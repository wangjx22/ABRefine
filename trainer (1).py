"""Code."""
from typing_extensions import override
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
from copy import deepcopy

from utils.exponential_moving_average import ExponentialMovingAverage, tensor_tree_map
from utils.baseTrainerv2 import BaseTrainer
from model.Model import ScoreModel
from model.equiformer_v2_model import EquiformerV2
from model.loss import KLLoss, HingeRankingLoss, RankNetLoss, LogisticRankingLoss
from dataset.datasetv2 import collate_batch_data as collate_batch_datav2
from dataset.datasetv3 import collate_batch_data as collate_batch_datav3
from dataset.equiformerv2_dataset import data_list_collater
import utils.mpu_utils as mpu
from utils import dist_utils
from utils.logger import Logger

logger = Logger.logger

class Trainer(BaseTrainer):
    """Define Class Trainer."""

    def __init__(self, args, config_runtime, config_model, config_data):
        """Run __init__ method."""
        # code.
        super().__init__(args)
        self.args = args
        self.cached_weights = None

        self.config_data = config_data
        self.config_model = config_model
        self.config_runtime = config_runtime

        loss_config = config_runtime.get("loss_fn", "mse")
        # to support previous version of yaml, loss_fn can receive a str type argument.
        # the new version of yaml should use a list type argument.
        # e.g. [{"loss": "mse", "output": 0, "target": "RMSD_CA"}]
        self.build_loss(loss_config)

        self.check_output_and_start_iter()

        # if "loss_fn" in config_runtime and "kl" in config_runtime["loss_fn"]:
        #     self.loss = KLLoss(reverse="reverse" in config_runtime["loss_fn"])
        # else:
        #     self.loss = torch.nn.MSELoss(reduction="none")

        fast_eval_interval = self.config_runtime.get(f"fast_eval_interval",-1)
        fast_eval_stop_iter = self.config_runtime.get(f"fast_eval_stop_iter",-1)
        self.args.fast_eval_interval = fast_eval_interval
        self.args.fast_eval_stop_iter = fast_eval_stop_iter

    def check_output_and_start_iter(self):
        """Run check_output_and_start_iter method."""
        # code.
        output2target = {}
        for item in self.loss:
            if item['output'] in output2target.keys():
                assert output2target[item['output']] == item['target'], "output and target should be consistent"
            else:
                output2target[item["output"]] = item["target"]

        self.output2target = output2target

        output2start_iters = {}
        for item in self.loss:
            if item['output'] in output2start_iters.keys():
                # if multiple loss are training the same output, the output is enabled for scoring when the first loss is used.
                # so we choose the min start_iter
                new_start_iter = min(item['start_iter'], output2start_iters[item['output']])
                output2start_iters[item['output']] = new_start_iter
            else:
                output2start_iters[item["output"]] = item["start_iter"]

        self.output2start_iters = output2start_iters

    def get_pred_involved_index(self, iter):
        """Run get_pred_involved_index method."""
        # code.
        pred_involved_index = [i for i in range(len(self.output2start_iters.keys())) if self.output2start_iters[i] <= iter]
        return pred_involved_index

    def build_loss(self, loss_config):
        """Run build_loss method."""
        # code.
        if loss_config.__class__ is str:
            # only one loss func is used. and the model should only output 1 logit.
            logger.warning("str type loss config is deprecated, please use list type loss config")

            loss_config = loss_config.lower()

            if 'rank' in loss_config:
                assert self.config_data['dataset_type'] == 'Datasetv3'

            if "kl" in loss_config:
                self.loss = [{'loss': KLLoss(reverse="reverse" in loss_config),
                              "output": 0,
                              "target": "y",
                              "weight": 1.0,
                              "start_iter": 0}]
            elif loss_config == 'mse':
                self.loss = [{'loss': torch.nn.MSELoss(reduction="none"),
                             'output': 0,
                             'target': 'y',
                             'weight': 1.0,
                             "start_iter": 0}]
            elif loss_config == 'mae':
                self.loss = [{"loss": torch.nn.L1Loss(reduction="none"),
                            "output": 0,
                            "target": 'y',
                            "weight": 1.0,
                            "start_iter": 0}]
            elif loss_config == 'hingeranking':
                raise NotImplementedError(f"Loss {loss_config} not support str type.")
            elif loss_config == 'logisticranking':
                raise NotImplementedError(f"Loss {loss_config} not support str type.")
            elif loss_config == 'ranknet':
                raise NotImplementedError(f"Loss {loss_config} not support str type.")
            else:
                raise NotImplementedError(f"Loss {loss_config} not implemented.")

        elif loss_config.__class__ is list:
            self.loss = []
            for item in loss_config:
                loss_type = item['type']
                loss_type = loss_type.lower()

                if 'rank' in loss_type:
                    assert self.config_data['dataset_type'] == 'Datasetv3'

                if loss_type == 'mse':
                    self.loss.append({'loss': torch.nn.MSELoss(reduction="none"),
                                      'output': item['output'],
                                      'target': item['target'],
                                      'weight': item['weight'],
                                      'start_iter': item.get('start_iter', 0),
                                      'type': loss_type})
                elif "kl" in loss_type:
                    self.loss.append({"loss": KLLoss(reverse="reverse" in loss_type),
                                      "output": item['output'],
                                      "target": item['target'],
                                      "weight": item['weight'],
                                      'start_iter': item.get('start_iter', 0),
                                      'type': loss_type})
                elif loss_type == 'mae':
                    self.loss.append({"loss": torch.nn.L1Loss(reduction="none"),
                                      "output": item['output'],
                                      "target": item['target'],
                                      "weight": item['weight'],
                                      'start_iter': item.get('start_iter', 0),
                                      'type': loss_type})
                elif loss_type == 'hingeranking':
                    self.loss.append({"loss": HingeRankingLoss(reduction="none"),
                                      "output": item['output'],
                                      "target": item['target'],
                                      "weight": item['weight'],
                                      'start_iter': item.get('start_iter',0),
                                      'type': loss_type})
                elif loss_type == 'logisticranking':
                    self.loss.append({"loss": LogisticRankingLoss(reduction="none"),
                                      "output": item['output'],
                                      "target": item['target'],
                                      "weight": item['weight'],
                                      'start_iter': item.get('start_iter',0),
                                      'type': loss_type})
                elif loss_type == 'ranknet':
                    self.loss.append({"loss": RankNetLoss(reduction="none"),
                                      "output": item['output'],
                                      "target": item['target'],
                                      "weight": item['weight'],
                                      'start_iter': item.get('start_iter',0),
                                      'type': loss_type})
                else:
                    raise NotImplementedError(f"Loss type {loss_type} not implemented.")
        else:
            raise TypeError("Wrong loss_fn type given.")

        # self.loss is a list of dict, each dict contains the loss function, and the information to describe this loss.
        # Previous version, self.loss is a single loss function. it will not be supported anymore in later versions.
        logger.info(f"Loss functions: {self.loss}")

    def on_before_zero_grad(self):
        """Run on_before_zero_grad method."""
        # code.
        self.ema.update(self.model)

    def on_validation_start(self):
        """Run on_validation_start method."""
        # code.
        # model.state_dict() contains references to model weights rather
        # than copies. Therefore, we need to clone them before calling
        # load_state_dict().
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())

        missing_keys, unexpected_keys = self.model.load_state_dict(
            self.ema.state_dict()["params"], strict=False
        )
        assert len(unexpected_keys) == 0
        assert len(missing_keys) == 0

    def on_validation_end(self):
        """Run on_validation_end method."""
        # code.
        missing_keys, unexpected_keys = self.model.load_state_dict(
            self.cached_weights, strict=False
        )
        assert len(unexpected_keys) == 0
        assert len(missing_keys) == 0
        del self.cached_weights

    def on_train_start(self):
        """Run on_train_start method."""
        # code.
        pass

    def build_model(self):
        """Run build_model method."""
        # code.
        if self.config_model.model_name == "ScoreModel":
            model = ScoreModel(self.config_model)
        elif self.config_model.model_name == "equiformer_v2":
            model = EquiformerV2(**self.config_model)

        model.cuda(torch.cuda.current_device())

        self.out_weight = self.config_model.get("out_weight", [1.0])
        self.out_weight = torch.Tensor(self.out_weight).to(torch.cuda.current_device())


        # Model Initialize
        def initialize_weights_xavier(m):
            """Run initialize_weights_xavier method."""
            # code.
            if isinstance(m, nn.Linear):
                logger.info(f"initializing {m}")
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        def initialize_weights_xavier_kaiming(layer):
            """Run initialize_weights_xavier_kaiming method."""
            # code.
            logger.info(f"initializing {layer}")
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        do_init = self.config_model.get("init_model", 'none')
        if do_init.lower() == 'xavier':
            model.apply(initialize_weights_xavier)
        elif do_init.lower() == 'kaiming' or do_init.lower() == 'he':
            model.apply(initialize_weights_xavier_kaiming)

        # logger.info(model.fc3.weight)
        # logger.info(model.fc3.bias)
        return model

    def forward_step(self, data_iterator):
        """Run forward_step method."""
        # code.
        timers = self.timers
        timers("batch generator1").start()
        batch = next(data_iterator)
        timers("batch generator1").stop()
        if batch is None:
            return None

        if self.config_model.model_name == 'ScoreModel':
            # If use ScoreModel, collate fn is from datasetv2 or datasetv3.
            # Then, the <batch> is one object, no matter it contains one batch of samples or a pair of batch of samples.
            # So, for k in batch, k is the key. if this key ends with _0, indicates that it contains a pair of batch of samples, then rank_flag=True
            # else, rank_flag = False.
            rank_flag = False
            for k in batch:
                if k.endswith("_0"):
                    rank_flag=True
                if "pdb" not in k:
                    batch[k] = batch[k].to(torch.cuda.current_device())
        elif self.config_model.model_name == 'equiformer_v2':
            # If use equiformer_v2 model, collate fn is from equiformer_dataset
            # Then, the <batch> is a list of objects.
            # So, if len(batch)==1, indicates it contains only one batch of samples.
            # else, rank_flag=False
            # Remember that the equiformerv2_dataset object is derived from datasetv3.
            # so for training, no matter whether ranking loss is used, the len(batch) is 2
            # But if we use regression loss rather than ranking loss, this loss will be computed both on batch[0] and batch[1], so it works.
            if len(batch) == 1:
                rank_flag = False
            elif len(batch) == 2:
                rank_flag = True
            else:
                raise RuntimeError(f"Wrong length of batch, please check: {batch}")
        else:
            raise RuntimeError(f"Unknown model name: {self.config_model.model_name}")

        if self.config_data['dataset_type'] == "Datasetv2" or rank_flag == False:
            # Original case (Datasetv2), no ranking loss used. Only one protein is fetched per sample.
            # Or valid/test set even if ranking loss is used (Datasetv3 and equiformerv2_dataset). Only one protein is fetched per sample.
            # Or just the valid/test set for datasetv3 and equiformerv2_dataset,
            # since they provide a pair of batch of samples for training, no matter the ranking loss is used.
            if batch.__class__ == list:
                # from equiformerv2_dataset (val/test)
                batch = batch[0].to(torch.cuda.current_device())
            # else, from Datasetv2 or Datasetv3 (val/test), tensors in batch have been moved to cuda

            sample_loss_weight = torch.Tensor(batch['loss_weight']).to(torch.cuda.current_device()) #[B]
            z = self.model(batch)  # [B, out]
            loss = torch.zeros(z.size()[0]).to(torch.cuda.current_device()) # [B]
            loss_metrics = {}
            for creterion in self.loss:
                if 'rank' in creterion['type']:
                    # for valid/test set, we will only get one protein per sample,
                    # so that the ranking loss will not be computed in evaluation.
                    continue

                if self.args.iteration < creterion['start_iter']:
                    # Not only for ranking loss, other regression loss can also be set to be started after several iters.
                    # so we need to compare current iteration with the start_iter.
                    continue

                l = creterion['loss'](z[:,creterion['output']], batch[creterion['target'].lower()])
                l = l * sample_loss_weight
                loss_metrics.update({f"{creterion['target']}_{creterion['type']}_loss": l.detach().cpu().tolist()})
                loss += creterion['weight'] * l

            # z: [B, out], self.out_weight: [out]
            assert z.size()[-1] == self.out_weight.size()[0], "The dimension of output and output_weight should be the same."

            pred_involved_index = self.get_pred_involved_index(self.args.iteration)
            pred = torch.matmul(z[:,pred_involved_index], self.out_weight[pred_involved_index])

            metrics = {
                "pdb_fpath": batch["pdb"],
                "pred": pred.detach().tolist(),
                "rmsd_ca_95": batch["rmsd_ca_95"].detach().tolist(),
                "rmsd_ca": batch["rmsd_ca"].detach().tolist(),
                "cdr3_rmsd_ca_95": batch["cdr3_rmsd_ca_95"].detach().tolist(),
                "cdr3_rmsd_ca": batch["cdr3_rmsd_ca"].detach().tolist(),
            }
            metrics.update(loss_metrics)
            return loss.mean(), metrics

        else:
            if self.config_model.model_name == 'ScoreModel':
                batch0 = {}
                batch1 = {}
                for k in batch.keys():
                    if k.endswith("_0"):
                        batch0.update({k[:-2]: batch[k]})
                    elif k.endswith("_1"):
                        batch1.update({k[:-2]: batch[k]})
                    else:
                        raise KeyError(f"Unknown key: {k}")
            elif self.config_model.model_name == 'equiformer_v2':
                batch0 = batch[0].to(torch.cuda.current_device())
                batch1 = batch[1].to(torch.cuda.current_device())
            else:
                raise RuntimeError(f"Unknown model name: {self.config_model.model_name}")

            z0 = self.model(batch0)  # [B, out]
            z1 = self.model(batch1)

            assert z0.size()[0] == z1.size()[0], "Batch size mismatch"

            loss = torch.zeros(z0.size()[0],2).to(torch.cuda.current_device())  # [B]
            loss_metrics = {}

            sample_loss_weight0 = torch.Tensor(batch0['loss_weight']).to(torch.cuda.current_device())  # [B]
            sample_loss_weight1 = torch.Tensor(batch1['loss_weight']).to(torch.cuda.current_device())  # [B]

            for creterion in self.loss:
                if 'rank' not in creterion['type']:
                    # not ranking loss, compute loss per sample
                    # check the iteration and start iter
                    if self.args.iteration < creterion['start_iter']:
                        continue

                    # if the criterion is not ranking loss, regression loss is computed on both z0 and z1
                    l0 = creterion['loss'](z0[:, creterion['output']],
                                          batch0[creterion['target'].lower()])
                    l1 = creterion['loss'](z1[:, creterion['output']],
                                           batch1[creterion[
                                               'target'].lower()])

                    l = torch.cat([(l0 * sample_loss_weight0).unsqueeze(-1),(l1 * sample_loss_weight1).unsqueeze(-1)], dim=1)

                    loss_metrics.update({f"{creterion['target']}_{creterion['type']}_loss": l.view(-1).detach().cpu().tolist()})
                    loss += creterion['weight'] * l

                else:
                    if self.args.iteration < creterion['start_iter']:
                        continue

                    label0 = batch0[creterion['target'].lower()]
                    label1 = batch1[creterion['target'].lower()]

                    score0 = z0[:, creterion['output']]
                    score1 = z1[:, creterion['output']]

                    if creterion['type'] != 'ranknet':
                        sign = torch.sign(label0 - label1)
                        score_dev = (score0 - score1) * sign
                        l = creterion['loss'](score_dev)
                    else:
                        score_dev = score0 - score1
                        label = (label0 > label1).long()
                        l = creterion['loss'](score_dev, label)

                    l = l.unsqueeze(-1)
                    l = l.repeat(1,2)
                    loss_metrics.update({f"{creterion['target']}_{creterion['type']}_loss": l.view(-1).detach().cpu().tolist()})
                    scores = torch.cat((score0, score1), dim=-1)
                    loss_metrics.update({f"scores_loss": scores.detach().cpu().tolist()})
                    score_dev = score_dev.repeat(2)
                    loss_metrics.update({f"abs_score_dev_loss": torch.abs(score_dev).detach().cpu().tolist()})
                    loss += creterion['weight'] * l

            # z: [B, out], self.out_weight: [out]
            assert z0.size()[-1] == self.out_weight.size()[0], "The dimension of z0 and self.out_weight must be the same"
            assert z1.size()[-1] == self.out_weight.size()[0], "The dimension of z1 and self.out_weight must be the same"

            pred_involved_index = self.get_pred_involved_index(self.args.iteration)
            pred0 = torch.matmul(z0[:, pred_involved_index], self.out_weight[pred_involved_index])
            pred1 = torch.matmul(z1[:, pred_involved_index], self.out_weight[pred_involved_index])

            metrics = {
                "pdb_fpath": batch0["pdb"] + batch1["pdb"],
                "pred": pred0.detach().tolist() + pred1.detach().tolist(),
                "rmsd_ca_95": batch0["rmsd_ca_95"].detach().tolist() + batch1["rmsd_ca_95"].detach().tolist(),
                "rmsd_ca": batch0["rmsd_ca"].detach().tolist() + batch1["rmsd_ca"].detach().tolist(),
                "cdr3_rmsd_ca_95": batch0["cdr3_rmsd_ca_95"].detach().tolist() + batch1["cdr3_rmsd_ca_95"].detach().tolist(),
                "cdr3_rmsd_ca": batch0["cdr3_rmsd_ca"].detach().tolist() + batch1["cdr3_rmsd_ca"].detach().tolist(),
            }
            metrics.update(loss_metrics)

            return loss.mean(), metrics

    @override
    def _make_loaders(self):
        """Run _make_loaders method."""
        # code.
        if self.config_data["dataset_type"] == "Datasetv2":
            from dataset.datasetv2 import Dataset
            Dataset = Dataset
        elif self.config_data["dataset_type"] == 'Datasetv3':
            from dataset.datasetv3 import Dataset
            Dataset = Dataset
        elif self.config_data['dataset_type'] == 'equiformerv2':
            from dataset.equiformerv2_dataset import Dataset
            Dataset = Dataset
        else:
            raise NotImplementedError(f"Other choice of dataset is not available.Please choose from ['Datasetv2', 'efDataset'].")
        logger.info(f"Building train dataset...")
        train_data = Dataset(
            self.config_data["train_filepath"],
            self.config_data,
            mode="train",
        )


        if self.config_data.get("fast_warm_up_set", False):
            fast_warm_up_config = deepcopy(self.config_data)
            target = self.config_data['fast_warm_up_set']['target']
            if target == 'RMSD_CA':
                bin = [0, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 200000]
            elif target == "CDR3_RMSD_CA":
                bin = [0,0.6,1.0,1.4,1.8,2.2,2.6,3.0,3.5,4.0,4.5,5.0,5.5,6.0,7.0,8.0,100000]
            size_ratio = self.config_data['fast_warm_up_set']['size_ratio']

            tmp_config = {"target": target, "bin": bin, "size_ratio": size_ratio}

            fast_warm_up_config.update({"down_sampling":tmp_config})

            logger.info(f"Building fast warm up train dataset...")
            fast_warm_up_train_data = Dataset(self.config_data['train_filepath'],
                                                fast_warm_up_config,
                                              mode='train')
            logger.info(f"Fast warmup train dataset is built.")
            logger.info(f"First sample of fast warmup train dataset:\n{fast_warm_up_train_data[0]}")
        else:
            fast_warm_up_train_data = None

        valid_datas = []
        logger.info(f"Building valid datasets...")
        for i, valid_filepath in enumerate(self.config_data["valid_filepath"].split(',')):
            valid_datas.append(
                Dataset(valid_filepath, self.config_data, mode=f"valid{i}")
            )


        logger.info(f"Building test datasets...")
        if self.config_data.get("test_filepath", None):
            test_datas = []
            for i, test_filepath in enumerate(self.config_data["test_filepath"].split(',')):
                test_datas.append(
                    Dataset(test_filepath, self.config_data, mode=f"test{i}")
                )
        else:
            test_datas = None

        # only when training on multiple GPUs, we need a distributed sampler. Else, default, no sampler.
        train_sampler = None
        fast_warm_up_sampler = None
        valid_samplers = []
        if dist_utils.get_world_size() > 1:
            train_sampler = DistributedSampler(
                train_data,
                rank=dist_utils.get_rank(),
                num_replicas=dist_utils.get_world_size(),
                shuffle=False,
            )
            if fast_warm_up_train_data is not None:
                fast_warm_up_sampler = DistributedSampler(
                    fast_warm_up_train_data,
                    rank=dist_utils.get_rank(),
                    num_replicas=dist_utils.get_world_size(),
                    shuffle=False,
                )

            for i, valid_data in enumerate(valid_datas):
                valid_samplers.append(
                    DistributedSampler(
                        valid_data,
                        rank=dist_utils.get_rank(),
                        num_replicas=dist_utils.get_world_size(),
                        shuffle=False,
                    )
                )

        if test_datas is not None:
            test_samplers = []
            if dist_utils.get_world_size() > 1:
                for i, test_data in enumerate(test_datas):
                    test_samplers.append(
                        DistributedSampler(
                            test_data,
                            rank=dist_utils.get_rank(),
                            num_replicas=dist_utils.get_world_size(),
                            shuffle=False,
                        )
                    )
        else:
            test_samplers = None

        if self.config_data['dataset_type'] == 'Datasetv2':
            train_collate_fn = collate_batch_datav2
            valid_collate_fn = collate_batch_datav2
        elif self.config_data['dataset_type'] == 'Datasetv3':
            train_collate_fn = collate_batch_datav3
            valid_collate_fn = collate_batch_datav2
        elif self.config_data['dataset_type'] == 'equiformerv2':
            train_collate_fn = data_list_collater
            valid_collate_fn = data_list_collater


        train_dataloader = DataLoader(
            train_data,
            batch_size=self.config_data['trn_batch_size'],
            num_workers=self.config_data['num_workers'],
            shuffle=True if train_sampler is None else False,
            collate_fn=train_collate_fn,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=self.config_data['prefetch_factor'],
            sampler=train_sampler,
        )
        if fast_warm_up_train_data is not None:
            fast_warm_up_train_dataloader = DataLoader(
                fast_warm_up_train_data,
                batch_size=self.config_data['trn_batch_size'],
                num_workers=self.config_data['num_workers'],
                shuffle=True if fast_warm_up_sampler is None else False,
                collate_fn=train_collate_fn,
                drop_last=True,
                pin_memory=True,
                prefetch_factor=self.config_data['prefetch_factor'],
                sampler=fast_warm_up_sampler,
            )
        else:
            fast_warm_up_train_dataloader = None

        val_dataloaders = []
        for i, valid_data in enumerate(valid_datas):
            val_dataloaders.append(
                DataLoader(
                    valid_data,
                    batch_size=self.config_data['val_batch_size'],
                    num_workers=self.config_data['num_workers'],
                    shuffle=False,
                    collate_fn=valid_collate_fn,
                    pin_memory=True,
                    prefetch_factor=self.config_data['prefetch_factor'],
                    sampler=valid_samplers[i] if valid_samplers else None,
                )
            )

        if test_datas is not None:
            test_dataloaders = []
            for i, test_data in enumerate(test_datas):
                test_dataloaders.append(
                    DataLoader(
                        test_data,
                        batch_size=self.config_data['val_batch_size'],
                        num_workers=self.config_data['num_workers'],
                        shuffle=False,
                        collate_fn=valid_collate_fn,
                        pin_memory=True,
                        prefetch_factor=self.config_data['prefetch_factor'],
                        sampler=test_samplers[i] if test_samplers else None,
                    )
                )
        else:
            test_dataloaders = None

        world_size = torch.distributed.get_world_size(
            group=mpu.get_data_parallel_group()
        )
        global_batch_size = self.args.batch_size * world_size
        self.args.train_epoch_len = (
                self.args.train_iters * global_batch_size * self.args.gradient_accumulation_steps
        )
        batch_size = 1

        def setup_for_eval_metric(data):
            """Run setup_for_eval_metric method."""
            # code.
            batch_size = global_batch_size
            eval_batch_size = batch_size
            if self.args.eval_batch_size is not None:
                eval_batch_size = self.args.eval_batch_size * world_size
            eval_batch_size = eval_batch_size if eval_batch_size != 0 else batch_size
            self.eval_ds_len = len(data)
            last_len = len(data) % eval_batch_size  # 0 ~ bs-1
            batch_per_worker = eval_batch_size // world_size  # mbs
            last_shape = [batch_per_worker] * (
                    last_len // batch_per_worker
            )  # len() = 0 ~ ws-1
            if last_len != 0:  # 1~bs-1
                if last_len % batch_per_worker != 0:  # (1~bs-1)%mbs>0
                    last_shape.append(last_len % batch_per_worker)  # (1~bs-1)%mbs
                drop_number = world_size - (
                        (last_len - 1) // batch_per_worker + 1
                )  # (2//mbs-1) ~ ws-1
                for j in range(drop_number):
                    last_shape.append(1)
            else:
                drop_number = 0
            return last_shape, drop_number

        self.args.val_last_shape, self.args.val_drop_number = [], []
        for valid_data in valid_datas:
            val_last_shape, val_drop_number = setup_for_eval_metric(valid_data)
            self.args.val_last_shape.append(val_last_shape)
            self.args.val_drop_number.append(val_drop_number)

        if test_datas is not None:
            self.args.test_last_shape, self.args.test_drop_number = [], []
            for test_data in test_datas:
                test_last_shape, test_drop_number = setup_for_eval_metric(test_data)
                self.args.test_last_shape.append(test_last_shape)
                self.args.test_drop_number.append(test_drop_number)

            return train_dataloader, val_dataloaders, test_dataloaders, fast_warm_up_train_dataloader
        else:
            return train_dataloader, val_dataloaders, None, fast_warm_up_train_dataloader

    def set_client_state(self, client_states):
        """Run set_client_state method."""
        # code.
        logger.info(f"self.ema.device{self.ema.device}")
        if 'ema' in client_states:
            self.ema.load_state_dict(client_states["ema"])
            self.ema.to(self.ema.device)
            logger.info(f"client_state[ema]: successfully add ema to client state")
        super().set_client_state(client_states)

    def get_client_state(self):
        """Run get_client_state method."""
        # code.
        sd = super().get_client_state()
        sd["ema"] = self.ema.state_dict()
        return sd

    def after_deepspeed_initialize(self):
        """Run after_deepspeed_initialize method."""
        # code.
        self.ema = ExponentialMovingAverage(
            model=self.model,
            decay=self.config_runtime['ema_decay'],
        )
        self.ema.to(torch.cuda.current_device())
