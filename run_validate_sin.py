import sys
sys.path.append('/nfs_beijing_ai/jinxian/rama-scoring1.3.0')
import argparse
import os
import random
from dataset.sin_equiformerv2_dataset import *
from utils.general import read_yaml_to_dict
from utils import dist_utils
from utils.logger import Logger
from np.protein import from_pdb_string, update_pdb_with_new_coords

def get_refine_pdb_by_inference(pdb_fpath, output_path, align_tran, predicted_tran, start_model_backbone_mask, init_model, start_atom_mask, gt_model, gt_res_mask, gt_atom_mask, start2gt_model_tran, start2gt_model_rot):
    """
    iter add the r and t to each atom temp. update through add R and T to all atom one step. 
    Args:
        pdb_fpath:
            file path of pdb, generate a new pdb if true
        output_path:
            the refine pdb path
        align_tran:
            [gt res, 37, 3], tensor
        predicted_tran:
            [start backbone, 3],  tensor
        all_atom_positions:
            [start res, 37, 3],  tensor
        start_model_backbone_mask:
            [start model res * 37], tensor
        init_model:
            [start_residue, 37, 3], tensor
        start_atom_mask:
            [start residue, 37], tensor
        gt_model:
            [gt_residue, 37, 3], tensor
        gt_res_mask:
            [full residue], tensor
        atom_mask:
            [gt_residue, 37],  tensor
        gt_atom_mask:
            [gt_residue, 37], tensor
        start2gt_model_rot:
            [3,3] ,tensor
        start2gt_model_tran:
            [3], tensor

    Return:
        rmsd   
    """

    
    # 计算gt_backbone_atom_positions，并且筛选predicted_tran, init_model_backbone与其对应
    restore_predicted_tran2start_model_shape = torch.zeros(init_model.shape).to(predicted_tran.device).view(-1,3)
    #根据backbone list去按照位置填入
    
    # 获取start_model_backbone_mask中为 1 的位置
    indices = start_model_backbone_mask.nonzero(as_tuple=True)[0]  # shape [n], n 是 A 中 1 的数量

    # 将 predicted_tran 的值赋给 restore_predicted_tran2start_model_shape 的相应位置
    restore_predicted_tran2start_model_shape[indices] = predicted_tran
    
    # 计算 refine model的rmsd
    refine_backbone_atom = init_model+restore_predicted_tran2start_model_shape.reshape(init_model.shape)
    
    #接下来去更新pdb
    refine_pdb_fpath = update_pdb_with_new_coords(pdb_fpath, refine_backbone_atom, output_path)
    
    return refine_pdb_fpath


def get_args():
    """Run get_args method."""
    # code.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/model_zoo_nb_no4_95_testset_0522_copy.csv",
        help="path of coarse graining",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/model/ckpt/run_train_sin_k8s_trial3_2024.11.18-12.13.00_Step_100000_improve_rat_0.0831341980535005_imporve_num_168.ckpt",
        help="path of coarse graining",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/inference/",
        help="path of coarse graining",
    )
    parser.add_argument(
        "--interface_cut",
        type=int,
        default=225,
        help="interface cutoff",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--yaml-path",
        type=str,
        default="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/yamls/refine_train_yamls/run_train_sin_k8s_trial3.yaml",
        help="path of yaml file",
    )
    parser.add_argument(
        "--refine_pdb_path",
        type=str,
        default="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/inference/pdb_file/",
        help="path of refine file",
    )

    args = parser.parse_args()

    return args

def main(inference_file_path, model_path, output_path, yaml_path, config_runtime, config_model, config_data):
    dataset_type = config_data['dataset_type']

    set_random_seed(config_runtime['seed'])
    test_file_path = config_data['test_filepath']
    valid_file_path = config_data['valid_filepath']

    valid_dataset = PDBDataset(csv_file=valid_file_path, data_type='test', dataset_type=dataset_type, sample_num=-1, valid_head=-1)
    test_dataset = PDBDataset(csv_file=test_file_path, data_type='test', dataset_type=dataset_type, sample_num=-1, valid_head=-1)

    # print('total_train_data:', len(train_dataset))
    print('total_valid_data:' , len(valid_dataset))
    print('total_test_data:' , len(test_dataset))


    def custom_collate_fn(data_list):
        # 过滤掉 None 值
        data_list = [data for data in data_list if data is not None]
        # 将过滤后的数据组合成一个批次（可以根据具体需求进行处理）
        if not data_list:
            return None

        return data_list  #将数据进一步处理成批量形式


    valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle= False,
            collate_fn=custom_collate_fn,
            pin_memory=True,
        )

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle= False,
            collate_fn=custom_collate_fn,
            pin_memory=True,
        )
    
    
    print('valid_dataloader length:', len(valid_dataloader))
    print('test_dataloader length:', len(test_dataloader))
    
    local_rank = torch.device('cuda')
    # device = torch.device(f'cuda:{rank}')
    model = EquiformerV2(**config_model).to(local_rank)
    if config_model['model_reload_path'] != False:
        model.load_state_dict(torch.load(config_model['model_reload_path']))
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    ema = EMA(model.module, decay=0.9999)

    distmap_loss_fn = nn.CrossEntropyLoss(reduction='sum' if distmap_loss_mean_or_sum == 'sum' else 'mean')

    loss_file = "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test_data_results/"+'Validation_Refine_'+config_model['model_reload_path']+".txt"
    with open(loss_file, "w") as file:
        file.write("Step, start pdb name, rmsd_start_CA, rmsd_refine_CA, rmsd_CA_improve_rat, start rmsd, refine rmsd, improve rat, start CDR CA rmsd, refine CDR CA rmsd, CDR1_impove_rmsd, CDR2_impove_rmsd, CDR3_impove_rmsd \n")  
    
    loss_file_test = "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test_data_results/"+'Test_Refine_'+config_model['model_reload_path']+".txt"
    with open(loss_file_test, "w") as file:
        file.write("Step, start pdb name, rmsd_start_CA, rmsd_refine_CA, rmsd_CA_improve_rat, start rmsd, refine rmsd, improve rat, start CDR CA rmsd, refine CDR CA rmsd, CDR1_impove_rmsd, CDR2_impove_rmsd, CDR3_impove_rmsd \n")  
    
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print('To Validation')
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            pred_distmap, predicted_atom_trans  = model(data.to(local_rank))

            if pred_distmap == None:
                distmap_loss = torch.tensor(0.0).to(local_rank)
                tri_loss = torch.tensor(0.0).to(local_rank)
            else:
                distmap_loss, tri_loss = distmap_loss_function(
                    pred_distmap=pred_distmap, 
                    gt_model=data.gt_atom_positions.to(local_rank), 
                    start_model_backbone_mask=data.start_model_backbone_mask.to(local_rank),
                    distmap_loss_fn=distmap_loss_fn,
                    cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
                    cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
                    cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank),
                    use_cdr_mask=distmap_loss_use_cdr_mask
                    )

            
            rmsd, rmsd_start, rmsd_refine_CA, rmsd_start_CA, tran_mse_loss, rot_pred_tran_mse, rot_pred_angle, cg_loss_all, rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after, CDR1_impove_rmsd, CDR2_impove_rmsd, CDR3_impove_rmsd = computer_rmsd4test_only_trans(
                align_tran=data.align_atom_trans.to(local_rank),
                predicted_tran=predicted_atom_trans, 
                start_model_backbone_mask= data.start_model_backbone_mask.to(local_rank),
                init_model=data.all_atom_positions.to(local_rank), 
                start_atom_mask=data.start_atom_mask.to(local_rank), 
                gt_model=data.gt_atom_positions.to(local_rank), 
                gt_res_mask=data.gt_res_mask.to(local_rank), 
                gt_atom_mask=data.gt_atom_mask.to(local_rank),
                res_names_list = data.res_names_list.to(local_rank),
                start2gt_model_tran = data.start2gt_model_tran.to(local_rank),
                start2gt_model_rot = data.start2gt_model_rot.to(local_rank),
                cdr_mask_by_res_mask=data.cdr_mask_by_res_mask.to(local_rank),
                cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
                cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
                cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank), 
                config_runtime=config_runtime
                )
            
            pdb_name = data.pdb_name[0]
            CA_improve_rmsd = (rmsd_start_CA.item() - rmsd_refine_CA.item())
            all_pdb_improve_rmsd += CA_improve_rmsd

            CDR_CA_improve_rmsd += (rmsd_refine_CA_CDR_before.item() - rmsd_refine_CA_CDR_after.item())
            CDR1_impove_rmsd_all_test += CDR1_impove_rmsd.item()
            CDR2_impove_rmsd_all_test += CDR2_impove_rmsd.item()
            CDR3_impove_rmsd_all_test += CDR3_impove_rmsd.item()
            backbone_improve_rmsd = (rmsd_start.item() - rmsd.item())

            with open(loss_file_test, "a") as file:
                file.write(f"Step:{step}, {pdb_name}, {rmsd_start_CA:.4f}, {rmsd_refine_CA}, {CA_improve_rmsd:.4f}, {rmsd_start:.4f}, {rmsd}, {backbone_improve_rmsd:.4f}, {rmsd_refine_CA_CDR_before:.4f}, {rmsd_refine_CA_CDR_after:.4f}, {CDR1_impove_rmsd:.4f}, {CDR2_impove_rmsd:.4f}, {CDR3_impove_rmsd:.4f}\n")
            
           
        for data in valid_dataloader:
            pred_distmap, predicted_atom_trans  = model(data.to(local_rank))
            
            if pred_distmap == None:
                distmap_loss = torch.tensor(0.0).to(local_rank)
                tri_loss = torch.tensor(0.0).to(local_rank)
            else:
                distmap_loss, tri_loss = distmap_loss_function(
                    pred_distmap=pred_distmap, 
                    gt_model=data.gt_atom_positions.to(local_rank), 
                    start_model_backbone_mask=data.start_model_backbone_mask.to(local_rank),
                    distmap_loss_fn= distmap_loss_fn,
                    cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
                    cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
                    cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank),
                    use_cdr_mask=distmap_loss_use_cdr_mask
                    )

            rmsd, rmsd_start, rmsd_refine_CA, rmsd_start_CA, tran_mse_loss, rot_pred_tran_mse, rot_pred_angle, cg_loss_all, rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after, CDR1_impove_rmsd, CDR2_impove_rmsd, CDR3_impove_rmsd = computer_rmsd4test_only_trans(
                align_tran=data.align_atom_trans.to(local_rank),
                predicted_tran=predicted_atom_trans, 
                start_model_backbone_mask= data.start_model_backbone_mask.to(local_rank),
                init_model=data.all_atom_positions.to(local_rank), 
                start_atom_mask=data.start_atom_mask.to(local_rank), 
                gt_model=data.gt_atom_positions.to(local_rank), 
                gt_res_mask=data.gt_res_mask.to(local_rank), 
                gt_atom_mask=data.gt_atom_mask.to(local_rank),
                res_names_list = data.res_names_list.to(local_rank),
                start2gt_model_tran = data.start2gt_model_tran.to(local_rank),
                start2gt_model_rot = data.start2gt_model_rot.to(local_rank),
                cdr_mask_by_res_mask=data.cdr_mask_by_res_mask.to(local_rank),
                cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
                cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
                cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank), 
                config_runtime=config_runtime
                )
          
            

            pdb_name = data.pdb_name[0]
            CA_improve_rmsd = (rmsd_start_CA.item() - rmsd_refine_CA.item())
            all_pdb_improve_rmsd += CA_improve_rmsd

            CDR_CA_improve_rmsd += (rmsd_refine_CA_CDR_before.item() - rmsd_refine_CA_CDR_after.item())
            CDR1_impove_rmsd_all_test += CDR1_impove_rmsd.item()
            CDR2_impove_rmsd_all_test += CDR2_impove_rmsd.item()
            CDR3_impove_rmsd_all_test += CDR3_impove_rmsd.item()
            backbone_improve_rmsd = (rmsd_start.item() - rmsd.item())

            with open(loss_file, "a") as file:
                file.write(f"Step:{step}, {pdb_name}, {rmsd_start_CA:.4f}, {rmsd_refine_CA}, {CA_improve_rmsd:.4f}, {rmsd_start:.4f}, {rmsd}, {backbone_improve_rmsd:.4f}, {rmsd_refine_CA_CDR_before:.4f}, {rmsd_refine_CA_CDR_after:.4f}, {CDR1_impove_rmsd:.4f}, {CDR2_impove_rmsd:.4f}, {CDR3_impove_rmsd:.4f}\n")

if __name__ == '__main__':
    # yaml_path='/yamls/refine_train_yamls/run_train_sin_k8s_trial3.yaml'
    # yaml_args = read_yaml_to_dict(yaml_path)
    # # set_random_seed(yaml_args['config_runtime'].seed)
    # logger.info(yaml_args)
    # inference_file_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/model_zoo_nb_no4_95_testset_0522.csv'
    # model_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/model/ckpt/2024.10.31-10.19.03_Step_46000_improve_rat_0.04649861819781495_imporve_num_420.ckpt'
    # # 输出的pdb综合数据csv存储路径
    # output_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/inference/'
    # # refine pdb的保存路径
    # refine_pdb_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/inference/pdb_file/'
    args = get_args()
    print(args)
    yaml_path = args.yaml_path
    yaml_args = read_yaml_to_dict(args.yaml_path)
    inference_file_path= args.input_path
    model_path = args.ckpt_path
    output_path= args.output_path
    main(inference_file_path, model_path, output_path, yaml_path, yaml_args["config_runtime"], yaml_args["config_model"], yaml_args["config_data"])