import sys
sys.path.append('/nfs_beijing_ai/jinxian/rama-scoring1.3.0')
import argparse
import os
import random
import time
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
        default="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/model_zoo_nb_no4_95_testset_0522.csv",
        help="path of coarse graining",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/model/ckpt/run_train_sin_k8s_trial3_yaml_run_train_sin_k8s_trial6_2024.12.16-09.24.27_Step_16000_improve_rmsd_-0.11802367705944156_imporve_num_211.ckpt",
        help="path of coarse graining",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test/",
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
        default="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/yamls/refine_train_yamls/run_train_sin_k8s_trial6.yaml",
        help="path of yaml file",
    )
    # parser.add_argument(
    #     "--refine_pdb_path",
    #     type=str,
    #     default="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/inference/pdb_file/",
    #     help="path of refine file",
    # )

    args = parser.parse_args()

    return args

def main(inference_file_path, model_path, output_path, yaml_path, config_runtime, config_model, config_data):
    dataset_type = config_data['dataset_type']

    set_random_seed(config_runtime['seed'])

    inference_file_name = inference_file_path.split('/')[-1].split('.')[0]
    model_name = model_path.split('/')[-1].split('.ck')[0]
    yaml_name = yaml_path.split('/')[-1].split('.')[0]
    # 输出的pdb综合数据csv存储路径
    # output_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/inference/'
    
    # 创建用于存储refine pdb fpath以及各项指标的csv文件
    output_pdb_save_fpath = output_path +'refinefile_'+ inference_file_name +'_refinemodel_'+ model_name+ '_yaml_name_'+yaml_name+ '.txt'
    # 获取output目录路径
    output_dir = os.path.dirname(output_pdb_save_fpath)
    # 如果目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 结果指标文件
    with open(output_pdb_save_fpath, "w") as file:
        file.write("start_pdb_fpath, refine_pdb_fpath, rmsd_start_CA, rmsd_refine_CA, rmsd_CA_improve, rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after, rmsd_CA_CDR_refine_improve, CDR1_impove_rmsd, CDR2_impove_rmsd, CDR3_impove_rmsd\n")
       

    # refine pdb的保存路径
    refine_pdb_path = output_path +'refinefile_'+ inference_file_name +'_refinemodel_'+ model_name+ '_yaml_name_'+yaml_name+'/'
    
    # 创建用于存储refine pdb的路径
    output_dir = os.path.dirname(refine_pdb_path)
    # 如果目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    # 创建inference dataset
    # inference_dataset = PDBDataset(csv_file=inference_file_path, data_type='inference', dataset_type=dataset_type, sample_num=-1, valid_head=-1)
    valid_dataset = PDBDataset(csv_file=inference_file_path, data_type='test', dataset_type=dataset_type, sample_num=-1, valid_head=-1)
    print('total_valid_dataset:' , len(valid_dataset))

    def custom_collate_fn(data_list):
        # 过滤掉 None 值
        data_list = [data for data in data_list if data is not None]
        # 将过滤后的数据组合成一个批次（可以根据具体需求进行处理）
        if not data_list:
            return None
        return data_list  # 将数据进一步处理成批量形式

    # inference_dataloader = DataLoader(
    #         inference_dataset,
    #         batch_size=1,
    #         num_workers=4,
    #         shuffle= False,
    #         collate_fn=custom_collate_fn,
    #         pin_memory=True,
    #         # prefetch_factor=args.prefetch_factor,
    #     )

    valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=1,
            num_workers=4,
            shuffle= False,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            # prefetch_factor=args.prefetch_factor,
        )

    device = torch.device('cuda')
    model = EquiformerV2(**config_model)
    model.load_state_dict(torch.load(model_path), strict=False)

    model.to(device)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model.eval()
    refine_pdb_fpath_list = []
    impover_list = []
    rmsd_CA_CDR_refine_improve_all = 0
    RMSD_CA_improve_all = 0
    with torch.no_grad():
        for data in tqdm(valid_dataloader):
            pred_distmap, predicted_atom_trans_valid  = model(data.to(device))
            pdb_name = data.pdb_name[0]
            # print(pdb_name)
            if pred_distmap == None:
                distmap_loss = torch.tensor(0.0).to(device)
                tri_loss = torch.tensor(0.0).to(device)
            else:
                distmap_loss, tri_loss = distmap_loss_fuction(
                    pred_distmap=pred_distmap, 
                    gt_model=data.gt_atom_positions.to(device), 
                    start_model_backbone_mask=data.start_model_backbone_mask.to(device)
                    )
                
            rmsd, rmsd_start, rmsd_refine_CA, rmsd_start_CA, tran_mse_loss, rot_pred_tran_mse, rot_pred_angle, cg_loss_all, rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after, CDR1_impove_rmsd, CDR2_impove_rmsd, CDR3_impove_rmsd = computer_rmsd4test_only_trans(
                align_tran=data.align_atom_trans.to(device),
                predicted_tran=predicted_atom_trans_valid, 
                start_model_backbone_mask= data.start_model_backbone_mask.to(device),
                init_model=data.all_atom_positions.to(device), 
                start_atom_mask=data.start_atom_mask.to(device), 
                gt_model=data.gt_atom_positions.to(device), 
                gt_res_mask=data.gt_res_mask.to(device), 
                gt_atom_mask=data.gt_atom_mask.to(device),
                res_names_list = data.res_names_list.to(device),
                start2gt_model_tran = data.start2gt_model_tran.to(device),
                start2gt_model_rot = data.start2gt_model_rot.to(device),
                cdr1_mask=data.cdr1_mask_backbone_atom.to(device), 
                cdr2_mask=data.cdr2_mask_backbone_atom.to(device), 
                cdr3_mask=data.cdr3_mask_backbone_atom.to(device), 
                config_runtime=config_runtime
                )
            
            
            RMSD_CA_improve = (rmsd_start_CA.item() - rmsd_refine_CA.item())
            RMSD_CA_improve_all += RMSD_CA_improve
            rmsd_CA_CDR_refine_improve = rmsd_refine_CA_CDR_before - rmsd_refine_CA_CDR_after
            rmsd_CA_CDR_refine_improve_all += rmsd_CA_CDR_refine_improve

            refine_pdb_fpath = get_refine_pdb_by_inference(
                pdb_fpath=pdb_name, 
                output_path=refine_pdb_path, # 输出路径应该根据指定的来存，名称中最好包括模型名,输入文件和yaml的名字
                align_tran=None,
                predicted_tran=predicted_atom_trans_valid, 
                start_model_backbone_mask= data.start_model_backbone_mask.to(device),
                init_model=data.all_atom_positions.to(device), 
                start_atom_mask=None, 
                gt_model=None, 
                gt_res_mask=None, 
                gt_atom_mask=None,
                start2gt_model_tran = None,
                start2gt_model_rot = None
                )
            # refine_pdb_fpath_list.append(refine_pdb_fpath)
            with open(output_pdb_save_fpath, "a") as file:
                file.write(f"{pdb_name}, {refine_pdb_fpath}, {rmsd_start_CA}, {rmsd_refine_CA}, {RMSD_CA_improve}, {rmsd_refine_CA_CDR_before}, {rmsd_refine_CA_CDR_after}, {rmsd_CA_CDR_refine_improve}, {CDR1_impove_rmsd}, {CDR2_impove_rmsd}, {CDR3_impove_rmsd}\n")
            
    print('mean of RMSD_CA_improve_all:', RMSD_CA_improve_all/len(valid_dataset))
    print('mean of rmsd_CA_CDR_refine_improve_all:', rmsd_CA_CDR_refine_improve_all/len(valid_dataset))
    # # 创建 DataFrame，并指定第一行为列名
    # df = pd.DataFrame(refine_pdb_fpath_list)

    # # 将 DataFrame 保存为 CSV 文件
    # df.to_csv(output_pdb_save_fpath, index=False)

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
    
    