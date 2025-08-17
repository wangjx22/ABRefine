import os
import shutil
import csv
from pathlib import Path
import pandas as pd

def copy_files_from_csv(csv_path, target_dir, column_name="pred_path"):
    """
    从CSV文件复制指定列的文件到目标目录
    
    参数:
        csv_path (str): CSV文件路径
        target_dir (str): 目标存储目录
        column_name (str): 要读取的列名(默认为"fpath")
    """
    # 创建目标目录(如果不存在)
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    error_count = 0
    
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # 验证列是否存在
        if column_name not in reader.fieldnames:
            raise ValueError(f"CSV文件中不存在'{column_name}'列")
        
        for row in reader:
            src_path = row[column_name].strip()  # 去除前后空格
            
            # 跳过空路径
            if not src_path:
                print(f"⚠️ 跳过空路径 (行号: {reader.line_num})")
                error_count += 1
                continue
            
            # 处理路径
            src_path = os.path.expanduser(src_path)  # 支持~符号
            src_path = os.path.abspath(src_path)
            
            try:
                # 检查文件是否存在
                if not os.path.exists(src_path):
                    print(f"❌ 文件不存在: {src_path} (行号: {reader.line_num})")
                    error_count += 1
                    continue
                
                if not os.path.isfile(src_path):
                    print(f"⛔ 路径不是文件: {src_path} (行号: {reader.line_num})")
                    error_count += 1
                    continue
                
                # 生成目标路径
                filename = os.path.basename(src_path)
                dest_path = os.path.join(target_dir, filename)
                
                # 执行复制
                shutil.copy2(src_path, dest_path)  # 保留元数据
                copied_count += 1
                print(f"✅ 已复制: {copied_count}")
                
            except Exception as e:
                print(f"🔥 复制失败 [{src_path}]: {str(e)}")
                error_count += 1
    
    # 输出总结
    print("\n操作完成:")
    print(f"• 成功复制: {copied_count} 文件")
    print(f"• 失败次数: {error_count}")
    print(f"• 输出目录: {os.path.abspath(target_dir)}")

# 使用示例
if __name__ == "__main__":
    # # 输入参数配置
    # csv_file = "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/nb_train_torchMD_2071_all_95_0511_w_pdbid.csv"  # 替换为你的CSV路径
    # output_dir = "/nfs_beijing_ai/jinxian/ziqiao_NB_torchMD"    # 指定输出目录
    
    # copy_files_from_csv(csv_file, output_dir)


    #将torchMD处理之前的文件保存
    df = pd.read_csv("/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/trainset_gt_22059_nb15_nb22_vh15_vh9_vl11_model_decoys_0423.csv")  # 替换为你的 CSV 文件路径

    # 筛选 chain_type 为 "vhh" 的行
    vhh_df = df[df["chain_type"].str.lower() == "vhh"]  # 不区分大小写匹配

    # 定义目标文件夹路径
    target_dir = "/nfs_beijing_ai/jinxian/ziqiao_NB_before_torchMD_nb15_decoys"  # 替换为你需要保存的目录
    os.makedirs(target_dir, exist_ok=True)  # 自动创建目录

    # 遍历符合条件的行
    for _, row in vhh_df.iterrows():
        try:
            # 获取源文件路径和新文件名
            src_path = row["nb15_decoys"]
            new_name = f"{row['id']}_nb15_decoys.pdb"
            dst_path = os.path.join(target_dir, new_name)
            
            # 检查源文件是否存在
            if not os.path.exists(src_path):
                print(f"警告：文件 {src_path} 不存在，跳过")
                continue
                
            # 复制并重命名文件
            shutil.copy(src_path, dst_path)
            print(f"已保存：{new_name}")
            
        except Exception as e:
            print(f"处理 ID {row['id']} 时出错：{str(e)}")

    print("处理完成！检查输出目录：", os.path.abspath(target_dir))