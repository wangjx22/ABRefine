import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import os

def calculate_seq_stats(csv_file_path):
    """
    计算CSV文件中full_seq列的平均字符串长度和统计信息
    
    参数:
        csv_file_path (str): CSV文件路径
    
    返回:
        dict: 包含统计信息的字典
    """
    # 检查文件是否存在
    if not os.path.isfile(csv_file_path):
        return {"error": f"文件 '{csv_file_path}' 不存在"}
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 检查full_seq列是否存在
        if 'full_seq' not in df.columns:
            return {"error": "CSV文件中缺少 'full_seq' 列"}
        
        print(f"成功读取文件: {csv_file_path}")
        print(f"总行数: {len(df)}")
        
        # 计算序列长度
        df['seq_length'] = df['full_seq'].str.len()
        
        # 处理NaN值（空序列）
        nan_count = df['seq_length'].isna().sum()
        if nan_count > 0:
            print(f"警告: 发现 {nan_count} 个空序列，已用长度0替代")
            df['seq_length'] = df['seq_length'].fillna(0)
        
        # 基本统计信息
        stats = {
            "total_rows": len(df),
            "empty_seqs": nan_count,
            "avg_length": df['seq_length'].mean(),
            "min_length": df['seq_length'].min(),
            "max_length": df['seq_length'].max(),
            "std_dev": df['seq_length'].std(),
            "long_seqs": len(df[df['seq_length'] > 1000]),
            "percentiles": {
                "25%": df['seq_length'].quantile(0.25),
                "50%": df['seq_length'].quantile(0.50),
                "75%": df['seq_length'].quantile(0.75),
                "95%": df['seq_length'].quantile(0.95)
            }
        }
        
        return stats
        
    except Exception as e:
        return {"error": f"处理文件时出错: {str(e)}"}

def print_stats(stats):
    """打印统计信息"""
    if "error" in stats:
        print(f"\n错误: {stats['error']}")
        return
    
    print("\n===== 序列长度统计 =====")
    print(f"总行数: {stats['total_rows']}")
    print(f"空序列数: {stats['empty_seqs']}")
    print(f"平均长度: {stats['avg_length']:.2f} 个字符")
    print(f"最小长度: {stats['min_length']}")
    print(f"最大长度: {stats['max_length']}")
    print(f"标准差: {stats['std_dev']:.2f}")
    print(f"长度超过1000的序列数: {stats['long_seqs']} ({stats['long_seqs']/stats['total_rows']:.2%})")
    
    print("\n百分位数:")
    print(f"  25%: {stats['percentiles']['25%']:.2f}")
    print(f"  50% (中位数): {stats['percentiles']['50%']:.2f}")
    print(f"  75%: {stats['percentiles']['75%']:.2f}")
    print(f"  95%: {stats['percentiles']['95%']:.2f}")
    
    

# 设置您的CSV文件路径
csv_file_path = "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/train_set_split_by_protein_with_RMSD.csv"  # 替换为您的实际文件路径

# 计算统计信息
stats = calculate_seq_stats(csv_file_path)

# 打印结果
if "error" not in stats:
    print_stats(stats)
    
else:
    print(stats["error"])

def calculate_average_sequence_length(path):
    """
    计算指定路径下所有FASTA文件的平均序列长度（每个文件两条序列长度之和的平均值）。
    
    参数:
        path (str): 包含FASTA文件的目录路径
    
    返回:
        float: 所有文件的平均序列长度和
    """
    if not os.path.isdir(path):
        return f"错误: 路径 '{path}' 不存在或不是目录。"
    
    file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if not file_list:
        return "错误: 目录中没有文件。"
    
    total_length_sum = 0  # 所有文件的序列长度和的总和
    file_count = 0        # 有效文件计数
    _1k_num = 0
    
    for filename in file_list:
        filepath = os.path.join(path, filename)
        try:
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f.readlines()]  # 读取并清理行
                
                # 检查文件格式: 至少4行，且第1/3行以'>'开头
                if len(lines) < 4:
                    print(f"警告: 跳过 '{filename}' (行数不足)")
                    continue
                if not (lines[0].startswith('>') and lines[2].startswith('>')):
                    print(f"警告: 跳过 '{filename}' (格式不符)")
                    continue
                
                # 提取两条序列并计算长度和
                seq1_len = len(lines[1])  # 第2行是第一条序列
                seq2_len = len(lines[3])  # 第4行是第二条序列
                length_sum = seq1_len + seq2_len
                
                total_length_sum += length_sum
                file_count += 1
                # print(f"文件 '{filename}': 序列长度和 = {length_sum} (H链: {seq1_len}, L链: {seq2_len})")
                if length_sum > 1000:
                    _1k_num+=1
        except Exception as e:
            print(f"警告: 跳过 '{filename}' (读取错误: {str(e)})")
    
    if file_count == 0:
        return "错误: 无有效FASTA文件处理。"
    
    average = total_length_sum / file_count
    return average, _1k_num

# 设置您的FASTA文件路径
path = "/nfs_beijing_ai/jinxian/AB_No_miss_res_fasta"  # 替换为您的实际路径

# 计算并打印结果
average_length, _1k_num = calculate_average_sequence_length(path)
if isinstance(average_length, float):
    print(f"\n所有文件的平均序列长度和: {average_length:.2f}")
    print('_1k_num:', _1k_num)
else:
    print(average_length)
    print('_1k_num:', _1k_num)
