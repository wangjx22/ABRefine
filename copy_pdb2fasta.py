import csv
import os
import argparse
from Bio.PDB import PDBParser, Select, PDBIO
import warnings
from Bio import BiopythonWarning

# 忽略Biopython的无关警告
warnings.filterwarnings("ignore", category=BiopythonWarning)

class ChainSelector(Select):
    """自定义链选择器"""
    def __init__(self, chains):
        self.chains = chains.upper().split(',')  # 支持多链选择
    
    def accept_chain(self, chain):
        return chain.get_id() in self.chains

def process_pdb(input_path, output_path, chains):
    """处理PDB文件并保存指定链"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("temp", input_path)
        
        # 验证链是否存在
        available_chains = {chain.id for chain in structure.get_chains()}
        selected_chains = set(chains.upper().split(','))
        
        if not selected_chains.issubset(available_chains):
            missing = selected_chains - available_chains
            raise ValueError(f"链 {', '.join(missing)} 不存在于文件中")
        
        # 保存处理后的结构
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_path, ChainSelector(chains))
        return True
    except Exception as e:
        print(f"处理错误: {str(e)}")
        return False

def copy_pdb_files(csv_file, target_dir, chain_col='chain', path_col='path'):
    os.makedirs(target_dir, exist_ok=True)
    stats = {'success': 0, 'errors': 0, 'existing': 0}
    processed = set()

    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # 验证必要列是否存在
        required_cols = {path_col, chain_col}
        if not required_cols.issubset(reader.fieldnames):
            missing = required_cols - set(reader.fieldnames)
            print(f"错误：CSV缺少必要列 {', '.join(missing)}")
            return

        for row in reader:
            source_path = row[path_col].strip()
            chains = row[chain_col].strip()
            if not all([source_path, chains]):
                stats['errors'] += 1
                continue

            try:
                # 生成标准化文件名
                normalized = os.path.normpath(source_path)
                base_name = normalized.replace(os.sep, '_')
                new_name = f"{base_name}_{chains.replace(',', '-')}.pdb"
                dest_path = os.path.join(target_dir, new_name)

                # 检查重复
                if new_name in processed:
                    print(f"重复记录跳过: {new_name}")
                    stats['errors'] += 1
                    continue
                
                if os.path.exists(dest_path):
                    print(f"文件已存在跳过: {new_name}")
                    stats['existing'] += 1
                    continue

                # 处理并保存文件
                if process_pdb(source_path, dest_path, chains):
                    processed.add(new_name)
                    print(f"成功处理: {new_name}")
                    stats['success'] += 1
                else:
                    stats['errors'] += 1

            except Exception as e:
                print(f"处理 {source_path} 时出错: {str(e)}")
                stats['errors'] += 1

    # 打印统计报告
    print("\n操作统计:")
    print(f"成功处理: {stats['success']}")
    print(f"已存在文件: {stats['existing']}")
    print(f"错误数量: {stats['errors']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDB链提取工具")
    parser.add_argument("--csv_file", help="输入CSV文件路径")
    parser.add_argument("--target_dir", help="输出目录路径")
    parser.add_argument("--path_col", default="path", help="路径列名称（默认：path）")
    parser.add_argument("--chain_col", default="chain", help="链列名称（默认：chain）")
    
    args = parser.parse_args()
    
    copy_pdb_files(
        args.csv_file,
        args.target_dir,
        chain_col=args.chain_col,
        path_col=args.path_col
    )