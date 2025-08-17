# Bin 0: distance < 2
# Bin1-24: 2-8A, each hot is 0.25A
# Bin25-36: 8-20A, each hot is 1 A
# Bine37: distance >20
import torch
import torch.nn.functional as F
import torch.nn as nn

def create_nonuniform_bin_edges():
    # 非等距分布：
    # ≤2.0 Å: bin 0
    # 2.0~8.0 Å 每0.25 Å一个bin => 24个bin
    # 8.0~20.0 Å 每1.0 Å一个bin => 12个bin
    # >20.0 Å: 最后一个bin
    
    edges = []
    # 2.0~8.0, step=0.25
    dist = 2.0
    while dist < 8.0:
        dist += 0.25
        edges.append(dist)
    
    # 8.0~20.0, step=1.0
    dist = 8.0
    while dist < 20.0:
        dist += 1.0
        edges.append(dist)
        
    # edges 包含从2.25, 2.5, ... 到20.0的上界
    return edges

def distance_to_bin_idx(dist, bin_edges):
    # bin 0: ≤2.0
    if dist <= 2.0:
        return 0
    
    # 遍历bin_edges找到合适的bin
    for i, edge in enumerate(bin_edges):
        if dist <= edge:
            # bin i+1 对应的区间为 (bin_edges[i-1], bin_edges[i]] (i从0计)
            # 由于0号bin是 ≤2.0，故i=0时对应第一条 >2.0的边界
            # 因此 bin = i+1
            return i + 1
    
    # 如果超过了最后一个边界，即 >20.0
    return len(bin_edges) + 1


if __name__ == '__main__':
    
    # 创建bin edges
    bin_edges = create_nonuniform_bin_edges()
    num_bins = len(bin_edges) + 2  # +2是因为包括 ≤2.0 和 >20.0 的bin

    # 假设我们有一个batch的预测logits和真实距离
    batch_size = 4
    logits = torch.randn(batch_size, num_bins)  # [batch_size, num_bins]

    # 假设真实距离值
    true_distances = [2.1, 25.0, 19.8, 2.0]

    # 将真实距离转换为对应的bin index
    true_indices = torch.tensor([distance_to_bin_idx(d, bin_edges) for d in true_distances], dtype=torch.long)

    # 使用CrossEntropyLoss计算
    # 注意：PyTorch的CrossEntropyLoss期望输入为logits（未softmax），target为类别索引
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, true_indices)

    print("Number of bins:", num_bins)
    print("Bin edges:", bin_edges)
    print("True indices:", true_indices)
    print("Loss:", loss.item())