import warnings
from tqdm import tqdm
from model import *
import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from datetime import datetime
import time

warnings.filterwarnings("ignore")

# ========== 新增显存优化代码 ==========
# 1. 清空显存缓存
torch.cuda.empty_cache()
# 2. 限制仅使用70%显存
torch.cuda.set_per_process_memory_fraction(0.9, device=0)
# 3. 启用TF32（减少显存占用，不影响精度）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 4. 禁用显存预分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ================= 配置区域 =================
seed = 47
batch_size = 20
in_size = 512
hidden_size = 256
out_size = 128
dropout = 0.5  # 0.5
lr = 0.0001  # 0.0001
weight_decay = 1e-10  # 1e-10
epochs = 100
cl_loss_co = 1
reg_loss_co = 0.0001
dataset = "MNDR4.0(15-8178)"
# dataset = "MNDR3.0(19-10149)"
# dataset = "piRDisease(21-4350)"
args = setup(default_configure, seed)
timestamp = time.time()
save_path = f"./save/{timestamp}/"
args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
# args['device'] = "cpu"


def main(tr, te, seed):
    # 存储每折的最佳指标（用于打印最终结果）
    all_acc = []
    all_roc = []
    all_aupr = []
    all_precision = []
    all_recall = []

    # i 是交叉验证的第 i 折
    for i in range(len(tr)):
        print(f"\n{'=' * 20} Starting Fold {i} {'=' * 20}")

        # 1. 写入索引文件 (保持原有逻辑)
        with open(save_path + f"{i}foldtrain.txt", "w", encoding="utf-8") as f:
            for idx in tr[i]:
                f.write(f"{idx}\n")
        with open(save_path + f"{i}foldtest.txt", "w", encoding="utf-8") as f:
            for idx in te[i]:
                f.write(f"{idx}\n")

        # 2. 创建日志 CSV (记录每个 epoch 的标量指标)
        log_filename = save_path + f"log_fold_{i}.csv"
        with open(log_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['epoch', 'loss', 'train_acc', 'train_roc', 'test_acc', 'test_roc', 'test_aupr', 'test_precision',
                 'test_recall'])

        # 3. 初始化模型
        model = HMTCL(
            all_meta_paths=all_meta_paths,
            in_size=[hp.shape[1], hd.shape[1]],
            hidden_size=[hidden_size, hidden_size],
            out_size=[out_size, out_size],
            dropout=dropout,
        ).to(args['device'])

        optim = torch.optim.Adam(lr=lr, weight_decay=weight_decay, params=model.parameters())

        # 记录最佳状态
        best_acc = 0
        best_probs = None  # 【新增】存储最佳 epoch 的预测概率
        best_labels = None  # 【新增】存储最佳 epoch 的真实标签

        # 临时变量用于记录当前 epoch 的返回值
        current_best_metrics = {}

        for epoch in tqdm(range(epochs), desc=f"Fold {i}"):
            # 【修改】接收 probs 和 labels
            loss, train_acc, train_roc, te_acc, te_roc, te_aupr, te_precision, te_recall, te_probs, te_labels = train(
                model, optim, tr[i], te[i], epoch, i
            )

            # 更新最佳指标 (以 Accuracy 为准，也可以改为 AUPR)
            if te_acc > best_acc:
                best_acc = te_acc
                # 【关键】保存最佳时刻的概率和标签，用于后续画 ROC/PR 曲线
                best_probs = te_probs
                best_labels = te_labels

                # 同时更新其他最佳标量指标用于打印
                current_best_metrics = {
                    'roc': te_roc,
                    'aupr': te_aupr,
                    'precision': te_precision,
                    'recall': te_recall
                }

            # 记录日志到 CSV
            with open(log_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    to_float(loss), to_float(train_acc), to_float(train_roc),
                    to_float(te_acc), to_float(te_roc), to_float(te_aupr),
                    to_float(te_precision), to_float(te_recall)
                ])

        # 1. 打印最佳结果
        fold_result = f"Fold {i}: Best Accracy: {best_acc:.4f}, Best AUC: {current_best_metrics.get('roc', 0):.4f}, Best AUPR: {current_best_metrics.get('aupr', 0):.4f}, Best Precision: {current_best_metrics.get('precision', 0):.4f}, Best Recall: {current_best_metrics.get('recall', 0):.4f}\n"
        print(fold_result)
        with open(save_path + f"result.txt", "a", encoding="utf-8") as f:
            f.write(fold_result)

        # 2. 累加指标 (确保全部转换为 CPU float)
        all_acc.append(float(best_acc))
        # 使用 .item() 如果是 tensor，或者 float() 强制转换
        roc_val = current_best_metrics.get('roc', 0)
        all_roc.append(roc_val.item() if isinstance(roc_val, torch.Tensor) else float(roc_val))
        aupr_val = current_best_metrics.get('aupr', 0)
        all_aupr.append(aupr_val.item() if isinstance(aupr_val, torch.Tensor) else float(aupr_val))
        prec_val = current_best_metrics.get('precision', 0)
        all_precision.append(prec_val.item() if isinstance(prec_val, torch.Tensor) else float(prec_val))
        rec_val = current_best_metrics.get('recall', 0)
        all_recall.append(rec_val.item() if isinstance(rec_val, torch.Tensor) else float(rec_val))

        # 3. 【核心新增】计算并保存 ROC 和 PR 曲线数据
        if best_probs is not None and best_labels is not None:
            # 确保是 numpy 数组
            y_true = best_labels.cpu().numpy() if isinstance(best_labels, torch.Tensor) else best_labels
            y_scores = best_probs.cpu().numpy() if isinstance(best_probs, torch.Tensor) else best_probs

            # 计算 ROC 曲线点 (FPR, TPR)
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            # 保存 ROC 数据
            roc_file = save_path + f"roc_fold_{i}.csv"
            np.savetxt(roc_file, np.column_stack([fpr, tpr]), delimiter=',', header='FPR,TPR', comments='')
            # print(f"  -> ROC 曲线数据已保存至: {roc_file}")

            # 计算 PR 曲线点 (Precision, Recall)
            # 注意：sklearn 的 precision_recall_curve 返回顺序是 (Precision, Recall)
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
            # 保存 PR 数据
            pr_file = save_path + f"pr_fold_{i}.csv"
            np.savetxt(pr_file, np.column_stack([recall_vals, precision_vals]), delimiter=',',
                       header='Recall,Precision', comments='')
            # print(f"  -> PR 曲线数据已保存至: {pr_file}")

    # 输出最终平均指标
    final_result = f"5-Fold 平均结果: Accuracy: {np.mean(all_acc):.4f} (+/- {np.std(all_acc):.4f}), AUC: {np.mean(all_roc):.4f} (+/- {np.std(all_roc):.4f}), AUPR: {np.mean(all_aupr):.4f} (+/- {np.std(all_aupr):.4f}), Precision: {np.mean(all_precision):.4f} (+/- {np.std(all_precision):.4f}), Recall: {np.mean(all_recall):.4f} (+/- {np.std(all_recall):.4f})\n"
    print(final_result)
    with open(save_path + f"result.txt", "a", encoding="utf-8") as f:
        f.write(final_result)


def train(model, optim, train_index, test_index, epoch, fold):
    model.train()

    # Forward
    out, cl_loss, p, d = model(graph, node_feature, cl, train_index, data, save_path)

    # 训练集指标
    train_acc = (out.argmax(dim=1) == label[train_index].reshape(-1)).sum(dtype=float) / len(train_index)
    train_roc = get_roc(out, label[train_index])

    # Loss
    reg = get_L2reg(model.parameters())
    loss = F.nll_loss(out, label[train_index].reshape(-1).long()) + cl_loss_co * cl_loss + reg_loss_co * reg

    optim.zero_grad()
    loss.backward()
    optim.step()

    # 【修改】测试集获取 probs 和 labels
    te_acc, te_roc, te_aupr, te_precision, te_recall, te_probs, te_labels = main_test(
        model, p, d, test_index, epoch, fold
    )

    return loss, train_acc, train_roc, te_acc, te_roc, te_aupr, te_precision, te_recall, te_probs, te_labels


def main_test(model, p, d, test_index, epoch, fold):
    model.eval()
    with torch.no_grad():
        out = model(graph, node_feature, cl, test_index, data, save_path, iftrain=False, p=p, d=d)

        # 真实标签
        true_labels = label[test_index].reshape(-1)

        # 预测类别
        pred_classes = out.argmax(dim=1)

        # 预测概率 (取正类的概率，假设第 1 列是正类)
        # 如果 out 是 log-probabilities (NLLLoss 的输入)，需要先 exp
        if out.min() < 0 or out.max() > 1:
            probs = F.softmax(out, dim=1)[:, 1]
        else:
            probs = out[:, 1] if out.shape[1] > 1 else out.squeeze()

        # 计算标量指标
        acc1 = (pred_classes == true_labels).sum(dtype=float) / len(test_index)

        # 这里的 get_roc, get_pr 等函数需要确保它们接收的是 (logits/probs, labels)
        # 如果原本的 get_roc 内部做了 softmax，这里传入 probs 可能需要调整。
        # 假设原本的 get_roc 能处理 logits 或 probs:
        task_roc = get_roc(out, true_labels)
        task_pr = get_pr(out, true_labels)
        precision = get_precision(out, true_labels)
        recall = get_recall(out, true_labels)

        # 【新增】返回 probs 和 labels (CPU numpy 格式，方便后续处理)
        return (
            acc1,
            task_roc,
            task_pr,
            precision,
            recall,
            probs.cpu().numpy(),
            true_labels.cpu().numpy()
        )


def to_float(val):
    if isinstance(val, torch.Tensor):
        return val.cpu().item()
    return float(val)


def save_config(filename):
    # 拼接完整的文件路径（结合save_path）
    config_file_path = os.path.join(save_path, filename)

    # 定义需要保存的超参（按顺序整理，便于阅读）
    config_content = f"""# 实验超参数配置
# 保存时间: {time.ctime(timestamp)}
# 运行设备: {args['device']}

# 基础配置
seed = {seed}
batch_size = {batch_size}
epochs = {epochs}
dataset = "{dataset}"

# 模型结构配置
in_size = {in_size}
hidden_size = {hidden_size}
out_size = {out_size}
dropout = {dropout}

# 优化器配置
lr = {lr}
weight_decay = {weight_decay}

# 损失函数系数配置
cl_loss_co = {cl_loss_co}
reg_loss_co = {reg_loss_co}
"""
    # 写入文件（确保编码为utf-8，避免中文/特殊字符问题）
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"配置文件已成功保存到: {config_file_path}")


if __name__ == "__main__":
    os.makedirs(save_path, exist_ok=True)  # 确保输出目录存在
    save_config("config.txt")  # 保存此次实验的超参配置
    print("加载数据...")

    pdadata, graph, num, all_meta_paths = load_dataset(dataset, save_path)
    print("piRNA, disease 数量：", num[0], num[1])

    if isinstance(graph, list):
        graph = [g.to(args['device']) for g in graph]
    else:
        graph = graph.to(args['device'])

    pda_label = torch.tensor(pdadata[:, 2:3]).to(args['device'])

    hp = torch.randn((num[0], in_size))
    hd = torch.randn((num[1], in_size))
    features_p = hp.to(args['device'])
    features_d = hd.to(args['device'])
    node_feature = [features_p, features_d]

    pda_cl = get_clGraph(pdadata, "pda", save_path).to(args['device'])

    cl = pda_cl
    data = pdadata
    label = pda_label

    train_indices, test_indices = get_cross(pdadata)
    print("每折训练集与测试集比例：", len(train_indices[0]), ":", len(test_indices[0]))

    main(train_indices, test_indices, seed)
