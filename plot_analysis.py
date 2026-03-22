import os
import glob
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import auc, average_precision_score

# ================= 配置区域 =================
LOG_DIR = "./save/1773998480.2194908"
OUTPUT_DIR = LOG_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOLD_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
AVG_COLOR = '#000000'
# 局部放大图的位置和大小 [left, bottom, width, height] (相对于整个图的比例)
# ROC_POS = [0.3, 0.35, 0.4, 0.4]
# PR_POS = [0.3, 0.35, 0.4, 0.4]
# 局部放大图的x/y轴范围（可根据数据调整）
# ROC_XLIM = [-0.01, 0.15]
# ROC_YLIM = [0.89, 1.01]
# PR_XLIM = [0.85, 1.01]
# PR_YLIM = [0.85, 1.01]
# 针对MNDR v4.0数据集：
ROC_POS = [0.3, 0.25, 0.4, 0.4]
PR_POS = [0.3, 0.25, 0.4, 0.4]
ROC_XLIM = [0.15, 0.45]
ROC_YLIM = [0.89, 1.01]
PR_XLIM = [0.85, 1.01]
PR_YLIM = [0.7, 0.9]


def load_curve_data(prefix):
    """加载所有 fold 的曲线数据"""
    pattern = os.path.join(LOG_DIR, f"{prefix}_fold_*.csv")
    files = sorted(glob.glob(pattern))
    folds_data = []
    for f in files:
        data = np.genfromtxt(f, delimiter=',', skip_header=1)  # 跳过表头读取
        folds_data.append(data)
    return folds_data


def calculate_metric(data, metric_type):
    """计算单折数据的 AUC 或 AUPR"""
    if metric_type == 'roc':
        # ROC 曲线：FPR(列0), TPR(列1)
        fpr = data[:, 0]
        tpr = data[:, 1]
        return auc(fpr, tpr)
    else:
        # PR 曲线：Recall(列0), Precision(列1)
        recall = data[:, 0]
        precision = data[:, 1]
        # 方法1：用梯形法则计算 PR 曲线下面积（AUPR）
        # 注意sklearn的average_precision_score需要原始标签和预测值，这里用曲线积分更合适
        return np.trapz(precision[::-1], recall[::-1])


def plot_average_curve(title, folds_data, metric_type, xlabel, ylabel, filename):
    """
    绘制单折曲线 + 插值平均曲线
    metric_type: 'roc' or 'pr'
    """
    if not folds_data:
        print(f"未找到 {metric_type} 数据文件")
        return

    # 创建画布（用fig对象来添加嵌入子图）
    fig, ax = plt.subplots(figsize=(10, 8))

    # 定义统一的网格用于插值
    if metric_type == 'roc':
        mean_x = np.linspace(0, 1, 100)  # 统一 FPR 网格
        col_x, col_y = 0, 1  # CSV 列索引: FPR, TPR
        metric_name = 'AUC'
    else:  # pr
        mean_x = np.linspace(0, 1, 100)  # 统一 Recall 网格
        col_x, col_y = 0, 1  # CSV 列索引: Recall, Precision
        metric_name = 'AUPR'

    tprs = []
    fold_metrics = []  # 存储每个fold的AUC/AUPR值

    # 绘制每一条 Fold 的曲线（带metric值）
    for i, data in enumerate(folds_data):
        x = data[:, col_x]
        y = data[:, col_y]

        # 计算当前fold的AUC/AUPR
        fold_metric = calculate_metric(data, metric_type)
        fold_metrics.append(fold_metric)

        # 绘制单折曲线，图例显示Fold编号+metric值（保留4位小数）
        plt.plot(x, y, color=FOLD_COLORS[i % len(FOLD_COLORS)],
                 lw=1, alpha=0.6, label=f'Fold {i + 1} ({metric_name}={fold_metric:.4f})')

        # 2. 插值到统一网格 (为了计算平均线)
        if metric_type == 'roc':
            y_interp = np.interp(mean_x, x, y)
            tprs.append(y_interp)
        else:
            # PR曲线：翻转x/y使其递增
            x_inc = x[::-1]
            y_inc = y[::-1]
            y_interp = np.interp(mean_x, x_inc, y_inc)
            tprs.append(y_interp)

    # 计算平均值和标准差
    mean_y = np.mean(tprs, axis=0)
    std_y = np.std(tprs, axis=0)

    # 计算平均metric值
    mean_metric = np.mean(fold_metrics)
    std_metric = np.std(fold_metrics)
    # label_avg = f'Mean ({metric_name} = {mean_metric:.4f} ± {std_metric:.4f})'  # 带标准差
    label_avg = f'Mean ({metric_name} = {mean_metric:.4f})'

    # 绘制平均线
    plt.plot(mean_x, mean_y, color=AVG_COLOR, linestyle='--', lw=1, label=label_avg)

    # 绘制标准差阴影（可选，取消注释即可启用）
    # plt.fill_between(mean_x, mean_y - std_y, mean_y + std_y, color='gray', alpha=0.15, label='Std Dev')

    # 美化
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')

    # ROC 曲线添加局部放大图（若不需要注释掉即可）
    if metric_type == 'roc':

        # ROC 图添加随机猜测对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3, label='Random Guess')

        # 创建嵌入的子图（局部放大图）
        inset_ax = fig.add_axes(ROC_POS, facecolor='white')

        # 绘制每个fold的曲线到放大图
        for i, data in enumerate(folds_data):
            x = data[:, col_x]
            y = data[:, col_y]
            inset_ax.plot(x, y, color=FOLD_COLORS[i % len(FOLD_COLORS)],
                          lw=1, alpha=0.6)

        # 绘制平均线到放大图
        inset_ax.plot(mean_x, mean_y, color=AVG_COLOR, linestyle='--', lw=1)

        # 绘制随机猜测线到放大图
        inset_ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)

        # 设置放大图的范围
        inset_ax.set_xlim(ROC_XLIM)
        inset_ax.set_ylim(ROC_YLIM)

        # 放大图美化
        inset_ax.grid(True, alpha=0.3, linestyle='--')
        # inset_ax.set_title('Zoomed View (FPR: 0-0.1)', fontsize=10)  # 放大图标题

    # PR 曲线局部放大图
    if metric_type == 'pr':

        inset_ax = fig.add_axes(PR_POS, facecolor='white')

        for i, data in enumerate(folds_data):
            x = data[:, col_x]
            y = data[:, col_y]
            inset_ax.plot(x, y, color=FOLD_COLORS[i % len(FOLD_COLORS)], lw=1, alpha=0.6)

        inset_ax.plot(mean_x, mean_y, color=AVG_COLOR, linestyle='--', lw=1)

        inset_ax.set_xlim(PR_XLIM)
        inset_ax.set_ylim(PR_YLIM)

        inset_ax.grid(True, alpha=0.3, linestyle='--')
        # inset_ax.set_title('Zoomed: High Recall & Precision', fontsize=10)

    save_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
    plt.savefig(save_path, dpi=300)
    print(f"已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    # 1. 绘制 ROC 曲线
    print("正在处理 ROC 数据...")
    roc_folds = load_curve_data("roc")
    plot_average_curve("Receiver Operating Characteristic", roc_folds, 'roc', 'False Positive Rate',
                       'True Positive Rate', 'auc_curve')

    # 2. 绘制 PR 曲线
    print("正在处理 PR 数据...")
    pr_folds = load_curve_data("pr")
    plot_average_curve("Precision-Recall Curve", pr_folds, 'pr', 'Recall', 'Precision', 'aupr_curve')

    print("所有曲线绘制完成！")
