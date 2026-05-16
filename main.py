import warnings
from model import *
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import pandas as pd
from datetime import datetime


k_fold = 5
seed = 42
in_size = 512
hidden_size = 256
out_size = 128
dropout = 0.5
lr = 0.001
weight_decay = 1e-5
epochs = 150
cl_loss_co = 1
reg_loss_co = 0.001
topk = 2
transformer_heads = 2
dataset = "piRDisease(21-4350)"
args = setup(default_configure, seed)
current_time = datetime.now().strftime("%m-%d-%H-%M")
save_path = f"./save/{dataset}-{current_time}/"
args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
os.makedirs(save_path, exist_ok=True)


class HMTCLLightning(L.LightningModule):
    def __init__(self, model, cl_loss_co, reg_loss_co, lr, weight_decay,
                 graph, node_feature, cl, data, label, train_index, test_index, topk, save_path):
        super().__init__()
        self.model = model
        self.cl_loss_co = cl_loss_co
        self.reg_loss_co = reg_loss_co
        self.lr = lr
        self.weight_decay = weight_decay

        self.graph = graph
        self.node_feature = node_feature
        self.cl = cl
        self.data = data
        self.label = label
        self.train_index = train_index
        self.test_index = test_index
        self.topk = topk
        self.save_path = save_path

        self.best_acc = 0.0
        self.best_metrics = {}
        self.all_probs = None
        self.all_labels = None

        self.cached_p = None
        self.cached_d = None

    def training_step(self, batch, batch_idx):
        out, cl_loss, p, d = self.model(self.graph, self.node_feature, self.cl, self.train_index, self.data,
                                        self.topk, self.save_path)

        self.cached_p = p
        self.cached_d = d

        train_labels = self.label[self.train_index].reshape(-1)
        train_acc = (out.argmax(dim=1) == train_labels).sum(dtype=float) / len(self.train_index)
        train_roc = get_roc(out, train_labels)

        reg_loss = get_L2reg(self.model.parameters())

        loss = F.nll_loss(out, train_labels.long()) + self.cl_loss_co * cl_loss + self.reg_loss_co * reg_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_auc", train_roc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        with torch.no_grad():
            out = self.model(
                self.graph, self.node_feature, self.cl, self.test_index, self.data, self.topk, self.save_path,
                iftrain=False, p=self.cached_p, d=self.cached_d
            )

            true_labels = self.label[self.test_index].reshape(-1)
            pred_classes = out.argmax(dim=1)
            probs = F.softmax(out, dim=1)[:, 1] if out.shape[1] > 1 else out.squeeze()

            acc = (pred_classes == true_labels).sum(dtype=float) / len(self.test_index)
            auc = get_roc(out, true_labels)
            aupr = get_pr(out, true_labels)
            precision = get_precision(out, true_labels)
            recall = get_recall(out, true_labels)

        if acc > self.best_acc:
            self.best_acc = float(acc.cpu()) if isinstance(acc, torch.Tensor) else float(acc)
            self.best_metrics = {
                "acc": float(acc.cpu()) if isinstance(acc, torch.Tensor) else float(acc),
                "auc": float(auc.cpu()) if isinstance(auc, torch.Tensor) else float(auc),
                "aupr": float(aupr.cpu()) if isinstance(aupr, torch.Tensor) else float(aupr),
                "precision": float(precision.cpu()) if isinstance(precision, torch.Tensor) else float(precision),
                "recall": float(recall.cpu()) if isinstance(recall, torch.Tensor) else float(recall)
            }
            self.all_probs = probs.cpu().numpy()
            self.all_labels = true_labels.cpu().numpy()

        self.log("val_acc", acc, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)
        return acc

    def train_dataloader(self):
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor([0])), batch_size=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor([0])), batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def main(tr, te, seed):
    all_acc, all_roc, all_aupr, all_precision, all_recall = [], [], [], [], []

    for fold in range(len(tr)):
        print(f"\n{'=' * 20} Fold {fold} {'=' * 20}")
        fold_path = os.path.join(save_path, f"fold_{fold}")
        os.makedirs(fold_path, exist_ok=True)

        with open(os.path.join(save_path, f"{fold}foldtrain.txt"), "w") as f:
            f.write("\n".join(map(str, tr[fold])))
        with open(os.path.join(save_path, f"{fold}foldtest.txt"), "w") as f:
            f.write("\n".join(map(str, te[fold])))

        model = HMTCL(
            all_meta_paths=all_meta_paths,
            in_size=[hp.shape[1], hd.shape[1]],
            hidden_size=[hidden_size, hidden_size],
            out_size=[out_size, out_size],
            dropout=dropout,
            transformer_heads=transformer_heads
        ).to(args['device'])

        lit_model = HMTCLLightning(
            model=model,
            cl_loss_co=cl_loss_co,
            reg_loss_co=reg_loss_co,
            lr=lr,
            weight_decay=weight_decay,
            graph=graph,
            node_feature=node_feature,
            cl=cl,
            data=data,
            label=label,
            train_index=tr[fold],
            test_index=te[fold],
            topk=topk,
            save_path=save_path
        )

        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            logger=CSVLogger(fold_path),
            enable_progress_bar=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
        )

        trainer.fit(lit_model)

        best = lit_model.best_metrics
        y_scores = lit_model.all_probs
        y_true = lit_model.all_labels

        # 保存测试集预测结果
        test_indices_fold = te[fold]
        pred_label = (y_scores > 0.5).astype(int)

        pirna_idx = pdadata[test_indices_fold, 0]
        disease_idx = pdadata[test_indices_fold, 1]
        true_label = pdadata[test_indices_fold, 2]

        save_pred = pd.DataFrame({
            "test_index": test_indices_fold,
            "piRNA_index": pirna_idx,
            "disease_index": disease_idx,
            "true_label": true_label,
            "pred_prob": y_scores,
            "pred_label": pred_label
        })

        pred_file = os.path.join(save_path, f"predictions_fold_{fold}.csv")
        save_pred.to_csv(pred_file, index=False)
        print(f"Fold {fold} 预测结果（含RNA/疾病ID）已保存 → {pred_file}")

        # 保存 ROC / PR 曲线
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        np.savetxt(os.path.join(save_path, f"roc_fold_{fold}.csv"),
                   np.column_stack([fpr, tpr]), delimiter=',', header='FPR,TPR', comments='')
        pre_vals, rec_vals, _ = precision_recall_curve(y_true, y_scores)
        np.savetxt(os.path.join(save_path, f"pr_fold_{fold}.csv"),
                   np.column_stack([rec_vals, pre_vals]), delimiter=',', header='Recall,Precision', comments='')

        all_acc.append(best["acc"])
        all_roc.append(best["auc"])
        all_aupr.append(best["aupr"])
        all_precision.append(best["precision"])
        all_recall.append(best["recall"])

        res = f"Fold {fold}: Acc={best['acc']:.4f} AUC={best['auc']:.4f} AUPR={best['aupr']:.4f} Precision={best['precision']:.4f} Recall={best['recall']:.4f}"
        print(res)
        with open(os.path.join(save_path, "result.txt"), "a") as f:
            f.write(res + "\n")

    final = f"\n5折平均结果: Acc={np.mean(all_acc):.4f}±{np.std(all_acc):.4f} | AUC={np.mean(all_roc):.4f}±{np.std(all_roc):.4f} | AUPR={np.mean(all_aupr):.4f}±{np.std(all_aupr):.4f} | Precision={np.mean(all_precision):.4f}±{np.std(all_precision):.4f} | Recall={np.mean(all_recall):.4f}±{np.std(all_recall):.4f}"
    print(final)
    with open(os.path.join(save_path, "result.txt"), "a") as f:
        f.write(final + "\n")


def save_config(filename):
    config_content = f"""# 实验配置
seed={seed} epoch={epochs} dataset={dataset}
in_size={in_size} hidden={hidden_size} out={out_size} dropout={dropout}
lr={lr} wd={weight_decay} cl_loss={cl_loss_co} reg={reg_loss_co} topk={topk} transformer_heads={transformer_heads} device={args['device']}
"""
    with open(os.path.join(save_path, filename), "w") as f:
        f.write(config_content)
    print("配置已保存到：", save_path)


if __name__ == "__main__":
    save_config("config.txt")

    pdadata, graph, num, all_meta_paths = load_dataset(dataset, save_path)
    pd.DataFrame(pdadata, columns=["piRNA_index", "disease_index", "label"]).to_csv(
        os.path.join(save_path, "pdadata.csv"), index=False
    )
    print("piRNA/disease:", num[0], num[1])

    if isinstance(graph, list):
        graph = [g.to(args['device']) for g in graph]
    else:
        graph = graph.to(args['device'])

    pda_label = torch.tensor(pdadata[:, 2:3]).to(args['device'])
    hp = torch.randn((num[0], in_size)).to(args['device'])
    hd = torch.randn((num[1], in_size)).to(args['device'])
    node_feature = [hp, hd]
    pda_cl = get_clGraph(pdadata, "pda", save_path).to(args['device'])

    cl = pda_cl
    data = pdadata
    label = pda_label

    train_indices, test_indices = get_cross(pdadata)
    print("数据比例:", len(train_indices[0]), ":", len(test_indices[0]))

    main(train_indices, test_indices, seed)
