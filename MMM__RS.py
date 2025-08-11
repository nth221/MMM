from math import inf
import math
import dill
import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import jaccard_score
from torch.optim import Adam
import os
import torch
import time
from MMM import MMM
from cnn_pretrain import select_model_output_dim
from util import evaluate_ground_truth_ddi_rate, evaluate_ground_truth_ddi_rate_per_admission, llprint, multi_label_metric, multi_label_metric_string, ddi_rate_score, get_n_params, buildMPNN, plot_pr_curve, rank_aware_loss
import torch.nn.functional as F
import sys
import pickle
import random
from datetime import datetime

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def llprint(message):
    sys.stdout.write('\r' + message)
    sys.stdout.flush()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def convert_to_ATC3_prob_dict(y_prob_batch, cid_vocab, cid_to_atc3):
    atc3_prob_list = []

    for prob_array in y_prob_batch: 
        atc3_to_probs = defaultdict(list)
        
        for idx, prob in enumerate(prob_array):
            cid = cid_vocab.get(idx)
            atc3 = cid_to_atc3.get(cid)
            if atc3:
                atc3_to_probs[atc3].append(prob)

        atc3_prob_dict = {}
        for atc3, probs in atc3_to_probs.items():
            probs_np = np.array(probs)
            weights = softmax(probs_np)
            atc3_prob = np.sum(weights * probs_np)
            atc3_prob_dict[atc3] = float(atc3_prob)

        atc3_prob_list.append(atc3_prob_dict)

    return atc3_prob_list


def convert_dicts_to_array(prob_dict_list, label_vocab):
    atc3_idx = {label: i for i, label in enumerate(label_vocab)}
    arr = np.zeros((len(prob_dict_list), len(label_vocab)))

    for i, prob_dict in enumerate(prob_dict_list):
        for atc3, prob in prob_dict.items():
            if atc3 in atc3_idx:
                arr[i, atc3_idx[atc3]] = prob

    return arr

def convert_gt_sets_to_array(gt_set_list, label_vocab):
    atc3_idx = {label: i for i, label in enumerate(label_vocab)}
    arr = np.zeros((len(gt_set_list), len(label_vocab)))
    for i, gt_set in enumerate(gt_set_list):
        for atc3 in gt_set:
            if atc3 in atc3_idx:
                arr[i, atc3_idx[atc3]] = 1
    return arr
def eval(model, data_eval, voc_size, epoch, med_voc, ddi_adj):
    model.eval()
    y_true_atc3_all, y_scores_atc3_all = [], []
    y_true_cid_all,  y_scores_cid_all  = [], []
    with open("/data/MMM.u2/mcwon/SafeDrug/data/output/cid_to_atc3.pkl", "rb") as f:
        cid_to_atc3 = pickle.load(f)        
    cid_vocab = med_voc.idx2word 

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1, auroc = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0

    val_loss_sum, count = 0.0, 0  
    with torch.no_grad():      
        for step, input in enumerate(data_eval):
            y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

            for adm_idx, adm in enumerate(input):
                target_output, loss_ddi = model(input[: adm_idx + 1])

                y_gt_tmp = np.zeros(voc_size[2])
                y_gt_tmp[adm[2]] = 1
                y_gt.append(y_gt_tmp)

                target_output_sigmoid = F.sigmoid(target_output)
                target_output_np = target_output_sigmoid.detach().cpu().numpy()[0]
                y_pred_prob.append(target_output_np)

                # binary predictions
                y_pred_tmp = target_output_np.copy()
                y_pred_tmp[y_pred_tmp >= 0.5] = 1
                y_pred_tmp[y_pred_tmp < 0.5] = 0
                y_pred.append(y_pred_tmp)
                
                y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                y_pred_label.append(sorted(y_pred_label_tmp))

                loss_bce_target = torch.FloatTensor(y_gt_tmp).unsqueeze(0).to(target_output.device)
                loss_multi_target = np.full((1, voc_size[2]), -1)
                for i, item in enumerate(adm[2]):
                    loss_multi_target[0][i] = item
                loss_multi_target = torch.LongTensor(loss_multi_target).to(target_output.device)

                loss_bce = F.binary_cross_entropy_with_logits(target_output, loss_bce_target)
                loss_multi = F.multilabel_margin_loss(target_output_sigmoid, loss_multi_target)
                
            
                current_ddi_rate = ddi_rate_score([[y_pred_label_tmp]], path="/data/MMM.u2/mcwon/SafeDrug/data/output/ddi_A_final.pkl")
                
                if current_ddi_rate <= args.target_ddi:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = max(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
                    loss = (
                        beta * (0.95 * loss_bce + 0.05 * loss_multi)
                        + (1 - beta) * loss_ddi
                    )
                
                val_loss_sum += loss.item()
                count += 1

                visit_cnt += 1
                med_cnt += len(y_pred_label_tmp)

            smm_record.append(y_pred_label)

            y_gt_ATC3 = convert_to_ATC3_prob_dict([np.where(gt == 1)[0] for gt in y_gt], cid_vocab, cid_to_atc3)
            y_pred_ATC3 = convert_to_ATC3_prob_dict([np.where(pred == 1)[0] for pred in y_pred], cid_vocab, cid_to_atc3)
            
            y_prob_ATC3 = convert_to_ATC3_prob_dict(y_pred_prob, cid_vocab, cid_to_atc3)
            all_atc3_labels = sorted(set().union(*[set(d.keys()) for d in y_prob_ATC3]))

            adm_ja_ATC3, adm_prauc_ATC3, adm_avg_p_ATC3, adm_avg_r_ATC3, adm_avg_f1_ATC3, adm_auroc_ATC3 = multi_label_metric_string(
                y_gt_ATC3, y_pred_ATC3, y_prob_ATC3
            )

            adm_ja_CID, adm_prauc_CID, adm_avg_p_CID, adm_avg_r_CID, adm_avg_f1_CID, adm_auroc_CID = multi_label_metric(
                np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
            )
            
            ja.append((adm_ja_CID, adm_ja_ATC3))
            prauc.append((adm_prauc_CID, adm_prauc_ATC3))
            avg_p.append((adm_avg_p_CID, adm_avg_p_ATC3))
            avg_r.append((adm_avg_r_CID, adm_avg_r_ATC3))
            avg_f1.append((adm_avg_f1_CID, adm_avg_f1_ATC3))
            auroc.append((adm_auroc_CID, adm_auroc_ATC3))
            llprint(f"evaluation step: {step + 1} / {len(data_eval)}")

    ddi_rate = ddi_rate_score(smm_record, path="/data/MMM.u2/mcwon/SafeDrug/data/output/ddi_A_final.pkl")
    val_loss_avg = val_loss_sum / count
    

    y_true_cid_all.append(np.array(y_gt))
    y_scores_cid_all.append(np.array(y_pred_prob))
    y_true_atc3_all.append(np.array(y_gt_ATC3))
    y_scores_atc3_all.append(np.array(y_prob_ATC3))

    ja_cid_mean, ja_atc_mean = np.mean([x[0] for x in ja]), np.mean([x[1] for x in ja])
    prauc_cid_mean, prauc_atc_mean = np.mean([x[0] for x in prauc]), np.mean([x[1] for x in prauc])
    p_cid_mean, p_atc_mean = np.mean([x[0] for x in avg_p]), np.mean([x[1] for x in avg_p])
    r_cid_mean, r_atc_mean = np.mean([x[0] for x in avg_r]), np.mean([x[1] for x in avg_r])
    f1_cid_mean, f1_atc_mean = np.mean([x[0] for x in avg_f1]), np.mean([x[1] for x in avg_f1])
    auroc_cid_mean, auroc_atc_mean = np.mean([x[0] for x in auroc]), np.mean([x[1] for x in auroc])

    
    all_med = med_cnt/visit_cnt

    print(f"\n===== Epoch {epoch+1} Eval =====")
    print(f"CID → DDI: {ddi_rate:.4f}, Jaccard: {ja_cid_mean:.4f}, PRAUC: {prauc_cid_mean:.4f}, F1: {f1_cid_mean:.4f}, Precison: {p_cid_mean:.4f}, Recall: {r_cid_mean:.4f}, AUROC: {auroc_cid_mean:.4f},  #Med: {all_med:.4f} ")
    print(f"ATC3 → DDI: {ddi_rate:.4f}, Jaccard: {ja_atc_mean:.4f}, PRAUC: {prauc_atc_mean:.4f}, F1: {f1_atc_mean:.4f}, Precison: {p_atc_mean:.4f}, Recall: {r_atc_mean:.4f}, AUROC: {auroc_atc_mean:.4f}, #Med: {all_med:.4f}")
    


    return (ddi_rate,
            ja_cid_mean, prauc_cid_mean, p_cid_mean, r_cid_mean, f1_cid_mean, auroc_cid_mean,
            ja_atc_mean, prauc_atc_mean, p_atc_mean, r_atc_mean, f1_atc_mean, auroc_atc_mean,
            med_cnt/visit_cnt, val_loss_avg)

def main(args):

    # load data
    data_path = "/data/MMM.u2/mcwon/SafeDrug/data/output/records_final.pkl"
    voc_path = "/data/MMM.u2/mcwon/SafeDrug/data/output/voc_final.pkl"

    ddi_adj_path = "/data/MMM.u2/mcwon/SafeDrug/data/output/ddi_A_final.pkl"
    ddi_mask_path = "/data/MMM.u2/mcwon/SafeDrug/data/output/ddi_mask_H.pkl"
    molecule_path = "/data/MMM.u2/mcwon/SafeDrug/data/output/cidtoSMILES.pkl"
    
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            torch.cuda.set_device(device) 
        except RuntimeError:
            print("GPU out of memory! Run in CPU mode.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        

    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    molecule = dill.load(open(molecule_path, "rb"))

    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    evaluate_ground_truth_ddi_rate(data, ddi_adj)
    evaluate_ground_truth_ddi_rate_per_admission(data, ddi_adj)
    
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]

    elf_dim = select_model_output_dim(model_name_ELF)

    
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    model = MMM(
        voc_size,
        ddi_adj,
        ddi_mask_H,
        emb_dim=args.dim,
        elf_in_dim=elf_dim,
        device=device,
    )
    

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, "rb")))
        model.to(device=device)
        tic = time.time()

        result = []

        for _ in range(10):
            test_sample = np.random.choice(data_test, round(len(data_test) * 0.8), replace=True)
            
            ddi, ja_c, pr_c, p_c, r_c, f1_c, roc_c, ja_a, pr_a, p_a, r_a, f1_a, roc_a, avg_med, val_loss = eval(
                model, test_sample, voc_size, 0, med_voc, ddi_adj
            )
            
            result.append([ddi, ja_c, f1_c, pr_c, roc_c, ja_a, f1_a, pr_a, roc_a, avg_med])  

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        metrics = [
            "DDI Rate (CID)",
            "Jaccard (CID)", "F1-score (CID)", "PRAUC (CID)", "AUROC (CID)",
            "Jaccard (ATC3)", "F1-score (ATC3)", "PRAUC (ATC3)", "AUROC (ATC3)",
            "Avg # Drugs"
        ]

        outstring = "\n=== Performance (mean ± std) ===\n"
        for name, m, s in zip(metrics, mean, std):
            outstring += f"{name}: {m:.4f} ± {s:.4f}\n"

        print(outstring)
        print("test time: {:.2f} seconds".format(time.time() - tic))
        return


    model.to(device=device)
    
    optimizer = Adam(list(model.parameters()), lr=args.lr)
    history = defaultdict(list)
   
    best_epoch, best_loss = 0, inf 
    
    EPOCH = 150
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch + 1))

        model.train()
        loss_sum, count = 0.0, 0 
        for step, input in enumerate(data_train):

            for idx, adm in enumerate(input):
                seq_input = input[: idx + 1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx2, item in enumerate(adm[2]):
                    loss_multi_target[0][idx2] = item

                result, loss_ddi = model(seq_input)

                loss_bce = F.binary_cross_entropy_with_logits(
                    result, torch.FloatTensor(loss_bce_target).to(device)
                )
                loss_multi = F.multilabel_margin_loss(
                    F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
                )
                
                
                result = F.sigmoid(result).detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score([[y_label.tolist()]], path=ddi_adj_path)
                
                if current_ddi_rate <= args.target_ddi:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = max(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
                    loss = (
                        beta * (0.95 * loss_bce + 0.05 * loss_multi)
                        + (1 - beta) * loss_ddi
                    )
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                loss_sum += loss.item()
                count += 1

            llprint("\rtraining step: {} / {}".format(step + 1, len(data_train)))

        avg_epoch_loss = loss_sum / count
        print("\nAverage Loss in Epoch {}: {:.6f}".format(epoch + 1, avg_epoch_loss))
    
        tic2 = time.time()
        
        ddi, ja_c, pr_c, p_c, r_c, f1_c, roc_c, ja_a, pr_a, p_a, r_a, f1_a, roc_a, avg_med, val_loss = eval(
            model, data_eval, voc_size, epoch, med_voc, ddi_adj
        )
        print(
            "training time: {}, evaluation time: {}".format( 
                time.time() - tic, time.time() - tic2
            )
        )

        history["ddi_rate"].append(ddi)
        history["adm_ja_CID"].append(ja_c); history["adm_ja_ATC3"].append(ja_a)
        history["adm_prauc_CID"].append(pr_c); history["adm_prauc_ATC3"].append(pr_a)
        history["adm_f1_CID"].append(f1_c); history["adm_f1_ATC3"].append(f1_a)
        history["med"].append(avg_med)
        history["adm_auroc_CID"].append(roc_c); history["adm_auroc_ATC3"].append(roc_a)
        history["prauc_cid"].append(pr_c); history["prauc_atc"].append(pr_a)
        history["loss"].append(avg_epoch_loss)
        history["val_loss"].append(val_loss)

        torch.save(
            model.state_dict(),
            open(
                os.path.join(
                    "saved",
                    args.model_name,
                    "Epoch_{}_TARGET_{:.2}_JA-ATC3_{:.4f}_JA-CID_{:.4f}_DDI_{:.4f}.model".format(
                        epoch+1, args.target_ddi, ja_a, ja_c, ddi  
                    ),
                ),
                "wb",
            ),
        )

        if val_loss < best_loss:
            best_epoch = epoch + 1
            best_loss = val_loss
            
       

        print("best_epoch: {}".format(best_epoch))

    dill.dump(
        history,
        open(
            os.path.join(
                "saved", args.model_name, "history_{}.pkl".format(args.model_name)
            ),
            "wb",
        ),
    )

if __name__ == "__main__":
    # six random seed setting
    seed_sets = [
        (1001, 1002, 1003, 1004),
        (2001, 2002, 2003, 2004),
        (3001, 3002, 3003, 3004),
        (4001, 4002, 4003, 4004),
        (5001, 5002, 5003, 5004),
        (6001, 6002, 6003, 6004),
    ]

    for idx, (s1, s2, s3, s4) in enumerate(seed_sets, 1):
        print(f"\n=== Running experiment Set#{idx} ===")

        torch.manual_seed(s1)
        np.random.seed(s2)
        random.seed(s3)
        torch.cuda.manual_seed_all(s4)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        model_name = f"MMM_Set#{idx}_0.04"
        
        model_name_ELF = 'efficientnet_v2_l'

        if not os.path.exists(os.path.join("saved", model_name)):
            os.makedirs(os.path.join("saved", model_name))
            
        # args
        parser = argparse.ArgumentParser()
        parser.add_argument("--Test", action="store_true", default=False, help="test mode")
        parser.add_argument("--model_name", type=str, default=model_name, help="model name")
        parser.add_argument("--CNN_model", type=str, default=model_name_ELF, help="CNN model")
        parser.add_argument("--resume_path", type=str, default=None, help="resume path")
        parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
        parser.add_argument("--target_ddi", type=float, default=0.04, help="target ddi")
        parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
        parser.add_argument("--dim", type=int, default=64, help="dimension")
        parser.add_argument("--cuda", type=int, default=1, help="which cuda")

        args = parser.parse_args()

        if len(sys.argv) > 1:
            args = parser.parse_args()
            print(f"\n>> Test mode: Testing {args.resume_path}")
            main(args)
            sys.exit(0)
        

        print(f"\n==== Start Seed Set #{idx} ====")
        main(args)