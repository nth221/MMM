from xml.dom.pulldom import ErrorHandler
import pandas as pd
import dill
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS
import os
import pickle
from tqdm import tqdm


#random seed
import random
random.seed(2048)
np.random.seed(2048)

##### process medications #####
# load med data
def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={"NDC": "category"})
    
    med_pd.drop(
        columns=[
            "ROW_ID", "DRUG_TYPE", "DRUG_NAME_POE", "DRUG_NAME_GENERIC",
            "FORMULARY_DRUG_CD", "PROD_STRENGTH", "DOSE_VAL_RX",
            "DOSE_UNIT_RX", "FORM_VAL_DISP", "FORM_UNIT_DISP", "GSN",
            "FORM_UNIT_DISP", "ROUTE", "ENDDATE"
        ],
        axis=1,
        inplace=True,
    )
    med_pd.drop(index=med_pd[med_pd["NDC"] == "0"].index, axis=0, inplace=True)
    med_pd.fillna(method="pad", inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd["ICUSTAY_ID"] = med_pd["ICUSTAY_ID"].astype("int64")
    med_pd["STARTDATE"] = pd.to_datetime(
        med_pd["STARTDATE"], format="%Y-%m-%d %H:%M:%S"
    )
    med_pd.sort_values(
        by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE"], inplace=True
    )
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.drop(columns=["ICUSTAY_ID"])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


# NDC - CID - SMILES mapping
def NDCtoCID_Mapping(med_pd, ndc2cid_csv_path):
    ndc2cid_df = pd.read_csv(ndc2cid_csv_path, dtype=str)
    merged_df = med_pd.merge(ndc2cid_df, on="NDC", how="left")
    merged_df = merged_df.dropna(subset=["CID"])
    merged_df["CID"] = merged_df["CID"].apply(lambda x: f"CID{x.zfill(9)}")
    
    smiles_max_len = 400
    if "SMILES" in merged_df.columns:
        merged_df = merged_df[merged_df["SMILES"].str.len() <= smiles_max_len]

    return merged_df


# RXCUI -> ATC3 mapping
def codeMapping2atc4(med_pd):
    with open(ndc2RXCUI_file, "r") as f:
        ndc2RXCUI = eval(f.read())
    med_pd["RXCUI"] = med_pd["NDC"].map(ndc2RXCUI)
    med_pd.dropna(inplace=True)

    RXCUI2atc4 = pd.read_csv(RXCUI2atc4_file)
    RXCUI2atc4 = RXCUI2atc4.drop(columns=["YEAR", "MONTH", "NDC"])
    RXCUI2atc4.drop_duplicates(subset=["RXCUI"], inplace=True)
    med_pd.drop(index=med_pd[med_pd["RXCUI"].isin([""])].index, axis=0, inplace=True)

    med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(RXCUI2atc4, on=["RXCUI"])
    med_pd.drop(columns=["NDC", "RXCUI"], inplace=True)
    med_pd["ATC3"] = med_pd["ATC4"].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd


# medication data merge
def Merge_on_RXCUI(med_pd, RXCUI2atc3_file):
    rxcui_atc_df = pd.read_csv(RXCUI2atc3_file)
    med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
    rxcui_atc_df["RXCUI"] = rxcui_atc_df["RXCUI"].astype("int64")
    med_df = med_pd.merge(rxcui_atc_df, on="RXCUI", how="left")
    med_df = med_df.dropna(subset=["ATC3"])
    return med_df


# CID -> ATC3 mapping dictionary
def MakeDictionaryCIDtoATC3(med_pd):
    filtered_df = med_pd.dropna(subset=["CID", "ATC3"]).drop_duplicates(subset=["CID"])
    cid_to_atc3 = dict(zip(filtered_df["CID"], filtered_df["ATC3"]))

    with open("/data/MMM.u2/mcwon/SafeDrug/data/output/cid_to_atc3.pkl", "wb") as f:
        pickle.dump(cid_to_atc3, f)
        
    return cid_to_atc3


# visit >= 2
def process_visit_lg2(med_pd):
    a = (
        med_pd[["SUBJECT_ID", "HADM_ID"]]
        .groupby(by="SUBJECT_ID")["HADM_ID"]
        .unique()
        .reset_index()
    )
    a["HADM_ID_Len"] = a["HADM_ID"].map(lambda x: len(x))
    a = a[a["HADM_ID_Len"] > 1]
    return a


# most common medications
def filter_300_most_med(med_pd):
    med_count = (
        med_pd.groupby(by=["CID"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    med_pd = med_pd[med_pd["CID"].isin(med_count.loc[:299, "CID"])]

    return med_pd.reset_index(drop=True)


##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=["SEQ_NUM", "ROW_ID"], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=["SUBJECT_ID", "HADM_ID"], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)
    

    def filter_2000_most_diag(diag_pd):
        diag_count = (
            diag_pd.groupby(by=["ICD9_CODE"])
            .size()
            .reset_index()
            .rename(columns={0: "count"})
            .sort_values(by=["count"], ascending=False)
            .reset_index(drop=True)
        )
        diag_pd = diag_pd[diag_pd["ICD9_CODE"].isin(diag_count.loc[:1999, "ICD9_CODE"])] 

        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)
    return diag_pd


##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={"ICD9_CODE": "category"})
    pro_pd.drop(columns=["ROW_ID"], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], inplace=True)
    pro_pd.drop(columns=["SEQ_NUM"], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def filter_1000_most_pro(pro_pd):
    pro_count = (
        pro_pd.groupby(by=["ICD9_CODE"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    pro_pd = pro_pd[pro_pd["ICD9_CODE"].isin(pro_count.loc[:1000, "ICD9_CODE"])] 

    return pro_pd.reset_index(drop=True)


def combine_process(med_pd, diag_pd, pro_pd, adm_pd):  

    med_pd_key = med_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    diag_pd_key = diag_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    pro_pd_key = pro_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    combined_key = combined_key.merge(pro_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")

    diag_pd = diag_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    med_pd = med_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    pro_pd = pro_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")

    diag_pd = diag_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"].unique().reset_index()
    
    med_pd = med_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])[["CID"]].agg(lambda x: list(sorted(set(x)))).reset_index()
    pro_pd = pro_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"].unique().reset_index().rename(columns={"ICD9_CODE": "PRO_CODE"})

    med_pd["CID"] = med_pd["CID"].map(lambda x: list(x))
    pro_pd["PRO_CODE"] = pro_pd["PRO_CODE"].map(lambda x: list(x))

    data = diag_pd.merge(med_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    data = data.merge(pro_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    data["CID_num"] = data["CID"].map(lambda x: len(x))

    
    adm_pd["ADMITTIME"] = pd.to_datetime(adm_pd["ADMITTIME"])
    data = data.merge(adm_pd[["SUBJECT_ID", "HADM_ID", "ADMITTIME"]], on=["SUBJECT_ID", "HADM_ID"], how="left")

    data = data.sort_values(by=["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)

    print("sorting complete !")
    print(data[["SUBJECT_ID", "HADM_ID", "ADMITTIME", "ICD9_CODE"]].head(10))
    print(data[["CID", "SUBJECT_ID"]].head(10))
    
    return data



def statistics(data):
    print("#patients ", data["SUBJECT_ID"].unique().shape)
    print("#clinical events ", len(data))

    diag = data["ICD9_CODE"].values
    med = data["CID"].values
    pro = data["PRO_CODE"].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    print("#diagnosis(진단개수)", len(unique_diag))
    print("#med(약물개수) ", len(unique_med))
    print("#procedure(수술개수)", len(unique_pro))

    (
        avg_diag,
        avg_med,
        avg_pro,
        max_diag,
        max_med,
        max_pro,
        cnt,
        max_visit,
        avg_visit,
    ) = [0 for i in range(9)]

    for subject_id in data["SUBJECT_ID"].unique():
        item_data = data[data["SUBJECT_ID"] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row["ICD9_CODE"]))
            y.extend(list(row["CID"]))
            z.extend(list(row["PRO_CODE"]))
        x, y, z = set(x), set(y), set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print("#avg of diagnoses ", avg_diag / cnt)
    print("#avg of medicines ", avg_med / cnt)
    print("#avg of procedures ", avg_pro / cnt)
    print("#total visits", cnt)
    print("#avg of vists ", avg_visit / len(data["SUBJECT_ID"].unique()))

    print("#max of diagnoses ", max_diag)
    print("#max of medicines ", max_med)
    print("#max of procedures ", max_pro)
    print("#max of visit ", max_visit)


#### make final record ####
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc() 
    pro_voc = Voc()

    for _, row in df.iterrows():
        diag_voc.add_sentence(sorted(row["ICD9_CODE"]))
        med_voc.add_sentence(sorted(row["CID"]))
        pro_voc.add_sentence(sorted(row["PRO_CODE"]))

    dill.dump(
        obj={
            "diag_voc": diag_voc,
            "med_voc": med_voc,
            "pro_voc": pro_voc,
        },
        file=open(vocabulary_file, "wb"),
    )
    
    
    return diag_voc, med_voc, pro_voc


# create final records
def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = []  
    all_p = {} 
    debug_limit = 5 

    for idx, subject_id in enumerate(sorted(df["SUBJECT_ID"].unique())):
        item_df = df[df["SUBJECT_ID"] == subject_id]
        patient = []
        for _, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row["ICD9_CODE"]])
            admission.append([pro_voc.word2idx[i] for i in row["PRO_CODE"]])
            admission.append([med_voc.word2idx[i] for i in row["CID"]])
            patient.append(admission)
        records.append(patient)
        all_p[subject_id] = patient

        if idx < debug_limit:
            print(f"debugging - SUBJECT_ID: {subject_id}")
            print(all_p[subject_id])
        elif idx == debug_limit:
            print(f"... 생략 중 (앞 {debug_limit}명까지만 출력) ...")

    dill.dump(obj=records, file=open(ehr_sequence_file, "wb"))
    
    return records

def get_ddi_mask(cid_to_smiles, med_voc):
    total_cids = len(med_voc.idx2word)
    print(f"Total CIDs to process: {total_cids}")

    fraction = []
    success_count = 0
    
    for idx, cid in tqdm(med_voc.idx2word.items(), desc="BRICS fragment 생성 중", mininterval=0.1, dynamic_ncols=True):
        tempF = set()
        smi = cid_to_smiles.get(str(cid), None)

        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                frags = BRICS.BRICSDecompose(Chem.MolFromSmiles(smi))
                if frags:
                    tempF.update(frags)
                    success_count += 1
        except Exception as e:
            tqdm.write(f"BRICS 실패 (CID {cid}) → {e}")

        fraction.append(tempF)

    print(f"SMILES 성공적으로 처리된 CID 수: {success_count}/{total_cids}")

    fracSet = list(set().union(*fraction))
    frac_index_map = {frac: idx for idx, frac in enumerate(fracSet)}

    ddi_mask = np.zeros((len(med_voc.idx2word), len(fracSet)))
    for i, fragList in tqdm(enumerate(fraction), total=len(fraction), desc="DDI mask 행렬 생성 중", mininterval=0.1, dynamic_ncols=True):
        for frag in fragList:
            j = frac_index_map.get(frag)
            if j is not None:
                ddi_mask[i, j] = 1

    return ddi_mask

def get_ddi_matrix(med_voc, ddi_table_csv):
    ddi_df = pd.read_csv(ddi_table_csv)
    vocab_size = len(med_voc.idx2word)

    ddi_adj = np.zeros((vocab_size, vocab_size))
    ddi_severity_adj = np.zeros((vocab_size, vocab_size))

    matched_pairs = 0
    skipped_pairs = 0

    for _, row in tqdm(ddi_df.iterrows(), total=len(ddi_df), desc="DDI 행렬 생성 중"):
        cid1 = row["STITCH 1"]
        cid2 = row["STITCH 2"]
        count = row["SideEffectCount"]

        if cid1 in med_voc.word2idx and cid2 in med_voc.word2idx:
            i = med_voc.word2idx[cid1]
            j = med_voc.word2idx[cid2]
            ddi_adj[i, j] = ddi_adj[j, i] = 1
            ddi_severity_adj[i, j] = ddi_severity_adj[j, i] = count
            matched_pairs += 1
        else:
            skipped_pairs += 1

    print(f"DDI 쌍: {matched_pairs}")
    print(f"DDI 아닌 쌍: {skipped_pairs}")

    return ddi_adj, ddi_severity_adj


def make_cid2smiles_csv(vocabulary_file, med_pd, output_csv_path):
    
    with open(vocabulary_file, "rb") as f:
        vocabs = dill.load(f)

    if "med_voc" not in vocabs:
        raise ValueError("vocabulary 파일에 'med_voc'가 없습니다.")

    med_voc = vocabs["med_voc"]

    if hasattr(med_voc, "idx2word"):
        unique_cids = set(med_voc.idx2word.values())
    else:
        raise ValueError("med_voc에 idx2word 속성이 없습니다.")

    print(f"med_voc에서 추출한 CID 개수: {len(unique_cids)}")

    cid_smiles_df = med_pd[["CID", "SMILES"]].drop_duplicates()
    cid_smiles_df = cid_smiles_df[cid_smiles_df["CID"].isin(unique_cids)]
    cid_smiles_df = cid_smiles_df.dropna()

    print(f"SMILES가 매핑된 CID 개수: {len(cid_smiles_df)}")

    cid_smiles_df.to_csv(output_csv_path, index=False)
    print(f"CID-SMILES CSV 저장 완료: {output_csv_path}")



def compute_medication_imbalance(records, med_voc,
                                 failed_cids_path="/data/MMM.u2/mcwon/SafeDrug/data/input/failed_cids_real.csv"):
    total_counts = np.zeros(len(med_voc.idx2word))
    total_visits = 0

    for patient in records:
        for visit in patient:
            med_list = visit[2] 
            total_visits += 1
            for med in med_list:
                total_counts[med] += 1

    pos_ratios = total_counts / total_visits

    med_ids = [med_voc.idx2word[i] for i in range(len(med_voc.idx2word))]
    df = pd.DataFrame({
        'CID': med_ids,
        'Count': total_counts.astype(int),
        'count_Ratio': pos_ratios
    })

    df_sorted = df.sort_values(by='count_Ratio', ascending=False).reset_index(drop=True)
    df_sorted['Percentile_Rank'] = df_sorted.index / len(df_sorted) * 100
    df_sorted['Rank'] = df_sorted.index + 1

    top_10_percent_cutoff = int(len(df_sorted) * 0.1)
    top_10_percent = df_sorted.head(top_10_percent_cutoff)
    bottom_10_percent = df_sorted.tail(top_10_percent_cutoff)

    print(f"총 방문 수: {total_visits}")
    print(f"총 약물 개수: {len(med_voc.idx2word)}")
    print("\n[상위 10% 약물]")
    print(top_10_percent)

    print("\n[하위 10% 약물]")
    print(bottom_10_percent)

    print("\n양성 비율이 1% 미만인 약물 수:", (df_sorted['count_Ratio'] < 0.01).sum())
    print("양성 비율이 50% 이상인 약물 수:", (df_sorted['count_Ratio'] > 0.5).sum())

    try:
        failed_cids_df = pd.read_csv(failed_cids_path)
        failed_cids = failed_cids_df['CID'].astype(str).tolist()

        df_sorted['CID'] = df_sorted['CID'].astype(str)
        failed_info = df_sorted[df_sorted['CID'].isin(failed_cids)]

        print(f"\n[실패한 CID {len(failed_cids)}개 중, 데이터프레임에 존재하는 수: {len(failed_info)}개]")
        print(failed_info[['CID', 'Count', 'count_Ratio', 'Percentile_Rank', 'Rank']])

    except Exception as e:
        print(f"\n 실패한 CID 파일을 불러오는 데 문제가 발생했습니다: {e}")

    return df_sorted

def filter_300_most_med_statistics(med_pd):
    cid_usage_csv="/data/MMM.u2/mcwon/SafeDrug/data/input/top300_cid_usage.csv"
    top300_df = pd.read_csv(cid_usage_csv, dtype={"CID": str})

    top300_cids = top300_df["CID"].tolist()

    med_pd = med_pd[med_pd["CID"].isin(top300_cids)].reset_index(drop=True)
    
    print(f"✔ top300_cid_usage.csv 기준으로 약물 필터링 완료 ({len(top300_cids)}개)")
    print(f"  → 필터링된 med_pd의 행 수: {len(med_pd)}")
    
    return med_pd



if __name__ == "__main__":

    # input files
    med_file = "/data/MMM.u2/mcwon/SafeDrug/SafeDrug Data/predata/PRESCRIPTIONS.csv"
    diag_file = "/data/MMM.u2/mcwon/SafeDrug/SafeDrug Data/DIAGNOSES_ICD.csv"
    procedure_file = "/data/MMM.u2/mcwon/SafeDrug/SafeDrug Data/PROCEDURES_ICD.csv"
    RXCUI2atc4_file = "/data/MMM.u2/mcwon/SafeDrug/data/input/RXCUI2atc4.csv"
    ndc2RXCUI_file = "/data/MMM.u2/mcwon/SafeDrug/data/input/ndc2RXCUI.txt"
    ddi_file = "/data/MMM.u2/mcwon/SafeDrug/data/input/drug-DDI.csv"
    drugbankinfo = "/data/MMM.u2/mcwon/SafeDrug/data/input/drugbank_drugs_info.csv"
    NDC_CID_SMILES_file = "/data/MMM.u2/mcwon/SafeDrug/data/input/ndc2cid_smiles.csv"
    NDC_RXCUI_CID_SMILES_file = "/data/MMM.u2/mcwon/SafeDrug/data/input/ndc2rxcui2inn2cid2smiles.csv"
    RXCUI2atc3_file = "/data/MMM.u2/mcwon/SafeDrug/data/input/RXCUI2atc3.csv"
    # output files
    ddi_adjacency_file = "/data/MMM.u2/mcwon/SafeDrug/data/output/ddi_A_final.pkl"
    ehr_adjacency_file = "/data/MMM.u2/mcwon/SafeDrug/data/output/ehr_adj_final.pkl"
    ehr_sequence_file = "/data/MMM.u2/mcwon/SafeDrug/data/output/records_final.pkl"
    vocabulary_file = "/data/MMM.u2/mcwon/SafeDrug/data/output/voc_final.pkl"
    ddi_mask_H_file = "/data/MMM.u2/mcwon/SafeDrug/data/output/ddi_mask_H.pkl"
    ddi_table_csv = "/data/MMM.u2/mcwon/SafeDrug/data/output/ddi_detailed_table.csv" # make ddi table 
    cid2smiles_file = "/data/MMM.u2/mcwon/SafeDrug/data/output/cidtoSMILES.pkl"


    # for med
    med_pd = med_process(med_file)
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(
        med_pd_lg2[["SUBJECT_ID"]], on="SUBJECT_ID", how="inner"
    ).reset_index(drop=True)
    
    #med processing
    med_pd = NDCtoCID_Mapping(med_pd, NDC_RXCUI_CID_SMILES_file)


    # Merge: NDC_RXCUI_CID_SMILES_file - RXCUI2atc3_file
    med_pd = Merge_on_RXCUI(med_pd, RXCUI2atc3_file)
    med_pd = med_pd.dropna(subset=['ATC3'])
    
    excepted_CIDs = {
        "CID005311498",   # gbw missing
        "CID000061739",   # 상처 소독제
        "CID000003821",   # 전신마취제
        "CID000004943",    # 전신마취제
        "CID005284607",     # 비타민
        "CID000517044",     # 포타슘
        "CID000517045",      # 소듐 
        "CID000005510",    # 향진균제 
        "CID005288783", 
        "CID000057469",
        "CID000441244",
        "CID005311051",
        "CID000006256",
        "CID000002216",
        "CID000004100",
        "CID000039468",
        "CID000068844",
        "CID005282226",
        "CID005284549",
        "CID005311027",
        "CID005311221",
        "CID000002905",
        "CID000005593",
        "CID000004935"      # 국소 마취제 
        }

    failed_cids_file = "/data/MMM.u2/mcwon/SafeDrug/data/input/failed_cids_final.csv"
    failed_cids_df = pd.read_csv(failed_cids_file)
    failed_cids = set(failed_cids_df["CID"].astype(str).str.strip())

    all_excluded_CIDs = excepted_CIDs.union(failed_cids)
    med_pd = med_pd[~med_pd["CID"].isin(all_excluded_CIDs)]
    med_pd = filter_300_most_med_statistics(med_pd)

    cid_to_smiles = dict(
    sorted(med_pd[["CID", "SMILES"]].drop_duplicates().dropna().values.tolist())
    )
    with open(cid2smiles_file, "wb") as f:
        pickle.dump(cid_to_smiles, f)

    cid_to_atc3 = MakeDictionaryCIDtoATC3(med_pd)
    
    print("complete medication processing")

    diag_pd = diag_process(diag_file)

    print("complete diagnosis processing")

    pro_pd = procedure_process(procedure_file)

    print("complete procedure processing")

    # combine
    adm_pd = pd.read_csv("/data/MMM.u2/mcwon/SafeDrug/data/input/ADMISSIONS.csv")
    data = combine_process(med_pd, diag_pd, pro_pd, adm_pd)
    
    lg2_subject_ids = process_visit_lg2(data)[["SUBJECT_ID"]]
    data = data.merge(lg2_subject_ids, on="SUBJECT_ID", how="inner").reset_index(drop=True)
    
    statistics(data)
    print("complete combining")

    # create vocab
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    print("obtain voc")

    # create ehr sequence data
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)
    print("obtain ehr sequence data")

    make_cid2smiles_csv(
        vocabulary_file=vocabulary_file,
        med_pd=med_pd,
        output_csv_path="/data/MMM.u2/mcwon/SafeDrug/ELF_data/med_info(ELF).csv"
    )

    pos_ratio_all = compute_medication_imbalance(records, med_voc,
                                failed_cids_path="/data/MMM.u2/mcwon/SafeDrug/data/input/failed_cids_final.csv")
    # get ddi_mask_H
    ddi_mask_H = get_ddi_mask(cid_to_smiles, med_voc)
    dill.dump(ddi_mask_H, open(ddi_mask_H_file, "wb"))

    ddi_adj, ddi_severity_adj = get_ddi_matrix(med_voc, ddi_table_csv)
    dill.dump(ddi_adj, open(ddi_adjacency_file, "wb"))
    
