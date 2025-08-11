'''
cnn_pretrain.py
ELF map -> pre-train model을 통해서 freeze embedding을 뽑기 위함.
CNN 사전 학습 모델을 이용해서 이미지 임베딩을 뽑는 코드.
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import pickle
import dill

# pretrained model name
model_name = 'efficientnet_v2_l'

# pooling: "max" or "mean"
pooling = 'max'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

transform = None

if 'resnet' in model_name:
    transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ])
elif 'efficientnet' in model_name:
    transform = transforms.Compose([
    transforms.Resize(480, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(480),  
    transforms.ToTensor(),  
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

elif 'mobilenet' in model_name:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

def load_model(model_name, pretrained=True):
    model_dict = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,

        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'efficientnet_b5': models.efficientnet_b5,
        'efficientnet_b6': models.efficientnet_b6,
        'efficientnet_b7': models.efficientnet_b7,
        'efficientnet_v2_m': models.efficientnet_v2_m,
        'efficientnet_v2_l': models.efficientnet_v2_l,
        
        'mobilenet_v3_S': models.mobilenet_v3_small
    }
    
    if model_name not in model_dict:
        raise ValueError(f"지원되지 않는 모델입니다: {model_name}. 사용 가능한 모델: {list(model_dict.keys())}")
    
    if pretrained:
        if "resnet" in model_name:
            if model_name == 'resnet50':
                weights = getattr(models, f"ResNet50_Weights").DEFAULT
            elif model_name == 'resnet101':                        
                weights = getattr(models, f"ResNet101_Weights").DEFAULT
            elif model_name == 'resnet152':                        
                weights = getattr(models, f"ResNet152_Weights").DEFAULT

        elif "efficientnet" in model_name:
            if model_name == 'efficientnet_v2_m':
                weights = getattr(models, f"EfficientNet_V2_M_Weights").DEFAULT
            elif model_name == 'efficientnet_v2_l':                        
                weights = getattr(models, f"EfficientNet_V2_L_Weights").DEFAULT
                
        elif "mobilenet" in model_name:
            if model_name == 'mobilenet_v3_S':
                weights = getattr(models, f"MobileNet_V3_Small_Weights").DEFAULT
        else:
            weights = None
    else:
        weights = None
    
    model = model_dict[model_name](weights=weights) 

    if "efficientnet" in model_name:
        model = torch.nn.Sequential(
            *list(model.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1))
        )
        feature_dim = 1280
    else:
        model = torch.nn.Sequential(*list(model.children())[:-1])
        feature_dim = 2048
    
    model.eval()
    return model, feature_dim

def get_molecule_embedding(folder_path, model, model_name, device):
    slice_embeddings = []
    slice_files = sorted(os.listdir(folder_path))

    for slice_file in slice_files:
        img_path = os.path.join(folder_path, slice_file)
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  

        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model(image)
            if "efficientnet" in model_name:
                embedding = embedding.squeeze(-1).squeeze(-1)  
                embedding = embedding.view(-1)  
            else:
                embedding = embedding.view(-1)

            slice_embeddings.append(embedding)
    
    if len(slice_embeddings) > 0:
        molecule_embedding = torch.stack(slice_embeddings)
        if pooling=='mean':
            mean_pooled_embedding = torch.mean(molecule_embedding, dim=0)
        elif pooling == 'max':
            mean_pooled_embedding, _ = torch.max(molecule_embedding, dim=0)
    else:
        mean_pooled_embedding = torch.zeros(model[-1].out_features)  
    
    return mean_pooled_embedding.cpu().numpy()

def clean_name(name):
    if name.endswith('_crop'):
        return name[:-5]
    return name

def sort_molecules_by_med_voc(molecule_folders, med_voc):
    def get_idx(x):
        key = clean_name(x)
        return med_voc.word2idx.get(key, float('inf')) 

    valid_folders = [f for f in molecule_folders if clean_name(f) in med_voc.word2idx]
    return sorted(valid_folders, key=get_idx)

def process_molecule_folder(root_folder, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_dim = load_model(model_name, pretrained=True)
    print(f"Feature dimension: {feature_dim}")
    model = model.to(device)
    
    molecule_embeddings_list = []

    
    with open('/data/MMM.u2/mcwon/SafeDrug/data/output/voc_final.pkl', 'rb') as f:
        voc_data = dill.load(f)

    med_voc = voc_data['med_voc']

    print(f"총 med_voc 약물 수: {len(med_voc.word2idx)}")

    molecule_fol = sorted(os.listdir(root_folder))
    molecule_folders = sort_molecules_by_med_voc(molecule_fol, med_voc)

    print(f"폴더에서 찾은 유효한 약물 폴더 수: {len(molecule_folders)}")

    found_set = set()
    missing_image_dirs = []

    for molecule in molecule_folders:
        molecule_path = os.path.join(root_folder, molecule)
        if not os.path.isdir(molecule_path):
            print(f"{molecule_path}는 폴더가 아님. 건너뜀.")
            continue

        molecule_name = molecule.replace('_crop', '')

        if molecule_name not in med_voc.word2idx:
            print(f"[경고] {molecule_name} not found in med_voc. Skipping.")
            continue

        image_files = [f for f in os.listdir(molecule_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) == 0:
            print(f"[주의] {molecule_name} 폴더에 이미지 없음. Skipping.")
            missing_image_dirs.append(molecule_name)
            continue
        print(f"Processing: {molecule_name} ({len(image_files)} slices)")
        embedding = get_molecule_embedding(molecule_path, model, model_name, device)
        molecule_embeddings_list.append((molecule_name, embedding))
        found_set.add(molecule_name)
        print(f"{molecule_name} Embedding Extracted ({feature_dim}D)")
    all_med_cids = set(med_voc.word2idx.keys())
    missing_cids = all_med_cids - found_set
    print("\n--- 요약 ---")
    print(f"총 med_voc 개수: {len(all_med_cids)}")
    print(f"임베딩 생성된 약물 수: {len(found_set)}")
    print(f"이미지가 없어 누락된 약물 수: {len(missing_image_dirs)}")
    print(f"누락된 약물 목록 (med_voc 기준 존재하지만 임베딩 안됨): {missing_cids}")
    print(f"실제로 저장된 임베딩 개수: {len(molecule_embeddings_list)}")

    return molecule_embeddings_list

def apply_reduction(embeddings_list, reducer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reducer = reducer.to(device)
    reduced_embeddings_list = []
    
    for molecule_name, embedding in embeddings_list:
        embedding_tensor = torch.tensor(embedding).float().unsqueeze(0).to(device)
        reduced_embedding = reducer(embedding_tensor).cpu().detach().numpy().flatten()
        reduced_embeddings_list.append((molecule_name, reduced_embedding))
    
    return reduced_embeddings_list



def select_model_output_dim(model_name):
    """
    지정한 모델 이름에 따른 출력 차원(feature_dim)을 반환합니다.
    """
    model_output_dims = {
        
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048,

        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
        'efficientnet_b3': 1536,
        'efficientnet_b4': 1792,
        'efficientnet_b5': 2048,
        'efficientnet_b6': 2304,
        'efficientnet_b7': 2560,
        'efficientnet_v2_m': 1280,
        'efficientnet_v2_l': 1280,

        'mobilenet_v3_S': 576,
        'mobilenet_v3_L': 960
    }

    if model_name not in model_output_dims:
        raise ValueError(f"지원되지 않는 모델입니다: {model_name}. 지원 가능한 모델: {list(model_output_dims.keys())}")
    
    return model_output_dims[model_name]


if __name__ == "__main__":
    root_folder = "/data/MMM.u2/mcwon/MIMIC-III_molden_IMGoutput_total_Cropped"

    embeddings_list = process_molecule_folder(root_folder, model_name)    

    with open(f"/data/MMM.u2/mcwon/SafeDrug/data/output/elf_embedding_mobilenet.pkl", "wb") as f:
        pickle.dump(embeddings_list, f)