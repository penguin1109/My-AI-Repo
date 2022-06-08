from tqdm import tqdm
import torch.cuda.amp as amp

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train() # 학습시키고자 하는 모델을 gradient update가 되도록 training mode로 변경
    scaler = amp.GradScaler()

    dataset_size = 0;running_loss = 0.0;
    train_loop = tqdm(enumerate(dataloader, total = len(dataloader), desc = 'Training'))
    for step, data in train_loop:
        image = data['image'].to(device) # 예측에 사용해야 하는 이미지 (그런데 siamese net을 사용하는 경우 triplet ranking loss를 위해 이미지가 3개 필요하고 아니면 그냥 1개만 사용
        label = data['label'].to(device) # 예상이지만 해당 얼굴 사진의 피사체의 나이 정보가 target으로 주어질 것이다.
