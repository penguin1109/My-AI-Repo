import torch.optim as optim

def build_model(*args):
    return

def build_optimizer(model, *args):
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    return optimizer

def build_scheduler(optimizer, *args):
    ## optimizer의 learning rate를 조건에 맞추어 스케줄링해야 하기 때문에 생성할 때 optimizer을 필요로 한다.
    return