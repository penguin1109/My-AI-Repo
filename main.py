import argparse, torch

def main():
    parser = argparse.ArgumentParser()

    ## model parameters
    parser.add_argument("--model_name", type = str, help = "Write the Model Type You Want To Use",
                        choices = ['attention', 'residual', 'siamese']) ## 네트워크 유형에 따라 데이터셋도 달라지게 됨
    # 현재 SOTA를 달성한 ViT와 efficientnet, resnext등을 사용해 보고자 한다.
    parser.add_argument("--bbone",type = str, help = "The CNN Backbone Model You Want To Use",
                        choices = ['resnext101_32x8d', 'efficientnetb0', 'vit_base_patch16_224', 'gluon_seresnext101_32x4d', 'resnet50'], default = 'efficientnetb0')

    ## training parameters
    parser.add_argument("--mode", "--m", type = int, help = "pretrain : 0 finetune : 1 predict : 2", default = 1)
    parser.add_argument("--img_size", type = int, default = 224)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--learning_rate", "--lr", type = float, default = 3e-4)
    parser.add_argument("--optimizer", type = str, default = 'AdamW')
    parser.add_argument("--scheduler", type = str, default = 'None') ## 처음에는 learning rate scheduler을 사용하지는 않음

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = main()
    optimizer = args.optimizer