from nbnet import NBNet
"""
contains the code for experimental denoiser models
UNet and the UNet family

TODO
(1) ResUNet
(2) UNet
(3) UNetPP
(4) AttUNet
(5) ASPPUNet 
(6) NBNet (O)
"""
NN_ARCHS = [
    "unet", "unetpp", "resunet", 
    "attunet", "asppunet", "nbnet"
]


def get_model(nn_arch , seg = seg, num_channels=1):
    if nn_arch == "unet8":
        return UNet(num_channels, num_of_layers=8, seg=seg)
    elif nn_arch == "unetpp":
        return UNetPP(num_channels, num_of_layers=12, seg=seg)
    elif nn_arch == "resunet":
        return ResUNet(num_channels, num_of_layers=17, seg=seg)
    elif nn_arch == "attunet":
        return AttUNet(num_channels, num_of_layers=25, seg=seg)
    elif nn_arch == "asppunet":
        return ASPPUNet(num_channels, num_channels, 8, seg=seg)
    elif nn_arch == "nbnet":
        return NBNet(num_channels, num_channels, 16, seg=seg)
    else:
        print("No supported NN architecture")
        return None
