import os
import torch
import torchvision.transforms as transforms
import torch.utils.data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trans_224 = transforms.Compose([
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])
trans_256 = transforms.Compose([
    transforms.Resize((256, 256)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
trans_DRCT_imagenet_224 = transforms.Compose([
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# y = model(img).sigmoid()  224  1
def get_UnivFD_model():
    from UnivFD import get_model
    model = get_model("CLIP:ViT-L/14")
    model_path = "/UniversalFakeDetect-main/pretrained_weights/fc_weights.pth"
    state_dict = torch.load(model_path, map_location='cpu')
    model.fc.load_state_dict(state_dict)
    model.to(device).eval()
    return model, trans_224, 1


# y = model(img).sigmoid()  256  1
def get_CNNSpot_model():
    from CNNSpot.resnet import resnet50
    CNNSpot_model = resnet50(num_classes=1)
    CNNSpot_path = '/CNNSpot/weights/blur_jpg_prob0.5.pth'
    CNNSpot_state_dict = torch.load(CNNSpot_path, map_location='cpu')
    CNNSpot_model.load_state_dict(CNNSpot_state_dict['model'])
    CNNSpot_model.to(device).eval()
    model = CNNSpot_model
    return model, trans_256, 1


# y = model(img).sigmoid()  256 1
def get_ResNetND_model():
    from ResNetND.resnet50nodown import resnet50nodown
    ProGAN_path = '/ProGAN/weights/gandetection_resnet50nodown_progan.pth'
    ProGAN_model = resnet50nodown(device, ProGAN_path)
    return ProGAN_model, trans_256, 1


# model(img).softmax(dim=1)[:, -1:]  center 224 trans_DRCT_imagenet_224 2
def get_DRCT_model(dateset="DRCT", model_Flag="ConvB"):  # DRCT GenImage ConvB UnivFD
    from DRCT.models import get_models
    if "DRCT" in dateset:
        if "ConvB" in model_Flag:
            model = get_models(model_name="convnext_base_in22k", num_classes=2, freeze_extractor=True, embedding_size=1024)
            m_path = "/xcy/DRCT_2M/pretrained/DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth"
        else:
            model = get_models(model_name="clip-ViT-L-14", num_classes=2, freeze_extractor=True, embedding_size=1024)
            m_path = "/xcy/DRCT_2M/pretrained/DRCT-2M/sdv14/clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth"
    else:
        if "ConvB" in model_Flag:
            model = get_models(model_name="convnext_base_in22k", num_classes=2, freeze_extractor=True, embedding_size=1024)
            m_path = "/xcy/DRCT_2M/pretrained/GenImage/sdv14/convnext_base_in22k_224_drct_amp_crop/last_acc0.9991.pth"
        else:
            model = get_models(model_name="clip-ViT-L-14", num_classes=2, freeze_extractor=True, embedding_size=1024)
            m_path = "/xcy/DRCT_2M/pretrained/GenImage/sdv14/clip-ViT-L-14_224_drct_amp_crop/2_acc0.9558.pth"
    model.load_state_dict(torch.load(m_path, map_location='cpu'), strict=False)
    model.to(device).eval()
    return model, trans_DRCT_imagenet_224, 2


# y = model(img).softmax(dim=1)[:, -1:]  224 2
def get_FatFormer_model():
    from FatFormer import build_model
    import argparse
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--dataset_path', type=str, default='../dataset')
    parser.add_argument('--img_resolution', type=int, default=256)
    parser.add_argument('--crop_resolution', type=int, default=224)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--backbone', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_vit_adapter', type=int, default=3)
    parser.add_argument('--num_context_embedding', type=int, default=8)
    parser.add_argument('--init_context_embedding', type=str, default="")
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--clip_vision_width', type=int, default=1024)
    parser.add_argument('--frequency_encoder_layer', type=int, default=2)
    parser.add_argument('--decoder_layer', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default="")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int, help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    FAT_args = parser.parse_args()
    FAT_model = build_model(FAT_args)
    FAT_model = FAT_model.to(device)
    model_without_ddp = FAT_model
    checkpoint = torch.load("/xcy/DiFF/fatformer_4class_ckpt.pth", map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    FAT_model.eval()
    return FAT_model, trans_224, 2


def get_FatFormer_visual():
    from FatFormer import build_model_visual as build_model
    import argparse
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--dataset_path', type=str, default='../dataset')
    parser.add_argument('--img_resolution', type=int, default=256)
    parser.add_argument('--crop_resolution', type=int, default=224)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--backbone', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_vit_adapter', type=int, default=3)
    parser.add_argument('--num_context_embedding', type=int, default=8)
    parser.add_argument('--init_context_embedding', type=str, default="")
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--clip_vision_width', type=int, default=1024)
    parser.add_argument('--frequency_encoder_layer', type=int, default=2)
    parser.add_argument('--decoder_layer', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default="")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int, help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    FAT_args = parser.parse_args()
    FAT_model = build_model(FAT_args)
    FAT_model = FAT_model.to(device)
    model_without_ddp = FAT_model
    checkpoint = torch.load("/xcy/DiFF/fatformer_4class_ckpt.pth", map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    FAT_model.eval()
    return FAT_model, trans_224


def get_FatFormer_classifier():
    from FatFormer import build_model_classifier as build_model
    import argparse
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--dataset_path', type=str, default='../dataset')
    parser.add_argument('--img_resolution', type=int, default=256)
    parser.add_argument('--crop_resolution', type=int, default=224)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--backbone', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_vit_adapter', type=int, default=3)
    parser.add_argument('--num_context_embedding', type=int, default=8)
    parser.add_argument('--init_context_embedding', type=str, default="")
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--clip_vision_width', type=int, default=1024)
    parser.add_argument('--frequency_encoder_layer', type=int, default=2)
    parser.add_argument('--decoder_layer', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default="")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int, help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    FAT_args = parser.parse_args()
    FAT_model = build_model(FAT_args)
    FAT_model = FAT_model.to(device)
    model_without_ddp = FAT_model
    checkpoint = torch.load("/xcy/DiFF/fatformer_4class_ckpt.pth", map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    FAT_model.eval()
    return FAT_model, trans_224
