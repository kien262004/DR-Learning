import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from networks.resnet_big import SupConResNet


def parse_option():
    parser = argparse.ArgumentParser('tsne inference')

    parser.add_argument('--ckpt', type=str, required=True,
                        help='path to checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'])
    parser.add_argument('--data_folder', type=str, default='./datasets/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--size', type=int, default=32)

    return parser.parse_args()


def set_loader(opt):

    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2675, 0.2565, 0.2761)

    transform = transforms.Compose([
        transforms.Resize((opt.size, opt.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if opt.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            root=opt.data_folder,
            train=True,
            transform=transform,
            download=True
        )
    elif opt.dataset == 'cifar100':
        dataset = datasets.CIFAR100(
            root=opt.data_folder,
            train=True,
            transform=transform,
            download=True
        )
    else:
        dataset = datasets.ImageFolder(
            root=opt.data_folder,
            transform=transform
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )

    return loader


def set_model(opt):

    model = SupConResNet(name=opt.model)

    ckpt = torch.load(opt.ckpt, map_location="cpu", weights_only=False)

    state_dict = ckpt['model']
    
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}

    model.load_state_dict(state_dict)

    model = model.encoder  # only encoder for feature extraction

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    model.eval()

    return model


def extract_features(loader, model):

    features = []
    labels = []

    with torch.no_grad():
        for images, target in loader:

            if torch.cuda.is_available():
                images = images.cuda()

            output = model(images)

            features.append(output.cpu().numpy())
            labels.append(target.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def plot_tsne(features, labels):

    print("Running t-SNE...")
    n_samples = features.shape[0]

    perplexity = min(30, (n_samples - 1) // 3)
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000,
        random_state=0
    )

    tsne_features = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))

    scatter = plt.scatter(
        tsne_features[:, 0],
        tsne_features[:, 1],
        c=labels,
        cmap="tab10",
        s=5
    )

    plt.colorbar(scatter)
    plt.title("t-SNE visualization of SupCon features")
    # plt.show()
    os.makedirs('visualize', exist_ok=True)

    plt.savefig('visualize/image.png', dpi=300)
    plt.close()


def main():
    opt = parse_option()
    loader = set_loader(opt)
    model = set_model(opt)
    features, labels = extract_features(loader, model)
    plot_tsne(features, labels)


if __name__ == "__main__":
    main()