from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms

from WideResNet_pytorch.wideresnet import WideResNet
from augmentations import augmentations, augmentations_all
PATH = "./ckpt/wrn40-2.ckpt"

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    return res

class AugMixData(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, js_loss=True, n_js=3, level=3, alpha=1, mixture_width=3, mixture_depth=0):
        self.dataset = dataset
        self.preprocess = preprocess
        self.js_loss = js_loss
        self.n_js = n_js
        self.level = level
        self.alpha = alpha
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.js_loss:
            xs = [self.preprocess(x), self.augmix(x)]
            while len(xs) < self.n_js:
                xs.append(self.augmix(x))
            return xs, y
        else:
            return self.augmix(x), y

    def __len__(self):
        return len(self.dataset)

    def augmix(self, img):
        aug_list = augmentations if True else augmentations_all
        ws = np.float32(np.random.dirichlet([self.alpha] * self.mixture_width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        mixed_image = torch.zeros_like(self.preprocess(img))
        for i in range(self.mixture_width):
            aug_img = img.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for d in range(depth):
                op = np.random.choice(aug_list)
                aug_img = op(aug_img, self.level)
            mixed_image += ws[i] * self.preprocess(aug_img)
        return m * self.preprocess(img) + (1-m) * mixed_image

def compute_js_loss(logits, targets, js_coeff=12.):
    logit, *aug_logits = logits
    loss = F.cross_entropy(logit, targets)
    p_augs = [ F.softmax(logit, dim=1) for logit in logits ]
    p_mixture = sum(p_augs) / float(len(p_augs))
    p_mixture = torch.clamp(p_mixture, 1e-7, 1).log()
    loss += js_coeff * sum([ F.kl_div(p_mixture, p, reduction="batchmean") for p in p_augs]) / float(len(p_augs))

    # logits_clean, logits_aug1, logits_aug2 = logits
    # loss = F.cross_entropy(logits_clean, targets)
    # p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1), F.softmax(logits_aug1, dim=1), F.softmax(logits_aug2, dim=1)
    # # Clamp mixture distribution to avoid exploding KL divergence
    # p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    # loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
    #             F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
    #             F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
    return loss

def test(model, test_data, eval_batch_size=512):
    model.eval()
    losses, acces = [], []
    for corruption in CORRUPTIONS:
        total_loss = 0.
        total_acc = 0.
        test_data.data = np.load('./data/cifar/CIFAR-100-C/%s.npy' % corruption)
        test_data.targets = torch.LongTensor(np.load('./data/cifar/CIFAR-100-C/labels.npy'))
        n_data = len(test_data)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=eval_batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True)
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.cuda(), targets.cuda()
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
                acc = accuracy(logits, targets)[0]
                total_loss += float(loss.data) * len(images) / n_data
                total_acc += float(acc.data) * len(images) / n_data
        losses.append(total_loss)
        acces.append(total_acc)
    return losses, acces
def main():
    # torch.manual_seed(2020)
    # np.random.seed(2020)
    epochs = 100
    js_loss = True
    batch_size = 256
    low_mem = False
    # 1. dataload
    # basic augmentation & preprocessing
    train_base_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4)
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
    ])
    # load data
    train_data = datasets.CIFAR100('./data/cifar', train=True, transform=train_base_aug, download=True)
    test_data = datasets.CIFAR100('./data/cifar', train=False, transform=preprocess, download=True)
    train_data = AugMixData(train_data, preprocess, js_loss=js_loss, n_js=3, level=3, alpha=1, mixture_width=3, mixture_depth=0)
    train_loader = torch.utils.data.DataLoader(
                   train_data,
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=4,
                   pin_memory=True)
    # 2. model
    # wideresnet 40-2
    model = WideResNet(depth=40, num_classes=100, widen_factor=2, drop_rate=0.0)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.SGD(
                  model.parameters(),
                  0.1,
                  momentum=0.9,
                  weight_decay=0.0005,
                  nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader), eta_min=1e-6, last_epoch=-1)

    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    # training model with cifar100
    losses = []
    top1s = []
    top5s = []
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            targets = targets.cuda()
            if js_loss:
                if low_mem:
                    logits = [ model(imgs.cuda()) for imgs in images ]
                else:
                    all_images = torch.cat(images, 0).cuda()
                    all_logits = model(all_images)
                    logits = torch.split(all_logits, images[0].size(0))
                loss = compute_js_loss(logits, targets, js_coeff=12.)
            else:
                images = images.cuda()
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
            # update
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            if i % 100 == 0 or i+1 == len(train_loader):
                print("[E{:d}-{:d}/{:d}]Train Loss: {:.4f}, lr: {:.6f}".format(
                                    epoch, i, len(train_loader), loss.item(), get_lr(optimizer)))

        top1, top5 = accuracy(logits[0].cpu().detach() if type(logits) == tuple else logits.cpu().detach(), targets.cpu().detach(), (1,5))
        top1s.append(top1)
        top5s.append(top5)
        # evaluate on cifar100-c
        test_losses, test_acces = test(model, test_data)
        print("--Evaluation-- E{:d}\n    [Training] top1: {:.2f}, top5: {:.2f}".format(epoch, top1, top5))
        print("    [Corrupted Test] mean_loss: {:.4f}, mean_acc: {:.2f}".format(np.mean(test_losses), np.mean(test_acces)))
        for test_loss, test_acc, corruption in zip(test_losses, test_acces, CORRUPTIONS):
            print("        ({:}) loss: {:.4f}, acc: {:.2f}".format(corruption, test_loss, test_acc))

        torch.save({
            "epoch": epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            "top1s": top1s,
            "top5s": top5s
        }, PATH)


if __name__=="__main__":
    main()
