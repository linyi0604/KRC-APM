
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np


from config import Config
from utils import Logger, AverageMeter, evaluate, data_augmentadion_localizer, diceCoeff
from dataset.dataset import BUSIDataset_image_mask
import segmentation_models_pytorch as smp
from networks.networks import Localizer





def train_test(times):
    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id


    train_set = BUSIDataset_image_mask(
            config.image_path, config.mask_path, config.train_mapping_path,
            transform=data_augmentadion_localizer())

    test_set = BUSIDataset_image_mask(
        config.image_path, config.mask_path, config.test_mapping_path,
        transform=data_augmentadion_localizer(train=False))
    

    train_loader = DataLoader(train_set,
                            batch_size=config.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=config.num_workers)
    test_loader = DataLoader(test_set,
                            batch_size=config.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers)
    
    logger = Logger("./logs/localizer_%s.log"%times)
    localizer = Localizer().cuda()
    criterion = torch.nn.BCEWithLogitsLoss()
    metric = smp.utils.metrics.IoU(threshold=0.5)
    optimizer = torch.optim.Adam(localizer.parameters(), lr=config.localizer_learning_rate)
    best_mean = 0
    best_message = ""
    # train the confidence localizer
    for epoch in range(config.LOCALIZER_EPOCH):
        localizer.train()
        train_loss = AverageMeter()
        for step, (image, mask, label) in enumerate(train_loader):
        
            image = image.cuda()
            mask = mask.cuda()
            pre_mask = localizer(image)
            loss = criterion(pre_mask, mask)
            train_loss.update(loss.item(), image.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        message = ""
        message += "epoch: %2d  \n" % epoch
        message += "    train:  loss: %.5f \n" % train_loss.avg

        # test the localizer
        localizer.eval()
        test_loss = AverageMeter()
        test_iou = AverageMeter()
        test_dice = AverageMeter()
        with torch.no_grad():
            for step, (image, mask, label) in enumerate(test_loader):
                image = image.cuda()
                mask = mask.cuda()
                pre_mask = localizer(image)

                loss = criterion(pre_mask, mask)
                ious = metric(pre_mask.cpu(), mask.cpu())
                dice = diceCoeff(pre_mask.cpu(), mask.cpu())
                test_loss.update(loss.item(), image.size(0))
                test_iou.update(ious, image.size(0))
                test_dice.update(dice, image.size(0))

        message += "    test:  loss: %.5f  \n" % test_loss.avg
        message += "        iou \t dice_coefficient  \n"
        message += "        %.2f%% \t %.2f%%   \n" % (test_iou.avg*100, test_dice.avg*100)
        logger.write(message)

        mean = np.mean([test_iou.avg, test_dice.avg])
        if mean > best_mean:
            best_mean = mean
            best_message = "best test performance until now: \n" + message + "\n"
            torch.save(localizer.state_dict(), config.localizer_state_path + "localizer_%s.pth"%times)
        logger.write(best_message)
  

                
            

if __name__ == "__main__":
    
    for i in range(5):
        train_test(i)
