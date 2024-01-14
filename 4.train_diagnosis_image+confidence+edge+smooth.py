import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from config import Config
from utils import Logger, AverageMeter, evaluate
from dataset.dataset import BUSIDataset_image_addition3
from networks.networks import VGG16_fusion


class Config(object):
    def __init__(self) -> None:
        self.log_path = "./logs/"
        self.gpu_id = "1"
        self.image_path = "../../datasets/UDIAT/image/"
        self.mask_path = "../../datasets/UDIAT/mask/"
        self.train_mapping_path = "../../datasets/UDIAT/train_mapping.txt"
        self.test_mapping_path = "../../datasets/UDIAT/test_mapping.txt"
        self.cut_image_path = "./cut_results/cut_image/"
        self.confidecen_path = "./cut_results/cut_pre_mask/"
        self.cut_mask_path = "./cut_results/cut_mask/"
        self.edge_path = "./artificial_prior/edge/"
        self.smooth_path = "./artificial_prior/dilated/"


        self.model_state_path = "./model_state/"
        self.localizer_state_path = "./model_state/"
        
        self.input_channel=4
        self.class_num = 2
        self.network_input_size = (224, 224)
        self.batch_size = 32
        self.num_workers = 32
        self.localizer_learning_rate = 0.001
        self.LOCALIZER_EPOCH = 200
        self.learning_rate = 0.001
        self.EPOCH = 120



def train_test(times):
    config = Config()
    logger = Logger(config.log_path + "diagnoser_image+confidence+edge+smooth_%s.log"%times)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    train_set = BUSIDataset_image_addition3(config.cut_image_path, config.smooth_path, config.edge_path, config.confidecen_path, config.train_mapping_path, 
                        transform=transforms.Compose([
                        transforms.Resize(config.network_input_size),
                        transforms.ToTensor()]))
    test_set = BUSIDataset_image_addition3(config.cut_image_path, config.smooth_path, config.edge_path, config.confidecen_path, config.train_mapping_path, 
                        transform=transforms.Compose([
                        transforms.Resize(config.network_input_size),
                        transforms.ToTensor()]))
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
    
    model = VGG16_fusion(class_num=config.class_num, input_channel=config.input_channel, pretrained=True).cuda()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.learning_rate, weight_decay=5e-4, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()


    # training
    model.train()
    best_mean = 0
    best_log = ""
    for epoch in range(config.EPOCH):
        train_loss = AverageMeter()
        for step, (img, label) in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()
            output = model(img)
            loss = loss_function(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item(), img.size(0))

        message = ""
        message += "epoch: %2d  \n" % epoch
        message += "    train:  loss: %.5f \n" % train_loss.avg
        # logger.write(message)

        # test after each epoch training
        with torch.no_grad():
            model.eval()
            test_loss = AverageMeter()
            labels = []
            predictions = []
            scores = []
            for step, (img, label) in enumerate(test_loader):
                img = img.cuda()
                label = label.cuda()
                output = model(img)
                loss = loss_function(output, label)

                test_loss.update(loss.item(), img.size(0))
                labels += label.cpu().tolist()
                predictions += torch.argmax(output, dim=1).cpu().tolist()
                scores += torch.softmax(output, dim=1).cpu().tolist()
             

            auc, accuracy, precision, specificity, sensitivity, f1, mean = evaluate(predictions, labels, scores)
            message += "    test:  loss: %.5f\n" % test_loss.avg
            message += "        auc \t accuracy \t precision \t specificity \t sensitivity \t F1 \n"
            message += "        %.2f%% \t %.2f%% \t %.2f%% \t %.2f%% \t %.2f%% \t %.2f%% \n" % (auc*100, accuracy*100, precision*100, specificity*100, sensitivity*100, f1*100)
            logger.write(message)

            if mean > best_mean:
                best_mean = mean
                best_log = "best test performance until now: \n" + message + "\n"
                # torch.save(model.state_dict(), config.model_state_path + "model.pkl")
            logger.write(best_log)
                
            

if __name__ == "__main__":
    
    for i in range(5):
        train_test(i)
