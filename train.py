import os
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from my_dataset import MyDataSet
from utils import read_split_data, plot_data_loader_image

#from model import resnet34,resnet50,resnet101
import torchvision.models as models

#root = "..\\data"  # 数据集所在根目录
root = "..\\BUSI"  # 数据集所在根目录
def main(i,train_images_path, train_images_label, val_images_path, val_images_label):
    i=i

    #device= "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

    data_transform = {
        "train": transforms.Compose([
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.RandomAffine(degrees=(0, 180), scale=(0.9, 1.1)),
                                     transforms.Resize((256, 256)),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    batch_size = 5

    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 4
    print('Using {} dataloader workers'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               )

    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])
    val_num = len(val_data_set)
    validate_loader = torch.utils.data.DataLoader(val_data_set,
                                                  batch_size=2,
                                                  shuffle=False,
                                                  num_workers=nw)


    #choose model
    """
    # use densenet121
    net = models.densenet121(pretrained=True)
    in_channel = net.classifier.in_features
    net.classifier = nn.Linear(in_channel, 2)
    
    
    # use resnet50
    net = models.resnet50(pretrained=True)
    # print(net)
    # net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    
    # use vit
    import timm
    net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
    # net =timm.create_model('efficientnet_b0',pretrained=True,num_classes=2)

    # use swin transformer
    import timm
    net = timm.create_model('swin_base_patch4_window7_224', pretrained=True,num_classes=2,drop_path_rate = 0.2)
    
    
    # use my model(MCRUNet)
    from new_model_transfer import tranfer
    net = tranfer(in_ch=3, out_ch=2)
    
    
    # use mobilenet_v3_small
    net = models.mobilenet_v3_small(pretrained=True)
    in_channel = net.classifier._modules['3'].in_features
    net.classifier._modules['3'] = nn.Linear(in_channel, 2)
    
    
    # use efficientnet
    import timm
    net = timm.create_model('efficientnet_b3', pretrained=True, num_classes=2)
    """
    # use my model(MCRUNet)
    from my_model import MCRUNet
    net = MCRUNet(in_ch=3, out_ch=2)

    net.to(device)
    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    lr = 0.0001
    optimizer = optim.Adam(params, lr=lr)

    epochs = 80
    best_acc = 0
    beat_epoch = 0


    #Use a learning rate strategy
    #from torch.optim.lr_scheduler import CosineAnnealingLR,LambdaLR
    #scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    #scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1), verbose=True)


    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train

        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        #scheduler.step()

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        y_true = []
        y_predict = []
        y_score = []
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                y_score_a_size = torch.softmax(outputs, dim=1)
                for l in range(len(y_score_a_size)):
                    y_score.append(y_score_a_size[l][1].cpu().detach().numpy().tolist())
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                y_true.append(val_labels.cpu().detach().numpy().tolist())
                y_predict.append(predict_y.cpu().detach().numpy().tolist())
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        y_true = np.array(y_true).reshape(-1)
        y_predict = np.array(y_predict).reshape(-1)
        val_accurate = acc / val_num
        print("第%d个epoch的学习率：%f" % (epoch+1, optimizer.param_groups[0]['lr']))
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        #print('accuracy: {}'.format(accuracy_score(y_true, y_predict)))

        print("accuracy_score:", accuracy_score(y_true, y_predict))
        print("precision_score:", metrics.precision_score(y_true, y_predict))
        print("recall_score:", metrics.recall_score(y_true, y_predict))
        print("f1_score:", metrics.f1_score(y_true, y_predict))
        print("roc_auc_score:", metrics.roc_auc_score(y_true,y_score))

        b = metrics.roc_auc_score(y_true,y_score)
        save_path = './result/MCRUNet_p' + str(i) +'_'+str(round(b, 4))+'_'+str(epoch+1)+ '.pth'

        if epoch >=40:
            if b >= best_acc:
                beat_epoch =epoch+1
                best_acc = b
                torch.save(net.state_dict(), save_path)

    print('Finished Training,best epoch :',beat_epoch)



if __name__ == '__main__':

    #训练模型
    train_images_label = np.load('.\\list\\train_images_label.npy')
    train_images_label = train_images_label.tolist()

    train_images_path = np.load('.\\list\\train_images_path.npy')
    train_images_path = train_images_path.tolist()

    val_images_path = np.load('.\\list\\val_images_path.npy')
    val_images_path = val_images_path.tolist()

    val_images_label = np.load('.\\list\\val_images_label.npy')
    val_images_label = val_images_label.tolist()

    for i in range(0,5):
        import time

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main(i,train_images_path, train_images_label, val_images_path, val_images_label)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    
