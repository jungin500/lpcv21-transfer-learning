import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys, os
import cv2
import numpy as np
import albumentations as A

sys.path.append(os.path.join(os.getcwd(), "targets", "mobilenetv3ssd"))
sys.path.append(os.path.join(os.getcwd(), 'baseline', 'solution'))
sys.path.append(os.path.join(os.getcwd(), 'baseline', 'solution', 'yolov5'))

from targets.mobilenetv3ssd.model import SSD300, MultiBoxLoss
from targets.mobilenetv3ssd.mb3utils import *

from baseline.solution.yolov5.models.experimental import attempt_load as load_yolov5_ensemble_model
from baseline.solution.yolov5.utils.general import non_max_suppression, scale_coords, xywh2xyxy
from baseline.solution import main as yolomain


# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = 2  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Learning parameters
checkpoint =None #  path to model checkpoint, None if none
batch_size = 32  # batch size 
# iterations = 120000  # number of iterations to train  120000
workers = 8  # number of workers for loading data in the DataLoader 4
print_freq = 10  # print training status every __ batches
lr =5e-4  # learning rate
#decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


import os
import random
from trutils.trutils import IdentityTransform, CvToTensor, SubfolderImageOnlyDataset


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        print("checkpoint none")
        start_epoch = 0
        model = SSD300(n_classes=n_classes)

        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        # differnet optimizer           
        # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
        #                             lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr':  lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)                            

        #optimizer = torch.optim.SGD(params=[{'params':model.parameters(), 'lr': 2 * lr}, {'params': model.parameters}],  lr=lr, momentum=momentum, weight_decay=weight_decay) 


    else:
        print("checkpoint load")
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']


    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Dataset and Dataloaders
    train_dataset = SubfolderImageOnlyDataset(
        input_folder='./data/',
        transform = A.Compose([
            A.RandomCrop(width=960, height=540),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2)
        ]),
        request_sizes = [(640, 384), (300, 300)]
        )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)

    dataset_image_output_size = (540, 960)  # img_h, img_w
    input_image_shape = dataset_image_output_size

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # now it is mobilenet v3,VGG paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations,
    epochs = 600
    # decay_lr_at =[154, 193]
    # print("decay_lr_at:",decay_lr_at)
    print("epochs:",epochs)

    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    print("learning rate.  The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))    
    # Epochs,I try to use different learning rate shcheduler
    #different scheduler six way you could try
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = (epochs // 7) + 1) 
    scheduler = ReduceLROnPlateau(optimizer,mode="min",factor=0.1,patience=15,verbose=True, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        # if epoch in decay_lr_at:
        #     adjust_learning_rate_epoch(optimizer,epoch)
        

        # One epoch's training
        train_transfer(
              dataloader=train_loader,
              student_model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              input_image_shape=input_image_shape)
        print("epoch loss:",train_loss)      
        scheduler.step(train_loss)      

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def normalize_wh(x, shape):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    y[:, 0] /= shape[0]
    y[:, 1] /= shape[1]
    y[:, 2] /= shape[0]
    y[:, 3] /= shape[1]

    return y


def train_transfer(dataloader, student_model, criterion, optimizer, epoch, input_image_shape):
    
    # Teacher model
    # load model
    teacher_model = load_yolov5_ensemble_model(
        r'C:\Workspace\study-projects\LPCV\lpcv21-transfer-learning\baseline\solution\yolov5\weights\best.pt'
        ).to(device).eval()

    # Get names and colors
    names = teacher_model.module.names if hasattr(teacher_model, 'module') else teacher_model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # teacher_model parameters
    # '''
    # Namespace(
    #     weights='C:\\Workspace\\study-projects\\LPCV\\lpcv21-transfer-learning\\baseline\\solution/yolov5/weights/best.pt',
    #     data='C:\\Workspace\\study-projects\\LPCV\\lpcv21-transfer-learning\\baseline\\solution/ballPerson.yaml',
    #     source='..\\..\\videos\\5p2b_01A1\\5p2b_01A1.m4v',
    #     output='./outputs', img_size=640, conf_thres=0.4, iou_thres=0.5, fourcc='mp4v', device='', view_img=False,
    #     save_txt=False, classes=[0, 1], agnostic_nms=False, augment=False,
    #     config_deepsort='C:\\Workspace\\study-projects\\LPCV\\lpcv21-transfer-learning\\baseline\\solution/deep_sort/configs/deep_sort.yaml',
    #     groundtruths='..\\..\\videos\\5p2b_01A1\\5p2b_01A1.csv', save_img=False, skip_frames=1)
    # '''
    # Model Input:  torch.Size([1, 3, 384, 640])

    # Student model
    student_model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    global train_loss
    # Batches
    # for i, (images, boxes, labels, _) in enumerate(train_loader):
    # get images, boxes, labels from teacher model!\
    for i, images in enumerate(dataloader):
        image_original = images[0].numpy()
        image_teacher = images[1]
        image_student = images[2]

        batch_size = image_original.shape[0]

        # Grab bbox from teacher model
        teacher_pred = teacher_model(image_teacher.to(device))[0]
        teacher_pred = non_max_suppression(teacher_pred, 0.4, 0.5, classes=[0, 1], agnostic=False)

        # Postprocess per images
        batch_xyminmax = []
        batch_confss = []
        batch_clses = []

        for det_id, det in enumerate(teacher_pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to image_raw_size size
                det[:, :4] = scale_coords(image_teacher.shape[2:], det[:, :4], image_original[det_id].shape).round()
                bbox_xyminmax = []
                confs = []
                clses = []

                # Write results
                for *xyxy, conf, cls in det:
                    
                    # img_h, img_w, _ = image_original[i].shape  # get image shape
                    img_h, img_w = input_image_shape
                    x_c, y_c, bbox_w, bbox_h = yolomain.bbox_rel(img_w, img_h, *xyxy)
                    obj_minmax = [x_c - bbox_w / 2, y_c - bbox_h, x_c + bbox_w / 2, y_c + bbox_h / 2]

                    bbox_xyminmax.append(obj_minmax)
                    confs.append(conf)
                    clses.append(cls)
                    
                bbox_xyminmaxs = torch.Tensor(normalize_wh(bbox_xyminmax, input_image_shape))
                confss = torch.Tensor(confs)
                clses = torch.Tensor(clses)

                batch_xyminmax.append(bbox_xyminmaxs)
                batch_confss.append(confss)
                batch_clses.append(clses.type(torch.int64))

                # # draw boxes for visualization
                # bbox_items = bbox_xyminmax.shape[0]  # applies to all
                # for id in range(bbox_items):
                #     xywh = [int(val) for val in bbox_xyminmax[id].numpy()]
                #     conf = confss[id].numpy().item()  # 1 item
                #     cls = int(clses[id].numpy().item())  # 1 item

                #     xyxy = xywh2xyxy(np.expand_dims(xywh, 0))[0]

                #     image_original[i] = cv2.rectangle(image_original[i], (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                #     image_original[i] = cv2.putText(image_original[i], "%d: %.1f%%" % (cls, conf * 100), (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))

        # End of Postprocess

        # Show results
        # for image in image_original:
        #     cv2.imshow("Result", image)
        #     if cv2.waitKey(0) == ord('q'):  # q to quit
        #         raise StopIteration

        # Use batch_xywh, batch_conf, batch_cls as a label to train student model
        ''' Use:
        batch_xyminmax = [None for _ in range(batch_size)]
        batch_confss = [None for _ in range(batch_size)]
        batch_clses = [None for _ in range(batch_size)]
        '''
        
        data_time.update(time.time() - start)

        # if(i%200==0):
        #     adjust_learning_rate_iter(optimizer,epoch)
        #     print("batch id:",i)#([8, 3, 300, 300])
        #N=8
        # Move to default device
        images = image_student.to(device)  # (batch_size (N), 3, 300, 300)
        
        # MobileNetV3-SSD wants boxes and labels to be like:
        # boxes -> [B, I, 4] -> 4: [xmin, ymin, xmax, ymax]
        # labels -> [B, I]
        boxes = [b.to(device) for b in batch_xyminmax]
        labels = [l.squeeze().to(device) for l in batch_clses]
        # print("labels -> ", len(labels), type(labels[0]), labels[0].shape, labels[0])

        # print("boxes -> ", len(boxes), type(boxes[0]), boxes[0].shape, boxes[0], boxes[0].dtype)

        # Forward prop.
        predicted_locs, predicted_scores = student_model(images)  # (N, anchor_boxes_size, 4), (N, anchor_boxes_size, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
        train_loss=loss
        #print("training",train_loss)

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}][{3}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(dataloader), optimizer.param_groups[1]['lr'],
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses))

    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


def adjust_learning_rate_epoch(optimizer,cur_epoch):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    print("DECAYING learning rate. The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

#warmup ,how much learning rate.
def adjust_learning_rate_iter(optimizer,cur_epoch):

    if(cur_epoch==0 or cur_epoch==1 ):
        for param_group in optimizer.param_groups:
            param_group['lr'] =param_group['lr'] +  0.0001  
            print("DECAYING learning rate iter.  The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

      


if __name__ == '__main__':
    main()
