"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from tqdm.autonotebook import tqdm
import torch
from pycocotools.cocoeval import COCOeval
from apex import amp
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def train(model, train_loader, epoch, criterion, optimizer, scheduler, is_amp):
    model.train()
    num_iter_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader)
    for i, (img, _, _, gloc, glabel, *other) in enumerate(progress_bar):
        if torch.cuda.is_available():
            img = img.cuda()
            gloc = gloc.cuda()
            glabel = glabel.cuda()

        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()
        loss = criterion(ploc, plabel, gloc, glabel)

        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))


        if is_amp:
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()

def evaluate(model, test_loader, epoch, writer, encoder, nms_threshold):
    model.eval()
    detections = []
    category_ids = test_loader.dataset.coco.getCatIds()   
    for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):

        print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            # Get predictions
            ploc, plabel = model(img)
            ploc, plabel = ploc.float(), plabel.float()
            for idx in range(ploc.shape[0]):
                
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue

                height, width = img_size[idx]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    detections.append([img_id[idx], loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,
                                       (loc_[3] - loc_[1]) * height, prob_,
                                       category_ids[label_ - 1]])
                    
    detections = np.array(detections, dtype=np.float32)
    coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)

def cognata_eval(model, test_loader, epoch, writer, encoder, nms_threshold):
    model.eval()
    preds = []
    targets = []
    
    for nbatch, (img, img_id, img_size, _, _, gt_boxes) in enumerate(test_loader):

        print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            # Get predictions
            ploc, plabel = model(img)
            ploc, plabel = ploc.float(), plabel.float()
            for idx in range(ploc.shape[0]):
                dts = []
                labels = []
                scores = []
                
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 500)[0]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue

                height, width = img_size[idx]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    dts.append([loc_[0]* width, loc_[1]* height, loc_[2]* width, loc_[3]* height,])
                    labels.append(label_)
                    scores.append(prob_)
                
                dts = torch.tensor(dts, device='cuda')
                labels = torch.tensor(labels, device='cuda', dtype=torch.int32)
                scores = torch.tensor(scores, device='cuda')
                preds.append({'boxes': dts, 'labels': labels, 'scores': scores})
                targets.append({'boxes': gt_boxes[idx][:,:4].to(device='cuda'), 'labels': gt_boxes[idx][:, 4].to(device='cuda', dtype=torch.int32) })
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, backend='faster_coco_eval')
    metric.update(preds, targets)
    return metric.compute()
    '''
    print('start ap')
    all_preds = [None]*torch.distributed.get_world_size()
    all_targets = [None]*torch.distributed.get_world_size()
    print('first gather')
    torch.distributed.all_gather_object(all_preds, preds)
    print('second gather')
    torch.distributed.all_gather_object(all_targets, targets)
    if torch.distributed.get_rank() == 0:
        print('calculating')
        final_preds = []
        final_targets = []
        list(map(final_preds.extend, all_preds))
        list(map(final_targets.extend, all_targets))
        metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        metric.update(final_preds, final_targets)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(metric.compute())
    print('end')
    #torch.distributed.barrier()
    '''
