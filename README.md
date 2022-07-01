

<table class="tg">
<thead>
  <tr>
    <th class="tg-yagv">Crop Size<br></th>
    <th class="tg-yagv">Dataset</th>
    <th class="tg-yagv">Pretrained</th>
    <th class="tg-yagv">Train type</th>
    <th class="tg-yagv">mIoU</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh">512,1024</td>
    <td class="tg-baqh">Cityscapes</td>
    <td class="tg-baqh">No</td>
    <td class="tg-baqh">trainval</td>
    <td class="tg-baqh">75.1</td>
  </tr>
  <tr>
    <td class="tg-baqh">1024,1024</td>
    <td class="tg-baqh">Cityscapes</td>
    <td class="tg-baqh">No</td>
    <td class="tg-baqh">trainval</td>
    <td class="tg-baqh">76.1</td>
  </tr>
  <tr>
    <td class="tg-baqh">360,480</td>
    <td class="tg-baqh">CamVid</td>
    <td class="tg-baqh">No</td>
    <td class="tg-baqh">trainval</td>
    <td class="tg-baqh">69.1</td>
  </tr>
  <tr>
    <td class="tg-baqh">720,960</td>
    <td class="tg-baqh">CamVid</td>
    <td class="tg-baqh">No</td>
    <td class="tg-baqh">trainval</td>
    <td class="tg-baqh">72.0</td>
  </tr>
</tbody>
</table>

# How to run

## 1. Training
### 1.1 Cityscapes
> python train.py 

### 1.2 CamVid
> python python train.py --dataset camvid --train_type trainval --max_epochs 1000 --lr 1e-3 --batch_size 8

## 2. Testing
### 2.1 Cityscapes  
> python predict.py --dataset ${camvid, cityscapes} --checkpoint ${CHECKPOINT_FILE}

convert the training lable to class lable.
> python trainID2labelID.py
> Package the file into xxx.zip 
> Submit the zip file to https://www.cityscapes-dataset.com/
> You can get the results from the https://www.cityscapes-dataset.com/.
### 2.2 CamVid
> python test.py --dataset camvid --checkpoint ${CHECKPOINT_FILE}

## 4. fps
> python eval_forward_time.py --size 512,1024

 
 To be continue...
 
 ## Citation
 
 ## Reference
 
 https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks
 
 https://github.com/Reagan1311/DABNet
 

