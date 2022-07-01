
<table class="tg">
<thead>
  <tr>
    <th class="tg-amwm">Crop Size*</th>
    <th class="tg-amwm">Dataset</th>
    <th class="tg-amwm">Pretrained</th>
    <th class="tg-amwm">Train type</th>
    <th class="tg-amwm">mIoU</th>
    <th class="tg-amwm">Params</th>
    <th class="tg-amwm">Speed</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh">512,1024</td>
    <td class="tg-baqh">Cityscapes</td>
    <td class="tg-baqh">No</td>
    <td class="tg-baqh">trainval</td>
    <td class="tg-baqh">75.1</td>
    <td class="tg-baqh">1.35</td>
    <td class="tg-baqh">118.9</td>
  </tr>
  <tr>
    <td class="tg-c3ow">1024,1024</td>
    <td class="tg-c3ow">Cityscapes</td>
    <td class="tg-c3ow">No</td>
    <td class="tg-c3ow">trainval</td>
    <td class="tg-c3ow">76.1</td>
    <td class="tg-baqh">1.35</td>
    <td class="tg-baqh">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">360,480</td>
    <td class="tg-c3ow">CamVid</td>
    <td class="tg-c3ow">No</td>
    <td class="tg-c3ow">trainval</td>
    <td class="tg-c3ow">69.1</td>
    <td class="tg-baqh">1.35</td>
    <td class="tg-baqh">116.4</td>
  </tr>
  <tr>
    <td class="tg-c3ow">720,960</td>
    <td class="tg-c3ow">CamVid</td>
    <td class="tg-c3ow">No</td>
    <td class="tg-c3ow">trainval</td>
    <td class="tg-c3ow">72.0</td>
    <td class="tg-baqh">1.35</td>
    <td class="tg-baqh">120.8</td>
  </tr>
</tbody>
</table>

* Represents the resolution of the input image cropping in the training phase.

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
 

