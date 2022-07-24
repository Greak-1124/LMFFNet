
# Segmentation performance of LMFFNet
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

\* Represents the resolution of the input image cropping in the training phase. We found that when the randomly cropped image is 1024x1024 in the training phase, the network can perform better. If the input image of LMFFNet is randomly cropped to 1024x1024 resolution, 76.1% mIoU can be achieved on cityscapes.


# Preparation
You need to download the Cityscapes and CamVid datasets and place the symbolic links or datasets of the Cityscapes and CamVid datasets in the dataset directory. Our file directory is consistent with DABNet (https://github.com/Reagan1311/DABNet).
```
dataset
  ├── camvid
  |    ├── train
  |    ├── test
  |    ├── val 
  |    ├── trainannot
  |    ├── testannot
  |    ├── valannot
  |    ├── camvid_trainval_list.txt
  |    ├── camvid_train_list.txt
  |    ├── camvid_test_list.txt
  |    └── camvid_val_list.txt
  ├── cityscapes
  |    ├── gtCoarse
  |    ├── gtFine
  |    ├── leftImg8bit
  |    ├── cityscapes_trainval_list.txt
  |    ├── cityscapes_train_list.txt
  |    ├── cityscapes_test_list.txt
  |    └── cityscapes_val_list.txt           
```        
# How to run

## 1 Training
### 1.1 Cityscapes
> python train.py 

### 1.2 CamVid
> python python train.py --dataset camvid --train_type trainval --max_epochs 1000 --lr 1e-3 --batch_size 8

## 2 Testing
### 2.1 Cityscapes  
> python predict.py --dataset ${camvid, cityscapes} --checkpoint ${CHECKPOINT_FILE}

To convert the training lables to class lables.
> python trainID2labelID.py
> Package the file into xxx.zip 
> Submit the zip file to https://www.cityscapes-dataset.com/submit/.
> You can get the results from the https://www.cityscapes-dataset.com/submit/.
### 2.2 CamVid
> python test.py --dataset camvid --checkpoint ${CHECKPOINT_FILE}

## 4. fps
> python eval_forward_time.py --size 512,1024

 
 To be continue...
 
 ## Citation
@article{shi2022lmffnet,
  title={LMFFNet: A Well-Balanced Lightweight Network for Fast and Accurate Semantic Segmentation},
  author={Shi, Min and Shen, Jialin and Yi, Qingming and Weng, Jian and Huang, Zunkai and Luo, Aiwen and Zhou, Yicong},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}

 ## Reference
 
 https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks
 
 https://github.com/Reagan1311/DABNet
 

