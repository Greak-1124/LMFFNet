
# How to run

## 1. Training
### 1.1 Cityscapes
python train.py 

### 1.2 CamVid
python python train.py --dataset camvid --train_type trainval --max_epochs 1000 --lr 1e-3 --batch_size 8

## 2. Testing
python test.py --dataset ${camvid, cityscapes} --checkpoint ${CHECKPOINT_FILE}


## 3. Predicting & submit
p

## 4. fps
python eval_forward_time.py --size 512,1024

 
 To be continue...
 
 ## Citation
 
 ## Reference
 
 https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks
 
 https://github.com/Reagan1311/DABNet
 

