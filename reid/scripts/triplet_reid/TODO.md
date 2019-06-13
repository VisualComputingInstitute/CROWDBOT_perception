TODO
================================================


Started on 02/11/18

Open
------------------------------------------------


* BBox prediction
* Instance regression
* Cityscapes dataloader

* implement automatic saving
* implement warm-up LR scheduler
* file bug report for affine_grid


Working on
------------------------------------------------

* [ ] implement forward script
* [ ] compare resnet fpn architecture to maskrcnn-benchmark
* [ ] Panoptic Head
* [ ] augment bbox (needed?)

Done
-------------------------------------------------

* [x] Implemented ResNet-FPN
* [x] Implemented semantic head
* [x] implemented COCO stuff writer
* [x] restoring checkpoints
* [x] fixed possible batchnorm issue
* [x] wrote notebook to crop and pad in parallel
* [x] investigating memory issue on cluster -> goes up after one epoch but stays constant after that
* [x] implement experiment specific result folder
* [x] testing stuff head
* [x] implement evaluation from checkpoint
* [x] implement time report
* [x] implement COCO Panoptic dataloader
* [x] added semantic eval for panoptic dataset
* [x] implement top-k loss (credits: killian)
* [x] implement gt bboxes
* [x] COCO Panoptic writer