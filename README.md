# SEMobNet-Object-Detection

## To start training the model from scratch please follow below steps:

1. Please download the PASCAL VOC 2007 and VOC 2012 dataset from the PASCAL VOC website.
2. Place the downloaded dataset in the directory '\ssd_mobv2\data\VOCdevkit'
3. The dataset directory should look like this:
    ssd_mobv2\data\VOCdevkit\VOC2007
    ssd_mobv2\data\VOCdevkit\VOC2012
4. After doing this, please update the HOME directory in the config.py file at 'ssd_mobv2\data\config.py'
5. HOME variable should look something like:
    HOME = os.path.expanduser("C:/Users/Xpiee/ssd_mobv2/")
6. For training the MobileNetv2 with SSD300 (without SE block), please open and execute 'ssd_mobv2\MobileNet_SSD300_train.ipynb'
7. For training the SE MobileNetv2 with SSD300 (with SE block), please open and execute 'ssd_mobv2\SE_MobileNet_SSD300_train.ipynb'
8. For evaluating the trained network, open the 'ssd_mobv2\SE_Mob_SSD300_eval.ipynb' and update the 'args_trained_model' with the correct path to saved trained model.
