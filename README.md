# Object_detection-mask_detection

The data is downloaded from kaggle. Click [here](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection).

To detect masks from images, use detect.py. In it makes changes according to following :

a) On line 130, provide the path to model and labelmap.pbtxt. Both of them are already provided in the repo.

b) On line 133, provide the path to input directory it will automatically take images with extension with .png only.

- It will only use the images with .png extension. If you want to use other extensions(jpg, fpeg), make changes on line 102.
