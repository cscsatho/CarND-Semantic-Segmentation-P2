# **Semantic Segmentation Project**

## Project Writeup

[//]: # (Image References)
[image0]: ./img/loss.png
[image1]: ./runs/1526824770.2972243-final/um_000005.png
[image2]: ./runs/1526824770.2972243-final/um_000007.png
[image3]: ./runs/1526824770.2972243-final/um_000013.png
[image4]: ./runs/1526824770.2972243-final/umm_000057.png

---

**Labelling the pixels of a road in images using a Fully Convolutional Network (FCN)**

### Rubric points

#### Build the Neural Network

- _Does the project load the pretrained vgg model?_ - `load_vgg()` was implemented by loading model and then retrieving `input`, `keep`, and `layer3-4-7` tensors.
- _Does the project learn the correct features from the images?_ - `layers()` was implemented by applying deconvolution (transpose) on 1x1 convoluted `layer7`, then adding skip connection with 1x1 convoluted `layer4`, then applying deconvolution on the result, adding skip connection with 1x1 convoluted `layer3`, and finally applying deconvolution on the result.
- _Does the project optimize the neural network?_ - `optimize()` was implemented by first converting 4D logits (`nn_last_layer` and `correct_label`) to 2D. Then `softmax_cross_entropy` was fed to `reduce_mean` to get loss operation. This was then fed to the `adam_optimizer` along with the `learning_rate`.
- _Does the project train the neural network?_ - `train_nn()` was implemented by invoking `session.run()` on the given epochs/batches.

#### Neural Network Training

- _Does the project use reasonable hyperparameters?_
  - `NUM_CLASSES` = 2
  - `IMG_SHAPE` = (160, 576)
  - `BATCH_SZ` = 5
  - `EPOCHS` = 40
  - `LEARN_RATE` = 0.0005
  - `KEEP_PROB` = 0.75
- _Does the project train the model correctly?_ - It is getting smaller over time (blue: values, red: trendline):
![image0]
- Does the project correctly label the road? - Some sample images from the `runs/1526824770.2972243-final` folder:
![image1]
![image2]
![image3]
![image4]


