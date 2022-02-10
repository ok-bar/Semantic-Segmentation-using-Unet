# Semantic-Segmentation-using-Unet

The task of this project is segmentation for autonomous-driving using the city map dataset.
Each image contains both the real image and the mask, so first a split is needed.
![37DDB504-A1CC-431A-AE27-C44923ACE7E8_1_105_c](https://user-images.githubusercontent.com/51881832/153382025-1e229a3f-e3a5-48a7-80db-6d37a584d055.jpeg)


The real images after the separation:
![C2D556BE-BCDB-46E2-A892-CD6253108F07_1_105_c](https://user-images.githubusercontent.com/51881832/153382369-996aefad-39e9-4f00-be48-e33b6fffba9b.jpeg)
The segmented images after the separation :
![A280BBD3-5DC1-4C5D-B835-1DFE8AF195AC_1_201_a](https://user-images.githubusercontent.com/51881832/153382622-bde7377d-3f7b-4d53-96fb-17c69da678ea.jpeg)


# Labeling using K-means

Although each pixel in these images is labeled, it was still necessary to separate the different colors to labels. K-means will be used to categorize these colored labels into ten different groups. This is a hyperparameter. The next few labeled images will be shown, followed by rescaling of the original image.

![6CC177AC-AB50-483F-B14B-27A97A499CF8_1_201_a](https://user-images.githubusercontent.com/51881832/153383127-c04c43b5-6697-41bd-842e-437a64337700.jpeg)

# Buliding the model

Here I'm using UNet since the position of the object is crucial for this task. The feature map is upsampled to the size of the original input image using a transposed convolution layer that preserves the spatial information. It also includes skip connections, which help to keep information that would otherwise be lost during encoding. This model is kind of mini-VGG since it use vgg as a backbone.

After 128 epochs no improvemnt was shown in the training section:
![128BA548-952B-4291-AB14-7357B01B6595_1_201_a](https://user-images.githubusercontent.com/51881832/153384155-b3a499a6-972e-4712-b1c9-32c732d2f4f1.jpeg)

The results are as shown:
![1898D9CA-1677-4B93-8651-B370B82FA9AB_1_201_a](https://user-images.githubusercontent.com/51881832/153384616-b9b85f35-6fe7-43ee-a06c-75ebaa44ebfb.jpeg)

# Conclusions
Despite the fact that the results are not amazing, they are not so bad. More layers, as well as augmentations and different class clustering, are required for even better results.
