# Face-Analysis-using-deep-learning
Using FER and Adience datasets, set of deep learning models which provide emotions, age and gender properties of all the faces in a image or video

Models trained on Adience dataset and FER2013 dataset utilized for gender, age and emotion recognition.

Trained my custom model on Adience dataset which achieved 20% greater rank-1 accuracy in the age category
compared to the corresponding publication of dataset.

Trained a model on FER dataset achieving 66.07% accuracy which is equivalent to 5 position in the leadership board
of the former competition.

# Models Used
For the adience dataset,a sequential model with no special kinds like the inception or the resnet addition was created.
Strides and padding were used to decrease size and only 1 max pooling was utilized. 
The reason for strides and padding was because mxnet ( which I initally used but discarded because of errors ) which wanted me to clearly specify all the values of strides and padding. Thus by calucaltions I found 59 * 59 would ensure model is correct
I wanted to avoid the use of any special architectures and also the fc ( as they increase memory and time to train ) and thus created this model.
I kept a fc layer though with low # of nodes 512.
I followed VGG type of increase in filterss ( doubling always ) majorly 3*3 receptive fields with 5*5 at the beginning.

For the FER dataset, again a custom VGGNet inspired model was used. It has similar structure to the VGG-Net(doubling of filters as dimension is reduced by half).

# Training Logs
Many experiments have been done. Here is one of the sample outputs of one of my experiments on the Adience dataset. Training was stopped when learning appeared to stagnate or whether overfitting was observed.

Final Result for the adience dataset in age category

![result](https://user-images.githubusercontent.com/56476887/94838572-6f2d1200-0433-11eb-847b-06f35ffc319b.png)

Graphs when training was stopped at different epochs (because lowering of learning rate was required)

![result_stop1](https://user-images.githubusercontent.com/56476887/94838578-705e3f00-0433-11eb-96a0-8bbc0339d2df.png)
![result_stop2](https://user-images.githubusercontent.com/56476887/94838582-705e3f00-0433-11eb-907b-934d82805f20.png)
![result_stop3](https://user-images.githubusercontent.com/56476887/94838584-70f6d580-0433-11eb-9f69-aed43e9fd65d.png)

FER result

![vggnet_emotion2](https://user-images.githubusercontent.com/56476887/94838587-718f6c00-0433-11eb-8da7-612f68647315.png)

# Results
![emotions](https://user-images.githubusercontent.com/56476887/94844068-49a40680-043b-11eb-8936-ab9d412072f5.gif)
![Screenshot from 2020-07-28 19-43-26](https://user-images.githubusercontent.com/56476887/94838203-dc8c7300-0432-11eb-9ec9-6d819b250cb9.png)
![Screenshot from 2020-07-28 20-02-31](https://user-images.githubusercontent.com/56476887/94838209-de563680-0432-11eb-9aa6-640e3d91780d.png)
![Screenshot from 2020-07-28 20-09-38](https://user-images.githubusercontent.com/56476887/94838211-deeecd00-0432-11eb-9029-0407af0025a9.png)


More information about the learning rates used, choices like optimizer etc can be found in the text files.
