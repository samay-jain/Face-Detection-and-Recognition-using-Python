#Face Detection and Recognition using Python

![github upload](https://github.com/samay-jain/Face-Detection-and-Recognition-using-Python/assets/116068471/61235bbd-b9be-478c-8b29-ca8a12208dcd)

I have created the facial recognition model using the Python Pickle Module, which is open-source. Firstly, I have generated a dataset of my face using the Python file "CapImg.py", which captures 300 images automatically in a single execution. It automatically detects the faces present in the frame captured by the webcam using the file "haarcascade_frontalface_default.xml" present in this repository. It also asks for your name before capturing the images, and all these images are saved in a file named "dataset", which contains another folder name that you entered. This generated dataset will be later used for training the model.

In the next step, you need to download the "TrainModel.py" Python file that, on execution, generates the encodings of all the faces present in the dataset. This step is the most important in the whole project. The encodings of training are stored in the "encodings.pickle" file. You can download the "TrainModel.py" file using the link given below.

TrainModel.py file link: https://drive.google.com/file/d/1SQ9Hby13srU5gk3pgOT6q1hUjgXtDXzs/view?usp=drive_link

Later in the next step, you will just need to execute the model on a real-time webcam using the Python file "RecFace.py". It detects faces by using "haarcascade_frontalface_default.xml" and recognizes the faces present in the frame captured by the webcam by using the encodings in the "encodings.pickle" file.

All the information regarding how to execute or run the project is given in the txt file "How to run.txt" you can go through it for more information.
And the most important thing, you must have to install all the required libraries and modules before execution.
You can watch the final output of the project using the below link.

link to the output: https://drive.google.com/file/d/1_d9nlEwhH4SAlc9zw5yWAMGLcnuqvfp4/view?usp=sharing
