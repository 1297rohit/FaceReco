# FaceReco
Face Recognition Library using Class based structure

[![PyPI](https://badge.fury.io/py/FaceReco.png)](https://pypi.org/project/FaceReco/)

## Installing
pip install FaceReco

## Description
Face Recognition is a very popular topic. It has lot of use cases in the filed of biometric security. 
Now a days with the help of Deep learning face recognition has become very feasible to people. 
As deep learning is a very data intensive task and we may always not have such huge amount of data to work in case of face recognition 
so with the advancement in One Shot Learning, face recognition has become more practical and feasible. This Python Package make it even more feasible, simple 
and easy to use. We have eliminated all the steps to download the supporting files and setting up the supporting files. You can simply installed the python package and start doing face detection and recognition.

## Steps Explanation
To learn more about the tasks which are being performed on the backend head over to link : [Step by Step Face Recognition Code Implementation From Scratch In Python](https://towardsdatascience.com/step-by-step-face-recognition-code-implementation-from-scratch-in-python-cc95fa041120)

## Using The Package

#### Train Model

```python
import FaceReco.FaceReco as fr
fr_object1 =  fr.FaceReco()
fr_object1.train_model("lfw_selected/face")
``` 

#### Test Model

```python
fr_object1.test_model("lfw_selected/face2/Johnny_Depp_0002.jpg")
``` 

#### Load Saved Model

```python
fr_object2 =  fr.FaceReco()
fr_object2.load_model("Model_Object_1") #folder of saved model
fr_object2.test_model("lfw_selected/face2/Johnny_Depp_0002.jpg")
``` 
