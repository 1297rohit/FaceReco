import dlib, cv2,os
import numpy as np
from imutils.face_utils import FaceAligner
from . import supportFiles
from tqdm import tqdm
#from supportFiles import pose_predictor_model_location
#from supportFiles import face_recognition_model_location
#from supportFiles import configFile_location
#from supportFiles import modelFile_location

class FaceReco():
    number_of_object_count=0
    ## Constructor for the package
    def __init__(self):
        self.faces = list()
        self.labels = list()
        FaceReco.number_of_object_count = FaceReco.number_of_object_count+1
        self.serial_number = FaceReco.number_of_object_count
        posepre=supportFiles.pose_predictor_model_location()
        self.pose_predictor = dlib.shape_predictor(posepre)
        detect = supportFiles.face_recognition_model_location()
        self.face_encoder=dlib.face_recognition_model_v1(detect)
        mdf = supportFiles.modelFile_location()
        self.modelFile = mdf
        conf = supportFiles.configFile_location()
        self.configFile = conf
        
        self.detector = dlib.get_frontal_face_detector()
        #self.face_encoder = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')
        #self.pose_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
        self.fa = FaceAligner(self.pose_predictor)
        #self.modelFile = "./opencv_face_detector_uint8.pb"
        #self.configFile = "./opencv_face_detector.pbtxt"
        self.net = cv2.dnn.readNetFromTensorflow(self.modelFile, self.configFile)
        self.images = list()
    
    
    ## Method to detect face from image
    def faceDetection(self,im):
    
        img = cv2.imread(im)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frameHeight = img.shape[0]
        frameWidth = img.shape[1]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceAligned = self.fa.align(img, gray,dlib.rectangle(x1,y1,x2,y2))
                return faceAligned

            
    ## Method to recognize face when model is being trained
    def faceRecognitionTrainig(self, faceAligned):
        landmark = self.pose_predictor(faceAligned,dlib.rectangle(0,0,faceAligned.shape[0],faceAligned.shape[1]))
        face_descriptor = self.face_encoder.compute_face_descriptor(faceAligned, landmark, num_jitters=2)
        self.faces.append(face_descriptor)      
        return self.faces

    
    ## Method to recognize face when model is being tested
    def faceRecognitionTesting(self,faceAligned):
        landmark = self.pose_predictor(faceAligned,dlib.rectangle(0,0,faceAligned.shape[0],faceAligned.shape[1]))
        face_descriptor = self.face_encoder.compute_face_descriptor(faceAligned, landmark, num_jitters=2)
        score = np.linalg.norm(self.faces - np.array(face_descriptor), axis=1)
        imatches = np.argsort(score)
        score = score[imatches]
        return (self.labels[imatches][:10].tolist())
    
    
    ## Method called by user for trainig of model
    ## Parameter accepted by user :  path of the folder which contains images for training
    def train_model(self,trainpath):
        for im in tqdm(os.listdir(trainpath)):
            faceAligned = self.faceDetection(os.path.join(trainpath,im))
            self.faces = self.faceRecognitionTrainig(faceAligned)
            self.labels.append(im)
        self.faces = np.array(self.faces)
        self.labels = np.array(self.labels)
        folderpath = os.path.join(os.getcwd(),"Model_Object_"+str(self.serial_number))
        if(not(os.path.isdir(folderpath))):
            os.mkdir(folderpath)
        np.save(os.path.join(folderpath,'face_repr.npy'), self.faces)
        np.save(os.path.join(folderpath,'labels.npy'), self.labels)
        print("Model files has been stored at " + folderpath)
        
    ## Method called by user for loading the existing model which has been created by this package and stored at specified location
    ## Parameter : path of the folder where model files are stored
    ## return : return True if the model files exist at the specified location else False
    def load_model(self,folderpath):
        if(os.path.exists(folderpath)):
            temp_face = os.path.join(folderpath,'face_repr.npy')
            temp_labels = os.path.join(folderpath,'labels.npy')
            if(os.path.exists(temp_face) and os.path.exists(temp_labels)):
                self.faces = np.load(temp_face)
                self.labels = np.load(temp_labels)
                return True
            else:
                print("Model files cannot be found..\n")
                print("Please provide the path of the folder where the model file(s) (face_repr.npy and labels.npy) are stored!! ")
                return False
        else:
            print("Model files cannot be found..\n")
            print("Please provide a valid path!!")
            return False
    
    
    ## Method called by user for testing of the model
    ## Parameter : path of the given for testing
    ## return : list of the images
    def test_model(self,im):
        faceAligned = self.faceDetection(im)
        self.images = self.faceRecognitionTesting(faceAligned)
        return self.images
        
    
    
