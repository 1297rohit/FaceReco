from pkg_resources import resource_filename

def pose_predictor_model_location():
    
    return resource_filename(__name__, "./shape_predictor_68_face_landmarks.dat")

def modelFile_location():
    return resource_filename(__name__, "./opencv_face_detector_uint8.pb")

def face_recognition_model_location():
    return resource_filename(__name__, "./dlib_face_recognition_resnet_model_v1.dat")

def configFile_location():
    return resource_filename(__name__, "./opencv_face_detector.pbtxt")
