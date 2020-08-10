# import the necessary packages
from os import path

OUTPUT_BASE = "/home/hrushikesh/dl4cv/gender_age/output"

# define the path to the dlib facial landmark predictor
DLIB_LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"

# define the path to the age network + supporting files
AGE_MODEL = "/home/hrushikesh/dl4cv/gender_age/checkpoints/age/epoch_55.hdf5"
AGE_LABEL_ENCODER = path.sep.join([OUTPUT_BASE, "age/age_le.cpickle"])

# define the path to the gender network + supporting files
GENDER_MODEL = "/home/hrushikesh/dl4cv/gender_age/checkpoints/gender/expt2/epoch_25.hdf5"
GENDER_LABEL_ENCODER = path.sep.join([OUTPUT_BASE, "gender/gender_le.cpickle"])

