import numpy as np
import imutils
import cv2

# Define a function responsible for calculating the probabilites and outputting them
def age_gender_visualize(image, age_model, gender_model, wid=120):
    
    # Expand the dimensions of the image as the model accepts 4 dim input (the 4th for batch size) and convert to float32 as keras models donot accept uint* type
    image = np.float32(np.expand_dims(image, axis=0))
    
    # Obtain the age and gender predictions along with the maximum index of probability
    age_predictions = age_model.predict(image)
    age_predictions = age_predictions.flatten()
    age_max_index = age_predictions.argmax()

    
    # Define three lists: The 1st one is as per the label binarizer and other two are used to give age representation in output in ascending order
    age_values = ['0-2', '15-20', '25-32', '38-43', '48-53', '4-6', '60-inf', '8-13'] 
    new_age_values =  ['0-2', '4-6', '8-13', '15-20', '25-32', '38-43', '48-53', '60+'] 
    index = [0, 5, 7, 1, 2, 3, 4, 6] 
    age_predictions = age_predictions[index]
    
    # Store the predicted age 
    age_label = age_values[age_max_index]

    # Similar to age but there is no need to rearrange 
    gender_predictions = gender_model.predict(image)
    gender_predictions = gender_predictions.flatten()
    gender_max_ind = gender_predictions.argmax()

    gender_values = ['Male', 'Female']
    # Store the predicted age 
    gender_label = gender_values[gender_max_ind]

    # Combine both the  required labels 
    total_label = age_label + ' ' + gender_label

    # Convert the image into uint8 type because opencv only reads that but keras models require float datatype
    image = np.uint8(image[0,:,:,:])
    # resize the image 
    frame = image
    frame = imutils.resize(image, width=wid)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize two matrices used to represent our probabilities as well as our haarcascade face detector
    age_canvas = np.zeros((290, 300, 3), dtype="uint8")
    gender_canvas = np.zeros((90, 300, 3), dtype="uint8")
    
    detector = cv2.CascadeClassifier("/home/hrushikesh/case_studies/cascades/haarcascade_frontalface_default.xml")

    # Loop over the age labels + probabilities and draw them 
    for (i, (age, age_prob)) in enumerate(zip(new_age_values, age_predictions)):
        text = "{}: {}%".format(age, np.round(age_prob * 100,2))
        w = int(age_prob * 300)
        cv2.rectangle(age_canvas, (5, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(age_canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        
    # Similarly for gender 
    for (i, (gender, gender_prob)) in enumerate(zip(gender_values, gender_predictions)):
        text = "{}: {}%".format(gender, np.round(gender_prob * 100,2))
        w = int(gender_prob * 300)
        cv2.rectangle(gender_canvas, (5, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(gender_canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    together = np.vstack([gender_canvas, age_canvas])

    #''' 
    
    # draw the label on the the original image
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # Proceed only if faces are detected ( I noticed only 90% of images have been identified by the classifier )
    if len(rects) > 0:
        for i in range(len(rects)):
            (fX, fY, fW, fH) = rects[0]
            cv2.putText(frame, total_label, (fX, fY + fH - 5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
    #'''
    cv2.imshow("Image", frame)
    cv2.imshow("Gender and Age", together)
    cv2.waitKey(0)
