import cv2
import os

face_id = input('Enter Person Name: ')
vid_cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('E:/Projects/Face Recognition Project/facial-recognition-main/haarcascade_frontalface_default.xml')

count = 0

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

assure_path_exists("E:/Projects/Face Recognition Project/facial-recognition-main/dataset/"+face_id+"/")

while (True):
    _, image_frame = vid_cam.read()
    faces = face_detector.detectMultiScale(image_frame, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite("E:/Projects/Face Recognition Project/facial-recognition-main/dataset/"+face_id+"/" + str(face_id) + '.' + str(count) + ".jpg", image_frame[y:y + h, x:x + w])
        cv2.putText(image_frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', image_frame)
    
    if cv2.waitKey(1) == 13:
        break
    elif count >= 300:
        print("User is successfully added to the dataset")
        break

vid_cam.release()
cv2.destroyAllWindows()