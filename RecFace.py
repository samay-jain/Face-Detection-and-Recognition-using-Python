import cv2
import face_recognition
import pickle


data = pickle.loads(open("encodings.pickle", "rb").read())
vid_cam = cv2.VideoCapture(0)

while True:
    _, image_frame = vid_cam.read()
    rgb_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_names = []

    for face_location in face_locations:
        face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)

        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)

        face_names.append(name)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(image_frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image_frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for (top, right, bottom, left) in face_locations:
        if "Unknown" in face_names:
            cv2.rectangle(image_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(image_frame, "Unknown", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Face Recognition', image_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid_cam.release()
cv2.destroyAllWindows()