import cv2

def facebox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False, crop=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    bboxs = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
    return frame, bboxs


# ---- Model Paths ----
faceproto = "models/deploy.prototxt"
facemodel = "models/res10_300x300_ssd_iter_140000.caffemodel"
age_proto = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_proto = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"

# ---- Load Models ----
faceNet = cv2.dnn.readNet(facemodel, faceproto)
ageNet = cv2.dnn.readNet(age_model, age_proto)
genderNet = cv2.dnn.readNet(gender_model, gender_proto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0 - 10)', '(11 - 20)', '(21 - 30)', '(31 - 40)',
           '(41 - 50)', '(51 - 60)', '(61 - 70)', '(71 - 80)']
genderList = ['Male', 'Female']

# ---- Start Video ----
cap = cv2.VideoCapture(1)
padding = 20

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame, bboxs = facebox(faceNet, frame)

        for bbox in bboxs:
            # Crop face safely
            face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                         max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            if face.size == 0:
                continue  # skip invalid crops

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         MODEL_MEAN_VALUES, swapRB=False)

            # Predict gender
            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]

            # Predict age
            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]

            # Display result
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (bbox[0], bbox[1]-30),
                          (bbox[2], bbox[1]), (0, 255, 0), -1)
            cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), (0, 255, 0), 1)

        cv2.imshow('Age-Gender Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
