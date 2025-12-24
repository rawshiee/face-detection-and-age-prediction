import cv2



def facebox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame,1.0,(227,227),[104,117,123],swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bbox=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bbox.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
    return detection



faceproto="opencv_face_detector.pbtxt"
facemodel=".txt.pb"

faceNet=cv2.dnn.readNet(facemodel,faceproto)

cap = cv2.VideoCapture(0) # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open video stream or file.")
else:
    while True:
        ret, frame = cap.read() 
        detect=facebox(faceNet,frame)# Read a frame

        if not ret: # If frame not read correctly
            print("Error: Failed to grab frame.")
            break


        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()