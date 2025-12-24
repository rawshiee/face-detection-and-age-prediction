import cv2







video=cv2.VideoCapture(0)

while True:
    ret,frame=video.read(0)
    cv2.imshow('hi',frame)
    k=cv2.waitKey(1)
    if k==ord('m'):
        break
    
    video.release()
    cv2.destroyAllWindows()
