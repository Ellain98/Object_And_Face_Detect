import cv2               
import numpy as np     
import imutils           

def nothing(x):
    pass

def rotated(img, angle, rotPoint=None):
    (height,width) = img.shape[:2]
    
    if rotPoint is None:
        rotPoint = (width//2,height//2)
        
    rotMat=cv2.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions= (width,height)
    
    return cv2.warpAffine(img, rotMat, dimensions)

Objects_Counter=0
cap = cv2.VideoCapture(0)     #VideoCapture
cap.set(3,640)                #3-Width
cap.set(4,480)                #4-Height
cv2.namedWindow("Trackbars") 


cv2.createTrackbar("L - H","Trackbars", 20, 255, nothing)
cv2.createTrackbar("L - S","Trackbars", 70, 255, nothing)
cv2.createTrackbar("L - V","Trackbars", 120, 255, nothing)
cv2.createTrackbar("U - H","Trackbars", 30, 179, nothing)
cv2.createTrackbar("U - S","Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V","Trackbars", 255, 255, nothing)
cv2.createTrackbar("AREA","Trackbars", 1, 20, nothing)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while True:
     _,frame= cap.read() #Frame reading
     frame = rotated(frame, 180)
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Convertion into hsv spectrum h=hue s=saturation v=value
     
     LHT = cv2.getTrackbarPos("L - H", "Trackbars") 
     LST = cv2.getTrackbarPos("L - S", "Trackbars")
     LVT = cv2.getTrackbarPos("L - V", "Trackbars")
     UHT = cv2.getTrackbarPos("U - H", "Trackbars")
     UST = cv2.getTrackbarPos("U - S", "Trackbars")
     UVT = cv2.getTrackbarPos("U - V", "Trackbars")
     area_threeshold = cv2.getTrackbarPos("AREA", "Trackbars")
    
     Yellow_Lower = np.array([LHT,LST,LVT])
     Yellow_Higher = np.array([UHT,UST,UVT])
     
     #Green_Lower = np.array([40,70,80]) # leave this for green, comment rest
     #Green_Higher = np.array([70,255,255])

     #Red_Lower = np.array([0,50,120])
     #Red_Higher = np.array([10,255,255])
    
     #Blue_Lower = np.array([90,60,0])
     #Blue_Higher = np.array([121,255,255])


     Yellow_Mask = cv2.inRange(hsv,Yellow_Lower,Yellow_Higher)
     #Green_Mask = cv2.inRange(hsv,Green_Lower,Green_Higher)
     #Red_Mask = cv2.inRange(hsv,Red_Lower,Red_Higher)
     #Blue_Mask = cv2.inRange(hsv,Blue_Lower,Blue_Higher)

     Yellow_Contour = cv2.findContours(Yellow_Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Mask,prioritization=contour made out of points, method
     Yellow_Contour = imutils.grab_contours(Yellow_Contour)
         
     #Green_Contour = cv2.findContours(Green_Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     #Green_Contour = imutils.grab_contours(Green_Contour)
    
     #Red_Contour = cv2.findContours(Red_Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     #Red_Contour = imutils.grab_contours(Red_Contour)
    
     #Blue_Contour = cv2.findContours(Blue_Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     #Blue_Contour = imutils.grab_contours(Blue_Contour)
    
    
     faces = face_cascade.detectMultiScale(hsv, 1.3, 5) #(image,Input image size factor 1.3=reduces 30%, higher value = quality vs. quantity),

     for c in Yellow_Contour: #For each contour found there is an area/area search
         area = cv2.contourArea(c)
         if area > 200*area_threeshold: #any object below this area will not have outlines drawn
             approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
             cv2.drawContours(frame,[c],-1,(0,255,0), 2) #image,all input contours found,size,color,width

             M = cv2.moments(c)

             cx = int(M["m10"]/ M["m00"]) #coordinates of the central moment
             cy = int(M["m01"]/ M["m00"])
             
             
             cv2.circle(frame,(cx,cy),5,(255,255,255),-1)
             cv2.putText(frame,"Yellow", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)
             
             if len(approx) == 3:
                 cv2.putText(frame,"Triangle", (cx-60, cy-60), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)
             
             if 4 <= len(approx) <= 6:
                 cv2.putText(frame,"Rectangle", (cx-60, cy-60), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)
                
             if 10 <= len(approx) <= 20:
                 cv2.putText(frame,"Circle", (cx-60, cy-60), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)
                 
     faces = face_cascade.detectMultiScale(hsv, 1.3, 5) #(image,input image size factor 1.3=reduces 30%, higher value = quality vs. quantity),
     for (x, y, w, h) in faces: #values from facemultidetect coordinate x,y,width and height)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = hsv[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
     
    
     cv2.imshow("HSV",hsv)
     cv2.imshow("Obraz",frame)
     cv2.imshow("Obraz zmaskowany",Yellow_Mask)
     result = cv2.bitwise_and(frame,frame,mask=Yellow_Mask)
     cv2.imshow("Obraz zmaskowany koloryzowany", result)





     if cv2.waitKey(1) == ord('q'): # waits 1 ms and checks if pressed key is , ord returns the unicode code point for a one-character string
        break

cap.release()
cv2.destroyAllWindows()