import cv2
import numpy as np
#change
def recognize_gesture(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
       
        max_contour = max(contours, key=cv2.contourArea)
    
        hull = cv2.convexHull(max_contour)
        
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
        
        defects = cv2.convexityDefects(max_contour, cv2.convexHull(max_contour, returnPoints=False))
        finger_count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, _ = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])
             
                angle = np.degrees(np.arctan2(far[1] - start[1], far[0] - start[0]) - np.arctan2(end[1] - start[1], end[0] - start[0]))
                if angle < 0:
                    angle += 180
                if angle > 10 and angle < 90:
                    finger_count += 1
        
  
        cv2.putText(frame, str(finger_count+1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    
    return frame

def main():
    
    cap = cv2.VideoCapture(0)
    
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        
       
        frame = cv2.resize(frame, (640, 480))
        
       
        frame = recognize_gesture(frame)
        cv2.imshow('Hand Gesture Recognition', frame)
        
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()