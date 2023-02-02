import cv2
import imutils
import pytesseract
import streamlit as st 
from PIL import Image,ImageEnhance
import numpy as np 
import pandas as pd
import os
import csv
import datetime


# vehicle number plate ocr recognition
 
def vehicle(image):

    image=np.array(image.convert('RGB')) # convert to array

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray scale

    gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17) # bilateral filter is used for smoothening images and reducing noise

    edged = cv2.Canny(gray_image, 30, 200) # Canny Edge Detection is used to detect the edges in an image

    cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # finding contours is like finding white object from black background

    image1=image.copy() # create a copy of image to use later

    cv2.drawContours(image1,cnts,-1,(0,255,0),3) #sed to draw shape provided you have its boundary points

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30] #Contours are used for shape analysis and object detection and recognition

    screenCnt = None

    image2 = image.copy()

    cv2.drawContours(image2,cnts,-1,(0,255,0),3) 

    # crop,save and show the numberplate from the image
    i=7
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4: 
            screenCnt = approx
            x,y,w,h = cv2.boundingRect(c) 
            new_img=image[y:y+h,x:x+w]
            cv2.imwrite('./'+str(i)+'.png',new_img)
            i+=1
            break
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    Cropped_loc = './7.png'
    cv2.imshow("cropped", cv2.imread(Cropped_loc))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # convert the cropped image to string using pytesseract
    
    plate = pytesseract.image_to_string(Cropped_loc, lang='eng')

    # Remove the unwanted spaces

    def remove(string):
        return string.replace(" ", "")

    plate1=remove(plate)

    # Remove '\n\x0c' 
    str(plate1)
    plate=plate1.replace('\n\x0c','')

    # Append the plate to a list
    lst=[]
    lst.append(plate)
 
    # using now() to get current time
    current_time = datetime.datetime.now()

    # Append the current time to another list
    lst_tm=[]
    lst_tm.append(current_time)

    # create a data frame using the above details
    df = pd.DataFrame({'Licence_Plate': lst,'Entry_Date': lst_tm})

    field_names = ['Licence_Plate','Entry_Date']

    # write the dataframe to a csv file
    # check if the file present in the directory to elimate the header to print during each entry

    if (os.path.exists("vd.csv") == False):
        df.to_csv('vd.csv', mode='a', header=field_names)
    else:
        df.to_csv('vd.csv', mode='a', header=False)

    # read the created csv file
    df1=pd.read_csv('vd.csv',index_col=0)

    # display the df1 using streamlit
    st.dataframe(df1)

    # return the image value to show in stream lit
    img=Cropped_loc

    return img


def main():
    """VEHICLE ENTRY RECORDING SYSTEM"""

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">VEHICLE ENTRY RECORDING SYSTEM</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # create a uploader to input images 
    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)

    # create a save button and call the vehicle function
    if st.button("Save Vehicle Details"):
        result_img= vehicle(our_image)
        st.image(result_img)


# call the main function
if __name__ == '__main__':
    main()
