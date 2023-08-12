# Image_to_word
## A flask app in which you can select a image and convert it into Word file
The aim of this project is to use a tesseract Optical Capture Recognition (OCR) engine so that we can edit the text in image. Scanned text documents, pictures stored in mobile phones and 
pictures taken by any Android device. The purpose of this application is to recognize text in scanned text documents, text images,Besides recognizing text
our project also aims to maintain format of text in image by recognizing paragraph and fonts used in image. As resources are limited and make this project approachable this project recognizes only ten fonts. This application will allow its users to edit text from these aforementioned documents and modify it, instead of 
wasting time on retyping and reformatting it.

https://github.com/rohitaroradung/image_to_word/assets/35729151/057d903b-6e37-4639-bef8-706e29a11091

### Workflow of flask app
1.	Image is passed to flask from html document
2.	Image is divided into further images into paragraph images
3.	Text is extracted from each paragraph image
4.	Besides paragraph image is further divided into text lines and then further into word images
5.	Word image is passed into font recognization model to predict font in paragraph
6.	Then dictionary of text and font is created and passed to create document function
7.	Create document function processed the dictionary and return the document name create
8.	When dowload button is clicked ‘download_file(document name)’ will return the document.

#### Results and Discussion
For line,word and paragraph detection,opencv is used and for font recognization,deep learning model is used.
 **Line Detection**
<br/>We got pretty Good results in line detection. This technique works in almost every case depending on noise present in image.
* **Sample Image used for Line detection**
  
  ![image018](https://github.com/rohitaroradung/image_to_word/assets/35729151/b6ba4a4e-2826-45fe-95e1-3e1942460b86)

* **Results of Line Detection in Image**
  <br/>Blue Lines resemble uppers<br/>
  Green Lines resemble lowers
  
  ![image012](https://github.com/rohitaroradung/image_to_word/assets/35729151/af2acbe1-76ff-4120-9f84-57bd53ddd5ca)

**Word/Alphabet detection in line**
<br/>We got pretty Good results in this detection too. Technique used is similar to line detection and similarly works in almost every case depending on noise present in image

**Result got from Words/Alphabet deetection in line son sample Image**

![image015](https://github.com/rohitaroradung/image_to_word/assets/35729151/2fd54337-3d87-4d32-9850-f496362ba6cd)

 **Paragraph Detection**
 <br/>For this Detection, we got good results for this image but we cannot automate this detection, technique used in this detection highly depends on iterations argument of dilate function
 
![image019](https://github.com/rohitaroradung/image_to_word/assets/35729151/fc98ba77-9fb6-4c9a-be0f-4d5e89a92110)

**Font recognition mode**
<br/>For font Recognization, we don’t get very good results due to limitation of data for this problem, we used mobile net and add some layers after mobile net and re-train whole model for our dataset.
<br/>We got these accuracies in percentage
•	Train accuracy-95
•	Validation accuracy-67
•	Test accuracy-71

### Instructions for user

* Install tesseract ocr, follow instructions at https://tesseract-ocr.github.io/tessdoc/Installation.html
* Create a anaconda environment with python 3.7 or higher
* Use the pip install -r requirements.txt command to install all of the Python modules and packages listed in your requirements.txt file.
* Run app.py or use "flask run" command to run local flask app  

#### Issues known

*	Algorithm find it difficult to process the image with many graphics inside image
*	Quality of Ocr depends on Noise of image
*	Paragraph detection highly depends on number of iterations used in dilate function
*	Application assume that text is in paragraph form
*	Application does not process any tables, graphs in it

##### References
*	A gentle introduction to OCR by towards Data Science.
*	https://medium.com/brightlab-techblog/deep-learning-based-text-detection-and-recognition-in-research-lab-bb3d61797f16
*	https://medium.com/@hdinhofer/optical-character-recognition-ocr-a-branch-of-computer-vision-76887e1d6ab0



