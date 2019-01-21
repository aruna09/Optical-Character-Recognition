# Optical-Character-Recognition

This has been implemented as a part of the final project for the elective course- Digital Image processing.

## What is Optical Character Recognition?
OCR is a technology which was developed to read and identify text from a scanned document or a digital image.

## Where is it used?
OCR technology can be used to convert a hard copy of a document into its corresponding electronic version. For example, if you scan a multipage document into a digital image, such as a jpg file, an OCR program, will recognize the text and convert the document to an editable text file. 

While OCR technology was originally designed to recognize printed text, it can be used to recognize and verify handwritten text as well. For example, postal services such as USPS use OCR software to automatically process letters and packages based on the address.

Optical Character Recognition also aids by extracting relevant information and automatically enters it into electronic database instead of the conventional way of manually retyping the text.

## Our approach
The building blocks of our network consists of:

Two sets of Convolutional layer one dropout and a fully connected layer for output.


## Steps Followed:

Taking input: Inputting the train image set from folders using flow_from_directory() 

Preprocessing and Data Augmentation: ImageDataGenerator for real-time data augmentation

Training: fit_generator for training the model
          layer freezing by using dropout
          
          
Testing : Inputing images into the training model for predicting. 

## Future work

I would like to work on developing a frontend aspect to it. This will not only make the application whole and complete but also make it user friendly.
