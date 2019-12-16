"""
Some relevant tutorials:
https://cloud.google.com/vision/docs/fulltext-annotations
https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/functions/ocr/app/main.py
https://cloud.google.com/vision/docs/ocr
https://cloud.google.com/translate/docs/hybrid-glossaries-tutorial
"""

import io
import html
import cv2
import pyrealsense2 as rs
import numpy as np
from google.cloud import vision
from google.cloud.vision import types
from google.cloud import texttospeech

import image_retrieval

# Instantiates a client
client = vision.ImageAnnotatorClient()


def imageCap():
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
        
            #convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
            #colorwriter.write(color_image)
            #depthwriter.write(depth_colormap)
        
            cv2.imshow('Stream', depth_colormap)
            cv2.imshow('try',color_image)
        
            cv2.imwrite("what.jpg",color_image)
            
            x= depth_frame.get_width()/2
            y= depth_frame.get_height()/2
            
            depth_to_center= depth_frame.get_distance(int(x),int(y))
            print('distance between camera and the center point',depth_to_center)
            
            if cv2.waitKey(1) == ord("e"):
                break
               #depth_to_center= depth_frame.get_distance(int(x),int(y))
               #print('distance between camera and the center point',depth_to_center)
            
    finally:
    #colorwriter.release()
    #depthwriter.release()
        pipeline.stop()

#RETURN THE LABELS
def label_to_text(img):
    image = vision.types.Image(content=cv2.imencode('.jpg', img)[1].tostring())
    response = client.label_detection(image=image)
    labels = response.label_annotations
    label_list=list()
    print('Labels:')

    for label in labels:
        #print(label.description)
        label_list.append(label.description)
    print(label_list)
    string_label=''.join([str(elem)for elem in label_list])
    
    return string_label
    
    
    

# modified from https://cloud.google.com/translate/docs/hybrid-glossaries-tutorial
def pic_to_text(img):
    """Detects text in an image file

    ARGS
    img: input image as numpy array

    RETURNS
    String of text detected in image
    """

    # encode img to byte string and pass to client
    image = vision.types.Image(content=cv2.imencode('.jpg', img)[1].tostring())

    # For dense text, use document_text_detection
    # For less dense text, use text_detection
    response = client.document_text_detection(image=image)
    text = response.full_text_annotation.text

    return text

# modified from https://cloud.google.com/translate/docs/hybrid-glossaries-tutorial
def text_to_speech(text, outfile):
    """Converts plaintext to SSML and
    generates synthetic audio from SSML

    ARGS
    text: text to synthesize
    outfile: filename to use to store synthetic audio

    RETURNS
    nothing
    """

    # Replace special characters with HTML Ampersand Character Codes
    # These Codes prevent the API from confusing text with
    # SSML commands
    # For example, '<' --> '&lt;' and '&' --> '&amp;'
    escaped_lines = html.escape(text)

    # Convert plaintext to SSML in order to wait two seconds
    #   between each line in synthetic speech
    ssml = '<speak>{}</speak>'.format(
        escaped_lines.replace('\n', '\n<break time="2s"/>'))

    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Sets the text input to be synthesized
    synthesis_input = texttospeech.types.SynthesisInput(ssml=ssml)

    # Builds the voice request, selects the language code ("en-US") and
    # the SSML voice gender ("MALE")
    voice = texttospeech.types.VoiceSelectionParams(
        language_code='en-US',
        ssml_gender=texttospeech.enums.SsmlVoiceGender.MALE)

    # Selects the type of audio file to return
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3)

    # Performs the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(synthesis_input, voice, audio_config)

    # Writes the synthetic audio to the output file.
    with open(outfile, 'wb') as out:
        out.write(response.audio_content)                                                                                
        print('Audio content written to file ' + outfile)

def main():
    # Name of file that will hold synthetic speech
    outfile1 = 'audio_text_mode.mp3'
    outfile2 = 'audio_label_mode.mp3'
    # get image from realsense cam
    img = image_retrieval.getColorImg()
    # save img to file for debugging purposes
    with open('what.jpg', 'rb') as image_file:
        content = image_file.read()
    # photo -> object labels, detected text
    label = label_to_text(img)
    text_to_speak = pic_to_text(img)
    # detected text -> synthetic audio
    text_to_speech(text_to_speak, outfile1)
    
    text_to_speech(label, outfile2)
    
    print(text_to_speak)

if __name__ == '__main__':
    main()