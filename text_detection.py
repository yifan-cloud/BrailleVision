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

from google.cloud import vision
from google.cloud.vision import types
from google.cloud import texttospeech

import image_retrieval

# Instantiates a client
client = vision.ImageAnnotatorClient()

# modified from https://cloud.google.com/translate/docs/hybrid-glossaries-tutorial
def pic_to_text(img):
    """Detects text in an image file

    ARGS
    img: input image as numpy array

    RETURNS
    String of text detected in image
    """

    # encode img to byte string and pass to client
    image = client.image(content=cv2.imencode('.jpg', img)[1].tostring())

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
    # TODO: send audio to bluetooth
    # Name of file that will hold synthetic speech
    outfile = 'audio/output.mp3'

    # get image from realsense cam
    img = image_retrieval.getColorImg()
    
    # photo -> detected text
    text_to_speak = pic_to_text(img)
    # detected text -> synthetic audio
    text_to_speech(text_to_speak, outfile)

if __name__ == '__main__':
    main()