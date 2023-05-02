import openai
import json
import requests
from ibm_watson import SpeechToTextV1, TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')


def speech_to_text(audio_binary):
    # setup Watson Speech to Text Api

    authenticator = IAMAuthenticator(
        os.environ.get('IBM_SPEECH_TO_TEXT_APIKEY'))
    speech_to_text = SpeechToTextV1(
        authenticator=authenticator
    )

    speech_to_text.set_service_url(
        os.environ.get('IBM_SPEECH_TO_TEXT_URL')
    )
    # Set up the body of our HTTP request

    # send a HTTP Post Request
    # response = requests.post(api_url, params=params, data=audio_binary).json()
    # speech_to_text.set_detailed_response(True)
    response = speech_to_text.recognize(
        audio=audio_binary,
        model='en-US_Multimedia',
    )

    # Access response from methodName
    # print(json.dumps(response.get_result(), indent=2))

    # Parse the response to get our transcribed text
    results = response.get_result()
    text = 'null'
    while bool(results):
        # print('speech to text response:', response)
        text = results.get('results').pop().get(
            'alternatives').pop().get('transcript')
        return text


def text_to_speech(text, voice=""):

    authenticator = IAMAuthenticator(
        os.environ.get('IBM_TEXT_TO_SPEECH_APIKEY'))

    text_to_speech = TextToSpeechV1(authenticator=authenticator)

    text_to_speech.set_service_url(os.environ.get('IBM_TEXT_TO_SPEECH_URL'))

    print('text: ', text)
    print('voice: ', voice)
    response = text_to_speech.synthesize(
        text=text, voice='en-US_MichaelV3Voice', accept='audio/mp3').get_result()

    print('text to speech response:', response)
    return response.content


def openai_process_message(user_message):
    # Set the prompt for OpenAI Api
    prompt = "\"Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations. " + user_message + "\""
    # Call the OpenAI Api to process our prompt
    openai_response = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, max_tokens=4000)
    # print("openai response:", openai_response)
    # Parse the response to get the response text for our prompt
    response_text = openai_response.choices[0].text
    return response_text
