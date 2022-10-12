from django.shortcuts import render
from .form import UploadForm
from .models import Upload

import re

from string import punctuation

import easyocr

from pickle import load
from numpy import argmax
from keras_preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

reader = easyocr.Reader(['en'])
# load the tokenizer
tokenizer = load(open('./media/model/tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model('./media/model/my_model.h5')


def extract_features(filename):
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature

# map an integer to a word


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image


def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


def out(image, model, tokenizer):
    # pre-define the max sequence length (from training)
    max_length = 34
    # load and prepare the photograph
    photo = extract_features(image)
    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    stopwords = ['startseq', 'endseq']
    querywords = description.split()
    resultwords = [word for word in querywords if word.lower()
                   not in stopwords]
    result = ' '.join(resultwords)
    return result


def index(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            obj = form.instance

            image_url = "./media/"+str(obj.image)

            result = reader.readtext(image_url)

            text_in_image = ""
            for line in result:
                text_in_image = text_in_image + " " + line[1]

            content_in_image = out(image_url, model, tokenizer)

            isText = True
            if len(text_in_image) == 0:
                isText = False

            if isText:
                text_in_image = text_in_image.lower()
                text_in_image = re.sub(r"https?://\S+", "", text_in_image)
                text_in_image = re.sub(
                    f"[{re.escape(punctuation)}]", "", text_in_image)
                text_in_image = " ".join(text_in_image.split())
                text_in_image = re.sub(r"\b[0-9]+\b\s*", "", text_in_image)
                text_in_image = " ".join(
                    [w for w in text_in_image.split() if not w.isdigit()])
                text_in_image = " ".join(
                    [w for w in text_in_image.split() if w.isalpha()])
                if len(text_in_image) == 0:
                    isText = False

            return render(request, "index.html", {"obj": obj, "text_in_image": text_in_image, "content_in_image": content_in_image, "isText": isText})
    else:
        form = UploadForm()
        img = Upload.objects.all()
        final_img = ''
        for x in img:
            final_img = x
    return render(request, "index.html", {"form": form, "img": final_img})
