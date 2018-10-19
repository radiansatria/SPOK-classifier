import numpy as np
import flask
from flask_cors import CORS

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras_contrib.layers import CRF

from dictionary import Dictionary
from tokenize_with_nltk import sequence_to_ids_predict,tokenize

app = flask.Flask(__name__)
CORS(app)

model = None

WORDDICT_PATH="3840word_dict_with_20000embedding_word.txt"
LABELDICT_PATH="label_dict.txt"

word_dict = Dictionary()
label_dict = Dictionary()

MAX_SEQUENCE_LENGTH = 100

def create_custom_objects():
    instanceHolder = {"instance": None}
    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)
    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)
    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)
    def viterbi_acc(*args):
        method = getattr(instanceHolder["instance"], "viterbi_acc")
        return method(*args)
    return {"ClassWrapper": ClassWrapper ,"CRF": ClassWrapper, "loss": loss, "accuracy":accuracy,"viterbi_acc":viterbi_acc}

@app.before_first_request
def load_keras():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    
    global model
    
    print("Loading model..")
    model = load_model('final_model.h5',
                            custom_objects=create_custom_objects())
    model._make_predict_function()
    print("Model loaded.")
 

@app.before_first_request
def load_dictionary():
    print("loading word & label dictionary...")
    word_dict.load(WORDDICT_PATH)
    label_dict.load(LABELDICT_PATH)
    print("Word & label dictionary loaded.")
    print(word_dict.idx2str[0])
    print(label_dict.idx2str[0])

def prepare_text(text,wd):

    #tokenize
    raw_train_sents = tokenize(text)
    
    #ganti tiap token menjadi integer
    train_token=[sequence_to_ids_predict(sent,wd, True) for sent in raw_train_sents]
    
    #Pad input sequnces with 0
    x_train = pad_sequences(train_token, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    print('X train Shape {}'.format(x_train.shape))
    
    # return the processed text
    return x_train

#appending previous label
def append_last_label(label,offset_list,text_dict):
    
    if len(label) > 1:   
        if label[2:] == 'S':
            d = {'Subjek':text_dict['Subjek']}  
            offset_list.append(d)
            text_dict['Subjek'] = ''
        elif label[2:] == 'P':
            d = {'Predikat':text_dict['Predikat']} 
            offset_list.append(d)
            text_dict['Predikat'] = ''
        elif label[2:] == 'O':
            d = {'Objek':text_dict['Objek']} 
            offset_list.append(d)
            text_dict['Objek'] = ''
        elif label[2:] == 'K':
            d = {'Keterangan':text_dict['Keterangan']} 
            offset_list.append(d)
            text_dict['Keterangan'] = ''
        elif label[2:] == 'Pel':
            d = {'Pelengkap':text_dict['Pelengkap']} 
            offset_list.append(d)
            text_dict['Pelengkap'] = ''
    else:
        d = {'out':text_dict['out']}
        offset_list.append(d)
        text_dict['out'] = ''
    
    

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    
    if flask.request.method == "POST":
        if flask.request.form.get('text'):
            # read the text
            text = flask.request.form.get("text")
            
            # preprocess the image and prepare it for classification
            prepared_text = prepare_text(text,word_dict)
            token_text = tokenize(text)

            # classify the input text and then initialize the list
            # of predictions to return to the client
            preds = model.predict(prepared_text)
            preds = np.argmax(preds,axis=-1)

            data['predictions']=[]
            data['offset']=[]
            text_dict = {'Subjek':'', 'Predikat':'', 'Objek':'','Keterangan':'', 'Pelengkap':'', 'out':''}
            last = len(token_text[0]) - 1
            for i,p in enumerate(zip(token_text[0],preds[0])):
                label = label_dict.idx2str[p[1]]
                word = p[0]
                r = {'word':word,'label':label}

                if not data['predictions']:
                    if label == 'B-S': text_dict['Subjek'] = word
                    elif label == 'B-P': text_dict['Predikat'] = word
                    elif label == 'B-O': text_dict['Objek'] = word
                    elif label == 'B-K': text_dict['Keterangan'] = word
                    elif label == 'B-Pel': text_dict['Pelengkap'] = word
                    else: text_dict['out'] = word 
                else:
                    prev = data['predictions'][i-1]
                    prev_label = prev['label']
                    if len(label) > 1:
                        if (label[2:] == prev_label[2:] and label[0] != prev_label[0]) or (label[2:] == prev_label[2:] and prev_label[0] == 'I'): 
                            if label == 'I-S':text_dict['Subjek'] = text_dict['Subjek'] + ' ' + word
                            elif label == 'I-P':text_dict['Predikat'] = text_dict['Predikat'] + ' ' + word
                            elif label == 'I-O':text_dict['Objek'] = text_dict['Objek'] + ' ' + word
                            elif label == 'I-K':text_dict['Keterangan'] = text_dict['Keterangan'] + ' ' + word
                            elif label == 'I-Pel':text_dict['Pelengkap'] = text_dict['Pelengkap'] + ' ' + word
                        else:     
                            if label == 'B-S':
                                append_last_label(prev_label,data['offset'],text_dict)
                                text_dict['Subjek'] = word
                            elif label == 'B-P': 
                                append_last_label(prev_label,data['offset'],text_dict)
                                text_dict['Predikat'] = word
                            elif label == 'B-O': 
                                append_last_label(prev_label,data['offset'],text_dict)
                                text_dict['Objek'] = word
                            elif label == 'B-K': 
                                append_last_label(prev_label,data['offset'],text_dict)
                                text_dict['Keterangan'] = word
                            elif label == 'B-Pel': 
                                append_last_label(prev_label,data['offset'],text_dict)
                                text_dict['Pelengkap'] = word
                            
                    else:
                        append_last_label(prev_label,data['offset'],text_dict)          
                        text_dict['out'] = word

                if i == last: # append last data to offset list
                    append_last_label(label,data['offset'],text_dict)
                # returned predictions
                data['predictions'].append(r)

            
            
            
            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_keras()
    load_dictionary()
    app.run(debug=True)