
from nltk.tokenize import word_tokenize
import nltk


UNKNOWN_TOKEN = '*UNKNOWN*'



def get_sentences_with_nltk(filepath, use_se_marker=False):
  """ Read tokenized SRL sentences from file.
    File format: {predicate_id} [word0, word1 ...] ||| [label0, label1 ...]
    Return:
      A list of sentences, with structure: [[words], predicate, [labels]]
  """
  nltk.download('punkt')
  sentences = []
  with open(filepath) as f:
    for line in f.readlines():
      inputs = line.strip().split('|||')
      lefthand_input = word_tokenize(inputs[0].decode("utf-8")) #dataset 2424 tidak ada UnicodeDecode error gunakan inputs[0] saja.
      # If gold tags are not provided, create a sequence of dummy tags.
      righthand_input = inputs[1].strip().split() if len(inputs) > 1 \
                          else ['O' for _ in lefthand_input[1:]]
      
      predicate = int(lefthand_input[0]) 
      words = lefthand_input[1:]
      labels = righthand_input
      sentences.append((words, predicate, labels))
  return sentences

def tokenize(sequence):
  """ Read tokenized SRL sentences from file.
    File format: {predicate_id} [word0, word1 ...] ||| [label0, label1 ...]
    Return:
      A list of sentences, with structure: [[words], predicate, [labels]]
  """
  sentences = []
  words = word_tokenize(sequence)
  sentences.append((words))
  
  return sentences

def string_sequence_to_ids(str_seq, dictionary, lowercase=False, pretrained_embeddings=None):
  """token/kata yang tidak terdapat dalam wordvec akan dianggap sebagai UNKNOWN_TOKEN
  """
  ids = []
  for s in str_seq:
    if s is None:
      ids.append(-1)
      continue
    if lowercase:
      s = s.lower()
    if (pretrained_embeddings != None) and not (s in pretrained_embeddings) :
      s = UNKNOWN_TOKEN
    ids.append(dictionary.add(s))
  return ids

def label_to_ids(str_seq, dictionary):
  """jika menggunakan fungsi ini pastikan ketika menampilkan prediksi index label juga ditambah 1
  """
  ids = []
  for s in str_seq:
    if s is None:
      ids.append(-1)
      continue
    if s not in dictionary :
      dictionary.add(s)
    ids.append(dictionary.get_index(s) + 1)
  return ids

def sequence_to_ids_predict(str_seq, dictionary, lowercase=False):
  """token/kata yang tidak terdapat dalam wordvec akan dianggap sebagai UNKNOWN_TOKEN
  """
  ids = []
  for s in str_seq:
    if s is None:
      ids.append(-1)
      continue
    if lowercase:
      s = s.lower()
    if s not in dictionary.str2idx :
      s = UNKNOWN_TOKEN
    ids.append(dictionary.get_index(s))
  return ids