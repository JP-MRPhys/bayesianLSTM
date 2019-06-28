import os

import numpy as np
import pandas as pd
import spacy
from keras.preprocessing import sequence

nlp = spacy.load('en')


def get_filenames(dir):
    nii_files = [];
    for dirName, subdirList, fileList in os.walk(dir):
        for filename in fileList:
            name = os.path.join(dirName, filename)
            if ".txt" in filename.lower() in filename:  # we only want the short axis images
                nii_files.append(name)
            else:
                continue
    return nii_files


def read_text(file):
    data = open(file, 'r', encoding="ISO-8859-1").read()
    # data=data.split(' ') #.split('.')
    return data


def get_imbd_data(dir):
    pos = get_filenames(os.path.join(dir, 'pos'))
    neg = get_filenames(os.path.join(dir, 'neg'))

    idx = 0
    data = []
    rating = []
    index = []

    for f in pos:
        data.append(read_text(f))
        rating.append(1)
        index.append(idx)
        idx = idx + 1

    for f in neg:
        data.append(read_text(f))
        rating.append(0)
        index.append(idx)
        idx = idx + 1

    dataset = list(zip(index, data, rating))

    np.random.shuffle(dataset)
    data2 = pd.DataFrame(data=dataset, columns=['entry', 'text', 'sentiment'])

    return data2


def word_to_sentence_embedding(sentence):
    # print(len(sentence))

    data = np.zeros((len(sentence), 300))
    k = 0

    for word in sentence:
        print(word)
        data[k, :] = get_word_embedding(word)
        k = k + 1

    return data


def get_word_embedding(word):
    emd = nlp(word)
    return emd.vector


def sentence_embedding(sentence, embedding_dim):
    tokens = nlp(sentence)
    data = np.zeros((len(sentence), embedding_dim))
    k = 0
    for token in tokens:
        data[k, :] = token.vector
        k = k + 1

    return data


def get_training_batch(data, batch_size, embedding_dim, num_classes, maxlen):
    num_classes = num_classes
    x = np.zeros([batch_size, maxlen, embedding_dim])
    y = np.zeros([batch_size, num_classes])

    index = 0

    for idx, row in data.iterrows():
        x[index, :, :] = sequence.pad_sequences([sentence_embedding(row['text'], embedding_dim)], maxlen=maxlen)
        if row['sentiment']:
            y[index, :] = np.array([0, 1])
        else:
            y[index, :] = np.array([1, 0])

        index = index + 1

    # print(x.shape)
    # print(y.shape)

    return x, y


def load_glove():
    glove_filename = '/home/jehill/python/NLP/datasets/GloVE/glove.6B.300d.txt'

    glove_vocab = []
    glove_embed = []
    embedding_dict = {}

    file = open(glove_filename, 'r', encoding='UTF-8')

    for line in file.readlines():
        row = line.strip().split(' ')
        vocab_word = row[0]
        glove_vocab.append(vocab_word)
        embed_vector = [float(i) for i in row[1:]]  # convert to list of float
        embedding_dict[vocab_word] = embed_vector
        glove_embed.append(embed_vector)

    print('Loaded GLOVE')
    file.close()

    return glove_vocab, glove_embed, embedding_dict


def get_train_test_data():
    DataDir = '/home/jehill/python/NLP/datasets/'

    train_dir = os.path.join(DataDir, 'train')
    test_dir = os.path.join(DataDir, 'test')

    train_data = get_imbd_data(train_dir)
    test_data = get_imbd_data(test_dir)

    n_train = len(train_data)
    print("Number of the training points : " + str(n_train))
    print("Number of the training points : " + str(len(test_data)))

    return train_data, test_data


if __name__ == '__main__':
    # glove_vocab, glove_embed, word_embedding_dict= load_glove()
    DataDir = '/home/jehill/python/NLP/datasets/'

    train_dir = os.path.join(DataDir, 'train')
    test_dir = os.path.join(DataDir, 'test')

    train_data = get_imbd_data(train_dir)
    # test_data =get_imbd_data(test_dir)

    text = train_data.text.values

    """
    
    batch_size=50
    index=0

    for index in range(0, len(train_data), batch_size):
        print (index)
        BATCH_X,BATCH_Y=get_training_batch(train_data[index:index+batch_size],batch_size=batch_size,embedding_dim=384,num_classes=2,maxlen=1000)
        print(np.shape(BATCH_X))
        print(np.shape(BATCH_Y))
    
    """
