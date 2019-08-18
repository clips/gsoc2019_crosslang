""" In this part preprocessing of the text is made, and everything is separated to train and test.
It is also possible to make similar things with inbuild pytorch functions, but it seemed like a good idea to provide several
already preprocessed corpora, to speed up the process for people working with Russian without close knowledge of the language.
"""

import os

# Imports for preprocessing of Russian language texts
import nltk
import re
import csv
from string import punctuation
from pymystem3 import Mystem


russian_stopwords = nltk.corpus.stopwords.words("russian")
russian_stemmer = nltk.stem.snowball.SnowballStemmer("russian")
lemmatizer = Mystem()

# The function is to create from the given names of subcorpora a train and test dataset. There are quite a few functions which
#split in more efficient way, but I wanted to ensure that the elements even from the smallest corpora (as RT_media_corpus) will
#get in both test and train.

def create_train_and_test_folder(corpora_subset,new_train_filename,new_test_filename):
    # TODO: Not the most elegant solution, how to avoid chdir without getting os dependant?
    os.chdir("..")
    corpath =  os.getcwd() + "\\russian sexist corpora\\annotated\\"
    new_path = os.getcwd() + '\\sexism_detector\\TemporalCorpora\\'

    corpora_subset = [corpus.strip() for corpus in corpora_subset.split(',')]
    sex_tr = sex_te = non_sex_tr = non_sex_te = 0
    with open(new_path+new_train_filename, 'w',encoding="utf-8") as writeTrainFile,  open(new_path+new_test_filename, 'w',encoding="utf-8") as writeTestFile:
        fieldnames = ['Label', 'Text']
        writer_train = csv.DictWriter(writeTrainFile, fieldnames=fieldnames)
        writer_test = csv.DictWriter(writeTestFile, fieldnames=fieldnames)
        writer_train.writeheader()
        writer_test.writeheader()
        for corpus in corpora_subset:
            with open((corpath+corpus),encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                tmp_counter_sex = tmp_counter_nonsex = 0
                for row in reader:
                    label = row['Label']
                    if label=="sexist":
                        if tmp_counter_sex>10:
                            writer_test.writerow({'Label': str(label+";"),'Text':row['Text']})
                            sex_te+=1
                            tmp_counter_sex = 0
                        else:
                            writer_train.writerow({'Label': str(label+";"), 'Text': row['Text']})
                            sex_tr+=1
                            tmp_counter_sex+=1
                    elif label=="non_sexist":
                        if tmp_counter_nonsex > 15:
                            writer_test.writerow({'Label': str(label+";"), 'Text': row['Text']})
                            non_sex_te+=1
                            tmp_counter_nonsex = 0
                        else:
                            writer_train.writerow({'Label': str(label+";"), 'Text': row['Text']})
                            non_sex_tr+=1
                            tmp_counter_nonsex+=1
        print("Here we go...")
        print("Your train set has {} sexist comments and {} non_sexist comments".format(sex_tr,non_sex_tr))
        print("Your test set has {} sexist comments and {} non_sexist comments".format(sex_te, non_sex_te))

# vk preprocessing - remove with regex IDs and quote prefixes

def remove_with_regex(text):
    regex = re.compile("\[[i][d][0-9]*\|.*],")
    regex1 = re.compile('.*писал\(а\):')
    regexp = re.compile('[\r\n]')
    words = [re.sub(regex,"",word) and re.sub(regex1,"",word) and re.sub(regexp,"",word) for word in text.split()]
    return ' '.join(words)

# punctuation removed and words are lowercased

def remove_punctuation_and_make_lowercase(text):
    sentences = [(sentence.translate(str.maketrans('', '', punctuation))).lower() for sentence in text.split()]
    return ' '.join(sentences)

# Remove stopwords

def remove_stowords(text):
    text = [word for word in text.split() if word.lower() not in russian_stopwords]
    return ' '.join(text)

# Lemmatization is performed

def perform_lemmatization(text):
    text = lemmatizer.lemmatize(text)
    return ''.join(text)

# Stemmatization is performed

def stemmatize(text):
    text = [russian_stemmer.stem(word) for word in text.split()]
    return ' '.join(text)


# Some would say it would be much smarter to do it earlier. Some may be are correct.

def cleaning_up_the_nulls(filename,filename_new):
    with open('TemporalCorpora\\' + filename, encoding="utf-8") as file_to_check, open('TemporalCorpora\\'+ filename_new,'w', encoding="utf-8") as file_to_write_in:
        fieldnames = ['Label', 'Text']
        csv_reader = csv.DictReader(file_to_check)
        new_file = csv.DictWriter(file_to_write_in, fieldnames=fieldnames)
        new_file.writeheader()
        for row in csv_reader:
            if row['Text'] is None or row['Text']=="":
                continue
            else:
                new_file.writerow({'Label':row['Label'],'Text':row['Text']})

# Three preprocessing phases. It is not very efficient, and can be improved.

def preprocessing(text, punct_low_case, stop_regex, lemm, stemm):
    if punct_low_case == True:
        text = remove_punctuation_and_make_lowercase(text)
    if stop_regex == True:
        text = remove_with_regex(text)
        text = remove_stowords(text)
    if lemm == True:
        text = perform_lemmatization(text)
    elif stemm == True:
        text = stemmatize(text)
    return text

# I prefer to have all different preprocessing stage corpora avaliable

def save_preprocessing_phase(filename,filename_out,punctuationBool,stopwordsBool,lemmBool):
     with open('TemporalCorpora\\'+ filename +'.csv', encoding="utf-8") as main_file, open('TemporalCorpora\\'+filename_out+'.csv',"w",encoding='utf-8') as file_no_stop:
         fieldnames = ['Label', 'Text']
         csv_reader = csv.DictReader(main_file)
         no_stopwords = csv.DictWriter(file_no_stop, fieldnames=fieldnames)
         no_stopwords.writeheader()
         for row in csv_reader:
             no_stopwords.writerow({'Label':row['Label'],'Text': preprocessing(row['Text'],punctuationBool,stopwordsBool,lemmBool,False)})

