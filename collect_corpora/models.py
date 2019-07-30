import os
import csv
import nltk
import re
import csv
from string import punctuation
from pymystem3 import Mystem

russian_stopwords = nltk.corpus.stopwords.words("russian")
russian_stemmer = nltk.stem.snowball.SnowballStemmer("russian")

# I want to leave possible use all of the corpora, or some subset of them
# Our options are after the function

def create_train_and_test_folder(corpora_subset,new_train_filename,new_test_filename):
    # TODO: Not the most elegant solution, how to avoid chdir without getting os dependant?
    os.chdir("..")
    corpath =  os.getcwd() + "\\corpora\\annotated\\"
    new_path = os.getcwd() + '\\collect_corpora\\TemporalCorpora\\'

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
        print("It is over")
        print("Your train set has {} sexist comments and {} non_sexist comments".format(sex_tr,non_sex_tr))
        print("Your test set has {} sexist comments and {} non_sexist comments".format(sex_te, non_sex_te))


# vk preprocessing - remove with regex IDs and quote prefixes

def remove_with_regex(text):
    regex = re.compile("\[[i][d][0-9]*\|.*],")
    regex1 = re.compile('^.*[ писал(а):]$')
    words = [re.sub(regex,"",word) and re.sub(regex1,"",word) for word in text.split()]
    return ' '.join(words)

# punctuation removed and words are lowercased
def remove_punctuation_and_make_lowercase(text):
    sentences = [(sentence.translate(str.maketrans('', '', punctuation))).lower() for sentence in text.split()]
    return ' '.join(sentences)


# Remove stopwords
def remove_stowords(text):
    text = [word for word in text.split() if word.lower() not in russian_stopwords]
    return ' '.join(text)

lemmatizer = Mystem()

def perform_lemmatization(text):
    text = lemmatizer.lemmatize(text)
    return ''.join(text)

def stemmatize(text):
    text = [russian_stemmer.stem(word) for word in text.split()]
    return ' '.join(text)

# I want to separate it in three preprocessing phases
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

test = "ОЖП, разведенка с прицепом, РСП, разведенка, что тогда что, я не знаю, думать"
print(preprocessing(test,True,True,False,True))
#
#TODO: save all phases of preprocessing
def save_preprocessing_phase(filename,filename_out,punctuationBool,stopwordsBool,lemmBool):
     with open('TemporalCorpora\\'+ filename +'.csv', encoding="utf-8") as main_file, open('TemporalCorpora\\'+filename_out+'.csv',"w",encoding='utf-8') as file_no_stop:
         fieldnames = ['Label', 'Text']
         csv_reader = csv.DictReader(main_file)
         no_stopwords = csv.DictWriter(file_no_stop, fieldnames=fieldnames)
         no_stopwords.writeheader()
         for row in csv_reader:
             no_stopwords.writerow({'Label':row['Label'],'Text': preprocessing(row['Text'],punctuationBool,stopwordsBool,lemmBool,False)})









media_corpora = 'media_1.csv, media_2.csv, media_3.csv'
forum_corpora = 'ant_1.csv, ns_1.csv, ant_2.csv'
all_corpora = media_corpora + ', ' + forum_corpora

#create_train_and_test_folder(all_corpora,'tmp_train.csv','tmp_test.csv')
#save_preprocessing_phase('tmp_test','tmp_test_no_stopwords',False,True,False)
#save_preprocessing_phase('tmp_test_no_stopwords','tmp_test_no_punctuation',True,False,False)
#save_preprocessing_phase('tmp_test','tmp_test_just_lemm_and_no_punct',True,False,True)
#save_preprocessing_phase('tmp_test_no_punctuation','tmp_test_lemmatized',False,False,True)
#save_preprocessing_phase('tmp_train','tmp_train_no_stopwords',False,True,False)
#save_preprocessing_phase('tmp_train_no_stopwords','tmp_train_no_punctuation',True,False,False)
#save_preprocessing_phase('tmp_train_no_punctuation','tmp_train_lemmatized',False,False,True)
#save_preprocessing_phase('tmp_train','tmp_train_just_lemm_and_no_punct',True,False,True)
