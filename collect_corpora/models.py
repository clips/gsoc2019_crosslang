import os
import csv


# TODO: Move this to helper functions
# I want to leave possible use all of the corpora, or some subset of them
# Our options are after the function

def create_train_and_test_folder(corpora_subset,new_train_filename,new_test_filename):
    # TODO: Not the most elegant solution, how to avoid chdir without getting os dependant?
    os.chdir("..")
    corpath =  os.getcwd() + "\\corpora\\annotated\\"
    new_path = os.getcwd() + '\\collect_corpora\\TemporalCorpora\\'

    corpora_subset = [corpus.strip() for corpus in corpora_subset.split(',')]
    sex_tr = sex_te = non_sex_tr = non_sex_te = 0
    for corpus in corpora_subset:
        with open((corpath+corpus),encoding="utf-8") as csvfile, open(new_path+new_train_filename, 'w',encoding="utf-8") as writeTrainFile,  open(new_path+new_test_filename, 'a',encoding="utf-8") as writeTestFile:
            reader = csv.DictReader(csvfile)
            tmp_counter_sex = tmp_counter_nonsex = 0
            for row in reader:
                if row['Label']=="sexist":
                    if tmp_counter_sex>10:
                        writeTestFile.write(row['Label'] + '; ' + row['Text']+'\n')
                        sex_te+=1
                        tmp_counter_sex = 0
                    else:
                        writeTrainFile.write(row['Label'] + '; ' + row['Text']+'\n')
                        sex_tr+=1
                        tmp_counter_sex+=1
                elif row['Label']=="non_sexist":
                    if tmp_counter_nonsex > 15:
                        writeTestFile.write(row['Label'] + '; ' + row['Text']+'\n')
                        non_sex_te+=1
                        tmp_counter_nonsex = 0
                    else:
                        writeTrainFile.write(row['Label'] + '; ' + row['Text']+'\n')
                        non_sex_tr+=1
                        tmp_counter_nonsex+=1
    print("It is over")
    print("Your train set has {} sexist comments and {} non_sexist comments".format(sex_tr,non_sex_tr))
    print("Your test set has {} sexist comments and {} non_sexist comments".format(sex_te, non_sex_te))

media_corpora = 'media_1.csv, media_2.csv, media_3.csv'
forum_corpora = 'ant_1.csv, ns_1.csv, ant_2.csv'
all_corpora = media_corpora + ', ' + forum_corpora
create_train_and_test_folder(all_corpora,'tmp_train.txt','tmp_test.txt')


