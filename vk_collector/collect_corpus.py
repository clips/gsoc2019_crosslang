#!/usr/bin/python
import requests
import csv
from codes import vk_api_vers,service_token

def make_corpus(name,community, query_list, service_token,vk_api_vers):
    with open(name, 'a', newline='', encoding="utf-8") as csvfile:
        headers = ['Online_media','Post_id','Comment_id','Text']
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for query in query_list:
            # Step one: we get the list of id's of topics which have to do something with the US
            url = 'https://api.vk.com/method/wall.search?owner_id={}&own=1&owners_only=1&query=\'{}\'&access_token={}&v={}'.format(community,query,service_token,vk_api_vers)
            topic_ids = requests.get(url)
            for post_id in topic_ids.json()["response"]['items']:
                new_url = 'https://api.vk.com/method/wall.getComments?owner_id={}&post_id={}&need_likes=1&count=100&sort=asc&preview_length=0&access_token={}&v={}'.format(community,post_id['id'],service_token,vk_api_vers)
                comments = requests.get(new_url)
                for cur_comment in comments.json()["response"]['items']:
                    try:
                        writer.writerow({'Online_media': community,'Post_id':post_id['id'], 'Comment_id': cur_comment['id'], 'Text': cur_comment['text']})
                    except:
                        pass



query_list= {'США',"Мюллер",'Трамп', 'санкции','ЕС','Путин'}


make_corpus("all.csv",-40316705,query_list, service_token,vk_api_vers)

make_corpus('all.csv',-29534144,query_list,service_token,vk_api_vers)

make_corpus('all.csv',-76982440,query_list,service_token,vk_api_vers)
















