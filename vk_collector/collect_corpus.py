#!/usr/bin/python
import requests
import csv
from codes import vk_api_vers,service_token
from wiki_one_unfinished import get_html_from_link,handling_encoding
from bs4 import BeautifulSoup

# this is made from vk via vk API
def make_corpus(name,community, query_list, service_token,vk_api_vers):
    with open(name, 'a', newline='', encoding="utf-8") as csvfile:
        headers = ['Post_id','Comment_id','Label','Text']
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        sexist = 'sexist'
        for query in query_list:
            # Step one: we get the list of id's of topics which have to do something with the US
            url = 'https://api.vk.com/method/wall.search?owner_id={}&own=1&owners_only=1&query=\'{}\'&access_token={}&v={}'.format(community,query,service_token,vk_api_vers)
            topic_ids = requests.get(url)
            for post_id in topic_ids.json()["response"]['items']:
                new_url = 'https://api.vk.com/method/wall.getComments?owner_id={}&post_id={}&need_likes=1&count=100&sort=asc&preview_length=0&access_token={}&v={}'.format(community,post_id['id'],service_token,vk_api_vers)
                comments = requests.get(new_url)

                for cur_comment in comments.json()["response"]['items']:
                    try:
                        writer.writerow({'Post_id':post_id['id'], 'Comment_id': cur_comment['id'],'Label': sexist, 'Text': cur_comment['text']})
                    except:
                        pass


query_list= {'сексизм',"meToo",'сексуальные домогательства', 'декриминализация побоев','изнасилование','феминизм','Шурыгина','домогательства'}


#make_corpus("sexism_RT.csv",-40316705,query_list, service_token,vk_api_vers)

#make_corpus('sexism_lentahc.csv',-29534144,query_list,service_token,vk_api_vers)

#make_corpus('sexism_medusa.csv',-76982440,query_list,service_token,vk_api_vers)



def make_corpus_ant_forum(name,link_to_topic):
    with open(name, 'w', newline='', encoding="utf-8") as csvfile:
        headers = ['Label', 'Text']
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        html = handling_encoding(get_html_from_link(link_to_topic))
        soup = BeautifulSoup(html, 'html.parser')
        number_of_pages = soup.find_all("div", {"class": "pagination"})
        number_of_pages = str(number_of_pages)
        number_of_pages = number_of_pages.split(' ')[1]
        for x in range(0, int(number_of_pages)):
            print(x)
            x = x + 25

make_corpus_ant_forum('test.csv','https://antiwomen.ru/viewtopic.php?f=1&t=34252')



















