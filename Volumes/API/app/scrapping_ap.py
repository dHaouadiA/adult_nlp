
import urllib.request
import requests, sys
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime as dt
def scrap_raw_text(url):
    """This function return the raw text of a given website

    Arguments:
        url {[str]} -- [website Url]
    """
    hdr = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A'}

    print('scraping from {}'.format(url))
    if (chek_Url(url)):
        try:
            #page= requests.get(url)
            req = Request(url,headers=hdr)
            page = urlopen(req)
            #html_contents = page.content
            #soup = BeautifulSoup(html_contents, "html.parser")
            soup = BeautifulSoup(page)

            for script in soup(["script", "style","a","<div id=\"bottom\" >"]):
                script.extract()    
            text = soup.findAll(text=True)
            ws_corpus=""
            for p in text:
                ws_corpus+=' '+p
            
            ws_corpus=ws_corpus.replace('\n', ' ').replace('\r',' ')
            #print(ws_corpus)
        except Exception:
            return np.nan
    else:
        print('invalid url')
        return np.nan
    
    return  ws_corpus

from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
def get_innapropriate_links():
    site= "https://toppornsites.com/"
    hdr = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A'}
    req = Request(site,headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page)
    url_list=[]
    for a in soup.find_all('a', href=True):
        if  not (( "#" in a['href']) or ("javascript:;" in a['href'])):
            url_list.append(a['href'])
    return list(set(url_list))

def add_listUrl_to_ScrapingDBcsv(url_list):
    #Open csv file at start
    outfile = open('ScrapingDB.csv', 'a', newline='')
    w = csv.writer(outfile)  # Need to write the user input to the .csv file.
    label= 'Adult'
    for url in url_list:
        w.writerow([url, label])  
    df =pd.read_csv('ScrapingDB.csv')
    df.drop_duplicates(inplace=True)
    df.columns=['url','label']
    df.to_csv('ScrapingDB.csv', index= False)

def scrap_from_df_com(url_DB='df_com.csv'):
    df=pd.read_csv(url_DB)
    df['corpus']=df['url'].apply(scrap_raw_text)
    df['dateTime']=pd.Series([dt.datetime.now().strftime('%Y-%m-%d')]*len(df['url']))
    df=df.dropna()
    df.to_csv('df_com_Scrapped_Corpus.csv',mode='a', index=False)
    return df

# Check for URL Validation:
def chek_Url(url):
    import validators
    return validators.url(url)

def scrap_from_csv(url_DB='ScrapingDB.csv'):
    import datetime as dt
    df=pd.read_csv(url_DB)
    #add 'corpus' column from scrapped url 
    # apply : appliquer une fct 3lé column
    #df=df.head(10)
    df['corpus']=df['url'].apply(scrap_raw_text)
    #add new column 'datetime'
    df['dateTime']=pd.Series([dt.datetime.now().strftime('%Y-%m-%d')]*len(df['url']))
    df=df.dropna()
    df.to_csv('CorpusFromWebSite.csv',mode='a', index=False)
    return df

# Scraping csv file input:
import csv

def manualentry_Scrapingdb():

    #Open csv file at start
    outfile = open('ScrapingDB.csv', 'a', newline='')
    w = csv.writer(outfile)  # Need to write the user input to the .csv file.

    #Everything wrapped in a while True loop, you can change to any loop accordingly
    while True:
        url = input("Enter an url: ")  # Generate data for each column to fill in to the output file.
        while (not chek_Url(url)):
            url = input("Please a valid Url: ")
         
        label = input("Enter a label: ")  # Each line asks the user  add data do the line.
        print(url, label)  # Prints the line of user data

        input4 = input("Append to Scraping DB Y/N : ")  # Asks user to append the data to the output file.
        if input4.lower() == "y":
            w.writerow([url, label])  # <-This is the portion that seems to fall apart.
            print("Row added")
        if input4.lower() == "n":
            print("SKIPPING. RESTARTING....")
        #If you see stop, stop writing, close the file and exit
        if input4.lower() == 'stop':
            print('Not writing anymore! Stopping')
            outfile.close()
            exit()
        else:
            print("Invalid entry restarting program.") 

