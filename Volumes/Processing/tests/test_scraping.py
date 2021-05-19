def testScraping(url='http://www.customweather.com/'):
    import scrapping_ap 
    try:
        text=scrapping_ap.scrap_raw_text(url)
        return True,text
    except Exception as e:
        return False,str(e)

def testUrlValidators(url='http://google'):
    import scrapping_ap
    try:
        return scrapping_ap.chek_Url(url)
    except Exception as e:
        return False, str(e)

def testManualEntry_scrapingDB():
    import scrapping_ap
    try:
        scrapping_ap.manualentry_Scrapingdb()
        return True
    except Exception as e:
        return  False, str(e)

def test_get_innapropriate_links():
    import scrapping_ap
    try:
        ret=scrapping_ap.get_innapropriate_links()
        return True,ret
    except Exception as e:
        return False, str(e)

def test_add_listUrl_to_ScrapingDBcsv():
    import scrapping_ap
    try:
        url_list = scrapping_ap.get_innapropriate_links()
        scrapping_ap.add_listUrl_to_ScrapingDBcsv(url_list)
        return True
        #we have to add a test that can check number of rows changed in csv file
    except Exception as e:
        return  False, str(e)

def test_cleaning_url_classification():
    import scrapping_ap
    try:
        return scrapping_ap.cleaning_url_classification()
    except Exception as e:
        return False, str(e)

def test_extract_english_sites():
    import scrapping_ap
    try:
        return scrapping_ap.extract_english_sites()
    except Exception as e:
        return False, str(e)

def test_concat_ScrapingDB_UrlClassification_EnglishSites():
    import scrapping_ap
    try:
        return scrapping_ap.concat_ScrapingDB_UrlClassification_EnglishSites()
    except Exception as e:
        return False, str(e)

def test_reduce_data_size():
    import scrapping_ap
    try:
        return scrapping_ap.reduce_data_size()
    except Exception as e:
        return False, str(e)

def test_scrap_from_db_URL():
    import scrapping_ap
    try:
        return scrapping_ap.scrap_from_db_URL()
    except Exception as e:
        return False, str(e)




if __name__== '__main__':
   #print(testUrlValidators())
   #print(testManualEntry_scrapingDB())
    #print(test_get_innapropriate_links())
    #print(test_add_listUrl_to_ScrapingDBcsv())
    #print(test_scrap_from_csv())
    #print(test_cleaning_url_classification())
       # print(test_extract_english_sites())
    #print(test_concat_ScrapingDB_UrlClassification_EnglishSites())
    #print(test_reduce_data_size())
    print(test_scrap_from_db_URL())


