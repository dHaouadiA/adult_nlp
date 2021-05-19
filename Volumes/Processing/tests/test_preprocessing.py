from Processing.src import preprocessing_ap

def test_ngrams(s='this is ngram text, ngram text are awesome', n=2):
    try:
            ngrams=preprocessing_ap.ngrams_with_nltk(s, n)
            return True, ngrams
    except Exception as e:
        return False, str(e)

def test_bagofwords(texts=['John likes to watch movies. Mary likes too.',
             'John also likes to watch football games.']):    
    try:
        bag=preprocessing_ap.bag_of_word(texts)
        return True,bag
    except Exception as e:
        return False, str(e)
#unitest for preprocessing functions
def test_strip_html_tags():    
    assert(preprocessing_ap.strip_html_tags('<h1>Title</h1> <![ endif]  Google T>')=='Title ')
    #return preprocessing_ap.strip_html_tags('<h1>Title</h1> <![ endif]  Google T>')


def test_remove_extra_whitespaces():
    assert(preprocessing_ap.remove_extra_whitespaces('hi my      name            is                             assouma')=='hi my name is assouma')
    assert(preprocessing_ap.remove_extra_whitespaces('hi my      name            is                             assouma')=='hi my   name      is assouma')

def test_remove_accented_chars():
    assert (preprocessing_ap.remove_accented_chars('àîûêùùùîï')=='aiueuuuii')

def test_removePunctuation():
    assert(preprocessing_ap.removePunctuation('hi!are u ok????!!')==('hiare u ok'))

def test_remove_stop_words():
    assert(preprocessing_ap.remove_stop_words('i am going to go to the store and park')==['going', 'go', 'store', 'park'])

def test_text_preprocessing():
    #assert(preprocessing_ap.text_preprocessing('hi  <html></html>      everyone!!! asma hù  123')==['hi', 'everyone', 'asma', 'hu'])
    return preprocessing_ap.text_preprocessing("<![ endif]  Google T")


def test_tokenization_with_RegularExpression():
    assert(pap.word_tokenization_with_RegularExpression('it enable humans')==['it', 'enable', 'humans'])

def test_remove_contractions():
    return pap.remove_contractions("won't've")

def test_preprocessed_CorpusFromWebSite():
    try:
        return True, pap.preprocessed_CorpusFromWebSite(csv_file='corpusfromwebsiteCleaned.csv')
    except Exception as e:
        return False, str(e)

def test_detectlang():
    try:
        english_corpus=pap.detect_english()
        return True, english_corpus
    except Exception as e: 
        return  False, str(e)


if __name__== "__main__":
    # print(test_remove_contractions())
    #print(test_strip_html_tags())
    #print(test_remove_extra_whitespaces())
    #print(test_remove_accented_chars())
    #print(test_removePunctuation())
    #print(test_remove_stop_words())
    # print(test_text_preprocessing())
    #print(test_tokenization_with_RegularExpression())
    print(test_preprocessed_CorpusFromWebSite())
    #print(test_detectlang())
    
   
