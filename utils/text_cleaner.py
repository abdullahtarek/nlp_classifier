import string
import re

class Cleaner():
    ''' clean text before feed it into Bert model for embedding'''
    def __init__(self):
        pass
    def remove_digits(self, doc):
        '''there are numbers concatenated to words so we need remove them'''
        pattern = '[0-9]'
        doc = re.sub(pattern, '', doc)
        return doc
    
    def remove_punctuations(self, doc):
        english_punctuations = string.punctuation
        punctuations_list = english_punctuations
        translator = str.maketrans('', '', punctuations_list)
        return doc.translate(translator)
    
    def remove_double_spaces(self, doc):
        return re.sub(' +', ' ', doc)

    def clean(self, doc):
        ''' put all the above functions togther to complete the cleaning process '''
        # the received doc as dict
        # we will be using content and titles only

        doc = self.remove_punctuations(doc)
        doc = self.remove_double_spaces(doc)
        doc = doc.lower()
        doc = doc.strip()
        return doc