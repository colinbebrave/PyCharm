import string
def load_words(file_name):
    print 'Loading word list from file...'
    # inFile: file
    in_file = open(file_name, 'r', 0)
    # line: string
    line = in_file.readline()
    # word_list: list of strings
    word_list = line.split()
    print '  ', len(word_list), 'words loaded.'
    in_file.close()
    return word_list

def is_word(word_list, word):
    word = word.lower()
    word = word.strip(" !@#$%^&*()-_+={}[]|\:;'<>?,./\"")
    return word in word_list

def get_story_string():
    """
    Returns: a joke in encrypted text.
    """
    f = open("story.txt", "r")
    story = str(f.read())
    f.close()
    return story

WORDLIST_FILENAME = 'words.txt'

class Message(object):
    def __init__(self, text):
        self.message_text = text
        self.valid_words = load_words(WORDLIST_FILENAME)

    def get_message_text(self):
        return self.message_text

    def get_valid_words(self):
        return self.valid_words[:]
        
    def build_shift_dict(self, shift):
        lowerlist = string.ascii_lowercase
        upperlist = string.ascii_uppercase
        adict = {}
        for i in lowerlist:
            if lowerlist.index(i) + shift <= 25:
                adict[i] = lowerlist[lowerlist.index(i) + shift]
            elif lowerlist.index(i) + shift > 25:
                adict[i] = lowerlist[lowerlist.index(i) + shift - 25 - 1]
        for j in upperlist:
            if upperlist.index(j) + shift <= 25:
                adict[j] = upperlist[upperlist.index(j) + shift]
            elif upperlist.index(j) + shift > 25:
                adict[j] = upperlist[upperlist.index(j) + shift - 25 - 1]
        return adict

    def apply_shift(self, shift):
        letterlist = string.ascii_letters
        astring = ''
        for char in self.message_text:
            if char in letterlist:
                anotherstring = self.build_shift_dict(shift)[char]
                astring = astring + anotherstring
            else:
                astring = astring + char
        return astring

class PlaintextMessage(Message):
    def __init__(self, text, shift):
        Message.__init__(self,text)
        self.shift = shift
        self.encrypting_dict = Message.build_shift_dict(self,shift)
        self.message_text_encrypted = Message.apply_shift(self,shift)

    def get_shift(self):
        return self.shift

    def get_encrypting_dict(self):
        return self.encrypting_dict.copy()

    def get_message_text_encrypted(self):
        return self.message_text_encrypted

    def change_shift(self, shift):
        self.shift = shift
        self.encrypting_dict = Message.build_shift_dict(self,shift)
        self.message_text_encrypted = Message.apply_shift(self,shift)


class CiphertextMessage(Message):
    def __init__(self, text):
        Message.__init__(self,text)

    def decrypt_message(self):
        bestshift = 0
        bestcount = 0
        for i in range(26):
            aText = self.apply_shift(i)
            aList = aText.split(" ")
            count = 0
            for item in aList:
                if item in self.valid_words:
                    count += 1
            if count > bestcount:
                bestcount = count
                bestshift = i
        return bestshift,self.apply_shift(bestshift)

def decrypt_story():
    story = get_story_string()
    cipher = CiphertextMessage(story)
    return cipher.decrypt_message()