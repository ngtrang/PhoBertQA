import string
import re
import unicodedata

def normalize_text(text):
    text = text.replace("\n", "")
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\xa0', ' ') # space
    return text

def remove_multi_space(text):
    text = text.replace("\t", " ")
    text = re.sub("\s\s+", " ", text)
    # handle exception when line just all of punctuation
    if len(text) == 0:
        return text
    if text[0] == " ":
        text = text[1:]
    if len(text) == 0:
        pass
    else:
        if text[-1] == " ":
            text = text[:-1]

    return "".join(text)

def handle_text(text):
    text = normalize_text(text)
    text = remove_multi_space(text)
    return text

def handle_text_lower(text):
    text = normalize_text(text)
    text = remove_multi_space(text)
    text = text.lower()
    return text

if __name__ == "__main__":
    text = "xin Chào"
    print(handle_text_lower(text))
