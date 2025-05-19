import string
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # rest of your code
    return text

