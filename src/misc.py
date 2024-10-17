import json, os

def log(text, dir = 'log.txt'):
    with open(dir, 'a+') as f:
        f.write(text)