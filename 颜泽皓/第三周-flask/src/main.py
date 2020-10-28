from flask import Flask, request, render_template
from ProcessSentence import *
import RetTypes as RetTy

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process/')
def search():
    input: str = request.args.get('input')
    (entities, relations) = ProcessSentence(input)
    return render_template('index.html',
                           input=input,
                           entities=entities,
                           relations=relations
                           )

if __name__ == '__main__':
   app.run()
