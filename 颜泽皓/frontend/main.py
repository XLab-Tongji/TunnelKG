from typing import Any, Text
from flask import Flask, request, jsonify
from ProcessSentence import process_sentence, process_entity

app = Flask(__name__)


@app.route('/')
def index():
    return app.send_static_file("index.html")


@app.route('/sentence/', methods=['POST'])
def sentence() -> Text:
    input_text: str = str(request.get_data(), encoding="utf-8")

    return jsonify(process_sentence(input_text))


@app.route('/entity/', methods=['POST'])
def entity() -> Any:
    data = request.get_json()
    entries: list[tuple[int, int]] = data["entities"]
    chosen_index: int = data["chosen_index"]
    sentence: str = data["sentence"]

    return jsonify(process_entity(entries, chosen_index, sentence))


if __name__ == '__main__':
    app.run()
