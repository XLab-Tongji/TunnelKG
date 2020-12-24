import json
from urllib.parse import quote

import requests
from gensim.models import KeyedVectors

NER_URL = "http://10.60.38.173:12001/chineseNER/"
RE_URL = "http://10.60.38.173:12002/re/"
WORD_VEC_MODEL = KeyedVectors.load("./data/w2v_model.model")
# WORD_VEC_MODEL = KeyedVectors.load_word2vec_format("./data/added_w2v_model.bin")


def most_unrelated_entity(input: str, entity_entries: list[dict]) -> str:
    entries = []
    for entry in entity_entries:
        entries.append(input[entry["range"][0]:entry["range"][1]])
    try:
        return WORD_VEC_MODEL.doesnt_match(entries)
    except ValueError:
        return ""


def ner_interface(input: str) -> list[dict]:
    entity_entries = None

    try:
        r = requests.get(str().join([NER_URL, quote(input)]))
        data = r.json()
    except requests.RequestException:
        data = []
    entity_entries = [
        {"range": (entry["start"], entry["stop"]),
            "type": entry["type"]}
        for entry in data
    ]
    entity_entries.sort(key=lambda entry: entry["range"][0])
    return entity_entries


def process_sentence(input: str) -> dict:
    mue = ""
    entity_entries = ner_interface(input)
    if entity_entries:
        mue = most_unrelated_entity(input, entity_entries)
    return {"entities": entity_entries, "MUE": mue}


def related_words(entity: str) -> list[tuple[str, float]]:
    try:
        related_words: list[tuple[str, float]
                            ] = WORD_VEC_MODEL.most_similar(entity)
        for i in range(len(related_words)):
            related_words[i] = (related_words[i][0],
                                round(related_words[i][1], 4))
        return related_words
    except Exception:
        return []


def ralations(
    entries: list[tuple[int, int]],
    chosen_index: int,
    sentence: str
) -> list[str]:
    send_data = {
        "sentence": sentence,
        "entities": entries,
        "chosen_entity": chosen_index
    }
    try:
        r = requests.post(RE_URL, json=send_data)
        data = r.json()
    except requests.RequestException:
        data = [""] * len(entries)
    return data


def process_entity(
    entries: list[tuple[int, int]],
    chosen_index: int,
    sentence: str
) -> dict[str, list[str]]:
    chosen_entity = sentence[entries[chosen_index][0]:
                             entries[chosen_index][1]]
    return {"chosen_index": chosen_index,
            "related_words": related_words(chosen_entity),
            "ralations": ralations(entries, chosen_index, sentence)}
