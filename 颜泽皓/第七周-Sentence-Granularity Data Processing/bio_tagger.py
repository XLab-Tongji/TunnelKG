import csv
from typing import IO
from ahocorasick import Automaton, load as AutomatonLoad
from pos import PoS


class BioTagger(object):
    __POS_BYTE_LEN = 5
    __TRIE_ENTRY_BIT_LEN = 10
    __BEGIN_MARK = 1 << __POS_BYTE_LEN
    __POS_MASK = 31
    __BYTE_ORDER = "big"
    __DELIMS = ['。', '！', '．', '？']

    @staticmethod
    def __serializer(lenth: int, pos: PoS) -> bytes:
        entry: int = (lenth << BioTagger.__POS_BYTE_LEN) | pos.value

        return entry.to_bytes(BioTagger.__TRIE_ENTRY_BIT_LEN,
                              BioTagger.__BYTE_ORDER)

    @staticmethod
    def __deserializer(entry_bytes: bytes) -> tuple[int, PoS]:
        entry = int.from_bytes(entry_bytes,
                               BioTagger.__BYTE_ORDER)
        pos = PoS(entry & BioTagger.__POS_MASK)
        lenth = entry >> BioTagger.__POS_BYTE_LEN

        return (lenth, pos)

    @staticmethod
    def __write_tags(tags: list[int], text: str, opath: str, opath_text: str):
        tag_text: str
        bio_tag: str
        pos_text: str
        target_text: str
        is_entity: bool
        is_begin: bool
        ofile_text: IO = None

        with open(opath, encoding="utf-8", mode="w") as ofile:
            if opath_text:
                ofile_text = open(opath_text, encoding="utf-8", mode="w")
            for index in range(len(text)):
                is_entity = tags[index] != 0
                is_begin = tags[index] & BioTagger.__BEGIN_MARK
                if is_entity:
                    bio_tag = 'B' if is_begin else 'I'
                else:
                    bio_tag = 'O'
                pos_text = (PoS(tags[index] & BioTagger.__POS_MASK).name
                            if is_entity
                            else "")
                target_text = str().join([bio_tag,
                                          '-' if is_entity else "",
                                          pos_text])
                tag_text = text[index]
                if ofile_text:
                    tag_text = str().join([tag_text,
                                           ' '])
                else:
                    tag_text = str().join([tag_text,
                                           '_',
                                           target_text,
                                           ' '])
                if text[index] in BioTagger.__DELIMS:
                    tag_text = str().join([tag_text, '\n'])
                ofile.write(tag_text)
                if ofile_text:
                    target_text = str().join([target_text, ' '])
                    if text[index] in BioTagger.__DELIMS:
                        target_text = str().join([target_text, '\n'])
                    ofile_text.write(target_text)
            if ofile_text:
                ofile_text.close()

    def __init__(self, path: str = None):
        self.__trie: Automaton

        if path:
            self.__trie = AutomatonLoad(path, BioTagger.__deserializer)
        else:
            self.__trie = Automaton(value_type=PoS, key_type=str)

    def save(self, path: str) -> None:
        self.__trie.save(path, BioTagger.__serializer)

    def parse_csv(self, path: str) -> None:
        with open(path, encoding="utf_8_sig") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.__trie.add_word(row[0], (len(row[0]), PoS[row[1]]))

    def make_automaton(self) -> None:
        self.__trie.make_automaton()

    def tag(self, ipath: str, opath: str, opath_text: str = None) -> None:
        text: str
        tags: list[int]
        start_index: int

        with open(ipath, encoding="utf-8") as ifile:
            text = ifile.read()
            tags = [0] * len(text)
            result_list = list(self.__trie.iter(text))
            for end_index, (lenth, pos) in result_list:
                start_index = end_index - lenth + 1
                start_tag = tags[start_index]
                if not start_tag or start_tag & BioTagger.__BEGIN_MARK:
                    tags[start_index] = BioTagger.__BEGIN_MARK | pos.value
                    for index in range(start_index + 1, end_index + 1):
                        tags[index] = pos.value
            BioTagger.__write_tags(tags, text, opath, opath_text)
