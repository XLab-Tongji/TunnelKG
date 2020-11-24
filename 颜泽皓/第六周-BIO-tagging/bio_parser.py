"""Part of Speech

This module provides parser of BIO tagged text.

    Typical usage example:

        my_list = BioParser.parse("input.txt")
"""
from pos import PoS


class BioParser(object):
    """BIO tagged text parser
    """
    @staticmethod
    def parse(ipath: str) -> list[tuple[str, PoS]]:
        """parse the BIO tagged text

        Memory overflow ignored.
        "O" tokens are not compressed.

        Args:
            ipath: path of BIO tagged text

        Returns:
            A list of tuples. Each contain a token string and its PoS.
            example:

            [("token0", PoS.ad),
             ("token1", PoS.UKN)]
        """
        token_list: list = []
        token: list[str]
        index = 0
        tag: list[str]
        bio: str
        prev_token: str

        with open(ipath, mode="r", encoding="utf-8") as ifile:
            token_list = ifile.read().strip().split(' ')
            while index < len(token_list):
                token = token_list[index].split('_')
                tag = token[1].split('-')
                bio = token[1][0]
                if bio == 'O':
                    token_list[index] = (token[0],
                                         PoS.UKN)
                    index += 1
                elif bio == 'I':
                    prev_token = token_list[index - 1]
                    token_list[index - 1] = (str().join([prev_token[0],
                                                         token[0]]),
                                             prev_token[1])
                    del token_list[index]
                elif bio == 'B':
                    token_list[index] = (token[0],
                                         PoS[tag[1]])
                    index += 1
        return token_list
