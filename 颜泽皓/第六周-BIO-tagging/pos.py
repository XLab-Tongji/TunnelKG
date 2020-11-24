"""Part of Speech

This module provides PoS, enumuration Part of Speech.
"""
from enum import Enum, unique, auto


@unique
class PoS(Enum):
    """enumuration Part of Speech 

    This set of PoS is a subset of which defined by jieba.
    """
    UKN = 0  # unknown
    a = auto()
    ad = auto()
    an = auto()
    b = auto()
    d = auto()
    f = auto()
    i = auto()
    j = auto()
    l = auto()
    m = auto()
    mq = auto()
    n = auto()
    ng = auto()
    nr = auto()
    nrt = auto()
    ns = auto()
    nt = auto()
    nz = auto()
    q = auto()
    r = auto()
    s = auto()
    t = auto()
    v = auto()
    vn = auto()
    x = auto()
    z = auto()
