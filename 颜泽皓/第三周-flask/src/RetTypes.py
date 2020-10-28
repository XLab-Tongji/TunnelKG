'''rettypes.py
This file defines alias of the return types.
'''


EntityType = str # entity category
RelationType = str  # entity category
Word = tuple[str, EntityType]
Relation = tuple[RelationType, int, int]
AnlysisResult = tuple[list[Word], list[Relation]]
