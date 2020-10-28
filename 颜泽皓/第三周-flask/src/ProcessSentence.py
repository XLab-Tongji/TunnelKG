# typedef
import RetTypes as RetTy


def ProcessSentence(input: str) -> RetTy.AnlysisResult:
	words: list[RetTy.Word] = list()
	relations: list[RetTy.Relation] = list()

	# TODO: implemetation
	# test example begins
	words = [("使用", ""),
			 ("直径", ""),
			 ("足够", ""),
			 ("大", ""),
			 ("的", ""),
			 ("盾构", "机械"),
			 ("开凿", ""),
			 ("隧道", "结构"),
			 ("。", ""),
             (input, ""),
			 ("。", ""),]
	relations = [("建造", 7, 5),
				 ("建造", 7, 5),
				 ("建造", 7, 5),
				 ("建造", 7, 5),
				 ("建造", 7, 5),
				 ("建造", 7, 5),
				 ("建造", 7, 5),
				 ("建造", 7, 5),
				 ("建造", 7, 5),
				 ("建造", 7, 5),
				 ("建造", 7, 5),
				 ("建造", 7, 5)]
	# test example ends
	return (words, relations)
