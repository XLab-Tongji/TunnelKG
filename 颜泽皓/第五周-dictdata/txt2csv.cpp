#define _CRT_SECURE_NO_WARNINGS
#define _HAS_AUTO_PTR_ETC 1
#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING

#include "txt2csv.h"

#include <locale>
#include <set>
#include <string>
#include <fstream>

#include "cppjieba/Jieba.hpp"

const char* const DICT_PATH = "D:/GitHub/cppjieba/dict/jieba.dict.utf8";
const char* const HMM_PATH = "D:/GitHub/cppjieba/dict/hmm_model.utf8";
const char* const USER_DICT_PATH = "D:/GitHub/cppjieba/dict/user.dict.utf8";
const char* const IDF_PATH = "D:/GitHub/cppjieba/dict/idf.utf8";
const char* const STOP_WORD_PATH = "D:/GitHub/cppjieba/dict/stop_words.utf8";

void Txt2Csv(
	const char* ipath,
	const char* filenames[],
	size_t file_num,
	const char* opath
)
{
	cppjieba::Jieba* pjieba = new cppjieba::Jieba(
		DICT_PATH,
		HMM_PATH,
		USER_DICT_PATH,
		IDF_PATH,
		STOP_WORD_PATH
	);
	std::ofstream ofstrm(opath);
	std::ifstream ifstrm;
	std::set<std::string>* pset_word = new std::set<std::string>();
	std::string word;
	std::locale loc;

	for (size_t ifile = 0; ifile < file_num; ifile++)
	{
		ifstrm.open(std::string(ipath) + filenames[ifile] + ".txt");
		while (std::getline(ifstrm, word))
		{
			if (pset_word->find(word) == pset_word->end())
			{
				ofstrm << word << ',' << std::flush;
				ofstrm << filenames[ifile] << ',' << std::flush;
				ofstrm << pjieba->LookupTag(word) << ',' << std::flush;
				ofstrm << std::endl;
				pset_word->insert(word);
			}
		}
		ifstrm.close();
	}
	ofstrm.close();
	delete pset_word;
	pset_word = nullptr;
	delete pjieba;
	pjieba = nullptr;
}
