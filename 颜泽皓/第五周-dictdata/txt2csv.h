#ifndef TXT_2_CSV_H
#define TXT_2_CSV_H

/**
 * @brief Word list to csv table.
 * Removes duplicates.
 * @param ipath input txt folder
 * @param filenames input txt base name
 * @param file_num input file number
 * @param opath output csv path
*/
void Txt2Csv(
	const char* ipath,
	const char* filenames[],
	size_t file_num,
	const char* opath
);

#endif // !TXT_2_CSV_H
