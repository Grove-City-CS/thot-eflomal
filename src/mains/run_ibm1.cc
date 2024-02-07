
#include <iostream>
#include <chrono>
#include <ctime>
#include <stdlib.h>

#include "sw_models/Ibm1AlignmentModel.h"

/// @brief Get the most likely translations (and their scores) for a src word
/// @param model used to make the translations
/// @param n number of translations (max) to return
/// @param srcWord word to translate
/// @param[out] bestTranslations up to n pairs of (targetWord, score) will be appended
void getMostLikelyTranslations(AlignmentModel& model, const std::string& srcWord,
    const int n, std::vector< std::pair<std::string, Score> > &bestTranslations) {
    
    NbestTableNode<WordIndex> results;
    model.getEntriesForSource(model.stringToSrcWordIndex(srcWord), results);
    
    int numAdded = 0;
    for (auto iter = results.begin(); iter != results.end() && numAdded < n; iter++ ) {

        std::pair<std::string, Score> res;
        res.second = iter->first;
        res.first = model.wordIndexToTrgString(iter->second);
        bestTranslations.push_back(res);
        numAdded++;
    }
}


int main(int argc, const char** argv) {
    if (argc != 6) {
        std::cerr << "USAGE: " << argv[0] << " [srcVocabFile] [trgVocabFile] [srcSentencesFile] [trgSentencesFile] [numIterations]\n";
        return -1;
    }

    const int verbosity = 2;

    const char* srcVocabFile = argv[1];
    const char* trgVocabFile = argv[2];
    const char* srcFile = argv[3];
    const char* trgFile = argv[4];
    const int numIterations = atoi(argv[5]);

    Ibm1AlignmentModel *model = new Ibm1AlignmentModel();

    model->loadGIZASrcVocab(srcVocabFile, (verbosity>1));
    if (verbosity > 0) {
        std::cout << "Loaded " << model->getSrcVocabSize() << " source vocabulary words\n";
    }

    model->loadGIZATrgVocab(trgVocabFile, (verbosity>1));
    if (verbosity > 0) {
        std::cout << "Loaded " << model->getTrgVocabSize() << " target vocabulary words\n";
    }
    
    std::pair<unsigned int, unsigned int> sentenceRange;
    bool wasError = model->readSentencePairs(srcFile, trgFile, "", sentenceRange, (verbosity>1));

    if (wasError) {
        std::cerr << "Error reading sentence pairs from src=" <<
            srcFile << " and tgt=" << trgFile << std::endl;
        return -2;
    }

    if (verbosity > 0) {
        std::cout << "Loaded sentence pairs " << sentenceRange.first << " through " << sentenceRange.second << std::endl;
    }

    if (verbosity > 0) {
        std::time_t the_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::ctime(&the_time_t);
        std::cout << " Beginning training " << numIterations << " iterations..." << std::endl;
    }

    model->startTraining((verbosity > 1));
    for (int iter = 1; iter <= numIterations; ++iter) {
        if (verbosity > 1 && iter % 1 == 0) {
            std::cout << "Starting training iteration " << iter << std::endl;
        }

        model->train();
    }
    model->endTraining();

    if (verbosity > 0) {
        std::time_t the_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::ctime(&the_time_t);
        std::cout << "Done training " << numIterations << " iterations." << std::endl;
    }

    std::string line;
    while (std::getline(std::cin, line) && line.size() > 0) {
        std::vector< std::pair<std::string, Score> > bestTranslations;

        getMostLikelyTranslations(*model, line, 5, bestTranslations);
        
        std::cout << line << ":";
        for (auto iter=bestTranslations.begin(); iter != bestTranslations.end(); iter++) {
            std::cout << iter->first << "," << iter->second << ",";
        }
        std::cout << std::endl;
    }

}