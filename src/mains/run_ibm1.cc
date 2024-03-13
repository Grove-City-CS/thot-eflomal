
#include <iostream>
#include <chrono>
#include <ctime>
#include <stdlib.h>

#include "sw_models/Ibm1AlignmentModel.h"
#include "sw_models/Ibm1Eflomal.h"

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

/// @brief Print to stdout the most likely translations (and their scores) for each word
/// @param model used to make the translations
/// @param srcWords words to translate
void printPredictions(AlignmentModel& model, const std::vector<std::string> &srcWords) {
    for (auto& word : srcWords) {
        std::vector< std::pair<std::string, Score> > bestTranslations;

        getMostLikelyTranslations(model, word, 5, bestTranslations);
        
        std::cout << word << ":";
        for (auto iter=bestTranslations.begin(); iter != bestTranslations.end(); iter++) {
            std::cout << iter->first << "," << iter->second << ",";
        }
        std::cout << std::endl;
    }
}


/// @brief Get words from a file, appending them to a vector
/// @param filename file name containing the words
/// @param[out] words Words from the file will be appended to this vector
void fillWordsFromFile(std::string filename, std::vector<std::string> &words) {
    std::fstream file;
    string word;
 
    // opening file
    file.open(filename.c_str());
 
    // extracting words from the file
    while (file >> word) {
        words.push_back(word);
    }
    file.close();
}

// Reads src words from stdin (one per line)
// Will print to stdout the most likely translations of those words
int main(int argc, const char** argv) {
    if (argc != 8) {
        std::cerr << "USAGE: " << argv[0] << " [useEflomal: 0 or 1] [srcVocabFile] [trgVocabFile] [srcSentencesFile] [trgSentencesFile] [numIterations] [srcTestWordsFile]\n";
        return -1;
    }

    const int verbosity = 2;

    int nextArg = 1;
    bool useEflomal = (strtol(argv[nextArg], NULL, 10) != 0);
    nextArg++;
    const char* srcVocabFile = argv[nextArg];
    nextArg++;
    const char* trgVocabFile = argv[nextArg];
    nextArg++;
    const char* srcFile = argv[nextArg];
    nextArg++;
    const char* trgFile = argv[nextArg];
    nextArg++;
    const int numIterations = atoi(argv[nextArg]);
    nextArg++;
    const char* testWordsFile = argv[nextArg];
    nextArg++;

    Ibm1AlignmentModel *model;
    if (useEflomal) {
        if (verbosity > 0) {
            std::cout << "Using EFLOMAL..." << std::endl;
        }
        model = new Ibm1Eflomal();
    }
    else {
        if (verbosity > 0) {
            std::cout << "Using original IBM model..." << std::endl;
        }
        model = new Ibm1AlignmentModel();
    }

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
        std::cout << "Beginning pre-training setup... " << std::endl;
    }

    model->startTraining((verbosity > 1));

    if (verbosity > 0) {
        std::time_t the_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << std::ctime(&the_time_t);
        std::cout << "Beginning training for " << numIterations << " iterations..." << std::endl;
    }
    
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

    {
        std::vector<std::string> words;

        fillWordsFromFile(testWordsFile, words);
        printPredictions(*model, words);
    }

}


