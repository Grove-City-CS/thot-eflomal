#pragma once

#include "nlp_common/AwkInputStream.h"
#include "sw_models/SentenceHandler.h"

#include <fstream>
#include <string.h>

/// @brief Collection of sentence pairs, in one matched pair of source/target files and/or in memory
class LightSentenceHandler : public SentenceHandler
{
public:
  // Constructor
  LightSentenceHandler();

  // Functions to read and add sentence pairs


  /// @brief Reset sentence handler and re-initialize with sentences from given files
  /// @param srcFileName Path to file with source language sentences, each on its own line, with whitespace delimited tokens
  /// @param trgFileName Path to file with target language sentences, each on its own line, with whitespace delimited tokens.
  ///                    Contents should correspond with those in @param srcFileName.
  /// @param sentCountsFile If not "", path to file with a frequency count on each line, corresponding to sentences in @param srcFileName.
  ///                       If "", default frequency of 1 for each sentence pair is used.
  /// @param[out] sentRange Filled in with (0, number of sentences in @param srcFileName - 1)
  /// @param verbose 1 for verbose output; 0 for normal operation
  /// @return THOT_OK upon success or THOT_ERROR upon error
  bool readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
                         std::pair<unsigned int, unsigned int>& sentRange, int verbose = 0) override;


  /// @brief Add an in-memory sentence to this collection
  /// @param srcSentStr Sentence in source language; each element is a token
  /// @param trgSentStr Sentence in target language; each element is a token
  /// @param c Frequency count (should be 1 unless the sentence pair is repeated)
  /// @param verbose 1 for verbose output; 0 for normal operation
  /// @return (i, i) where i is the index of the new sentence pair in this collection
  std::pair<unsigned int, unsigned int> addSentencePair(std::vector<std::string> srcSentStr,
                                                        std::vector<std::string> trgSentStr, Count c,
                                                        int verbose = 0) override;


  /// @brief The number of sentence pairs, including in-file and in-memory
  /// @return The number of sentence pairs in this collection
  unsigned int numSentencePairs() override;
  // NOTE: the whole valid range in a given moment is
  // [ 0 , numSentPairs() )
  

  /// @brief Get one sentence pair and count
  ///
  /// NOTE: For efficient access of multiple sentences, retrieve in order of increasing index.
  ///
  /// @param n Index of the sentence pair. Must be in [0, numSentencePairs() - 1]. Indices for in-file sentences precede in-memory sentences.
  /// @param[out] srcSentStr Filled with the sentence in the sourse language, one element per token
  /// @param[out] trgSentStr Filled with the sentence in the target language, one element per token
  /// @param[out] c Filled with the frequency count for the sentence pair
  /// @return THOT_OK upon success or THOT_ERROR upon error
  int getSentencePair(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                      Count& c) override;


  /// @brief Get one source sentence. For efficiency, getSentencePair is preferred.
  /// @param n Index of the source sentence
  /// @param[out] srcSentStr Filled with the sentence in the source language, one element per token
  /// @return THOT_OK upon success or THOT_ERROR upon error
  int getSrcSentence(unsigned int n, std::vector<std::string>& srcSentStr) override;

  /// @brief Get one target sentence. For efficiency, getSentencePair is preferred.
  /// @param n Index of the target sentence
  /// @param[out] trgSentStr Filled with the sentence in the target language, one element per token
  /// @return THOT_OK upon success or THOT_ERROR upon error
  int getTrgSentence(unsigned int n, std::vector<std::string>& trgSentStr) override;


  /// @brief Get one frequency count. For efficiency, getSentencePair is preferred.
  /// @param n Index of the sentence pair
  /// @param[out] c Filled with the frequency of the sentence pair 
  /// @return THOT_OK upon success or THOT_ERROR upon error
  int getCount(unsigned int n, Count& c) override;

  /// @brief Print all sentence pairs and frequency counts to files
  /// @param srcSentFile Path to a file to create, writing all source sentences there, one per line, with space-delimited tokens
  /// @param trgSentFile Path to a file to create, writing all target sentences there, one per line, with space-delimited tokens
  /// @param sentCountsFile Path to a file to create, writing all frequency counts there, one per line
  /// @return THOT_OK upon success or THOT_ERROR upon error
  bool printSentencePairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile) override;

  /// @brief Clear all sentences from this collection
  void clear() override;

protected:
  AwkInputStream awkSrc;
  AwkInputStream awkTrg;
  AwkInputStream awkSrcTrgC;

  bool countFileExists;

  // Sentence indices [0, nsPairsInFiles-1] are in file; [nsPairsInFiles, numSentencePairs()-1] are in memory
  size_t nsPairsInFiles;

  // A cursor to keep track of the line for the open input streams
  size_t currFileSentIdx;

  // Holds all the in-memory sentence pairs
  std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> sentPairCont;
  std::vector<Count> sentPairCount;

  void rewindFiles();
  bool getNextLineFromFiles();
  int nthSentPairFromFiles(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                           Count& c);
};
