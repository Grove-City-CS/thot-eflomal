#pragma once

#include "nlp_common/SingleWordVocab.h"
#include "nlp_common/WordClasses.h"
#include "sw_models/AlignmentModel.h"
#include "sw_models/LightSentenceHandler.h"

#include <memory>
#include <set>
#include <yaml-cpp/yaml.h>

class AlignmentModelBase : public virtual AlignmentModel
{
public:
  /**
   * @brief Determines if it is safe to read from the model in parallel
   * 
   * The default implementation returns a value of true. Classes where this is not 
   * true must override this method and have it return false.
   * 
   * @return true if it it is safe or false if it is not
   */
  bool modelReadsAreProcessSafe() override;

  /**
   * @brief Set the value of the variational Bayes boolean for this Alignment Model
   * 
   * @details
   * The Variational Bayes boolean refers to whether the model should
   * use this technique during its expectation maximization steps.
   * 
   * @param variationalBayes the new value for this variable
   */
  void setVariationalBayes(bool variationalBayes) override;

  /**
   * @brief Get the Variational Bayes value for this Alignment Model
   * 
   * @details
   * The Variational Bayes boolean refers to whether the model should
   * use this technique during its expectation maximization steps.
   * 
   * @return true 
   * @return false 
   */
  bool getVariationalBayes() override;

  /**
   * @brief Read's matching sentence pairs in the source and target languages.
   * 
   * @details
   * The sentence pairs read by this method are added to the sentenceHandler as 
   * vectors of words using addSentencePair. The input file formats are designed 
   * to correspond to the three input files used by the GIZA++ system.
   * 
   * @see AlignmentModelBase::addSentencePair
   * @see AlignmentModelBase::sentenceHandler
   * 
   * @param srcFileName path to a file in the source language (newline and space delimited)
   * @param trgFileName path to a file in the target language (newline and space delimited)
   * @param sentCountsFile path to a newline delimited file of frequencies for how many times
   *                       the corresponding sentence pair occurs in the corpus.
   *                       @default: with "" 1 for every sentence pair
   * @param sentRange starting and ending indices of sentences to be read
   * @param verbose   how much additional output should be printed [0/1]
   * @return true if there was an error
   * @return false if there was no error
   */
  bool readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
                         std::pair<unsigned int, unsigned int>& sentRange, int verbose = 0) override;
  std::pair<unsigned int, unsigned int> addSentencePair(std::vector<std::string> srcSentStr,
                                                        std::vector<std::string> trgSentStr, Count c) override;
  unsigned int numSentencePairs() override;

  // NOTE: the whole valid range in a given moment is
  // [ 0 , numSentPairs() )
  int getSentencePair(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                      Count& c) override;

  PositionIndex getMaxSentenceLength() override;

  // Functions to print sentence pairs
  bool printSentencePairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile) override;

  // Returns log-likelihood. The first double contains the
  // loglikelihood for all sentences, and the second one, the same
  // loglikelihood normalized by the number of sentences
  std::pair<double, double> loglikelihoodForAllSentences(int verbosity = 0) override;

  // Scoring functions for a given alignment
  using AlignmentModel::computeLogProb;
  LgProb computeLogProb(const char* srcSentence, const char* trgSentence, const WordAlignmentMatrix& aligMatrix,
                        int verbose = 0) override;
  LgProb computeLogProb(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                        const WordAlignmentMatrix& aligMatrix, int verbose = 0) override;

  // Scoring functions without giving an alignment
  using AlignmentModel::computeSumLogProb;
  LgProb computeSumLogProb(const char* srcSentence, const char* trgSentence, int verbose = 0) override;
  LgProb computeSumLogProb(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                           int verbose = 0) override;
  LgProb computePhraseSumLogProb(const std::vector<WordIndex>& srcPhrase, const std::vector<WordIndex>& trgPhrase,
                                 int verbose = 0) override;

// Best-alignment functions
  /**
   * @brief Outputs a file with the word alignment in the GIZA format for each sentence pair in the src file and trg file 
   *
   * @see AlignmentModelBase::addSrcSymbol
   * @see AlignmentModelBase::printAligInGizaFormat
   *
   * @param sourceTestFileName path to a file in the source language (newline and space delimited)
   * @param targetTestFilename path to a file in the target language (newline and space delimited)
   * @param outFileName path to a file where GIZA alignment will be written
   * @return true if there was an error
   * @return false if there was no error
   */
  bool getBestAlignments(const char* sourceTestFileName, const char* targetTestFilename,
                         const char* outFileName) override;
  using AlignmentModel::getBestAlignment;
  // Obtains the best alignments for the sentence pairs given in
  // the files 'sourceTestFileName' and 'targetTestFilename'. The
  // results are stored in the file 'outFileName'
  
  /**
   * @brief Outputs an alignment log probability and the best word alignment matrix for a src and trg sentence 
   *
   * @details Converts source and target sentences (char*) to vectors of words (strings) and calls another
   * getBestAlignment method
   * 
   * @see StrProcUtils::charItemsToVector
   *
   * @param srcSentence A sentence of type string from the src corresponding to trgSentence
   * @param trgSentence A sentence of type string from the trg corresponding to srcSentence
   * @param bestWaMatrix A reference to a WordAlignmentMatrix where the word alignment matrix will be written
   * @return LgProb A log probability for the alignment calculated
   */
  LgProb getBestAlignment(const char* srcSentence, const char* trgSentence, WordAlignmentMatrix& bestWaMatrix) override;
  // Obtains the best alignment for the given sentence pair

  /**
   * @brief Outputs an alignment log probability and the best word alignment matrix for a src and trg sentence
   *
   * @details Converts source and target sentences from vectors of words (strings) to vectors of
   * WordIndex and then calls another getBestAlignment method
   * 
   * @see AlignmentModelBase::strVectorToSrcIndexVector
   * @see AlignmentModelBase::strVectorToTrgIndexVector
   *
   * @param srcSentence A sentence from the src represented as a vector of words (strings) that corresponds to trgSentence
   * @param trgSentence A sentence from the trg represented as a vector of words (strings) that corresponds to srcSentence
   * 
   * @param A reference to a WordAlignmentMatrix where the word alignment matrix will be written
   * @return LgProb A log probability for the alignment calculated
   */
  LgProb getBestAlignment(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                          WordAlignmentMatrix& bestWaMatrix) override;
  // Obtains the best alignment for the given sentence pair (input
  // parameters are now string vectors)

  /**
   * @brief Outputs an alignment log probability and the best word alignment matrix for a src and trg sentence
   *
   * @details calls AlignmentModel::getBestAlignment, which has a unique implementation in each alignment model
   *
   * @see WordAlignmentMatrix::init
   * @see WordAlignmentMatrix::putAligVec
   * @see AlignmentModel::getBestAlignment
   *
   * @param srcSentence A sentence from the src represented as a vector of WordIndex that corresponds to trgSentence
   * @param trgSentence A sentence from the trg represented as a vector of WordIndex that corresponds to srcSentence
   * @param bestWaMatrix A reference to a WordAlignmentMatrix where the word alignment matrix will be written
   * @return LgProb A log probability for the alignment calculated
   */
  LgProb getBestAlignment(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                          WordAlignmentMatrix& bestWaMatrix) override;

  /**
   * @brief Outputs an alignment log probability and the best alignment for a src and trg sentence
   *
   * @details Convert source and target sentences (char*) to vectors of words (strings) and call another
   * getBestAlignment() function
   *
   * @see StrProcUtils::charItemsToVector
   *
   * @param srcSentence A sentence of type string from the src corresponding to trgSentence
   * @param trgSentence A sentence of type string from the trg corresponding to srcSentence
   * @param bestAlignment A reference to a PositionIndex vector where the best alignment will be written
   * @return LgProb A log probability for the alignment calculated
   */
  LgProb getBestAlignment(const char* srcSentence, const char* trgSentence,
                          std::vector<PositionIndex>& bestAlignment) override;

  /**
   * @brief Outputs an alignment log probability and best alignment for a src and trg sentence
   *
   * @details Converts source and target sentences (char*) to vectors of words (strings) and calls another
   * getBestAlignment() method found in AlignmentModel.h
   *
   * @see StrProcUtils::strVectorToSrcIndexVector
   * @see StrProcUtils::strVectorToTrgIndexVector
   * @see AlignmentModel::getBestAlignment
   *
   * @param srcSentence A sentence from the src represented as a vector of words (strings) that corresponds to trgSentence
   * @param trgSentence A sentence from the trg represented as a vector of words (strings) that corresponds to srcSentence
   * @param bestAlignment A reference to a PositionIndex vector where the best alignment will be written
   * @return LgProb A log probability for the alignment calculated
   */
  LgProb getBestAlignment(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                          std::vector<PositionIndex>& bestAlignment) override;
  // Obtains the best alignment for the given sentence pair (input
  // parameters are now string vectors)

  /**
   * @brief Outputs a word alignment of these sentences in the GIZA format
   * 
   * @details
   * Specifically the alignment format described in part V section B.
   * @link https://github.com/moses-smt/giza-pp/blob/master/GIZA%2B%2B-v2/README
   * 
   * @param sourceSentence A space delimited string of word tokens in the source language
   * @param targetSentence A space delimited string of word tokens in the target language
   * @param p A Prob object (double) representing the alignment quality
   * @param alig A vector of indexes indicating alignment. The vector has 
   *             one item per token in the target sentence and the number stored 
   *             there is the index of the word from the source sentece that is 
   *             responsible for that word in the target sentence.
   * @param outS A reference to the output stream where the GIZA alignment will be written
   * @return a reference to the input parameter outS
   */
  std::ostream& printAligInGizaFormat(const char* sourceSentence, const char* targetSentence, Prob p,
                                      std::vector<PositionIndex> alig, std::ostream& outS) override;

  /**
   * @brief Load a GIZA vocabulary file for the source language into swVocab
   * 
   * @details
   * The vocabulary is stored in swVocab under the two maps:
   *   + srcWordIndexMapToString (IdxToStrVocab i.e. Map<WordIndex, String> i.e. unsigned int -> std::string)
   *   + stringToSrcWordIndexMap (StrToIdxVocab i.e. Map<String, WordIndex> i.e. std::string -> unsigned int)
   * 
   * @see AlignmentModelBase::swVocab (SingleWordVocab)
   * 
   * Specifically the vocabulary format described in part IV section A.
   * @link https://github.com/moses-smt/giza-pp/blob/master/GIZA%2B%2B-v2/README
   * 
   * @param srcInputVocabFileName a file in the GIZA vocabulary format
   * @param verbose how much additional output should be printed [0/1]
   * @return true if there was an error
   * @return false if there was no error
   */
  bool loadGIZASrcVocab(const char* srcInputVocabFileName, int verbose = 0) override;

  /**
   * @brief Load a GIZA vocabulary file for the target language into swVocab
   * 
   * @details
   * The vocabulary is stored in swVocab under the two maps:
   *   + trgWordIndexMapToString (IdxToStrVocab i.e. Map<WordIndex, String> i.e. unsigned int -> std::string)
   *   + stringToTrgWordIndexMap (StrToIdxVocab i.e. Map<String, WordIndex> i.e. std::string -> unsigned int)
   * 
   * @see AlignmentModelBase::swVocab (SingleWordVocab)
   * 
   * Specifically the vocabulary format described in part IV section A.
   * @link https://github.com/moses-smt/giza-pp/blob/master/GIZA%2B%2B-v2/README
   * 
   * @param trgInputVocabFileName a file in the GIZA vocabulary format
   * @param verbose how much additional output should be printed [0/1]
   * @return true if there was an error
   * @return false if there was no error
   */
  bool loadGIZATrgVocab(const char* trgInputVocabFileName, int verbose = 0) override;

  /**
   * @brief Writes the vocab in swVocab.stringToSrcWordIndexMap to the file in the GIZA vocabulary format
   * 
   * @see AlignmentModelBase::swVocab (SingleWordVocab)
   * 
   * @details
   * Specifically the vocabulary format described in part IV section A.
   * @link https://github.com/moses-smt/giza-pp/blob/master/GIZA%2B%2B-v2/README
   * 
   * @param srcOutputVocabFileName The file where the source vocabulary should be written
   * @return true if an error occurs
   * @return false if the operation is completed successfully
   */
  bool printGIZASrcVocab(const char* srcOutputVocabFileName) override;

  /**
   * @brief Writes the vocab in swVocab.stringToTrgWordIndexMap to the file in the GIZA vocabulary format
   * 
   * @details
   * Specifically the vocabulary format described in part IV section A.
   * @link https://github.com/moses-smt/giza-pp/blob/master/GIZA%2B%2B-v2/README
   * 
   * @param trgOutputVocabFileName The file where the target vocabulary should be written
   * @return true if an error occurs
   * @return false if the operation is completed successfully
   */
  bool printGIZATrgVocab(const char* trgOutputVocabFileName) override;

  /**
   * @brief Get the number of words in the source vocab
   * 
   * @details size of swVocab.stringToSrcWordIndexMap)
   * @see AlignmentModelBase::swVocab (SingleWordVocab)
   * 
   * @return size_t the number of words in the source vocabulary
   */
  size_t getSrcVocabSize() const override;

  /**
   * @brief Get the word index of a word in the source vocabulary
   * 
   * @see AlignmentModelBase::swVocab (SingleWordVocab)
   * 
   * @param s The word token from the source language
   * @return WordIndex that uniquely identifies that word (1 if not found)
   */
  WordIndex stringToSrcWordIndex(std::string s) const override;

  /**
   * @brief Get the word token (string) that goes with the given word index
   * 
   * @see AlignmentModelBase::swVocab (SingleWordVocab)
   * 
   * @param w The WordIndex (unsigned int) of a word in the source vocabulary
   * @return std::string of the word token from the source vocabulary or "UNKNOWN_WORD" if not found
   */
  std::string wordIndexToSrcString(WordIndex w) const override;

  /**
   * @brief Check if a given string is a word token in the source language
   * 
   * @see AlignmentModelBase::swVocab (SingleWordVocab)
   * 
   * @param s the string to be checked
   * @return true if the provided word token exists in the source language
   * @return false if the provided word token is not found
   */
  bool existSrcSymbol(std::string s) const override;

  /**
   * @brief Convert a vector of words (strings) from the source language to a vector of word indexes
   * 
   * @details
   * This conversion is done using swVocab.stringToSrcWordIndexMap.
   * If a word in the given vector is not already in the vocabulary it is added to the vocabulary.
   * @see AlignmentModelBase::addSrcSymbol
   * 
   * @param s a vector of word tokens (strings) in the source language
   * @return std::vector<WordIndex> 
   */
  std::vector<WordIndex> strVectorToSrcIndexVector(std::vector<std::string> s) override;

  /**
   * @brief Ensure a word token (string) is in the vocabulary of the source language
   * 
   * @param s The word to be included in the source language
   * @return WordIndex the new (or existing) word index for the given word
   */
  WordIndex addSrcSymbol(std::string s) override;

  /**
   * @brief Get the number of words in the target vocab
   * 
   * @details size of swVocab.stringToTrgWordIndexMap)
   * @see AlignmentModelBase::swVocab (SingleWordVocab)
   * 
   * @return size_t the number of words in the target vocabulary
   */
  size_t getTrgVocabSize() const override; // Returns the target vocabulary size

  /**
   * @brief Convert the given word (string) to its Word Index in the target vocabulary
   * 
   * @param t The word token (string) from the target vocabulary
   * @return WordIndex (unsigned int) that corresponds to that word (1 if not in vocab)
   */
  WordIndex stringToTrgWordIndex(std::string t) const override;

  /**
   * @brief Convert the given word index (unsigned int) to its corresponding word token (string) in the target vocab
   * 
   * @param w the word index (unsigned int) of the word to be retrieved
   * @return std::string the word corresponding to that index or "UNKNOWN_WORD" if not in the target vocabulary
   */
  std::string wordIndexToTrgString(WordIndex w) const override;

  /**
   * @brief Check if a given string is a word in the target language
   * 
   * @param t the word (string) to be checked
   * @return true if the word exists in the language
   * @return false if the word is not in the target vocabulary
   */
  bool existTrgSymbol(std::string t) const override;

  /**
   * @brief Convert a vector of words (strings) from the target language to a vector of word indexes
   * 
   * @details
   * This conversion is done using swVocab.stringToTrgWordIndexMap.
   * If a word in the given vector is not already in the vocabulary it is added to the vocabulary.
   * @see AlignmentModelBase::addTrgSymbol
   * 
   * @param t a vector of word tokens (strings) in the target language
   * @return std::vector<WordIndex> 
   */
  std::vector<WordIndex> strVectorToTrgIndexVector(std::vector<std::string> t) override;

  /**
   * @brief Ensure the given word token (string) is in the target vocabulary
   * 
   * @param t the word (token) to include in the target vocabulary
   * @return WordIndex the new (or existing) index of the given word
   */
  WordIndex addTrgSymbol(std::string t) override;

  /**
   * @brief Returns a copy of the given word index (unsigned int) vector with the NULL_WORD (0) prepended
   * 
   * @param vw The original word index (unsigned int) vector
   * @return std::vector<WordIndex> A copy of vw with a prepended WordIndex of 0 for NULL_WORD
   */
  std::vector<WordIndex> addNullWordToWidxVec(const std::vector<WordIndex>& vw) override;

  /**
   * @brief Returns a copy of the given word token (string) vector with the NULL_WORD_STR "NULL" prepended
   * 
   * @param vw The original word token (string) vector
   * @return std::vector<WordIndex> A copy of vw with a prepended word token of "NULL" for NULL_WORD_STR
   */
  std::vector<std::string> addNullWordToStrVec(const std::vector<std::string>& vw) override;

  WordClassIndex addSrcWordClass(const std::string& c) override;
  WordClassIndex addTrgWordClass(const std::string& c) override;
  void mapSrcWordToWordClass(WordIndex s, const std::string& c) override;
  void mapSrcWordToWordClass(WordIndex s, WordClassIndex c) override;
  void mapTrgWordToWordClass(WordIndex t, const std::string& c) override;
  void mapTrgWordToWordClass(WordIndex t, WordClassIndex c) override;

  bool load(const char* prefFileName, int verbose = 0) override;
  bool print(const char* prefFileName, int verbose = 0) override;

  void clear() override;
  // clear info about the whole sentence range without clearing
  // information about current model parameters
  void clearInfoAboutSentenceRange() override;

  virtual ~AlignmentModelBase()
  {
  }

protected:
  AlignmentModelBase();
  AlignmentModelBase(AlignmentModelBase& model);

  bool loadVariationalBayes(const std::string& filename);
  bool sentenceLengthIsOk(const std::vector<WordIndex> sentence);

  virtual std::string getModelTypeStr() const = 0;

  virtual void loadConfig(const YAML::Node& config);
  virtual bool loadOldConfig(const char* prefFileName, int verbose = 0);
  virtual void createConfig(YAML::Emitter& out);

  PositionIndex maxSentenceLength = 1024;
  double alpha;
  bool variationalBayes; /* whether to use Variational Bayes for EM */
  std::shared_ptr<SingleWordVocab> swVocab;
  std::shared_ptr<LightSentenceHandler> sentenceHandler;
  std::shared_ptr<WordClasses> wordClasses;
};
