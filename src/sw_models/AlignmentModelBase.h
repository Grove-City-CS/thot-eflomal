#pragma once

#include "nlp_common/SingleWordVocab.h"
#include "nlp_common/WordClasses.h"
#include "sw_models/AlignmentModel.h"
#include "sw_models/LightSentenceHandler.h"

#include <memory>
#include <set>
#include <yaml-cpp/yaml.h>

/// @brief 
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
   * @brief Adds matching sentence pairs in the source and target languages to
   *        this object's collection, replacing any previous sentence pairs.
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
   * @param[out] sentRange starting and ending indices of sentences to be read
   * @param verbose   how much additional output should be printed [0/1]
   * @return true if there was an error
   * @return false if there was no error
   */
  bool readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
                         std::pair<unsigned int, unsigned int>& sentRange, int verbose = 0) override;
  
  
  
  /**
   *  @brief Add an in-memory sentence to this object's collection
   *  @param srcSentStr Sentence in source language; each element is a token
   *  @param trgSentStr Sentence in target language; each element is a token
   *  @param c Frequency count (should be 1 unless the sentence pair is repeated)
   *  @param verbose 1 for verbose output; 0 for normal operation
   *  @return (i, i) where i is the index of the new sentence pair in this object's collection
   */
  std::pair<unsigned int, unsigned int> addSentencePair(std::vector<std::string> srcSentStr,
                                                        std::vector<std::string> trgSentStr, Count c) override;
  
  /**
   * @brief The number of sentence pairs in this object's collection, including in-file and in-memory
   * @return The number of sentence pairs in this object's collection
   */ 
  unsigned int numSentencePairs() override;

  /**
   * @brief Get one sentence pair and count
   * 
   *  NOTE: For efficient access of multiple sentences, retrieve in order of increasing index.
   * 
   *  @param n Index of the sentence pair. Must be in [0, numSentencePairs() - 1]. Indices for in-file sentences precede in-memory sentences.
   *  @param[out] srcSentStr Filled with the sentence in the sourse language, one element per token
   *  @param[out] trgSentStr Filled with the sentence in the target language, one element per token
   *  @param[out] c Filled with the frequency count for the sentence pair
   *  @return THOT_OK upon success or THOT_ERROR upon error
   */
  int getSentencePair(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                      Count& c) override;


  /**
   *  @brief Constant for max supported sentence length
   *  @return Constant for max supported sentence length
   */
  PositionIndex getMaxSentenceLength() override;


  /**
   *  @brief Print all sentence pairs and frequency counts to files
   *  @param srcSentFile Path to a file to create, writing all source sentences there, one per line, with space-delimited tokens
   *  @param trgSentFile Path to a file to create, writing all target sentences there, one per line, with space-delimited tokens
   *  @param sentCountsFile Path to a file to create, writing all frequency counts there, one per line
   *  @return THOT_OK upon success or THOT_ERROR upon error
   */
  bool printSentencePairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile) override;

  // Returns log-likelihood. The first double contains the
  // loglikelihood for all sentences, and the second one, the same
  // loglikelihood normalized by the number of sentences
  std::pair<double, double> loglikelihoodForAllSentences(int verbosity = 0) override;

  // Scoring functions for a given alignment
  using AlignmentModel::computeLogProb;

  /**
  * @brief Read a pointer to a source sentence and a target sentence, a pointer to a mapping of words in source
  * to words in target otherwise called an alignment
  * 
  * @details computing the probability of having a given alignment
  * 
  * @param srcSentence pointer to the place in memory containing the source sentences
  * @param trgSentence pointer to the place in memory containing the target sentences
  * @param WordAlignmentMatrix pointer to the alignment info (probably a vector)
  * @param verbose 0 add text else not
  * 
  * @return the probability of having a given alignment
  */
  LgProb computeLogProb(const char* srcSentence, const char* trgSentence, const WordAlignmentMatrix& aligMatrix,
                        int verbose = 0) override;
  

  /**
  * @brief Read a pointer to a source sentence and a target sentence, a pointer to a mapping of words in source
  * to words in target otherwise called an alignment
  * 
  * @details computing the probability of having a given alignment
  * @param srcSentence pointer to the vector of strings holding the source sentence
  * @param trgSentence pointer to the vector of strings holding the target sentence
  * @param WordAlignmentMatrix pointer to the alignment info (probably a vector)
  * @param verbose 0 add text else not
  * 
  * @return the probability of having a given alignment
  */
  LgProb computeLogProb(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                        const WordAlignmentMatrix& aligMatrix, int verbose = 0) override;

  // Scoring functions without giving an alignment
  using AlignmentModel::computeSumLogProb;



  /**
  * @brief Read a pointer to the source sentences, the target sentences and compute the probability 
  * of having the two senteces matching independently of the alignment
  * 
  * @details computing the probability of having a match between two sentences
  * independently of the alignment
  * 
  * @param srcSentence pointer to the address holding the source sentence
  * @param trgSentence pointer to the address holding the target sentence
  * @param verbose 0 add text else not
  * 
  * @return the probability of having a match between the source and target sentences
  * independently of the alignment
  */
  LgProb computeSumLogProb(const char* srcSentence, const char* trgSentence, int verbose = 0) override;

  /**
  * @brief Read a pointer to the source sentences, the target sentences and compute the probability 
  * of having the two senteces matching independently of the alignment
  * 
  * @details computing the probability of having a match between two sentences
  * independently of the alignment
  * 
  * @param srcSentence an adress to a vector of strings holding the source sentence
  * @param trgSentence an adress to a vector of strings holding the target sentence
  * @param verbose 0 add text else not
  * 
  * @return the probability of having a match between the source and target sentences 
  * independently of the alignment
  */
  LgProb computeSumLogProb(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                           int verbose = 0) override;

    /**
  * @brief Read a pointer to the source sentences, the target sentences and compute the probability 
  * of having the two senteces matching independently of the alignment
  * 
  * @details apparently there is a difference between sentence and phrase. But it is 
  * essentially the same thing as computeSumLogProb()
  * 
  * @param srcSentence a pointer to a vector of strings to the address holding the source sentence
  * @param trgSentence a pointer to a vector of strings to the address holding the target sentence
  * @param verbose 0 add text else not
  * 
  * @return the probability of having a match between the source and target phrases 
  * independently of the alignment
  */
  LgProb computePhraseSumLogProb(const std::vector<WordIndex>& srcPhrase, const std::vector<WordIndex>& trgPhrase,
                                 int verbose = 0) override;

  // Best-alignment functions
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
