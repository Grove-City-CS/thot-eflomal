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
  // Thread/Process safety related functions
  bool modelReadsAreProcessSafe() override;

  void setVariationalBayes(bool variationalBayes) override;
  bool getVariationalBayes() override;

  // Functions to read and add sentence pairs
/**
 * @brief Read matching sentence pairs in the source and target languages and adds them 
 * to the AlignmentModelBase::sentenceHandler using addSentencePair
 * 
 * @details
 * Alignment 
 * @param srcFileName path to a file in the ??? gformat storing the  sentences in the source langiahe 
 * @param trgFileName path to a file in the ??? format storing sentences in the target language
 * @param sentCountsFile path to a file in the  ??? formt with ??? information
 * @param sentRange starting and ending indiced of sentences to be read
 * @param verbose how much additional output should be printed [0/1]
 * @return true ?if the operation was successful
 * @return false ?otherwise
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
  LgProb getBestAlignment(const char* srcSentence, const char* trgSentence, WordAlignmentMatrix& bestWaMatrix) override;
  // Obtains the best alignment for the given sentence pair
  LgProb getBestAlignment(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                          WordAlignmentMatrix& bestWaMatrix) override;
  // Obtains the best alignment for the given sentence pair (input
  // parameters are now string vectors)
  LgProb getBestAlignment(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                          WordAlignmentMatrix& bestWaMatrix) override;
  LgProb getBestAlignment(const char* srcSentence, const char* trgSentence,
                          std::vector<PositionIndex>& bestAlignment) override;
  LgProb getBestAlignment(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                          std::vector<PositionIndex>& bestAlignment) override;
  // Obtains the best alignment for the given sentence pair (input
  // parameters are now string vectors)

  std::ostream& printAligInGizaFormat(const char* sourceSentence, const char* targetSentence, Prob p,
                                      std::vector<PositionIndex> alig, std::ostream& outS) override;
  // Prints the given alignment to 'outS' stream in GIZA format

  // Functions for loading vocabularies
  bool loadGIZASrcVocab(const char* srcInputVocabFileName, int verbose = 0) override;
  // Reads source vocabulary from a file in GIZA format
  bool loadGIZATrgVocab(const char* trgInputVocabFileName, int verbose = 0) override;
  // Reads target vocabulary from a file in GIZA format

  // Functions for printing vocabularies
  bool printGIZASrcVocab(const char* srcOutputVocabFileName) override;
  // Reads source vocabulary from a file in GIZA format
  bool printGIZATrgVocab(const char* trgOutputVocabFileName) override;
  // Reads target vocabulary from a file in GIZA format

  // Source and target vocabulary functions
  size_t getSrcVocabSize() const override; // Returns the source vocabulary size
  WordIndex stringToSrcWordIndex(std::string s) const override;
  std::string wordIndexToSrcString(WordIndex w) const override;
  bool existSrcSymbol(std::string s) const override;
  std::vector<WordIndex> strVectorToSrcIndexVector(std::vector<std::string> s) override;
  WordIndex addSrcSymbol(std::string s) override;

  size_t getTrgVocabSize() const override; // Returns the target vocabulary size
  WordIndex stringToTrgWordIndex(std::string t) const override;
  std::string wordIndexToTrgString(WordIndex w) const override;
  bool existTrgSymbol(std::string t) const override;
  std::vector<WordIndex> strVectorToTrgIndexVector(std::vector<std::string> t) override;
  WordIndex addTrgSymbol(std::string t) override;

  // Utilities
  std::vector<WordIndex> addNullWordToWidxVec(const std::vector<WordIndex>& vw) override;
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
  bool variationalBayes;
  std::shared_ptr<SingleWordVocab> swVocab;
  std::shared_ptr<LightSentenceHandler> sentenceHandler;
  std::shared_ptr<WordClasses> wordClasses;
};
