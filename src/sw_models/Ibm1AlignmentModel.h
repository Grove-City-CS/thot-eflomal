#pragma once

#include "sw_models/AlignmentModelBase.h"
#include "sw_models/IncrAlignmentModel.h"
#include "sw_models/LexCounts.h"
#include "sw_models/LexTable.h"
#include "sw_models/NormalSentenceLengthModel.h"
#include "sw_models/anjiMatrix.h"

#include <memory>
#include <unordered_map>

class Ibm1AlignmentModel : public AlignmentModelBase
{
  friend class IncrIbm1AlignmentTrainer;

public:
  Ibm1AlignmentModel();

  void startTraining(int verbosity = 0) override;
  void train(int verbosity = 0) override;
  void endTraining() override;

  // Returns log-likelihood. The first double contains the
  // loglikelihood for all sentences, and the second one, the same
  // loglikelihood normalized by the number of sentences
  std::pair<double, double> loglikelihoodForPairRange(std::pair<unsigned int, unsigned int> sentPairRange,
                                                      int verbosity = 0) override;

  // returns p(t|s)
  Prob pts(WordIndex s, WordIndex t) override;
  // returns log(p(t|s))
  LgProb logpts(WordIndex s, WordIndex t) override;

  // alignment model functions
  Prob aProbIbm1(PositionIndex slen, PositionIndex tlen);
  LgProb logaProbIbm1(PositionIndex slen, PositionIndex tlen);

  // Sentence length model functions
  Prob getSentenceLengthProb(PositionIndex slen, PositionIndex tlen) override;
  LgProb getSentenceLengthLgProb(PositionIndex slen, PositionIndex tlen) override;

  bool getEntriesForSource(WordIndex s, NbestTableNode<WordIndex>& trgtn) override;

  using AlignmentModel::getBestAlignment;
  LgProb getBestAlignment(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                          std::vector<PositionIndex>& bestAlignment) override;
  using AlignmentModel::getAlignmentLgProb;
  LgProb getAlignmentLgProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                            const WordAlignmentMatrix& aligMatrix, int verbose = 0) override;
  using AlignmentModel::getSumLgProb;
  LgProb getSumLgProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                      int verbose = 0) override;

  bool load(const char* prefFileName, int verbose = 0) override;
  bool print(const char* prefFileName, int verbose = 0) override;

  void clear() override;
  void clearTempVars() override;
  void clearSentenceLengthModel() override;
  // clear info about the whole sentence range without clearing
  // information about current model parameters
  void clearInfoAboutSentenceRange() override;

  virtual ~Ibm1AlignmentModel()
  {
  }

protected:
  const std::size_t ThreadBufferSize = 10000;

  Ibm1AlignmentModel(Ibm1AlignmentModel& model);

  std::vector<WordIndex> getSrcSent(unsigned int n);
  std::vector<WordIndex> getTrgSent(unsigned int n);

  // given a vector with source words, returns a extended vector including extra NULL words
  virtual std::vector<WordIndex> extendWithNullWord(const std::vector<WordIndex>& srcWordIndexVec);

  virtual bool sentenceLengthIsOk(const std::vector<WordIndex> sentence);

  double unsmoothed_pts(WordIndex s, WordIndex t);
  double unsmoothed_logpts(WordIndex s, WordIndex t);

  LgProb getIbm1BestAlignment(const std::vector<WordIndex>& nSrcSentIndexVector,
                              const std::vector<WordIndex>& trgSentIndexVector, std::vector<PositionIndex>& bestAlig);
  LgProb getIbm1AlignmentLgProb(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent,
                                const std::vector<PositionIndex>& alig, int verbose = 0);
  LgProb getIbm1SumLgProb(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent, int verbose = 0);

  // Batch EM functions
  virtual void initSentencePair(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg);
  virtual void initSourceWord(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex i);
  virtual void initTargetWord(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex j);
  virtual void initWordPair(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex i,
                            PositionIndex j);
  virtual void addTranslationOptions(std::vector<std::vector<WordIndex>>& insertBuffer);
  virtual void batchUpdateCounts(const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs);
  virtual double getCountNumerator(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                   PositionIndex i, PositionIndex j);
  virtual void incrementWordPairCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                       PositionIndex i, PositionIndex j, double count);
  virtual void batchMaximizeProbs();

  std::string lexNumDenFileExtension = ".ibm_lexnd";

  // model parameters
  std::shared_ptr<NormalSentenceLengthModel> sentLengthModel;
  std::shared_ptr<LexTable> lexTable;

  // EM counts
  LexCounts lexCounts;
};
