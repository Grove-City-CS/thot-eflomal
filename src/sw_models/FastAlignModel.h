#ifndef _FastAlignModel_h
#define _FastAlignModel_h

#if HAVE_CONFIG_H
#include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "_swAligModel.h"
#include "ttables.h"
#include "BestLgProbForTrgWord.h"

class FastAlignModel : public _swAligModel<std::vector<Prob>>
{
public:
  typedef _swAligModel<std::vector<Prob>>::PpInfo PpInfo;

  void trainSentPairRange(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void trainAllSents(int verbosity = 0);
  void clearInfoAboutSentRange(void);

  LgProb obtainBestAlignment(std::vector<WordIndex> srcSentIndexVector, std::vector<WordIndex> trgSentIndexVector,
    WordAligMatrix& bestWaMatrix);

  Prob pts(WordIndex s, WordIndex t);
  LgProb logpts(WordIndex s, WordIndex t);

  Prob aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  LgProb logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);

  Prob sentLenProb(unsigned int slen, unsigned int tlen);
  LgProb sentLenLgProb(unsigned int slen, unsigned int tlen);

  LgProb calcLgProb(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent, int verbose = 0);
  LgProb calcLgProbForAlig(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
    WordAligMatrix aligMatrix, int verbose = 0);

  void initPpInfo(unsigned int slen, const std::vector<WordIndex>& tSent, PpInfo& ppInfo);
  void partialProbWithoutLen(unsigned int srcPartialLen, unsigned int slen, const std::vector<WordIndex>& s_,
    const std::vector<WordIndex>& tSent, PpInfo& ppInfo);
  LgProb lpFromPpInfo(const PpInfo& ppInfo);
  void addHeurForNotAddedWords(int numSrcWordsToBeAdded, const std::vector<WordIndex>& tSent, PpInfo& ppInfo);
  void sustHeurForNotAddedWords(int numSrcWordsToBeAdded, const std::vector<WordIndex>& tSent, PpInfo& ppInfo);

  bool load(const char* prefFileName, int verbose = 0);
  bool print(const char* prefFileName, int verbose = 0);

  void clearSentLengthModel(void);
  void clearTempVars(void);
  void clear(void);

  bool variationalBayes = true;
  double probAlignNull = 0.08;
  double alpha = 0.01;

private:
  const std::size_t ThreadBufferSize = 10000;

  void initialPass(std::pair<unsigned int, unsigned int> sentPairRange);
  void addTranslationOptions(std::vector<std::vector<WordIndex>>& insertBuffer);
  void updateFromPairs(const SentPairCont& pairs);
  Sentence getSrcSent(unsigned int n);
  Sentence getTrgSent(unsigned int n);
  double computeAZ(PositionIndex j, PositionIndex slen, PositionIndex tlen);
  Prob aProb(double az, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  LgProb lgProbOfBestTransForTrgWord(WordIndex t);
  void printParams(const std::string& filename);
  void loadParams(const std::string& filename);

  TTable s2t;
  double diagonalTension = 4.0;
  double meanSrcLenMultipler = 1.0;

  double empFeat = 0;
  double nTrgTokens = 0;
  int iter = 0;
  std::vector<std::pair<std::pair<short, short>, unsigned>> sizeCounts;

  BestLgProbForTrgWord bestLgProbForTrgWord;
};

#endif