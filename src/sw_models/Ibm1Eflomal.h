#include "sw_models/Ibm1AlignmentModel.h"

#include "nlp_common/ErrorDefs.h"
#include "sw_models/Md.h"
#include "sw_models/MemoryLexTable.h"
#include "sw_models/SwDefs.h"

#include <algorithm>

using namespace std;

#pragma once

#include "sw_models/AlignmentModelBase.h"
#include "sw_models/IncrAlignmentModel.h"
#include "sw_models/LexCounts.h"
#include "sw_models/LexTable.h"
#include "sw_models/NormalSentenceLengthModel.h"
#include "sw_models/anjiMatrix.h"

#include <memory>
#include <unordered_map>

class Ibm1Eflomal: public Ibm1AlignmentModel {
  
public:
  Ibm1Eflomal();

protected:

  // batchUpdateCounts sets the LexCounts from the sentences and the LexTable.
  virtual void batchUpdateCounts(const vector<pair<vector<WordIndex>, vector<WordIndex>>>& pairs) override;

  using Ibm1AlignmentModel::addTranslationOptions;
  virtual void addTranslationOptions(vector<vector<WordIndex>>& insertBuffer) override;

  using Ibm1AlignmentModel::batchMaximizeProbs;
  virtual void batchMaximizeProbs();

  using Ibm1AlignmentModel::clearTempVars;
  virtual void clearTempVars() override;

  using Ibm1AlignmentModel::initSentencePair;
  virtual void initSentencePair(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg) override;

    /*
   * There is one links array per src-tgt sentence pair
   *
   * Each links array is of the same length as the target sentence
   *
   * Let i be a sentence pair index in the translation corpus
   * Let j be the index of a word in the target sentence of that pair
   * links[i][j] is the index of the word in the source sentece of that pair
   *             which is responsible for the presence of word j
   *
   */
  std::vector<std::vector<PositionIndex>> links;
  // clear vector in temp vars
  // inside init sentence pair, append random alignment

  /*
   * There is one entry for every non-zero mapping from a source word to its
   * consequential target word. All missing entries are assumed to be 0.
   *
   * An entry of 57 at index Pair(66, 42) means that the word represented by
   * the word token 66 is attributed to the word represented by the word token
   * 42 a total of 57 times accross the corpus.
   *
   */
  std::map<std::pair<WordIndex, WordIndex>, int> counts;
  // TODO: int data type?

  /*
   * Dirichlet represents the current estimates of the dirichlet priors in the model
   *
   * dirichlet[t] gives the estimated distribution of source word tokens
   *              for the target word token t.
   *
   * A value of dirichlet[t][s] is the probability that s is the cause of t
   *
   * If dirichlet[t][s] is not present because s is not in the map, the value of
   * this entry is assumed to be some small base probability (eflomal: LEX_ALPHA)
   *
   */
  std::vector<std::map<WordIndex, float>> dirichlet;
  // TODO: initialize dirichlet to size of vocab???

  /*
   * priors[t] is a map from source word tokens to probabilities
   *
   * priors[t][s] is the probability that t is caused by s
   *
   */
  std::vector<std::map<WordIndex, float>> priors;

private:
  // data structures needed for the
  const float NULL_ALPHA = 0.001;
  const float LEX_ALPHA = 0.001;
  const int NULL_LINK = 0xFFFF;
  const float NULL_PRIOR= 0.02;

};

