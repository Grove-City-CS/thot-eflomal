/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez
 
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
*/
 
/********************************************************************/
/*                                                                  */
/* Module: PhrLocalSwLiTm                                           */
/*                                                                  */
/* Prototypes file: PhrLocalSwLiTm.h                                */
/*                                                                  */
/* Description: Declares the PhrLocalSwLiTm class                   */
/*              This class implements a statistical machine         */
/*              translation model which combines a phrase model     */
/*              and a local single word model via linear            */
/*              interpolation. Training of new samples is carried   */
/*              out using an interlaced training scheme.            */
/*                                                                  */
/********************************************************************/

/**
 * @file PhrLocalSwLiTm.h
 *
 * @brief Declares the PhrLocalSwLiTm class.  This class implements a
 * statistical machine translation model that combines a phrase model
 * with a local single word model via linear interpolation. Training of
 * new samples is carried out using an interlaced training scheme.
 */

#ifndef _PhrLocalSwLiTm_h
#define _PhrLocalSwLiTm_h

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

extern "C" {
#include "step_by_step_dhs.h"
}

#include "PhrHypNumcovJumps01EqClassF.h"
#include "_phrSwTransModel.h"
#include "PhrLocalSwLiTmHypRec.h"
#include <BaseStepwiseAligModel.h>
#include "_wbaIncrPhraseModel.h"
#include "BaseIncrPhraseModel.h"
#include <PhrasePair.h>
#include <PhraseExtractParameters.h>
#include "EditDistForVec.h"

//--------------- Constants ------------------------------------------

#define PHRSWLITM_LGPROB_SMOOTH         -9999999
#define PHRSWLITM_DEFAULT_LR            0.5
#define PHRSWLITM_DEFAULT_LR_ALPHA_PAR  0.75
#define PHRSWLITM_DEFAULT_LR_PAR1       0.99
#define PHRSWLITM_DEFAULT_LR_PAR2       0.75
#define PHRSWLITM_LR_RESID_WER          0.2
#define PHRSWLITM_DHS_FTOL              0.001
#define PHRSWLITM_DHS_SCALE_PAR         1

//--------------- typedefs -------------------------------------------

typedef PhrHypNumcovJumps01EqClassF HypEqClassF;

//--------------- Classes --------------------------------------------

//--------------- PhrLocalSwLiTm class

/**
 * @brief The PhrLocalSwLiTm class implements a statistical translation
 * model specialized for phrase-based translation that combines a phrase
 * model with a single word model via linear interpolation. Training of
 * new samples is carried out using an interlaced training scheme.
 */

class PhrLocalSwLiTm: public _phrSwTransModel<PhrLocalSwLiTmHypRec<HypEqClassF> >
{
 public:

  typedef _phrSwTransModel<PhrLocalSwLiTmHypRec<HypEqClassF> >::Hypothesis Hypothesis;
  typedef _phrSwTransModel<PhrLocalSwLiTmHypRec<HypEqClassF> >::HypScoreInfo HypScoreInfo;
  typedef _phrSwTransModel<PhrLocalSwLiTmHypRec<HypEqClassF> >::HypDataType HypDataType;

  // class functions

      // Constructor
  PhrLocalSwLiTm(void);

      // Virtual object copy
  BaseSmtModel<PhrLocalSwLiTmHypRec<HypEqClassF> >* clone(void);

      // Init alignment model
  bool loadAligModel(const char* prefixFileName);

      // Print models
  bool printAligModel(std::string printPrefix);

  void clear(void);

      // Functions to update linear interpolation weights
  int updateLinInterpWeights(std::string srcDevCorpusFileName,
                             std::string trgDevCorpusFileName,
                             int verbose=0);

  ////// Hypotheses-related functions

      // Misc. operations with hypothesis
  Hypothesis nullHypothesis(void);
  HypDataType nullHypothesisHypData(void);
  bool obtainPredecessorHypData(HypDataType& hypd);
  bool isCompleteHypData(const HypDataType& hypd)const;

      // Model weights functions
  void setWeights(Vector<float> wVec);
  void getWeights(Vector<pair<std::string,float> >& compWeights);
  unsigned int getNumWeights(void);
  void printWeights(ostream &outS);

      // Functions for performing on-line training
  void setOnlineTrainingPars(OnlineTrainingPars _onlineTrainingPars,
                             int verbose=0);
  int onlineTrainFeatsSentPair(const char *srcSent,
                               const char *refSent,
                               const char *sysSent,
                               int verbose=0);

      // Destructor
  ~PhrLocalSwLiTm();

 protected:

      // Training-related data members
  Vector<Vector<std::string> > vecSrcSent;
  Vector<Vector<std::string> > vecTrgSent;  
  Vector<Vector<std::string> > vecSysSent;  
  Vector<Vector<PhrasePair> > vecVecInvPhPair;
  unsigned int stepNum;

      // Functions related to linear interpolation weights updating
  int extractPhrPairsFromDevCorpus(std::string srcDevCorpusFileName,
                                   std::string trgDevCorpusFileName,
                                   Vector<Vector<PhrasePair> >& invPhrPairs,
                                   int verbose/*=0*/);
  double phraseModelPerplexity(const Vector<Vector<PhrasePair> >& invPhrPairs,
                               int verbose=0);
  int new_dhs_eval(const Vector<Vector<PhrasePair> >& invPhrPairs,
                   FILE* tmp_file,
                   double* x,
                   double& obj_func);

      // Function lo load and print lambda values
  bool load_lambdas(const char* lambdaFileName);
  bool print_lambdas(const char* lambdaFileName);
  ostream& print_lambdas(ostream &outS);

      // Misc. operations with hypothesis
  unsigned int
    numberOfUncoveredSrcWordsHypData(const HypDataType& hypd)const;

      // Scoring functions
  Score incrScore(const Hypothesis& prev_hyp,
                  const HypDataType& new_hypd,
                  Hypothesis& new_hyp,
                  Vector<Score>& scoreComponents);
      // Phrase model scoring functions
  Score smoothedPhrScore_s_t_(const Vector<WordIndex>& s_,
                              const Vector<WordIndex>& t_);
  Score smoothedPhrScore_t_s_(const Vector<WordIndex>& s_,
                              const Vector<WordIndex>& t_);

      // Functions to score n-best translations lists
  Score nbestTransScore(const Vector<WordIndex>& s_,
                        const Vector<WordIndex>& t_);
  Score nbestTransScoreLast(const Vector<WordIndex>& s_,
                            const Vector<WordIndex>& t_);

  PositionIndex getLastSrcPosCoveredHypData(const HypDataType& hypd);
      // Get the index of last source position which was covered

      // Functions for translating with references or prefixes
  bool hypDataTransIsPrefixOfTargetRef(const HypDataType& hypd,
                                       bool& equal)const;

      // Specific phrase-based functions
  void extendHypDataIdx(PositionIndex srcLeft,
                        PositionIndex srcRight,
                        const Vector<WordIndex>& trgPhraseIdx,
                        HypDataType& hypd);

      // Functions for performing on-line training
  int extractConsistentPhrasePairs(const Vector<std::string>& srcSentStrVec,
                                   const Vector<std::string>& refSentStrVec,
                                   Vector<PhrasePair>& vecInvPhPair,
                                   bool verbose=0);
  int incrTrainFeatsSentPair(const char *srcSent,
                             const char *refSent,
                             int verbose=0);
  int minibatchTrainFeatsSentPair(const char *srcSent,
                                  const char *refSent,
                                  const char *sysSent,
                                  int verbose=0);
  int batchRetrainFeatsSentPair(const char *srcSent,
                                const char *refSent,
                                int verbose=0);
  float calculateNewLearningRate(int verbose=0);
  float werBasedLearningRate(int verbose=0);
  unsigned int map_n_am_suff_stats(unsigned int n);
  int addNewTransOpts(unsigned int n,
                      int verbose=0);
};

#endif
