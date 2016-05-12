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
/* Module: _phraseBasedTransModel                                   */
/*                                                                  */
/* Prototypes file: _phraseBasedTransModel.h                        */
/*                                                                  */
/* Description: Declares the _phraseBasedTransModel class.          */
/*              This class is a succesor of the BasePbTransModel    */
/*              class.                                              */
/*                                                                  */
/********************************************************************/

/**
 * @file _phraseBasedTransModel.h
 *
 * @brief Declares the _phraseBasedTransModel class.  This class is a
 * succesor of the BasePbTransModel class.
 */

#ifndef __phraseBasedTransModel_h
#define __phraseBasedTransModel_h

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "PhraseModelInfo.h"
#include "NgramCacheTable.h"
#include "LangModelInfo.h"
#include "Prob.h"
#include "SourceSegmentation.h"
#include "PhrNbestTransTable.h"
#include "PhrNbestTransTableRef.h"
#include "PhrNbestTransTablePref.h"
#include "PhraseCacheTable.h"
#include "PhrasePairCacheTable.h"
#include "ScoreCompDefs.h"
#include <math.h>
#include <set>

//--------------- Constants ------------------------------------------

#define NO_HEURISTIC            0
#define LOCAL_T_HEURISTIC       4
#define LOCAL_TD_HEURISTIC      6
#define MODEL_IDLE_STATE        1
#define MODEL_TRANS_STATE       2
#define MODEL_TRANSREF_STATE    3
#define MODEL_TRANSVER_STATE    4
#define MODEL_TRANSPREFIX_STATE 5

//--------------- Classes --------------------------------------------

//--------------- _phraseBasedTransModel class

/**
 * @brief The _phraseBasedTransModel class is a predecessor of the
 * BasePbTransModel class.
 */

template<class HYPOTHESIS>
class _phraseBasedTransModel: public BasePbTransModel<HYPOTHESIS>
{
 public:

  typedef typename BasePbTransModel<HYPOTHESIS>::Hypothesis Hypothesis;
  typedef typename BasePbTransModel<HYPOTHESIS>::HypScoreInfo HypScoreInfo;
  typedef typename BasePbTransModel<HYPOTHESIS>::HypDataType HypDataType;

  // class functions

  // Constructor
  _phraseBasedTransModel();
      
  // class methods

      // Init language and alignment models
  virtual bool loadLangModel(const char* prefixFileName);
  virtual bool loadAligModel(const char* prefixFileName);

      // Print models
  virtual bool printLangModel(std::string printPrefix);
  virtual bool printAligModel(std::string printPrefix);
  
  virtual void clear(void);

      // Actions to be executed before the translation
  void pre_trans_actions(std::string srcsent);
  void pre_trans_actions_ref(std::string srcsent,
                             std::string refsent);
  void pre_trans_actions_ver(std::string srcsent,
                             std::string refsent);
  void pre_trans_actions_prefix(std::string srcsent,
                                std::string prefix);

      // Word prediction functions
  void addSentenceToWordPred(Vector<std::string> strVec,
                             int verbose=0);
  pair<Count,std::string> getBestSuffix(std::string input);
  pair<Count,std::string> getBestSuffixGivenHist(Vector<std::string> hist,
                                                 std::string input);

  ////// Hypotheses-related functions
  
      // Heuristic-related functions
  void setHeuristic(unsigned int _heuristicId);
  void addHeuristicToHyp(Hypothesis& hyp);
  void sustractHeuristicToHyp(Hypothesis& hyp);

      // Printing functions and data conversion
  void printHyp(const Hypothesis& hyp,
                ostream &outS,
                int verbose=false);
  Vector<std::string> getTransInPlainTextVec(const Hypothesis& hyp, set<PositionIndex>& unknownWords)const;
      
      // Model weights functions
  Vector<Score> scoreCompsForHyp(const Hypothesis& hyp);
  
        // Specific phrase-based functions
  void extendHypData(PositionIndex srcLeft,
                     PositionIndex srcRight,
                     const Vector<std::string>& trgPhrase,
                     HypDataType& hypd);

      // Destructor
  ~_phraseBasedTransModel();

 protected:
      // Language model members
  LangModelInfo* langModelInfoPtr;
  NgramCacheTable cachedNgramScores;
  
      // Phrase model members
  PhraseModelInfo* phrModelInfoPtr;
  PhrasePairCacheTable cachedDirectPhrScores;
  PhrasePairCacheTable cachedInversePhrScores;
  

  // Data members related to caching n-best translation scores
  //
      // Cached n-best lm scores
  PhraseCacheTable cnbLmScores;
      
      // Cached translation table to store phrase N-best translations
  PhrNbestTransTable cPhrNbestTransTable; 
      // The same as cPhrNbestTransTable but to be used in assisted
      // translation
  PhrNbestTransTableRef cPhrNbestTransTableRef;
  PhrNbestTransTablePref cPhrNbestTransTablePref;

      // Cached n-best translations scores (these cached scores are
      // those generated by the nbestTransScore() and
      // nbestTransScoreLast() functions)
  PhrasePairCacheTable cnbestTransScore;
  PhrasePairCacheTable cnbestTransScoreLast;

  // End of caching data members for N-best translations
  
      // Set of unseen words
  std::set<std::string> unseenWordsSet;
      
      // Mapping between phrase and language model vocabularies
  map<WordIndex,ngramWordIndex> tmToLmVocMap;

      // Heuristic function to be used
  unsigned int heuristicId;
      // Heuristic probability vector
  Vector<Vector<Score> > heuristicScoreVec; 

  unsigned int state; // state information
    
      // Temporary data structure to store the source sentence during
      // each translation process
  Vector<std::string> srcSentVec;
  Vector<WordIndex> srcSentIdVec;
  Vector<WordIndex> nsrcSentIdVec;

      // Temporary data structure to store the reference sentence during
      // each translation process
  Vector<std::string> refSentVec;
  Vector<WordIndex> nrefSentIdVec;
  Vector<LgProb> refHeurLmLgProb;
    
      // Temporary data structure to store the prefix sentence during
      // each translation process
  bool lastCharOfPrefIsBlank;
  Vector<std::string> prefSentVec;
  Vector<WordIndex> nprefSentIdVec;
  Vector<LgProb> prefHeurLmLgProb;

      // Training-related data members
  Vector<Vector<std::string> > wordPredSentVec;

      // Protected functions

      // Word prediction functions
  void incrAddSentenceToWordPred(Vector<std::string> strVec,
                                 int verbose=0);
  void minibatchAddSentenceToWordPred(Vector<std::string> strVec,
                                      int verbose=0);
  void batchAddSentenceToWordPred(Vector<std::string> strVec,
                                  int verbose=0);

  ////// Hypotheses-related functions

      // Expansion-related functions

      // Specific phrase-based functions
  virtual void extendHypDataIdx(PositionIndex srcLeft,
                                PositionIndex srcRight,
                                const Vector<WordIndex>& trgPhraseIdx,
                                HypDataType& hypd)=0;

  bool getHypDataVecForGap(const Hypothesis& hyp,
                           PositionIndex srcLeft,
                           PositionIndex srcRight,
                           Vector<HypDataType>& hypDataTypeVec,
                           float N);
      // Get N-best translations for a subphrase of the source sentence
      // to be translated .  If N is between 0 and 1 then N represents a
      // threshold. 
  bool getHypDataVecForGapRef(const Hypothesis& hyp,
                              PositionIndex srcLeft,
                              PositionIndex srcRight,
                              Vector<HypDataType>& hypDataTypeVec,
                              float N);
      // This function is identical to the previous function but is to
      // be used when the translation process is conducted by a given
      // reference sentence
  virtual bool getHypDataVecForGapVer(const Hypothesis& hyp,
                                      PositionIndex srcLeft,
                                      PositionIndex srcRight,
                                      Vector<HypDataType>& hypDataTypeVec,
                                      float N);
      // This function is identical to the previous function but is to
      // be used when the translation process is performed to verify the
      // coverage of the model given a reference sentence
  bool getHypDataVecForGapPref(const Hypothesis& hyp,
                               PositionIndex srcLeft,
                               PositionIndex srcRight,
                               Vector<HypDataType>& hypDataTypeVec,
                               float N);
      // This function is identical to the previous function but is to
      // be used when the translation process is conducted by a given
      // prefix

  virtual bool getTransForHypUncovGap(const Hypothesis& hyp,
                                      PositionIndex srcLeft,
                                      PositionIndex srcRight,
                                      NbestTableNode<Vector<WordIndex> >& nbt,
                                      float N);
      // Get N-best translations for a subphrase of the source sentence
      // to be translated .  If N is between 0 and 1 then N represents a
      // threshold.  The result of the search is cached in the data
      // member cPhrNbestTransTable
  virtual bool getTransForHypUncovGapRef(const Hypothesis& hyp,
                                         PositionIndex srcLeft,
                                         PositionIndex srcRight,
                                         NbestTableNode<Vector<WordIndex> >& nbt,
                                         float N);
      // This function is identical to the previous function but is to
      // be used when the translation process is conducted by a given
      // reference sentence
  virtual bool getTransForHypUncovGapVer(const Hypothesis& hyp,
                                         PositionIndex srcLeft,
                                         PositionIndex srcRight,
                                         NbestTableNode<Vector<WordIndex> >& nbt,
                                         float N);
      // This function is identical to the previous function but is to
      // be used when the translation process is performed to verify the
      // coverage of the model given a reference sentence
  virtual bool getTransForHypUncovGapPref(const Hypothesis& hyp,
                                          PositionIndex srcLeft,
                                          PositionIndex srcRight,
                                          NbestTableNode<Vector<WordIndex> >& nbt,
                                          float N);
      // This function is identical to the previous function but is to
      // be used when the translation process is conducted by a given
      // prefix

      // Functions for translating with references or prefixes
  virtual bool hypDataTransIsPrefixOfTargetRef(const HypDataType& hypd,
                                               bool& equal)const=0;
  void transUncovGapPrefNoGen(const Hypothesis& hyp,
                              PositionIndex srcLeft,
                              PositionIndex srcRight,
                              NbestTableNode<Vector<WordIndex> >& nbt);
  void genListOfTransLongerThanPref(Vector<WordIndex> s_,
                                    unsigned int ntrgSize,
                                    NbestTableNode<Vector<WordIndex> >& nbt);
  bool trgWordVecIsPrefix(const Vector<WordIndex>& wiVec1,
                          bool lastWiVec1WordIsComplete,
                          const std::string& lastWiVec1Word,
                          const Vector<WordIndex>& wiVec2,
                          bool& equal);
      // returns true if target word vector wiVec1 is a prefix of wiVec2
  bool isPrefix(std::string str1,std::string str2);
      // returns true if string str1 is a prefix of string str2
  
  PositionIndex getLastSrcPosCovered(const Hypothesis& hyp);
      // Get the index of last source position which was covered
  virtual PositionIndex getLastSrcPosCoveredHypData(const HypDataType& hypd)=0;
      // The same as the previous function, but given an object of
      // HypDataType

      // Language model scoring functions
  Score wordPenaltyScore(unsigned int tlen);
  Score sumWordPenaltyScore(unsigned int tlen);
  Score nbestLmScoringFunc(const Vector<WordIndex>& target);
  Score getNgramScoreGivenState(const Vector<WordIndex>& target,
                                LM_State &state);
  Score getScoreEndGivenState(LM_State &state);
  LgProb getSentenceLgProb(const Vector<WordIndex>& target,
                           int verbose=0);

      // Phrase model scoring functions
  Score phrScore_s_t_(const Vector<WordIndex>& s_,
                      const Vector<WordIndex>& t_);
      // obtains the logarithm of pstWeight*ps_t_ 
  Score phrScore_t_s_(const Vector<WordIndex>& s_,
                      const Vector<WordIndex>& t_);
      // obtains the logarithm of ptsWeight*pt_s_
  Score srcJumpScore(unsigned int offset);
      // obtains score for source jump
  Score srcSegmLenScore(unsigned int k,
                        const SourceSegmentation& srcSegm,
                        unsigned int srcLen,
                        unsigned int lastTrgSegmLen);
      // obtains the log-probability for the length of the k'th source
      // segment
  Score trgSegmLenScore(unsigned int x_k,
                        unsigned int x_km1,
                        unsigned int trgLen);
      // obtains the log-probability for the length of a target segment
  
      // Functions to generate n-best translations lists
  virtual bool getNbestTransFor_s_(Vector<WordIndex> s_,
                                   NbestTableNode<Vector<WordIndex> >& nbt,
                                   float N);
      // Get N-best translations for a given source phrase s_.
      // If N is between 0 and 1 then N represents a threshold
      
      // Functions to score n-best translations lists
  virtual Score nbestTransScore(const Vector<WordIndex>& s_,
                                const Vector<WordIndex>& t_)=0;
  virtual Score nbestTransScoreLast(const Vector<WordIndex>& s_,
                                    const Vector<WordIndex>& t_)=0;
      // Cached functions to score n-best translations lists
  Score nbestTransScoreCached(const Vector<WordIndex>& s_,
                              const Vector<WordIndex>& t_);
  Score nbestTransScoreLastCached(const Vector<WordIndex>& s_,
                                  const Vector<WordIndex>& t_);

      // Functions related to pre_trans_actions
  virtual void clearTempVars(void);
  bool lastCharIsBlank(std::string str);
  void verifyDictCoverageForSentence(Vector<std::string>& sentenceVec,
                                     int maxSrcPhraseLength=MAX_SENTENCE_LENGTH_ALLOWED);
      // Verifies dictionary coverage for the sentence to translate.  It
      // is possible to impose an additional constraint consisting of a
      // maximum length for the source phrases.
  void manageUnseenSrcWord(std::string srcw);
  bool unseenSrcWord(std::string srcw);
  bool unseenSrcWordGivenPosition(unsigned int j);
  Score unkWordScoreHeur(void);
  void initHeuristic(unsigned int maxSrcPhraseLength);
      // Initialize heuristic for the sentence to be translated
  
      // Heuristic related functions
  virtual Score calcHeuristicScore(const Hypothesis& hyp);
  void initHeuristicLocalt(int maxSrcPhraseLength);
  Score heurLmScoreLt(Vector<WordIndex>& t_);
  Score heurLmScoreLtNoAdmiss(Vector<WordIndex>& t_);
  Score calcRefLmHeurScore(const Hypothesis& hyp);
  Score calcPrefLmHeurScore(const Hypothesis& hyp);
  Score heuristicLocalt(const Hypothesis& hyp);
  void initHeuristicLocaltd(int maxSrcPhraseLength);
  Score heuristicLocaltd(const Hypothesis& hyp);

      // Vocabulary functions
  WordIndex stringToSrcWordIndex(std::string s)const;
  std::string wordIndexToSrcString(WordIndex w)const;
  Vector<std::string> srcIndexVectorToStrVector(Vector<WordIndex> srcidxVec)const;
  WordIndex stringToTrgWordIndex(std::string s)const;
  std::string wordIndexToTrgString(WordIndex w)const;
  Vector<std::string> trgIndexVectorToStrVector(Vector<WordIndex> trgidxVec)const;
  std::string phraseToStr(const Vector<WordIndex>& phr)const;
  Vector<std::string> phraseToStrVec(const Vector<WordIndex>& phr)const;
  ngramWordIndex tmVocabToLmVocab(WordIndex w);
  void initTmToLmVocabMap(void);
};

//--------------- _phraseBasedTransModel class functions
//

template<class HYPOTHESIS>
_phraseBasedTransModel<HYPOTHESIS>::_phraseBasedTransModel()
{
      // Create pointer to PhraseModelInfo
  phrModelInfoPtr=new PhraseModelInfo;

      // Create pointer to LangModelInfo
  langModelInfoPtr=new LangModelInfo;

      // Set state info
  state=MODEL_IDLE_STATE;

      // Initially, no heuristic is used
  heuristicId=NO_HEURISTIC;

      // Initialize tm to lm vocab map
  initTmToLmVocabMap();
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::loadLangModel(const char* prefixFileName)
{
  std::string penFile;
  std::string predFile;
  int err;
  
  langModelInfoPtr->langModelPars.languageModelFileName=prefixFileName;
  
      // Initializes language model
  if(langModelInfoPtr->lmodel.load(prefixFileName)==ERROR)
    return ERROR;
  
      // load WordPenaltyModel info
  penFile=prefixFileName;
  penFile=penFile+".wpm";
  err=langModelInfoPtr->wordPenaltyModel.load(penFile.c_str());
  if(err==ERROR)
  {
    cerr<<"Warning: File for initializing the word penalty model not provided!"<<endl;
    cerr<<"Using word penalty model based on a geometric distribution."<<endl;
  }
  
      // load WordPredictor info
  predFile=prefixFileName;
  predFile=predFile+".wp";
  err=langModelInfoPtr->wordPredictor.load(predFile.c_str());
  if(err==ERROR)
  {
    cerr<<"Warning: File for initializing the word predictor not provided!"<<endl;
  }
  return OK;
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::loadAligModel(const char* prefixFileName)
{
      // Save parameters
  phrModelInfoPtr->phraseModelPars.srcTrainVocabFileName="";
  phrModelInfoPtr->phraseModelPars.trgTrainVocabFileName="";
  phrModelInfoPtr->phraseModelPars.readTablePrefix=prefixFileName;
  
      // Load phrase model
  if(this->phrModelInfoPtr->invPbModel.load(prefixFileName)!=0)
  {
    cerr<<"Error while reading phrase model file\n";
    return ERROR;
  }  
      
  return OK;
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::printLangModel(std::string printPrefix)
{
  bool retVal=langModelInfoPtr->lmodel.print(printPrefix.c_str());
  if(retVal==ERROR) return ERROR;

  return OK;
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::printAligModel(std::string printPrefix)
{
  bool retVal=this->phrModelInfoPtr->invPbModel.print(printPrefix.c_str());
  if(retVal==ERROR) return ERROR;

  return OK;
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::clear(void)
{
  this->phrModelInfoPtr->invPbModel.clear();
  langModelInfoPtr->lmodel.clear();
  langModelInfoPtr->wordPenaltyModel.clear();
  langModelInfoPtr->wordPredictor.clear();
      // Set state info
  state=MODEL_IDLE_STATE;
}

//---------------------------------
template<class HYPOTHESIS>
PositionIndex _phraseBasedTransModel<HYPOTHESIS>::getLastSrcPosCovered(const Hypothesis& hyp)
{
  return getLastSrcPosCoveredHypData(hyp.getData());
}

//---------------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::wordPenaltyScore(unsigned int tlen)
{
  if(langModelInfoPtr->langModelPars.wpScaleFactor!=0)
  {
    return langModelInfoPtr->langModelPars.wpScaleFactor*(double)langModelInfoPtr->wordPenaltyModel.wordPenaltyScore(tlen);
  }
  else
  {
    return 0;
  }
}

//---------------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::sumWordPenaltyScore(unsigned int tlen)
{
  if(langModelInfoPtr->langModelPars.wpScaleFactor!=0)
  {
    return langModelInfoPtr->langModelPars.wpScaleFactor*(double)langModelInfoPtr->wordPenaltyModel.sumWordPenaltyScore(tlen);
  }
  else
  {
    return 0;
  }
}

//---------------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::nbestLmScoringFunc(const Vector<WordIndex>& target)
{
      // Warning: this function may become a bottleneck when the list of
      // translation options is large
  
  PhraseCacheTable::iterator pctIter;
  pctIter=cnbLmScores.find(target);
  if(pctIter!=cnbLmScores.end())
  {
        // Score was previously stored in the cache table
    return pctIter->second;
  }
  else
  {
        // Score is not stored in the cache table
    Vector<WordIndex> hist;
    LM_State state;    
    langModelInfoPtr->lmodel.getStateForWordSeq(hist,state);
    Score scr=getNgramScoreGivenState(target,state);
    cnbLmScores[target]=scr;
    return scr;
  }
}

//---------------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::getNgramScoreGivenState(const Vector<WordIndex>& target,
                                                                  LM_State &state)
{
  if(langModelInfoPtr->langModelPars.lmScaleFactor!=0)
  {
        // Score not present in cache table
    Vector<WordIndex> target_lm;
    Score unweighted_result=0;

        // target_lm stores the target sentence using indices of the language model
    for(unsigned int i=0;i<target.size();++i)
    {
      target_lm.push_back(tmVocabToLmVocab(target[i]));
    }
      
    for(unsigned int i=0;i<target_lm.size();++i)
    {
          // Try to find score in cache table
      NgramCacheTable::iterator nctIter=cachedNgramScores.find(make_pair(target_lm[i],state));
      if(nctIter!=cachedNgramScores.end())
      {
        unweighted_result+=nctIter->second;
      }
      else
      {
#ifdef WORK_WITH_ZERO_GRAM_PROB
        Score scr=log((double)langModelInfoPtr->lmodel.getZeroGramProb());
#else
        Score scr=(double)langModelInfoPtr->lmodel.getNgramLgProbGivenState(target_lm[i],state);
#endif
            // Increase score
        unweighted_result+=scr;
            // Update cache table
        cachedNgramScores[make_pair(target_lm[i],state)]=scr;
      }
    }
        // Return result
    return langModelInfoPtr->langModelPars.lmScaleFactor*unweighted_result;
  }
  
  else
  {
    return 0;
  }
}

//---------------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::getScoreEndGivenState(LM_State &state)
{
  if(langModelInfoPtr->langModelPars.lmScaleFactor==0) return 0;
  else
  {
        // Try to find score in cache table
    NgramCacheTable::iterator nctIter=cachedNgramScores.find(make_pair(S_END,state));
    if(nctIter!=cachedNgramScores.end())
    {
      return nctIter->second;
    }
    else
    {
#ifdef WORK_WITH_ZERO_GRAM_PROB
      Score result=langModelInfoPtr->langModelPars.lmScaleFactor*log((double)langModelInfoPtr->lmodel.getZeroGramProb());
#else
      Score result=langModelInfoPtr->langModelPars.lmScaleFactor*(double)langModelInfoPtr->lmodel.getLgProbEndGivenState(state);	
#endif
      cachedNgramScores[make_pair(S_END,state)]=result;
      return result;
    }
  }
}

//---------------------------------------
template<class HYPOTHESIS>
LgProb _phraseBasedTransModel<HYPOTHESIS>::getSentenceLgProb(const Vector<WordIndex>& target,
                                                             int verbose)
{
  LgProb lmLgProb=0;
 	 
  Vector<ngramWordIndex> s;
  unsigned int i;
  for(i=0;i<target.size();++i)
    s.push_back(tmVocabToLmVocab(target[i]));
  lmLgProb=(double)langModelInfoPtr->lmodel.getSentenceLog10Prob(s,verbose)*M_LN10; 

  return lmLgProb;  
}

//---------------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::phrScore_s_t_(const Vector<WordIndex>& s_,
                                                        const Vector<WordIndex>& t_)
{
  Score score=0;
  
  if(this->phrModelInfoPtr->phraseModelPars.pstWeight!=0)
  {
        // Check if score of phrase pair is stored in cache table
    PhrasePairCacheTable::iterator ppctIter=cachedInversePhrScores.find(make_pair(s_,t_));
    if(ppctIter!=cachedInversePhrScores.end()) return ppctIter->second;
    else
    {
          // Score has not been cached previously
      score+=this->phrModelInfoPtr->phraseModelPars.pstWeight * (double)this->phrModelInfoPtr->invPbModel.logpt_s_(t_,s_);
      cachedInversePhrScores[make_pair(s_,t_)]=score;
    }
  }
  return score;
}

//---------------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::phrScore_t_s_(const Vector<WordIndex>& s_,
                                                        const Vector<WordIndex>& t_)
{
  Score score=0;
  
  if(phrModelInfoPtr->phraseModelPars.ptsWeight!=0)
  {
        // Check if score of phrase pair is stored in cache table
    PhrasePairCacheTable::iterator ppctIter=cachedDirectPhrScores.find(make_pair(s_,t_));
    if(ppctIter!=cachedDirectPhrScores.end()) return ppctIter->second;
    else
    {
          // Score has not been cached previously
      score=this->phrModelInfoPtr->phraseModelPars.ptsWeight * (double)this->phrModelInfoPtr->invPbModel.logps_t_(t_,s_);
      cachedDirectPhrScores[make_pair(s_,t_)]=score;
    }
  }
  return score;
}

//---------------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::srcJumpScore(unsigned int offset)
{
  Score score=0;
  
  if(phrModelInfoPtr->phraseModelPars.srcJumpWeight!=0)
    score=this->phrModelInfoPtr->phraseModelPars.srcJumpWeight * (double)this->phrModelInfoPtr->invPbModel.trgCutsLgProb(offset);
  return score;  
}

//---------------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::srcSegmLenScore(unsigned int k,
                                                          const SourceSegmentation& srcSegm,
                                                          unsigned int srcLen,
                                                          unsigned int lastTrgSegmLen)
{
  if(phrModelInfoPtr->phraseModelPars.srcSegmLenWeight!=0)
  {
    return this->phrModelInfoPtr->phraseModelPars.srcSegmLenWeight * (double)this->phrModelInfoPtr->invPbModel.trgSegmLenLgProb(k,srcSegm,srcLen,lastTrgSegmLen);
  }
  else return 0;
}

//---------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::trgSegmLenScore(unsigned int x_k,
                                                          unsigned int x_km1,
                                                          unsigned int trgLen)
{
  if(phrModelInfoPtr->phraseModelPars.trgSegmLenWeight!=0)
  {
    return this->phrModelInfoPtr->phraseModelPars.trgSegmLenWeight * (double)this->phrModelInfoPtr->invPbModel.srcSegmLenLgProb(x_k,x_km1,trgLen);
  }
  else return 0;  
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::clearTempVars(void)
{
  // Clear input information
  srcSentVec.clear();
  srcSentIdVec.clear();
  nsrcSentIdVec.clear();
  refSentVec.clear();
  nrefSentIdVec.clear();
  refHeurLmLgProb.clear();
  prefSentVec.clear();
  nprefSentIdVec.clear();
  prefHeurLmLgProb.clear();

  // Clear set of unseen words
  unseenWordsSet.clear();
  
  // Clear data structures that are used 
  // for fast access.

      // Clear language models data members
  cachedNgramScores.clear();
  
      // Clear phrase model caching data members
  cachedDirectPhrScores.clear();
  cachedInversePhrScores.clear();
  
      // Init the map between TM and LM vocabularies
  initTmToLmVocabMap();

      // Clear information of the heuristic used in the translation
  heuristicScoreVec.clear();

      // Clear n-best translation table
  cPhrNbestTransTable.clear();

      // Clear temporary variables of the language model
  langModelInfoPtr->lmodel.clearTempVars();

      // Clear temporary variables of the phrase model
  this->phrModelInfoPtr->invPbModel.clearTempVars();

      // Clear n-best assisted translation for translation with
      // reference
  cPhrNbestTransTableRef.clear();

      // Clear n-best assisted translation for translation with
      // prefix
  cPhrNbestTransTablePref.clear();

      // Clear cached n-best lm scores
  cnbLmScores.clear();
  
      // Clear cached n-best translations scores
  cnbestTransScore.clear();
  cnbestTransScoreLast.clear();
}

//---------------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::lastCharIsBlank(std::string str)
{
  if(str.size()==0) return false;
  else
  {
    if(str[str.size()-1]==' ') return true;
    else return false;
  }
}

//---------------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::verifyDictCoverageForSentence(Vector<std::string>& sentenceVec,
                                                                       int /*maxSrcPhraseLength*/)
{
      // Manage source words without translation options
  for(unsigned int j=0;j<sentenceVec.size();++j)
  {
    NbestTableNode<Vector<WordIndex> > ttNode;
    std::string s=sentenceVec[j];
    Vector<WordIndex> s_;
    s_.push_back(stringToSrcWordIndex(s));
    this->phrModelInfoPtr->invPbModel.getNbestTransFor_t_(s_,ttNode);
    if(ttNode.size()==0)
    {
      manageUnseenSrcWord(s);
    }
  }
  
      // Clear temporary variables of the phrase model
  this->phrModelInfoPtr->invPbModel.clearTempVars();
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::manageUnseenSrcWord(std::string srcw)
{
      // Visualize warning depending on the verbosity level
  if(this->verbosity>0)
  {
    cerr<<"Warning! word "<<srcw<<" has been marked as unseen."<<endl;
  }
      // Add word to the set of unseen words
  unseenWordsSet.insert(srcw);
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::unseenSrcWord(std::string srcw)
{
  std::set<std::string>::iterator setIter;

  setIter=unseenWordsSet.find(srcw);
  if(setIter!=unseenWordsSet.end())
    return true;
  else
    return false;
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::unseenSrcWordGivenPosition(unsigned int j)
{
  return unseenSrcWord(srcSentVec[j-1]);
}

//---------------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::unkWordScoreHeur(void)
{
  Score result=0;

      // Obtain phrase scores
  Vector<WordIndex> s_;
  Vector<WordIndex> t_;

      // Init s_ and t_
  s_.push_back(UNK_WORD);
  t_.push_back(UNK_WORD);
  
      // p(t_|s_) phrase score
  result+=this->phrScore_t_s_(s_,t_); 
  
      // p(s_|t_) phrase score
  result+=this->phrScore_s_t_(s_,t_); 

      // Obtain lm scores
  Vector<WordIndex> hist;
  LM_State state;    
  langModelInfoPtr->lmodel.getStateForWordSeq(hist,state);
  t_.clear();
  t_.push_back(UNK_WORD);
  result+=getNgramScoreGivenState(t_,state);

      // Return result
  return result;
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::initHeuristic(unsigned int maxSrcPhraseLength)
{
  switch(heuristicId)
  {
    case LOCAL_T_HEURISTIC:
      initHeuristicLocalt(maxSrcPhraseLength);
      break;
    case LOCAL_TD_HEURISTIC:
      initHeuristicLocaltd(maxSrcPhraseLength);
      break;
  }
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::initHeuristicLocalt(int maxSrcPhraseLength)
{
  Vector<Score> row;
  NbestTableNode<PhraseTransTableNodeData> ttNode;
  NbestTableNode<PhraseTransTableNodeData>::iterator ttNodeIter;
  Score compositionProduct;
  Score bestScore_ts=0;
  Score score_ts;
  Vector<WordIndex> s_;
    
  unsigned int J=nsrcSentIdVec.size()-1;
  heuristicScoreVec.clear();
      // Initialize row vector    
  for(unsigned int j=0;j<J;++j) row.push_back(-FLT_MAX);
      // Insert rows into t-heuristic table     
  for(unsigned int j=0;j<J;++j) heuristicScoreVec.push_back(row);
     
      // Fill the t-heuristic table
  for(unsigned int y=0;y<J;++y)
  {
    for(unsigned int x=J-y-1;x<J;++x)
    {
          // obtain source phrase
      unsigned int segmRightMostj=y;
      unsigned int segmLeftMostj=J-x-1; 
      s_.clear();

          // obtain score for best translation
      if((segmRightMostj-segmLeftMostj)+1>(unsigned int)maxSrcPhraseLength)
      {
        ttNode.clear();
      }
      else
      {
        for(unsigned int j=segmLeftMostj;j<=segmRightMostj;++j)
          s_.push_back(nsrcSentIdVec[j+1]);
  
            // obtain translations for s_
        getNbestTransFor_s_(s_,ttNode,this->W);
        if(ttNode.size()!=0) // Obtain best p(s_|t_)
        {
          bestScore_ts=-FLT_MAX;
          for(ttNodeIter=ttNode.begin();ttNodeIter!=ttNode.end();++ttNodeIter)
          {
                // Obtain phrase to phrase translation probability
            score_ts=phrScore_s_t_(s_,ttNodeIter->second)+phrScore_t_s_(s_,ttNodeIter->second);
                // Obtain language model heuristic estimation
//            score_ts+=heurLmScoreLt(ttNodeIter->second);
            score_ts+=heurLmScoreLtNoAdmiss(ttNodeIter->second);
            
            if(bestScore_ts<score_ts) bestScore_ts=score_ts;
          }
        }
      }
      
          // Check source phrase length     
      if(x==J-y-1)
      {
            // source phrase has only one word
        if(ttNode.size()!=0)
        {
          heuristicScoreVec[y][x]=bestScore_ts;
        }
        else
        {
          heuristicScoreVec[y][x]=unkWordScoreHeur();
        }
      }
      else
      {
            // source phrase has more than one word
        if(ttNode.size()!=0)
        {
          heuristicScoreVec[y][x]=bestScore_ts;
        }
        else
        {
          heuristicScoreVec[y][x]=-FLT_MAX;
        }
        for(unsigned int z=J-x-1;z<y;++z) 
        {
          compositionProduct=heuristicScoreVec[z][x]+heuristicScoreVec[y][J-2-z];
          if(heuristicScoreVec[y][x]<compositionProduct)
          {
            heuristicScoreVec[y][x]=compositionProduct; 
          }
        }   
      }       
    } 
  }

/* #ifdef THOT_DEBUG */
/*   cerr<<"Table with heuristic values: "<<endl; */
/*   for(unsigned int y=0;y<J;++y) */
/*   { */
/*     for(unsigned int x=0;x<J;++x) */
/*     { */
/*       if((double)heuristicScoreVec[y][x]==-FLT_MAX) fprintf(stderr,"     -inf ");  */
/*       else fprintf(stderr,"%.8f ",(double)heuristicScoreVec[y][x]); */
/*     } */
/*     cerr<<endl; */
/*   } */
/* #endif */
}

//---------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::heurLmScoreLt(Vector<WordIndex>& t_)
{
  if(langModelInfoPtr->langModelPars.lmScaleFactor==0)
  {
    return 0;
  }
  else
  {
    Vector<WordIndex> lmHist;
    unsigned int i;
    LM_State lmState;
    LgProb lp=0;
  
    if(t_.size()>2)
    {
      langModelInfoPtr->lmodel.getStateForBeginOfSentence(lmState);
      langModelInfoPtr->lmodel.getNgramLgProbGivenState(tmVocabToLmVocab(t_[0]),lmState);
      langModelInfoPtr->lmodel.getNgramLgProbGivenState(tmVocabToLmVocab(t_[1]),lmState);
    }
    for(i=2;i<t_.size();++i)
    {
      lp=lp+(double)langModelInfoPtr->lmodel.getNgramLgProbGivenState(tmVocabToLmVocab(t_[i]),lmState);
    }
    return lp*(LgProb)langModelInfoPtr->langModelPars.lmScaleFactor;
  }
}

//---------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::heurLmScoreLtNoAdmiss(Vector<WordIndex>& t_)
{
  Vector<WordIndex> hist;
  LM_State state;    
  langModelInfoPtr->lmodel.getStateForWordSeq(hist,state);
  Score scr=getNgramScoreGivenState(t_,state);
  return scr;
}

//---------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::calcRefLmHeurScore(const Hypothesis& hyp)
{
  if(langModelInfoPtr->langModelPars.lmScaleFactor==0)
  {
    return 0;
  }
  else
  {
    if(refHeurLmLgProb.empty())
    {
          // Fill vector with lm components for the reference sentence
      LgProb lp=0;
      LM_State lmState;
      langModelInfoPtr->lmodel.getStateForBeginOfSentence(lmState);

      refHeurLmLgProb.push_back(NULL_WORD);
      for(unsigned int i=1;i<nrefSentIdVec.size();++i)
      {
        lp+=langModelInfoPtr->lmodel.getNgramLgProbGivenState(tmVocabToLmVocab(nrefSentIdVec[i]),lmState);
        refHeurLmLgProb.push_back(lp);
      }
    }
        // Return heuristic value
    unsigned int len=hyp.partialTransLength();
    LgProb lp=refHeurLmLgProb.back()-refHeurLmLgProb[len];

    return (LgProb)langModelInfoPtr->langModelPars.lmScaleFactor*lp;
  }
}

//---------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::calcPrefLmHeurScore(const Hypothesis& hyp)
{
  if(langModelInfoPtr->langModelPars.lmScaleFactor==0)
  {
    return 0;
  }
  else
  {
    if(prefHeurLmLgProb.empty())
    {
          // Fill vector with lm components for the reference sentence
      LgProb lp=0;
      LM_State lmState;
      langModelInfoPtr->lmodel.getStateForBeginOfSentence(lmState);
      
      prefHeurLmLgProb.push_back(0);
      for(unsigned int i=1;i<nprefSentIdVec.size();++i)
      {
        lp+=langModelInfoPtr->lmodel.getNgramLgProbGivenState(tmVocabToLmVocab(nprefSentIdVec[i]),lmState);
        prefHeurLmLgProb.push_back(lp);
      }
    }
        // Return heuristic value
    LgProb lp;
    unsigned int len=hyp.partialTransLength();
    if(len>=nprefSentIdVec.size()-1) lp=0;
    else
    {
      lp=prefHeurLmLgProb.back()-prefHeurLmLgProb[len];
    }
    return (LgProb)langModelInfoPtr->langModelPars.lmScaleFactor*lp;
  }
}

//---------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::heuristicLocalt(const Hypothesis& hyp)
{
  if(state==MODEL_TRANS_STATE)
  {
    LgProb result=0;
    unsigned int J;
    Vector<pair<PositionIndex,PositionIndex> > gaps;
    
    J=srcSentVec.size();
    this->extract_gaps(hyp,gaps);
    for(unsigned int i=0;i<gaps.size();++i)
    {
      result+=heuristicScoreVec[gaps[i].second-1][J-gaps[i].first];	
    }
    return result;
  }
  else
  {
        // TO-DO
    return 0;
  }
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::initHeuristicLocaltd(int maxSrcPhraseLength)
{
  initHeuristicLocalt(maxSrcPhraseLength);
}
//---------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::heuristicLocaltd(const Hypothesis& hyp)
{
  Score result=0;
  unsigned int J;
  Vector<pair<PositionIndex,PositionIndex> > gaps;
  pair<PositionIndex,PositionIndex> leftGap;
  pair<PositionIndex,PositionIndex> rightGap;
  PositionIndex lastSrcPosCovered;

  if(state==MODEL_TRANS_STATE)
  {
        // Get local t heuristic information
    J=srcSentVec.size();
    this->extract_gaps(hyp,gaps);
    for(unsigned int i=0;i<gaps.size();++i)
    {
      result+=heuristicScoreVec[gaps[i].second-1][J-gaps[i].first];	
    }
  }
  else
  {
        // TO-DO
  }
      // Distortion heuristic information
  lastSrcPosCovered=getLastSrcPosCovered(hyp);
  leftGap.first=0;
  leftGap.second=0;
  rightGap.first=MAX_SENTENCE_LENGTH_ALLOWED+1;
  rightGap.second=MAX_SENTENCE_LENGTH_ALLOWED+1;
  for(unsigned int i=0;i<gaps.size();++i)
  {
    if(gaps[i].second<lastSrcPosCovered && leftGap.second<gaps[i].second)
      leftGap=gaps[i];
    if(gaps[i].first>lastSrcPosCovered && rightGap.first>gaps[i].first)
      rightGap=gaps[i];
  }
  if(leftGap.first!=0)
    result+=srcJumpScore(abs((int)leftGap.first-((int)lastSrcPosCovered+1)));
  if(rightGap.first!=MAX_SENTENCE_LENGTH_ALLOWED+1)
    result+=srcJumpScore(abs((int)rightGap.first-((int)lastSrcPosCovered+1)));
  return result;
}
//---------------------------------
template<class HYPOTHESIS>
WordIndex _phraseBasedTransModel<HYPOTHESIS>::stringToSrcWordIndex(std::string s)const
{
  return this->phrModelInfoPtr->invPbModel.stringToTrgWordIndex(s);    
}

//---------------------------------
template<class HYPOTHESIS>
std::string _phraseBasedTransModel<HYPOTHESIS>::wordIndexToSrcString(WordIndex w)const
{
  return this->phrModelInfoPtr->invPbModel.wordIndexToTrgString(w);      
}

//--------------------------------- 
template<class HYPOTHESIS>
Vector<std::string> _phraseBasedTransModel<HYPOTHESIS>::srcIndexVectorToStrVector(Vector<WordIndex> srcidxVec)const
{
  Vector<std::string> vStr;
  unsigned int i;

  for(i=0;i<srcidxVec.size();++i)
    vStr.push_back(wordIndexToSrcString(srcidxVec[i])); 	 
	
  return vStr;
}

//--------------------------------- 
template<class HYPOTHESIS>
WordIndex _phraseBasedTransModel<HYPOTHESIS>::stringToTrgWordIndex(std::string s)const
{
  return this->phrModelInfoPtr->invPbModel.stringToSrcWordIndex(s);
}

//--------------------------------- 
template<class HYPOTHESIS>
std::string _phraseBasedTransModel<HYPOTHESIS>::wordIndexToTrgString(WordIndex w)const
{
  return this->phrModelInfoPtr->invPbModel.wordIndexToSrcString(w);  
}

//--------------------------------- 
template<class HYPOTHESIS>
Vector<std::string> _phraseBasedTransModel<HYPOTHESIS>::trgIndexVectorToStrVector(Vector<WordIndex> trgidxVec)const
{
  Vector<std::string> vStr;
  unsigned int i;

  for(i=0;i<trgidxVec.size();++i)
    vStr.push_back(wordIndexToTrgString(trgidxVec[i])); 	 
	
  return vStr;
}

//---------------------------------
template<class HYPOTHESIS>
std::string _phraseBasedTransModel<HYPOTHESIS>::phraseToStr(const Vector<WordIndex>& phr)const
{
  std::string s;
  Vector<std::string> svec;

  svec=phraseToStrVec(phr);
  for(unsigned int i=0;i<svec.size();++i)
   {
    if(i==0) s=svec[0];
    else s=s+" "+svec[i];
  }
  return s;  
}

//---------------------------------
template<class HYPOTHESIS>
Vector<std::string> _phraseBasedTransModel<HYPOTHESIS>::phraseToStrVec(const Vector<WordIndex>& phr)const
{
  return trgIndexVectorToStrVector(phr);
}

//---------------------------------
template<class HYPOTHESIS>
ngramWordIndex _phraseBasedTransModel<HYPOTHESIS>::tmVocabToLmVocab(WordIndex w)
{
  std::map<WordIndex,ngramWordIndex>::const_iterator mapIter;
  
  mapIter=tmToLmVocMap.find(w);
  if(mapIter==tmToLmVocMap.end())
  {
        // w not found
        // Obtain string from index
    std::string s=wordIndexToTrgString(w);
        // Add string to the lm vocabulary if necessary
    if(!langModelInfoPtr->lmodel.existSymbol(s))
      langModelInfoPtr->lmodel.addSymbol(s);
        // Map tm word to lm word
    ngramWordIndex nw=langModelInfoPtr->lmodel.stringToWordIndex(s);
    tmToLmVocMap[w]=nw;
    return nw;
  }
  else
  { // w found
    return mapIter->second;
  }  
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::initTmToLmVocabMap(void)
{
  tmToLmVocMap.clear();
  tmToLmVocMap[UNK_WORD]=langModelInfoPtr->lmodel.stringToWordIndex(UNK_SYMBOL_STR);
}

//--------------- _phraseBasedTransModel class methods
//

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::pre_trans_actions(std::string srcsent)
{
  WordIndex w;

      // Clear temporary variables
  clearTempVars();

      // Set state info
  state=MODEL_TRANS_STATE;
  
      // Store source sentence to be translated
  srcSentVec=StrProcUtils::stringToStringVector(srcsent);

      // Verify coverage for source
  if(this->verbosity>0)
    cerr<<"Verify model coverage for source sentence..."<<endl; 
  verifyDictCoverageForSentence(srcSentVec,this->A);

      // Store source sentence as an array of WordIndex.
      // Note: this must be done after verifying the coverage for the
      // source sentence since it may contain unknown words

      // Init source sentence index vector after the coverage has been
      // verified
  srcSentIdVec.clear();
  nsrcSentIdVec.clear();
  nsrcSentIdVec.push_back(NULL_WORD);
  for(unsigned int i=0;i<srcSentVec.size();++i)
  {
    w=stringToSrcWordIndex(srcSentVec[i]);
    srcSentIdVec.push_back(w);
    nsrcSentIdVec.push_back(w);
  }

      // Initialize heuristic (the source sentence must be previously
      // stored)
  initHeuristic(this->A);
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::pre_trans_actions_ref(std::string srcsent,
                                                               std::string refsent)
{
  WordIndex w;

      // Clear temporary variables
  clearTempVars();

        // Set state info
  state=MODEL_TRANSREF_STATE;

      // Store source sentence to be translated
  srcSentVec=StrProcUtils::stringToStringVector(srcsent);

      // Verify coverage for source
  if(this->verbosity>0)
    cerr<<"Verify model coverage for source sentence..."<<endl; 
  verifyDictCoverageForSentence(srcSentVec,this->A);

      // Init source sentence index vector after the coverage has been
      // verified
  srcSentIdVec.clear();
  nsrcSentIdVec.clear();
  nsrcSentIdVec.push_back(NULL_WORD);
  for(unsigned int i=0;i<srcSentVec.size();++i)
  {
    w=stringToSrcWordIndex(srcSentVec[i]);
    srcSentIdVec.push_back(w);
    nsrcSentIdVec.push_back(w);
  }

      // Store reference sentence
  refSentVec=StrProcUtils::stringToStringVector(refsent);

  nrefSentIdVec.clear();
  nrefSentIdVec.push_back(NULL_WORD);
  for(unsigned int i=0;i<refSentVec.size();++i)
  {
    w=stringToTrgWordIndex(refSentVec[i]);
    if(w==UNK_WORD && this->verbosity>0)
      cerr<<"Warning: word "<<refSentVec[i]<<" is not contained in the phrase model vocabulary, ensure that your language model contains the unknown-word token."<<endl;
    nrefSentIdVec.push_back(w);
  }

      // Initialize heuristic (the source sentence must be previously
      // stored)
  initHeuristic(this->A);
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::pre_trans_actions_ver(std::string srcsent,
                                                               std::string refsent)
{
  WordIndex w;

      // Clear temporary variables
  clearTempVars();

        // Set state info
  state=MODEL_TRANSVER_STATE;

      // Store source sentence to be translated
  srcSentVec=StrProcUtils::stringToStringVector(srcsent);

      // Verify coverage for source
  if(this->verbosity>0)
    cerr<<"Verify model coverage for source sentence..."<<endl; 
  verifyDictCoverageForSentence(srcSentVec,this->A);

      // Init source sentence index vector after the coverage has been
      // verified
  srcSentIdVec.clear();
  nsrcSentIdVec.clear();
  nsrcSentIdVec.push_back(NULL_WORD);
  for(unsigned int i=0;i<srcSentVec.size();++i)
  {
    w=stringToSrcWordIndex(srcSentVec[i]);
    srcSentIdVec.push_back(w);
    nsrcSentIdVec.push_back(w);
  }

      // Store reference sentence
  refSentVec=StrProcUtils::stringToStringVector(refsent);

  nrefSentIdVec.clear();
  nrefSentIdVec.push_back(NULL_WORD);
  for(unsigned int i=0;i<refSentVec.size();++i)
  {
    w=stringToTrgWordIndex(refSentVec[i]);
    if(w==UNK_WORD && this->verbosity>0)
      cerr<<"Warning: word "<<refSentVec[i]<<" is not contained in the phrase model vocabulary, ensure that your language model contains the unknown-word token."<<endl;
    nrefSentIdVec.push_back(w);
  }

      // Initialize heuristic (the source sentence must be previously
      // stored)
  initHeuristic(this->A);
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::pre_trans_actions_prefix(std::string srcsent,
                                                                  std::string prefix)
{
  WordIndex w;

      // Clear temporary variables
  clearTempVars();

      // Set state info
  state=MODEL_TRANSPREFIX_STATE;

      // Store source sentence to be translated
  srcSentVec=StrProcUtils::stringToStringVector(srcsent);

      // Verify coverage for source
  if(this->verbosity>0)
    cerr<<"Verify model coverage for source sentence..."<<endl; 
  verifyDictCoverageForSentence(srcSentVec,this->A);


      // Init source sentence index vector after the coverage has been
      // verified
  srcSentIdVec.clear();
  nsrcSentIdVec.clear();
  nsrcSentIdVec.push_back(NULL_WORD);
  for(unsigned int i=0;i<srcSentVec.size();++i)
  {
    w=stringToSrcWordIndex(srcSentVec[i]);
    srcSentIdVec.push_back(w);
    nsrcSentIdVec.push_back(w);
  }

      // Store prefix sentence
  if(lastCharIsBlank(prefix)) lastCharOfPrefIsBlank=true;
  else lastCharOfPrefIsBlank=false;
  prefSentVec=StrProcUtils::stringToStringVector(prefix);

  nprefSentIdVec.clear();
  nprefSentIdVec.push_back(NULL_WORD);
  for(unsigned int i=0;i<prefSentVec.size();++i)
  {
    w=stringToTrgWordIndex(prefSentVec[i]);
    if(w==UNK_WORD && this->verbosity>0)
      cerr<<"Warning: word "<<prefSentVec[i]<<" is not contained in the phrase model vocabulary, ensure that your language model contains the unknown-word token."<<endl;
    nprefSentIdVec.push_back(w);
  }

      // Initialize heuristic (the source sentence must be previously
      // stored)
  initHeuristic(this->A);
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::addSentenceToWordPred(Vector<std::string> strVec,
                                                               int verbose/*=0*/)
{
  switch(this->onlineTrainingPars.onlineLearningAlgorithm)
  {
    case BASIC_INCR_TRAINING:
      incrAddSentenceToWordPred(strVec,verbose);
      break;
    case MINIBATCH_TRAINING:
      minibatchAddSentenceToWordPred(strVec,verbose);
      break;
    case BATCH_RETRAINING:
      batchAddSentenceToWordPred(strVec,verbose);
      break;
    default:
      cerr<<"Warning: requested online update of word predictor with id="<<this->onlineTrainingPars.onlineLearningAlgorithm<<" is not implemented."<<endl;
      break;
  }
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::incrAddSentenceToWordPred(Vector<std::string> strVec,
                                                                   int verbose/*=0*/)
{
  if(verbose)
    cerr<<"Adding a new sentence to word predictor..."<<endl;
  langModelInfoPtr->wordPredictor.addSentence(strVec);
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::minibatchAddSentenceToWordPred(Vector<std::string> strVec,
                                                                        int verbose/*=0*/)
{
      // Store sentence
  wordPredSentVec.push_back(strVec);
  
      // Check if a mini-batch has to be processed
      // (onlineTrainingPars.learnStepSize determines the size of the
      // mini-batch)
  unsigned int batchSize=(unsigned int)this->onlineTrainingPars.learnStepSize;
  if(!wordPredSentVec.empty() &&
     (wordPredSentVec.size()%batchSize)==0)
  {
    if(verbose)
      cerr<<"Adding "<<batchSize<<" sentences to word predictor..."<<endl;
    
    for(unsigned int i=0;i<wordPredSentVec.size();++i)
      langModelInfoPtr->wordPredictor.addSentence(wordPredSentVec[i]);
    wordPredSentVec.clear();
  }
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::batchAddSentenceToWordPred(Vector<std::string> strVec,
                                                                    int verbose/*=0*/)
{
      // Store sentence
  wordPredSentVec.push_back(strVec);
  
      // Check if a mini-batch has to be processed
      // (onlineTrainingPars.learnStepSize determines the size of the
      // mini-batch)
  unsigned int batchSize=(unsigned int)this->onlineTrainingPars.learnStepSize;
  if(!wordPredSentVec.empty() &&
     (wordPredSentVec.size()%batchSize)==0)
  {
    if(verbose)
      cerr<<"Adding "<<batchSize<<" sentences to word predictor..."<<endl;
    
    for(unsigned int i=0;i<wordPredSentVec.size();++i)
      langModelInfoPtr->wordPredictor.addSentence(wordPredSentVec[i]);
    wordPredSentVec.clear();
  }
}

//---------------------------------
template<class HYPOTHESIS>
pair<Count,std::string>
_phraseBasedTransModel<HYPOTHESIS>::getBestSuffix(std::string input)
{
  return langModelInfoPtr->wordPredictor.getBestSuffix(input);
}

//---------------------------------
template<class HYPOTHESIS>
pair<Count,std::string>
_phraseBasedTransModel<HYPOTHESIS>::getBestSuffixGivenHist(Vector<std::string> hist,
                                                           std::string input)
{
  WordPredictor::SuffixList suffixList;
  WordPredictor::SuffixList::iterator suffixListIter;
  LgProb lp;
  LgProb maxlp=-FLT_MAX;
  pair<Count,std::string> bestCountSuffix;

      // Get suffix list for input
  langModelInfoPtr->wordPredictor.getSuffixList(input,suffixList);
  if(suffixList.size()==0)
  {
        // There are not any suffix
    return make_pair(0,"");
  }
  else
  {
        // There are one or more suffixes
    LM_State lmState;
    LM_State aux;

        // Initialize language model state given history
    langModelInfoPtr->lmodel.getStateForBeginOfSentence(lmState);
    for(unsigned int i=0;i<hist.size();++i)
    {
      langModelInfoPtr->lmodel.getNgramLgProbGivenState(langModelInfoPtr->lmodel.stringToWordIndex(hist[i]),lmState);
    }

        // Obtain probability for each suffix given history
    for(suffixListIter=suffixList.begin();suffixListIter!=suffixList.end();++suffixListIter)
    {
      std::string lastw;
      
      aux=lmState;
      lastw=input+suffixListIter->second;
      lp=langModelInfoPtr->lmodel.getNgramLgProbGivenState(langModelInfoPtr->lmodel.stringToWordIndex(lastw),aux);
      if(maxlp<lp)
      {
        bestCountSuffix.first=suffixListIter->first;
        bestCountSuffix.second=suffixListIter->second;
        maxlp=lp;
      }
    }
        // Return best suffix
    return bestCountSuffix;
  }
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::getHypDataVecForGap(const Hypothesis& hyp,
                                                             PositionIndex srcLeft,
                                                             PositionIndex srcRight,
                                                             Vector<HypDataType>& hypDataTypeVec,
                                                             float N)
{
  NbestTableNode<Vector<WordIndex> > ttNode;
  NbestTableNode<Vector<WordIndex> >::iterator ttNodeIter;
  HypDataType hypData=hyp.getData();
  HypDataType newHypData;

  hypDataTypeVec.clear();
  
  getTransForHypUncovGap(hyp,srcLeft,srcRight,ttNode,N);

  if(this->verbosity>=2)
  {
    cerr<<"  trying to cover from src. pos. "<<srcLeft<<" to "<<srcRight<<"; ";
    cerr<<"Filtered "<<ttNode.size()<<" translations"<<endl;
  }

  for(ttNodeIter=ttNode.begin();ttNodeIter!=ttNode.end();++ttNodeIter)
  {
    if(this->verbosity>=3)
    {
      cerr<<"   ";
      for(unsigned int i=srcLeft;i<=srcRight;++i) cerr<<this->srcSentVec[i-1]<<" ";
      cerr<<"||| ";
      for(unsigned int i=0;i<ttNodeIter->second.size();++i)
        cerr<<this->wordIndexToTrgString(ttNodeIter->second[i])<<" ";
      cerr<<"||| "<<ttNodeIter->first<<endl;
    }

    newHypData=hypData;
    extendHypDataIdx(srcLeft,srcRight,ttNodeIter->second,newHypData);
    hypDataTypeVec.push_back(newHypData);
  }
  if(hypDataTypeVec.empty()) return false;
  else return true;
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::getHypDataVecForGapRef(const Hypothesis& hyp,
                                                                PositionIndex srcLeft,
                                                                PositionIndex srcRight,
                                                                Vector<HypDataType>& hypDataTypeVec,
                                                                float N)
{
  NbestTableNode<Vector<WordIndex> > ttNode;
  NbestTableNode<Vector<WordIndex> >::iterator ttNodeIter;
  HypDataType hypData=hyp.getData();
  HypDataType newHypData;

  hypDataTypeVec.clear();
  
  getTransForHypUncovGapRef(hyp,srcLeft,srcRight,ttNode,N);

  if(this->verbosity>=2)
  {
    cerr<<"  trying to cover from src. pos. "<<srcLeft<<" to "<<srcRight<<"; ";
    cerr<<"Filtered "<<ttNode.size()<<" translations"<<endl;
  }

  for(ttNodeIter=ttNode.begin();ttNodeIter!=ttNode.end();++ttNodeIter)
  {
    if(this->verbosity>=3)
    {
      cerr<<"   ";
      for(unsigned int i=srcLeft;i<=srcRight;++i) cerr<<this->srcSentVec[i-1]<<" ";
      cerr<<"||| ";
      for(unsigned int i=0;i<ttNodeIter->second.size();++i)
        cerr<<this->wordIndexToTrgString(ttNodeIter->second[i])<<" ";
      cerr<<"||| "<<ttNodeIter->first<<endl;
    }

    newHypData=hypData;
    extendHypDataIdx(srcLeft,srcRight,ttNodeIter->second,newHypData);
    bool equal;
    if(hypDataTransIsPrefixOfTargetRef(newHypData,equal))
    {
      if((this->isCompleteHypData(newHypData) && equal) || !this->isCompleteHypData(newHypData))
        hypDataTypeVec.push_back(newHypData);
    }
  }
  if(hypDataTypeVec.empty()) return false;
  else return true;  
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::getHypDataVecForGapVer(const Hypothesis& hyp,
                                                                PositionIndex srcLeft,
                                                                PositionIndex srcRight,
                                                                Vector<HypDataType>& hypDataTypeVec,
                                                                float N)
{
  NbestTableNode<Vector<WordIndex> > ttNode;
  NbestTableNode<Vector<WordIndex> >::iterator ttNodeIter;
  HypDataType hypData=hyp.getData();
  HypDataType newHypData;

  hypDataTypeVec.clear();
  
  getTransForHypUncovGapVer(hyp,srcLeft,srcRight,ttNode,N);

  if(this->verbosity>=2)
  {
    cerr<<"  trying to cover from src. pos. "<<srcLeft<<" to "<<srcRight<<"; ";
    cerr<<"Filtered "<<ttNode.size()<<" translations"<<endl;
  }

  for(ttNodeIter=ttNode.begin();ttNodeIter!=ttNode.end();++ttNodeIter)
  {
    if(this->verbosity>=3)
    {
      cerr<<"   ";
      for(unsigned int i=srcLeft;i<=srcRight;++i) cerr<<this->srcSentVec[i-1]<<" ";
      cerr<<"||| ";
      for(unsigned int i=0;i<ttNodeIter->second.size();++i)
        cerr<<this->wordIndexToTrgString(ttNodeIter->second[i])<<" ";
      cerr<<"||| "<<ttNodeIter->first<<endl;
    }

    newHypData=hypData;
    extendHypDataIdx(srcLeft,srcRight,ttNodeIter->second,newHypData);
    bool equal;
    if(hypDataTransIsPrefixOfTargetRef(newHypData,equal))
    {
      if((this->isCompleteHypData(newHypData) && equal) || !this->isCompleteHypData(newHypData))
        hypDataTypeVec.push_back(newHypData);
    }
  }
  if(hypDataTypeVec.empty()) return false;
  else return true;  
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::getHypDataVecForGapPref(const Hypothesis& hyp,
                                                                 PositionIndex srcLeft,
                                                                 PositionIndex srcRight,
                                                                 Vector<HypDataType>& hypDataTypeVec,
                                                                 float N)
{
  NbestTableNode<Vector<WordIndex> > ttNode;
  NbestTableNode<Vector<WordIndex> >::iterator ttNodeIter;
  HypDataType hypData=hyp.getData();
  HypDataType newHypData;

  hypDataTypeVec.clear();
  
  getTransForHypUncovGapPref(hyp,srcLeft,srcRight,ttNode,N);

  if(this->verbosity>=2)
  {
    cerr<<"  trying to cover from src. pos. "<<srcLeft<<" to "<<srcRight<<"; ";
    cerr<<"Filtered "<<ttNode.size()<<" translations"<<endl;
  }

  for(ttNodeIter=ttNode.begin();ttNodeIter!=ttNode.end();++ttNodeIter)
  {
    if(this->verbosity>=3)
    {
      cerr<<"   ";
      for(unsigned int i=srcLeft;i<=srcRight;++i) cerr<<this->srcSentVec[i-1]<<" ";
      cerr<<"||| ";
      for(unsigned int i=0;i<ttNodeIter->second.size();++i)
        cerr<<this->wordIndexToTrgString(ttNodeIter->second[i])<<" ";
      cerr<<"||| "<<ttNodeIter->first<<endl;
    }

    newHypData=hypData;
    extendHypDataIdx(srcLeft,srcRight,ttNodeIter->second,newHypData);
    hypDataTypeVec.push_back(newHypData);
  }
  if(hypDataTypeVec.empty()) return false;
  else return true;
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::getTransForHypUncovGap(const Hypothesis& /*hyp*/,
                                                                PositionIndex srcLeft,
                                                                PositionIndex srcRight,
                                                                NbestTableNode<Vector<WordIndex> >& nbt,
                                                                float N)
{
      // Check if source phrase has only one word and this word has been
      // marked as unseen word
  if(srcLeft==srcRight && unseenSrcWord(srcSentVec[srcLeft-1]))
  {
    Vector<WordIndex> unkWordVec;
    unkWordVec.push_back(UNK_WORD);
    nbt.insert(0,unkWordVec);
    return false;
  }
  else
  {
        // search translations for s on translation table
    NbestTableNode<PhraseTransTableNodeData> *transTableNodePtr;
    Vector<WordIndex> s_;
    
    for(unsigned int i=srcLeft;i<=srcRight;++i)
    {
      s_.push_back(nsrcSentIdVec[i]);
    }
    
    transTableNodePtr=cPhrNbestTransTable.getTranslationsForKey(make_pair(srcLeft,srcRight));
    if(transTableNodePtr!=NULL)
    {// translation present in the cache translation table
      nbt=*transTableNodePtr;
      if(nbt.size()==0) return false;
      else return true;
    }
    else
    {   
      getNbestTransFor_s_(s_,nbt,N);
      cPhrNbestTransTable.insertEntry(make_pair(srcLeft,srcRight),nbt);
      if(nbt.size()==0) return false;
      else return true;
    }
  }
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::getTransForHypUncovGapRef(const Hypothesis& hyp,
                                                                   PositionIndex srcLeft,
                                                                   PositionIndex srcRight,
                                                                   NbestTableNode<Vector<WordIndex> >& nbt,
                                                                   float N)
{
  Vector<WordIndex> s_;
  Vector<WordIndex> t_;
  Vector<WordIndex> ntarget;

      // Obtain source phrase
  for(unsigned int i=srcLeft;i<=srcRight;++i)
  {
    s_.push_back(nsrcSentIdVec[i]);
  }

      // Obtain length limits for target phrase
  unsigned int minTrgSize=0;
  if(s_.size()>this->E) minTrgSize=s_.size()-this->E;
  unsigned int maxTrgSize=s_.size()+this->E;

  ntarget=hyp.getPartialTrans();	

  nbt.clear();
  if(ntarget.size()>nrefSentIdVec.size()) return false;
  if(this->numberOfUncoveredSrcWords(hyp)-(srcRight-srcLeft+1)>0)
  {
        // This is not the last gap to be covered
    NbestTableNode<PhraseTransTableNodeData> *transTableNodePtr;
    PhrNbestTransTableRefKey pNbtRefKey;

    pNbtRefKey.srcLeft=srcLeft;
    pNbtRefKey.srcRight=srcRight;
    pNbtRefKey.ntrgSize=ntarget.size();
        // The number of gaps to be covered AFTER covering
        // s_{srcLeft}...s_{srcRight} is obtained to ensure that the
        // resulting hypotheses have at least as many gaps as reference
        // words to add
    if(this->nonMonotonicity==1)
      pNbtRefKey.numGaps=1;
    else
    {
      Bitset<MAX_SENTENCE_LENGTH_ALLOWED> key=hyp.getKey();
      for(unsigned int i=srcLeft;i<=srcRight;++i) key.set(i);
      pNbtRefKey.numGaps=this->get_num_gaps(key);
    }
     
        // Search the required translations in the cache translation
        // table
    
    transTableNodePtr=cPhrNbestTransTableRef.getTranslationsForKey(pNbtRefKey);
    if(transTableNodePtr!=NULL)
    {// translations present in the cache translation table
      nbt=*transTableNodePtr;
    }
    else
    {// translations not present in the cache translation table
      for(PositionIndex i=ntarget.size();i<nrefSentIdVec.size()-pNbtRefKey.numGaps;++i)
      {
        t_.push_back(nrefSentIdVec[i]);
        if(t_.size()>=minTrgSize && t_.size()<=maxTrgSize)
        {
          Score scr=nbestTransScoreCached(s_,t_);
          nbt.insert(scr,t_);
        }
      }
          // Prune the list
      if(N>=1)
        while(nbt.size()>(unsigned int) N) nbt.removeLastElement();
      else
      {
        Score bscr=nbt.getScoreOfBestElem();
        nbt.pruneGivenThreshold(bscr+(double)log(N));
      }
          // Store the list in cPhrNbestTransTableRef
      cPhrNbestTransTableRef.insertEntry(pNbtRefKey,nbt);
    }
  }
  else
  {
        // The last gap will be covered
    for(PositionIndex i=ntarget.size();i<nrefSentIdVec.size();++i)
      t_.push_back(nrefSentIdVec[i]);
    if(t_.size()>=minTrgSize && t_.size()<=maxTrgSize)
    {
      Score scr=nbestTransScoreCached(s_,t_);
      nbt.insert(scr,t_);
    }
  }
  if(nbt.size()==0) return false;
  else return true;
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::getTransForHypUncovGapVer(const Hypothesis& hyp,
                                                                   PositionIndex srcLeft,
                                                                   PositionIndex srcRight,
                                                                   NbestTableNode<Vector<WordIndex> >& nbt,
                                                                   float N)
{
  return getTransForHypUncovGap(hyp,srcLeft,srcRight,nbt,N);
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::getTransForHypUncovGapPref(const Hypothesis& hyp,
                                                                    PositionIndex srcLeft,
                                                                    PositionIndex srcRight,
                                                                    NbestTableNode<Vector<WordIndex> >& nbt,
                                                                    float N)
{
  unsigned int ntrgSize=hyp.getPartialTrans().size();
      // Check if the prefix has been generated
  if(ntrgSize<nprefSentIdVec.size())
  {
        // The prefix has not been generated
    NbestTableNode<PhraseTransTableNodeData> *transTableNodePtr;
    PhrNbestTransTablePrefKey pNbtPrefKey;

    pNbtPrefKey.srcLeft=srcLeft;
    pNbtPrefKey.srcRight=srcRight;
    pNbtPrefKey.ntrgSize=ntrgSize;
    if(this->numberOfUncoveredSrcWords(hyp)-(srcRight-srcLeft+1)>0)
      pNbtPrefKey.lastGap=false;
    else pNbtPrefKey.lastGap=true;
    
        // Search the required translations in the cache translation
        // table
    transTableNodePtr=cPhrNbestTransTablePref.getTranslationsForKey(pNbtPrefKey);
    if(transTableNodePtr!=NULL)
    {// translations present in the cache translation table
      nbt=*transTableNodePtr;
    }
    else
    {
          // Obtain list
      transUncovGapPrefNoGen(hyp,srcLeft,srcRight,nbt);
      
          // Prune the list
      if(N>=1)
        while(nbt.size()>(unsigned int) N) nbt.removeLastElement();
      else
      {
        Score bscr=nbt.getScoreOfBestElem();
        nbt.pruneGivenThreshold(bscr+(double)log(N));
      }
          // Store the list in cPhrNbestTransTablePref
      cPhrNbestTransTablePref.insertEntry(pNbtPrefKey,nbt);
    }
    if(nbt.size()==0) return false;
    else return true;
  }
  else
  {
        // The prefix has been completely generated, the nbest list
        // is obtained as if no prefix was given
    return getTransForHypUncovGap(hyp,srcLeft,srcRight,nbt,N);
  }
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::transUncovGapPrefNoGen(const Hypothesis& hyp,
                                                                PositionIndex srcLeft,
                                                                PositionIndex srcRight,
                                                                NbestTableNode<Vector<WordIndex> >& nbt)
{
  Vector<WordIndex> s_;
    
      // Obtain source phrase
  nbt.clear();
  for(unsigned int i=srcLeft;i<=srcRight;++i)
  {
    s_.push_back(nsrcSentIdVec[i]);
  }
      // Obtain length limits for target phrase
  unsigned int minTrgSize=0;
  if(s_.size()>this->E) minTrgSize=s_.size()-this->E;
  unsigned int maxTrgSize=s_.size()+this->E;
    
  unsigned int ntrgSize=hyp.getPartialTrans().size();

      // Check if we are covering the last gap of the hypothesis
  if(this->numberOfUncoveredSrcWords(hyp)-(srcRight-srcLeft+1)>0)
  {
        // This is not the last gap to be covered.

        // Add translations with length in characters greater than the
        // prefix length.
    genListOfTransLongerThanPref(s_,ntrgSize,nbt);

        // Add translations with length lower than the prefix length.
    Vector<WordIndex> t_;
    if(nprefSentIdVec.size()>1)
    {
      for(PositionIndex i=ntrgSize;i<nprefSentIdVec.size()-1;++i)
      {
        t_.push_back(nprefSentIdVec[i]);
        if(t_.size()>=minTrgSize && t_.size()<=maxTrgSize)
        {
          Score scr=nbestTransScoreCached(s_,t_);
          nbt.insert(scr,t_);
        }
      }
    }
  }
  else
  {
        // This is the last gap to be covered.

        // Add translations with length in characters greater than the
        // prefix length.
    genListOfTransLongerThanPref(s_,ntrgSize,nbt);
  }
  
      // Insert the remaining prefix itself in nbt
  Vector<WordIndex> remainingPref;
  for(unsigned int i=ntrgSize;i<nprefSentIdVec.size();++i)
    remainingPref.push_back(nprefSentIdVec[i]);
  nbt.insert(nbestTransScoreLastCached(s_,remainingPref),remainingPref);
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::genListOfTransLongerThanPref(Vector<WordIndex> s_,
                                                                      unsigned int ntrgSize,
                                                                      NbestTableNode<Vector<WordIndex> >& nbt)
{
  THOT_CURR_PBM_TYPE::SrcTableNode srctn;
  THOT_CURR_PBM_TYPE::SrcTableNode::iterator srctnIter;
  Vector<WordIndex> remainingPref;

      // clear nbt
  nbt.clear();
  
      // Store the remaining prefix to be generated in remainingPref
  for(unsigned int i=ntrgSize;i<nprefSentIdVec.size();++i)
    remainingPref.push_back(nprefSentIdVec[i]);

      // Obtain translations for source segment s_
  this->phrModelInfoPtr->invPbModel.getTransFor_t_(s_,srctn);
  for(srctnIter=srctn.begin();srctnIter!=srctn.end();++srctnIter)
  {
        // Filter those translations whose length in words is
        // greater or equal than the remaining prefix length
    if(srctnIter->first.size()>=remainingPref.size())
    {
          // Filter those translations having "remainingPref"
          // as prefix
      bool equal;
      if(trgWordVecIsPrefix(remainingPref,
                            lastCharOfPrefIsBlank,
                            prefSentVec.back(),
                            srctnIter->first,
                            equal))
      {
            // Filter translations not exactly equal to "remainingPref"
        if(!equal)
        {
          Score scr=nbestTransScoreLastCached(s_,srctnIter->first);
          nbt.insert(scr,srctnIter->first);
        }
      }
    }
  }
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::trgWordVecIsPrefix(const Vector<WordIndex>& wiVec1,
                                                            bool lastWiVec1WordIsComplete,
                                                            const std::string& lastWiVec1Word,
                                                            const Vector<WordIndex>& wiVec2,
                                                            bool& equal)
{
  equal=false;
  
      // returns true if target word vector wiVec1 is a prefix of wiVec2
  if(wiVec1.size()>wiVec2.size()) return false;

  for(unsigned int i=0;i<wiVec1.size();++i)
  {
    if(wiVec1[i]!=wiVec2[i])
    {
      if(i==wiVec1.size()-1 && !lastWiVec1WordIsComplete)
      {
        if(!StrProcUtils::isPrefix(lastWiVec1Word,wordIndexToTrgString(wiVec2[i])))
          return false;
      }
      else return false;
    }
  }

  if(wiVec1.size()==wiVec2.size() &&
     lastWiVec1Word==wordIndexToTrgString(wiVec2.back()))
  {
    equal=true;
  }
  
  return true;
}

//---------------------------------
template<class HYPOTHESIS>
bool _phraseBasedTransModel<HYPOTHESIS>::getNbestTransFor_s_(Vector<WordIndex> s_,
                                                             NbestTableNode<Vector<WordIndex> >& nbt,
                                                             float N)
{
#ifdef THOT_USE_ONLY_PTS_FOR_NBEST
  bool b;
      // retrieve translations from table
  if(N>=1)
    b=phrModelInfoPtr->invPbModel.getNbestTransFor_t_(s_,nbt,(int)N);
  else
  {
    b=phrModelInfoPtr->invPbModel.getNbestTransFor_t_(s_,nbt);
    Score scr=nbt.getScoreOfBestElem();
    
    nbt.pruneGivenThreshold(scr+(double)log(N));
  }  
  return b;
#else
  THOT_CURR_PBM_TYPE::SrcTableNode srctn;
  THOT_CURR_PBM_TYPE::SrcTableNode::iterator srctnIter;
  bool ret;

      // Obtain the whole list of translations
  nbt.clear();
  ret=this->phrModelInfoPtr->invPbModel.getTransFor_t_(s_,srctn);
  if(!ret) return false;
  else
  {
    Score scr;

        // This loop may become a bottleneck if the number of translation
        // options is high
    for(srctnIter=srctn.begin();srctnIter!=srctn.end();++srctnIter)
    {
      scr=nbestTransScoreCached(s_,srctnIter->first);
      nbt.insert(scr,srctnIter->first);
    }
  }
      // Prune the list depending on the value of N
      // retrieve translations from table
  if(N>=1)
    while(nbt.size()>(unsigned int) N) nbt.removeLastElement();
  else
  {
    Score bscr=nbt.getScoreOfBestElem();    
    nbt.pruneGivenThreshold(bscr+(double)log(N));
  }
  return true;
#endif
}

//---------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::nbestTransScoreCached(const Vector<WordIndex>& s_,
                                                                const Vector<WordIndex>& t_)
{
  PhrasePairCacheTable::iterator ppctIter;
  ppctIter=cnbestTransScore.find(make_pair(s_,t_));
  if(ppctIter!=cnbestTransScore.end())
  {
        // Score was previously stored in the cache table
    return ppctIter->second;
  }
  else
  {
        // Score is not stored in the cache table
    Score scr=nbestTransScore(s_,t_);
    cnbestTransScore[make_pair(s_,t_)]=scr;
    return scr;
  }
}

//---------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::nbestTransScoreLastCached(const Vector<WordIndex>& s_,
                                                                    const Vector<WordIndex>& t_)
{
  PhrasePairCacheTable::iterator ppctIter;
  ppctIter=cnbestTransScoreLast.find(make_pair(s_,t_));
  if(ppctIter!=cnbestTransScoreLast.end())
  {
        // Score was previously stored in the cache table
    return ppctIter->second;
  }
  else
  {
        // Score is not stored in the cache table
    Score scr=nbestTransScoreLast(s_,t_);
    cnbestTransScoreLast[make_pair(s_,t_)]=scr;
    return scr;
  }
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::addHeuristicToHyp(Hypothesis& hyp)
{
  hyp.addHeuristic(calcHeuristicScore(hyp));
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::sustractHeuristicToHyp(Hypothesis& hyp)
{
  hyp.sustractHeuristic(calcHeuristicScore(hyp));
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::setHeuristic(unsigned int _heuristicId)
{
  heuristicId=_heuristicId;
}

//---------------------------------
template<class HYPOTHESIS>
Score _phraseBasedTransModel<HYPOTHESIS>::calcHeuristicScore(const Hypothesis& hyp)
{
  Score score=0;

  if(state==MODEL_TRANSREF_STATE)
  {
        // translation with reference
    score+=calcRefLmHeurScore(hyp);
  }
  if(state==MODEL_TRANSPREFIX_STATE)
  {
        // translation with prefix
    score+=calcPrefLmHeurScore(hyp);
  }

  switch(heuristicId)
  {
    case NO_HEURISTIC:
      break;
    case LOCAL_T_HEURISTIC:
      score+=heuristicLocalt(hyp);
      break;
    case LOCAL_TD_HEURISTIC:
      score+=heuristicLocaltd(hyp);
      break;
  }
  return score;
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::printHyp(const Hypothesis& hyp,
                                                  ostream &outS,
                                                  int verbose)
{
  Vector<std::string> trgStrVec;
  Vector<WordIndex> trans=hyp.getPartialTrans();
  SourceSegmentation sourceSegmentation;
  Vector<PositionIndex> targetSegmentCuts;
  Vector<pair<PositionIndex,PositionIndex> > amatrix;
  HypDataType hypDataType;
  Hypothesis auxHyp;
  Vector<Score> scoreComponents;
  
      // Obtain target string vector
  trgStrVec=trgIndexVectorToStrVector(hyp.getPartialTrans());

      // Print score
  outS <<"Score: "<<hyp.getScore()<<" ; ";
      // Print weights
  this->printWeights(outS);
  outS <<" ; ";
      // Obtain score components
  hypDataType=hyp.getData();
  this->incrScore(this->nullHypothesis(),hypDataType,auxHyp,scoreComponents);
      // Print score components
  for(unsigned int i=0;i<scoreComponents.size();++i)
    outS<<scoreComponents[i]<<" ";

      // Print score + heuristic
  addHeuristicToHyp(auxHyp);
  outS <<"; Score+heur: "<<auxHyp.getScore()<<" ";
    
      // Print warning if the alignment is not complete
  if(!this->isComplete(hyp)) outS<< "; Incomplete_alignment!";

      // Obtain phrase alignment
  this->aligMatrix(hyp,amatrix);
  this->getPhraseAlignment(amatrix,sourceSegmentation,targetSegmentCuts);

      // Print alignment information
  outS<<" | ";
  for(unsigned int i=1;i<trgStrVec.size();++i)
    outS<<trgStrVec[i]<<" ";
  outS << "| ";
  for(unsigned int k=0;k<sourceSegmentation.size();k++)
 	outS<<"( "<<sourceSegmentation[k].first<<" , "<<sourceSegmentation[k].second<<" ) "; 
  outS<< "| "; 
  for (unsigned int j=0; j<targetSegmentCuts.size(); j++)
    outS << targetSegmentCuts[j] << " ";
  
      // Print hypothesis key
  outS<<"| hypkey: "<<hyp.getKey()<<" ";

      // Print hypothesis equivalence class
  outS<<"| hypEqClass: "<<hyp.getEqClass()<<endl;

  if(verbose)
  {
    unsigned int numSteps=sourceSegmentation.size()-1;
    outS<<"----------------------------------------------"<<endl;
    outS<<"Score components for previous expansion steps:"<<endl;
    auxHyp=hyp;
    while(this->obtainPredecessor(auxHyp))
    {
      scoreComponents=scoreCompsForHyp(auxHyp);
      outS<<"Step "<<numSteps<<" : ";
      for(unsigned int i=0;i<scoreComponents.size();++i)
      {
        outS<<scoreComponents[i]<<" ";
      }
      outS<<endl;
      --numSteps;
    }
    outS<<"----------------------------------------------"<<endl;
  }
#ifdef THOT_DEBUG
      // Print debug information
  for(unsigned int i=0;i<hyp.hDebug.size();++i)
  {
    hyp.hDebug[i].print(outS);
  }
#endif 
}

//---------------------------------
template<class HYPOTHESIS>
void _phraseBasedTransModel<HYPOTHESIS>::extendHypData(PositionIndex srcLeft,
                                                       PositionIndex srcRight,
                                                       const Vector<std::string>& trgPhrase,
                                                       HypDataType& hypd)
{
  Vector<WordIndex> trgPhraseIdx;
  
  for(unsigned int i=0;i<trgPhrase.size();++i)
    trgPhraseIdx.push_back(stringToTrgWordIndex(trgPhrase[i]));
  extendHypDataIdx(srcLeft,srcRight,trgPhraseIdx,hypd);
}

//---------------------------------
template<class HYPOTHESIS>
Vector<std::string> _phraseBasedTransModel<HYPOTHESIS>::getTransInPlainTextVec(const Hypothesis& hyp,
                                                                               set<unsigned int>& unknownWords)const
{
  unknownWords.clear();

  Vector<WordIndex> nvwi;
  Vector<WordIndex> vwi;

      // Obtain vector of WordIndex
  nvwi=hyp.getPartialTrans();
  for(unsigned int j=1;j<nvwi.size();++j)
  {
    vwi.push_back(nvwi[j]);
  }
      // Obtain vector of strings
  Vector<std::string> trgVecStr=trgIndexVectorToStrVector(vwi);

      // Unknown words contained in trgVecStr. For this purpose, the
      // model state must be checked
  if(state==MODEL_TRANS_STATE || state==MODEL_TRANSPREFIX_STATE)
  {
        // Model is being used to translate a sentence or to translate a
        // sentence given a prefix
        
        // Remove unknown words from trgVecStr
    for(unsigned int j=0;j<trgVecStr.size();++j)
    {
      if(trgVecStr[j]==UNK_WORD_STR)
      {
        unknownWords.insert(j);
        if(state==MODEL_TRANSPREFIX_STATE && j<prefSentVec.size())
        {
              // Unknown word must be replaced by a prefix word
          trgVecStr[j]=prefSentVec[j];
        }
        else
        {
              // Find source word aligned with unknown word
          for(unsigned int i=0;i<srcSentVec.size();++i)
          {
            if(hyp.areAligned(i+1,j+1))
            {
              trgVecStr[j]=srcSentVec[i];
              break;
            }
          }
        }
      }
    }
    return trgVecStr;
  }
  else
  {
    if(state==MODEL_TRANSREF_STATE || state==MODEL_TRANSVER_STATE)
    {
          // Model is being used to generate a reference or to verify
          // model coverage
      for(unsigned int i=0;i<trgVecStr.size();++i)
      {
        if(i<refSentVec.size())
        {
          trgVecStr[i]=refSentVec[i];
          unknownWords.insert(i);
        }
      }
      return trgVecStr;
    }
    else return trgVecStr;
  }
}

//---------------------------------
template<class HYPOTHESIS>
Vector<Score>
_phraseBasedTransModel<HYPOTHESIS>::scoreCompsForHyp(const Hypothesis& hyp)
{
  HypDataType hypDataType;
  Hypothesis auxHyp;
  Vector<Score> scoreComponents;
  
      // Obtain score components
  hypDataType=hyp.getData();
  this->incrScore(this->nullHypothesis(),hypDataType,auxHyp,scoreComponents);

  return scoreComponents;
}

//---------------------------------
template<class HYPOTHESIS>
_phraseBasedTransModel<HYPOTHESIS>::~_phraseBasedTransModel()
{
      // Delete pointer to PhraseModelInfo
  delete phrModelInfoPtr;

      // Delete pointer to LangModelInfo
  delete langModelInfoPtr;
}

//-------------------------

#endif
