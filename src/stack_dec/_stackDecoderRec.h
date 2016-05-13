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
/* Module: _stackDecoderRec                                         */
/*                                                                  */
/* Prototypes file: _stackDecoderRec.h                              */
/*                                                                  */
/* Description: Declares the _stackDecoderRec abstract              */
/*              template class, this class is derived from the      */
/*              _stackDecoder class and implements a base class     */
/*              for obtaining stack decoders with recombination.    */
/*                                                                  */
/********************************************************************/

/**
 * @file _stackDecoderRec.h
 *
 * @brief Declares the _stackDecoderRec abstract template class, this
 * class is derived from the _stackDecoder class and implements a base
 * class for obtaining stack decoders with recombination.
 */

#ifndef __stackDecoderRec_h
#define __stackDecoderRec_h

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "_stackDecoder.h"
#include "HypStateDict.h"
#include "WordGraph.h"

//--------------- Constants ------------------------------------------


//--------------- Classes --------------------------------------------

/**
 * @brief The _stackDecoderRec abstract template class is derived from
 * the _stackDecoder class and implements a base class for obtaining
 * stack decoders with recombination.
 */

//--------------- _stackDecoderRec template class

template<class SMT_MODEL>
class _stackDecoderRec: public _stackDecoder<SMT_MODEL>
{
 public:

  typedef typename BaseStackDecoder<SMT_MODEL>::Hypothesis Hypothesis;

      // Constructor. 
  _stackDecoderRec(void); 

      // Function to retrieve word graph ptr
  WordGraph* getWordGraphPtr(void);

      // Function to set word graph ptr
  void setWordGraphPtr(WordGraph* _wordGraphPtr);

      // Functions to parameterize word graphs
  void enableWordGraph(void);
      // Enable word graph
  void disableWordGraph(void);
      // Disable word graph
  void includeScoreCompsInWg(void);
      // Include score componentes in word graph
  void excludeScoreCompsInWg(void);
      // Exclude score componentes in word graph
  unsigned int pruneWordGraph(float threshold);
      // Prune word graph using the given threshold. Returns number of
      // pruned arcs
     
      // Functions to print word graphs
  bool printWordGraph(const char* filename);
  

  void clear(void);
      // Remove all partial hypotheses contained in the stack/s

      // Destructor
  ~_stackDecoderRec();
  
 protected:

  bool wordGraphEnabled;
  bool scoreCompsInWgIncluded;
  WordGraph* wordGraphPtr;
  bool wgPtrOwnedByObject;
  HypStateDict<Hypothesis>* hypStateDictPtr;

  void pushGivenPredHyp(const Hypothesis& pred_hyp,
                        const Vector<Score>& scrComps,
                        const Hypothesis& succ_hyp);
      // Overriden function to allow word-graph generation
  void addArcToWordGraph(Hypothesis pred_hyp,
                         const Vector<Score>& scrComps,
                         Hypothesis succ_hyp);
      // Add an arc to the recombination graph.
  HypStateIndex getHypStateIndex(const Hypothesis& hyp,
                                 bool& existIndex);
      // Obtain hypothesis state index for hyp. If the index exist, the
      // value of existIndex will be set to true, and false otherwise
  bool printHypStateIdxInfo(const char* filename);
  void printHypStateIdxInfo(ostream &outS);
};

//--------------- _stackDecoderRec template class function definitions


//---------------------------------------
template<class SMT_MODEL>
_stackDecoderRec<SMT_MODEL>::_stackDecoderRec(void):_stackDecoder<SMT_MODEL>()
{
  wordGraphPtr=new WordGraph;
  wgPtrOwnedByObject=true;
  hypStateDictPtr=new HypStateDict<Hypothesis>;
  wordGraphEnabled=false;
  scoreCompsInWgIncluded=true;
}

//---------------------------------------
template<class SMT_MODEL>
void _stackDecoderRec<SMT_MODEL>::enableWordGraph(void)
{
  wordGraphEnabled=true;
}

//---------------------------------------
template<class SMT_MODEL>
void _stackDecoderRec<SMT_MODEL>::disableWordGraph(void)
{
  wordGraphEnabled=false;
}

//---------------------------------------
template<class SMT_MODEL>
void _stackDecoderRec<SMT_MODEL>::includeScoreCompsInWg(void)
{
  scoreCompsInWgIncluded=true;
}

//---------------------------------------
template<class SMT_MODEL>
void _stackDecoderRec<SMT_MODEL>::excludeScoreCompsInWg(void)
{
  scoreCompsInWgIncluded=false;
}

//---------------------------------------
template<class SMT_MODEL>
WordGraph* _stackDecoderRec<SMT_MODEL>::getWordGraphPtr(void)
{
  return wordGraphPtr;
}

//---------------------------------------
template<class SMT_MODEL>
void _stackDecoderRec<SMT_MODEL>::setWordGraphPtr(WordGraph* _wordGraphPtr)
{
  if(wgPtrOwnedByObject)
    delete wordGraphPtr;
  
  wordGraphPtr=_wordGraphPtr;
  wgPtrOwnedByObject=false;
}

//---------------------------------------
template<class SMT_MODEL>
unsigned int _stackDecoderRec<SMT_MODEL>::pruneWordGraph(float threshold)
{
      // Prune word graph
  unsigned int numPrunedArcs=wordGraphPtr->prune(threshold);
  return numPrunedArcs;
}

//---------------------------------------
template<class SMT_MODEL>
bool _stackDecoderRec<SMT_MODEL>::printWordGraph(const char* filename)
{
  int ret;

  if(scoreCompsInWgIncluded)
  {
        // Set weights of the components in the wordgraph (this may be
        // misplaced)
    Vector<pair<std::string,float> > compWeights;
    this->smtm_ptr->getWeights(compWeights);
    wordGraphPtr->setCompWeights(compWeights);
  }
      // Print word graph
  std::string filenameWordGraph=filename;
  filenameWordGraph=filenameWordGraph+".wg";
  ret=wordGraphPtr->print(filenameWordGraph.c_str(),true);
      // NOTE: if the second parameter of wordGraphPtr->print() is set to
      // true, only useful states (those that allow us to reach to a
      // final state) are printed
  if(ret==ERROR) return ERROR;
  
      // Print state index info
  std::string filenameHypStateIdx=filename;
  filenameHypStateIdx=filenameHypStateIdx+".idx";
  return printHypStateIdxInfo(filenameHypStateIdx.c_str());
}

//---------------------------------------
template<class SMT_MODEL>
void _stackDecoderRec<SMT_MODEL>::clear(void)
{
  _stackDecoder<SMT_MODEL>::clear();
  hypStateDictPtr->clear();
  wordGraphPtr->clear();
}

//---------------------------------------
template<class SMT_MODEL>
void _stackDecoderRec<SMT_MODEL>::pushGivenPredHyp(const Hypothesis& pred_hyp,
                                                   const Vector<Score>& scrComps,
                                                   const Hypothesis& succ_hyp)

{
  this->push(succ_hyp);
  addArcToWordGraph(pred_hyp,scrComps,succ_hyp);
}

//---------------------------------------
template<class SMT_MODEL>
void _stackDecoderRec<SMT_MODEL>::addArcToWordGraph(Hypothesis pred_hyp,
                                                    const Vector<Score>& scrComps,
                                                    Hypothesis succ_hyp)
{
  if(wordGraphEnabled)
  {
        // Obtain indices for predecessor and successor states, the
        // indices will not exist if the corresponding hypotheses has not
        // been inserted or recombined into the stack
    bool existIndexPred;
    bool existIndexSucc;
    HypStateIndex predStateIndex=getHypStateIndex(pred_hyp,existIndexPred);
    HypStateIndex succStateIndex=getHypStateIndex(succ_hyp,existIndexSucc);

    if(existIndexPred && existIndexSucc)
    {
          // The arc is added only if the hypotheses has been inserted or
          // recombined into the stack

          // Add heuristic to hypotheses
      this->smtm_ptr->addHeuristicToHyp(pred_hyp);
      this->smtm_ptr->addHeuristicToHyp(succ_hyp);
      
          // Set score for the initial state
      if(predStateIndex==INITIAL_STATE)
        wordGraphPtr->setInitialStateScore(pred_hyp.getScore());

          // Add final state if succ_hyp is complete
      bool succStateIndexComplete=this->smtm_ptr->isComplete(succ_hyp);
      if(succStateIndexComplete)
        wordGraphPtr->addFinalState(succStateIndex);
    
          // Obtain the score of the arc
      LgProb arcScore=succ_hyp.getScore()-pred_hyp.getScore();
      
          // Obtain the words associated to the arc
      set<unsigned int> unknownWords;
      Vector<std::string> predPartialTrans=this->smtm_ptr->getTransInPlainTextVec(pred_hyp,unknownWords);
      Vector<std::string> succPartialTrans=this->smtm_ptr->getTransInPlainTextVec(succ_hyp,unknownWords);
      Vector<std::string> words;

      bool unknown=false;
      for(unsigned int i=predPartialTrans.size();i<succPartialTrans.size();++i)
      {
        if(unknownWords.find(i+1)!=unknownWords.end())
          unknown=true;
        words.push_back(succPartialTrans[i]);
      }
      
      SourceSegmentation sourceSegmentation;
      Vector<PositionIndex> targetSegmentCuts;
      Vector<pair<PositionIndex, PositionIndex> > amatrix;
      // Obtain phrase alignment
      this->smtm_ptr->aligMatrix(succ_hyp,amatrix);
      this->smtm_ptr->getPhraseAlignment(amatrix,sourceSegmentation,targetSegmentCuts);
      PositionIndex srcStartIndex=0, srcEndIndex=0;
      if(!sourceSegmentation.empty())
      {
        pair<PositionIndex, PositionIndex> pair=sourceSegmentation.back();
        srcStartIndex=pair.first;
        srcEndIndex=pair.second;
      }

          // Include unweighted score components if requested
          // KNOWN BUG: the unweighted score components cannot be
          //            correctly calculated for those weights
          //            equal to zero, since the expand functions
          //            return weighted components
      if(scoreCompsInWgIncluded)
      {
            // Obtain weights
        Vector<pair<std::string,float> > compWeights;
        this->smtm_ptr->getWeights(compWeights);

            // Obtain components using unitary weights
        Vector<Score> scrCompsUnitary;
        for(unsigned int i=0;i<compWeights.size();++i)
        {
          if(compWeights[i].second!=0)
            scrCompsUnitary.push_back(scrComps[i]/compWeights[i].second);
          else
            scrCompsUnitary.push_back(0);
        }
                
            // Add arc with score components
        wordGraphPtr->addArcWithScrComps(predStateIndex,
                                         succStateIndex,
                                         words,
                                         srcStartIndex,
                                         srcEndIndex,
                                         unknown,
                                         arcScore,
                                         scrCompsUnitary);
      }
      else
      {
            // Add arc
        wordGraphPtr->addArc(predStateIndex,
                             succStateIndex,
                             words,
                             srcStartIndex,
                             srcEndIndex,
                             unknown,
                             arcScore);
      }
    }
  }
}

//--------------------------
template<class SMT_MODEL>
HypStateIndex _stackDecoderRec<SMT_MODEL>::getHypStateIndex(const Hypothesis& hyp,
                                                            bool& existIndex)
{
      // Obtain hypothesis state index for hyp
  typename HypStateDict<Hypothesis>::iterator hypStateDictIter;
  HypStateIndex hypStateIndex;
  typename Hypothesis::HypState hypState;

  hypState=hyp.getHypState();
  hypStateDictIter=hypStateDictPtr->find(hypState);
  if(hypStateDictIter==hypStateDictPtr->end())
  {
    hypStateIndex=0;
    existIndex=false;
    return hypStateIndex;
  }
  else
  {
    hypStateIndex=hypStateDictIter->second.hypStateIndex;
    existIndex=true;
    return hypStateIndex;
  }
}

//---------------------------------------
template<class SMT_MODEL> 
bool _stackDecoderRec<SMT_MODEL>::printHypStateIdxInfo(const char* filename)
{
  ofstream outS;

  outS.open(filename,ios::out);
  if(!outS)
  {
    cerr<<"Error while printing hypothesis state info file."<<endl;
    return ERROR;
  }
  else
  {
    printHypStateIdxInfo(outS);
    outS.close();	
    return OK;
  }
}

//---------------------------------------
template<class SMT_MODEL> 
void _stackDecoderRec<SMT_MODEL>::printHypStateIdxInfo(ostream &outS)
{
  typename HypStateDict<Hypothesis>::iterator hsdIter;

  outS<<"# SOURCE SENTENCE: "<<this->srcSentence<<endl;
  outS<<"#"<<endl;
  for(hsdIter=hypStateDictPtr->begin();hsdIter!=hypStateDictPtr->end();++hsdIter)
  {
    outS<<hsdIter->second.hypStateIndex<<" "<<hsdIter->second.coverage<<" ";
        // Subtract g value if decoder running in breadth-first mode
    if(this->breadthFirst)
    {
      double g=trunc((double)hsdIter->second.score/(double)G_EPSILON);
      outS<<hsdIter->second.score-(g*G_EPSILON);
    }
    else
      outS<<hsdIter->second.score;

    outS<<endl;
  }
}

//---------------------------------------
template<class SMT_MODEL>
_stackDecoderRec<SMT_MODEL>::~_stackDecoderRec(void)
{
  if(wgPtrOwnedByObject)
    delete wordGraphPtr;
  delete hypStateDictPtr;
}

#endif
