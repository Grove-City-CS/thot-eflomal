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
/* Module: IncrHmmAligModel                                         */
/*                                                                  */
/* Prototype file: IncrHmmAligModel.h                               */
/*                                                                  */
/* Description: Defines the IncrHmmAligModel class.                 */
/*              IncrHmmAligModel class allows to generate and       */
/*              access to the data of an Hmm statistical            */
/*              alignment model.                                    */
/*                                                                  */
/* Notes: 100% AC-DC powered                                        */
/*                                                                  */
/********************************************************************/

#ifndef _IncrHmmAligModel_h
#define _IncrHmmAligModel_h

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include <MathFuncs.h>
#include "_incrSwAligModel.h"
#include "SwModelsSlmTypes.h"
#include "anjiMatrix.h"
#include "anjm1ip_anjiMatrix.h"
#include "aSourceHmm.h"
#include "HmmAligInfo.h"
#include "DoubleMatrix.h"
#include "IncrLexTable.h"
#include "IncrHmmAligTable.h"
#include "ashPidxPairHashF.h"
#include "LexAuxVar.h"

#if __GNUC__>2
#include <ext/hash_map>
using __gnu_cxx::hash_map;
#else
#include <hash_map>
#endif

//--------------- Constants ------------------------------------------

#define EXP_VAL_LOG_MAX                   -0.01
#define EXP_VAL_LOG_MIN                   -9
#define DEFAULT_ALIG_SMOOTH_INTERP_FACTOR  0.3
#define DEFAULT_LEX_SMOOTH_INTERP_FACTOR   0.1

//--------------- typedefs -------------------------------------------


//--------------- function declarations ------------------------------

//--------------- Classes --------------------------------------------

//--------------- IncrHmmAligModel class

class IncrHmmAligModel: public _incrSwAligModel<Vector<Prob> >
{
  public:

   typedef _incrSwAligModel<Vector<Prob> >::PpInfo PpInfo;
   typedef map<WordIndex,Prob> SrcTableNode;
  
   // Constructor
   IncrHmmAligModel();

   void set_expval_maxnsize(unsigned int _expval_maxnsize);
       // Function to set a maximum size for the matrices of expected
       // values (by default the size is not restricted)

   // Functions to read and add sentence pairs
   unsigned int numSentPairs(void);

   // Functions to train model
   void trainSentPairRange(pair<unsigned int,unsigned int> sentPairRange,
                           int verbosity=0);
       // train model for range [uint,uint]
   void trainAllSents(int verbosity=0);
   void efficientBatchTrainingForRange(pair<unsigned int,unsigned int> sentPairRange,
                                       int verbosity=0);
   pair<double,double> loglikelihoodForPairRange(pair<unsigned int,unsigned int> sentPairRange,
                                                 int verbosity=0);
       // Returns log-likelihood. The first double contains the
       // loglikelihood for all sentences, and the second one, the same
       // loglikelihood normalized by the number of sentences
   void clearInfoAboutSentRange(void);
       // clear info about the whole sentence range without clearing
       // information about current model parameters
   
   // Functions to access model parameters

   void setLexSmIntFactor(double _lexSmoothInterpFactor);
       // Sets lexical smoothing interpolation factor
   Prob pts(WordIndex s,WordIndex t) override;
       // returns p(t|s)
   LgProb logpts(WordIndex s,WordIndex t) override;
       // returns log(p(t|s))

       // alignment model functions
   void setAlSmIntFactor(double _aligSmoothInterpFactor);
       // Sets alignment smoothing interpolation factor
   Prob aProb(PositionIndex prev_i,
              PositionIndex slen,
              PositionIndex i);
       // Returns p(i|prev_i,slen)
   virtual LgProb logaProb(PositionIndex prev_i,
                           PositionIndex slen,
                           PositionIndex i);
       // Returns log(p(i|prev_i,slen))

   // Sentence length model functions
   Prob sentLenProb(unsigned int slen,unsigned int tlen);
       // returns p(tlen|slen)
   LgProb sentLenLgProb(unsigned int slen,unsigned int tlen);

   // Functions to get translations for word
   bool getEntriesForTarget(WordIndex t,
                            SrcTableNode& srctn);
   
   // Functions to generate alignments 
   LgProb obtainBestAlignment(Vector<WordIndex> srcSentIndexVector,
                              Vector<WordIndex> trgSentIndexVector,
                              WordAligMatrix& bestWaMatrix);

   // Functions to calculate probabilities for alignments
   LgProb calcLgProbForAlig(const Vector<WordIndex>& sSent,
                            const Vector<WordIndex>& tSent,
                            WordAligMatrix aligMatrix,
                            int verbose=0);

   // Scoring functions without giving an alignment
   LgProb calcLgProb(const Vector<WordIndex>& sSent,
                     const Vector<WordIndex>& tSent,
                     int verbose=0);
   LgProb calcLgProbPhr(const Vector<WordIndex>& sPhr,
                        const Vector<WordIndex>& tPhr,
                        int verbose=0);
       // Scoring function for phrase pairs

   // Partial scoring functions
   void initPpInfo(unsigned int slen,
                   const Vector<WordIndex>& tSent,
                   PpInfo& ppInfo);
   void partialProbWithoutLen(unsigned int srcPartialLen,
                              unsigned int slen,
                              const Vector<WordIndex>& s_,
                              const Vector<WordIndex>& tSent,
                              PpInfo& ppInfo);
   LgProb lpFromPpInfo(const PpInfo& ppInfo);
   void addHeurForNotAddedWords(int numSrcWordsToBeAdded,
                                const Vector<WordIndex>& tSent,
                                PpInfo& ppInfo);
   void sustHeurForNotAddedWords(int numSrcWordsToBeAdded,
                                 const Vector<WordIndex>& tSent,
                                 PpInfo& ppInfo);

   // load function
   bool load(const char* prefFileName);
   
   // print function
   bool print(const char* prefFileName);

   // clear() function
   void clear(void);

   // clearTempVars() function
   void clearTempVars(void);

   // Destructor
   ~IncrHmmAligModel();

  protected:

   anjiMatrix lanji;
   anjiMatrix lanji_aux;
   anjm1ip_anjiMatrix lanjm1ip_anji;
   anjm1ip_anjiMatrix lanjm1ip_anji_aux;
   DoubleMatrix alpha_values;
   DoubleMatrix beta_values;   
       // Data structures for manipulating expected values

   LexAuxVar lexAuxVar;
   DoubleMatrix cachedLogProbtsDm;
       // EM algorithm auxiliary variables

   typedef hash_map<pair<aSourceHmm,PositionIndex>,pair<float,float>,ashPidxPairHashF> AligAuxVar;
   AligAuxVar aligAuxVar;
   DoubleMatrix cachedLogaProbDm;
       // EM algorithm auxiliary variables

   IncrLexTable incrLexTable;
       // Table with lexical parameters

   IncrHmmAligTable incrHmmAligTable;
       // Table with alignment parameters
   
   CURR_SLM_TYPE sentLengthModel;

   double aligSmoothInterpFactor;
   double lexSmoothInterpFactor;
   
   // Functions to get sentence pairs
   Vector<WordIndex> getSrcSent(unsigned int n);
       // get n-th source sentence
   virtual Vector<WordIndex> extendWithNullWord(const Vector<WordIndex>& srcWordIndexVec);
       // given a vector with source words, returns a extended vector
       // including extra NULL words
   virtual Vector<WordIndex> extendWithNullWordAlig(const Vector<WordIndex>& srcWordIndexVec);
       // the same as the previous one, but it is specific when calculating suff. statistics
       // for the alignment parameters
   PositionIndex getSrcLen(const Vector<WordIndex>& nsrcWordIndexVec);

   Vector<WordIndex> getTrgSent(unsigned int n);   
       // get n-th target sentence

      // Auxiliar functions to load and print models
   bool loadLexSmIntFactor(const char* lexSmIntFactorFile);
   bool printLexSmIntFactor(const char* lexSmIntFactorFile);
   bool loadAlSmIntFactor(const char* alSmIntFactorFile);
   bool printAlSmIntFactor(const char* alSmIntFactorFile);

   // Functions to handle nloglikelihood
   void set_nloglikelihood(unsigned int n,
                           double d);
   double get_nloglikelihood(unsigned int n);

   // Auxiliar scoring functions
   double unsmoothed_logpts(WordIndex s,
                            WordIndex t);
   double cached_logpts(PositionIndex i,
                        PositionIndex j,
                        const Vector<WordIndex>& nsrcSent,
                        const Vector<WordIndex>& trgSent);
       // Returns log(p(t|s)) without smoothing
   virtual double unsmoothed_logaProb(PositionIndex prev_i,
                                      PositionIndex slen,
                                      PositionIndex i);
   double cached_logaProb(PositionIndex prev_i,
                          PositionIndex slen,
                          PositionIndex i,
                          const Vector<WordIndex>& nsrcSent,
                          const Vector<WordIndex>& trgSent);
   void nullAligSpecialPar(unsigned int ip,
                           unsigned int slen,
                           aSourceHmm& asHmm,
                           unsigned int& i);
       // Given ip and slen values, returns (asHmm,i) pair expressing a
       // valid alignment with the null word
   
   void viterbiAlgorithm(const Vector<WordIndex>& nSrcSentIndexVector,
                         const Vector<WordIndex>& trgSentIndexVector,
                         Vector<Vector<LgProb> >& vitMatrix,
                         Vector<Vector<PositionIndex> >& predMatrix);
       // Execute the Viterbi algorithm to obtain the best HMM word
       // alignment
   LgProb bestAligGivenVitMatrices(PositionIndex slen,
                                   const Vector<Vector<LgProb> >& vitMatrix,
                                   const Vector<Vector<PositionIndex> >& predMatrix,
                                   Vector<PositionIndex>& bestAlig);
       // Obtain best alignment vector from Viterbi algorithm matrices
   LgProb forwardAlgorithm(const Vector<WordIndex>& nSrcSentIndexVector,
                           const Vector<WordIndex>& trgSentIndexVector,
                           int verbose=0);
       // Execute Forward algorithm to obtain the log-probability of a
       // sentence pair
   LgProb lgProbGivenForwardMatrix(const Vector<Vector<LgProb> >& forwardMatrix);
   virtual LgProb calcSumIBM1LgProb(const Vector<WordIndex>& sSent,
                                    const Vector<WordIndex>& tSent,
                                    int verbose);
   LgProb logaProbIbm1(PositionIndex slen,
                       PositionIndex tlen);
   LgProb noisyOrLgProb(const Vector<WordIndex>& sSent,
                        const Vector<WordIndex>& tSent,
                        int verbose);

   // EM-related functions
   void calcNewLocalSuffStats(pair<unsigned int,unsigned int> sentPairRange,
                              int verbosity=0);
   void calc_lanji(unsigned int n,
                   const Vector<WordIndex>& nsrcSent,
                   const Vector<WordIndex>& trgSent,
                   const Count& weight);
   void calc_lanjm1ip_anji(unsigned int n,
                           const Vector<WordIndex>& nsrcSent,
                           const Vector<WordIndex>& trgSent,
                           const Count& weight);
   void gatherAligSuffStats(unsigned int n,
                            unsigned int np,
                            const Vector<WordIndex>& nsrcSent,
                            const Vector<WordIndex>& trgSent,
                            const Count& weight);
   bool isFirstNullAligPar(PositionIndex ip,
                           unsigned int slen,
                           PositionIndex i);
   double calc_lanji_num(PositionIndex slen,
                         PositionIndex i,
                         PositionIndex j,
                         const Vector<WordIndex>& nsrcSent,
                         const Vector<WordIndex>& trgSent);
   double calc_lanjm1ip_anji_num_je1(PositionIndex slen,
                                     PositionIndex i,
                                     const Vector<WordIndex>& nsrcSent,
                                     const Vector<WordIndex>& trgSent);
   double calc_lanjm1ip_anji_num_jg1(PositionIndex ip,
                                     PositionIndex slen,
                                     PositionIndex i,
                                     PositionIndex j,
                                     const Vector<WordIndex>& nsrcSent,
                                     const Vector<WordIndex>& trgSent);
   void getHmmAligInfo(PositionIndex ip,
                       unsigned int slen,
                       PositionIndex i,
                       HmmAligInfo& hmmAligInfo);
   bool isValidAlig(PositionIndex ip,
                    unsigned int slen,
                    PositionIndex i);
   bool isNullAlig(PositionIndex ip,
                   unsigned int slen,
                   PositionIndex i);
   PositionIndex getModifiedIp(PositionIndex ip,
                               unsigned int slen,
                               PositionIndex i);
   double log_alpha(PositionIndex slen,
                    PositionIndex i,
                    PositionIndex j,
                    const Vector<WordIndex>& nsrcSent,
                    const Vector<WordIndex>& trgSent);
   double log_alpha_rec(PositionIndex slen,
                        PositionIndex i,
                        PositionIndex j,
                        const Vector<WordIndex>& nsrcSent,
                        const Vector<WordIndex>& trgSent);
   double log_beta(PositionIndex slen,
                   PositionIndex i,
                   PositionIndex j,
                   const Vector<WordIndex>& nsrcSent,
                   const Vector<WordIndex>& trgSent);
   double log_beta_rec(PositionIndex slen,
                       PositionIndex i,
                       PositionIndex j,
                       const Vector<WordIndex>& nsrcSent,
                       const Vector<WordIndex>& trgSent);
   void fillEmAuxVarsLex(unsigned int n,
                         unsigned int np,
                         PositionIndex i,
                         PositionIndex j,
                         const Vector<WordIndex>& nsrcSent,
                         const Vector<WordIndex>& trgSent,
                         const Count& weight);
   void fillEmAuxVarsAlig(unsigned int n,
                          unsigned int np,
                          PositionIndex slen,
                          PositionIndex ip,
                          PositionIndex i,
                          PositionIndex j,
                          const Count& weight);
   void updateParsLex(void);
   void updateParsAlig(void);
   virtual float obtainLogNewSuffStat(float lcurrSuffStat,
                                      float lLocalSuffStatCurr,
                                      float lLocalSuffStatNew);
};

#endif
