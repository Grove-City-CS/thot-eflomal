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

/**
 * @file FeatureHandler.h
 * 
 * @brief Class to handle log-linear model features.
 */

#ifndef _FeatureHandler_h
#define _FeatureHandler_h

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "OnlineTrainingPars.h"
#include "WeightUpdateUtils.h"
#include THOT_SMTMODEL_H // Define SmtModel type. It is set in
                         // configure by checking SMTMODEL_H
                         // variable (default value: SmtModel.h)
#include "SmtModelUtils.h"
#include "SrcPosJumpFeat.h"
#include "TrgPhraseLenFeat.h"
#include "SrcPhraseLenFeat.h"
#include "InversePhraseModelFeat.h"
#include "DirectPhraseModelFeat.h"
#include "_incrJelMerNgramLM.h"
#include "LangModelFeat.h"
#include "WordPenaltyFeat.h"
#include "FeaturesInfo.h"
#include "SwModelsInfo.h"
#include "PhraseModelsInfo.h"
#include "LangModelsInfo.h"
#include "WpModelInfo.h"
#include "StrProcUtils.h"

//--------------- FeatureHandler class

class FeatureHandler
{
 public:

      // Constructor
  FeatureHandler();
  
      // Functions to incorporate features
  int addWpFeat(int verbose);
  int addLmFeats(std::string lmFileName,
                 int verbose);
  int addTmFeats(std::string tmFilesPrefix,
                 int verbose);

      // Functions to print features
  bool print(std::string tmFileName,
             std::string lmFileName,
             int verbose=0);
  bool printAligModels(std::string tmFileName,
                       int verbose=0);
  bool printAligModelLambdas(std::string tmFileName,
                             int verbose=0);
  bool printLangModels(std::string lmFileName,
                       int verbose=0);

      // Functions to specify default model types
  int setWordPenSoFile(std::string soFileName);
  int setDefaultLangSoFile(std::string soFileName);
  int setDefaultTransSoFile(std::string soFileName);
  int setDefaultSingleWordSoFile(std::string soFileName);

      // Function to get pointers to features
  FeaturesInfo<SmtModel::HypScoreInfo>* getFeatureInfoPtr(void);
  
      // Functions to adjust weights
  int updateLmLinInterpWeights(std::string trgCorpusFileName,
                               int verbose=0);
  int updatePmLinInterpWeights(std::string srcCorpusFileName,
                               std::string trgCorpusFileName,
                               int verbose=0);

      // Functions for online training of features
  int onlineTrainFeats(OnlineTrainingPars onlineTrainingPars,
                       std::string srcSent,
                       std::string refSent,
                       std::string sysSent,
                       int verbose=0);

      // Functions to train word predictor
  void trainWordPred(std::vector<std::string> strVec);
       
      // Clear function
  void clear(void);

      // Destructor
  ~FeatureHandler();
  
 private:

      // Default model types
  std::string wpSoFile;
  std::string defaultLangSoFile;
  std::string defaultTransSoFile;
  std::string defaultSingleWordSoFile;

      // Model information
  SwModelsInfo swModelsInfo;
  PhraseModelsInfo phraseModelsInfo;
  LangModelsInfo langModelsInfo;
  WpModelInfo wpModelInfo;
  FeaturesInfo<SmtModel::HypScoreInfo> featuresInfo;

      // Training-related data members
  std::vector<std::vector<PhrasePair> > vecVecInvPhPair;

      // Auxiliary functions

      // Language model-related functions
  BaseNgramLM<LM_State>* createLmPtr(std::string soFileName);
  int createLangModelFeat(std::string featName,
                          const ModelDescriptorEntry& modelDescEntry,
                          WordPredictor* wordPredPtr,
                          LangModelFeat<SmtModel::HypScoreInfo>** langModelFeatPtrRef);
  bool process_lm_descriptor(std::string lmDescFile,
                             int verbose=0);
  bool process_lm_files_prefix(std::string lmFilesPrefix,
                               int verbose=0);
  bool loadWordPredInfo(std::string lmFilesPrefix);

      // Phrase model-related functions
  BasePhraseModel* createPmPtr(std::string soFileName);
  unsigned int getFeatureIdx(std::string featName);
  DirectPhraseModelFeat<SmtModel::HypScoreInfo>* getDirectPhraseModelFeatPtr(std::string directPhrModelFeatName);
  InversePhraseModelFeat<SmtModel::HypScoreInfo>* getInversePhraseModelFeatPtr(std::string invPhrModelFeatName);
  int createDirectPhrModelFeat(std::string featName,
                               const ModelDescriptorEntry& modelDescEntry,
                               DirectPhraseModelFeat<SmtModel::HypScoreInfo>** dirPmFeatPtrRef);
  int createInversePhrModelFeat(std::string featName,
                                const ModelDescriptorEntry& modelDescEntry,
                                BasePhraseModel* invPbModelPtr,
                                InversePhraseModelFeat<SmtModel::HypScoreInfo>** invPmFeatPtrRef);
  int createSrcPhraseLenFeat(std::string featName,
                             BasePhraseModel* basePhraseModelPtr,
                             SrcPhraseLenFeat<SmtModel::HypScoreInfo>** srcPhraseLenFeatRef);
  int createTrgPhraseLenFeat(std::string featName,
                             BasePhraseModel* basePhraseModelPtr,
                             TrgPhraseLenFeat<SmtModel::HypScoreInfo>** trgPhraseLenFeatRef);
  int createSrcPosJumpFeat(std::string featName,
                           BasePhraseModel* basePhraseModelPtr,
                           SrcPosJumpFeat<SmtModel::HypScoreInfo>** srcPosJumpFeatRef);
  bool process_tm_descriptor(std::string tmDescFile,
                             int verbose=0);
  bool process_tm_files_prefix(std::string tmFilesPrefix,
                               int verbose=0);

      // Weight updating related functions
  int updateLinWeightsForLmPtr(BaseNgramLM<LM_State>* lModelPtr,
                               std::string trgCorpusFileName,
                               int verbose=0);

      // Functions for performing on-line training
  int incrTrainFeatsSentPair(OnlineTrainingPars onlineTraininingPars,
                             std::string srcSent,
                             std::string refSent,
                             int verbose=0);
  int trainLangModel(BaseNgramLM<LM_State>* lModelPtr,
                     float learnStepSize,
                     std::vector<std::string> refSentStrVec,
                     int verbose=0);
  int trainAligModel(BasePhraseModel* invPbModelPtr,
                     BaseSwAligModel<PpInfo>* swAligModelPtr,
                     BaseSwAligModel<PpInfo>* invSwAligModelPtr,
                     OnlineTrainingPars onlineTrainingPars,
                     std::vector<std::string> srcSentStrVec,
                     std::vector<std::string> refSentStrVec,
                     int verbose=0);
  void updateAligModelsSrcVoc(BasePhraseModel* invPbModelPtr,
                              BaseSwAligModel<PpInfo>* swAligModelPtr,
                              BaseSwAligModel<PpInfo>* invSwAligModelPtr,
                              std::vector<std::string> srcSentStrVec);
  void updateAligModelsTrgVoc(BasePhraseModel* invPbModelPtr,
                              BaseSwAligModel<PpInfo>* swAligModelPtr,
                              BaseSwAligModel<PpInfo>* invSwAligModelPtr,
                              std::vector<std::string> trgSentStrVec);
  WordIndex addSrcSymbolToAligModels(BasePhraseModel* invPbModelPtr,
                                     BaseSwAligModel<PpInfo>* swAligModelPtr,
                                     BaseSwAligModel<PpInfo>* invSwAligModelPtr,
                                     std::string s);
  WordIndex addTrgSymbolToAligModels(BasePhraseModel* invPbModelPtr,
                                     BaseSwAligModel<PpInfo>* swAligModelPtr,
                                     BaseSwAligModel<PpInfo>* invSwAligModelPtr,
                                     std::string t);
  int addNewTransOpts(BasePhraseModel* invPbModelPtr,
                      BaseSwAligModel<PpInfo>* swAligModelPtr,
                      BaseSwAligModel<PpInfo>* invSwAligModelPtr,
                      int n,
                      int verbose=0);
  unsigned int map_n_am_suff_stats(unsigned int n);

      // Auxiliary functions to print models
  bool printLambdasForModel(std::string modelFileName,
                            std::string featName,
                            std::string invFeatName,
                            int verbose=0);
       
      // Memory management functions
  void deleteWpModelPtr(void);
  void deleteLangModelPtrs(void);
  void deletePhrModelPtrs(void);
  void deleteSwModelPtrs(void);
};
  
#endif
