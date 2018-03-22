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
 * @file thot_li_weight_upd.cc
 * 
 * @brief Implements a linear interpolation weight updater for
 * phrase-based models given a development corpus.
 */

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include THOT_SMTMODEL_H // Define SmtModel type. It is set in
                         // configure by checking SMTMODEL_H
                         // variable (default value: SmtModel.h)
#include "StdFeatureHandler.h"
#include "_pbTransModel.h"
#include "PhrLocalSwLiTm.h"
#include "ModelDescriptorUtils.h"
#include "DynClassFactoryHandler.h"
#include "ErrorDefs.h"
#include "options.h"
#include <iostream>
#include <fstream>
#include <iomanip>

//--------------- Constants ------------------------------------------

struct thot_liwu_pars
{
  std::string testCorpusFile;
  std::string refCorpusFile;
  std::string phrModelFilePrefix;
  int verbosity;
};

//--------------- Function Declarations ------------------------------

int handleParameters(int argc,
                     char *argv[],
                     thot_liwu_pars& pars);
int takeParameters(int argc,
                   char *argv[],
                   thot_liwu_pars& pars);
int checkParameters(thot_liwu_pars& pars);
bool featureBasedImplIsEnabled(void);
int initPhrModelLegacyImpl(std::string phrModelFilePrefix);
int initPhrModelFeatImpl(std::string phrModelFilePrefix);
void set_default_models(void);
int load_model_features(std::string phrModelFilePrefix);
void releaseMemLegacyImpl(void);
void releaseMemFeatImpl(void);
int update_li_weights_legacy_impl(const thot_liwu_pars& pars);
int update_li_weights_feat_impl(const thot_liwu_pars& pars);
void printUsage(void);
void version(void);

//--------------- Global variables -----------------------------------

DynClassFactoryHandler dynClassFactoryHandler;
PhraseModelInfo* phrModelInfoPtr;
SwModelInfo* swModelInfoPtr;
PhrLocalSwLiTm* phrLocalSwLiTmPtr;

    // Variables related to feature-based implementation
StdFeatureHandler stdFeatureHandler;

//--------------- Function Definitions -------------------------------

//--------------------------------
int main(int argc,char *argv[])
{
  thot_liwu_pars pars;

  if(handleParameters(argc,argv,pars)==THOT_ERROR)
  {
    return THOT_ERROR;
  }
  else
  {
        // Print parameters
    std::cerr<<"-tm option is "<<pars.phrModelFilePrefix<<std::endl;
    std::cerr<<"-t option is "<<pars.testCorpusFile<<std::endl;
    std::cerr<<"-r option is "<<pars.refCorpusFile<<std::endl;
    std::cerr<<"-v option is "<<pars.verbosity<<std::endl;
    
        // Update language model weights
    if(featureBasedImplIsEnabled())
      return update_li_weights_feat_impl(pars);
    else
      return update_li_weights_legacy_impl(pars);
  }
}

//--------------------------------
int handleParameters(int argc,
                     char *argv[],
                     thot_liwu_pars& pars)
{
  if(argc==1 || readOption(argc,argv,"--version")!=-1)
  {
    version();
    return THOT_ERROR;
  }
  if(readOption(argc,argv,"--help")!=-1)
  {
    printUsage();
    return THOT_ERROR;   
  }
  if(takeParameters(argc,argv,pars)==THOT_ERROR)
  {
    return THOT_ERROR;
  }
  else
  {
    if(checkParameters(pars)==THOT_OK)
    {
      return THOT_OK;
    }
    else
    {
      return THOT_ERROR;
    }
  }
}

//--------------------------------
int takeParameters(int argc,
                   char *argv[],
                   thot_liwu_pars& pars)
{
      // Take language model file name
  int err=readSTLstring(argc,argv, "-tm", &pars.phrModelFilePrefix);
  if(err==THOT_ERROR)
    return THOT_ERROR;
  
      // Take language model file name
  err=readSTLstring(argc,argv, "-t", &pars.testCorpusFile);
  if(err==THOT_ERROR)
    return THOT_ERROR;

      // Take language model file name
  err=readSTLstring(argc,argv, "-r", &pars.refCorpusFile);
  if(err==THOT_ERROR)
    return THOT_ERROR;

  if(readOption(argc,argv,"-v")==THOT_OK)
    pars.verbosity=true;
  else
    pars.verbosity=false;
    
  return THOT_OK;
}

//--------------------------------
int checkParameters(thot_liwu_pars& pars)
{  
  if(pars.phrModelFilePrefix.empty())
  {
    std::cerr<<"Error: parameter -tm not given!"<<std::endl;
    return THOT_ERROR;   

  }

  if(pars.testCorpusFile.empty())
  {
    std::cerr<<"Error: parameter -t not given!"<<std::endl;
    return THOT_ERROR;   
  }

  if(pars.refCorpusFile.empty())
  {
    std::cerr<<"Error: parameter -r not given!"<<std::endl;
    return THOT_ERROR;   
  }

  return THOT_OK;
}

//--------------------------
bool featureBasedImplIsEnabled(void)
{
  BasePbTransModel<SmtModel::Hypothesis>* tmpSmtModelPtr=new SmtModel();
  _pbTransModel<SmtModel::Hypothesis>* pbtm_ptr=dynamic_cast<_pbTransModel<SmtModel::Hypothesis>* >(tmpSmtModelPtr);
  if(pbtm_ptr)
  {
    delete tmpSmtModelPtr;
    return true;
  }
  else
  {
    delete tmpSmtModelPtr;
    return false;
  }
}

//--------------------------------
int initPhrModelLegacyImpl(std::string phrModelFilePrefix)
{
      // Show static types
  std::cerr<<"Static types:"<<std::endl;
  std::cerr<<"- SMT model type (SmtModel): "<<SMT_MODEL_TYPE_NAME<<" ("<<THOT_SMTMODEL_H<<")"<<std::endl;
  std::cerr<<"- Language model state (LM_Hist): "<<LM_STATE_TYPE_NAME<<" ("<<THOT_LM_STATE_H<<")"<<std::endl;
  std::cerr<<"- Partial probability information for single word models (PpInfo): "<<PPINFO_TYPE_NAME<<" ("<<THOT_PPINFO_H<<")"<<std::endl;

      // Initialize weight updater
  phrLocalSwLiTmPtr=new PhrLocalSwLiTm;

      // Initialize class factories
  int err=dynClassFactoryHandler.init_smt(THOT_MASTER_INI_PATH);
  if(err==THOT_ERROR)
    return THOT_ERROR;

      // Obtain info about translation model entries
  unsigned int numTransModelEntries;
  std::vector<ModelDescriptorEntry> modelDescEntryVec;
  if(extractModelEntryInfo(phrModelFilePrefix.c_str(),modelDescEntryVec)==THOT_OK)
  {
    numTransModelEntries=modelDescEntryVec.size();
  }
  else
  {
    numTransModelEntries=1;
  }
  
      // Instantiate pointers
  phrModelInfoPtr=new PhraseModelInfo;
  phrModelInfoPtr->invPbModelPtr=dynClassFactoryHandler.basePhraseModelDynClassLoader.make_obj(dynClassFactoryHandler.basePhraseModelInitPars);
  if(phrModelInfoPtr->invPbModelPtr==NULL)
  {
    std::cerr<<"Error: BasePhraseModel pointer could not be instantiated"<<std::endl;
    return THOT_ERROR;
  }

      // Add one swm pointer per each translation model entry
  swModelInfoPtr=new SwModelInfo;
  for(unsigned int i=0;i<numTransModelEntries;++i)
  {
    swModelInfoPtr->swAligModelPtrVec.push_back(dynClassFactoryHandler.baseSwAligModelDynClassLoader.make_obj(dynClassFactoryHandler.baseSwAligModelInitPars));
    if(swModelInfoPtr->swAligModelPtrVec[0]==NULL)
    {
      std::cerr<<"Error: BaseSwAligModel pointer could not be instantiated"<<std::endl;
      return THOT_ERROR;
    }
  }

      // Add one inverse swm pointer per each translation model entry
  for(unsigned int i=0;i<numTransModelEntries;++i)
  {
    swModelInfoPtr->invSwAligModelPtrVec.push_back(dynClassFactoryHandler.baseSwAligModelDynClassLoader.make_obj(dynClassFactoryHandler.baseSwAligModelInitPars));
    if(swModelInfoPtr->invSwAligModelPtrVec[0]==NULL)
    {
      std::cerr<<"Error: BaseSwAligModel pointer could not be instantiated"<<std::endl;
      return THOT_ERROR;
    }
  }

      // Link pointers
  phrLocalSwLiTmPtr->link_pm_info(phrModelInfoPtr);
  phrLocalSwLiTmPtr->link_swm_info(swModelInfoPtr);
  
  return THOT_OK;
}

//--------------------------------
int initPhrModelFeatImpl(std::string phrModelFilePrefix)
{
      // Show static types
  std::cerr<<"Static types:"<<std::endl;
  std::cerr<<"- SMT model type (SmtModel): "<<SMT_MODEL_TYPE_NAME<<" ("<<THOT_SMTMODEL_H<<")"<<std::endl;
  std::cerr<<"- Language model state (LM_Hist): "<<LM_STATE_TYPE_NAME<<" ("<<THOT_LM_STATE_H<<")"<<std::endl;
  std::cerr<<"- Partial probability information for single word models (PpInfo): "<<PPINFO_TYPE_NAME<<" ("<<THOT_PPINFO_H<<")"<<std::endl;

      // Initialize class factories
  int ret=dynClassFactoryHandler.init_smt(THOT_MASTER_INI_PATH);
  if(ret==THOT_ERROR)
    return THOT_ERROR;
  
      // Set default models for feature handler
  set_default_models();
  
      // Load model features
  load_model_features(phrModelFilePrefix);
    
  return THOT_OK;
}

//---------------
void set_default_models(void)
{
  stdFeatureHandler.setDefaultTransSoFile(dynClassFactoryHandler.basePhraseModelSoFileName);
  stdFeatureHandler.setDefaultSingleWordSoFile(dynClassFactoryHandler.baseSwAligModelSoFileName);
}

//---------------
int load_model_features(std::string phrModelFilePrefix)
{
      // Load bilingual log-linear model features
  int verbosity=false;
  int ret=stdFeatureHandler.loadBilingualFeats(phrModelFilePrefix,verbosity);
  if(ret==THOT_ERROR)
    return THOT_ERROR;

  return THOT_OK;
}

//--------------------------------
void releaseMemLegacyImpl(void)
{
  delete phrModelInfoPtr->invPbModelPtr;
  delete phrModelInfoPtr;
  for(unsigned int i=0;i<swModelInfoPtr->swAligModelPtrVec.size();++i)
    delete swModelInfoPtr->swAligModelPtrVec[i];
  for(unsigned int i=0;i<swModelInfoPtr->swAligModelPtrVec.size();++i)
    delete swModelInfoPtr->invSwAligModelPtrVec[i];
  delete swModelInfoPtr;
  delete phrLocalSwLiTmPtr;

  dynClassFactoryHandler.release_smt();
}

//--------------------------------
void releaseMemFeatImpl(void)
{
      // Delete features information
  stdFeatureHandler.clear();
  
      // Release class factory handler
  dynClassFactoryHandler.release_smt();
}

//--------------------------------
int update_li_weights_legacy_impl(const thot_liwu_pars& pars)
{
  int retVal;

      // Initialize phrase model
  retVal=initPhrModelLegacyImpl(pars.phrModelFilePrefix);
  if(retVal==THOT_ERROR)
    return THOT_ERROR;
  
      // Load model
  retVal=phrLocalSwLiTmPtr->loadAligModel(pars.phrModelFilePrefix.c_str());
  if(retVal==THOT_ERROR)
    return THOT_ERROR;
  
      // Update weights
  retVal=phrLocalSwLiTmPtr->updateLinInterpWeights(pars.testCorpusFile,pars.refCorpusFile,pars.verbosity);
  if(retVal==THOT_ERROR)
    return THOT_ERROR;

      // Print updated weights
  retVal=phrLocalSwLiTmPtr->printAligModel(pars.phrModelFilePrefix.c_str());
  if(retVal==THOT_ERROR)
    return THOT_ERROR;

      // Release phrase model
  releaseMemLegacyImpl();

  return THOT_OK;
}

//--------------------------------
int update_li_weights_feat_impl(const thot_liwu_pars& pars)
{
      // Initialize phrase model
  int retVal=initPhrModelFeatImpl(pars.phrModelFilePrefix);
  if(retVal==THOT_ERROR)
    return THOT_ERROR;

      // Update weights
  retVal=stdFeatureHandler.updatePmLinInterpWeights(pars.testCorpusFile,pars.refCorpusFile,pars.verbosity);
  if(retVal==THOT_ERROR)
    return THOT_ERROR;
  
      // Print updated weights
  retVal=stdFeatureHandler.printAligModelLambdas(pars.phrModelFilePrefix);
  if(retVal==THOT_ERROR)
    return THOT_ERROR;
  
      // Release phrase model
  releaseMemFeatImpl();

  return THOT_OK;
}

//--------------------------------
void printUsage(void)
{
  std::cerr<<"thot_li_weight_upd -tm <string> -t <string> -r <string>"<<std::endl;
  std::cerr<<"                   [-v] [--help] [--version]"<<std::endl;
  std::cerr<<std::endl;
  std::cerr<<"-tm <string>       Prefix or descriptor of translation model files."<<std::endl;
  std::cerr<<"                   (Warning: current weights will be overwritten)."<<std::endl;
  std::cerr<<"-t <string>        File with test sentences."<<std::endl;
  std::cerr<<"-r <string>        File with reference sentences."<<std::endl;
  std::cerr<<"-v                 Enable verbose mode."<<std::endl;
  std::cerr<<"--help             Display this help and exit."<<std::endl;
  std::cerr<<"--version          Output version information and exit."<<std::endl;
}

//--------------------------------
void version(void)
{
  std::cerr<<"thot_li_weight_upd is part of the thot package"<<std::endl;
  std::cerr<<"thot version "<<THOT_VERSION<<std::endl;
  std::cerr<<"thot is GNU software written by Daniel Ortiz"<<std::endl;
}
