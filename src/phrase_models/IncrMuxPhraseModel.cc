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
/* Module: IncrMuxPhraseModel                                       */
/*                                                                  */
/* Definitions file: IncrMuxPhraseModel.cc                          */
/*                                                                  */
/********************************************************************/


//--------------- Include files --------------------------------------

#include "IncrMuxPhraseModel.h"
#ifdef THOT_DISABLE_DYNAMIC_LOADING
#include "IncrPhraseModel.h"
#include "WbaIncrPhraseModel.h"
#endif
#ifdef _WIN32
#include <direct.h>
#endif

//--------------- Global variables -----------------------------------


//--------------- IncrMuxPhraseModel class method definitions

IncrMuxPhraseModel::IncrMuxPhraseModel(void)
{
}

//-------------------------
bool IncrMuxPhraseModel::load(const char *prefix,
                              int verbose/*=0*/)
{
      // Load phrase model entries
  int retval=loadTmEntries(prefix,verbose);
  if(retval==ERROR) return ERROR;

  return OK;
}

//-------------------------
bool IncrMuxPhraseModel::loadTmEntries(const char *fileName,
                                       int verbose)
{
  Vector<ModelDescriptorEntry> modelDescEntryVec;
  if(extractModelEntryInfo(fileName,modelDescEntryVec)==OK)
  {
    for(unsigned int i=0;i<modelDescEntryVec.size();++i)
    {
      if(verbose)
        cerr<<"* Reading TM entry: "<<modelDescEntryVec[i].modelType<<" "<<modelDescEntryVec[i].absolutizedModelFileName<<" "<<modelDescEntryVec[i].statusStr<<endl;
      int ret=loadTmEntry(modelDescEntryVec[i].modelType,
                          modelDescEntryVec[i].absolutizedModelFileName,
                          modelDescEntryVec[i].statusStr);
      if(ret==ERROR)
        return ERROR;
    }
        // Check if main model was found
    if(modelIndex!=MAIN_MUX_PMODEL_INDEX)
    {
      if(verbose)
        cerr<<"Error: the first model entry should be marked as main"<<endl;
      return ERROR;
    }
    else
      return OK;

    return OK;
  }
  else
  {
    if(verbose)
      cerr<<"Error while loading descriptor file"<<endl;
    return ERROR;
  }
}

//-------------------------
bool IncrMuxPhraseModel::loadTmEntry(std::string tmType,
                                     std::string modelFileName,
                                     std::string statusStr)
{
      // Create pointer to model
  BasePhraseModel* tmPtr=createTmPtr(tmType);
  if(tmPtr==NULL)
    return ERROR;
    
      // Store file pointer
  modelPtrVec.push_back(tmPtr);

      // Add global to local map
  GlobalToLocalDataMap gtlDataMap;
  srcGtlDataMapVec.push_back(gtlDataMap);
  trgGtlDataMapVec.push_back(gtlDataMap);

      // Load model from file
  int ret=modelPtrVec.back()->load(modelFileName.c_str());
  if(ret==ERROR) return ERROR;
        
      // Store lm type
  tmTypeVec.push_back(tmType);

      // Store model file name
  modelFileNameVec.push_back(modelFileName);
  
      // Store status
  modelStatusVec.push_back(statusStr);
  
      // Check if model is main
  if(statusStr=="main")
    modelIndex=modelPtrVec.size()-1;
  
  return OK;
}

//-------------------------
bool IncrMuxPhraseModel::print(const char* prefix)
{
  int ret=printTmEntries(prefix);
  if(ret==ERROR) return ERROR;
  return OK;
}

//-------------------------
bool IncrMuxPhraseModel::printTmEntries(const char *fileName)
{
  ofstream outF;

  outF.open(fileName,ios::out);
  if(!outF)
  {
    cerr<<"Error while printing model to file."<<endl;
    return ERROR;
  }
  else
  {
        // Print header
    outF<<"thot tm descriptor"<<endl;

        // Print pm entries
    for(unsigned int i=0;i<tmTypeVec.size();++i)
    {
          // Print descriptor entry
      std::string currModelFileName=obtainFileNameForTmEntry(fileName,i);
      outF<<tmTypeVec[i]<<" "<<currModelFileName<<" "<<modelStatusVec[i]<<endl;

          // Print translation model
      bool ret=printTm(fileName,i);
      if(ret==ERROR)
        return ERROR;
    }
    return OK;
  }
}

//-------------------------
bool IncrMuxPhraseModel::printTm(const char* fileDescName,
                                 unsigned int entry_index)
{
      // Obtain directory name for model entry
  std::string currDirName=obtainDirNameForTmEntry(fileDescName,entry_index);

      // Obtain model file name
  std::string currModelFileName=obtainFileNameForTmEntry(fileDescName,entry_index);

      // Check if directory already exists. Create directory when
      // necessary
  struct stat info;
  if(stat(currDirName.c_str(),&info) != 0)
  {
        // No file or directory with given name exists
        // Create directory
#ifdef _WIN32
    int ret = _mkdir(currDirName.c_str());
#else
    int ret = mkdir(currDirName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
#endif
    if(ret!=0)
    {
      cerr<<"Error while printing model, directory "<<currDirName<<" could not be created."<<endl;
      return ERROR;
    }
  }
  else
  {
    if(info.st_mode & S_IFREG)
    {
          // A file with the same name existed
      cerr<<"Error while printing model, directory "<<currDirName<<" could not be created."<<endl;
      return ERROR;
    }
  }
      // Print model files
  return modelPtrVec[entry_index]->print(currModelFileName.c_str());
}

//---------------
std::string IncrMuxPhraseModel::obtainFileNameForTmEntry(const std::string fileDescName,
                                                         unsigned int entry_index)
{
      // Obtain directory name for model entry
  std::string currDirName=obtainDirNameForTmEntry(fileDescName,entry_index);

      // Obtain model file name
  std::string currModelFileName=currDirName+"/src_trg";

  return currModelFileName;
}

//---------------
std::string IncrMuxPhraseModel::obtainDirNameForTmEntry(const std::string fileDescName,
                                                        unsigned int entry_index)
{
      // Obtain directory name for model entry
  std::string fileDescDirName=extractDirName(fileDescName);

      // Obtain directory name
  std::string currDirName=fileDescDirName+"/"+modelStatusVec[entry_index];

  return currDirName;
}

//-------------------------
BasePhraseModel* IncrMuxPhraseModel::createTmPtr(std::string tmType)
{
#ifdef THOT_DISABLE_DYNAMIC_LOADING
  BasePhraseModel* tmPtr = NULL;
  if (tmType == "IncrPhraseModel")
    tmPtr = new IncrPhraseModel;
  else if (tmType == "WbaIncrPhraseModel")
    tmPtr = new WbaIncrPhraseModel;
  return tmPtr;
#else
  SimpleDynClassLoaderMap::iterator iter=simpleDynClassLoaderMap.find(tmType);
  if(iter!=simpleDynClassLoaderMap.end())
  {
    return iter->second.make_obj("");
  }
  else
  {
        // Declare dynamic class loader instance
    SimpleDynClassLoader<BasePhraseModel> simpleDynClassLoader;
  
        // Open module
    bool verbosity=false;
    if(!simpleDynClassLoader.open_module(tmType,verbosity))
    {
      cerr<<"Error: so file ("<<tmType<<") could not be opened"<<endl;
      return NULL;
    }

        // Create tm file pointer
    BasePhraseModel* tmPtr=simpleDynClassLoader.make_obj("");
    if(tmPtr==NULL)
    {
      cerr<<"Error: BasePhraseModel pointer could not be instantiated"<<endl;
      simpleDynClassLoader.close_module();
    
      return NULL;
    }
        // Store class loader in map
    simpleDynClassLoaderMap.insert(std::make_pair(tmType,simpleDynClassLoader));
    
    return tmPtr;
  }
#endif
}

//-------------------------
void IncrMuxPhraseModel::deleteModelPointers(void)
{
  for(unsigned int i=0;i<modelPtrVec.size();++i)
  {
    delete modelPtrVec[i];
  }
  modelPtrVec.clear();
}

#ifndef THOT_DISABLE_DYNAMIC_LOADING
//-------------------------
void IncrMuxPhraseModel::closeDynamicModules(void)
{
  SimpleDynClassLoaderMap::iterator iter;
  for(iter=simpleDynClassLoaderMap.begin();iter!=simpleDynClassLoaderMap.end();++iter)
    iter->second.close_module(false);
  simpleDynClassLoaderMap.clear();
}
#endif

//-------------------------
void IncrMuxPhraseModel::clear(void)
{
  srcGtlDataMapVec.clear();
  trgGtlDataMapVec.clear();
  modelIndex=INVALID_MUX_PMODEL_INDEX;
  swVocPtr->clear();
  tmTypeVec.clear();
  modelFileNameVec.clear();
  modelStatusVec.clear();
  deleteModelPointers();
#ifndef THOT_DISABLE_DYNAMIC_LOADING
  closeDynamicModules();
#endif
}

//-------------------------
std::string IncrMuxPhraseModel::getModelPrefixFileName(int idx)
{
  if(idx<(int)modelPtrVec.size())
    return modelFileNameVec[idx];
  else
    return "";
}

//-------------------------
std::string IncrMuxPhraseModel::getModelStatus(int idx)
{
  if(idx<(int)modelPtrVec.size())
    return modelStatusVec[idx];
  else
    return "";
}

//-------------------------
IncrMuxPhraseModel::~IncrMuxPhraseModel()
{
  clear();
}

//-------------------------
