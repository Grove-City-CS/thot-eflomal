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
/* Module: thot_ms_dec                                              */
/*                                                                  */
/* Definitions file: thot_ms_dec.cc                                 */
/*                                                                  */
/* Description: Implements a translation system which translates a  */
/*              test corpus using a multiple-stack decoder          */
/*                                                                  */
/********************************************************************/


//--------------- Include files --------------------------------------

#include "_stackDecoderRec.h"
#include "BaseStackDecoder.h"
#include "SmtModelTypes.h"
#include "BasePbTransModel.h"
#include "_phrSwTransModel.h"
#include "_phraseBasedTransModel.h"
#include "StackDecSwModelTypes.h"
#include "StackDecPbModelTypes.h"
#include "StackDecLmTypes.h"
#include "SwModelInfo.h"
#include "PhraseModelInfo.h"
#include "LangModelInfo.h"
#include "WordPenaltyModel.h"
#include "KbMiraLlWu.h"
#include "MultiStackTypes.h"

#include "ctimer.h"
#include "options.h"
#include "ErrorDefs.h"
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <vector>
#include <string>
#include <map>
#include <set>

using namespace std;

//--------------- Constants ------------------------------------------

#define PMSTACK_W_DEFAULT 10

#ifdef MULTI_STACK_USE_GRAN
 #define PMSTACK_S_DEFAULT 128
#else
 #define PMSTACK_S_DEFAULT 10
#endif

#define PMSTACK_A_DEFAULT 10
#define PMSTACK_I_DEFAULT 1
#define PMSTACK_G_DEFAULT 0
#define PMSTACK_H_DEFAULT LOCAL_TD_HEURISTIC
#define PMSTACK_NOMON_DEFAULT 0

//--------------- Type definitions -----------------------------------

struct thot_ms_dec_pars
{
  bool be;
  float W;
  int A,nomon,S,I,G,heuristic,verbosity;
  std::string sourceSentencesFile;
  std::string languageModelFileName;
  std::string transModelPref;
  std::string wordGraphFileName;
  std::string outFile;
  float wgPruningThreshold;
  Vector<float> weightVec;

  thot_ms_dec_pars()
    {
      W=PMSTACK_W_DEFAULT;
      S=PMSTACK_S_DEFAULT;
      A=PMSTACK_A_DEFAULT;
      nomon=PMSTACK_NOMON_DEFAULT;
      I=PMSTACK_I_DEFAULT;
      G=PMSTACK_G_DEFAULT;
      heuristic=PMSTACK_H_DEFAULT;
      be=0;
      wgPruningThreshold=DISABLE_WORDGRAPH;
      wgPruningThreshold=UNLIMITED_DENSITY;
      verbosity=0;
    }
};

//--------------- Function Declarations ------------------------------

int init_translator(const thot_ms_dec_pars& tdp);
void release_translator(void);
int translate_corpus(const thot_ms_dec_pars& tdp);
Vector<string> stringToStringVector(string s);
void version(void);
int handleParameters(int argc,
                     char *argv[],
                     thot_ms_dec_pars& pars);
int takeParameters(int argc,
                   char *argv[],
                   thot_ms_dec_pars& tdp);
int takeParametersFromCfgFile(std::string cfgFileName,
                              thot_ms_dec_pars& tdp);
void takeParametersGivenArgcArgv(int argc,
                                 char *argv[],
                                 thot_ms_dec_pars& tdp);
int checkParameters(const thot_ms_dec_pars& tdp);
void printParameters(const thot_ms_dec_pars& tdp);
void printUsage(void);
void printConfig(void);

//--------------- Global variables -----------------------------------

LangModelInfo* langModelInfoPtr;
PhraseModelInfo* phrModelInfoPtr;
SwModelInfo* swModelInfoPtr;
BaseLogLinWeightUpdater* llWeightUpdaterPtr;
BasePbTransModel<CURR_MODEL_TYPE::Hypothesis>* smtModelPtr;
BaseStackDecoder<CURR_MODEL_TYPE>* stackDecoderPtr;
_stackDecoderRec<CURR_MODEL_TYPE>* stackDecoderRecPtr;

//--------------- Function Definitions -------------------------------

//--------------- main function
int main(int argc, char *argv[])
{
      // Take and check parameters
  thot_ms_dec_pars tdp;
  if(handleParameters(argc,argv,tdp)==ERROR)
  {
    return ERROR;
  }
  else
  {
        // init translator    
    if(init_translator(tdp)==ERROR)
    {      
      cerr<<"Error during the initialization of the translator"<<endl;
      return ERROR;
    }
    else
    {
      int ret=translate_corpus(tdp);
      release_translator();
      if(ret==ERROR) return ERROR;
      else return OK;
    }
  }
}

//---------------
int init_translator(const thot_ms_dec_pars& tdp)
{
  int err;
  
  cerr<<"\n- Initializing model and test corpus...\n\n";

  langModelInfoPtr=new LangModelInfo;
  langModelInfoPtr->lModelPtr=new THOT_CURR_LM_TYPE;
  phrModelInfoPtr=new PhraseModelInfo;
  phrModelInfoPtr->invPbModelPtr=new THOT_CURR_PBM_TYPE;
  swModelInfoPtr=new SwModelInfo;
  swModelInfoPtr->swAligModelPtr=new CURR_SWM_TYPE;
  swModelInfoPtr->invSwAligModelPtr=new CURR_SWM_TYPE;
  langModelInfoPtr->wpModelPtr=new WordPenaltyModel;
  llWeightUpdaterPtr=new KbMiraLlWu;
      // Instantiate smt model
  smtModelPtr=new CURR_MODEL_TYPE();
      // Link pointers
  smtModelPtr->link_ll_weight_upd(llWeightUpdaterPtr);
  _phraseBasedTransModel<CURR_MODEL_TYPE::Hypothesis>* base_pbtm_ptr=dynamic_cast<_phraseBasedTransModel<CURR_MODEL_TYPE::Hypothesis>* >(smtModelPtr);
  if(base_pbtm_ptr)
  {
    base_pbtm_ptr->link_lm_info(langModelInfoPtr);
    base_pbtm_ptr->link_pm_info(phrModelInfoPtr);
  }
  _phrSwTransModel<CURR_MODEL_TYPE::Hypothesis>* base_pbswtm_ptr=dynamic_cast<_phrSwTransModel<CURR_MODEL_TYPE::Hypothesis>* >(smtModelPtr);
  if(base_pbswtm_ptr)
  {
    base_pbswtm_ptr->link_swm_info(swModelInfoPtr);
  }
  
  err=smtModelPtr->loadLangModel(tdp.languageModelFileName.c_str());
  if(err==ERROR)
  {
    release_translator();
    return ERROR;
  }

  err=smtModelPtr->loadAligModel(tdp.transModelPref.c_str());
  if(err==ERROR)
  {
    release_translator();
    return ERROR;
  }

      // Set heuristic
  smtModelPtr->setHeuristic(tdp.heuristic);

      // Set weights
  smtModelPtr->setWeights(tdp.weightVec);
  smtModelPtr->printWeights(cerr);
  cerr<<endl;

      // Set model parameters
  smtModelPtr->set_W_par(tdp.W);
  smtModelPtr->set_A_par(tdp.A);
  smtModelPtr->set_U_par(tdp.nomon);

      // Set verbosity
  smtModelPtr->setVerbosity(tdp.verbosity);
    
      // Create a translator instance
  stackDecoderPtr=new CURR_MSTACK_TYPE<CURR_MODEL_TYPE>();

      // Determine if the translator incorporates hypotheses recombination
  stackDecoderRecPtr=dynamic_cast<_stackDecoderRec<CURR_MODEL_TYPE>*>(stackDecoderPtr);

      // Link translation model
  stackDecoderPtr->link_smt_model(smtModelPtr);
    
      // Set translator parameters
  stackDecoderPtr->set_S_par(tdp.S);
  stackDecoderPtr->set_I_par(tdp.I);
  stackDecoderPtr->set_G_par(tdp.G);

      // Enable best score pruning if the decoder is not going to obtain
      // n-best translations or word-graphs
  if(tdp.wgPruningThreshold==DISABLE_WORDGRAPH)
    stackDecoderPtr->useBestScorePruning(true);

      // Set breadthFirst flag
  stackDecoderPtr->set_breadthFirst(!tdp.be);

  if(stackDecoderRecPtr)
  {
        // Enable word graph according to wgPruningThreshold
    if(tdp.wordGraphFileName!="")
    {
      if(tdp.wgPruningThreshold!=DISABLE_WORDGRAPH)
        stackDecoderRecPtr->enableWordGraph();
    }
  }
      // Set translator verbosity
  stackDecoderPtr->setVerbosity(tdp.verbosity);

  return OK;
}

//---------------
void release_translator(void)
{
  delete langModelInfoPtr->lModelPtr;
  delete langModelInfoPtr->wpModelPtr;
  delete langModelInfoPtr;
  delete phrModelInfoPtr->invPbModelPtr;
  delete phrModelInfoPtr;
  delete swModelInfoPtr->swAligModelPtr;
  delete swModelInfoPtr->invSwAligModelPtr;
  delete swModelInfoPtr;
  delete stackDecoderPtr;
  delete llWeightUpdaterPtr;
  delete smtModelPtr;
}

//---------------
int translate_corpus(const thot_ms_dec_pars& tdp)
{
  CURR_MODEL_TYPE::Hypothesis result;     // Results of the translation
  CURR_MODEL_TYPE::Hypothesis anotherTrans;     // Another results of the translation
  int sentNo=0;    
  double elapsed_ant,elapsed,ucpu,scpu,total_time=0;
      
  ifstream testCorpusFile;                // Test corpus file stream
  string srcSentenceString,s;
  
    
      // Open test corpus file
  testCorpusFile.open(tdp.sourceSentencesFile.c_str());    
  testCorpusFile.seekg(0, ios::beg);

  cerr<<"\n- Translating test corpus sentences...\n\n";

  if(!testCorpusFile)
  {
    cerr<<"Test corpus error!"<<endl;
    return ERROR;
  }
  else
  {
        // Open output file if required
    ofstream outS;
    if(!tdp.outFile.empty())
    {
      outS.open(tdp.outFile.c_str(),ios::out);
      if(!outS) cerr<<"Error while opening output file."<<endl;
    }
    
        // Translate corpus sentences
    while(!testCorpusFile.eof())
    {
      getline(testCorpusFile,srcSentenceString);

          // Discard last sentence if it is empty
      if(srcSentenceString=="" && testCorpusFile.eof())
        break;
      
      ++sentNo;
        
      if(tdp.verbosity)
      {
        cerr<<sentNo<<endl<<srcSentenceString<<endl;
        ctimer(&elapsed_ant,&ucpu,&scpu);
      }
       
          //------- Translate sentence
      result=stackDecoderPtr->translate(srcSentenceString);

          //--------------------------
      if(tdp.verbosity) ctimer(&elapsed,&ucpu,&scpu);

      if(tdp.outFile.empty())
        cout<<smtModelPtr->getTransInPlainText(result)<<endl;
      else
        outS<<smtModelPtr->getTransInPlainText(result)<<endl;
          
      if(tdp.verbosity)
      {
        smtModelPtr->printHyp(result,cerr,tdp.verbosity);
#         ifdef THOT_STATS
        stackDecoderPtr->printStats();
#         endif

        cerr<<"- Elapsed Time: "<<elapsed-elapsed_ant<<endl<<endl;
        total_time+=elapsed-elapsed_ant;
      }

      if(stackDecoderRecPtr)
      {
            // Print wordgraph if the -wg option was given
        if(tdp.wordGraphFileName!="")
        {
          char wgFileNameForSent[256];
          sprintf(wgFileNameForSent,"%s_%06d",tdp.wordGraphFileName.c_str(),sentNo);
          stackDecoderRecPtr->pruneWordGraph(tdp.wgPruningThreshold);
          stackDecoderRecPtr->printWordGraph(wgFileNameForSent);
        }
      }

#ifdef THOT_ENABLE_GRAPH
      char printGraphFileName[256];
      ofstream graphOutS;
      sprintf(printGraphFileName,"sent%d.graph_file",sentNo);
      graphOutS.open(printGraphFileName,ios::out);
      if(!graphOutS) cerr<<"Error while printing search graph to file."<<endl;
      else
      {
        stackDecoderPtr->printSearchGraphStream(graphOutS);
        graphOutS<<"Stack ID. Out\n";
        stackDecoderPtr->printGraphForHyp(result,graphOutS);
        graphOutS.close();        
      }
#endif        
    }
        // Close output file
    if(!tdp.outFile.empty())
    {
      outS.close();
    }

        // Close test corpus file
    testCorpusFile.close();
  }

  if(tdp.verbosity)
  {
    cerr<<"- Time per sentence: "<<total_time/sentNo<<endl;
  }

  return OK;
}

//---------------
int handleParameters(int argc,
                     char *argv[],
                     thot_ms_dec_pars& tdp)
{
  if(argc==1 || readOption(argc,argv,"--version")!=-1)
  {
    version();
    return ERROR;
  }
  if(readOption(argc,argv,"--help")!=-1)
  {
    printUsage();
    return ERROR;   
  }
  if(readOption(argc,argv,"--config")!=-1)
  {
    printConfig();
    return ERROR;   
  }
  if(takeParameters(argc,argv,tdp)==ERROR)
  {
    return ERROR;
  }
  else
  {
    if(checkParameters(tdp)==OK)
    {
      printParameters(tdp);
      return OK;
    }
    else
    {
      return ERROR;
    }
  }
}

//---------------
int takeParameters(int argc,
                   char *argv[],
                   thot_ms_dec_pars& tdp)
{
      // Check if a configuration file was provided
  std::string cfgFileName;
  int err=readSTLstring(argc,argv, "-c", &cfgFileName);
  if(!err)
  {
        // Process configuration file
    err=takeParametersFromCfgFile(cfgFileName,tdp);
    if(err==ERROR) return ERROR;
  }
      // process command line parameters
  takeParametersGivenArgcArgv(argc,argv,tdp);
  return OK;
}

//---------------
int takeParametersFromCfgFile(std::string cfgFileName,
                              thot_ms_dec_pars& tdp)
{
      // Extract parameters from configuration file
    std::string comment="#";
    int cfgFileArgc;
    Vector<std::string> cfgFileArgvStl;
    int ret=extractParsFromFile(cfgFileName.c_str(),cfgFileArgc,cfgFileArgvStl,comment);
    if(ret==ERROR) return ERROR;

        // Create argv for cfg file
    char** cfgFileArgv=(char**) malloc(cfgFileArgc*sizeof(char*));
    for(unsigned int i=0;i<cfgFileArgvStl.size();++i)
    {
      cfgFileArgv[i]=(char*) malloc((cfgFileArgvStl[i].size()+1)*sizeof(char));
      strcpy(cfgFileArgv[i],cfgFileArgvStl[i].c_str());
    }
        // Process extracted parameters
    takeParametersGivenArgcArgv(cfgFileArgc,cfgFileArgv,tdp);

        // Release allocated memory
    for(unsigned int i=0;i<cfgFileArgvStl.size();++i)
    {
      free(cfgFileArgv[i]);
    }
    free(cfgFileArgv);

        // Return without error
    return OK;
}

//---------------
void takeParametersGivenArgcArgv(int argc,
                                 char *argv[],
                                 thot_ms_dec_pars& tdp)
{
     // Takes W 
 int err=readFloat(argc,argv, "-W", &tdp.W);

     // Takes S parameter 
 err=readInt(argc,argv, "-S", &tdp.S);

     // Takes A parameter 
 err=readInt(argc,argv, "-A", &tdp.A);

     // Takes U parameter 
 err=readInt(argc,argv, "-nomon", &tdp.nomon);

     // Takes I parameter 
 err=readInt(argc,argv, "-I", &tdp.I);

     // Takes I parameter 
 err=readInt(argc,argv, "-G", &tdp.G);

     // Takes h parameter 
 err=readInt(argc,argv, "-h", &tdp.heuristic);

     // Take language model file name
 err=readSTLstring(argc,argv, "-lm", &tdp.languageModelFileName);

     // Take read table prefix 
 err=readSTLstring(argc,argv, "-tm", &tdp.transModelPref);
 
     // Take file name with the sentences to be translated 
 err=readSTLstring(argc,argv, "-t",&tdp.sourceSentencesFile);

      // Take output file name
 err=readSTLstring(argc,argv, "-o",&tdp.outFile);
 
       // read -be option
 err=readOption(argc,argv,"-be");
 if(err!=-1)
 {
   tdp.be=1;
 }      
     
     // Take -tmw parameter
 err=readFloatSeq(argc,argv, "-tmw", tdp.weightVec);

     // Take -wg parameter
 err=readSTLstring(argc,argv, "-wg", &tdp.wordGraphFileName);
 if(err!=-1)
 {
       // Take -wgp parameter 
   err=readFloat(argc,argv, "-wgp", &tdp.wgPruningThreshold);
 }

     // Take verbosity parameter
 err=readOption(argc,argv,"-v");
 if(err==-1)
 {
       // -v not found
   err=readOption(argc,argv,"-v1");
   if(err==-1)
   {
         // -v1 not found
     err=readOption(argc,argv,"-v2");
     if(err==-1)
     {
           // -v2 not found
       tdp.verbosity=0;
     }
     else
     {
           // -v2 found
       tdp.verbosity=3;
     }
   }
   else
   {
         // -v1 found
     tdp.verbosity=2;
   }
 }
 else
 {
       // -v found
   tdp.verbosity=1;
 }
}

//---------------
int checkParameters(const thot_ms_dec_pars& tdp)
{
  if(tdp.languageModelFileName.empty())
  {
    cerr<<"Error: parameter -lm not given!"<<endl;
    return ERROR;   
  }
  
  if(tdp.transModelPref.empty())
  {
    cerr<<"Error: parameter -tm not given!"<<endl;
    return ERROR;   
  }

  if(tdp.sourceSentencesFile.empty())
  {
    cerr<<"Error: parameter -t not given!"<<endl;
    return ERROR;   
  }
  
  return OK;
}

//---------------
void printParameters(const thot_ms_dec_pars& tdp)
{
 cerr<<"W: "<<tdp.W<<endl;   
 cerr<<"S: "<<tdp.S<<endl;   
 cerr<<"A: "<<tdp.A<<endl;
 cerr<<"I: "<<tdp.I<<endl;
#ifdef MULTI_STACK_USE_GRAN
 cerr<<"G: "<<tdp.G<<endl;
#endif
 cerr<<"h: "<<tdp.heuristic<<endl;
 cerr<<"be: "<<tdp.be<<endl;
 cerr<<"nomon: "<<tdp.nomon<<endl;
 cerr<<"weight vector:";
 for(unsigned int i=0;i<tdp.weightVec.size();++i)
   cerr<<" "<<tdp.weightVec[i];
 cerr<<endl;
 cerr<<"lmfile: "<<tdp.languageModelFileName<<endl;   
 cerr<<"tm files prefix: "<<tdp.transModelPref<<endl;
 cerr<<"test file: "<<tdp.sourceSentencesFile<<endl;
 if(tdp.wordGraphFileName!="")
 {
   cerr<<"word graph file prefix: "<<tdp.wordGraphFileName<<endl;
   if(tdp.wgPruningThreshold==UNLIMITED_DENSITY)
     cerr<<"word graph pruning threshold: word graph density unrestricted"<<endl;
   else
     cerr<<"word graph pruning threshold: "<<tdp.wgPruningThreshold<<endl;
 }
 else
 {
   cerr<<"word graph file prefix not given (wordgraphs will not be generated)"<<endl;
 }
 cerr<<"verbosity level: "<<tdp.verbosity<<endl;
}

//---------------
Vector<string> stringToStringVector(string s)
{
 Vector<string> vs;	
 string aux="";
 unsigned int i;	

 for(i=0;i<s.size();++i)
    {
	 if(s[i]!=' ') aux+=s[i];
         else if(aux!="") {vs.push_back(aux); aux="";}		 
	}
 
 if(aux!="") vs.push_back(aux); 	
 return vs;	
}

//---------------
void printConfig(void)
{
  cerr <<"* Translator configuration:"<<endl;
      // Print translation model information
  cerr<< "  - Statistical machine translation model type: "<<CURR_MODEL_LABEL<<endl;
  if(strlen(CURR_MODEL_NOTES)!=0)
  {
    cerr << "  - Model notes: "<<CURR_MODEL_NOTES<<endl;
  }

      // Print language model information
  cerr << "  - Language model type: "<<THOT_CURR_LM_LABEL<<endl;
  if(strlen(THOT_CURR_LM_NOTES)!=0)
  {
    cerr << "  - Language model notes: "<<THOT_CURR_LM_NOTES<<endl;
  }

      // Print phrase-based model information
  cerr << "  - Phrase-based model type: "<<THOT_CURR_PBM_LABEL<<endl;
  if(strlen(THOT_CURR_PBM_NOTES)!=0)
  {
    cerr << "  - Phrase-based model notes: "<<THOT_CURR_PBM_NOTES<<endl;
  }

      // Print single-word model information
  cerr << "  - Single-word model type: "<<CURR_SWM_LABEL<<endl;
  if(strlen(CURR_SWM_NOTES)!=0)
  {
    cerr << "  - Single-word model notes: "<<CURR_SWM_NOTES<<endl;
  }

      // Print decoding algorithm information
  cerr << "  - Translator type: "<<CURR_MSTACK_LABEL<<endl;
  if(strlen(CURR_MSTACK_NOTES)!=0)
  {
    cerr << "  - Translator notes: "<<CURR_MSTACK_NOTES<<endl;
  }
  cerr << endl;
}

//---------------
void printUsage(void)
{
  cerr << "thot_ms_dec      [-c <string>] [-tm <string>] [-lm <string>]"<<endl;
  cerr << "                 -t <string> [-o <string>]"<<endl;
  cerr << "                 [-W <float>] [-S <int>] [-A <int>]"<<endl;
  cerr << "                 [-I <int>] [-G <int>] [-h <int>]"<<endl;
  cerr << "                 [-be] [ -nomon <int>] [-tmw <float> ... <float>]"<<endl;
  cerr << "                 [-wg <string> [-wgp <float>] ]"<<endl;
  cerr << "                 [-v|-v1|-v2]"<<endl;
  cerr << "                 [--help] [--version] [--config]"<<endl<<endl;
  cerr << " -c <string>           : Configuration file (command-line options override"<<endl;
  cerr << "                         configuration file options)."<<endl;
  cerr << " -tm <string>          : Prefix of the translation model files."<<endl;
  cerr << " -lm <string>          : Language model file name."<<endl;
  cerr << " -t <string>           : File with the test sentences."<<endl;
  cerr << " -o <string>           : File to store translations (if not given, they are"<<endl;
  cerr << "                         printed to the standard output)."<<endl;
  cerr << " -W <float>            : Maximum number of translation options to be considered"<<endl;
  cerr << "                         per each source phrase ("<<PMSTACK_W_DEFAULT<<" by default)."<<endl;
  cerr << " -S <int>              : Maximum number of hypotheses that can be stored in"<<endl;
  cerr << "                         each stack ("<<PMSTACK_S_DEFAULT<<" by default)."<<endl;    
  cerr << " -A <int>              : Maximum length in words of the source phrases to be"<<endl;
  cerr << "                         translated ("<<PMSTACK_A_DEFAULT<<" by default)."<<endl;
  cerr << " -I <int>              : Number of hypotheses expanded at each iteration"<<endl;
  cerr << "                         ("<<PMSTACK_I_DEFAULT<<" by default)."<<endl;
#ifdef MULTI_STACK_USE_GRAN
  cerr << " -G <int>              : Granularity parameter ("<<PMSTACK_G_DEFAULT<<"by default)."<<endl;
#else
  cerr << " -G <int>              : Parameter not available with the given configuration."<<endl;
#endif
  cerr << " -h <int>              : Heuristic function used: "<<NO_HEURISTIC<<"->None, "<<LOCAL_T_HEURISTIC<<"->LOCAL_T, "<<endl;
  cerr << "                         "<<LOCAL_TD_HEURISTIC<<"->LOCAL_TD ("<<PMSTACK_H_DEFAULT<<" by default)."<<endl;
  cerr << " -be                   : Execute a best-first algorithm (breadth-first search"<<endl;
  cerr << "                         is executed by default)."<<endl;
  cerr << " -nomon <int>          : Perform a non-monotonic search, allowing the decoder"<<endl;
  cerr << "                         to skip up to <int> words from the last aligned source"<<endl;
  cerr << "                         words. If <int> is equal to zero, then a monotonic"<<endl;
  cerr << "                         search is performed ("<<PMSTACK_NOMON_DEFAULT<<" is the default value)."<<endl;
  cerr << " -tmw <float>...<float>: Set model weights, the number of weights and their"<<endl;
  cerr << "                         meaning depends on the model type (use --config"<<endl;
  cerr << "                         option)."<<endl;
  cerr << " -wg <string>          : Print word graph after each translation, the prefix" <<endl;
  cerr << "                         of the files is given as parameter."<<endl;
  cerr << " -wgp <float>          : Prune word-graph using the given threshold.\n";
  cerr << "                         Threshold=0 -> no pruning is performed.\n";
  cerr << "                         Threshold=1 -> only the best arc arriving to each\n";
  cerr << "                                       state is retained.\n";
  cerr << "                         If not given, the number of arcs is not\n";
  cerr << "                         restricted.\n";
  cerr << " -v|-v1|-v2            : verbose modes."<<endl;
  cerr << " --help                : Display this help and exit."<<endl;
  cerr << " --version             : Output version information and exit."<<endl;
  cerr << " --config              : Show current configuration."<<endl;
}

//---------------
void version(void)
{
  cerr<<"thot_ms_dec is part of the thot package "<<endl;
  cerr<<"thot version "<<THOT_VERSION<<endl;
  cerr<<"thot is GNU software written by Daniel Ortiz"<<endl;
}
