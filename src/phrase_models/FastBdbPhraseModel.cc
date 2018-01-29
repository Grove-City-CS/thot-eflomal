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
 * @file FastBdbPhraseModel.cc
 * 
 * @brief Definitions file for FastBdbPhraseModel.h
 */

//--------------- Include files --------------------------------------

#include "FastBdbPhraseModel.h"

//--------------- Global variables -----------------------------------


//--------------- FastBdbPhraseModel class function definitions

//-------------------------
FastBdbPhraseModel::FastBdbPhraseModel(void)
{
}

//-------------------------
Count FastBdbPhraseModel::cSrcTrg(const std::vector<WordIndex>& s,
                                  const std::vector<WordIndex>& t)
{
  bool found;
  return fastBdbPhraseTable.getSrcTrgInfo(s,t,found);
}

//-------------------------
Count FastBdbPhraseModel::cSrc(const std::vector<WordIndex>& s)
{
  bool found;
  return fastBdbPhraseTable.getSrcInfo(s,found);
}

//-------------------------
Count FastBdbPhraseModel::cTrg(const std::vector<WordIndex>& t)
{
  bool found;
  return fastBdbPhraseTable.getTrgInfo(t,found);
}

//-------------------------
Count FastBdbPhraseModel::cHSrcHTrg(const std::vector<std::string>& hs,
                                    const std::vector<std::string>& ht)
{
  std::vector<WordIndex> s;
  std::vector<WordIndex> t;

      // Generate vector of source WordIndex
  for(unsigned int i=0;i<hs.size();++i)
  {
    if(!existSrcSymbol(hs[i])) return 0;
    else s.push_back(stringToSrcWordIndex(hs[i]));
  }

      // Generate vector of target WordIndex
  for(unsigned int i=0;i<ht.size();++i)
  {
    if(!existTrgSymbol(ht[i])) return 0;
    else t.push_back(stringToTrgWordIndex(ht[i]));
  }

  return cSrcTrg(s,t);
}

//-------------------------
Count FastBdbPhraseModel::cHSrc(const std::vector<std::string>& hs)
{
  std::vector<WordIndex> s;

      // Generate vector of source WordIndex
  for(unsigned int i=0;i<hs.size();++i)
  {
    if(!existSrcSymbol(hs[i])) return 0;
    else s.push_back(stringToSrcWordIndex(hs[i]));
  }
  return cSrc(s);
}

//-------------------------
Count FastBdbPhraseModel::cHTrg(const std::vector<std::string>& ht)
{
  std::vector<WordIndex> t;

      // Generate vector of target WordIndex
  for(unsigned int i=0;i<ht.size();++i)
  {
    if(!existTrgSymbol(ht[i])) return 0;
    else t.push_back(stringToTrgWordIndex(ht[i]));
  }
  return cTrg(t);  
}

//-------------------------
PhrasePairInfo FastBdbPhraseModel::infSrcTrg(const std::vector<WordIndex>& s,
                                             const std::vector<WordIndex>& t,
                                             bool& found)
{
  PhrasePairInfo ppInfo;
  ppInfo.first=fastBdbPhraseTable.getSrcInfo(s,found);
  ppInfo.second=fastBdbPhraseTable.getSrcTrgInfo(s,t,found);
  return ppInfo;
}

//-------------------------
Prob FastBdbPhraseModel::pk_tlen(unsigned int tlen,
                             unsigned int k)
{
  Prob p=segLenTable.pk_tlen(tlen,k);

  if((double) p < SEGM_SIZE_PROB_SMOOTH)
    return SEGM_SIZE_PROB_SMOOTH;
  else return p;
}

//-------------------------
LgProb FastBdbPhraseModel::srcSegmLenLgProb(unsigned int x_k,
                                            unsigned int x_km1,
                                            unsigned int srcLen)
{
  return srcSegmLenTable.srcSegmLenLgProb(x_k,x_km1,srcLen);
}

//-------------------------
LgProb FastBdbPhraseModel::trgCutsLgProb(int offset)
{
  return trgCutsTable.trgCutsLgProb(offset);
}

//-------------------------
LgProb FastBdbPhraseModel::trgSegmLenLgProb(unsigned int k,
                                            const SentSegmentation& trgSegm,
                                            unsigned int trgLen,
                                            unsigned int lastSrcSegmLen)
{
  return trgSegmLenTable.trgSegmLenLgProb(k,trgSegm,trgLen,lastSrcSegmLen);
}

//-------------------------
LgProb FastBdbPhraseModel::logpt_s_(const std::vector<WordIndex>& s,
                                    const std::vector<WordIndex>& t)
{
  LgProb lp=fastBdbPhraseTable.logpTrgGivenSrc(s,t);
  if((double)lp<LOG_PHRASE_PROB_SMOOTH)
    return LOG_PHRASE_PROB_SMOOTH;
  else
    return lp;
}

//-------------------------
LgProb FastBdbPhraseModel::logps_t_(const std::vector<WordIndex>& s,
                                    const std::vector<WordIndex>& t)
{
  LgProb lp=fastBdbPhraseTable.logpSrcGivenTrg(s,t);
  if((double)lp<LOG_PHRASE_PROB_SMOOTH)
    return LOG_PHRASE_PROB_SMOOTH;
  else
    return lp;
}

//-------------------------
bool FastBdbPhraseModel::getTransFor_s_(const std::vector<WordIndex>& /*s*/,
                                        FastBdbPhraseModel::TrgTableNode& trgtn)
{
  trgtn.clear();
  std::cerr<<"Warning: getTransFor_s_() function not implemented for this class"<<std::endl;
  return false;
}

//-------------------------
bool FastBdbPhraseModel::getTransFor_t_(const std::vector<WordIndex>& t,
                                        FastBdbPhraseModel::SrcTableNode& srctn)
{
  return fastBdbPhraseTable.getEntriesForTarget(t,srctn);
}

//-------------------------
bool FastBdbPhraseModel::getNbestTransFor_s_(const std::vector<WordIndex>& /*s*/,
                                             NbestTableNode<PhraseTransTableNodeData>& nbt)
{
  nbt.clear();
  std::cerr<<"Warning: getNbestTransFor_s_() function not implemented for this class"<<std::endl;
  return false;
}

//-------------------------	
bool FastBdbPhraseModel::getNbestTransFor_t_(const std::vector<WordIndex>& t,
                                             NbestTableNode<PhraseTransTableNodeData>& nbt,
                                             int N/*=-1*/) 
{  
  return fastBdbPhraseTable.getNbestForTrg(t,nbt,N);
}

//-------------------------
bool FastBdbPhraseModel::load(const char *prefix)
{
  std::string mainFileName;
  if(fileIsDescriptor(prefix,mainFileName))
  {
    std::string descFileName=prefix;
    std::string absolutizedMainFileName=absolutizeModelFileName(descFileName,mainFileName);
    return load_given_prefix(absolutizedMainFileName.c_str());
  }
  else
  {
    return load_given_prefix(prefix);
  }
}

//-------------------------
bool FastBdbPhraseModel::load_given_prefix(const char *prefix)
{
  bool ret;

      // Clear previous tables
  fastBdbPhraseTable.clear();
  segLenTable.clear();

      // Load source vocabulary
  std::string srcvocabfile=prefix;
  srcvocabfile=srcvocabfile+".fbdb_svcb";
  ret=loadSrcVocab(srcvocabfile.c_str());
  if(ret==THOT_ERROR) return THOT_ERROR;
  
      // Load target vocabulary
  std::string trgvocabfile=prefix;
  trgvocabfile=trgvocabfile+".fbdb_tvcb";
  ret=loadTrgVocab(trgvocabfile.c_str());
  if(ret==THOT_ERROR) return THOT_ERROR;
  
      // Load translation table
  ret=fastBdbPhraseTable.init(prefix);
  if(ret==THOT_ERROR) return THOT_ERROR;

      // Load segmentation length table
  std::string seglenfile=prefix;
  seglenfile=seglenfile+".seglentable";
  load_seglentable(seglenfile.c_str());

      // Load source phrase length table
  std::string srcSegmLenFile=prefix;
  srcSegmLenFile=srcSegmLenFile+".srcsegmlentable";
  srcSegmLenTable.load(srcSegmLenFile.c_str());

      // Load target cuts table
  std::string trgCutsTableFile=prefix;
  trgCutsTableFile=trgCutsTableFile+".trgcutstable";
  trgCutsTable.load(trgCutsTableFile.c_str());

      // Load target phrase length table
  std::string trgSegmLenFile=prefix;
  trgSegmLenFile=trgSegmLenFile+".trgsegmlentable";
  trgSegmLenTable.load(trgSegmLenFile.c_str());

      // Store prefix of model files
  prefixOfModelFiles=prefix;

  return THOT_OK;
}

//-------------------------
bool FastBdbPhraseModel::load_seglentable(const char *segmLengthTableFileName)
{
  return segLenTable.load_seglentable(segmLengthTableFileName);
}

//-------------------------
bool FastBdbPhraseModel::print(const char *prefix)
{
  std::string prefixStl=prefix;
  if(prefixOfModelFiles==prefixStl)
  {
    fastBdbPhraseTable.enableFastSearch();
    return THOT_OK;
  }
  else
  {
    std::cerr<<"Warning: print() function not implemented for this model"<<std::endl;
    return THOT_ERROR;
  }
}

//-------------------------
size_t FastBdbPhraseModel::getSrcVocabSize(void)const
{
  return singleWordVocab.getSrcVocabSize();	
}

//-------------------------
bool FastBdbPhraseModel::loadSrcVocab(const char *srcInputVocabFileName)
{
  return singleWordVocab.loadSrcVocab(srcInputVocabFileName);
}

//-------------------------
bool FastBdbPhraseModel::loadTrgVocab(const char *trgInputVocabFileName)
{
  return singleWordVocab.loadTrgVocab(trgInputVocabFileName);
}

//-------------------------
WordIndex FastBdbPhraseModel::stringToSrcWordIndex(std::string s)const
{	
 return singleWordVocab.stringToSrcWordIndex(s);
}

//-------------------------
std::string FastBdbPhraseModel::wordIndexToSrcString(WordIndex w)const
{
 return singleWordVocab.wordIndexToSrcString(w);
}

//-------------------------
bool FastBdbPhraseModel::existSrcSymbol(std::string s)const
{
 return singleWordVocab.existSrcSymbol(s);
}

//-------------------------
std::vector<WordIndex> FastBdbPhraseModel::strVectorToSrcIndexVector(const std::vector<std::string>& s)
{
  std::vector<WordIndex> swVec;
  
  for(unsigned int i=0;i<s.size();++i)
    swVec.push_back(addSrcSymbol(s[i]));
  
  return swVec;
}

//-------------------------
std::vector<std::string> FastBdbPhraseModel::srcIndexVectorToStrVector(const std::vector<WordIndex>& s)
{
 std::vector<std::string> vStr;
 unsigned int i;

 for(i=0;i<s.size();++i)
    vStr.push_back((wordIndexToSrcString(s[i]))); 	 
	
 return vStr;
}
//-------------------------
WordIndex FastBdbPhraseModel::addSrcSymbol(std::string s)
{
 return singleWordVocab.addSrcSymbol(s);
}

//-------------------------
bool FastBdbPhraseModel::printSrcVocab(const char *outputFileName)
{
 return singleWordVocab.printSrcVocab(outputFileName);
}

//-------------------------
size_t FastBdbPhraseModel::getTrgVocabSize(void)const
{
 return singleWordVocab.getTrgVocabSize();	
}

//-------------------------
WordIndex FastBdbPhraseModel::stringToTrgWordIndex(std::string t)const
{
 return singleWordVocab.stringToTrgWordIndex(t);
}

//-------------------------
std::string FastBdbPhraseModel::wordIndexToTrgString(WordIndex w)const
{
 return singleWordVocab.wordIndexToTrgString(w);
}

//-------------------------
bool FastBdbPhraseModel::existTrgSymbol(std::string t)const
{
 return singleWordVocab.existTrgSymbol(t);
}

//-------------------------
std::vector<WordIndex> FastBdbPhraseModel::strVectorToTrgIndexVector(const std::vector<std::string>& t)
{
  std::vector<WordIndex> twVec;
  
  for(unsigned int i=0;i<t.size();++i)
    twVec.push_back(addTrgSymbol(t[i]));
  
  return twVec;
}
//-------------------------
std::vector<std::string> FastBdbPhraseModel::trgIndexVectorToStrVector(const std::vector<WordIndex>& t)
{
 std::vector<std::string> vStr;
 unsigned int i;

 for(i=0;i<t.size();++i)
    vStr.push_back((wordIndexToTrgString(t[i]))); 	 
	
 return vStr;
}
//-------------------------
WordIndex FastBdbPhraseModel::addTrgSymbol(std::string t)
{
 return singleWordVocab.addTrgSymbol(t);
}
//-------------------------
bool FastBdbPhraseModel::printTrgVocab(const char *outputFileName)
{
 return singleWordVocab.printTrgVocab(outputFileName);
}

//-------------------------
std::vector<std::string> FastBdbPhraseModel::stringToStringVector(std::string s)
{
 std::vector<std::string> vs;	
 std::string aux;
 unsigned int i=0;	
 bool end=false;
	
 while(!end)
 {
  aux="";	 
  while(s[i]!=' ' && s[i]!='	' && i<s.size()) 
  {
    aux+=s[i];
    ++i;		   
  }
  if(aux!="") vs.push_back(aux);	   
  while((s[i]==' ' || s[i]=='	') && i<s.size()) ++i;	 
  if(i>=s.size()) end=true;	  
 }
 
 return vs;	
}

//-------------------------
std::vector<std::string> FastBdbPhraseModel::extractCharItemsToVector(char *ch)const
{
 unsigned int i=0;
 std::string s;	
 std::vector<std::string> v,u;

 while(ch[i]!=0)
 {
   s=""; 
   while(ch[i]==' ' && ch[i]!=0) {++i;}	  
   while(ch[i]!=' ' && ch[i]!=0) {s=s+ch[i]; ++i;}
   if(s!="") v.push_back(s);  
 }	  
	 
 for(i=0;i<v.size();++i)
 {
   u.push_back(v[v.size()-1-i]);
 }
 return u;
}

//-------------------------
size_t FastBdbPhraseModel::size(void)
{
  return fastBdbPhraseTable.size();
}

//-------------------------
void FastBdbPhraseModel::clear(void)
{
  singleWordVocab.clear();
  fastBdbPhraseTable.clear();
  segLenTable.clear();
  prefixOfModelFiles.clear();
}

//-------------------------
FastBdbPhraseModel::~FastBdbPhraseModel()
{
}

//-------------------------
