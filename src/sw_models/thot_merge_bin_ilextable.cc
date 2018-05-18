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
 * @file thot_merge_bin_ilextable.cc
 * 
 * @brief Merges counts given in a set of sorted incremental lexical
 * tables.
 */

//--------------- Include files --------------------------------------

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <queue>
#include <set>
#include "options.h"
#include "SwDefs.h"
#include <MathFuncs.h>

//--------------- Constants ------------------------------------------

#define RECORD_READ     0
#define NO_RECORDS_LEFT 1

//--------------- Type definitions -----------------------------------

struct Entry
{
  WordIndex s;
  WordIndex t;
  float numer;
  float denom;
  unsigned int id;
};

struct SortBySrcAndTrg
{
  bool operator() (const Entry& a,
                   const Entry& b)const
    {
      if(b.s<a.s) return 1;
      if(a.s<b.s) return 0;

      if(b.t<a.t) return 1;
      if(a.t<b.t) return 0;

      return 0;
    }
};

typedef std::priority_queue<Entry,std::vector<Entry>,SortBySrcAndTrg> MergePrQueue;

typedef std::set<unsigned int> ChunkSet;

//--------------- Function Declarations ------------------------------

int openFiles(void);
void initPrQueue(MergePrQueue& entryPrQueue);
int getNextEntry(MergePrQueue& entryPrQueue,
                 Entry& entry);
void printCounts(WordIndex firstSrc,
                 const std::vector<WordIndex>& trgWordVec,
                 float lcSrc,
                 const std::vector<float>& lcSrcTrgVec);
void clear();
int readTableRecord(std::ifstream& inF,
                    WordIndex& s,
                    WordIndex& t,
                    float& numer,
                    float& denom);
int TakeParameters(int argc,char *argv[]);
void printUsage(void);
void printDesc(void);

//--------------- Global variables -----------------------------------

std::vector<std::string> fileNameVec;
std::vector<std::ifstream*> ifstreamPtrVec;
std::vector<bool> eofFlagVec;

//--------------- Function Definitions -------------------------------

//--------------- main function
int main(int argc,char *argv[])
{
  if(TakeParameters(argc,argv)==THOT_OK)
  {
        // Open files
    int ret=openFiles();
    if(ret==THOT_ERROR)
      return THOT_ERROR;
    
        // Process entries contained in the set of files...

        // Initialize priority queue
    MergePrQueue entryPrQueue;
    initPrQueue(entryPrQueue);
    
        // while loop
    bool end=false;
    bool first_entry=true;
    WordIndex firstSrc=0;
    float lcSrc=SMALL_LG_NUM;
    std::vector<WordIndex> trgWordVec;
    std::vector<float> lcSrcTrgVec;
    ChunkSet chunkSet;
    
    while(!end)
    {
      Entry entry;
      ret=getNextEntry(entryPrQueue,entry);
      if(ret==RECORD_READ)
      {
//        printf("** %d %d %g %g %g\n",entry.s,entry.t,entry.numer,entry.denom,exp(entry.numer-entry.denom));

            // Verify if it is the first entry of the table
        if(first_entry==1)
        {
          firstSrc=entry.s;
          first_entry=false;
          trgWordVec.push_back(entry.t);
          lcSrcTrgVec.push_back(entry.numer);
        }
        else
        {
              // A new source word has appeared?
          if(firstSrc!=entry.s)
          {
                // Print counts
            printCounts(firstSrc,trgWordVec,lcSrc,lcSrcTrgVec);

                // Reset variables
            firstSrc=entry.s;
            chunkSet.clear();
            trgWordVec.clear();
            lcSrcTrgVec.clear();
            trgWordVec.push_back(entry.t);
            lcSrcTrgVec.push_back(entry.numer);
            lcSrc=SMALL_LG_NUM;
          }
          else
          {
            trgWordVec.push_back(entry.t);
            lcSrcTrgVec.push_back(entry.numer);
          }
        }
        ChunkSet::const_iterator csConstIter=chunkSet.find(entry.id);
        if(csConstIter==chunkSet.end()) 
        {
          chunkSet.insert(entry.id);
          lcSrc=MathFuncs::lns_sumlog(lcSrc,entry.denom);
        }
      }
      else end=true;
    }
        // Print last group of counts
    printCounts(firstSrc,trgWordVec,lcSrc,lcSrcTrgVec);

        // Close files and release pointers
    clear();

    return THOT_OK;
  }
  else return THOT_ERROR;
}

//--------------- openFiles() function
int openFiles(void)
{
  for(unsigned int i=0;i<fileNameVec.size();++i)
  {
        // Create file stream
    std::ifstream* ifstreamPtr=new std::ifstream;
    ifstreamPtrVec.push_back(ifstreamPtr);
    ifstreamPtrVec[i]->open(fileNameVec[i].c_str(), std::ios::in | std::ios::binary);
    if(! *ifstreamPtrVec[i])
    {
      std::cerr<<"Error in file with incremental lexical table, file "<<fileNameVec[i]<<" does not exist.\n";
      return THOT_ERROR;    
    }
    
        // Create flag for file
    eofFlagVec.push_back(false);
  }
  
  return THOT_OK;
}

//--------------- initPrQueue() function
void initPrQueue(MergePrQueue& entryPrQueue)
{
  for(unsigned int i=0;i<ifstreamPtrVec.size();++i)
  {
    Entry entry;
    int ret=readTableRecord(*ifstreamPtrVec[i],entry.s,entry.t,entry.numer,entry.denom);
    if(ret==RECORD_READ)
    {
      entry.id=i;
      entryPrQueue.push(entry);
    }
    else
    {
      eofFlagVec[i]=true;
    }
  }
}

//--------------- getNextEntry() function
int getNextEntry(MergePrQueue& entryPrQueue,
                 Entry& entry)
{
      // Check if queue is not empty
  if(!entryPrQueue.empty())
  {
        // Obtain top of the queue and pop it
    entry=entryPrQueue.top();
    entryPrQueue.pop();
        // Push next entry of corresponding file if there exists
    Entry nextEntry;
    int ret=readTableRecord(*ifstreamPtrVec[entry.id],nextEntry.s,nextEntry.t,nextEntry.numer,nextEntry.denom);
    if(ret==RECORD_READ)
    {
      nextEntry.id=entry.id;
      entryPrQueue.push(nextEntry);
    }
    else
    {
      eofFlagVec[entry.id]=true;
    }
            // Print entry
    return RECORD_READ;
  }
  else
    return NO_RECORDS_LEFT;
}

//--------------- printCounts() function
void printCounts(WordIndex firstSrc,
                 const std::vector<WordIndex>& trgWordVec,
                 float lcSrc,
                 const std::vector<float>& lcSrcTrgVec)
{
  WordIndex firstTrg=trgWordVec[0];
  float glcSrcTrg=lcSrcTrgVec[0];
  for(unsigned int n=1;n<trgWordVec.size();++n)
  {
    if(firstTrg==trgWordVec[n])
    {
          // Accumulate count of target phrase for additional chunk
      glcSrcTrg=MathFuncs::lns_sumlog(glcSrcTrg,lcSrcTrgVec[n]);
    }
    else
    {
          // Print count for current target phrase
//      printf("%d %d %g %g\n",firstSrc,firstTrg,glcSrcTrg,lcSrc);
      std::cout.write((char*)&firstSrc,sizeof(WordIndex));
      std::cout.write((char*)&firstTrg,sizeof(WordIndex));
      std::cout.write((char*)&glcSrcTrg,sizeof(float));
      std::cout.write((char*)&lcSrc,sizeof(float));
 
          // Initialize variables for next target phrase
      firstTrg=trgWordVec[n];
      glcSrcTrg=lcSrcTrgVec[n];
    }
  }
      // Print last target phrase
//  printf("%d %d %g %g\n",firstSrc,firstTrg,glcSrcTrg,lcSrc);
  std::cout.write((char*)&firstSrc,sizeof(WordIndex));
  std::cout.write((char*)&firstTrg,sizeof(WordIndex));
  std::cout.write((char*)&glcSrcTrg,sizeof(float));
  std::cout.write((char*)&lcSrc,sizeof(float));
}

//--------------- clear() function
void clear(void)
{
  for(unsigned int i=0;i<ifstreamPtrVec.size();++i)
  {
    ifstreamPtrVec[i]->close();
    delete ifstreamPtrVec[i];
  }
}

//--------------- readTableRecord() function
int readTableRecord(std::ifstream& inF,
                    WordIndex& s,
                    WordIndex& t,
                    float& numer,
                    float& denom)
{
  if(inF.read((char*)&s,sizeof(WordIndex)))
  {
    inF.read((char*)&t,sizeof(WordIndex));
    inF.read((char*)&numer,sizeof(float));
    inF.read((char*)&denom,sizeof(float));
    return RECORD_READ;
  }
  else return NO_RECORDS_LEFT;
}

//--------------- TakeParameters function
int TakeParameters(int argc,char *argv[])
{
 int err;

 if(argc==1)
 {
   printDesc();
   return THOT_ERROR;   
 }

     /* Verify --help option */
 err=readOption(argc,argv,"--help");
 if(err!=-1)
 {
   printUsage();
   return THOT_ERROR;
 }

     /* Takes the table file names */
 for(int i=1;i<argc;++i)
 {
   std::string fileName=argv[i];
   fileNameVec.push_back(fileName);
 }

 return THOT_OK;  
}

//--------------- printDesc() function
void printDesc(void)
{
  printf("thot_merge_bin_ilextable written by Daniel Ortiz\n");
  printf("A tool to merge the counts of a set of sorted incremental lexical tables\n");
  printf("type \"thot_merge_bin_ilextable --help\" to get usage information.\n");
}

//--------------- printUsage() function
void printUsage(void)
{
  printf("Usage: thot_merge_bin_ilextable <sorted_ilextable_1> [<sorted_ilextable_2> ...]\n");
  printf("                                [--help]\n\n");
  printf("--help                     Display this help and exit.\n\n");
}

//--------------------------------
