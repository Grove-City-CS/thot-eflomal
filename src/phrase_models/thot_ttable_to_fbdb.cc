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
/* Module: thot_ttable_to_fastbdb.cc                                */
/*                                                                  */
/* Definitions file: thot_ttable_to_fastbdb.cc                      */
/*                                                                  */
/* Description: Converts a translation table to BDB format.         */
/*                                                                  */   
/********************************************************************/


//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include <FastBdbPhraseTable.h>
#include "PhraseDefs.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include "options.h"
#include <awkInputStream.h>

//--------------- Constants ------------------------------------------


//--------------- Function Declarations ------------------------------

int TakeParameters(int argc,char *argv[]);
void printUsage(void);
int extractEntryInfo(awkInputStream& awk,
                     Vector<WordIndex>& srcPhr,
                     Vector<WordIndex>& trgPhr,
                     Count& jointCount);
int process_ttable(void);

//--------------- Type definitions -----------------------------------


//--------------- Global variables -----------------------------------

std::string outputFilesPrefix;

//--------------- Function Definitions -------------------------------

//---------------
int main(int argc,char *argv[])
{
  if(TakeParameters(argc,argv)==OK)
  {
    return process_ttable();
  }
  else return ERROR;
}

//---------------
int extractEntryInfo(awkInputStream& awk,
                     Vector<WordIndex>& srcPhr,
                     Vector<WordIndex>& trgPhr,
                     Count& jointCount)
{
  unsigned int i;

      // Obtain source phrase
  srcPhr.clear();
  for(i=1;i<=awk.NF;++i)
  {
    if(awk.dollar(i)=="|||")
      break;
    else
      srcPhr.push_back(atoi(awk.dollar(i).c_str()));
  }
  if(i==awk.NF)
    return ERROR;

      // Obtain target phrase
  trgPhr.clear();
  i+=1;
  for(;i<=awk.NF;++i)
  {
    if(awk.dollar(i)=="|||")
      break;
    else
      trgPhr.push_back(atoi(awk.dollar(i).c_str()));
  }
  if(i!=awk.NF-2)
    return ERROR;

      // Obtain joint count
  jointCount=atof(awk.dollar(awk.NF).c_str());

  return OK;
}

//---------------
int process_ttable(void)
{
      // Read standard input
  awkInputStream awk;
  if(awk.open_stream(stdin)==ERROR)
  {
    cerr<<"Error while reading from standard input!\n";
    return ERROR;
  }
  else
  {
        // Initialize fast BDB phrase table class
    FastBdbPhraseTable fastBdbPt;
    fastBdbPt.init(outputFilesPrefix.c_str());
    
        // Process translation table
    while(awk.getln())
    {
      // if(awk.FNR % 1000 ==0)
      //   cerr<<"Processing entry "<<awk.FNR<<endl;

      Vector<WordIndex> srcPhr;
      Vector<WordIndex> trgPhr;
      Count jointCount;
      int ret=extractEntryInfo(awk,srcPhr,trgPhr,jointCount);
      if(ret==OK)
        fastBdbPt.incrCountsOfEntry(srcPhr,trgPhr,jointCount);
    }

    fastBdbPt.enableFastSearch();
    
    return OK;
  }
}

//---------------
int TakeParameters(int argc,char *argv[])
{
 int err;

     /* Verify --help option */
 err=readOption(argc,argv,"--help");
 if(err!=-1)
 {
   printUsage();
   return ERROR;
 }

     /* Takes the output files prefix */
 err=readSTLstring(argc,argv, "-o", &outputFilesPrefix);
 if(err==-1)
 {
   printUsage();
   return ERROR;
 }

 return OK;  
}

//---------------
void printUsage(void)
{
  printf("Usage: thot_ttable_to_fastbdb -o <string> [--help]\n\n");
  printf("-o <string>                   Prefix of output files.\n\n");
  printf("--help                        Display this help and exit.\n\n");
}

//--------------------------------
