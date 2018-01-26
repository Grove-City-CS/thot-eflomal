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
 * @file EU_PrePosProcessor1.cc
 * 
 * @brief Definitions file for EU_PrePosProcessor1.h
 */

//--------------- Include files --------------------------------------

#include "EU_PrePosProcessor1.h"


//--------------- EU_PrePosProcessor1 class functions
//

//---------------------------------------
EU_PrePosProcessor1::EU_PrePosProcessor1()
{
  
}

//---------------------------------------
bool EU_PrePosProcessor1::loadCapitInfo(const char* filename)
{
  std::ifstream *caps=0;
  
  std::cerr<<"Initializing capitalization from file "<<filename<<" ... ";

  caps= new std::ifstream(filename);
  if ( caps )
  {
    EUpostprocInitializeCapitalization(*caps);
    delete caps;
    std::cerr<<"THOT_OK"<<std::endl;
    
    return THOT_OK;
  }
  else
  {
    std::cerr<<"THOT_ERROR"<<std::endl;
    
    return THOT_ERROR;
  }
}

//---------------------------------------
std::string EU_PrePosProcessor1::preprocLine(std::string str,
                                             bool caseconv,
                                             bool keepPreprocInfo)
{
      // Store state info
  if(keepPreprocInfo)
    lastPreprocStrKept=str;

  return EUpprocLine(str.c_str(),caseconv,keepPreprocInfo);
      // when last argument of EUpprocLine is true its internal tables
      // are modified
}

//---------------------------------------
std::string EU_PrePosProcessor1::postprocLine(std::string str,
                                              bool caseconv)
{
      // Restore preprocessing info if necessary
  if(!lastPreprocStrKept.empty())
    EUpprocLine(lastPreprocStrKept.c_str(),caseconv,true);
  
  return EUpostprocLine(str.c_str(),caseconv);
}

//---------------------------------------
bool EU_PrePosProcessor1::isCategory(std::string word)
{
  if(word.size()==0) return false;
  else
  {
    if(word[0]=='_')
      return true;
    else
      return false;
  }
}

//---------------------------------------
EU_PrePosProcessor1::~EU_PrePosProcessor1()
{
  
}
