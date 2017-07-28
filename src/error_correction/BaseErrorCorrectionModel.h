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
/* Module: BaseErrorCorrectionModel                                 */
/*                                                                  */
/* Prototypes file: BaseErrorCorrectionModel.h                      */
/*                                                                  */
/* Description: Declares the BaseErrorCorrectionModel class,        */
/*              this class is a base class for implementing error   */
/*              correction models for strings                       */
/*                                                                  */
/********************************************************************/

/**
 * @file BaseErrorCorrectionModel.h
 * 
 * @brief Defines the BaseErrorCorrectionModel class, this class is a
 * base class for implementing error correction model for strings
 */

#ifndef _BaseErrorCorrectionModel_h
#define _BaseErrorCorrectionModel_h

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include <string>
#include "myVector.h"
#include <ErrorDefs.h>
#include "WordAndCharLevelOps.h"
#include <StatModelDefs.h>
#include <Score.h>

//--------------- Constants ------------------------------------------


//--------------- Classes --------------------------------------------

//--------------- BaseErrorCorrectionModel template class

/**
 * @brief The BaseErrorCorrectionModel class is a base class for
 * implementing error correction models for strings
 */

class BaseErrorCorrectionModel
{
 public:

      // Declarations related to dynamic class loading
  typedef BaseErrorCorrectionModel* create_t(std::string);
  typedef std::string type_id_t(void);

      // Basic functions
  virtual Score similarity(Vector<std::string> x,
                           Vector<std::string> y)=0;
      // Calculates similarity between x and y
  virtual Score similarityGivenPrefix(Vector<std::string> x,
                                      Vector<std::string> y)=0;
      // Calculates similarity between x and y, where y is taken as a
      // prefix
  virtual void correctStrGivenPref(Vector<std::string> uncorrStrVec,
                                   Vector<std::string> prefStrVec,
                                   Vector<std::string>& correctedStrVec,
                                   Vector<pair<PositionIndex, PositionIndex> >& sourceSegmentation,
                                   Vector<PositionIndex>& targetSegmentCuts)=0;
      // Corrects string 'uncorrStrVec' given the prefix 'prefStrVec'
      // storing the results in 'correctedStrVec'
  
      // Model weights functions
  virtual void setWeights(Vector<float> wVec)=0;
  virtual unsigned int getNumWeights(void)=0;
  virtual void printWeights(ostream &outS)=0;

      // load() and print() functions
  virtual bool load(const char *prefix,
                    int verbose=0)=0;
  virtual bool print(const char *prefix)=0;

      // Functions for performing on-line training
  virtual int trainStrPair(const char* x,
                           const char* y,
                           int verbose=0);

      // clear() function
  virtual void clear(void){};
  
      // Destructor
  virtual ~BaseErrorCorrectionModel(){};
};

#endif
