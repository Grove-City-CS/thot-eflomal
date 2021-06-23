/*
thot package for statistical machine translation
Copyright (C) 2013-2017 Daniel Ortiz-Mart\'inez, Adam Harasimowicz

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
 * @file _incrHmmP0AligModel.h
 *
 * @brief Defines the _incrHmmP0AligModel class.  _incrHmmP0AligModel
 * class allows to generate and access to the data of a Hmm statistical
 * alignment model with fixed p0 probability.
 *
 */

#pragma once

//--------------- Include files --------------------------------------

#include "sw_models/_incrHmmAligModel.h"

//--------------- Constants ------------------------------------------

#define DEFAULT_HMM_P0 0.1

//--------------- typedefs -------------------------------------------

//--------------- function declarations ------------------------------

//--------------- Classes --------------------------------------------

//--------------- _incrHmmP0AligModel class

class _incrHmmP0AligModel : public _incrHmmAligModel
{
public:
  // Constructor
  _incrHmmP0AligModel();

  // Set hmm p0 value
  void set_hmm_p0(Prob _hmm_p0);

  // load function
  bool load(const char* prefFileName, int verbose = 0);

  // print function
  bool print(const char* prefFileName, int verbose = 0);

  // clear() function
  void clear(void);

protected:
  Prob hmm_p0;

  bool loadHmmP0(const char* hmmP0FileName, int verbose);
  bool printHmmP0(const char* hmmP0FileName);

  std::vector<WordIndex> extendWithNullWordAlig(const std::vector<WordIndex>& srcWordIndexVec);
  double unsmoothed_logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
};

