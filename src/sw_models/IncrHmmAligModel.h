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
 * @file IncrHmmAligModel.h
 *
 * @brief Defines the IncrHmmAligModel class.  IncrHmmAligModel class
 * allows to generate and access to the data of an HMM statistical
 * alignment model.
 *
 */

#ifndef _IncrHmmAligModel_h
#define _IncrHmmAligModel_h

//--------------- Include files --------------------------------------

#include "sw_models/IncrLexTable.h"
#include "sw_models/_incrHmmAligModel.h"

//--------------- Constants ------------------------------------------

//--------------- typedefs -------------------------------------------

//--------------- function declarations ------------------------------

//--------------- Classes --------------------------------------------

//--------------- IncrHmmAligModel class

class IncrHmmAligModel : public _incrHmmAligModel
{
public:
  // Constructor
  IncrHmmAligModel();

  void clearSentLengthModel(void);
};

#endif
