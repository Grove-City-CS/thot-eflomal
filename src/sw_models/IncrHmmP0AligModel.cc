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
 * @file IncrHmmP0AligModel.cc
 *
 * @brief Definitions file for IncrHmmP0AligModel.h
 */

//--------------- Include files --------------------------------------

#include "sw_models/IncrHmmP0AligModel.h"

#include "sw_models/MemoryLexTable.h"

//--------------- IncrHmmP0AligModel class function definitions

//-------------------------
IncrHmmP0AligModel::IncrHmmP0AligModel() : _incrHmmP0AligModel()
{
  // Create table with lexical parameters
  lexTable = new MemoryLexTable();
  lexNumDenFileExtension = ".hmm_lexnd";
}
