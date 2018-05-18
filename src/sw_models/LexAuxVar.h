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
 
#ifndef _LexAuxVar_h
#define _LexAuxVar_h

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "SwDefs.h"

#ifdef THOT_DISABLE_SPACE_EFFICIENT_LEXDATA_STRUCTURES

#if __GNUC__>2
#include <ext/hash_map>
using __gnu_cxx::hash_map;
#else
#include <hash_map>
#endif

#else

#include <OrderedVector.h>

#endif

//--------------- typedefs -------------------------------------------


#ifdef THOT_DISABLE_SPACE_EFFICIENT_LEXDATA_STRUCTURES
typedef hash_map<WordIndex,std::pair<float,float> > LexAuxVarElem;
typedef std::vector<LexAuxVarElem> LexAuxVar;
#else
typedef OrderedVector<WordIndex,std::pair<float,float> > LexAuxVarElem;
typedef std::vector<LexAuxVarElem> LexAuxVar;
#endif

//--------------- Classes ---------------------------------------------


#endif
