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
 
#ifndef _ashPidxPairHashF_h
#define _ashPidxPairHashF_h

//--------------- Include files --------------------------------------

#include "aSourceHmm.h"

//--------------- Classes ---------------------------------------------

class ashPidxPairHashF
{
 public:
   enum
   {
     bucket_size = 1
   };

  std::size_t operator() (const std::pair<aSourceHmm,PositionIndex>&  a1) const
  {
    return (size_t) (a1.second*16384)+(256*a1.first.prev_i)+a1.first.slen;
  }

  bool operator() (const std::pair<aSourceHmm,PositionIndex>& left,const std::pair<aSourceHmm,PositionIndex>& right) const
  {
    return left < right;
  }
};

#endif
