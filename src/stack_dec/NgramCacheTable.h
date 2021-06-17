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
 * @file NgramCacheTable.h
 *
 * @brief Class for caching information related to n-gram language
 * model.
 */

#ifndef _NgramCacheTable_h
#define _NgramCacheTable_h

//--------------- Include files --------------------------------------

#include THOT_LM_STATE_H // Define LM_State type. It is set in
                         // configure by checking LM_STATE_H
                         // variable (default value: LM_State.h)
#include <Score.h>
#include <map>
#include <utility>

//--------------- Classes --------------------------------------------

typedef std::map<pair<WordIndex, LM_State>, Score> NgramCacheTable;

#endif
