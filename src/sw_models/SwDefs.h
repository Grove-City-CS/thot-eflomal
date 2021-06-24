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
 * @file SwDefs.h
 *
 * @brief Constants, typedefs and basic classes used in the single-word
 * model classes.
 *
 */

#pragma once

//--------------- Include files --------------------------------------

#include "nlp_common/Count.h"
#include "nlp_common/PositionIndex.h"
#include "sw_models/SentPairCont.h"

#include <map>
#include <string>
#include <vector>

//--------------- Constants ------------------------------------------

constexpr double SW_PROB_SMOOTH = 1e-7;
const double SW_LOG_PROB_SMOOTH = log(SW_PROB_SMOOTH);
constexpr PositionIndex IBM_SWM_MAX_SENT_LENGTH = 1024;
constexpr PositionIndex HMM_SWM_MAX_SENT_LENGTH = 200;
constexpr PositionIndex IBM4_SWM_MAX_SENT_LENGTH = 200;
