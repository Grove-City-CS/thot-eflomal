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
 * @file WeightUpdateUtils.h
 * @brief Defines string processing utilities
 */

#ifndef _WeightUpdateUtils_h
#define _WeightUpdateUtils_h

extern "C" {
#include "step_by_step_dhs.h"
}

#include "InversePhraseModelFeat.h"
#include "DirectPhraseModelFeat.h"
#include "PhraseExtractUtils.h"
#include "WordGraph.h"
#include "LM_State.h"
#include "BaseSwAligModel.h"
#include "BasePhraseModel.h"
#include "BaseNgramLM.h"
#include "PhrasePair.h"
#include "BaseLogLinWeightUpdater.h"
#include <stdio.h>
#include <string>
#include <vector>

//--------------- Constants ------------------------------------------

#define NBLIST_SIZE_FOR_LLWEIGHT_UPDATE 1000
#define PHRSWLITM_DHS_FTOL                 0.001
#define PHRSWLITM_DHS_SCALE_PAR            1

namespace WeightUpdateUtils
{
  void updateLogLinearWeights(std::string refSent,
                              WordGraph* wgPtr,
                              BaseLogLinWeightUpdater* llWeightUpdaterPtr,
                              const std::vector<std::pair<std::string,float> >& compWeights,
                              std::vector<float>& newWeights,
                              int verbose=0);
  template <class THypScoreInfo>
  int updatePmLinInterpWeights(std::string srcCorpusFileName,
                               std::string trgCorpusFileName,
                               DirectPhraseModelFeat<THypScoreInfo>* dirPhrModelFeatPtr,
                               InversePhraseModelFeat<THypScoreInfo>* invPhrModelFeatPtr,
                               int verbose=0);  
}

#endif
