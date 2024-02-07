#include "sw_models/Ibm1AlignmentModel.h"

#include "nlp_common/ErrorDefs.h"
#include "sw_models/Md.h"
#include "sw_models/MemoryLexTable.h"
#include "sw_models/SwDefs.h"

#include <algorithm>

using namespace std;

#pragma once

#include "sw_models/AlignmentModelBase.h"
#include "sw_models/IncrAlignmentModel.h"
#include "sw_models/LexCounts.h"
#include "sw_models/LexTable.h"
#include "sw_models/NormalSentenceLengthModel.h"
#include "sw_models/anjiMatrix.h"

#include <memory>
#include <unordered_map>

class SamplingIBM1: public Ibm1AlignmentModel {
  
public:
  SamplingIBM1();
  void initializeLinkAndCounts(std::vector<std::pair<std::vector<int>, std::vector<int>>> pairs);


protected:

  // batch.UpdateCounts sets the LexCounts from the sentences and the LexTable.
  virtual void batchUpdateCounts(const vector<pair<vector<WordIndex>, vector<WordIndex>>>& pairs) override;

private:
  // data structures needed for the 
  std::vector<std::vector<WordIndex>> links;
  std::map<std::pair<u_int32_t, u_int32_t>, float> counts;
  std::vector<std::map<u_int32_t, float>> dirichlet;
  std::vector<std::map<u_int32_t, float>> priors;

};

