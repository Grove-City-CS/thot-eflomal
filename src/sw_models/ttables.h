// Copyright 2013 by Chris Dyer
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef _TTABLES_H_
#define _TTABLES_H_

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include "hashtables.h"
#include "SingleWordVocab.h"

struct Md
{
  static double digamma(double x)
  {
    double result = 0, xx, xx2, xx4;
    for (; x < 7; ++x)
      result -= 1 / x;
    x -= 1.0 / 2.0;
    xx = 1.0 / x;
    xx2 = xx * xx;
    xx4 = xx2 * xx2;
    result += log(x) + (1. / 24.) * xx2 - (7.0 / 960.0) * xx4 + (31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4
      * xx4;
    return result;
  }
  static inline double log_poisson(unsigned x, const double& lambda)
  {
    assert(lambda > 0.0);
    return std::log(lambda) * x - lgamma(x + 1) - lambda;
  }
};

class TTable
{
public:
  TTable() : frozen_(false), probs_initialized_(false) {}

  typedef std::vector<Word2Double> Word2Word2Double;

  inline double prob(const WordIndex e, const WordIndex f) const
  {
    return probs_initialized_ ? ttable[e].find(f)->second : 1e-9;
  }

  inline double safeProb(const WordIndex& e, const WordIndex& f) const
  {
    if (e < static_cast<int>(ttable.size())) {
      const Word2Double& cpd = ttable[e];
      const Word2Double::const_iterator it = cpd.find(f);
      if (it == cpd.end()) return 1e-9;
      return it->second;
    }
    else {
      return 1e-9;
    }
  }

  inline void getEntriesForF(const WordIndex f, std::map<WordIndex, Prob>& entries) const
  {
    for (WordIndex i = 0; i < ttable.size(); ++i) {
      const Word2Double& cpd = ttable[i];
      for (Word2Double::const_iterator it = cpd.begin(); it != cpd.end(); ++it)
      {
        if (it->first == f)
          entries[i] = it->second;
      }
    }
  }

  inline void setMaxE(const WordIndex e)
  {
    // NOT thread safe
    if (e >= counts.size())
      counts.resize((size_t)e + 1);
  }

  inline void insert(const WordIndex e, const WordIndex f)
  {
    // NOT thread safe
    if (e >= counts.size())
      counts.resize((size_t)e + 1);
    counts[e][f] = 0;
  }

  inline void increment(const WordIndex e, const WordIndex f, const double x)
  {
#pragma omp atomic
    counts[e].find(f)->second += x;
  }

  void normalizeVB(const double alpha)
  {
    ttable.swap(counts);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ttable.size(); ++i) {
      double tot = 0;
      Word2Double& cpd = ttable[i];
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        tot += it->second + alpha;
      if (!tot) tot = 1;
      const double digamma_tot = Md::digamma(tot);
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        it->second = exp(Md::digamma(it->second + alpha) - digamma_tot);
    }
    clearCounts();
    probs_initialized_ = true;
  }

  void normalize()
  {
    ttable.swap(counts);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ttable.size(); ++i) {
      double tot = 0;
      Word2Double& cpd = ttable[i];
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        tot += it->second;
      if (!tot) tot = 1;
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        it->second /= tot;
    }
    clearCounts();
    probs_initialized_ = true;
  }

  void freeze()
  {
    // duplicate all values in counts into ttable
    // later updates to both are semi-threadsafe
    assert(!frozen_);
    if (!frozen_) {
      ttable.resize(counts.size());
      for (unsigned i = 0; i < counts.size(); ++i) {
        ttable[i] = counts[i];
      }
    }
    frozen_ = true;
  }
  // adds counts from another TTable - probabilities remain unchanged
  TTable& operator+=(const TTable& rhs)
  {
    if (rhs.counts.size() > counts.size()) counts.resize(rhs.counts.size());
    for (unsigned i = 0; i < rhs.counts.size(); ++i) {
      const Word2Double& cpd = rhs.counts[i];
      Word2Double& tgt = counts[i];
      for (Word2Double::const_iterator j = cpd.begin(); j != cpd.end(); ++j) {
        tgt[j->first] += j->second;
      }
    }
    return *this;
  }
  void exportToFile(const std::string& filename, SingleWordVocab& vocab, double BEAM_THRESHOLD) const
  {
    std::ofstream file(filename);
    for (unsigned i = 0; i < ttable.size(); ++i) {
      const std::string& a = vocab.wordIndexToSrcString(i);
      const Word2Double& cpd = ttable[i];
      double max_p = -1;
      for (auto& it : cpd)
        if (it.second > max_p) max_p = it.second;
      const double threshold = -log(max_p) * BEAM_THRESHOLD;
      for (auto& it : cpd) {
        const std::string& b = vocab.wordIndexToTrgString(it.first);
        double c = log(it.second);
        if (c >= threshold)
          file << a << '\t' << b << '\t' << c << std::endl;
      }
    }
    file.close();
  }

  void clear()
  {
    ttable.clear();
    counts.clear();
    frozen_ = false;
    probs_initialized_ = false;
  }

private:
  void clearCounts()
  {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < counts.size(); ++i) {
      for (auto& cnt : counts[i]) {
        cnt.second = 0.0;
      }
    }
  }

  Word2Word2Double ttable;
  Word2Word2Double counts;
  bool frozen_; // Disallow new e,f pairs to be added to counts
  bool probs_initialized_; // If we can use the values in probs

public:
  void deserializeLogProbsFromText(const std::string& filename, SingleWordVocab& d);
};

#endif
