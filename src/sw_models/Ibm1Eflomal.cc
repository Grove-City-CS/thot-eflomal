#include "sw_models/Ibm1Eflomal.h"

#include "nlp_common/ErrorDefs.h"
#include "sw_models/Md.h"
#include "sw_models/MemoryLexTable.h"
#include "sw_models/SwDefs.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <vector>

using namespace std;

Ibm1Eflomal::Ibm1Eflomal()
{
}

void Ibm1Eflomal::batchUpdateCounts(const vector<pair<vector<WordIndex>, vector<WordIndex>>>& pairs)
{
  for (int line_idx = 0; line_idx < (int)pairs.size(); ++line_idx)
  {
    vector<WordIndex> osrc = pairs[line_idx].first;
    vector<WordIndex> src = extendWithNullWord(osrc);
    vector<WordIndex> trg = pairs[line_idx].second;
    vector<float> ps(src.size());
    for (PositionIndex j = 0; j < trg.size(); j++)
    {
      WordIndex t = trg[j];
      // get the word index in the source sentence that previously mapped to this word
      WordIndex old_i = links[line_idx][j];
      // get that word token from the source sentence
      WordIndex old_s = -1;
      if (old_i == NULL_LINK)
      {
        old_s = NULL_WORD;
      }
      else
      {
        old_s = src[old_i];
      }
      pair<WordIndex, WordIndex> pairToUpdate = {t, old_s};
      if (counts.find(pairToUpdate) != counts.end()) // this should always be true
      {
        if (counts[pairToUpdate] <= 1)
        {
          // if counts reaches 0 clear these entries for RAM
          counts.erase(pairToUpdate);
          if (dirichlet.size() > t)
          {
            dirichlet[t].erase(old_s);
          }
        }
        else
        {
          // decrease the count and dirichlet prior of the word old_s
          counts.insert({pairToUpdate, counts.at(pairToUpdate) - 1});
          float dirichletVal = LEX_ALPHA; // handle case where dirichlet is null
          if (dirichlet.size() > t && dirichlet[t].find(old_s) != dirichlet[t].end())
          {
            dirichletVal = dirichlet[t][old_s];
          }
          float newDirichletVal = 1 / (1 / dirichletVal - 1);
          if (dirichlet.size() <= t)
          {
            while (dirichlet.size() <= t) // is this the best way to do this?
            {
              map<WordIndex, float> map;
              dirichlet.push_back(map);
            }
          }
          dirichlet[t].insert({old_s, newDirichletVal});
        }
      }

      // update probabilities assuming this pair is unmapped
      float ps_sum = 0.0;
      for (size_t i = 0; i < src.size(); i++)
      {
        WordIndex s = src[i]; // ith word in src
        // get the number of times that t is caused by s
        int n = 0;
        if (counts.find(pair<WordIndex, WordIndex>(t, s)) != counts.end())
        {
          n = counts.at(pair<WordIndex, WordIndex>(t, s));
        }
        // get the prior count of t caused by s
        float alpha = LEX_ALPHA;
        if (priors.size() > t) // do we care about priors?
        {
          alpha += priors[t][s];
        }
        // multiply the estimated probabilities (dirichlet) by the counts to get quality
        float dirichletVal = LEX_ALPHA;
        if (dirichlet.size() > t && dirichlet[t].find(s) != dirichlet[t].end())
        {
          dirichletVal = dirichlet[t][s];
        }
        ps_sum += dirichletVal * (alpha + n);
        // add this number to the cumulative probability distribution
        ps.push_back(ps_sum);
        // include null word in the sum
        double dirichletValNull = LEX_ALPHA;
        if (dirichlet.size() > t && dirichlet[t].find(NULL_WORD) != dirichlet[t].end())
        {
          dirichletValNull = dirichlet[t][NULL_WORD];
        }
        int nullWordCount = 0;
        if (counts.find(pair<WordIndex, WordIndex>(t, NULL_WORD)) != counts.end())
        {
          nullWordCount = counts.at(pair<WordIndex, WordIndex>(t, NULL_WORD));
        }
        ps_sum += NULL_PRIOR * dirichletValNull * (NULL_ALPHA + nullWordCount);
      }
      ps.push_back(ps_sum);

      // determine based on ps_sum which source token caused the target token

      // the probability of any i is proportional to its probability in ps
      size_t new_i = random_categorical_from_cumulative(ps);
      int new_s = -1;
      if (new_i < src.size())
      {
        new_s = src[new_i];
        links[line_idx][j] = new_i;
      }
      else
      {
        new_s = NULL_WORD;
        links[line_idx][j] = NULL_LINK;
      }

      // increase the count and dirichlet variables to reflect the new i and s

      pair<WordIndex, WordIndex> pairToUpdate2 = {t, new_s};
      if (counts.find(pairToUpdate2) != counts.end())
      {
        counts[pairToUpdate2] = counts[pairToUpdate2] + 1;
      }
      else
      {
        counts.insert({pairToUpdate2, 1});
      }
      float dirichletVal = LEX_ALPHA;
      if (dirichlet.size() > t && dirichlet[t].find(new_s) != dirichlet[t].end())
      {
        dirichletVal = dirichlet[t][new_s];
      }
      if (dirichlet.size() <= t)
      {
        while (dirichlet.size() <= t) // is this the best way to do this?
        {
          map<WordIndex, float> map;
          dirichlet.push_back(map);
        }
      }
      dirichlet[t][new_s] = 1 / (1 / dirichletVal + 1);
    }
  }
}

// difference from original: get rid of everything to do with lexCounts, keep lexTable
void Ibm1Eflomal::addTranslationOptions(vector<vector<WordIndex>>& insertBuffer)
{
  WordIndex maxSrcWordIndex = (WordIndex)insertBuffer.size() - 1;
  lexTable->reserveSpace(maxSrcWordIndex);
}

// use values from dirichlet instead of lexCounts
void Ibm1Eflomal::batchMaximizeProbs()
{
#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)dirichlet.size(); ++s)
  {
    double denom = 0;
    map<WordIndex, float> elem = dirichlet[s];
    for (auto& pair : elem)
    {
      double numer = pair.second; // numerator dirichlet, denominator for src word and sum across
      if (variationalBayes)
        numer += alpha;
      denom += numer;
      lexTable->setNumerator(s, pair.first, (float)log(numer));
      pair.second = 0.0;
    }
    if (denom == 0)
      denom = 1;
    lexTable->setDenominator(s, (float)log(denom));
  }
}

// clear links instead of lexCounts
void Ibm1Eflomal::clearTempVars()
{
  links.clear();
}

void Ibm1Eflomal::initSentencePair(const vector<WordIndex>& src, const vector<WordIndex>& trg)
{
  vector<PositionIndex> newLink;
  for (size_t j = 0; j < trg.size(); j++)
  {
    size_t linkIndex = rand() % (src.size() + 1); // src does not include the null word, so add 1 to the length
    newLink.push_back(linkIndex);                 // assumes that we are adding links for each sentence chronologically

    // add the trg word / src word pair to counts, accounting for src not containing the null word
    WordIndex srcWord = 0;
    if (linkIndex != 0)
    {
      srcWord = src[linkIndex - 1];
    }
    pair<WordIndex, WordIndex> pairToUpdate = {trg[j], srcWord};
    if (counts.find(pairToUpdate) != counts.end())
    {
      counts[pairToUpdate] = counts[pairToUpdate] + 1;
    }
    else
    {
      counts[pairToUpdate] = 1;
    }
  }
  links.push_back(newLink);
}

size_t Ibm1Eflomal::random_categorical_from_cumulative(vector<float> ps)
{
  float max = ps[ps.size() - 1];
  double randomVal = rand() / (RAND_MAX + 1.0) * max;
  for (size_t i = 0; i < ps.size() - 1; i++)
  {
    if (ps[i] >= randomVal)
    {
      return i;
    }
  }
  return ps.size() - 1;
}
