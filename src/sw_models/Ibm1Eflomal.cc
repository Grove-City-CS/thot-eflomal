#include "sw_models/Ibm1Eflomal.h"
#include <iostream>
#include "nlp_common/ErrorDefs.h"
#include "sw_models/Md.h"
#include "sw_models/MemoryLexTable.h"
#include "sw_models/SwDefs.h"
#include <vector>
#include <map>
#include <algorithm>
#include <random>

using namespace std;

Ibm1Eflomal::Ibm1Eflomal()
{
}

void Ibm1Eflomal::batchUpdateCounts(const vector<pair<vector<WordIndex>, vector<WordIndex>>>& pairs)
 {
    for (int line_idx = 0; line_idx < (int)pairs.size(); ++line_idx)
    {
        vector<WordIndex> src = pairs[line_idx].first;
        vector<WordIndex> nsrc = Ibm1AlignmentModel::extendWithNullWord(src);
        vector<WordIndex> trg = pairs[line_idx].second;
        vector<float> ps(nsrc.size());
        for (PositionIndex j = 1; j <= trg.size(); ++j)
        {
          WordIndex t = trg[j];
          // get the word index in the source sentence that previously mapped to this word
          WordIndex old_i = Ibm1Eflomal::links[line_idx][j];
          // get that word token from the source sentence
          WordIndex old_s = -1;
          if (old_i == Ibm1Eflomal::NULL_LINK)
          {
            old_s = NULL_WORD;
          }
          else
          {
            old_s = src[old_i];
          }
          std::pair<WordIndex, WordIndex> pairToUpdate = {t, old_s};
          if (std::find(Ibm1Eflomal::counts.begin(), Ibm1Eflomal::counts.end(), pairToUpdate)
              != Ibm1Eflomal::counts.end()) // this should always be true
          {
            if (counts.at(pairToUpdate) <= 1)
            {
              // if counts reaches 0 clear these entries for RAM
              counts.erase(pairToUpdate);
              if (dirichlet.size() > t) // TODO: check if dirichlet[t] is null
              {
                dirichlet[t].erase(old_s);
              }   
            }
            else
            {
              // decrease the count and dirichlet prior of the word old_s         
              counts.insert(std::pair<std::pair<WordIndex, WordIndex>, int>(pairToUpdate, counts.at(pairToUpdate) - 1));
              float dirichletVal = LEX_ALPHA; // handle case where dirichlet is null
              if (dirichlet.size() > t && dirichlet[t].find(old_s) != dirichlet[t].end()) // TODO: check if dirichlet[t] is null
              {
                dirichletVal = dirichlet[t].at(old_s);
              }
              float newDirichletVal = 1 / (1 / dirichletVal - 1);
              // TODO: create more space in dirichlet???
              dirichlet[t].insert(std::pair<WordIndex, float>(old_s, newDirichletVal));
            }
            
          }

          // update probabilities assuming this pair is unmapped
          float ps_sum = 0.0;
          for (int i = 0; i < src.size(); i++)
          {
              WordIndex s = src[i]; // ith word in src
            // get the number of times that t is caused by s
              int n = 0;
              if (counts.find(std::pair<WordIndex, WordIndex>(t, s)) != counts.end())
              {
              n = counts.at(std::pair<WordIndex, WordIndex>(t, s));
              }
              float alpha = LEX_ALPHA;
              // TODO: get stuff with priors
              // multiply the estimated probabilities (dirichlet) by the counts to get quality
              float dirichletVal = LEX_ALPHA;
              if (dirichlet.size() > t && dirichlet[t].find(s) != dirichlet[t].end()) // TODO: stuff
              {
              dirichletVal = dirichlet[t].at(s);
              }
              ps_sum += dirichletVal * (alpha + n);
              // add this number to the cumulative probability distribution
              ps.push_back(ps_sum);
              // include null word in the sum
              double dirichletValNull = LEX_ALPHA;
              if (dirichlet.size() > t && dirichlet[t].find(NULL_WORD) != dirichlet[t].end()) // TODO: stuff
              {
              dirichletValNull = dirichlet[t].at(NULL_WORD);
              }
              int nullWordCount = 0;
              if (counts.find(std::pair<WordIndex, WordIndex>(t, NULL_WORD)) != counts.end())
              {
              nullWordCount = counts.at(std::pair<WordIndex, WordIndex>(t, NULL_WORD));
              }
              ps_sum += NULL_PRIOR * dirichletValNull * (NULL_ALPHA + nullWordCount);
          }
          ps.push_back(ps_sum);

          // determine based on ps_sum which source token caused the target token

          // the probability of any i is proportional to its probability in ps
          int new_i = random_categorical_from_cumulative(ps);
          int new_s = -1;
          if (new_i < src.size())
          {
            new_s = src[new_i];
            links[line_idx][j] = new_i;
          }
          else
          {
            new_s = NULL_WORD;
            Ibm1Eflomal::links[line_idx][j] = NULL_LINK; // TODO: find NULL_LINK and add variable
          }

          // increase the count and dirichlet variables to reflect the new i and s
          //Ibm1Eflomal::counts[t][new_s]++;
          Ibm1Eflomal::dirichlet[t][new_s] = 1.0 / (1.0 / Ibm1Eflomal::dirichlet[t][new_s] + 1.0);
        }
    }
 }

 // get rid of everything to do with lexCounts, keep lexTable
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
        std::map<WordIndex, float> elem = dirichlet[s];
        for (auto& pair : elem)
        {
          double numer = pair.second; // numerator dirichlet, denominator for src word and sum across
          if (variationalBayes)
            numer += alpha;
          denom += numer;
          lexTable->setNumerator(s, pair.first, numer); // originally has log...don't think I need that?
          pair.second = 0.0;
        }
        if (denom == 0)
          denom = 1;
        lexTable->setDenominator(s, denom); // originally has log...don't think I need that?
    }
 }

// clear links instead of lexCounts
void Ibm1Eflomal::clearTempVars()
 {
    links.clear();
 }

void Ibm1Eflomal::initSentencePair(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg)
{
  std::vector<PositionIndex> newLink;
  for (int j = 0; j < trg.size(); j++)
    {
        int linkIndex = rand() % src.size(); // should be in range [0, src.length)
        newLink.push_back(linkIndex);
        std::pair<WordIndex, WordIndex> pairToUpdate = {trg[j], src[linkIndex]};
        if (counts.find(pairToUpdate) != counts.end())
        {
          counts[pairToUpdate] = counts[pairToUpdate] + 1;
        }
        else
        {
          counts[pairToUpdate] = 0;
        }
    }
    links.push_back(newLink);
}

int random_categorical_from_cumulative(vector<float> ps)
 {
    float max = ps[ps.size() - 1];
    double randomVal = rand() / (RAND_MAX + 1.0) * max;
    for (int i = 0; i < ps.size() - 1; i++)
    {
        if (ps[i] >= randomVal)
        {
          return i;
        }
    }
    return ps.size() - 1;
 }
    



