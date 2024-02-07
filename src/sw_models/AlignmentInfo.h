#pragma once

#include "nlp_common/PositionIndex.h"

#include <vector>

struct CeptNode
{
public:
  PositionIndex prev;
  PositionIndex next;
};

class AlignmentInfo
{
public:
  AlignmentInfo(PositionIndex slen, PositionIndex tlen)
      : slen{slen}, tlen{tlen}, alignment(tlen, 0), positionSum(size_t{slen} + 1, 0), fertility(size_t{slen} + 1, 0),
        heads(size_t{slen} + 1, 0), ceptNodes(size_t{tlen} + 1)
  {
    // Alignment where null word generated all words in target sentence

    fertility[0] = tlen;
    for (PositionIndex j = 1; j <= tlen; ++j)
    {
      if (j > 1)
        ceptNodes[j].prev = j - 1;
      if (j < tlen)
        ceptNodes[j].next = j + 1;
    }
    heads[0] = 1;
  }

  PositionIndex getSourceLength() const
  {
    return slen;
  }

  PositionIndex getTargetLength() const
  {
    return tlen;
  }

  const std::vector<PositionIndex>& getAlignment() const
  {
    return alignment;
  }

  void setAlignment(const std::vector<PositionIndex>& alignment)
  {
    for (PositionIndex j = 1; j <= tlen; ++j)
      set(j, alignment[j - 1]);
  }

  PositionIndex get(PositionIndex j) const
  {
    return alignment[j - 1];
  }

  void set(PositionIndex j, PositionIndex i)
  {
    PositionIndex iOld = alignment[j - 1];
    positionSum[iOld] -= j;

    // unlink j from the old cept's linked list
    PositionIndex prev = ceptNodes[j].prev;
    PositionIndex next = ceptNodes[j].next;
    if (next > 0)
      ceptNodes[next].prev = prev;
    if (prev > 0)
      ceptNodes[prev].next = next;
    else
      heads[iOld] = next;

    // link j into the proper place in the new cept's linked list
    next = heads[i];
    prev = 0;
    while (next > 0 && next < j)
    {
      prev = next;
      next = ceptNodes[next].next;
    }

    ceptNodes[j].prev = prev;
    ceptNodes[j].next = next;
    if (prev > 0)
      ceptNodes[prev].next = j;
    else
      heads[i] = j;
    if (next > 0)
      ceptNodes[next].prev = j;

    fertility[iOld]--;
    positionSum[i] += j;
    fertility[i]++;
    alignment[j - 1] = i;
  }

  PositionIndex getFertility(PositionIndex i) const
  {
    return fertility[i];
  }

  bool isHead(PositionIndex j) const
  {
    PositionIndex i = get(j);
    return heads[i] == j;
  }

  PositionIndex getCenter(PositionIndex i) const
  {
    if (i == 0)
      return 0;

    return (positionSum[i] + fertility[i] - 1) / fertility[i];
  }

  PositionIndex getPrevCept(PositionIndex i) const
  {
    if (i == 0)
      return 0;
    PositionIndex k = i - 1;
    while (k > 0 && fertility[k] == 0)
      k--;
    return k;
  }

  PositionIndex getNextCept(PositionIndex i) const
  {
    PositionIndex k = i + 1;
    while (k < slen + 1 && fertility[k] == 0)
      k++;
    return k;
  }

  PositionIndex getPrevInCept(PositionIndex j) const
  {
    return ceptNodes[j].prev;
  }

  bool isValid(PositionIndex maxFertility) const
  {
    if (2 * fertility[0] > tlen)
      return false;

    for (PositionIndex i = 1; i <= slen; ++i)
    {
      if (fertility[i] >= maxFertility)
        return false;
    }
    return true;
  }

private:
  /// @brief  Length of the source sentence, not including null word
  PositionIndex slen;

  /// @brief Length of the target sentence
  PositionIndex tlen;

  /// @brief alignment[j] is the source word index for target word at index j
  /// j is in [0, tlen-1]
  /// alignment[j] is in [0, slen]
  std::vector<PositionIndex> alignment;

  /// @brief positionSum[i] is the sum of the 1-indexed positions of the target words that map to source word i
  /// i is in [0, slen]
  std::vector<PositionIndex> positionSum;

  /// @brief fertility[i] is the number of target words that map to source word i
  /// i is in [0, slen]
  std::vector<PositionIndex> fertility;


  /* From https://www.nltk.org/_modules/nltk/translate/ibm4.html:
  :Cept:
    A source word with non-zero fertility i.e. aligned to one or more
    target words.
  :Tablet:
    The set of target word(s) aligned to a cept.
  :Head of cept:
    The first word of the tablet of that cept.
  */

  /// @brief heads[i] is the head of the cept i
  /// i is in [0, slen] 
  std::vector<PositionIndex> heads;

  /// @brief ceptNodes[j].prev is the target index of the preceding word in
  /// the same tablet as j; .next is the following word in the same tablet
  /// Either of them set to 0 indicates no prev or next, since indices in target
  /// sentence are 1-indexed
  std::vector<CeptNode> ceptNodes;
};
