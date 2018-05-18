/*
thot package for statistical machine translation
Copyright (C) 2017 Adam Harasimowicz, Daniel Ortiz-Mart\'inez

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
 * @file HatTriePhraseTable.cc
 * 
 * @brief Definitions file for HatTriePhraseTable.h
 */

//--------------- Include files --------------------------------------

#include "HatTriePhraseTable.h"

//--------------- Function definitions

//-------------------------
HatTriePhraseTable::HatTriePhraseTable(void)
{
}

//-------------------------
std::string HatTriePhraseTable::vectorToStdString(const std::vector<WordIndex>& vec)const
{
    std::vector<WordIndex> str;
    for(size_t i = 0; i < vec.size(); i++) {
        // Use WORD_INDEX_MODULO_BYTES bytes to encode index
        for(int j = WORD_INDEX_MODULO_BYTES - 1; j >= 0; j--) {
            str.push_back(1 + (vec[i] / (unsigned int) pow(WORD_INDEX_MODULO_BASE, j) % WORD_INDEX_MODULO_BASE));
        }
    }

    std::string s(str.begin(), str.end());

    return s;
}

//-------------------------
std::vector<WordIndex> HatTriePhraseTable::stringToVector(const std::string s)const
{
    std::vector<WordIndex> vec;

    for(size_t i = 0; i < s.size();)  // A string length is WORD_INDEX_MODULO_BYTES * n + 1
    {
        unsigned int wi = 0;
        for(int j = WORD_INDEX_MODULO_BYTES - 1; j >= 0; j--, i++) {
            wi += (((unsigned char) s[i]) - 1) * (unsigned int) pow(WORD_INDEX_MODULO_BASE, j);
        }

        vec.push_back(wi);
    }

    return vec;
}

//-------------------------
std::string HatTriePhraseTable::vectorToKey(const std::vector<WordIndex>& vec)const
{
    return vectorToStdString(vec);
}

//-------------------------
std::vector<WordIndex> HatTriePhraseTable::keyToVector(const std::string key)const
{
    return stringToVector(key);
}

//-------------------------
std::vector<WordIndex> HatTriePhraseTable::getSrc(const std::vector<WordIndex>& s)
{
    // Prepare s vector as (UNUSED_WORD, s)
    std::vector<WordIndex> s_vec;
    s_vec.push_back(UNUSED_WORD);
    s_vec.insert(s_vec.end(), s.begin(), s.end());

    return s_vec;
}

//-------------------------
std::vector<WordIndex> HatTriePhraseTable::getSrcTrg(const std::vector<WordIndex>& s,
                                                     const std::vector<WordIndex>& t)
{
    // Prepare (s,t) vector as (UNUSED_WORD, s, UNUSED_WORD, t)
    std::vector<WordIndex> st_vec = getSrc(s);
    st_vec.push_back(UNUSED_WORD);
    st_vec.insert(st_vec.end(), t.begin(), t.end());

    return st_vec;
}

//-------------------------
std::vector<WordIndex> HatTriePhraseTable::getTrgSrc(const std::vector<WordIndex>& s,
                                                     const std::vector<WordIndex>& t)
{
    // Prepare (s,t) vector as (t, UNUSED_WORD, s)
    std::vector<WordIndex> st_vec = t;
    st_vec.push_back(UNUSED_WORD);
    st_vec.insert(st_vec.end(), s.begin(), s.end());

    return st_vec;
}

//-------------------------
bool HatTriePhraseTable::getNbestForSrc(const std::vector<WordIndex>& s,
                                        NbestTableNode<PhraseTransTableNodeData>& nbt)
{
    HatTriePhraseTable::TrgTableNode::iterator iter;

    bool found;
    Count s_count;
    HatTriePhraseTable::TrgTableNode node;
    LgProb lgProb;

    // Make sure that collection does not contain any old elements
    nbt.clear();

    found = getEntriesForSource(s, node);
    s_count = cSrc(s);

    if(found) {
        // Generate transTableNode
        for(iter = node.begin(); iter != node.end(); iter++)
        {
            std::vector<WordIndex> t = iter->first;
            PhrasePairInfo ppi = (PhrasePairInfo) iter->second;
            float c_st = (float) ppi.second.get_c_st();
            lgProb = log(c_st / (float) s_count);
            nbt.insert(lgProb, t); // Insert pair <log probability, target phrase>
        }

#   ifdef DO_STABLE_SORT_ON_NBEST_TABLE
        // Performs stable sort on n-best table, this is done to ensure
        // that the n-best lists generated by cache models and
        // conventional models are identical. However this process is
        // time consuming and must be avoided if possible
        nbt.stableSort();
#   endif
        return true;
    }
    else
    {
        // Cannot find the source phrase
        return false;
    }
}
//-------------------------
bool HatTriePhraseTable::getNbestForTrg(const std::vector<WordIndex>& t,
                                        NbestTableNode<PhraseTransTableNodeData>& nbt,
                                        int N)
{
    HatTriePhraseTable::SrcTableNode::iterator iter;

    bool found;
    Count t_count;
    HatTriePhraseTable::SrcTableNode node;
    LgProb lgProb;

    // Make sure that collection does not contain any old elements
    nbt.clear();

    found = getEntriesForTarget(t, node);
    t_count = cTrg(t);

    if(found) {
        // Generate transTableNode
        for(iter = node.begin(); iter != node.end(); iter++)
        {
            std::vector<WordIndex> s = iter->first;
            PhrasePairInfo ppi = (PhrasePairInfo) iter->second;
            float c_st = (float) ppi.second.get_c_st();
            lgProb = log(c_st / (float) t_count);
            nbt.insert(lgProb, s); // Insert pair <log probability, source phrase>
        }

#   ifdef DO_STABLE_SORT_ON_NBEST_TABLE
        // Performs stable sort on n-best table, this is done to ensure
        // that the n-best lists generated by cache models and
        // conventional models are identical. However this process is
        // time consuming and must be avoided if possible
        nbt.stableSort();
#   endif

        while(nbt.size() > (unsigned int) N && N >= 0)
        {
            // node contains N inverse translations, remove last element
            nbt.removeLastElement();
        }

        return true;
    }
    else
    {
        // Cannot find the target phrase
        return false;
    }
}

//-------------------------
void HatTriePhraseTable::addTableEntry(const std::vector<WordIndex>& s,
                                       const std::vector<WordIndex>& t,
                                       PhrasePairInfo inf)
{
    Count t_count = cTrg(t);

    addSrcInfo(s, inf.first.get_c_s());  // src
    // Values for target are not summed with the old one thus they have to aggregated here
    addTrgInfo(t, (t_count + inf.second).get_c_s());  // trg
    addSrcTrgInfo(s, t, inf.second.get_c_st());  // (src, trg)
}

//-------------------------
void HatTriePhraseTable::addSrcInfo(const std::vector<WordIndex>& s,
                                    Count s_inf)
{
    std::string srcKey = vectorToKey(getSrc(s));
    phraseTable[srcKey.c_str()] = s_inf;
}

//-------------------------
void HatTriePhraseTable::addTrgInfo(const std::vector<WordIndex>& t,
                                    Count t_inf)
{
    std::string trgKey = vectorToKey(t);
    phraseTable[trgKey.c_str()] = t_inf;
}

//-------------------------
void HatTriePhraseTable::addSrcTrgInfo(const std::vector<WordIndex>& s,
                                       const std::vector<WordIndex>& t,
                                       Count st_inf)
{
    std::string trgSrcKey = vectorToKey(getTrgSrc(s, t));
    phraseTable[trgSrcKey.c_str()] = st_inf;
}

//-------------------------
void HatTriePhraseTable::incrCountsOfEntry(const std::vector<WordIndex>& s,
                                           const std::vector<WordIndex>& t,
                                           Count c)
{
    // Retrieve previous states
    Count s_count = cSrc(s);
    Count t_count = cTrg(t);
    Count src_trg_count = cSrcTrg(s, t);

    // Update counts
    addSrcInfo(s, s_count + c);  // src
    addTrgInfo(t, t_count + c);  // trg
    addSrcTrgInfo(s, t, (src_trg_count + c).get_c_st());  // (src, trg)
}

//-------------------------
PhrasePairInfo HatTriePhraseTable::infSrcTrg(const std::vector<WordIndex>& s,
                                             const std::vector<WordIndex>& t,
                                             bool& found)
{
    PhrasePairInfo ppi;

    ppi.first = getSrcInfo(s, found);
    if(!found)
    {
        ppi.second = 0;
        return ppi;
    }
    else
    {
        ppi.second = getSrcTrgInfo(s, t, found);
        return ppi;
    }
}

//-------------------------
Count HatTriePhraseTable::getSrcInfo(const std::vector<WordIndex>& s,
                                     bool &found)
{
    std::string srcKey = vectorToKey(getSrc(s));
    PhraseTable::iterator iter = phraseTable.find(srcKey.c_str());

    if (iter == phraseTable.end())  // Check if s exists in collection
    {
        found = false;
        return 0;
    }
    else
    {
        found = true;
        return iter.value();
    }
}

//-------------------------
Count HatTriePhraseTable::getTrgInfo(const std::vector<WordIndex>& t,
                                     bool &found)
{
    std::string trgKey = vectorToKey(t);
    PhraseTable::iterator iter = phraseTable.find(trgKey.c_str());

    if (iter == phraseTable.end())  // Check if t exists in collection
    {
        found = false;
        return 0;
    }
    else
    {
        found = true;
        return iter.value();
    }
}

//-------------------------
Count HatTriePhraseTable::getSrcTrgInfo(const std::vector<WordIndex>& s,
                                        const std::vector<WordIndex>& t,
                                        bool &found)
{
    std::string trgSrcKey = vectorToKey(getTrgSrc(s, t));
    PhraseTable::iterator iter = phraseTable.find(trgSrcKey);

    // // Check if entry for (s, t) pair exists
    if (iter == phraseTable.end())
    {
        found = false;
        return 0;
    }
    else
    {
        found = true;
        return iter.value();
    }
}

//-------------------------
Prob HatTriePhraseTable::pTrgGivenSrc(const std::vector<WordIndex>& s,
                                      const std::vector<WordIndex>& t)
{
    // p(s|t) = count(s,t) / count(t)
    Count st_count = cSrcTrg(s, t);
    if ((float) st_count > 0)
    {
        Count s_count = cSrc(s);
        if ((float) s_count > 0)
            return ((float) st_count) / ((float) s_count);
        else
            return PHRASE_PROB_SMOOTH;
    }
    else return PHRASE_PROB_SMOOTH;
}

//-------------------------
LgProb HatTriePhraseTable::logpTrgGivenSrc(const std::vector<WordIndex>& s,
                                           const std::vector<WordIndex>& t)
{
    return log((double) pTrgGivenSrc(s, t));
}

//-------------------------
Prob HatTriePhraseTable::pSrcGivenTrg(const std::vector<WordIndex>& s,
                                      const std::vector<WordIndex>& t)
{
    Count count_s_t_ = cSrcTrg(s, t);
    if((float) count_s_t_ > 0)
    {
        Count count_t_ = cTrg(t);
	    if((float) count_t_ > 0)
        {
            return (float) count_s_t_ / (float) count_t_;
        }
	    else return PHRASE_PROB_SMOOTH;
    }
    else return PHRASE_PROB_SMOOTH;
}

//-------------------------
LgProb HatTriePhraseTable::logpSrcGivenTrg(const std::vector<WordIndex>& s,
                                           const std::vector<WordIndex>& t)
{
    return log((double) pSrcGivenTrg(s, t));
}

//-------------------------
bool HatTriePhraseTable::getEntriesForTarget(const std::vector<WordIndex>& t,
                                             HatTriePhraseTable::SrcTableNode& srctn)
{
    bool found;
    srctn.clear();  // Make sure that structure does not keep old values

    // Prepare iterators
    const std::vector<WordIndex> emptyVec;
    std::vector<WordIndex> trgSrcPrefix = getTrgSrc(emptyVec, t);
    std::string trgSrcPrefixStr = vectorToKey(trgSrcPrefix);

    auto prefixIterators = phraseTable.equal_prefix_range(trgSrcPrefixStr);

    for(auto iter = prefixIterators.first; iter != prefixIterators.second; iter++)
    {
        std::vector<WordIndex> vec = keyToVector(iter.key());
        std::vector<WordIndex> s(vec.begin() + t.size() + 1, vec.end());

        PhrasePairInfo ppi = infSrcTrg(s, t, found);
        if (!found || fabs(ppi.first.get_c_s()) < EPSILON || fabs(ppi.second.get_c_s()) < EPSILON)
            continue;

        srctn.insert(std::pair<std::vector<WordIndex>, PhrasePairInfo>(s, ppi));
    }

    return srctn.size();
}

//-------------------------
bool HatTriePhraseTable::getEntriesForSource(const std::vector<WordIndex>& s,
                                             HatTriePhraseTable::TrgTableNode& trgtn)
{
    trgtn.clear();  // Make sure that structure does not keep old values

    std::vector<WordIndex> srcVec = getSrc(s);  // (UNUSED_WORD, s)

    // Scan (s, t) collection to find matching elements for a given s
    for (auto iter = phraseTable.begin(); iter != phraseTable.end(); iter++)
    {
        std::vector<WordIndex> phrase = keyToVector(iter.key());

        if (phrase.size() <= srcVec.size())  // Phrase is to short to contain given source and target
            continue;

        std::vector<WordIndex> srcPhrase(phrase.end() - srcVec.size(), phrase.end());
        if (srcPhrase != srcVec)  // Found source does not match to given source
            continue;

        std::vector<WordIndex> trgPhrase(phrase.begin(), phrase.end() - srcVec.size());

        PhrasePairInfo ppi;
        ppi.first = cTrg(trgPhrase);  // t count
        ppi.second = iter.value();  // (s, t) count

        if ((int) ppi.first.get_c_s() == 0 || (int) ppi.second.get_c_s() == 0)
            continue;

        trgtn.insert(std::pair<std::vector<WordIndex>, PhrasePairInfo>(trgPhrase, ppi));
    }

    return trgtn.size();
}

//-------------------------
Count HatTriePhraseTable::cSrcTrg(const std::vector<WordIndex>& s,
                                  const std::vector<WordIndex>& t)
{
    bool found;
    return getSrcTrgInfo(s, t, found).get_c_st();
}

//-------------------------
Count HatTriePhraseTable::cSrc(const std::vector<WordIndex>& s)
{
    bool found;
    return getSrcInfo(s, found).get_c_s();
}

//-------------------------
Count HatTriePhraseTable::cTrg(const std::vector<WordIndex>& t)
{
    bool found;
    return getTrgInfo(t, found).get_c_st();
}

//-------------------------
void HatTriePhraseTable::print(void)
{
    size_t i;

    for (PhraseTable::iterator iter = phraseTable.begin(); iter != phraseTable.end(); iter++)
    {
        std::vector<WordIndex> s;
        std::vector<WordIndex> t;
        std::vector<WordIndex> phrase = keyToVector(iter.key());

        // Retrieve target
        for(i = 0; i < phrase.size(); i++)
        {
            if (phrase[i] == UNUSED_WORD)
                break;

            t.push_back(phrase[i]);
        }

        // Retrieve source
        // Increase at the begining i by 1 to skip UNUSED_WORD so it will not be included in source phrase
        for(i += 1; i < phrase.size(); i++)
        {
            s.push_back(phrase[i]);
        }

        Count c = iter.value();
        // Print on standard output
        printVector(s);
        std::cout << " ||| ";
        printVector(t);
        std::cout << " ||| ";
        std::cout << c.get_c_s() << std::endl;
    }
}

//-------------------------
void HatTriePhraseTable::printVector(const std::vector<WordIndex>& vec) const
{
    for (size_t i = 0; i < vec.size(); i ++)
    {
        std::cout << vec[i] << " ";
    }
}

//-------------------------
size_t HatTriePhraseTable::size(void)
{
    return phraseTable.size();
}
//-------------------------
void HatTriePhraseTable::clear(void)
{
    phraseTable.clear();
}

//-------------------------
bool HatTriePhraseTable::isTargetPhrase(const std::vector<WordIndex>& vec) const
{
    for(size_t i = 0; i < vec.size(); i++)
    {
        if (vec[i] == UNUSED_WORD)
            return false;
    }

    return true;
}

//-------------------------
HatTriePhraseTable::~HatTriePhraseTable(void)
{
}

//-------------------------
HatTriePhraseTable::const_iterator HatTriePhraseTable::begin(void) const
{
    if (phraseTable.size()==0)
      return end();
    // Shift the iterator to the first target phrase
    PhraseTable::const_iterator iterTrgBegin = phraseTable.begin();
    while (!isTargetPhrase(keyToVector(iterTrgBegin.key())))
    {
        iterTrgBegin++;
    }

    HatTriePhraseTable::const_iterator iter(this, iterTrgBegin);

    return iter;
}
//-------------------------
HatTriePhraseTable::const_iterator HatTriePhraseTable::end(void) const
{
    HatTriePhraseTable::const_iterator iter(this, phraseTable.end());

    return iter;
}

// const_iterator function definitions
//--------------------------
bool HatTriePhraseTable::const_iterator::operator++(void) //prefix
{
    if (ptPtr != NULL && trgIter != ptPtr->phraseTable.end())
    {
        // Shift iterator to the next target phrase
        do
        {
            trgIter++;
            if (trgIter == ptPtr->phraseTable.end())
                return false;
        } while(!ptPtr->isTargetPhrase(ptPtr->keyToVector(trgIter.key())) || trgIter.value().get_c_s() == 0);

        return true;
    }
    else
    {
        return false;
    }
}
//--------------------------
bool HatTriePhraseTable::const_iterator::operator++(int)  //postfix
{
    return operator++();
}
//--------------------------
int HatTriePhraseTable::const_iterator::operator==(const const_iterator& right)
{
    return (
        ptPtr == right.ptPtr &&
        trgIter == right.trgIter
    );
}
//--------------------------
int HatTriePhraseTable::const_iterator::operator!=(const const_iterator& right)
{
    return !((*this) == right);
}

//--------------------------
HatTriePhraseTable::PhraseInfoElement HatTriePhraseTable::const_iterator::operator*(void)
{
    return *operator->();
}

//--------------------------
const HatTriePhraseTable::PhraseInfoElement*
HatTriePhraseTable::const_iterator::operator->(void)
{
    std::vector<WordIndex> t;
    Count c = 0;

    if (ptPtr != NULL && trgIter != ptPtr->phraseTable.end())
    {
        t = ptPtr->keyToVector(trgIter.key());
        c = trgIter.value();
    }

    dataItem.first = t;
    dataItem.second = c;

    return &dataItem;
}

//-------------------------
