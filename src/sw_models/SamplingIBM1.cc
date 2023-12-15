#include "sw_models/SamplingIBM1.h"
#include <iostream>
#include "nlp_common/ErrorDefs.h"
#include "sw_models/Md.h"
#include "sw_models/MemoryLexTable.h"
#include "sw_models/SwDefs.h"
#include <vector>
#include <map>
#include <algorithm>
#include <random>

SamplingIBM1::SamplingIBM1(){
    links = std::vector<std::vector<WordIndex>>();
    counts = std::map<std::pair<u_int32_t, u_int32_t>, float>();
    dirichlet = std::vector<std::map<u_int32_t, float>>();
    priors = std::vector<std::map<u_int32_t, float>>();
}


// missing null word
void SamplingIBM1::initializeLinkAndCounts(std::vector<std::pair<std::vector<int>, std::vector<int>>> pairs) {
     links = std::vector<std::vector<WordIndex>>();
     counts = std::map<std::pair<WordIndex, WordIndex>, float>();
    
    // Use a random_device to seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < sizeof(pairs); i++) {
        std::vector<WordIndex> ar = std::vector<WordIndex>();
        links.push_back(ar);
        int srcLength = pairs[i].first.size();
        int trgLength = pairs[i].second.size(); 
        
        //Define a uniform distribution for integers
        std::uniform_int_distribution<> distribution(1, srcLength);
        for (size_t j = 0; j < trgLength; j++) { 
            
        // Generate a random number
        int linkIndex = distribution(gen);
        
        //adding the index to the links 
        links[i].push_back(linkIndex); 
        
        // create a random match for one of the token 
        std::pair<int, int> pairToUpdate = std::pair(pairs[i].second[j], pairs[i].first[linkIndex]); 
        if (counts.find(pairToUpdate)!=counts.end()) { 
            counts[pairToUpdate] = counts[pairToUpdate] + 1;
        } 
        else { counts[pairToUpdate] = 1;} 
        } 
    }   
    std::cout << " the link data structure: " << std::endl; 
    for (int i = 0; i < links.size(); i++){ 
        std::cout << "For "<< i << std::endl; 
        for (int j = 0; j < links[i].size(); j++){ 
            std::cout << links[i][j]; 
            std::cout << " "; 
        } 
        std::cout << std::endl; 
    }
    
    std::cout << "The count data structure: " << std::endl; 
    for (const auto& el : counts) { 
        std::cout << " (" << el.first.first <<", " << el.first.second << "): " << el.second; 
        }
        return;
} 

// virtual void batchMaximizeProbs() {

// }
// virtual void batchUpdateCounts(const vector<pair<vector<WordIndex>, vector<WordIndex>>>& pairs){

// }
    



