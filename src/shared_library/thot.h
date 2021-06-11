#ifndef _THOT_H_
#define _THOT_H_

#if defined _WIN32 || defined __CYGWIN__
#if defined THOT_EXPORTING
#if defined __GNUC__
#define THOT_API __attribute__((dllexport))
#else
#define THOT_API __declspec(dllexport)
#endif
#else
#if defined __GNUC__
#define THOT_API __attribute__((dllimport))
#else
#define THOT_API __declspec(dllimport)
#endif
#endif
#elif __GNUC__ >= 4
#define THOT_API __attribute__((visibility("default")))
#else
#define THOT_API
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  THOT_API void* smtModel_create(const char* swAlignClassName);

  THOT_API bool smtModel_loadTranslationModel(void* smtModelHandle, const char* tmFileNamePrefix);

  THOT_API bool smtModel_loadLanguageModel(void* smtModelHandle, const char* lmFileName);

  THOT_API void smtModel_setNonMonotonicity(void* smtModelHandle, unsigned int nomon);

  THOT_API void smtModel_setW(void* smtModelHandle, float w);

  THOT_API void smtModel_setA(void* smtModelHandle, unsigned int a);

  THOT_API void smtModel_setE(void* smtModelHandle, unsigned int e);

  THOT_API void smtModel_setHeuristic(void* smtModelHandle, unsigned int heuristic);

  THOT_API void smtModel_setOnlineTrainingParameters(void* smtModelHandle, unsigned int algorithm,
                                                     unsigned int learningRatePolicy, float learnStepSize,
                                                     unsigned int emIters, unsigned int e, unsigned int r);

  THOT_API void smtModel_setWeights(void* smtModelHandle, const float* weights, unsigned int capacity);

  THOT_API void* smtModel_getSingleWordAlignmentModel(void* smtModelHandle);

  THOT_API void* smtModel_getInverseSingleWordAlignmentModel(void* smtModelHandle);

  THOT_API bool smtModel_saveModels(void* smtModelHandle);

  THOT_API void smtModel_close(void* smtModelHandle);

  THOT_API void* decoder_create(void* smtModelHandle);

  THOT_API void decoder_setS(void* decoderHandle, unsigned int s);

  THOT_API void decoder_setBreadthFirst(void* decoderHandle, bool breadthFirst);

  THOT_API void decoder_setG(void* decoderHandle, unsigned int g);

  THOT_API void* decoder_translate(void* decoderHandle, const char* sentence);

  THOT_API unsigned int decoder_translateNBest(void* decoderHandle, unsigned int n, const char* sentence,
                                               void** results);

  THOT_API void* decoder_getWordGraph(void* decoderHandle, const char* sentence);

  THOT_API void* decoder_getBestPhraseAlignment(void* decoderHandle, const char* sentence, const char* translation);

  THOT_API bool decoder_trainSentencePair(void* decoderHandle, const char* sourceSentence, const char* targetSentence);

  THOT_API void decoder_close(void* decoderHandle);

  THOT_API unsigned int tdata_getTarget(void* dataHandle, char* target, unsigned int capacity);

  THOT_API unsigned int tdata_getPhraseCount(void* dataHandle);

  THOT_API unsigned int tdata_getSourceSegmentation(void* dataHandle, unsigned int** sourceSegmentation,
                                                    unsigned int capacity);

  THOT_API unsigned int tdata_getTargetSegmentCuts(void* dataHandle, unsigned int* targetSegmentCuts,
                                                   unsigned int capacity);

  THOT_API unsigned int tdata_getTargetUnknownWords(void* dataHandle, unsigned int* targetUnknownWords,
                                                    unsigned int capacity);

  THOT_API double tdata_getScore(void* dataHandle);

  THOT_API unsigned int tdata_getScoreComponents(void* dataHandle, double* scoreComps, unsigned int capacity);

  THOT_API void tdata_destroy(void* dataHandle);

  THOT_API unsigned int wg_getString(void* wgHandle, char* wordGraphStr, unsigned int capacity);

  THOT_API double wg_getInitialStateScore(void* wgHandle);

  THOT_API void wg_destroy(void* wgHandle);

  THOT_API void* swAlignModel_create(const char* className);

  THOT_API void* swAlignModel_open(const char* className, const char* prefFileName);

  THOT_API void swAlignModel_setVariationalBayes(void* swAlignModelHandle, bool variationalBayes);

  THOT_API bool swAlignModel_getVariationalBayes(void* swAlignModelHandle);

  THOT_API unsigned int swAlignModel_getSourceWordCount(void* swAlignModelHandle);

  THOT_API unsigned int swAlignModel_getSourceWord(void* swAlignModelHandle, unsigned int index, char* wordStr,
                                                   unsigned int capacity);

  THOT_API unsigned int swAlignModel_getTargetWordCount(void* swAlignModelHandle);

  THOT_API unsigned int swAlignModel_getTargetWord(void* swAlignModelHandle, unsigned int index, char* wordStr,
                                                   unsigned int capacity);

  THOT_API void swAlignModel_addSentencePair(void* swAlignModelHandle, const char* sourceSentence,
                                             const char* targetSentence);

  THOT_API void swAlignModel_train(void* swAlignModelHandle, unsigned int numIters);

  THOT_API void swAlignModel_clearTempVars(void* swAlignModelHandle);

  THOT_API void swAlignModel_save(void* swAlignModelHandle, const char* prefFileName);

  THOT_API float swAlignModel_getTranslationProbability(void* swAlignModelHandle, const char* srcWord,
                                                        const char* trgWord);

  THOT_API float swAlignModel_getTranslationProbabilityByIndex(void* swAlignModelHandle, unsigned int srcWordIndex,
                                                               unsigned int trgWordIndex);

  THOT_API float swAlignModel_getIbm2AlignmentProbability(void* swAlignModelHandle, unsigned int j, unsigned int sLen,
                                                          unsigned int tLen, unsigned int i);

  THOT_API float swAlignModel_getHmmAlignmentProbability(void* swAlignModelHandle, unsigned int prevI,
                                                         unsigned int sLen, unsigned int i);

  THOT_API float swAlignModel_getBestAlignment(void* swAlignModelHandle, const char* sourceSentence,
                                               const char* targetSentence, bool** matrix, unsigned int* iLen,
                                               unsigned int* jLen);

  THOT_API void* swAlignModel_getTranslations(void* swAlignModelHandle, const char* srcWord, float threshold);

  THOT_API void* swAlignModel_getTranslationsByIndex(void* swAlignModelHandle, unsigned int srcWordIndex,
                                                     float threshold);

  THOT_API void swAlignModel_close(void* swAlignModelHandle);

  THOT_API unsigned int swAlignTrans_getCount(void* swAlignTransHandle);

  THOT_API unsigned int swAlignTrans_getTranslations(void* swAlignTransHandle, unsigned int* wordIndices, float* probs,
                                                     unsigned int capacity);

  THOT_API void swAlignTrans_destroy(void* swAlignTransHandle);

  THOT_API bool giza_symmetr1(const char* lhsFileName, const char* rhsFileName, const char* outputFileName,
                              bool transpose);

  THOT_API bool phraseModel_generate(const char* alignmentFileName, int maxPhraseLength, const char* tableFileName,
                                     int n);

  THOT_API void* langModel_open(const char* prefFileName);

  THOT_API float langModel_getSentenceProbability(void* lmHandle, const char* sentence);

  THOT_API void langModel_close(void* lmHandle);

  THOT_API void* llWeightUpdater_create();

  THOT_API void llWeightUpdater_updateClosedCorpus(void* llWeightUpdaterHandle, const char** references,
                                                   const char*** nblists, const double*** scoreComps,
                                                   const unsigned int* nblistLens, float* weights,
                                                   unsigned int numSents, unsigned int numWeights);

  THOT_API void llWeightUpdater_close(void* llWeightUpdaterHandle);

#ifdef __cplusplus
}
#endif

#endif