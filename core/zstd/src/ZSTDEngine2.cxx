
#include "zdict.h"

#include "RConfig.h"
#include "ROOT/ZSTDEngine.hxx"
#include "RZip.h"

using namespace ROOT::Internal;

#include <iostream>
#include <ctime>
using namespace std;

size_t ZSTDCompressionEngine::with_dict(
                  const void *buffer, size_t size,
                  Ctx_ptr & fCtx1, DCtx_ptr & fDCtx1,
                  Dict_ptr & fDict1, DDict_ptr & fDDict1,
                  float & time, float & timeD) {

   size_t newSize = size*2;
   void * decomp_buffer = (void *)malloc(newSize);

   clock_t begin = clock();
   size_t retval = ZSTD_compress_usingCDict(fCtx1.get(), fCur, fCap, buffer, size, fDict1.get());
   clock_t end = clock();
   time = double(end - begin) / CLOCKS_PER_SEC;
   if (!ZSTD_isError(retval)) {
      begin = clock();
      ZSTD_decompress_usingDDict(fDCtx1.get(), decomp_buffer, newSize, fCur, retval, fDDict1.get());
      end = clock();
      timeD = double(end - begin) / CLOCKS_PER_SEC;
   }
   free(decomp_buffer);

   return retval;
}

size_t ZSTDCompressionEngine::with_simple(
                  const void *buffer, size_t size,
                  Ctx_ptr & fCtx1, DCtx_ptr & fDCtx1,
                  int level, float & time, float & timeD) {

   size_t newSize = size*2;
   void * decomp_buffer = (void *)malloc(newSize);

   clock_t begin = clock();
   size_t retval = ZSTD_compressCCtx(fCtx1.get(), fCur, fCap, buffer, size, 2*level);
   clock_t end = clock();
   time = double(end - begin) / CLOCKS_PER_SEC;
   if (!ZSTD_isError(retval)) {
      begin = clock();
      size_t r = ZSTD_decompressDCtx(fDCtx1.get(), decomp_buffer, newSize, fCur, retval);
      end = clock();
      timeD = double(end - begin) / CLOCKS_PER_SEC;
   }
   free(decomp_buffer);

   return retval;
}

ssize_t ZSTDCompressionEngine::StreamFull(const void *buffer, size_t size) {
   clock_t begin;
   clock_t end;

   /*if (fDict.get()) {
      cout << nBasket << "----------StreamFull with Dict------------" << endl;
   }
   else {
      cout << nBasket << "----------StreamFull------------" << endl;
   }*/
   float dictGenerationTime = 0;
   float dictGenerationTimeLow = 0;
   float dictGenerationTimeHigh = 0;
   float dictGenerationTime2 = 0;
   
   if (!fDict.get()) {
      begin = clock();
      fDict.reset(ZSTD_createCDict(buffer, size, 2*5));
      fDDict.reset(ZSTD_createDDict(buffer, size));
      end = clock();
      dictGenerationTime = (end - begin) / CLOCKS_PER_SEC;

      begin = clock();
      fDictLow.reset(ZSTD_createCDict(buffer, size, 2*3));
      fDDictLow.reset(ZSTD_createDDict(buffer, size));
      end = clock();
      dictGenerationTimeLow = (end - begin) / CLOCKS_PER_SEC;

      begin = clock();
      fDictHigh.reset(ZSTD_createCDict(buffer, size, 2*8));
      fDDictHigh.reset(ZSTD_createDDict(buffer, size));
      end = clock();
      dictGenerationTimeHigh = (end - begin) / CLOCKS_PER_SEC;

      begin = clock();
      fDict2.reset(ZSTD_createCDict(buffer, size/2, 2*5));
      fDDict2.reset(ZSTD_createDDict(buffer, size/2));
      end = clock();
      dictGenerationTime2 = (end - begin) / CLOCKS_PER_SEC;

      fCtx.reset(ZSTD_createCCtx());
      fCtxLow.reset(ZSTD_createCCtx());
      fCtxHigh.reset(ZSTD_createCCtx());
      fCtxDict.reset(ZSTD_createCCtx());
      fCtxDictLow.reset(ZSTD_createCCtx());
      fCtxDictHigh.reset(ZSTD_createCCtx());
      fCtxDict2.reset(ZSTD_createCCtx());

      fDCtx.reset(ZSTD_createDCtx());
      fDCtxLow.reset(ZSTD_createDCtx());
      fDCtxHigh.reset(ZSTD_createDCtx());
      fDCtxDict.reset(ZSTD_createDCtx());
      fDCtxDictLow.reset(ZSTD_createDCtx());
      fDCtxDictHigh.reset(ZSTD_createDCtx());
      fDCtxDict2.reset(ZSTD_createDCtx());
   }

   /////////////////////////////////COMPRESSION///////////////////
   float simpleTime, simpleTimeD;
   size_t retval = with_simple(buffer, size, fCtx, fDCtx, 5, simpleTime, simpleTimeD);

   float simpleTimeLow, simpleTimeLowD;
   size_t retvalLow = with_simple(buffer, size, fCtxLow, fDCtxLow, 3, simpleTimeLow, simpleTimeLowD);

   float simpleTimeHigh, simpleTimeHighD;
   size_t retvalHigh = with_simple(buffer, size, fCtxHigh, fDCtxHigh, 8, simpleTimeHigh, simpleTimeHighD);


   float dictTime, dictTimeD;
   size_t retvalDict = with_dict(buffer, size, fCtxDict, fDCtxDict,
      fDict, fDDict, dictTime, dictTimeD);

   float dictTime2, dictTime2D;
   size_t retvalDict2 = with_dict(buffer, size, fCtxDict2, fDCtxDict2,
      fDict2, fDDict2, dictTime2, dictTime2D);

   float dictTimeLow, dictTimeLowD;
   size_t retvalDictLow = with_dict(buffer, size, fCtxDictLow, fDCtxDictLow,
      fDictLow, fDDictLow, dictTimeLow, dictTimeLowD);

   float dictTimeHigh, dictTimeHighD;
   size_t retvalDictHigh = with_dict(buffer, size, fCtxDictHigh, fDCtxDictHigh,
      fDictHigh, fDDictHigh, dictTimeHigh, dictTimeHighD);

   /////////////////////////////////END COMPRESSION///////////////////

   //cout << "---->Dict size: "<< ZSTD_sizeof_CDict(fDict.get()) << endl;

   if (!ZSTD_isError(retval)
      && !ZSTD_isError(retvalLow)
      && !ZSTD_isError(retvalHigh)
      && !ZSTD_isError(retvalDict)
      && !ZSTD_isError(retvalDict2)
      && !ZSTD_isError(retvalDictLow)
      && !ZSTD_isError(retvalDictHigh)
      ) {

      if (nBasket == 0) { //All first baskets are compressed the same way
         retvalDict = retval;
         retvalDictLow = retval;
         retvalDictHigh = retval;
         retvalDict2 = retval;

         dictTime = simpleTime;
         dictTimeLow = simpleTime;
         dictTimeHigh = simpleTime;
         dictTime2 = simpleTime;

         dictTimeD = simpleTimeD;
         dictTimeLowD = simpleTimeD;
         dictTimeHighD = simpleTimeD;
         dictTime2D = simpleTimeD;

         /////////
         dictTime += dictGenerationTime;
         dictTimeLow += dictGenerationTimeLow;
         dictTimeHigh += dictGenerationTimeHigh;
         dictTime2 += dictGenerationTime2;
      }

      cout.precision(11);
      cout << nBranch << "," << nBasket 
      << "," << retval + kROOTHeaderSize << "," << simpleTime << "," << simpleTimeD
      << "," << retvalLow + kROOTHeaderSize << "," << simpleTimeLow << "," << simpleTimeLowD
      << "," << retvalHigh + kROOTHeaderSize << "," << simpleTimeHigh << "," << simpleTimeHighD
      << "," << retvalDict + kROOTHeaderSize << "," << dictTime << "," << dictTimeD
      << "," << retvalDictLow + kROOTHeaderSize << "," << dictTimeLow << "," << dictTimeLowD
      << "," << retvalDictHigh + kROOTHeaderSize << "," << dictTimeHigh << "," << dictTimeHighD
      << "," << retvalDict2 + kROOTHeaderSize << "," << dictTime2 << "," << dictTime2D
      << "," << size << endl;
   }


   if (R__unlikely(fCap < kROOTHeaderSize)) return -1;
   char *originalLocation = fCur;
   fCur += kROOTHeaderSize;
   fCap -= kROOTHeaderSize;

   /*size_t retval = fDict.get() ? ZSTD_compress_usingCDict(fCtx.get(), fCur, fCap, buffer, size, fDict.get()) :
                                 ZSTD_compressCCtx(fCtx.get(), fCur, fCap, buffer, size, 2*fLevel);*/
   if (R__unlikely(ZSTD_isError(retval))) {
      //cout << "Error compressing : "<< ZSTD_getErrorName(retval) << endl;
      return -1;
   }

   fCur += retval;
   fCap -= retval;

   nBasket++;
   WriteROOTHeader(originalLocation, "ZS", Version(), retval, size);
   // Note: return size should include header.
   return static_cast<ssize_t>(retval+kROOTHeaderSize);
}

void ZSTDCompressionEngine::Reset() {
   //cout << nBasket << "----------Reset------------" << endl; 

   fDict.release();
   fTrainingBuffer.reset();
   fTrainingSize = 0;
}


bool ZSTDCompressionEngine::Train(const void *buffer, size_t size) {
   // From the zstd headers, we have the following recommendations:
   //   - 100KB is a good size.
   //   - Total size of all samples should be about 100x the target dictionary size.
   //cout << nBasket << "----------Train------------" << size << endl; 
   size_t dictCapacity = std::min(std::max(size/100, static_cast<size_t>(16*1024)), static_cast<size_t>(100*1024));
   dictCapacity = static_cast<size_t>(16*1024);

   /*void *dictBuffer = malloc(dictCapacity);
   if (!dictBuffer) return false;


   const size_t samplesSizes[1] = {size};

   size_t retval = ZDICT_trainFromBuffer(dictBuffer, dictCapacity, buffer, samplesSizes, 1);

   if (ZDICT_isError(retval)) {
      cout << "Error training : "<< ZDICT_getErrorName(retval)<< endl;
      return false;
   } 

   fTrainingBuffer.reset(dictBuffer);
   fTrainingSize = retval;  // Note: returned dictionary size may be less than capacity.

   // Create the "digested" dictionary usable by the compression context.
   fDict.reset(ZSTD_createCDict(fTrainingBuffer.get(), fTrainingSize, 2*fLevel));
   return fDict.get();*/
   return false;
}


bool ZSTDCompressionEngine::GetTraining(void *&buffer, size_t &size) {
   //cout << nBasket << "----------GetTraining------------" << endl; 

   /*buffer = fTrainingBuffer.get();
   size = fTrainingSize;
   return buffer != nullptr;*/
   return true;
}

bool ZSTDCompressionEngine::SetTraining(const void *, size_t) {
   return false;
}

ssize_t ZSTDDecompressionEngine::StreamFull(const void *buffer, size_t size) {
   // Find the start of the ZSTD block by skipping the ROOT header.
   const char *src = static_cast<const char*>(buffer);
   src += CompressionEngine::kROOTHeaderSize;
   size -= CompressionEngine::kROOTHeaderSize;

   if (!fDict.get())
      fDict.reset(ZSTD_createDDict(buffer, size));

   size_t retval = fDict ? ZSTD_decompress_usingDDict(fCtx.get(), fCur, fCap, src, size, fDict.get()) :
                           ZSTD_decompressDCtx(fCtx.get(), fCur, fCap, src, size);
   if (R__unlikely(ZSTD_isError(retval))) {
      return -1;
   }
   fCur += retval;
   fCap -= retval;
   return static_cast<ssize_t>(retval);
}


void ZSTDDecompressionEngine::Reset() {
   fDict.release();
}


bool ZSTDDecompressionEngine::SetTraining(const void *buffer, size_t size) {
   fDict.reset(ZSTD_createDDict(buffer, size));
   return fDict.get() != nullptr;
}
