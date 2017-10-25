
#include "zdict.h"

#include "RConfig.h"
#include "ROOT/ZSTDEngine.hxx"
#include "RZip.h"

using namespace ROOT::Internal;


ssize_t ZSTDCompressionEngine::StreamFull(const void *buffer, size_t size) {
   if (R__unlikely(fCap < kROOTHeaderSize)) return -1;
   char *originalLocation = fCur;
   fCur += kROOTHeaderSize;
   fCap -= kROOTHeaderSize;

   size_t retval = fDict.get() ? ZSTD_compress_usingCDict(fCtx.get(), fCur, fCap, buffer, size, fDict.get()) :
                                 ZSTD_compressCCtx(fCtx.get(), fCur, fCap, buffer, size, 2*fLevel);
   if (R__unlikely(ZSTD_isError(retval))) {
      return -1;
   }

   fCur += retval;
   fCap -= retval;

   WriteROOTHeader(originalLocation, "ZS", Version(), retval, size);
   // Note: return size should include header.
   return static_cast<ssize_t>(retval+kROOTHeaderSize);
}

void ZSTDCompressionEngine::Reset() {
   fDict.release();
   fTrainingBuffer.reset();
   fTrainingSize = 0;
}


bool ZSTDCompressionEngine::Train(const void *buffer, size_t size) {
   // From the zstd headers, we have the following recommendations:
   //   - 100KB is a good size.
   //   - Total size of all samples should be about 100x the target dictionary size.
   size_t dictCapacity = std::min(std::max(size/100, static_cast<size_t>(16*1024)), static_cast<size_t>(100*1024));
   void *dictBuffer = malloc(dictCapacity);
   if (!dictBuffer) return false;

   const size_t samplesSizes[1] = {size};
   size_t retval = ZDICT_trainFromBuffer(dictBuffer, dictCapacity, buffer, samplesSizes, 1);

   if (ZDICT_isError(retval)) return false;

   fTrainingBuffer.reset(dictBuffer);
   fTrainingSize = retval;  // Note: returned dictionary size may be less than capacity.

   // Create the "digested" dictionary usable by the compression context.
   fDict.reset(ZSTD_createCDict(fTrainingBuffer.get(), fTrainingSize, 2*fLevel));
   return fDict.get();
}


bool ZSTDCompressionEngine::GetTraining(void *&buffer, size_t &size) {
   buffer = fTrainingBuffer.get();
   size = fTrainingSize;
   return buffer != nullptr;
}

bool ZSTDCompressionEngine::SetTraining(const void *, size_t) {
   return false;
}

ssize_t ZSTDDecompressionEngine::StreamFull(const void *buffer, size_t size) {
   // Find the start of the ZSTD block by skipping the ROOT header.
   const char *src = static_cast<const char*>(buffer);
   src += CompressionEngine::kROOTHeaderSize;
   size -= CompressionEngine::kROOTHeaderSize;

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
