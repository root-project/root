
#include "zdict.h"

#include "RConfig.h"
#include "ROOT/ZSTDEngine.hxx"
#include "RZip.h"
#include <iostream>

using namespace ROOT::Internal;

ssize_t ZSTDCompressionEngine::StreamFull(const void *buffer, size_t size) {
   if (R__unlikely(fCap < kROOTHeaderSize))
      return -1;

   char *originalLocation = fCur;
   fCur += kROOTHeaderSize;
   fCap -= kROOTHeaderSize;

   size_t retval = fDict.get() ? ZSTD_compress_usingCDict(fCtx.get(), fCur, fCap, buffer, size, fDict.get()) :
                                 ZSTD_compressCCtx(fCtx.get(), fCur, fCap, buffer, size, 2*fLevel);
   
   if (R__unlikely(ZSTD_isError(retval))) {
      std::cout << "Error compressing : "<< ZSTD_getErrorName(retval) << std::endl;
      return -1;
   }

   if (!fDict.get()) {
      fDict.reset(ZSTD_createCDict(buffer, size, 2*fLevel));
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
   return false;
}

bool ZSTDCompressionEngine::GetTraining(void *&buffer, size_t &size) {
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

   size_t retval = fDict.get() ? ZSTD_decompress_usingDDict(fCtx.get(), fCur, fCap, src, size, fDict.get()) :
                           ZSTD_decompressDCtx(fCtx.get(), fCur, fCap, src, size);

   if (R__unlikely(ZSTD_isError(retval))) {
      std::cout << "Error decompressing : "<< ZSTD_getErrorName(retval) << std::endl;
      return -1;
   }
   if (!fDict.get()) {
      fDict.reset(ZSTD_createDDict(fCur, retval));
   }
   fCur += retval;
   fCap -= retval;
   return static_cast<ssize_t>(retval);
}


void ZSTDDecompressionEngine::Reset() {
   fDict.release();
}


bool ZSTDDecompressionEngine::SetTraining(const void *buffer, size_t size) {
   return false;
}
