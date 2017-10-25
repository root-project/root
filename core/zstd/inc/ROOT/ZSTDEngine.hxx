/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/Compression.hxx"

#include <zstd.h>

#include <memory>

namespace ROOT {
namespace Internal {

class ZSTDCompressionEngine final: public CompressionEngine {

public:
   ZSTDCompressionEngine(int level) : CompressionEngine(level), fCtx(ZSTD_createCCtx(), &ZSTD_freeCCtx) {}

   virtual ~ZSTDCompressionEngine() {}

   virtual char Version() const override {return '\1';}

   virtual ssize_t StreamFull(const void *buffer, size_t size) override;

   virtual void Reset() override;

   virtual bool Train(const void *buffer, size_t size) override;

   virtual bool GetTraining(void *&buffer, size_t &size) override;

   virtual bool SetTraining(const void *buffer, size_t size) override;

private:
   using Ctx_ptr = std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)>;
   using Dict_ptr = std::unique_ptr<ZSTD_CDict, decltype(&ZSTD_freeCDict)>;
   using Train_ptr = std::unique_ptr<void, decltype(&free)>;
   
   Ctx_ptr   fCtx{nullptr, &ZSTD_freeCCtx};    // Compression context.
   Dict_ptr  fDict{nullptr, &ZSTD_freeCDict};  // Compression training dictionary.
   Train_ptr fTrainingBuffer{nullptr, &free};  // Dictionary contents
   size_t    fTrainingSize{0};                 // Dictionary size
};

class ZSTDDecompressionEngine final : public DecompressionEngine {

public:
   ZSTDDecompressionEngine() : fCtx(ZSTD_createDCtx(), &ZSTD_freeDCtx) {}

   virtual ~ZSTDDecompressionEngine() {}

   virtual bool VersionCompat(char version) const override {return version == '\1';}

   virtual ssize_t StreamFull(const void *buffer, size_t size) override;

   virtual void Reset() override;

   virtual bool SetTraining(const void *buffer, size_t size) override;

private:
   using Ctx_ptr = std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)>;
   using Dict_ptr = std::unique_ptr<ZSTD_DDict, decltype(&ZSTD_freeDDict)>;

   Ctx_ptr  fCtx{nullptr, &ZSTD_freeDCtx};    // Compression context.
   Dict_ptr fDict{nullptr, &ZSTD_freeDDict};  // Compression training dictionary.
};

}  // Internal
}  // ROOT
