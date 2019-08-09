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
   ZSTDCompressionEngine(int level) : CompressionEngine(level), fCtx(ZSTD_createCCtx(), &ZSTD_freeCCtx) {
      static int countedBranches = 0;
      nBranch = countedBranches;
      countedBranches++;
   }

   virtual ~ZSTDCompressionEngine() {}

   virtual char Version() const override {return '\1';}

   virtual ssize_t StreamFull(const void *buffer, size_t size) override;

   virtual void Reset() override;

   virtual bool Train(const void *buffer, size_t size) override;

   virtual bool GetTraining(void *&buffer, size_t &size) override;

   virtual bool SetTraining(const void *buffer, size_t size) override;

   using Ctx_ptr = std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)>;
   using Dict_ptr = std::unique_ptr<ZSTD_CDict, decltype(&ZSTD_freeCDict)>;

   using DCtx_ptr = std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)>;
   using DDict_ptr = std::unique_ptr<ZSTD_DDict, decltype(&ZSTD_freeDDict)>;
   using Train_ptr = std::unique_ptr<void, decltype(&free)>;

private:

   
   Ctx_ptr   fCtx{nullptr, &ZSTD_freeCCtx};    // Compression context.
   Ctx_ptr   fCtxLow{nullptr, &ZSTD_freeCCtx};
   Ctx_ptr   fCtxHigh{nullptr, &ZSTD_freeCCtx};
   Ctx_ptr   fCtxDict{nullptr, &ZSTD_freeCCtx};    // Compression context.
   Ctx_ptr   fCtxDict2{nullptr, &ZSTD_freeCCtx};    // Compression context.
   Ctx_ptr   fCtxDictLow{nullptr, &ZSTD_freeCCtx};    // Compression context.
   Ctx_ptr   fCtxDictHigh{nullptr, &ZSTD_freeCCtx};    // Compression context.
   
   DCtx_ptr   fDCtx{nullptr, &ZSTD_freeDCtx};    // Compression context.
   DCtx_ptr   fDCtxLow{nullptr, &ZSTD_freeDCtx};    // Compression context.
   DCtx_ptr   fDCtxHigh{nullptr, &ZSTD_freeDCtx};    // Compression context.
   DCtx_ptr   fDCtxDict{nullptr, &ZSTD_freeDCtx};    // Compression context.
   DCtx_ptr   fDCtxDict2{nullptr, &ZSTD_freeDCtx};    // Compression context.
   DCtx_ptr   fDCtxDictLow{nullptr, &ZSTD_freeDCtx};    // Compression context.
   DCtx_ptr   fDCtxDictHigh{nullptr, &ZSTD_freeDCtx};    // Compression context.


   Dict_ptr  fDict{nullptr, &ZSTD_freeCDict};  // Compression training dictionary.
   Dict_ptr  fDict2{nullptr, &ZSTD_freeCDict};  // Compression training dictionary.
   Dict_ptr  fDictLow{nullptr, &ZSTD_freeCDict};  // Compression training dictionary.
   Dict_ptr  fDictHigh{nullptr, &ZSTD_freeCDict};  // Compression training dictionary.
   
   DDict_ptr  fDDict{nullptr, &ZSTD_freeDDict};  // Compression training dictionary.
   DDict_ptr  fDDict2{nullptr, &ZSTD_freeDDict};  // Compression training dictionary.
   DDict_ptr  fDDictLow{nullptr, &ZSTD_freeDDict};  // Compression training dictionary.
   DDict_ptr  fDDictHigh{nullptr, &ZSTD_freeDDict};  // Compression training dictionary.

   Train_ptr fTrainingBuffer{nullptr, &free};  // Dictionary contents
   size_t    fTrainingSize{0};                 // Dictionary size*/
   int nBasket = 0;
   int nBranch = 0;

   size_t with_dict(
                  const void *buffer, size_t size,
                  Ctx_ptr & fCtx1, DCtx_ptr & fDCtx1,
                  Dict_ptr & fDict1, DDict_ptr & fDDict1,
                  float & time, float & timeD);

   size_t with_simple(
                  const void *buffer, size_t size,
                  Ctx_ptr & fCtx1, DCtx_ptr & fDCtx1,
                  int level, float & time, float & timeD);


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
   int n = 0;
};

}  // Internal
}  // ROOT
