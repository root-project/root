// @(#)root/io:$Id$
// Author: Philippe Canal, May 2011

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMemFile
#define ROOT_TMemFile

#include "TFile.h"
#include <vector>
#include <memory>

class TMemFile : public TFile {
public:
   using ExternalDataPtr_t = std::shared_ptr<const std::vector<char>>;
   /// A read-only memory range which we do not control.
   struct ZeroCopyView_t {
      const char *fStart;
      const size_t fSize;
      explicit ZeroCopyView_t(const char * start, const size_t size) : fStart(start), fSize(size) {}
   };

protected:
   struct TMemBlock {
   private:
      TMemBlock(const TMemBlock&) = delete;            // Not implemented
      TMemBlock &operator=(const TMemBlock&) = delete; // Not implemented.
   public:
      TMemBlock() = default;
      TMemBlock(Long64_t size, TMemBlock *previous = nullptr);
      TMemBlock(UChar_t* externalBuffer, Long64_t size);
      ~TMemBlock();

      void CreateNext(Long64_t size);

      TMemBlock *fPrevious{nullptr};
      TMemBlock *fNext{nullptr};
      UChar_t   *fBuffer{nullptr};
      Long64_t   fSize{0};
   };
   TMemBlock    fBlockList;               ///< Collection of memory blocks of size fgDefaultBlockSize
   ExternalDataPtr_t fExternalData;       ///< shared file data / content
   Bool_t       fIsOwnedByROOT{kFALSE};   ///< if this is a C-style memory region
   Long64_t     fSize{0};                 ///< Total file size (sum of the size of the chunks)
   Long64_t     fSysOffset{0};            ///< Seek offset in file
   TMemBlock   *fBlockSeek{nullptr};      ///< Pointer to the block we seeked to.
   Long64_t     fBlockOffset{0};          ///< Seek offset within the block

   constexpr static Long64_t fgDefaultBlockSize = 2 * 1024 * 1024;
   Long64_t fDefaultBlockSize = fgDefaultBlockSize;

   Bool_t IsExternalData() const { return !fIsOwnedByROOT; }

   Long64_t MemRead(Int_t fd, void *buf, Long64_t len) const;

   // Overload TFile interfaces.
   Int_t    SysOpen(const char *pathname, Int_t flags, UInt_t mode) override;
   Int_t    SysClose(Int_t fd) override;
   Int_t    SysReadImpl(Int_t fd, void *buf, Long64_t len);
   Int_t    SysWriteImpl(Int_t fd, const void *buf, Long64_t len);
   Int_t    SysRead(Int_t fd, void *buf, Int_t len) override;
   Int_t    SysWrite(Int_t fd, const void *buf, Int_t len) override;
   Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence) override;
   Int_t    SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime) override;
   Int_t    SysSync(Int_t fd) override;

   void ResetObjects(TDirectoryFile *, TFileMergeInfo *) const;

   enum class EMode {
      kCreate,
      kRecreate,
      kUpdate,
      kRead
   };

   bool NeedsToWrite(EMode mode) const { return mode != EMode::kRead; }
   bool NeedsExistingFile(EMode mode) const { return mode == EMode::kUpdate || mode == EMode::kRead; }

   EMode ParseOption(Option_t *option);

   TMemFile &operator=(const TMemFile&) = delete; // Not implemented.

public:
   TMemFile(const char *name, Option_t *option = "", const char *ftitle = "",
            Int_t compress = ROOT::RCompressionSetting::EDefaults::kUseGeneralPurpose, Long64_t defBlockSize = 0LL);
   TMemFile(const char *name, char *buffer, Long64_t size, Option_t *option = "", const char *ftitle = "",
            Int_t compress = ROOT::RCompressionSetting::EDefaults::kUseGeneralPurpose, Long64_t defBlockSize = 0LL);
   TMemFile(const char *name, ExternalDataPtr_t data);
   TMemFile(const char *name, const ZeroCopyView_t &datarange);
   TMemFile(const char *name, std::unique_ptr<TBufferFile> buffer);
   TMemFile(const TMemFile &orig);
   virtual ~TMemFile();

   virtual Long64_t CopyTo(void *to, Long64_t maxsize) const;
   virtual void     CopyTo(TBuffer &tobuf) const;
           Long64_t GetSize() const override;

           void ResetAfterMerge(TFileMergeInfo *) override;
           void ResetErrno() const override;

           void        Print(Option_t *option="") const override;

   ClassDefOverride(TMemFile, 0) // A ROOT file that reads/writes on a chunk of memory
};

#endif
