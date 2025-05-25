// Author: Jonas Hahnfeld

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RRawFileTFile
#define ROOT_RRawFileTFile

#include <ROOT/RRawFile.hxx>

#include <TFile.h>

#include <stdexcept>

namespace ROOT {
namespace Internal {

/**
 * \class RRawFileTFile RRawFileTFile.hxx
 * \ingroup IO
 *
 * The RRawFileTFile wraps an open TFile, but does not take ownership.
 */
class RRawFileTFile : public RRawFile {
private:
   TFile *fFile;

protected:
   void OpenImpl() final { fOptions.fBlockSize = 0; }

   size_t ReadAtImpl(void *buffer, size_t nbytes, std::uint64_t offset) final
   {
      if (fFile->ReadBuffer(static_cast<char *>(buffer), offset, nbytes)) {
         throw std::runtime_error("failed to read expected number of bytes");
      }
      return nbytes;
   }

   std::uint64_t GetSizeImpl() final { return fFile->GetSize(); }

public:
   RRawFileTFile(TFile *file) : RRawFile(file->GetEndpointUrl()->GetFile(), {}), fFile(file) {}

   std::unique_ptr<ROOT::Internal::RRawFile> Clone() const final { return std::make_unique<RRawFileTFile>(fFile); }
};

} // namespace Internal
} // namespace ROOT

#endif
