// @(#)root/io:$Id$
// Author: Philippe Canal, Witold Pokorski, and Guilherme Amadio

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TBufferMerger.hxx"

#include "TArrayC.h"
#include "TBufferFile.h"

namespace ROOT {
namespace Experimental {

TBufferMergerFile::TBufferMergerFile(TBufferMerger &m)
   : TMemFile(m.fFile->GetName(), "recreate", "", m.fFile->GetCompressionSettings()), fMerger(m)
{
}

TBufferMergerFile::~TBufferMergerFile()
{
}

Int_t TBufferMergerFile::Write(const char *name, Int_t opt, Int_t bufsize)
{
   Int_t nbytes = TMemFile::Write(name, opt, bufsize);

   if (nbytes) {
      TBufferFile *fBuffer = new TBufferFile(TBuffer::kWrite);

      fBuffer->WriteLong64(GetEND());
      CopyTo(*fBuffer);

      fMerger.Push(fBuffer);
      ResetAfterMerge(0);
   }
   return nbytes;
}

} // namespace Experimental
} // namespace ROOT
