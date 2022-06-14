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

#include "TBufferFile.h"

namespace ROOT {

TBufferMergerFile::TBufferMergerFile(TBufferMerger &m)
   : TMemFile(m.fMerger.GetOutputFile()->GetName(), "RECREATE", "",
              m.fMerger.GetOutputFile()->GetCompressionSettings()),
     fMerger(m)
{
}

TBufferMergerFile::~TBufferMergerFile()
{
}

Int_t TBufferMergerFile::Write(const char *name, Int_t opt, Int_t bufsize)
{
   // Make sure the compression of the basket is done in the unlocked thread and
   // not in the locked section.
   if (!fMerger.GetNotrees())
      TMemFile::Write(name, opt | TObject::kOnlyPrepStep, bufsize);

   // Instead of Writing the TTree, doing a memcpy, Pushing to the queue
   // then Reading and then deleting, let's see if we can just merge using
   // the live TTree.
   if (fMerger.TryMerge(this)) {
      ResetAfterMerge(0);
      return 0;
   }

   auto oldCompLevel = GetCompressionLevel();
   if (!fMerger.GetCompressTemporaryKeys())
      SetCompressionLevel(0);
   Int_t nbytes = TMemFile::Write(name, opt, bufsize);
   SetCompressionLevel(oldCompLevel);

   if (nbytes) {
      TBufferFile *buffer = new TBufferFile(TBuffer::kWrite, GetSize());
      CopyTo(*buffer);
      buffer->SetReadMode();
      fMerger.Push(buffer);
      ResetAfterMerge(0);
   }
   return nbytes;
}

} // namespace ROOT
