// @(#)root/io:$Id$
// Author: Rene Brun   19/05/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFileCacheWrite
#define ROOT_TFileCacheWrite

#include "TObject.h"

class TFile;

class TFileCacheWrite : public TObject {

protected:
   Long64_t      fSeekStart;      ///< Seek value of first block in cache
   Int_t         fBufferSize;     ///< Allocated size of fBuffer
   Int_t         fNtot;           ///< Total size of cached blocks
   TFile        *fFile;           ///< Pointer to file
   char         *fBuffer;         ///< [fBufferSize] buffer of contiguous prefetched blocks
   Bool_t        fRecursive;      ///< flag to avoid recursive calls

private:
   TFileCacheWrite(const TFileCacheWrite &);            //cannot be copied
   TFileCacheWrite& operator=(const TFileCacheWrite &);

public:
   TFileCacheWrite();
   TFileCacheWrite(TFile *file, Int_t buffersize);
   virtual ~TFileCacheWrite();
   virtual Bool_t      Flush();
   virtual Int_t       GetBytesInCache() const { return fNtot; }
   virtual void        Print(Option_t *option="") const;
   virtual Int_t       ReadBuffer(char *buf, Long64_t pos, Int_t len);
   virtual Int_t       WriteBuffer(const char *buf, Long64_t pos, Int_t len);
   virtual void        SetFile(TFile *file);

   ClassDef(TFileCacheWrite,1)  //TFile cache when writing
};

#endif
