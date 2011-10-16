// @(#)root/tree:$Id$
// Author: Rene Brun   19/01/96
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBasket.h"
#include "TBufferFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "TBufferFile.h"
#include "TMath.h"
#include "TTreeCache.h"
#include "TTreeCacheUnzip.h"
#include "TVirtualPerfStats.h"
#include "TTimeStamp.h"

// TODO: Copied from TBranch.cxx
#if (__GNUC__ >= 3) || defined(__INTEL_COMPILER)
#if !defined(R__unlikely)
  #define R__unlikely(expr) __builtin_expect(!!(expr), 0)
#endif
#if !defined(R__likely)
  #define R__likely(expr) __builtin_expect(!!(expr), 1)
#endif
#else
  #define R__unlikely(expr) expr
  #define R__likely(expr) expr
#endif

extern "C" void R__zipMultipleAlgorithm(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep, int compressionAlgorithm);
extern "C" void R__unzip(Int_t *nin, UChar_t *bufin, Int_t *lout, char *bufout, Int_t *nout);
extern "C" int R__unzip_header(Int_t *nin, UChar_t *bufin, Int_t *lout);

const Int_t  kMAXBUF = 0xFFFFFF;
const UInt_t kDisplacementMask = 0xFF000000;  // In the streamer the two highest bytes of
                                              // the fEntryOffset are used to stored displacement.

ClassImp(TBasket)

//_______________________________________________________________________
//
//  Manages buffers for branches of a Tree.
//  See picture in TTree.
//

//_______________________________________________________________________
TBasket::TBasket() : fCompressedBufferRef(0), fOwnsCompressedBuffer(kFALSE), fLastWriteBufferSize(0)
{
   // Default contructor.

   fDisplacement  = 0;
   fEntryOffset   = 0;
   fBufferRef     = 0;
   fBuffer        = 0;
   fHeaderOnly    = kFALSE;
   fBufferSize    = 0;
   fNevBufSize    = 0;
   fNevBuf        = 0;
   fLast          = 0;
   fBranch        = 0;
}

//_______________________________________________________________________
TBasket::TBasket(TDirectory *motherDir) : TKey(motherDir),fCompressedBufferRef(0), fOwnsCompressedBuffer(kFALSE), fLastWriteBufferSize(0)
{
   // Constructor used during reading.
   fDisplacement  = 0;
   fEntryOffset   = 0;
   fBufferRef     = 0;
   fBuffer        = 0;
   fHeaderOnly    = kFALSE;
   fBufferSize    = 0;
   fNevBufSize    = 0;
   fNevBuf        = 0;
   fLast          = 0;
   fBranch        = 0;
}

//_______________________________________________________________________
TBasket::TBasket(const char *name, const char *title, TBranch *branch) : 
   TKey(branch->GetDirectory()),fCompressedBufferRef(0), fOwnsCompressedBuffer(kFALSE), fLastWriteBufferSize(0)
{
   // Basket normal constructor, used during writing.

   SetName(name);
   SetTitle(title);
   fClassName   = "TBasket";
   fBufferSize  = branch->GetBasketSize();
   fNevBufSize  = branch->GetEntryOffsetLen();
   fNevBuf      = 0;
   fEntryOffset = 0;
   fDisplacement= 0;
   fBuffer      = 0;
   fBufferRef   = new TBufferFile(TBuffer::kWrite, fBufferSize);
   fVersion    += 1000;
   if (branch->GetDirectory()) {
      TFile *file = branch->GetFile();
      fBufferRef->SetParent(file);
   }
   fHeaderOnly  = kTRUE;
   fLast        = 0; // Must initialize before calling Streamer()
   if (branch->GetTree()) {
      fCompressedBufferRef = branch->GetTree()->GetTransientBuffer(fBufferSize);
      fOwnsCompressedBuffer = kFALSE;
      if (!fCompressedBufferRef) {
         fCompressedBufferRef = new TBufferFile(TBuffer::kRead, fBufferSize);
         fOwnsCompressedBuffer = kTRUE;
      }
   }
   Streamer(*fBufferRef);
   fKeylen      = fBufferRef->Length();
   fObjlen      = fBufferSize - fKeylen;
   fLast        = fKeylen;
   fBuffer      = 0;
   fBranch      = branch;
   fHeaderOnly  = kFALSE;
   if (fNevBufSize) {
      fEntryOffset = new Int_t[fNevBufSize];
      for (Int_t i=0;i<fNevBufSize;i++) fEntryOffset[i] = 0;
   }
   branch->GetTree()->IncrementTotalBuffers(fBufferSize);
}

//_______________________________________________________________________
TBasket::~TBasket()
{
   // Basket destructor.

   if (fDisplacement) delete [] fDisplacement;
   if (fEntryOffset)  delete [] fEntryOffset;
   if (fBufferRef) delete fBufferRef;
   fBufferRef = 0;
   fBuffer = 0;
   fDisplacement= 0;
   fEntryOffset = 0;
   // Note we only delete the compressed buffer if we own it
   if (fCompressedBufferRef && fOwnsCompressedBuffer) {
       delete fCompressedBufferRef;
       fCompressedBufferRef = 0;
   }
}

//_______________________________________________________________________
void TBasket::AdjustSize(Int_t newsize)
{
   // Increase the size of the current fBuffer up to newsize.
   
   if (fBuffer == fBufferRef->Buffer()) {
      fBufferRef->Expand(newsize);
      fBuffer = fBufferRef->Buffer();
   } else {
      fBufferRef->Expand(newsize);
   }
   fBranch->GetTree()->IncrementTotalBuffers(newsize-fBufferSize);
   fBufferSize  = newsize;
}

//_______________________________________________________________________
Long64_t TBasket::CopyTo(TFile *to) 
{
   // Copy the basket of this branch onto the file to.

   fBufferRef->SetWriteMode();
   Int_t nout = fNbytes - fKeylen;
   fBuffer = fBufferRef->Buffer();
   Create(nout, to);
   fBufferRef->SetBufferOffset(0);
   fHeaderOnly = kTRUE;
   Streamer(*fBufferRef);
   fHeaderOnly = kFALSE;
   Int_t nBytes = WriteFileKeepBuffer(to);

   return nBytes>0 ? nBytes : -1;
}

//_______________________________________________________________________
void TBasket::DeleteEntryOffset()
{
   //  Delete fEntryOffset array.

   if (fEntryOffset) delete [] fEntryOffset;
   fEntryOffset = 0;
   fNevBufSize  = 0;
}


//_______________________________________________________________________
Int_t TBasket::DropBuffers()
{
   // Drop buffers of this basket if it is not the current basket.
   if (!fBuffer && !fBufferRef) return 0;

   if (fDisplacement) delete [] fDisplacement;
   if (fEntryOffset)  delete [] fEntryOffset;
   if (fBufferRef)    delete fBufferRef;
   if (fCompressedBufferRef && fOwnsCompressedBuffer) delete fCompressedBufferRef;
   fBufferRef   = 0;
   fCompressedBufferRef = 0;
   fBuffer      = 0;
   fDisplacement= 0;
   fEntryOffset = 0;
   fBranch->GetTree()->IncrementTotalBuffers(-fBufferSize);
   return fBufferSize;
}

//_______________________________________________________________________
Int_t TBasket::GetEntryPointer(Int_t entry)
{
   // Get pointer to buffer for internal entry.

   Int_t offset;
   if (fEntryOffset) offset = fEntryOffset[entry];
   else              offset = fKeylen + entry*fNevBufSize;
   fBufferRef->SetBufferOffset(offset);
   return offset;
}

//_______________________________________________________________________
Int_t TBasket::LoadBasketBuffers(Long64_t pos, Int_t len, TFile *file)
{ 
   // Load basket buffers in memory without unziping.
   // This function is called by TTreeCloner.
   // The function returns 0 in case of success, 1 in case of error.

   if (fBufferRef) {
      // Reuse the buffer if it exist.
      fBufferRef->SetReadMode();
      fBufferRef->Reset();
      // We use this buffer both for reading and writing, we need to
      // make sure it is properly sized for writing.
      if (fBufferRef->BufferSize() < len) {
         fBufferRef->SetWriteMode();
         fBufferRef->Expand(len);
         fBufferRef->SetReadMode();
      }
   } else {
      fBufferRef = new TBufferFile(TBuffer::kRead, len);
   }
   fBufferRef->SetParent(file);
   char *buffer = fBufferRef->Buffer();
   file->Seek(pos);
   if (file->ReadBuffer(buffer,len)) {
      return 1; //error while reading
   }

   fBufferRef->SetReadMode();
   fBufferRef->SetBufferOffset(0);
   Streamer(*fBufferRef);

   return 0;
}

//_______________________________________________________________________
void TBasket::MoveEntries(Int_t dentries)
{
   // Remove the first dentries of this basket, moving entries at
   // dentries to the start of the buffer.
   
   Int_t i;

   if (dentries >= fNevBuf) return;
   Int_t bufbegin;
   Int_t moved;

   if (fEntryOffset) {
      bufbegin = fEntryOffset[dentries];
      moved = bufbegin-GetKeylen();

      // First store the original location in the fDisplacement array
      // and record the new start offset

      if (!fDisplacement) {
         fDisplacement = new Int_t[fNevBufSize];
      }
      for (i = 0; i<(fNevBufSize-dentries); ++i) {
         fDisplacement[i] = fEntryOffset[i+dentries];
         fEntryOffset[i]  = fEntryOffset[i+dentries] - moved;
      }
      for (i = fNevBufSize-dentries; i<fNevBufSize; ++i) {
         fDisplacement[i] = 0;      
         fEntryOffset[i]  = 0;
      }

   } else {
      // If there is no EntryOffset array, this means
      // that each entry has the same size and that 
      // it does not point to other objects (hence there
      // is no need for a displacement array).
      bufbegin = GetKeylen() + dentries*fNevBufSize;
      moved = bufbegin-GetKeylen();
   }
   TBuffer *buf = GetBufferRef();
   char *buffer = buf->Buffer();
   memmove(buffer+GetKeylen(),buffer+bufbegin,buf->Length()-bufbegin);
   buf->SetBufferOffset(buf->Length()-moved);
   fNevBuf -= dentries;
}

#define OLD_CASE_EXPRESSION fObjlen==fNbytes-fKeylen && GetBranch()->GetCompressionLevel()!=0 && file->GetVersion()<=30401
//_______________________________________________________________________
Int_t TBasket::ReadBasketBuffersUncompressedCase()
{
   // By-passing buffer unzipping has been requested and is
   // possible (only 1 entry in this basket). 
   fBuffer = fBufferRef->Buffer();
            
   // Make sure that the buffer is set at the END of the data
   fBufferRef->SetBufferOffset(fNbytes);
            
   // Indicate that this buffer is weird.
   fBufferRef->SetBit(TBufferFile::kNotDecompressed);
            
   // Usage of this mode assume the existance of only ONE
   // entry in this basket.
   delete [] fEntryOffset; fEntryOffset = 0;
   delete [] fDisplacement; fDisplacement = 0;
            
   fBranch->GetTree()->IncrementTotalBuffers(fBufferSize);
   return 0;
}

//_______________________________________________________________________
Int_t TBasket::ReadBasketBuffersUnzip(char* buffer, Int_t size, Bool_t mustFree, TFile* file)
{
   // We always create the TBuffer for the basket but it hold the buffer from the cache.
   if (fBufferRef) {
      fBufferRef->SetBuffer(buffer, size, mustFree);
      fBufferRef->SetReadMode();
      fBufferRef->Reset();
   } else {
      fBufferRef = new TBufferFile(TBuffer::kRead, size, buffer, mustFree);
   }
   fBufferRef->SetParent(file);

   Streamer(*fBufferRef);
   
   if (IsZombie()) {
      return -1;
   }
   
   Bool_t oldCase = OLD_CASE_EXPRESSION;

   if ((fObjlen > fNbytes-fKeylen || oldCase) && TestBit(TBufferFile::kNotDecompressed) && (fNevBuf==1)) {
      return TBasket::ReadBasketBuffersUncompressedCase();
   }

   fBuffer = fBufferRef->Buffer();
   return fObjlen+fKeylen;
}

//_______________________________________________________________________
static inline TBuffer* R__InitializeReadBasketBuffer(TBuffer* bufferRef, Int_t len, TFile* file)
{
   // Initialize a buffer for reading if it is not already initialized
   
   TBuffer* result;
   if (R__likely(bufferRef)) {
      bufferRef->SetReadMode();
      Int_t curBufferSize = bufferRef->BufferSize();
      if (curBufferSize < len) {
         // Experience shows that giving 5% "wiggle-room" decreases churn.
         bufferRef->Expand(Int_t(len*1.05));
      }
      bufferRef->Reset();
      result = bufferRef;
   } else {
      result = new TBufferFile(TBuffer::kRead, len);
   }
   result->SetParent(file);
   return result;
}

//_______________________________________________________________________
void inline TBasket::InitializeCompressedBuffer(Int_t len, TFile* file)
{
    // Initialize the compressed buffer; either from the TTree or create a local one.
    Bool_t compressedBufferExists = fCompressedBufferRef != NULL;
    fCompressedBufferRef = R__InitializeReadBasketBuffer(fCompressedBufferRef, len, file);
    if (R__unlikely(!compressedBufferExists)) {
        fOwnsCompressedBuffer = kTRUE;
    }
}

//_______________________________________________________________________
Int_t TBasket::ReadBasketBuffers(Long64_t pos, Int_t len, TFile *file)
{
   // Read basket buffers in memory and cleanup.
   //
   // Read a basket buffer. Check if buffers of previous ReadBasket
   // should not be dropped. Remember, we keep buffers in memory up to
   // fMaxVirtualSize.
   // The function returns 0 in case of success, 1 in case of error
   // This function was modified with the addition of the parallel
   // unzipping, it will try to get the unzipped file from the cache
   // receiving only a pointer to that buffer (so we shall not
   // delete that pointer), although we get a new buffer in case
   // it's not found in the cache.
   // There is a lot of code duplication but it was necesary to assure
   // the expected behavior when there is no cache.

   if(!fBranch->GetDirectory()) {
      return -1;
   }  
   
   Bool_t oldCase;
   char *rawUncompressedBuffer, *rawCompressedBuffer;
   Int_t uncompressedBufferLen;

   // See if the cache has already unzipped the buffer for us.
   TFileCacheRead *pf = file->GetCacheRead();
   if (pf) {
      Int_t res = -1;
      Bool_t free = kTRUE;
      char *buffer;
      res = pf->GetUnzipBuffer(&buffer, pos, len, &free);
      if (R__unlikely(res >= 0)) {
         len = ReadBasketBuffersUnzip(buffer, res, free, file);
         // Note that in the kNotDecompressed case, the above function will return 0;
         // In such a case, we should stop processing
         if (len <= 0) return -len;
         goto AfterBuffer;
      }
   }

   // Determine which buffer to use, so that we can avoid a memcpy in case of 
   // the basket was not compressed.
   TBuffer* readBufferRef;
   if (R__unlikely(fBranch->GetCompressionLevel()==0)) {
      readBufferRef = fBufferRef;
   } else {
      readBufferRef = fCompressedBufferRef;
   }

   // Initialize the buffer to hold the compressed data.
   readBufferRef = R__InitializeReadBasketBuffer(readBufferRef, len, file);
   if (!readBufferRef) {
       Error("ReadBasketBuffers", "Unable to allocate buffer.");
       return 1;
   }

   // Read from the file and unstream the header information.
   if (file->ReadBuffer(readBufferRef->Buffer(),pos,len)) {
      return 1;
   }
   Streamer(*readBufferRef);
   if (IsZombie()) {
      return 1;
   }

   rawCompressedBuffer = readBufferRef->Buffer();

   // Are we done?
   if (R__unlikely(readBufferRef == fBufferRef)) // We expect most basket to be compressed.
   {
      if (R__likely(fObjlen+fKeylen == fNbytes)) {
         // The basket was really not compressed as expected.
         goto AfterBuffer;
      } else {
         // Well, somehow the buffer was compressed anyway, we have the compressed data in the uncompressed buffer
         // Make sure the compressed buffer is initialized, and memcpy.
         InitializeCompressedBuffer(len, file);
         if (!fCompressedBufferRef) {
            Error("ReadBasketBuffers", "Unable to allocate buffer.");
            return 1;
         }
         fBufferRef->Reset();
         rawCompressedBuffer = fCompressedBufferRef->Buffer();
         memcpy(rawCompressedBuffer, fBufferRef->Buffer(), len);
      }
   }

   // Initialize buffer to hold the uncompressed data
   // Note that in previous versions we didn't allocate buffers until we verified
   // the zip headers; this is no longer beforehand as the buffer lifetime is scoped
   // to the TBranch.
   uncompressedBufferLen = len > fObjlen+fKeylen ? len : fObjlen+fKeylen;
   fBufferRef = R__InitializeReadBasketBuffer(fBufferRef, uncompressedBufferLen, file);
   rawUncompressedBuffer = fBufferRef->Buffer();
   fBuffer = rawUncompressedBuffer;

   oldCase = OLD_CASE_EXPRESSION;
   // Case where ROOT thinks the buffer is compressed.  Copy over the key and uncompress the object
   if (fObjlen > fNbytes-fKeylen || oldCase) {
      if (R__unlikely(TestBit(TBufferFile::kNotDecompressed) && (fNevBuf==1))) {
         return ReadBasketBuffersUncompressedCase();
      }

      // Optional monitor for zip time profiling.
      Double_t start = 0;
      if (R__unlikely(gPerfStats)) {
         start = TTimeStamp();
      }

      memcpy(rawUncompressedBuffer, rawCompressedBuffer, fKeylen);
      char *rawUncompressedObjectBuffer = rawUncompressedBuffer+fKeylen;
      UChar_t *rawCompressedObjectBuffer = (UChar_t*)rawCompressedBuffer+fKeylen;
      Int_t nin, nbuf;
      Int_t nout = 0, noutot = 0, nintot = 0;

      // Unzip all the compressed objects in the compressed object buffer.
      while (1) {
         // Check the header for errors.
         if (R__unlikely(R__unzip_header(&nin, rawCompressedObjectBuffer, &nbuf) != 0)) {
            Error("ReadBasketBuffers", "Inconsistency found in header (nin=%d, nbuf=%d)", nin, nbuf);
            break;
         }
         if (R__unlikely(oldCase && (nin > fObjlen || nbuf > fObjlen))) {
            //buffer was very likely not compressed in an old version
            memcpy(rawUncompressedBuffer+fKeylen, rawCompressedObjectBuffer+fKeylen, fObjlen);
            goto AfterBuffer;
         }

         R__unzip(&nin, rawCompressedObjectBuffer, &nbuf, rawUncompressedObjectBuffer, &nout);
         if (!nout) break;
         noutot += nout;
         nintot += nin;
         if (noutot >= fObjlen) break;
         rawCompressedObjectBuffer += nin;
         rawUncompressedObjectBuffer += nout;
      }

      // Make sure the uncompressed numbers are consistent with header.
      if (R__unlikely(noutot != fObjlen)) {
         Error("ReadBasketBuffers", "fNbytes = %d, fKeylen = %d, fObjlen = %d, noutot = %d, nout=%d, nin=%d, nbuf=%d", fNbytes,fKeylen,fObjlen, noutot,nout,nin,nbuf);
         fBranch->GetTree()->IncrementTotalBuffers(fBufferSize);
         return 1;
      }
      len = fObjlen+fKeylen;
      if (R__unlikely(gPerfStats)) {
         gPerfStats->FileUnzipEvent(file,pos,start,nintot,fObjlen);
      }
   } else {
      // Nothing is compressed - copy over wholesale.
      memcpy(rawUncompressedBuffer, rawCompressedBuffer, len);
   }

AfterBuffer:

   fBranch->GetTree()->IncrementTotalBuffers(fBufferSize);
   
   // Read offsets table if needed.
   if (!fBranch->GetEntryOffsetLen()) {
      return 0;
   }
   delete [] fEntryOffset;
   fEntryOffset = 0;
   fBufferRef->SetBufferOffset(fLast);
   fBufferRef->ReadArray(fEntryOffset);
   if (!fEntryOffset) {
      fEntryOffset = new Int_t[fNevBuf+1];
      fEntryOffset[0] = fKeylen;
      Warning("ReadBasketBuffers","basket:%s has fNevBuf=%d but fEntryOffset=0, pos=%lld, len=%d, fNbytes=%d, fObjlen=%d, trying to repair",GetName(),fNevBuf,pos,len,fNbytes,fObjlen);
      return 0;
   }
   // Read the array of diplacement if any.
   delete [] fDisplacement;
   fDisplacement = 0; 
   if (fBufferRef->Length() != len) {
      // There is more data in the buffer!  It is the displacement
      // array.  If len is less than TBuffer::kMinimalSize the actual
      // size of the buffer is too large, so we can not use the
      // fBufferRef->BufferSize()
      fBufferRef->ReadArray(fDisplacement);
   }        

   return 0;
}

//_______________________________________________________________________
Int_t TBasket::ReadBasketBytes(Long64_t pos, TFile *file)
{
   // Read basket buffers in memory and cleanup
   //
   // Read first bytes of a logical record starting at position pos
   // return record length (first 4 bytes of record).

   const Int_t len = 128;
   char buffer[len];
   Int_t keylen;
   file->GetRecordHeader(buffer, pos,len, fNbytes, fObjlen, keylen);
   fKeylen = keylen;
   return fNbytes;
}

//_______________________________________________________________________
void TBasket::Reset()
{
   // Reset the basket to the starting state. i.e. as it was after calling
   // the constructor (and potentially attaching a TBuffer.)
   // Reduce memory used by fEntryOffset and the TBuffer if needed ..
   
   // Name, Title, fClassName, fBranch 
   // stay the same.
   
   // Downsize the buffer if needed.
   Int_t curSize = fBufferRef->BufferSize();
   // fBufferLen at this point is already reset, so use indirect measurements
   Int_t curLen = (GetObjlen() + GetKeylen());
   Long_t newSize = -1;
   if (curSize > 2*curLen)
   {
      Long_t curBsize = fBranch->GetBasketSize();      
      if (curSize > 2*curBsize ) {
         Long_t avgSize = (Long_t)(fBranch->GetTotBytes() / (1+fBranch->GetWriteBasket())); // Average number of bytes per basket so far
         if (curSize > 2*avgSize) {
            newSize = curBsize;
            if (curLen > newSize) {
               newSize = curLen;
            }
            if (avgSize > newSize) {
               newSize = avgSize;
            }
            newSize = newSize + 512 - newSize%512;  // Wiggle room and alignment (512 is same as in OptimizeBaskets)
         }
      }
   }
   /*
      Philippe has asked us to keep this turned off until we finish memory fragmentation studies.
   // If fBufferRef grew since we last saw it, shrink it to 105% of the occupied size
   if (curSize > fLastWriteBufferSize) {
      if (newSize == -1) {
         newSize = Int_t(1.05*Float_t(fBufferRef->Length()));
      }
      fLastWriteBufferSize = newSize;
   }
   */
   if (newSize != -1) {
       fBufferRef->Expand(newSize,kFALSE);     // Expand without copying the existing data.
   }

   TKey::Reset();

   Int_t newNevBufSize = fBranch->GetEntryOffsetLen();
   if (newNevBufSize==0) {
      delete [] fEntryOffset;
      fEntryOffset = 0;
   } else if (newNevBufSize != fNevBufSize) {
      delete [] fEntryOffset;
      fEntryOffset = new Int_t[newNevBufSize];
   } else if (!fEntryOffset) {
      fEntryOffset = new Int_t[newNevBufSize];
   }
   fNevBufSize = newNevBufSize;

   fNevBuf      = 0;
   Int_t *storeEntryOffset = fEntryOffset;
   fEntryOffset = 0; 
   Int_t *storeDisplacement = fDisplacement;
   fDisplacement= 0; 
   fBuffer      = 0;

   fBufferRef->Reset();
   fBufferRef->SetWriteMode();

   fHeaderOnly  = kTRUE;
   fLast        = 0;  //Must initialize before calling Streamer()
   
   Streamer(*fBufferRef);

   fKeylen      = fBufferRef->Length();
   fObjlen      = fBufferSize - fKeylen;
   fLast        = fKeylen;
   fBuffer      = 0;
   fHeaderOnly  = kFALSE;
   fDisplacement= storeDisplacement;
   fEntryOffset = storeEntryOffset;
   if (fNevBufSize) {
      for (Int_t i=0;i<fNevBufSize;i++) fEntryOffset[i] = 0;
   }   
}

//_______________________________________________________________________
void TBasket::SetReadMode()
{
   // Set read mode of basket.

   fLast = fBufferRef->Length();
   fBufferRef->SetReadMode();
}

//_______________________________________________________________________
void TBasket::SetWriteMode()
{
   // Set write mode of basket.

   fBufferRef->SetWriteMode();
   fBufferRef->SetBufferOffset(fLast);
}

//_______________________________________________________________________
void TBasket::Streamer(TBuffer &b)
{
   // Stream a class object.

   char flag;
   if (b.IsReading()) {
      TKey::Streamer(b); //this must be first
      Version_t v = b.ReadVersion();
      b >> fBufferSize;
      b >> fNevBufSize;
      if (fNevBufSize < 0) {
         Error("Streamer","The value of fNevBufSize is incorrect (%d) ; trying to recover by setting it to zero",fNevBufSize);
         MakeZombie();
         fNevBufSize = 0;
      }
      b >> fNevBuf;
      b >> fLast;
      b >> flag;
      if (fLast > fBufferSize) fBufferSize = fLast;
      if (!flag) {
         return;
      }
      if (flag%10 != 2) {
         delete [] fEntryOffset;
         fEntryOffset = new Int_t[fNevBufSize];
         if (fNevBuf) b.ReadArray(fEntryOffset);
         if (20<flag && flag<40) {
            for(int i=0; i<fNevBuf; i++){
               fEntryOffset[i] &= ~kDisplacementMask;
            }
         }
         if (flag>40) {
            fDisplacement = new Int_t[fNevBufSize];
            b.ReadArray(fDisplacement);
         }
      }
      if (flag == 1 || flag > 10) {
         fBufferRef = new TBufferFile(TBuffer::kRead,fBufferSize);
         fBufferRef->SetParent(b.GetParent());
         char *buf  = fBufferRef->Buffer();
         if (v > 1) b.ReadFastArray(buf,fLast);
         else       b.ReadArray(buf);
         fBufferRef->SetBufferOffset(fLast);
         // This is now done in the TBranch streamer since fBranch might not
         // yet be set correctly.
         //   fBranch->GetTree()->IncrementTotalBuffers(fBufferSize);
      }
   } else {
      TKey::Streamer(b);   //this must be first
      b.WriteVersion(TBasket::IsA());
      if (fBufferRef) {
         Int_t curLast = fBufferRef->Length();
         if (!fHeaderOnly && !fSeekKey && curLast > fLast) fLast = curLast;
      }
      if (fLast > fBufferSize) fBufferSize = fLast;

//   static TStopwatch timer;
//   timer.Start(kFALSE);

//       //  Check may be fEntryOffset is equidistant
//       //  This attempts by Victor fails :(
//       int equidist = 0;
//       if (1 && fEntryOffset && fNevBuf>=3) {
//          equidist = 1;
//          int dist = fEntryOffset[1]-fEntryOffset[0];
//          int curr = fEntryOffset[1];
//          for (int i=1;i<fNevBuf;i++,curr+=dist) {
//             if (fEntryOffset[i]==curr) continue;
//             equidist = 0;
//             break;
//          }
//          if (equidist) {
//             fNevBufSize=dist;
//             delete [] fEntryOffset; fEntryOffset = 0;
//          }
//           if (equidist) {
//              fprintf(stderr,"detected an equidistant case fNbytes==%d fLast==%d\n",fNbytes,fLast);
//           }
//       }
//  also he add (a little further
//       if (!fEntryOffset || equidist)  flag  = 2;
    
//   timer.Stop();
//   Double_t rt1 = timer.RealTime();
//   Double_t cp1 = timer.CpuTime();
//   fprintf(stderr,"equidist cost :  RT=%6.2f s  Cpu=%6.2f s\n",rt1,cp1);

      b << fBufferSize;
      b << fNevBufSize;
      b << fNevBuf;
      b << fLast;
      if (fHeaderOnly) {
         flag = 0;
         b << flag;
      } else {
         flag = 1;
         if (!fEntryOffset)  flag  = 2;
         if (fBufferRef)     flag += 10;
         if (fDisplacement)  flag += 40;
         b << flag;

         if (fEntryOffset && fNevBuf) {
            b.WriteArray(fEntryOffset, fNevBuf);
            if (fDisplacement) b.WriteArray(fDisplacement, fNevBuf);
         }
         if (fBufferRef) {
            char *buf  = fBufferRef->Buffer();
            b.WriteFastArray(buf, fLast);
         }
      }
   }
}

//_______________________________________________________________________
void TBasket::Update(Int_t offset, Int_t skipped)
{
   // Update basket header and EntryOffset table.

   if (fEntryOffset) {
      if (fNevBuf+1 >= fNevBufSize) {
         Int_t newsize = TMath::Max(10,2*fNevBufSize);
         Int_t *newoff = TStorage::ReAllocInt(fEntryOffset, newsize,
                                              fNevBufSize);
         if (fDisplacement) {
            Int_t *newdisp = TStorage::ReAllocInt(fDisplacement, newsize,
                                                  fNevBufSize);
            fDisplacement = newdisp;
         }
         fEntryOffset  = newoff;
         fNevBufSize   = newsize;

         //Update branch only for the first 10 baskets
         if (fBranch->GetWriteBasket() < 10) {
            fBranch->SetEntryOffsetLen(newsize);
         }
      }
      fEntryOffset[fNevBuf] = offset;

      if (skipped!=offset && !fDisplacement){
         fDisplacement = new Int_t[fNevBufSize];
         for (Int_t i = 0; i<fNevBufSize; i++) fDisplacement[i] = fEntryOffset[i];
      }
      if (fDisplacement) {
         fDisplacement[fNevBuf] = skipped;
         fBufferRef->SetBufferDisplacement(skipped);
      }
   }

   fNevBuf++;
}

//_______________________________________________________________________
Int_t TBasket::WriteBuffer()
{
   // Write buffer of this basket on the current file.
   //
   // The function returns the number of bytes committed to the memory.
   // If a write error occurs, the number of bytes returned is -1.
   // If no data are written, the number of bytes returned is 0.
   //

   const Int_t kWrite = 1;

   TFile *file = fBranch->GetFile(kWrite);
   if (!file) return 0;
   if (!file->IsWritable()) { 
      return -1;
   }
   fMotherDir = file; // fBranch->GetDirectory();
  
   if (R__unlikely(fBufferRef->TestBit(TBufferFile::kNotDecompressed))) {
      // Read the basket information that was saved inside the buffer.
      Bool_t writing = fBufferRef->IsWriting();
      fBufferRef->SetReadMode();
      fBufferRef->SetBufferOffset(0);

      Streamer(*fBufferRef);
      if (writing) fBufferRef->SetWriteMode();
      Int_t nout = fNbytes - fKeylen;

      fBuffer = fBufferRef->Buffer();

      Create(nout,file);
      fBufferRef->SetBufferOffset(0);
      fHeaderOnly = kTRUE;

      Streamer(*fBufferRef);         //write key itself again
      int nBytes = WriteFileKeepBuffer();
      fHeaderOnly = kFALSE;
      return nBytes>0 ? fKeylen+nout : -1;
   }

   // Transfer fEntryOffset table at the end of fBuffer.
   fLast = fBufferRef->Length();
   if (fEntryOffset) {
      // Note: We might want to investigate the compression gain if we 
      // transform the Offsets to fBuffer in entry length to optimize 
      // compression algorithm.  The aggregate gain on a (random) CMS files
      // is around 5.5%. So the code could something like:
      //      for(Int_t z = fNevBuf; z > 0; --z) {
      //         if (fEntryOffset[z]) fEntryOffset[z] = fEntryOffset[z] - fEntryOffset[z-1];
      //      }
      fBufferRef->WriteArray(fEntryOffset,fNevBuf+1);
      if (fDisplacement) {
         fBufferRef->WriteArray(fDisplacement,fNevBuf+1);
         delete [] fDisplacement; fDisplacement = 0;
      }
   }

   Int_t lbuf, nout, noutot, bufmax, nzip;
   lbuf       = fBufferRef->Length();
   fObjlen    = lbuf - fKeylen;

   fHeaderOnly = kTRUE;
   fCycle = fBranch->GetWriteBasket();
   Int_t cxlevel = fBranch->GetCompressionLevel();
   Int_t cxAlgorithm = fBranch->GetCompressionAlgorithm();
   if (cxlevel > 0) {
     Int_t nbuffers = 1 + (fObjlen - 1) / kMAXBUF;
     Int_t buflen = fKeylen + fObjlen + 9 * nbuffers + 28; //add 28 bytes in case object is placed in a deleted gap
      InitializeCompressedBuffer(buflen, file);
      if (!fCompressedBufferRef) {
         Warning("WriteBuffer", "Unable to allocate the compressed buffer");
         return -1;
      }
      fCompressedBufferRef->SetWriteMode();
      fBuffer = fCompressedBufferRef->Buffer();
      char *objbuf = fBufferRef->Buffer() + fKeylen;
      char *bufcur = &fBuffer[fKeylen];
      noutot = 0;
      nzip   = 0;
      for (Int_t i = 0; i < nbuffers; ++i) {
         if (i == nbuffers - 1) bufmax = fObjlen - nzip;
         else bufmax = kMAXBUF;
         //compress the buffer
         R__zipMultipleAlgorithm(cxlevel, &bufmax, objbuf, &bufmax, bufcur, &nout, cxAlgorithm);
         
         // test if buffer has really been compressed. In case of small buffers 
         // when the buffer contains random data, it may happen that the compressed
         // buffer is larger than the input. In this case, we write the original uncompressed buffer
         if (nout == 0 || nout >= fObjlen) {
            nout = fObjlen;
            // We used to delete fBuffer here, we no longer want to since
            // the buffer (held by fCompressedBufferRef) might be re-used later.
            fBuffer = fBufferRef->Buffer();
            Create(fObjlen,file);
            fBufferRef->SetBufferOffset(0);

            Streamer(*fBufferRef);         //write key itself again
            if ((nout+fKeylen)>buflen) {
               Warning("WriteBuffer","Possible memory corruption due to compression algorithm, wrote %d bytes past the end of a block of %d bytes. fNbytes=%d, fObjLen=%d, fKeylen=%d",
                  (nout+fKeylen-buflen),buflen,fNbytes,fObjlen,fKeylen);
            }
            goto WriteFile;
         }
         bufcur += nout;
         noutot += nout;
         objbuf += kMAXBUF;
         nzip   += kMAXBUF;
      }
      nout = noutot;
      Create(noutot,file);
      fBufferRef->SetBufferOffset(0);

      Streamer(*fBufferRef);         //write key itself again
      memcpy(fBuffer,fBufferRef->Buffer(),fKeylen);
   } else {
      fBuffer = fBufferRef->Buffer();
      Create(fObjlen,file);
      fBufferRef->SetBufferOffset(0);

      Streamer(*fBufferRef);         //write key itself again
      nout = fObjlen;
   }

WriteFile:
   Int_t nBytes = WriteFileKeepBuffer();
   fHeaderOnly = kFALSE;
   return nBytes>0 ? fKeylen+nout : -1;
}

