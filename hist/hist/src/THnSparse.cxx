// @(#)root/hist:$Id$
// Author: Axel Naumann (2007-09-11)

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THnSparse.h"

#include "TArrayI.h"
#include "TAxis.h"
#include "TBrowser.h"
#include "TClass.h"
#include "TCollection.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TH3D.h"
#include "TF1.h"
#include "TInterpreter.h"
#include "TMath.h"
#include "TRandom.h"
#include "TVirtualPad.h"

#include "TError.h"

#include "HFitInterface.h"
#include "Fit/SparseData.h"
#include "Math/MinimizerOptions.h"
#include "Math/WrappedMultiTF1.h"

//______________________________________________________________________________
//
// THnSparseCoordCompression is a class used by THnSparse internally. It
// represents a compacted n-dimensional array of bin coordinates (indices).
// As the total number of bins in each dimension is known by THnSparse, bin
// indices can be compacted to only use the amount of bins needed by the total
// number of bins in each dimension. E.g. for a THnSparse with
// {15, 100, 2, 20, 10, 100} bins per dimension, a bin index will only occupy
// 28 bits (4+7+1+5+4+7), i.e. less than a 32bit integer. The tricky part is
// the fast compression and decompression, the platform-independent storage
// (think of endianness: the bits of the number 0x123456 depend on the
// platform), and the hashing needed by THnSparseArrayChunk.
//______________________________________________________________________________

class THnSparseCoordCompression {
public:
   THnSparseCoordCompression(Int_t dim, const Int_t* nbins);
   THnSparseCoordCompression(const THnSparseCoordCompression& other);
   ~THnSparseCoordCompression();

   ULong64_t GetHashFromBuffer(const Char_t* buf) const;
   Int_t     GetBufferSize() const { return fCoordBufferSize; }
   Int_t     GetNdimensions() const { return fNdimensions; }
   void      SetCoordFromBuffer(const Char_t* buf_in, Int_t* coord_out) const;
   ULong64_t SetBufferFromCoord(const Int_t* coord_in, Char_t* buf_out) const;

protected:
   Int_t     GetNumBits(Int_t n) const {
      // return the number of bits allocated by the number "n"
      Int_t r = (n > 0);
      while (n/=2) ++r;
      return r;
   }
private:
   Int_t  fNdimensions;     // number of dimensions
   Int_t  fCoordBufferSize; // size of coordbuf
   Int_t *fBitOffsets;      //[fNdimensions + 1] bit offset of each axis index
};


//______________________________________________________________________________
//______________________________________________________________________________


//______________________________________________________________________________
THnSparseCoordCompression::THnSparseCoordCompression(Int_t dim, const Int_t* nbins):
   fNdimensions(dim), fCoordBufferSize(0), fBitOffsets(0)
{
   // Initialize a THnSparseCoordCompression object with "dim" dimensions
   // and "bins" holding the number of bins for each dimension; it
   // stores the 

   fBitOffsets = new Int_t[dim + 1];

   int shift = 0;
   for (Int_t i = 0; i < dim; ++i) {
      fBitOffsets[i] = shift;
      shift += GetNumBits(nbins[i] + 2);
   }
   fBitOffsets[dim] = shift;
   fCoordBufferSize = (shift + 7) / 8;
}


//______________________________________________________________________________
THnSparseCoordCompression::THnSparseCoordCompression(const THnSparseCoordCompression& other)
{
   // Construct a THnSparseCoordCompression from another one
   fNdimensions = other.fNdimensions;
   fCoordBufferSize = other.fCoordBufferSize;
   fBitOffsets = new Int_t[fNdimensions + 1];
   memcpy(fBitOffsets, other.fBitOffsets, sizeof(Int_t) * fNdimensions);
}


//______________________________________________________________________________
THnSparseCoordCompression::~THnSparseCoordCompression()
{
   // destruct a THnSparseCoordCompression
   delete [] fBitOffsets;
}


//______________________________________________________________________________
void THnSparseCoordCompression::SetCoordFromBuffer(const Char_t* buf_in,
                                                  Int_t* coord_out) const
{
   // Given the compressed coordinate buffer buf_in, calculate ("decompact")
   // the bin coordinates and return them in coord_out.

   for (Int_t i = 0; i < fNdimensions; ++i) {
      const Int_t offset = fBitOffsets[i] / 8;
      Int_t shift = fBitOffsets[i] % 8;
      Int_t nbits = fBitOffsets[i + 1] - fBitOffsets[i];
      const UChar_t* pbuf = (const UChar_t*) buf_in + offset;
      coord_out[i] = *pbuf >> shift;
      Int_t subst = (Int_t) -1;
      subst = subst << nbits;
      nbits -= (8 - shift);
      shift = 8 - shift;
      for (Int_t n = 0; n * 8 < nbits; ++n) {
         ++pbuf;
         coord_out[i] += *pbuf << shift;
         shift += 8;
      }
      coord_out[i] &= ~subst;
   }
}


//______________________________________________________________________________
ULong64_t THnSparseCoordCompression::SetBufferFromCoord(const Int_t* coord_in,
                                                       Char_t* buf_out) const
{
   // Given the cbin coordinates coord_in, calculate ("compact")
   // the bin coordinates and return them in buf_in.
   // Return the hash value.

   if (fCoordBufferSize <= 8) {
      ULong64_t l64buf = 0;
      for (Int_t i = 0; i < fNdimensions; ++i) {
         l64buf += ((ULong64_t)((UInt_t)coord_in[i])) << fBitOffsets[i];
      }
      memcpy(buf_out, &l64buf, sizeof(Long64_t));
      return l64buf;
   }

   // else: doesn't fit into a Long64_t:
   memset(buf_out, 0, fCoordBufferSize);
   for (Int_t i = 0; i < fNdimensions; ++i) {
      const Int_t offset = fBitOffsets[i] / 8;
      const Int_t shift = fBitOffsets[i] % 8;
      ULong64_t val = coord_in[i];

      Char_t* pbuf = buf_out + offset;
      *pbuf += 0xff & (val << shift);
      val = val >> (8 - shift);
      while (val) {
         ++pbuf;
         *pbuf += 0xff & val;
         val = val >> 8;
      }
   }

   return GetHashFromBuffer(buf_out);
}

/*
//______________________________________________________________________________
ULong64_t THnSparseCoordCompression::GetHashFromCoords(const Int_t* coord) const
{
   // Calculate hash from bin indexes.

   // Bins are addressed in two different modes, depending
   // on whether the compact bin index fits into a Long64_t or not.
   // If it does, we can use it as a "perfect hash" for the TExMap.
   // If not we build a hash from the compact bin index, and use that
   // as the TExMap's hash.

   if (fCoordBufferSize <= 8) {
      // fits into a Long64_t
      ULong64_t hash1 = 0;
      for (Int_t i = 0; i < fNdimensions; ++i) {
         hash1 += coord[i] << fBitOffsets[i];
      }
      return hash1;
   }

   // else: doesn't fit into a Long64_t:
   memset(coord, 0, fCoordBufferSize);
   for (Int_t i = 0; i < fNdimensions; ++i) {
      const Int_t offset = fBitOffsets[i] / 8;
      const Int_t shift = fBitOffsets[i] % 8;
      ULong64_t val = coord[i];

      Char_t* pbuf = fCoordBuffer + offset;
      *pbuf += 0xff & (val << shift);
      val = val >> (8 - shift);
      while (val) {
         ++pbuf;
         *pbuf += 0xff & val;
         val = val >> 8;
      }
   }

   ULong64_t hash = 5381;
   Char_t* str = fCoordBuffer;
   while (str - fCoordBuffer < fCoordBufferSize) {
      hash *= 5;
      hash += *(str++);
   }
   return hash;
}
*/


//______________________________________________________________________________
ULong64_t THnSparseCoordCompression::GetHashFromBuffer(const Char_t* buf) const
{
   // Calculate hash from compact bin index.

   // Bins are addressed in two different modes, depending
   // on whether the compact bin index fits into a Long64_t or not.
   // If it does, we can use it as a "perfect hash" for the TExMap.
   // If not we build a hash from the compact bin index, and use that
   // as the TExMap's hash.

   if (fCoordBufferSize <= 8) {
      // fits into a Long64_t
      ULong64_t hash1 = 0;
      memcpy(&hash1, buf, fCoordBufferSize);
      return hash1;
   }

   // else: doesn't fit into a Long64_t:
   ULong64_t hash = 5381;
   const Char_t* str = buf;
   while (str - buf < fCoordBufferSize) {
      hash *= 5;
      hash += *(str++);
   }
   return hash;
}




//______________________________________________________________________________
//
// THnSparseCompactBinCoord is a class used by THnSparse internally. It
// maps between an n-dimensional array of bin coordinates (indices) and
// its compact version, the THnSparseCoordCompression.
//______________________________________________________________________________

class THnSparseCompactBinCoord: public THnSparseCoordCompression {
public:
   THnSparseCompactBinCoord(Int_t dim, const Int_t* nbins);
   ~THnSparseCompactBinCoord();
   Int_t*    GetCoord() { return fCurrentBin; }
   const Char_t*   GetBuffer() const { return fCoordBuffer; }
   ULong64_t GetHash() const { return fHash; }
   void UpdateCoord() {
      fHash = SetBufferFromCoord(fCurrentBin, fCoordBuffer);
   }
   void      SetCoord(const Int_t* coord) {
      memcpy(fCurrentBin, coord, sizeof(Int_t) * GetNdimensions());
      fHash = SetBufferFromCoord(coord, fCoordBuffer);
   }
   void      SetBuffer(const Char_t* buf) {
      memcpy(fCoordBuffer, buf, GetBufferSize());
      fHash = GetHashFromBuffer(fCoordBuffer);
   }

private:
   ULong64_t fHash;      // hash for current coordinates; 0 if not calculated
   Char_t *fCoordBuffer; // compact buffer of coordinates
   Int_t *fCurrentBin;   // current coordinates
};


//______________________________________________________________________________
//______________________________________________________________________________


//______________________________________________________________________________
THnSparseCompactBinCoord::THnSparseCompactBinCoord(Int_t dim, const Int_t* nbins):
   THnSparseCoordCompression(dim, nbins),
   fHash(0), fCoordBuffer(0), fCurrentBin(0)
{
   // Initialize a THnSparseCompactBinCoord object with "dim" dimensions
   // and "bins" holding the number of bins for each dimension.

   fCurrentBin = new Int_t[dim];
   size_t bufAllocSize = GetBufferSize();
   if (bufAllocSize < sizeof(Long64_t))
      bufAllocSize = sizeof(Long64_t);
   fCoordBuffer = new Char_t[bufAllocSize];
}


//______________________________________________________________________________
THnSparseCompactBinCoord::~THnSparseCompactBinCoord()
{
   // destruct a THnSparseCompactBinCoord

   delete [] fCoordBuffer;
   delete [] fCurrentBin;
}

//______________________________________________________________________________
//
// THnSparseArrayChunk is used internally by THnSparse.
//
// THnSparse stores its (dynamic size) array of bin coordinates and their
// contents (and possibly errors) in a TObjArray of THnSparseArrayChunk. Each
// of the chunks holds an array of THnSparseCompactBinCoord and the content
// (a TArray*), which is created outside (by the templated derived classes of
// THnSparse) and passed in at construction time.
//______________________________________________________________________________


ClassImp(THnSparseArrayChunk);

//______________________________________________________________________________
THnSparseArrayChunk::THnSparseArrayChunk(Int_t coordsize, bool errors, TArray* cont):
      fCoordinateAllocationSize(-1), fSingleCoordinateSize(coordsize), fCoordinatesSize(0),
      fCoordinates(0), fContent(cont),
      fSumw2(0)
{
   // (Default) initialize a chunk. Takes ownership of cont (~THnSparseArrayChunk deletes it),
   // and create an ArrayF for errors if "errors" is true.

   fCoordinateAllocationSize = fSingleCoordinateSize * cont->GetSize();
   fCoordinates = new Char_t[fCoordinateAllocationSize];
   if (errors) Sumw2();
}

//______________________________________________________________________________
THnSparseArrayChunk::~THnSparseArrayChunk()
{
   // Destructor
   delete fContent;
   delete [] fCoordinates;
   delete fSumw2;
}

//______________________________________________________________________________
void THnSparseArrayChunk::AddBin(Int_t idx, const Char_t* coordbuf)
{
   // Create a new bin in this chunk

   // When streaming out only the filled chunk is saved.
   // When reading back only the memory needed for that filled part gets
   // allocated. We need to check whether the allowed chunk size is
   // bigger than the allocated size. If fCoordinateAllocationSize is
   // set to -1 this chunk has been allocated by the  streamer and the
   // buffer allocation size is defined by [fCoordinatesSize]. In that
   // case we need to compare fCoordinatesSize to
   // fSingleCoordinateSize * fContent->GetSize()
   // to determine whether we need to expand the buffer.
   if (fCoordinateAllocationSize == -1 && fContent) {
      Int_t chunksize = fSingleCoordinateSize * fContent->GetSize();
      if (fCoordinatesSize < chunksize) {
         // need to re-allocate:
         Char_t *newcoord = new Char_t[chunksize];
         memcpy(newcoord, fCoordinates, fCoordinatesSize);
         delete [] fCoordinates;
         fCoordinates = newcoord;
      }
      fCoordinateAllocationSize = chunksize;
   }

   memcpy(fCoordinates + idx * fSingleCoordinateSize, coordbuf, fSingleCoordinateSize);
   fCoordinatesSize += fSingleCoordinateSize;
}

//______________________________________________________________________________
void THnSparseArrayChunk::Sumw2()
{
   // Turn on support of errors
   if (!fSumw2)
      fSumw2 = new TArrayD(fContent->GetSize());
}



//______________________________________________________________________________
//
//
//    Efficient multidimensional histogram.
//
// Use a THnSparse instead of TH1 / TH2 / TH3 / array for histogramming when
// only a small fraction of bins is filled. A 10-dimensional histogram with 10
// bins per dimension has 10^10 bins; in a naive implementation this will not
// fit in memory. THnSparse only allocates memory for the bins that have
// non-zero bin content instead, drastically reducing both the memory usage
// and the access time.
//
// To construct a THnSparse object you must use one of its templated, derived
// classes:
// THnSparseD (typedef for THnSparse<ArrayD>): bin content held by a Double_t,
// THnSparseF (typedef for THnSparse<ArrayF>): bin content held by a Float_t,
// THnSparseL (typedef for THnSparse<ArrayL>): bin content held by a Long_t,
// THnSparseI (typedef for THnSparse<ArrayI>): bin content held by an Int_t,
// THnSparseS (typedef for THnSparse<ArrayS>): bin content held by a Short_t,
// THnSparseC (typedef for THnSparse<ArrayC>): bin content held by a Char_t,
//
// They take name and title, the number of dimensions, and for each dimension
// the number of bins, the minimal, and the maximal value on the dimension's
// axis. A TH2 h("h","h",10, 0., 10., 20, -5., 5.) would correspond to
//   Int_t bins[2] = {10, 20};
//   Double_t xmin[2] = {0., -5.};
//   Double_t xmax[2] = {10., 5.};
//   THnSparse hs("hs", "hs", 2, bins, min, max);
//
// * Filling
// A THnSparse is filled just like a regular histogram, using
// THnSparse::Fill(x, weight), where x is a n-dimensional Double_t value.
// To take errors into account, Sumw2() must be called before filling the
// histogram.
// Bins are allocated as needed; the status of the allocation can be observed
// by GetSparseFractionBins(), GetSparseFractionMem().
//
// * Fast Bin Content Access
// When iterating over a THnSparse one should only look at filled bins to save
// processing time. The number of filled bins is returned by
// THnSparse::GetNbins(); the bin content for each (linear) bin number can
// be retrieved by THnSparse::GetBinContent(linidx, (Int_t*)coord).
// After the call, coord will contain the bin coordinate of each axis for the bin
// with linear index linidx. A possible call would be
//   cout << hs.GetBinContent(0, coord);
//   cout <<" is the content of bin [x = " << coord[0] "
//        << " | y = " << coord[1] << "]" << endl;
//
// * Efficiency
// TH1 and TH2 are generally faster than THnSparse for one and two dimensional
// distributions. THnSparse becomes competitive for a sparsely filled TH3
// with large numbers of bins per dimension. The tutorial hist/sparsehist.C
// shows the turning point. On a AMD64 with 8GB memory, THnSparse "wins"
// starting with a TH3 with 30 bins per dimension. Using a THnSparse for a
// one-dimensional histogram is only reasonable if it has a huge number of bins.
//
// * Projections
// The dimensionality of a THnSparse can be reduced by projecting it to
// 1, 2, 3, or n dimensions, which can be represented by a TH1, TH2, TH3, or
// a THnSparse. See the Projection() members. To only project parts of the
// histogram, call
//   THnSparse::GetAxis(12)->SetRange(from_bin, to_bin);
// See the important remark in THnSparse::IsInRange() when excluding under-
// and overflow bins!
//
// * Internal Representation
// An entry for a filled bin consists of its n-dimensional coordinates and
// its bin content. The coordinates are compacted to use as few bits as
// possible; e.g. a histogram with 10 bins in x and 20 bins in y will only
// use 4 bits for the x representation and 5 bits for the y representation.
// This is handled by the internal class THnSparseCompactBinCoord.
// Bin data (content and coordinates) are allocated in chunks of size
// fChunkSize; this parameter can be set when constructing a THnSparse. Each
// chunk is represented by an object of class THnSparseArrayChunk.
//
// Translation from an n-dimensional bin coordinate to the linear index within
// the chunks is done by GetBin(). It creates a hash from the compacted bin
// coordinates (the hash of a bin coordinate is the compacted coordinate itself
// if it takes less than 8 bytes, the size of a Long64_t.
// This hash is used to lookup the linear index in the TExMap member fBins;
// the coordinates of the entry fBins points to is compared to the coordinates
// passed to GetBin(). If they do not match, these two coordinates have the same
// hash - which is extremely unlikely but (for the case where the compact bin
// coordinates are larger than 4 bytes) possible. In this case, fBinsContinued
// contains a chain of linear indexes with the same hash. Iterating through this
// chain and comparing each bin coordinates with the one passed to GetBin() will
// retrieve the matching bin.


ClassImp(THnSparse);

//______________________________________________________________________________
THnSparse::THnSparse():
   fNdimensions(0), fChunkSize(1024), fFilledBins(0), fEntries(0),
   fTsumw(0), fTsumw2(-1.), fCompactCoord(0), fIntegral(0), fIntegralStatus(kNoInt)
{
   // Construct an empty THnSparse.
   fBinContent.SetOwner();
}

//______________________________________________________________________________
THnSparse::THnSparse(const char* name, const char* title, Int_t dim,
                     const Int_t* nbins, const Double_t* xmin, const Double_t* xmax,
                     Int_t chunksize):
   TNamed(name, title), fNdimensions(dim), fChunkSize(chunksize), fFilledBins(0),
   fAxes(dim), fEntries(0), fTsumw(0), fTsumw2(-1.), fTsumwx(dim), fTsumwx2(dim),
   fCompactCoord(0), fIntegral(0), fIntegralStatus(kNoInt)
{
   // Construct a THnSparse with "dim" dimensions,
   // with chunksize as the size of the chunks.
   // "nbins" holds the number of bins for each dimension;
   // "xmin" and "xmax" the minimal and maximal value for each dimension.
   // The arrays "xmin" and "xmax" can be NULL; in that case SetBinEdges()
   // must be called for each dimension.

   for (Int_t i = 0; i < fNdimensions; ++i) {
      TAxis* axis = new TAxis(nbins[i], xmin ? xmin[i] : 0., xmax ? xmax[i] : 1.);
      fAxes.AddAtAndExpand(axis, i);
   }
   SetTitle(title);
   fAxes.SetOwner();

   fCompactCoord = new THnSparseCompactBinCoord(dim, nbins);
   fBinContent.SetOwner();
}

//______________________________________________________________________________
THnSparse::~THnSparse() {
   // Destruct a THnSparse

   delete fCompactCoord;
   if (fIntegralStatus != kNoInt) delete [] fIntegral;
}

//______________________________________________________________________________
void THnSparse::AddBinContent(const Int_t* coord, Double_t v)
{
   // Add "v" to the content of bin with coordinates "coord"

   GetCompactCoord()->SetCoord(coord);
   Long64_t bin = GetBinIndexForCurrentBin(kTRUE);
   return AddBinContent(bin, v);
}

//______________________________________________________________________________
void THnSparse::AddBinContent(Long64_t bin, Double_t v)
{
   // Add "v" to the content of bin with index "bin"

   THnSparseArrayChunk* chunk = GetChunk(bin / fChunkSize);
   bin %= fChunkSize;
   v += chunk->fContent->GetAt(bin);
   return chunk->fContent->SetAt(v, bin);
}

//______________________________________________________________________________
THnSparseArrayChunk* THnSparse::AddChunk()
{
   //Create a new chunk of bin content
   THnSparseArrayChunk* chunk =
      new THnSparseArrayChunk(GetCompactCoord()->GetBufferSize(),
                              GetCalculateErrors(), GenerateArray());
   fBinContent.AddLast(chunk);
   return chunk;
}

//______________________________________________________________________________
THnSparse* THnSparse::CloneEmpty(const char* name, const char* title,
                                 const TObjArray* axes, Int_t chunksize,
                                 Bool_t keepTargetAxis) const
{
   // Create a new THnSparse object that is of the same type as *this,
   // but with dimensions and bins given by axes.
   // If keepTargetAxis is true, the axes will keep their original xmin / xmax,
   // else they will be restricted to the range selected (first / last).

   THnSparse* ret = (THnSparse*)IsA()->New();
   ret->SetNameTitle(name, title);
   ret->fNdimensions = axes->GetEntriesFast();
   ret->fChunkSize = chunksize;

   TIter iAxis(axes);
   const TAxis* axis = 0;
   Int_t pos = 0;
   Int_t *nbins = new Int_t[axes->GetEntriesFast()];
   while ((axis = (TAxis*)iAxis())) {
      TAxis* reqaxis = (TAxis*)axis->Clone();
      if (!keepTargetAxis && axis->TestBit(TAxis::kAxisRange)) {
         Int_t binFirst = axis->GetFirst();
         Int_t binLast = axis->GetLast();
         Int_t nBins = binLast - binFirst + 1;
         if (axis->GetXbins()->GetSize()) {
            // non-uniform bins:
            reqaxis->Set(nBins, axis->GetXbins()->GetArray() + binFirst - 1);
         } else {
            // uniform bins:
            reqaxis->Set(nBins, axis->GetBinLowEdge(binFirst), axis->GetBinUpEdge(binLast));
         }
         reqaxis->ResetBit(TAxis::kAxisRange);
      }

      nbins[pos] = reqaxis->GetNbins();
      ret->fAxes.AddAtAndExpand(reqaxis->Clone(), pos++);
   }
   ret->fAxes.SetOwner();

   ret->fCompactCoord = new THnSparseCompactBinCoord(pos, nbins);
   delete [] nbins;

   return ret;
}

//______________________________________________________________________________
TH1* THnSparse::CreateHist(const char* name, const char* title,
                           const TObjArray* axes,
                           Bool_t keepTargetAxis ) const {
   // Create an empty histogram with name and title with a given
   // set of axes. Create a TH1D/TH2D/TH3D, depending on the number
   // of elements in axes.

   const int ndim = axes->GetSize();

   TH1* hist = 0;
   // create hist with dummy axes, we fix them later.
   if (ndim == 1)
      hist = new TH1D(name, title, 1, 0., 1.);
   else if (ndim == 2)
      hist = new TH2D(name, title, 1, 0., 1., 1, 0., 1.);
   else if (ndim == 3)
      hist = new TH3D(name, title, 1, 0., 1., 1, 0., 1., 1, 0., 1.);
   else {
      Error("CreateHist", "Cannot create histogram %s with %d dimensions!", name, ndim);
      return 0;
   }

   TAxis* hax[3] = {hist->GetXaxis(), hist->GetYaxis(), hist->GetZaxis()};
   for (Int_t d = 0; d < ndim; ++d) {
      TAxis* reqaxis = (TAxis*)(*axes)[d];
      hax[d]->SetTitle(reqaxis->GetTitle());
      if (!keepTargetAxis && reqaxis->TestBit(TAxis::kAxisRange)) {
         Int_t binFirst = reqaxis->GetFirst();
         if (binFirst == 0) binFirst = 1;
         Int_t binLast = reqaxis->GetLast();
         Int_t nBins = binLast - binFirst + 1;
         if (reqaxis->GetXbins()->GetSize()) {
            // non-uniform bins:
            hax[d]->Set(nBins, reqaxis->GetXbins()->GetArray() + binFirst - 1);
         } else {
            // uniform bins:
            hax[d]->Set(nBins, reqaxis->GetBinLowEdge(binFirst), reqaxis->GetBinUpEdge(binLast));
         }
      } else {
         if (reqaxis->GetXbins()->GetSize()) {
            // non-uniform bins:
            hax[d]->Set(reqaxis->GetNbins(), reqaxis->GetXbins()->GetArray());
         } else {
            // uniform bins:
            hax[d]->Set(reqaxis->GetNbins(), reqaxis->GetXmin(), reqaxis->GetXmax());
         }
      }
   }

   hist->Rebuild();

   return hist;
}

//______________________________________________________________________________
THnSparse* THnSparse::CreateSparse(const char* name, const char* title,
                                   const TH1* h, Int_t chunkSize)
{
   // Create a THnSparse object from a histogram deriving from TH1.

   // Get the dimension of the TH1
   int ndim = h->GetDimension();

   // Axis properties
   int nbins[3] = {0,0,0};
   double minRange[3] = {0.,0.,0.};
   double maxRange[3] = {0.,0.,0.};
   TAxis* axis[3] = { h->GetXaxis(), h->GetYaxis(), h->GetZaxis() };
   for (int i = 0; i < ndim; ++i) {
      nbins[i]    = axis[i]->GetNbins();
      minRange[i] = axis[i]->GetXmin();
      maxRange[i] = axis[i]->GetXmax();
   }

   // Create the corresponding THnSparse, depending on the storage
   // type of the TH1. The class name will be "TH??\0" where the first
   // ? is 1,2 or 3 and the second ? indicates the stograge as C, S,
   // I, F or D.
   THnSparse* s = 0;
   const char* cname( h->ClassName() );
   if (cname[0] == 'T' && cname[1] == 'H' && cname[2] >= '1' && cname[2] <= '3' && cname[4] == 0) {
      if (cname[3] == 'F') {
         s = new THnSparseF(name, title, ndim, nbins, minRange, maxRange, chunkSize);
      } else if (cname[3] == 'D') {
         s = new THnSparseD(name, title, ndim, nbins, minRange, maxRange, chunkSize);
      } else if (cname[3] == 'I') {
         s = new THnSparseI(name, title, ndim, nbins, minRange, maxRange, chunkSize);
      } else if (cname[3] == 'S') {
         s = new THnSparseS(name, title, ndim, nbins, minRange, maxRange, chunkSize);
      } else if (cname[3] == 'C') {
         s = new THnSparseC(name, title, ndim, nbins, minRange, maxRange, chunkSize);
      }
   }
   if (!s) {
      ::Warning("THnSparse::CreateSparse", "Unknown Type of Histogram");
      return 0;
   }

   for (int i = 0; i < ndim; ++i) {
      s->GetAxis(i)->SetTitle(axis[i]->GetTitle());
   }

   // Get the array to know the number of entries of the TH1
   const TArray *array = dynamic_cast<const TArray*>(h);
   if (!array) {
      ::Warning("THnSparse::CreateSparse", "Unknown Type of Histogram");
      return 0;
   }

   // Fill the THnSparse with the bins that have content.
   for (int i = 0; i < array->GetSize(); ++i) {
      double value = h->GetBinContent(i);
      double error = h->GetBinError(i);
      if (!value && !error) continue;
      int x[3] = {0,0,0};
      h->GetBinXYZ(i, x[0], x[1], x[2]);
      s->SetBinContent(x, value);
      s->SetBinError(x, error);
   }

   return s;
}

//______________________________________________________________________________
void THnSparse::FillExMap()
{
   //We have been streamed; set up fBins
   TIter iChunk(&fBinContent);
   THnSparseArrayChunk* chunk = 0;
   THnSparseCoordCompression compactCoord(*GetCompactCoord());
   Long64_t idx = 0;
   if (2 * GetNbins() > fBins.Capacity())
      fBins.Expand(3 * GetNbins());
   while ((chunk = (THnSparseArrayChunk*) iChunk())) {
      const Int_t chunkSize = chunk->GetEntries();
      Char_t* buf = chunk->fCoordinates;
      const Int_t singleCoordSize = chunk->fSingleCoordinateSize;
      const Char_t* endbuf = buf + singleCoordSize * chunkSize;
      for (; buf < endbuf; buf += singleCoordSize, ++idx) {
         Long64_t hash = compactCoord.GetHashFromBuffer(buf);
         Long64_t linidx = fBins.GetValue(hash);
         if (linidx) {
            Long64_t nextidx = linidx;
            while (nextidx) {
               // must be a collision, so go to fBinsContinued.
               linidx = nextidx;
               nextidx = fBinsContinued.GetValue(linidx);
            }
            fBinsContinued.Add(linidx, idx + 1);
         } else {
            fBins.Add(hash, idx + 1);
         }
      }
   }
}

//______________________________________________________________________________
TFitResultPtr THnSparse::Fit(TF1 *f ,Option_t *option ,Option_t *goption)
{
//   Fit a THnSparse with function f
// 
//   since the data is sparse by default a likelihood fit is performed 
//   merging all the regions with empty bins for betetr performance efficiency
// 
//  Since the THnSparse is not drawn no graphics options are passed 
//  Here is the list of possible options 
// 
//                = "I"  Use integral of function in bin instead of value at bin center
//                = "X"  Use chi2 method (default is log-likelihood method)
//                = "U"  Use a User specified fitting algorithm (via SetFCN)
//                = "Q"  Quiet mode (minimum printing)
//                = "V"  Verbose mode (default is between Q and V)
//                = "E"  Perform better Errors estimation using Minos technique
//                = "B"  Use this option when you want to fix one or more parameters
//                       and the fitting function is like "gaus", "expo", "poln", "landau".
//                = "M"  More. Improve fit results
//                = "R"  Use the Range specified in the function range


   Foption_t fitOption;

   if (!TH1::FitOptionsMake(option,fitOption)) return 0;

   // The function used to fit cannot be stored in a THnSparse. It
   // cannot be drawn either. Perhaps in the future.
   fitOption.Nostore = true;
   // Use likelihood fit if not specified
   if (!fitOption.Chi2) fitOption.Like = true;
   // create range and minimizer options with default values 
   ROOT::Fit::DataRange range(GetNdimensions()); 
   for ( int i = 0; i < GetNdimensions(); ++i ) {
      TAxis *axis = GetAxis(i);
      range.AddRange(i, axis->GetXmin(), axis->GetXmax());
   }
   ROOT::Math::MinimizerOptions minOption; 
   
   return ROOT::Fit::FitObject(this, f , fitOption , minOption, goption, range); 
}

//______________________________________________________________________________
Long64_t THnSparse::GetBin(const Double_t* x, Bool_t allocate /* = kTRUE */)
{
   // Get the bin index for the n dimensional tuple x,
   // allocate one if it doesn't exist yet and "allocate" is true.

   THnSparseCompactBinCoord* cc = GetCompactCoord();
   Int_t *coord = cc->GetCoord();
   for (Int_t i = 0; i < fNdimensions; ++i)
      coord[i] = GetAxis(i)->FindBin(x[i]);
   cc->UpdateCoord();

   return GetBinIndexForCurrentBin(allocate);
}


//______________________________________________________________________________
Long64_t THnSparse::GetBin(const char* name[], Bool_t allocate /* = kTRUE */)
{
   // Get the bin index for the n dimensional tuple addressed by "name",
   // allocate one if it doesn't exist yet and "allocate" is true.

   THnSparseCompactBinCoord* cc = GetCompactCoord();
   Int_t *coord = cc->GetCoord();
   for (Int_t i = 0; i < fNdimensions; ++i)
      coord[i] = GetAxis(i)->FindBin(name[i]);
   cc->UpdateCoord();

   return GetBinIndexForCurrentBin(allocate);
}

//______________________________________________________________________________
Long64_t THnSparse::GetBin(const Int_t* coord, Bool_t allocate /*= kTRUE*/)
{
   // Get the bin index for the n dimensional coordinates coord,
   // allocate one if it doesn't exist yet and "allocate" is true.
   GetCompactCoord()->SetCoord(coord);
   return GetBinIndexForCurrentBin(allocate);
}

//______________________________________________________________________________
Double_t THnSparse::GetBinContent(const Int_t *coord) const {
   // Get content of bin with coordinates "coord"
   GetCompactCoord()->SetCoord(coord);
   Long64_t idx = const_cast<THnSparse*>(this)->GetBinIndexForCurrentBin(kFALSE);
   if (idx < 0) return 0.;
   THnSparseArrayChunk* chunk = GetChunk(idx / fChunkSize);
   return chunk->fContent->GetAt(idx % fChunkSize);
}

//______________________________________________________________________________
Double_t THnSparse::GetBinContent(Long64_t idx, Int_t* coord /* = 0 */) const
{
   // Return the content of the filled bin number "idx".
   // If coord is non-null, it will contain the bin's coordinates for each axis
   // that correspond to the bin.

   if (idx >= 0) {
      THnSparseArrayChunk* chunk = GetChunk(idx / fChunkSize);
      idx %= fChunkSize;
      if (chunk && chunk->fContent->GetSize() > idx) {
         if (coord) {
            THnSparseCompactBinCoord* cc = GetCompactCoord();
            Int_t sizeCompact = cc->GetBufferSize();
            cc->SetCoordFromBuffer(chunk->fCoordinates + idx * sizeCompact,
                                                  coord);
                                                                               
         }
         return chunk->fContent->GetAt(idx);
      }
   }
   if (coord)
      memset(coord, -1, sizeof(Int_t) * fNdimensions);
   return 0.;
}

//______________________________________________________________________________
Double_t THnSparse::GetBinError(const Int_t *coord) const {
   // Get error of bin with coordinates "coord" as
   // BEGIN_LATEX #sqrt{#sum weight^{2}}
   // END_LATEX
   // If errors are not enabled (via Sumw2() or CalculateErrors())
   // return sqrt(contents).

   if (!GetCalculateErrors())
      return TMath::Sqrt(GetBinContent(coord));

   GetCompactCoord()->SetCoord(coord);
   Long64_t idx = const_cast<THnSparse*>(this)->GetBinIndexForCurrentBin(kFALSE);
   if (idx < 0) return 0.;

   THnSparseArrayChunk* chunk = GetChunk(idx / fChunkSize);
   return TMath::Sqrt(chunk->fSumw2->GetAt(idx % fChunkSize));
}

//______________________________________________________________________________
Double_t THnSparse::GetBinError(Long64_t linidx) const {
   // Get error of bin addressed by linidx as
   // BEGIN_LATEX #sqrt{#sum weight^{2}}
   // END_LATEX
   // If errors are not enabled (via Sumw2() or CalculateErrors())
   // return sqrt(contents).

   if (!GetCalculateErrors())
      return TMath::Sqrt(GetBinContent(linidx));

   if (linidx < 0) return 0.;
   THnSparseArrayChunk* chunk = GetChunk(linidx / fChunkSize);
   linidx %= fChunkSize;
   if (!chunk || chunk->fContent->GetSize() < linidx)
      return 0.;

   return TMath::Sqrt(chunk->fSumw2->GetAt(linidx));
}


//______________________________________________________________________________
Long64_t THnSparse::GetBinIndexForCurrentBin(Bool_t allocate)
{
   // Return the index for fCurrentBinIndex.
   // If it doesn't exist then return -1, or allocate a new bin if allocate is set

   THnSparseCompactBinCoord* cc = GetCompactCoord();
   ULong64_t hash = cc->GetHash();
   if (fBinContent.GetSize() && !fBins.GetSize())
      FillExMap();
   Long64_t linidx = (Long64_t) fBins.GetValue(hash);
   while (linidx) {
      // fBins stores index + 1!
      THnSparseArrayChunk* chunk = GetChunk((linidx - 1)/ fChunkSize);
      if (chunk->Matches((linidx - 1) % fChunkSize, cc->GetBuffer()))
         return linidx - 1; // we store idx+1, 0 is "TExMap: not found"

      Long64_t nextlinidx = fBinsContinued.GetValue(linidx);
      if (!nextlinidx) break;

      linidx = nextlinidx;
   }
   if (!allocate) return -1;

   ++fFilledBins;

   // allocate bin in chunk
   THnSparseArrayChunk *chunk = (THnSparseArrayChunk*) fBinContent.Last();
   Long64_t newidx = chunk ? ((Long64_t) chunk->GetEntries()) : -1;
   if (!chunk || newidx == (Long64_t)fChunkSize) {
      chunk = AddChunk();
      newidx = 0;
   }
   chunk->AddBin(newidx, cc->GetBuffer());

   // store translation between hash and bin
   newidx += (fBinContent.GetEntriesFast() - 1) * fChunkSize;
   if (!linidx) {
      // fBins didn't find it
      if (2 * GetNbins() > fBins.Capacity())
         fBins.Expand(3 * GetNbins());
      fBins.Add(hash, newidx + 1);
   } else {
      // fBins contains one, but it's the wrong one;
      // add entry to fBinsContinued.
      fBinsContinued.Add(linidx, newidx + 1);
   }
   return newidx;
}

//______________________________________________________________________________
THnSparseCompactBinCoord* THnSparse::GetCompactCoord() const
{
   // Return THnSparseCompactBinCoord object.

   if (!fCompactCoord) {
      Int_t *bins = new Int_t[fNdimensions];
      for (Int_t d = 0; d < fNdimensions; ++d)
         bins[d] = GetAxis(d)->GetNbins();
      const_cast<THnSparse*>(this)->fCompactCoord
         = new THnSparseCompactBinCoord(fNdimensions, bins);
      delete [] bins;
   }
   return fCompactCoord;
}

//______________________________________________________________________________
void THnSparse::GetRandom(Double_t *rand, Bool_t subBinRandom /* = kTRUE */)
{
   // Generate an n-dimensional random tuple based on the histogrammed
   // distribution. If subBinRandom, the returned tuple will be additionally
   // randomly distributed within the randomized bin, using a flat
   // distribution.

   // check whether the integral array is valid
   if (fIntegralStatus != kValidInt)
      ComputeIntegral();

   // generate a random bin
   Double_t p = gRandom->Rndm();
   Long64_t idx = TMath::BinarySearch(GetNbins() + 1, fIntegral, p);
   const Int_t nStaticBins = 40;
   Int_t bin[nStaticBins];
   Int_t* pBin = bin;
   if (GetNdimensions() > nStaticBins) {
      pBin = new Int_t[GetNdimensions()];
   }
   GetBinContent(idx, pBin);

   // convert bin coordinates to real values
   for (Int_t i = 0; i < fNdimensions; i++) {
      rand[i] = GetAxis(i)->GetBinCenter(pBin[i]);

      // randomize the vector withing a bin
      if (subBinRandom)
         rand[i] += (gRandom->Rndm() - 0.5) * GetAxis(i)->GetBinWidth(pBin[i]);
   }
   if (pBin != bin) {
      delete [] pBin;
   }

   return;
}

//______________________________________________________________________________
Double_t THnSparse::GetSparseFractionBins() const {
   // Return the amount of filled bins over all bins

   Double_t nbinsTotal = 1.;
   for (Int_t d = 0; d < fNdimensions; ++d)
      nbinsTotal *= GetAxis(d)->GetNbins() + 2;
   return fFilledBins / nbinsTotal;
}

//______________________________________________________________________________
Double_t THnSparse::GetSparseFractionMem() const {
   // Return the amount of used memory over memory that would be used by a
   // non-sparse n-dimensional histogram. The value is approximate.

   Int_t arrayElementSize = 0;
   if (fFilledBins) {
      TClass* clArray = GetChunk(0)->fContent->IsA();
      TDataMember* dm = clArray ? clArray->GetDataMember("fArray") : 0;
      arrayElementSize = dm ? dm->GetDataType()->Size() : 0;
   }
   if (!arrayElementSize) {
      Warning("GetSparseFractionMem", "Cannot determine type of elements!");
      return -1.;
   }

   Double_t sizePerChunkElement = arrayElementSize + GetCompactCoord()->GetBufferSize();
   if (fFilledBins && GetChunk(0)->fSumw2)
      sizePerChunkElement += sizeof(Double_t); /* fSumw2 */

   Double_t size = 0.;
   size += fBinContent.GetEntries() * (GetChunkSize() * sizePerChunkElement + sizeof(THnSparseArrayChunk));
   size += + 3 * sizeof(Long64_t) * fBins.GetSize() /* TExMap */;

   Double_t nbinsTotal = 1.;
   for (Int_t d = 0; d < fNdimensions; ++d)
      nbinsTotal *= GetAxis(d)->GetNbins() + 2;

   return size / nbinsTotal / arrayElementSize;
}

//______________________________________________________________________________
Bool_t THnSparse::IsInRange(Int_t *coord) const
{
   // Check whether bin coord is in range, as defined by TAxis::SetRange().
   // Currently, TAxis::SetRange() does not allow to select all but over- and
   // underflow bins (it instead resets the axis to "no range selected").
   // Instead, simply call
   //    TAxis* axis12 = hsparse.GetAxis(12);
   //    axis12->SetRange(1, axis12->GetNbins());
   //    axis12->SetBit(TAxis::kAxisRange);
   // to deselect the under- and overflow bins in the 12th dimension.

   Int_t min = 0;
   Int_t max = 0;
   for (Int_t i = 0; i < fNdimensions; ++i) {
      TAxis *axis = GetAxis(i);
      if (!axis->TestBit(TAxis::kAxisRange)) continue;
      min = axis->GetFirst();
      max = axis->GetLast();
      if (min == 0 && max == 0) {
         // special case where TAxis::SetBit(kAxisRange) and
         // over- and underflow bins are de-selected.
         // first and last are == 0 due to axis12->SetRange(1, axis12->GetNbins());
         min = 1;
         max = axis->GetNbins();
      }
      if (coord[i] < min || coord[i] > max)
         return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
TH1D* THnSparse::Projection(Int_t xDim, Option_t* option /*= ""*/) const
{
   // Project all bins into a 1-dimensional histogram,
   // keeping only axis "xDim".
   // If "option" contains "E" errors will be calculated.
   //                      "A" ranges of the taget axes will be ignored.
   //                      "O" original axis range of the taget axes will be
   //                          kept, but only bins inside the selected range
   //                          will be filled.

   return (TH1D*) ProjectionAny(1, &xDim, false, option);
}

//______________________________________________________________________________
TH2D* THnSparse::Projection(Int_t yDim, Int_t xDim, Option_t* option /*= ""*/) const
{
   // Project all bins into a 2-dimensional histogram,
   // keeping only axes "xDim" and "yDim".
   //
   // WARNING: just like TH3::Project3D("yx") and TTree::Draw("y:x"),
   // Projection(y,x) uses the first argument to define the y-axis and the
   // second for the x-axis!
   //
   // If "option" contains "E" errors will be calculated.
   //                      "A" ranges of the taget axes will be ignored.

   const Int_t dim[2] = {xDim, yDim};
   return (TH2D*) ProjectionAny(2, dim, false, option);
}

//______________________________________________________________________________
TH3D* THnSparse::Projection(Int_t xDim, Int_t yDim, Int_t zDim,
                            Option_t* option /*= ""*/) const
{
   // Project all bins into a 3-dimensional histogram,
   // keeping only axes "xDim", "yDim", and "zDim".
   // If "option" contains "E" errors will be calculated.
   //                      "A" ranges of the taget axes will be ignored.
   //                      "O" original axis range of the taget axes will be
   //                          kept, but only bins inside the selected range
   //                          will be filled.

   const Int_t dim[3] = {xDim, yDim, zDim};
   return (TH3D*) ProjectionAny(3, dim, false, option);
}

//______________________________________________________________________________
THnSparse* THnSparse::Projection(Int_t ndim, const Int_t* dim,
                                 Option_t* option /*= ""*/) const
{
   // Project all bins into a ndim-dimensional THnSparse histogram,
   // keeping only axes in dim (specifying ndim dimensions)
   // If "option" contains "E" errors will be calculated.
   //                      "A" ranges of the taget axes will be ignored.
   //                      "O" original axis range of the taget axes will be
   //                          kept, but only bins inside the selected range
   //                          will be filled.

   return (THnSparse*) ProjectionAny(ndim, dim, true, option);
}


//______________________________________________________________________________
TObject* THnSparse::ProjectionAny(Int_t ndim, const Int_t* dim,
                                  Bool_t wantSparse, Option_t* option /*= ""*/) const
{
   // Project all bins into a ndim-dimensional THnSparse histogram,
   // keeping only axes in dim (specifying ndim dimensions)
   // If "option" contains "E" errors will be calculated.
   //                      "A" ranges of the taget axes will be ignored.
   //                      "O" original axis range of the taget axes will be
   //                          kept, but only bins inside the selected range
   //                          will be filled.

   TString name(GetName());
   name +="_proj";
   
   for (Int_t d = 0; d < ndim; ++d) {
     name += "_";
     name += dim[d];
   }

   TString title(GetTitle());
   Ssiz_t posInsert = title.First(';');
   if (posInsert == kNPOS) {
      title += " projection ";
      for (Int_t d = 0; d < ndim; ++d)
         title += GetAxis(dim[d])->GetTitle();
   } else {
      for (Int_t d = ndim - 1; d >= 0; --d) {
         title.Insert(posInsert, GetAxis(d)->GetTitle());
         if (d)
            title.Insert(posInsert, ", ");
      }
      title.Insert(posInsert, " projection ");
   }

   TObjArray newaxes(ndim);
   for (Int_t d = 0; d < ndim; ++d) {
      newaxes.AddAt(GetAxis(dim[d]),d);
   }

   THnSparse* sparse = 0;
   TH1* hist = 0;
   TObject* ret = 0;

   Bool_t* hadRange = 0;
   Bool_t ignoreTargetRange = (option && (strchr(option, 'A') || strchr(option, 'a')));
   Bool_t keepTargetAxis = ignoreTargetRange || (option && (strchr(option, 'O') || strchr(option, 'o')));
   if (ignoreTargetRange) {
      hadRange = new Bool_t[ndim];
      for (Int_t d = 0; d < ndim; ++d){
         TAxis *axis = GetAxis(dim[d]);
         hadRange[d] = axis->TestBit(TAxis::kAxisRange);
         axis->SetBit(TAxis::kAxisRange, kFALSE);
      }
   }

   if (wantSparse)
      ret = sparse = CloneEmpty(name, title, &newaxes, fChunkSize, keepTargetAxis);
   else
      ret = hist = CreateHist(name, title, &newaxes, keepTargetAxis); 

   if (keepTargetAxis) {
      // make the whole axes visible, i.e. unset the range
      if (wantSparse) {
         for (Int_t d = 0; d < ndim; ++d) {
            sparse->GetAxis(d)->SetRange(0, 0);
         }
      } else {
         hist->GetXaxis()->SetRange(0, 0);
         hist->GetYaxis()->SetRange(0, 0);
         hist->GetZaxis()->SetRange(0, 0);
      }
   }

   Bool_t haveErrors = GetCalculateErrors();
   Bool_t wantErrors = (option && (strchr(option, 'E') || strchr(option, 'e'))) || haveErrors;

   Int_t* bins  = new Int_t[ndim];
   Int_t  linbin = 0;
   Int_t* coord = new Int_t[fNdimensions];
   memset(coord, 0, sizeof(Int_t) * fNdimensions);

   Double_t err = 0.;
   Double_t preverr = 0.;
   Double_t v = 0.;

   for (Long64_t i = 0; i < GetNbins(); ++i) {
      v = GetBinContent(i, coord);

      if (!IsInRange(coord)) continue;

      for (Int_t d = 0; d < ndim; ++d) {
         bins[d] = coord[dim[d]];
         if (!keepTargetAxis && GetAxis(dim[d])->TestBit(TAxis::kAxisRange)) {
            bins[d] -= GetAxis(dim[d])->GetFirst() - 1;
         }
      }

      if (!wantSparse) {
         if (ndim == 1) linbin = bins[0];
         else if (ndim == 2) linbin = hist->GetBin(bins[0], bins[1]);
         else if (ndim == 3) linbin = hist->GetBin(bins[0], bins[1], bins[2]);
      }

      if (wantErrors) {
         if (haveErrors) {
            err = GetBinError(i);
            err *= err;
         } else err = v;
         if (wantSparse) {
            preverr = sparse->GetBinError(bins);
            sparse->SetBinError(bins, TMath::Sqrt(preverr * preverr + err));
         } else {
            preverr = hist->GetBinError(linbin);
            hist->SetBinError(linbin, TMath::Sqrt(preverr * preverr + err));
         }
      }

      // only _after_ error calculation, or sqrt(v) is taken into account!
      if (wantSparse)
         sparse->AddBinContent(bins, v);
      else
         hist->AddBinContent(linbin, v);
   }

   delete [] bins;
   delete [] coord;

   if (wantSparse)
      // need to check also when producing a THnSparse how to reset the number of entries
      sparse->SetEntries(fEntries);
   else  
      // need to reset the statistics which will set also the number of entries
      // according to the bins filled
      hist->ResetStats(); 

   if (hadRange) {
      // reset kAxisRange bit:
      for (Int_t d = 0; d < ndim; ++d)
         GetAxis(dim[d])->SetBit(TAxis::kAxisRange, hadRange[d]);

      delete [] hadRange;
   }

   return ret;
}

//______________________________________________________________________________
void THnSparse::Scale(Double_t c)
{
   // Scale contents and errors of this histogram by c:
   // this = this * c
   // It does not modify the histogram's number of entries.


   Double_t nEntries = GetEntries();
   // Scale the contents & errors
   Bool_t haveErrors = GetCalculateErrors();
   for (Long64_t i = 0; i < GetNbins(); ++i) {
      // Get the content of the bin from the current histogram
      Double_t v = GetBinContent(i);
      SetBinContent(i, c * v);
      if (haveErrors) {
         Double_t err = GetBinError(i);
         SetBinError(i, c * err);
      }
   }
   SetEntries(nEntries);
}

//______________________________________________________________________________
void THnSparse::AddInternal(const THnSparse* h, Double_t c, Bool_t rebinned)
{
   // Add() implementation for both rebinned histograms and those with identical
   // binning. See THnSparse::Add().

   if (fNdimensions != h->GetNdimensions()) {
      Warning("RebinnedAdd", "Different number of dimensions, cannot carry out operation on the histograms");
      return;
   }

   // Trigger error calculation if h has it
   if (!GetCalculateErrors() && h->GetCalculateErrors())
      Sumw2();
   Bool_t haveErrors = GetCalculateErrors();

   // Now add the contents: in this case we have the union of the sets of bins
   Int_t* coord = new Int_t[fNdimensions];
   memset(coord, 0, sizeof(Int_t) * fNdimensions);
   Double_t* x = 0;
   if (rebinned) {
      x = new Double_t[fNdimensions];
   }

   if (!fBins.GetSize() && fBinContent.GetSize()) {
      FillExMap();
   }

   // Expand the exmap if needed, to reduce collisions
   Long64_t numTargetBins = GetNbins() + h->GetNbins();
   if (2 * numTargetBins > fBins.Capacity()) {
      fBins.Expand(3 * numTargetBins);
   }

   // Add to this whatever is found inside the other histogram
   for (Long64_t i = 0; i < h->GetNbins(); ++i) {
      // Get the content of the bin from the second histogram
      Double_t v = h->GetBinContent(i, coord);

      Long64_t binidx = -1;
      if (rebinned) {
         // Get the bin center given a coord
         for (Int_t j = 0; j < fNdimensions; ++j)
            x[j] = h->GetAxis(j)->GetBinCenter(coord[j]);

         binidx = GetBin(x);
      } else {
         binidx = GetBin(coord);
      }

      if (haveErrors) {
         Double_t err1 = GetBinError(binidx);
         Double_t err2 = h->GetBinError(i) * c;
         SetBinError(binidx, TMath::Sqrt(err1 * err1 + err2 * err2));
      }
      // only _after_ error calculation, or sqrt(v) is taken into account!
      AddBinContent(binidx, c * v);
   }


   delete [] coord;
   delete [] x;

   Double_t nEntries = GetEntries() + c * h->GetEntries();
   SetEntries(nEntries);
}

//______________________________________________________________________________
void THnSparse::Add(const THnSparse* h, Double_t c)
{
   // Add contents of h scaled by c to this histogram:
   // this = this + c * h
   // Note that if h has Sumw2 set, Sumw2 is automatically called for this
   // if not already set.

   // Check consistency of the input
   if (!CheckConsistency(h, "Add")) return;

   AddInternal(h, c, kFALSE);
}

//______________________________________________________________________________
void THnSparse::RebinnedAdd(const THnSparse* h, Double_t c)
{
   // Add contents of h scaled by c to this histogram:
   // this = this + c * h
   // Note that if h has Sumw2 set, Sumw2 is automatically called for this
   // if not already set.
   // In contrast to Add(), RebinnedAdd() does not require consist binning of
   // this and h; instead, each bin's center is used to determine the target bin.

   AddInternal(h, c, kTRUE);
}


//______________________________________________________________________________
Long64_t THnSparse::Merge(TCollection* list)
{
   // Merge this with a list of THnSparses. All THnSparses provided
   // in the list must have the same bin layout!

   if (!list) return 0;
   if (list->IsEmpty()) return (Long64_t)GetEntries();

   Long64_t sumNbins = GetNbins();
   TIter iter(list);
   const TObject* addMeObj = 0;
   while ((addMeObj = iter())) {
      const THnSparse* addMe = dynamic_cast<const THnSparse*>(addMeObj);
      if (addMe) {
         sumNbins += addMe->GetNbins();
      }
   }
   if (!fBins.GetSize() && fBinContent.GetSize()) {
      FillExMap();
   }
   if (2 * sumNbins > fBins.Capacity()) {
      fBins.Expand(3 * sumNbins);
   }

   iter.Reset();
   while ((addMeObj = iter())) {
      const THnSparse* addMe = dynamic_cast<const THnSparse*>(addMeObj);
      if (!addMe)
         Error("Merge", "Object named %s is not THnSpase! Skipping it.",
               addMeObj->GetName());
      else
         Add(addMe);
   }
   return (Long64_t)GetEntries();
}


//______________________________________________________________________________
void THnSparse::Multiply(const THnSparse* h)
{
   // Multiply this histogram by histogram h
   // this = this * h
   // Note that if h has Sumw2 set, Sumw2 is automatically called for this
   // if not already set.

   // Check consistency of the input
   if(!CheckConsistency(h, "Multiply"))return;

   // Trigger error calculation if h has it
   Bool_t wantErrors = kFALSE;
   if (GetCalculateErrors() || h->GetCalculateErrors())
      wantErrors = kTRUE;

   if (wantErrors) Sumw2();

   Double_t nEntries = GetEntries();
   // Now multiply the contents: in this case we have the intersection of the sets of bins
   Int_t* coord = new Int_t[fNdimensions];
   for (Long64_t i = 0; i < GetNbins(); ++i) {
      // Get the content of the bin from the current histogram
      Double_t v1 = GetBinContent(i, coord);
      // Now look at the bin with the same coordinates in h
      Double_t v2 = h->GetBinContent(coord);
      SetBinContent(coord, v1 * v2);;
      if (wantErrors) {
         Double_t err1 = GetBinError(coord) * v2;
         Double_t err2 = h->GetBinError(coord) * v1;
         SetBinError(coord,TMath::Sqrt((err2 * err2 + err1 * err1)));
      }
   }
   SetEntries(nEntries);

   delete [] coord;
}

//______________________________________________________________________________
void THnSparse::Multiply(TF1* f, Double_t c)
{
    // Performs the operation: this = this*c*f1
    // if errors are defined, errors are also recalculated.
    //
    // Only bins inside the function range are recomputed.
    // IMPORTANT NOTE: If you intend to use the errors of this histogram later
    // you should call Sumw2 before making this operation.
    // This is particularly important if you fit the histogram after THnSparse::Multiply

   Int_t* coord = new Int_t[fNdimensions];
   memset(coord, 0, sizeof(Int_t) * fNdimensions);
   Double_t* points = new Double_t[fNdimensions];


   Bool_t wantErrors = GetCalculateErrors();
   if (wantErrors) Sumw2();

   ULong64_t nEntries = GetNbins();
   for (ULong64_t i = 0; i < nEntries; ++i) {
      Double_t value = GetBinContent(i, coord);

      // Get the bin co-ordinates given an coord
      for (Int_t j = 0; j < fNdimensions; ++j)
         points[j] = GetAxis(j)->GetBinCenter(coord[j]);

      if (!f->IsInside(points))
         continue;
      TF1::RejectPoint(kFALSE);

      // Evaulate function at points
      Double_t fvalue = f->EvalPar(points, NULL);

      SetBinContent(coord, c * fvalue * value);
      if (wantErrors) {
         Double_t error(GetBinError(i));
         SetBinError(coord, c * fvalue * error);
      }
   }
   
   delete [] points;
   delete [] coord;
}

//______________________________________________________________________________
void THnSparse::Divide(const THnSparse *h)
{
   // Divide this histogram by h
   // this = this/(h)
   // Note that if h has Sumw2 set, Sumw2 is automatically called for
   // this if not already set.
   // The resulting errors are calculated assuming uncorrelated content.

   // Check consistency of the input
   if (!CheckConsistency(h, "Divide"))return;

   // Trigger error calculation if h has it
   Bool_t wantErrors=GetCalculateErrors();
   if (!GetCalculateErrors() && h->GetCalculateErrors())
      wantErrors=kTRUE;

   // Remember original histogram statistics
   Double_t nEntries = fEntries;

   if (wantErrors) Sumw2();
   Bool_t didWarn = kFALSE;

   // Now divide the contents: also in this case we have the intersection of the sets of bins
   Int_t* coord = new Int_t[fNdimensions];
   memset(coord, 0, sizeof(Int_t) * fNdimensions);
   Double_t err = 0.;
   Double_t b22 = 0.;
   for (Long64_t i = 0; i < GetNbins(); ++i) {
      // Get the content of the bin from the first histogram
      Double_t v1 = GetBinContent(i, coord);
      // Now look at the bin with the same coordinates in h
      Double_t v2 = h->GetBinContent(coord);
      if (!v2) {
         v1 = 0.;
         v2 = 1.;
         if (!didWarn) {
            Warning("Divide(h)", "Histogram h has empty bins - division by zero! Setting bin to 0.");
            didWarn = kTRUE;
         }
      }
      SetBinContent(coord, v1 / v2);
      if (wantErrors) {
         Double_t err1 = GetBinError(coord) * v2;
         Double_t err2 = h->GetBinError(coord) * v1;
         b22 = v2 * v2;
         err = (err1 * err1 + err2 * err2) / (b22 * b22);
         SetBinError(coord, TMath::Sqrt(err));
      }
   }
   delete [] coord;
   SetEntries(nEntries);
}

//______________________________________________________________________________
void THnSparse::Divide(const THnSparse *h1, const THnSparse *h2, Double_t c1, Double_t c2, Option_t *option)
{
   // Replace contents of this histogram by multiplication of h1 by h2
   // this = (c1*h1)/(c2*h2)
   // Note that if h1 or h2 have Sumw2 set, Sumw2 is automatically called for
   // this if not already set.
   // The resulting errors are calculated assuming uncorrelated content.
   // However, if option ="B" is specified, Binomial errors are computed.
   // In this case c1 and c2 do not make real sense and they are ignored.


   TString opt = option;
   opt.ToLower();
   Bool_t binomial = kFALSE;
   if (opt.Contains("b")) binomial = kTRUE;

   // Check consistency of the input
   if (!CheckConsistency(h1, "Divide") || !CheckConsistency(h2, "Divide"))return;
   if (!c2) {
      Error("Divide","Coefficient of dividing histogram cannot be zero");
      return;
   }

   Reset();

   // Trigger error calculation if h1 or h2 have it
   if (!GetCalculateErrors() && (h1->GetCalculateErrors()|| h2->GetCalculateErrors() != 0))
      Sumw2();

   // Count filled bins
   Long64_t nFilledBins=0;

   // Now divide the contents: we have the intersection of the sets of bins

   Int_t* coord = new Int_t[fNdimensions];
   memset(coord, 0, sizeof(Int_t) * fNdimensions);
   Float_t w = 0.;
   Float_t err = 0.;
   Float_t b22 = 0.;
   Bool_t didWarn = kFALSE;

   for (Long64_t i = 0; i < h1->GetNbins(); ++i) {
      // Get the content of the bin from the first histogram
      Double_t v1 = h1->GetBinContent(i, coord);
      // Now look at the bin with the same coordinates in h2
      Double_t v2 = h2->GetBinContent(coord);
      if (!v2) {
         v1 = 0.;
         v2 = 1.;
         if (!didWarn) {
            Warning("Divide(h1, h2)", "Histogram h2 has empty bins - division by zero! Setting bin to 0.");
            didWarn = kTRUE;
         }
      }
      nFilledBins++;
      SetBinContent(coord, c1 * v1 / c2 / v2);
      if(GetCalculateErrors()){
         Double_t err1=h1->GetBinError(coord);
         Double_t err2=h2->GetBinError(coord);
         if (binomial) {
            if (v1 != v2) {
               w = v1 / v2;
               err2 *= w;
               err = TMath::Abs( ( (1. - 2.*w) * err1 * err1 + err2 * err2 ) / (v2 * v2) );
            } else {
               err = 0;
            }
         } else {
            c1 *= c1;
            c2 *= c2;
            b22 = v2 * v2 * c2;
            err1 *= v2;
            err2 *= v1;
            err = c1 * c2 * (err1 * err1 + err2 * err2) / (b22 * b22);
         }
         SetBinError(coord,TMath::Sqrt(err));
      }
   }

   delete [] coord;
   fFilledBins = nFilledBins;

   // Set as entries in the result histogram the entries in the numerator
   SetEntries(h1->GetEntries());
}

//______________________________________________________________________________
Bool_t THnSparse::CheckConsistency(const THnSparse *h, const char *tag) const
{
   // Consistency check on (some of) the parameters of two histograms (for operations).

   if (fNdimensions!=h->GetNdimensions()) {
      Warning(tag,"Different number of dimensions, cannot carry out operation on the histograms");
      return kFALSE;
   }
   for (Int_t dim = 0; dim < fNdimensions; dim++){
      if (GetAxis(dim)->GetNbins()!=h->GetAxis(dim)->GetNbins()) {
         Warning(tag,"Different number of bins on axis %i, cannot carry out operation on the histograms", dim);
         return kFALSE;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
void THnSparse::SetBinEdges(Int_t idim, const Double_t* bins)
{
   // Set the axis # of bins and bin limits on dimension idim

   TAxis* axis = (TAxis*) fAxes[idim];
   axis->Set(axis->GetNbins(), bins);
}

//______________________________________________________________________________
void THnSparse::SetBinContent(const Int_t* coord, Double_t v)
{
   // Set content of bin with coordinates "coord" to "v"

   GetCompactCoord()->SetCoord(coord);
   Long_t bin = GetBinIndexForCurrentBin(kTRUE);
   SetBinContent(bin, v);
}

//______________________________________________________________________________
void THnSparse::SetBinContent(Long64_t bin, Double_t v)
{
   // Set content of bin with index "bin" to "v"

   THnSparseArrayChunk* chunk = GetChunk(bin / fChunkSize);
   chunk->fContent->SetAt(v, bin % fChunkSize);
   ++fEntries;
}

//______________________________________________________________________________
void THnSparse::SetBinError(const Int_t* coord, Double_t e)
{
   // Set error of bin with coordinates "coord" to "e", enable errors if needed

   GetCompactCoord()->SetCoord(coord);
   Long_t bin = GetBinIndexForCurrentBin(kTRUE);
   SetBinError(bin, e);
}

//______________________________________________________________________________
void THnSparse::SetBinError(Long64_t bin, Double_t e)
{
   // Set error of bin with index "bin" to "e", enable errors if needed

   THnSparseArrayChunk* chunk = GetChunk(bin / fChunkSize);
   if (!chunk->fSumw2 ) {
      // if fSumw2 is zero GetCalculateErrors should return false
      if (GetCalculateErrors()) {
         Error("SetBinError", "GetCalculateErrors() logic error!");
      }
      Sumw2(); // enable error calculation
   }

   chunk->fSumw2->SetAt(e * e, bin % fChunkSize);
}

//______________________________________________________________________________
void THnSparse::Sumw2()
{
   // Enable calculation of errors

   if (GetCalculateErrors()) return;

   fTsumw2 = 0.;
   TIter iChunk(&fBinContent);
   THnSparseArrayChunk* chunk = 0;
   while ((chunk = (THnSparseArrayChunk*) iChunk()))
      chunk->Sumw2();
}

//______________________________________________________________________________
THnSparse* THnSparse::Rebin(Int_t group) const
{
   // Combine the content of "group" neighboring bins into
   // a new bin and return the resulting THnSparse.
   // For group=2 and a 3 dimensional histogram, all "blocks"
   // of 2*2*2 bins will be put into a bin.

   Int_t* ngroup = new Int_t[GetNdimensions()];
   for (Int_t d = 0; d < GetNdimensions(); ++d)
      ngroup[d] = group;
   THnSparse* ret = Rebin(ngroup);
   delete [] ngroup;
   return ret;
}

//______________________________________________________________________________
void THnSparse::SetTitle(const char *title)
{
   // Change (i.e. set) the title.
   //
   // If title is in the form "stringt;string0;string1;string2 ..."
   // the histogram title is set to stringt, the title of axis0 to string0,
   // of axis1 to string1, of axis2 to string2, etc, just like it is done
   // for TH1/TH2/TH3.
   // To insert the character ";" in one of the titles, one should use "#;"
   // or "#semicolon".

  fTitle = title;
  fTitle.ReplaceAll("#;",2,"#semicolon",10);
  
  Int_t endHistTitle = fTitle.First(';');
  if (endHistTitle >= 0) {
     // title contains a ';' so parse the axis titles
     Int_t posTitle = endHistTitle + 1;
     Int_t lenTitle = fTitle.Length();
     Int_t dim = 0;
     while (posTitle > 0 && posTitle < lenTitle && dim < fNdimensions){
        Int_t endTitle = fTitle.Index(";", posTitle);
        TString axisTitle = fTitle(posTitle, endTitle - posTitle);
        axisTitle.ReplaceAll("#semicolon", 10, ";", 1);
        GetAxis(dim)->SetTitle(axisTitle);
        dim++;
        if (endTitle > 0)
           posTitle = endTitle + 1;
        else
           posTitle = -1;
     }
     // Remove axis titles from histogram title
     fTitle.Remove(endHistTitle, lenTitle - endHistTitle);
  }
  
  fTitle.ReplaceAll("#semicolon", 10, ";", 1);

}
//______________________________________________________________________________
THnSparse* THnSparse::Rebin(const Int_t* group) const
{
   // Combine the content of "group" neighboring bins for each dimension
   // into a new bin and return the resulting THnSparse.
   // For group={2,1,1} and a 3 dimensional histogram, pairs of x-bins
   // will be grouped.

   Int_t ndim = GetNdimensions();
   TString name(GetName());
   for (Int_t d = 0; d < ndim; ++d)
      name += Form("_%d", group[d]);


   TString title(GetTitle());
   Ssiz_t posInsert = title.First(';');
   if (posInsert == kNPOS) {
      title += " rebin ";
      for (Int_t d = 0; d < ndim; ++d)
         title += Form("{%d}", group[d]);
   } else {
      for (Int_t d = ndim - 1; d >= 0; --d)
         title.Insert(posInsert, Form("{%d}", group[d]));
      title.Insert(posInsert, " rebin ");
   }

   TObjArray newaxes(ndim);
   newaxes.SetOwner();
   for (Int_t d = 0; d < ndim; ++d) {
      newaxes.AddAt(GetAxis(d)->Clone(),d);
      if (group[d] > 1) {
         TAxis* newaxis = (TAxis*) newaxes.At(d);
         Int_t newbins = (newaxis->GetNbins() + group[d] - 1) / group[d];
         if (newaxis->GetXbins() && newaxis->GetXbins()->GetSize()) {
            // variable bins
            Double_t *edges = new Double_t[newbins + 1];
            for (Int_t i = 0; i < newbins + 1; ++i)
               if (group[d] * i <= newaxis->GetNbins())
                  edges[i] = newaxis->GetXbins()->At(group[d] * i);
               else edges[i] = newaxis->GetXmax();
            newaxis->Set(newbins, edges);
            delete [] edges;
         } else {
            newaxis->Set(newbins, newaxis->GetXmin(), newaxis->GetXmax());
         }
      }
   }

   THnSparse* h = CloneEmpty(name.Data(), title.Data(), &newaxes, fChunkSize, kTRUE);
   Bool_t haveErrors = GetCalculateErrors();
   Bool_t wantErrors = haveErrors;

   Int_t* bins  = new Int_t[ndim];
   Int_t* coord = new Int_t[fNdimensions];
   memset(coord, 0, sizeof(Int_t) * fNdimensions);
   Double_t err = 0.;
   Double_t preverr = 0.;
   Double_t v = 0.;

   for (Long64_t i = 0; i < GetNbins(); ++i) {
      v = GetBinContent(i, coord);
      for (Int_t d = 0; d < ndim; ++d)
         bins[d] = TMath::CeilNint( (double) coord[d]/group[d] );

      if (wantErrors) {
         if (haveErrors) {
            err = GetBinError(i);
            err *= err;
         } else err = v;
         preverr = h->GetBinError(bins);
         h->SetBinError(bins, TMath::Sqrt(preverr * preverr + err));
      }

      // only _after_ error calculation, or sqrt(v) is taken into account!
      h->AddBinContent(bins, v);
   }

   delete [] bins;
   delete [] coord;

   h->SetEntries(fEntries);

   return h;

}

//______________________________________________________________________________
void THnSparse::Reset(Option_t * /*option = ""*/)
{
   // Clear the histogram
   fFilledBins = 0;
   fEntries = 0.;
   fTsumw = 0.;
   fTsumw2 = -1.;
   fBins.Delete();
   fBinsContinued.Clear();
   fBinContent.Delete();
   if (fIntegralStatus != kNoInt) {
      delete [] fIntegral;
      fIntegralStatus = kNoInt;
   }
}

//______________________________________________________________________________
Double_t THnSparse::ComputeIntegral()
{
   // Calculate the integral of the histogram

   // delete old integral
   if (fIntegralStatus != kNoInt) {
      delete [] fIntegral;
      fIntegralStatus = kNoInt;
   }

   // check number of bins
   if (GetNbins() == 0) {
      Error("ComputeIntegral", "The histogram must have at least one bin.");
      return 0.;
   }

   // allocate integral array
   fIntegral = new Double_t [GetNbins() + 1];
   fIntegral[0] = 0.;

   // fill integral array with contents of regular bins (non over/underflow)
   Int_t* coord = new Int_t[fNdimensions];
   for (Long64_t i = 0; i < GetNbins(); ++i) {
      Double_t v = GetBinContent(i, coord);

      // check whether the bin is regular
      bool regularBin = true;
      for (Int_t dim = 0; dim < fNdimensions; dim++)
         if (coord[dim] < 1 || coord[dim] > GetAxis(dim)->GetNbins()) {
            regularBin = false;
            break;
         }

      // if outlayer, count it with zero weight
      if (!regularBin) v = 0.;

      fIntegral[i + 1] = fIntegral[i] + v;
   }
   delete [] coord;

   // check sum of weights
   if (fIntegral[GetNbins()] == 0.) {
      Error("ComputeIntegral", "No hits in regular bins (non over/underflow).");
      delete [] fIntegral;
      return 0.;
   }

   // normalize the integral array
   for (Long64_t i = 0; i <= GetNbins(); ++i)
      fIntegral[i] = fIntegral[i] / fIntegral[GetNbins()];

   // set status to valid
   fIntegralStatus = kValidInt;
   return fIntegral[GetNbins()];
}

//______________________________________________________________________________
void THnSparse::PrintBin(Long64_t idx, Option_t* options) const
{
   // Print bin with linex index "idx".
   // For valid options see PrintBin(Long64_t idx, Int_t* bin, Option_t* options).
   Int_t* coord = new Int_t[fNdimensions];
   PrintBin(idx, coord, options);
   delete [] coord;
}

//______________________________________________________________________________
Bool_t THnSparse::PrintBin(Long64_t idx, Int_t* bin, Option_t* options) const
{
   // Print one bin. If "idx" is != -1 use that to determine the bin,
   // otherwise (if "idx" == -1) use the coordinate in "bin".
   // If "options" contains:
   //   '0': only print bins with an error or content != 0
   // Return whether the bin was printed (depends on options)
   
   Double_t v = -42;
   if (idx == -1) {
      idx = const_cast<THnSparse*>(this)->GetBin(bin, kFALSE);
      v = GetBinContent(idx);
   } else {
      v = GetBinContent(idx, bin);
   }

   if (options && strchr(options, '0') && v == 0.) {
      if (GetCalculateErrors()) {
         if (GetBinError(idx) == 0.) {
            // suppress zeros, and we have one.
            return kFALSE;
         }
      } else {
         // suppress zeros, and we have one.
         return kFALSE;
      }
   }

   TString coord;
   for (Int_t dim = 0; dim < fNdimensions; ++dim) {
      coord += bin[dim];
      coord += ',';
   }
   coord.Remove(coord.Length() - 1);

   if (GetCalculateErrors()) {
      Double_t err = 0.;
      if (idx != -1) {
         err = GetBinError(idx);
      }
      Printf("Bin at (%s) = %g (+/- %g)", coord.Data(), v, err);
   } else {
      Printf("Bin at (%s) = %g", coord.Data(), v);
   }

   return kTRUE;
}

//______________________________________________________________________________
void THnSparse::PrintEntries(Long64_t from /*=0*/, Long64_t howmany /*=-1*/,
                            Option_t* options /*=0*/) const
{
   // Print "howmany" entries starting at "from". If "howmany" is -1, print all.
   // If "options" contains:
   //   'x': print in the order of axis bins, i.e. (0,0,...,0), (0,0,...,1),...
   //   '0': only print bins with content != 0

   if (from < 0) from = 0;
   if (howmany == -1) howmany = GetNbins();

   Int_t* bin = new Int_t[fNdimensions];

   if (options && (strchr(options, 'x') || strchr(options, 'X'))) {
      Int_t* nbins = new Int_t[fNdimensions];
      for (Int_t dim = fNdimensions - 1; dim >= 0; --dim) {
         nbins[dim] = GetAxis(dim)->GetNbins();
         bin[dim] = from % nbins[dim];
         from /= nbins[dim];
      }

      for (Long64_t i = 0; i < howmany; ++i) {
         if (!PrintBin(-1, bin, options))
            ++howmany;
         // Advance to next bin:
         ++bin[fNdimensions - 1];
         for (Int_t dim = fNdimensions - 1; dim >= 0; --dim) {
            if (bin[dim] >= nbins[dim]) {
               bin[dim] = 0;
               if (dim > 0) {
                  ++bin[dim - 1];
               } else {
                  howmany = -1; // aka "global break"
               }
            }
         }
      }
      delete [] nbins;
   } else {
      for (Long64_t i = from; i < from + howmany; ++i) {
         if (!PrintBin(i, bin, options))
            ++howmany;
      }
   }
   delete [] bin;
}

//______________________________________________________________________________
void THnSparse::Print(Option_t* options) const
{
   // Print a THnSparse. If "option" contains:
   //   'a': print axis details
   //   'm': print memory usage
   //   's': print statistics
   //   'c': print its content, too (this can generate a LOT of output!)
   // Other options are forwarded to PrintEntries().

   Bool_t optAxis    = options && (strchr(options, 'A') || (strchr(options, 'a')));
   Bool_t optMem     = options && (strchr(options, 'M') || (strchr(options, 'm')));
   Bool_t optStat    = options && (strchr(options, 'S') || (strchr(options, 's')));
   Bool_t optContent = options && (strchr(options, 'C') || (strchr(options, 'c')));

   Printf("%s (*0x%lx): \"%s\" \"%s\"", IsA()->GetName(), (unsigned long)this, GetName(), GetTitle());
   Printf("  %d dimensions, %g entries in %lld filled bins", GetNdimensions(), GetEntries(), GetNbins());

   if (optAxis) {
      for (Int_t dim = 0; dim < fNdimensions; ++dim) {
         TAxis* axis = GetAxis(dim);
         Printf("    axis %d \"%s\": %d bins (%g..%g), %s bin sizes", dim,
                axis->GetTitle(), axis->GetNbins(), axis->GetXmin(), axis->GetXmax(),
                (axis->GetXbins() ? "variable" : "fixed"));
      }
   }

   if (optStat) {
      Printf("  %s error calculation", (GetCalculateErrors() ? "with" : "without"));
      if (GetCalculateErrors()) {
         Printf("    Sum(w)=%g, Sum(w^2)=%g", GetSumw(), GetSumw2());
         for (Int_t dim = 0; dim < fNdimensions; ++dim) {
            Printf("    axis %d: Sum(w*x)=%g, Sum(w*x^2)=%g", dim, GetSumwx(dim), GetSumwx2(dim));
         }
      }
   }

   if (optMem) {
      Printf("  coordinates stored in %d chunks of %d entries\n    %g of bins filled using %g of memory compared to an array",
             GetNChunks(), GetChunkSize(), GetSparseFractionBins(), GetSparseFractionMem());
   }

   if (optContent) {
      Printf("  BIN CONTENT:");
      PrintEntries(0, -1, options);
   }
}


//______________________________________________________________________________
void THnSparse::Browse(TBrowser *b)
{
   // Browse a THnSparse: create an entry (ROOT::THnSparseBrowsable) for each
   // dimension.
   if (fBrowsables.IsEmpty()) {
      for (Int_t dim = 0; dim < fNdimensions; ++dim) {
         fBrowsables[dim] = new ROOT::THnSparseBrowsable(this, dim);
      }
      fBrowsables.SetOwner();
   }

   for (Int_t dim = 0; dim < fNdimensions; ++dim) {
      b->Add(fBrowsables[dim]);
   }
}


//______________________________________________________________________________
//______________________________________________________________________________
ClassImp(ROOT::THnSparseBrowsable);

//______________________________________________________________________________
ROOT::THnSparseBrowsable::THnSparseBrowsable(THnSparse* hist, Int_t axis):
   TNamed(TString::Format("axis %d", axis),
          TString::Format("Projection on axis %d of sparse histogram", axis)),
   fHist(hist), fAxis(axis), fProj(0)
{
   // Construct a THnSparseBrowsable.
}

//______________________________________________________________________________
ROOT::THnSparseBrowsable::~THnSparseBrowsable()
{
   // Destruct a THnSparseBrowsable.
   delete fProj;
}

//______________________________________________________________________________
void ROOT::THnSparseBrowsable::Browse(TBrowser* b)
{
   // Browse an axis of a THnSparse, i.e. draw its projection.
   if (!fProj) {
      fProj = fHist->Projection(fAxis);
   }
   fProj->Draw(b ? b->GetDrawOption() : "");
   gPad->Update();
}

