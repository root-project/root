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
#include "TClass.h"
#include "TCollection.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TH3D.h"
#include "TInterpreter.h"
#include "TMath.h"
#include "TRandom.h"
#include <cassert>

//______________________________________________________________________________
//
// THnSparseCompactBinCoord is a class used by THnSparse internally. It
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

class THnSparseCompactBinCoord {
public:
   THnSparseCompactBinCoord(Int_t dim, const Int_t* nbins);
   ~THnSparseCompactBinCoord();
   void SetCoord(const Int_t* coord) { memcpy(fCurrentBin, coord, sizeof(Int_t) * fNdimensions); }
   ULong64_t GetHash();
   Int_t GetSize() const { return fCoordBufferSize; }
   Int_t* GetCoord() const { return fCurrentBin; }
   Char_t* GetBuffer() const { return fCoordBuffer; }
   void GetCoordFromBuffer(Int_t* coord) const;

protected:
   Int_t GetNumBits(Int_t n) const {
      // return the number of bits allocated by the number "n"
      Int_t r = (n > 0);
      while (n/=2) ++r;
      return r;
   }
private:
   Int_t  fNdimensions;     // number of dimensions
   Int_t *fBitOffsets;      //[fNdimensions + 1] bit offset of each axis index
   Char_t *fCoordBuffer;     // compact buffer of coordinates
   Int_t  fCoordBufferSize; // size of fBinCoordBuffer
   Int_t *fCurrentBin;      // current coordinates
};


//______________________________________________________________________________
//______________________________________________________________________________


//______________________________________________________________________________
THnSparseCompactBinCoord::THnSparseCompactBinCoord(Int_t dim, const Int_t* nbins):
   fNdimensions(dim), fBitOffsets(0), fCoordBuffer(0), fCoordBufferSize(0)
{
   // Initialize a THnSparseCompactBinCoord object with "dim" dimensions
   // and "bins" holding the number of bins for each dimension.

   fCurrentBin = new Int_t[dim];
   fBitOffsets = new Int_t[dim + 1];

   int shift = 0;
   for (Int_t i = 0; i < dim; ++i) {
      fBitOffsets[i] = shift;
      shift += GetNumBits(nbins[i] + 2);
   }
   fBitOffsets[dim] = shift;
   fCoordBufferSize = (shift + 7) / 8;
   fCoordBuffer = new Char_t[fCoordBufferSize];
}


//______________________________________________________________________________
THnSparseCompactBinCoord::~THnSparseCompactBinCoord()
{
   // destruct a THnSparseCompactBinCoord

   delete [] fBitOffsets;
   delete [] fCoordBuffer;
   delete [] fCurrentBin;
}

//______________________________________________________________________________
void THnSparseCompactBinCoord::GetCoordFromBuffer(Int_t* coord) const
{
   // Given the current fCoordBuffer, calculate ("decompact") the bin coordinates,
   // and return it in coord.

   for (Int_t i = 0; i < fNdimensions; ++i) {
      const Int_t offset = fBitOffsets[i] / 8;
      Int_t shift = fBitOffsets[i] % 8;
      Int_t nbits = fBitOffsets[i + 1] - fBitOffsets[i];
      UChar_t* pbuf = (UChar_t*) fCoordBuffer + offset;
      coord[i] = *pbuf >> shift;
      Int_t subst = (Int_t) -1;
      subst = subst << nbits;
      nbits -= (8 - shift);
      shift = 8 - shift;
      for (Int_t n = 0; n * 8 < nbits; ++n) {
         ++pbuf;
         coord[i] += *pbuf << shift;
         shift += 8;
      }
      coord[i] &= ~subst;
   }
}

//______________________________________________________________________________
ULong64_t THnSparseCompactBinCoord::GetHash()
{
   // Calculate hash for compact bin index of the current bin.

   memset(fCoordBuffer, 0, fCoordBufferSize);
   for (Int_t i = 0; i < fNdimensions; ++i) {
      const Int_t offset = fBitOffsets[i] / 8;
      const Int_t shift = fBitOffsets[i] % 8;
      ULong64_t val = fCurrentBin[i];

      Char_t* pbuf = fCoordBuffer + offset;
      *pbuf += 0xff & (val << shift);
      val = val >> (8 - shift);
      while (val) {
         ++pbuf;
         *pbuf += 0xff & val;
         val = val >> 8;
      }
   }

   // Bins are addressed in two different modes, depending
   // on whether the compact bin index fits into a Long_t or not.
   // If it does, we can use it as a "perfect hash" for the TExMap.
   // If not we build a hash from the compact bin index, and use that
   // as the TExMap's hash.
   // For the non-hash mode, the size of the compact bin index must be
   // smaller than Long_t on all supported platforms - not just the current
   // one, because we make this layout persistent, too. So just test for
   // its size <= 4.

   switch (fCoordBufferSize) {
   case 1: return fCoordBuffer[0];
   case 2: return fCoordBuffer[0] + (fCoordBuffer[1] <<  8l);
   case 3: return fCoordBuffer[0] + (fCoordBuffer[1] <<  8l)
              + (fCoordBuffer[2] << 16l);
   case 4: return fCoordBuffer[0] + (fCoordBuffer[1] <<  8l)
              + (fCoordBuffer[2] << 16l) + (fCoordBuffer[3] << 24l);
   }
   return TMath::Hash(fCoordBuffer, fCoordBufferSize);
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
void THnSparseArrayChunk::AddBin(ULong_t idx, const Char_t* coordbuf)
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
// if it takes less than 4 bytes, the minimal supported size of a Long_t).
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
      TString aname("axis");
      aname += i;
      axis->SetName(aname);
      axis->SetTitle(aname);
      fAxes.AddAtAndExpand(axis, i);
   }
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
   Long_t bin = GetBinIndexForCurrentBin(kTRUE);
   THnSparseArrayChunk* chunk = GetChunk(bin / fChunkSize);
   bin %= fChunkSize;
   v += chunk->fContent->GetAt(bin);
   return chunk->fContent->SetAt(v, bin);
}

//______________________________________________________________________________
THnSparseArrayChunk* THnSparse::AddChunk()
{
   //Create a new chunk of bin content
   THnSparseArrayChunk* first = 0;
   if (fBinContent.GetEntriesFast() > 0)
      first = GetChunk(0);
   THnSparseArrayChunk* chunk =
      new THnSparseArrayChunk(GetCompactCoord()->GetSize(),
                              GetCalculateErrors(), GenerateArray());
   fBinContent.AddLast(chunk);
   return chunk;
}

//______________________________________________________________________________
THnSparse* THnSparse::CloneEmpty(const char* name, const char* title,
                                 const TObjArray* axes, Int_t chunksize) const
{
   // Create a new THnSparse object that is of the same type as *this,
   // but with dimensions and bins given by axes.

   THnSparse* ret = (THnSparse*)IsA()->New();
   ret->SetNameTitle(name, title);
   ret->fNdimensions = axes->GetEntriesFast();
   ret->fChunkSize = chunksize;

   TIter iAxis(axes);
   const TAxis* axis = 0;
   Int_t pos = 0;
   Int_t *nbins = new Int_t[axes->GetEntriesFast()];
   while ((axis = (TAxis*)iAxis())) {
      nbins[pos] = axis->GetNbins();
      ret->fAxes.AddAtAndExpand(axis->Clone(), pos++);
   }
   ret->fAxes.SetOwner();

   ret->fCompactCoord = new THnSparseCompactBinCoord(pos, nbins);
   delete [] nbins;

   return ret;
}

//______________________________________________________________________________
TH1* THnSparse::CreateHist(const char* name, const char* title,
                           const TObjArray* axes) const {
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
      if (reqaxis->TestBit(TAxis::kAxisRange)) {
         Int_t binFirst = reqaxis->GetFirst();
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
Long_t THnSparse::GetBin(const Double_t* x, Bool_t allocate /* = kTRUE */)
{
   // Get the bin index for the n dimensional tuple x,
   // allocate one if it doesn't exist yet and "allocate" is true.

   Int_t *coord = GetCompactCoord()->GetCoord();
   for (Int_t i = 0; i < fNdimensions; ++i)
      coord[i] = GetAxis(i)->FindBin(x[i]);

   return GetBinIndexForCurrentBin(allocate);
}


//______________________________________________________________________________
Long_t THnSparse::GetBin(const char* name[], Bool_t allocate /* = kTRUE */)
{
   // Get the bin index for the n dimensional tuple addressed by "name",
   // allocate one if it doesn't exist yet and "allocate" is true.

   Int_t *coord = GetCompactCoord()->GetCoord();
   for (Int_t i = 0; i < fNdimensions; ++i)
      coord[i] = GetAxis(i)->FindBin(name[i]);

   return GetBinIndexForCurrentBin(allocate);
}

//______________________________________________________________________________
Long_t THnSparse::GetBin(const Int_t* coord, Bool_t allocate /*= kTRUE*/)
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
   Long_t idx = const_cast<THnSparse*>(this)->GetBinIndexForCurrentBin(kFALSE);
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
            Int_t sizeCompact = GetCompactCoord()->GetSize();
            memcpy(GetCompactCoord()->GetBuffer(), chunk->fCoordinates + idx * sizeCompact, sizeCompact);
            GetCompactCoord()->GetCoordFromBuffer(coord);
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
   Long_t idx = const_cast<THnSparse*>(this)->GetBinIndexForCurrentBin(kFALSE);
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
Long_t THnSparse::GetBinIndexForCurrentBin(Bool_t allocate)
{
   // Return the index for fCurrentBinIndex.
   // If it doesn't exist then return -1, or allocate a new bin if allocate is set

   Long_t hash = GetCompactCoord()->GetHash();
   Long_t linidx = (Long_t) fBins.GetValue(hash);
   while (linidx) {
      // fBins stores index + 1!
      THnSparseArrayChunk* chunk = GetChunk((linidx - 1)/ fChunkSize);
      if (chunk->Matches((linidx - 1) % fChunkSize, GetCompactCoord()->GetBuffer()))
         return linidx - 1; // we store idx+1, 0 is "TExMap: not found"

      Long_t nextlinidx = fBinsContinued.GetValue(linidx);
      if (!nextlinidx) break;

      linidx = nextlinidx;
   }
   if (!allocate) return -1;

   ++fFilledBins;

   // allocate bin in chunk
   THnSparseArrayChunk *chunk = (THnSparseArrayChunk*) fBinContent.Last();
   Long_t newidx = chunk ? ((Long_t) chunk->GetEntries()) : -1;
   if (!chunk || newidx == (Long_t)fChunkSize) {
      chunk = AddChunk();
      newidx = 0;
   }
   chunk->AddBin(newidx, GetCompactCoord()->GetBuffer());

   // store translation between hash and bin
   newidx += (fBinContent.GetEntriesFast() - 1) * fChunkSize;
   if (!linidx)
      // fBins didn't find it
      fBins.Add(hash, newidx + 1);
   else
      // fBins contains one, but it's the wrong one;
      // add entry to fBinsContinued.
      fBinsContinued.Add(linidx, newidx + 1);
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
   Int_t bin[20]; //FIXME in case a user requests more than 20 dimensions ::)
   GetBinContent(idx, bin);

   // convert bin coordinates to real values
   for (Int_t i = 0; i < fNdimensions; i++) {
      rand[i] = GetAxis(i)->GetBinCenter(bin[i]);

      // randomize the vector withing a bin
      if (subBinRandom)
         rand[i] += (gRandom->Rndm() - 0.5) * GetAxis(i)->GetBinWidth(bin[i]);
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
   Double_t size = 0.;
   if (fFilledBins) {
      TClass* clArray = GetChunk(0)->fContent->IsA();
      TDataMember* dm = clArray ? clArray->GetDataMember("fArray") : 0;
      arrayElementSize = dm ? dm->GetDataType()->Size() : 0;
   }
   if (!arrayElementSize) {
      Warning("GetSparseFractionMem", "Cannot determine type of elements!");
      return -1.;
   }

   size += fFilledBins * (GetCompactCoord()->GetSize() + arrayElementSize + 2 * sizeof(Long_t) /* TExMap */);
   if (fFilledBins && GetChunk(0)->fSumw2)
      size += fFilledBins * sizeof(Float_t); /* fSumw2 */

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

   return (TH1D*) ProjectionAny(1, &xDim, false, option);
}

//______________________________________________________________________________
TH2D* THnSparse::Projection(Int_t xDim, Int_t yDim, Option_t* option /*= ""*/) const
{
   // Project all bins into a 2-dimensional histogram,
   // keeping only axes "xDim" and "yDim".
   // If "option" contains "E" errors will be calculated.

   // y, x looks wrong, but it's what TH3::Project3D("xy") does
   const Int_t dim[2] = {yDim, xDim};
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

   const Int_t dim[3] = {xDim, yDim, zDim};
   return (TH3D*) ProjectionAny(3, dim, false, option);
}

//______________________________________________________________________________
THnSparse* THnSparse::Projection(Int_t ndim, const Int_t* dim,
                                 Option_t* option /*= ""*/) const
{
   // Project all bins into a ndim-dimensional histogram,
   // keeping only axes "dim".
   // If "option" contains "E" errors will be calculated.
   //                      "A" ranges of the taget axes will be ignored.

   return (THnSparse*) ProjectionAny(ndim, dim, true, option);
}


//______________________________________________________________________________
TObject* THnSparse::ProjectionAny(Int_t ndim, const Int_t* dim,
                                  Bool_t wantSparse, Option_t* option /*= ""*/) const
{
   // Project all bins into a 3-dimensional histogram,
   // keeping only axes "xDim", "yDim", and "zDim".
   // If "option" contains "E" errors will be calculated.
   //                      "A" ranges of the taget axes will be ignored.

   TString name(GetName());
   name += "_";
   for (Int_t d = 0; d < ndim; ++d)
      name += GetAxis(dim[d])->GetName();

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
   if (ignoreTargetRange) {
      hadRange = new Bool_t[ndim];
      for (Int_t d = 0; d < ndim; ++d){
         TAxis *axis = GetAxis(dim[d]);
         hadRange[d] = axis->TestBit(TAxis::kAxisRange);
         axis->SetBit(TAxis::kAxisRange, kFALSE);
      }
   }

   if (wantSparse)
      ret = sparse = CloneEmpty(name, title, &newaxes, fChunkSize);
   else
      ret = hist = CreateHist(name, title, &newaxes); 

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
         if (GetAxis(dim[d])->TestBit(TAxis::kAxisRange)) {
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
      sparse->SetEntries(fEntries);
   else
      hist->SetEntries(fEntries);

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


   Int_t* coord = new Int_t[fNdimensions];
   memset(coord, 0, sizeof(Int_t) * fNdimensions);

   // Scale the contents & errors
   Bool_t haveErrors = GetCalculateErrors();
   for (Long64_t i = 0; i < GetNbins(); ++i) {
      // Get the content of the bin from the current histogram
      Double_t v = GetBinContent(i, coord);
      SetBinContent(coord, c * v);
      if (haveErrors) {
         Double_t err = GetBinError(coord);
         SetBinError(coord, c * err);
      }
   }

   delete [] coord;
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
   RebinnedAdd(h, c);
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

   if (fNdimensions!=h->GetNdimensions()) {
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

   // Add to this whatever is found inside the other histogram
   for (Long64_t i = 0; i < h->GetNbins(); ++i) {
      // Get the content of the bin from the second histogram
      Double_t v = h->GetBinContent(i, coord);
      AddBinContent(coord, c * v);
      if (haveErrors) {
         Double_t err1 = GetBinError(coord);
         Double_t err2 = h->GetBinError(coord) * c;
         SetBinError(coord, TMath::Sqrt(err1 * err1 + err2 * err2));
      }
   }


   delete [] coord;

   Double_t nEntries = GetEntries() + c * h->GetEntries();
   SetEntries(nEntries);
}


//______________________________________________________________________________
Long64_t THnSparse::Merge(TCollection* list)
{
   // Merge this with a list of THnSparses. All THnSparses provided
   // in the list must have the same bin layout!

   if (!list) return 0;
   if (list->IsEmpty()) return (Long64_t)GetEntries();

   TIter iter(list);
   const TObject* addMeObj = 0;
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
   if (!GetCalculateErrors() && h->GetCalculateErrors())
      wantErrors = kTRUE;

   // Create a temporary histogram where to store the result
   TObjArray newaxes(fNdimensions);
   for (Int_t d = 0; d < fNdimensions; ++d) {
      newaxes.AddAt(GetAxis(d),d);
   }

   if (wantErrors) Sumw2();

   // Now multiply the contents: in this case we have the intersection of the sets of bins
   Int_t* coord = new Int_t[fNdimensions];
   memset(coord, 0, sizeof(Int_t) * fNdimensions);
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

   //now deposit the result in the original histogram....
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
   Bool_t wantErrors=kFALSE;
   if (!GetCalculateErrors() && h->GetCalculateErrors())
      wantErrors=kTRUE;

   // Remember original histogram statistics
   Double_t nEntries = fEntries;

   // Create a temporary histogram where to store the result
   TObjArray newaxes(fNdimensions);
   for (Int_t d = 0; d < fNdimensions; ++d) {
      newaxes.AddAt(GetAxis(d),d);
   }

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
   THnSparseArrayChunk* chunk = GetChunk(bin / fChunkSize);
   chunk->fContent->SetAt(v, bin % fChunkSize);
}

//______________________________________________________________________________
void THnSparse::SetBinError(const Int_t* coord, Double_t e)
{
   // Set error of bin with coordinates "coord" to "e", enable errors if needed

   GetCompactCoord()->SetCoord(coord);
   Long_t bin = GetBinIndexForCurrentBin(kTRUE);

   THnSparseArrayChunk* chunk = GetChunk(bin / fChunkSize);
   if (!chunk->fSumw2 ) {
      // if fSumw2 is zero GetCalcualteErrors should return false
      assert(!GetCalculateErrors() );
      Sumw2(); // enable error calculation
   }
   
   chunk->fSumw2->SetAt(e*e, bin % fChunkSize);
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
         } else {
            newaxis->Set(newbins, newaxis->GetXmin(), newaxis->GetXmax());
         }
      }
   }

   THnSparse* h = CloneEmpty(name.Data(), title.Data(), &newaxes, fChunkSize);
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
