// @(#)root/hist:$Name:  $:$Id: TH2.cxx,v 1.100 2006/10/03 10:05:02 brun Exp $
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TClass.h"
#include "TH2.h"
#include "TVirtualPad.h"
#include "TF2.h"
#include "TProfile.h"
#include "TRandom.h"
#include "TMatrixFBase.h"
#include "TMatrixDBase.h"
#include "THLimitsFinder.h"
#include "TError.h"
#include "TObjString.h"


ClassImp(TH2)

//______________________________________________________________________________
//
// Service class for 2-Dim histogram classes
//
//  TH2C a 2-D histogram with one byte per cell (char)
//  TH2S a 2-D histogram with two bytes per cell (short integer)
//  TH2I a 2-D histogram with four bytes per cell (32 bits integer)
//  TH2F a 2-D histogram with four bytes per cell (float)
//  TH2D a 2-D histogram with eight bytes per cell (double)
//

//______________________________________________________________________________
TH2::TH2()
{
   // Constructor.

   fDimension   = 2;
   fScalefactor = 1;
   fTsumwy      = fTsumwy2 = fTsumwxy = 0;
   SetBinsLength(9);
}

//______________________________________________________________________________
TH2::TH2(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                     ,Int_t nbinsy,Double_t ylow,Double_t yup)
     :TH1(name,title,nbinsx,xlow,xup)
{
   // see comments in the TH1 base class constructors

   fDimension   = 2;
   fScalefactor = 1;
   fTsumwy      = fTsumwy2 = fTsumwxy = 0;
   if (nbinsy <= 0) nbinsy = 1;
   fYaxis.Set(nbinsy,ylow,yup);
   fNcells      = fNcells*(nbinsy+2); // fNCells is set in the TH1 constructor
   if (fgDefaultSumw2) Sumw2();
}

//______________________________________________________________________________
TH2::TH2(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                     ,Int_t nbinsy,Double_t ylow,Double_t yup)
     :TH1(name,title,nbinsx,xbins)
{
   // see comments in the TH1 base class constructors
   fDimension   = 2;
   fScalefactor = 1;
   fTsumwy      = fTsumwy2 = fTsumwxy = 0;
   if (nbinsy <= 0) nbinsy = 1;
   fYaxis.Set(nbinsy,ylow,yup);
   fNcells      = fNcells*(nbinsy+2); // fNCells is set in the TH1 constructor
   if (fgDefaultSumw2) Sumw2();
}

//______________________________________________________________________________
TH2::TH2(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                     ,Int_t nbinsy,const Double_t *ybins)
     :TH1(name,title,nbinsx,xlow,xup)
{
   // see comments in the TH1 base class constructors
   fDimension   = 2;
   fScalefactor = 1;
   fTsumwy      = fTsumwy2 = fTsumwxy = 0;
   if (nbinsy <= 0) nbinsy = 1;
   if (ybins) fYaxis.Set(nbinsy,ybins);
   else       fYaxis.Set(nbinsy,0,1);
   fNcells      = fNcells*(nbinsy+2); // fNCells is set in the TH1 constructor
   if (fgDefaultSumw2) Sumw2();
}

//______________________________________________________________________________
TH2::TH2(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                           ,Int_t nbinsy,const Double_t *ybins)
     :TH1(name,title,nbinsx,xbins)
{
   // see comments in the TH1 base class constructors
   fDimension   = 2;
   fScalefactor = 1;
   fTsumwy      = fTsumwy2 = fTsumwxy = 0;
   if (nbinsy <= 0) nbinsy = 1;
   if (ybins) fYaxis.Set(nbinsy,ybins);
   else       fYaxis.Set(nbinsy,0,1);
   fNcells      = fNcells*(nbinsy+2); // fNCells is set in the TH1 constructor
   if (fgDefaultSumw2) Sumw2();
}

//______________________________________________________________________________
TH2::TH2(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                           ,Int_t nbinsy,const Float_t *ybins)
     :TH1(name,title,nbinsx,xbins)
{
   // see comments in the TH1 base class constructors
   fDimension   = 2;
   fScalefactor = 1;
   fTsumwy      = fTsumwy2 = fTsumwxy = 0;
   if (nbinsy <= 0) nbinsy = 1;
   if (ybins) fYaxis.Set(nbinsy,ybins);
   else       fYaxis.Set(nbinsy,0,1);
   fNcells      = fNcells*(nbinsy+2); // fNCells is set in the TH1 constructor.
   if (fgDefaultSumw2) Sumw2();
}

//______________________________________________________________________________
TH2::TH2(const TH2 &h) : TH1()
{
   // Copy constructor.
   // The list of functions is not copied. (Use Clone if needed)
   Copy((TObject&)h);
}

//______________________________________________________________________________
TH2::~TH2()
{
   // Destructor.
}

//______________________________________________________________________________
Int_t TH2::BufferEmpty(Int_t action)
{
// Fill histogram with all entries in the buffer.
// action = -1 histogram is reset and refilled from the buffer (called by THistPainter::Paint)
// action =  0 histogram is filled from the buffer
// action =  1 histogram is filled and buffer is deleted
//             The buffer is automatically deleted when the number of entries
//             in the buffer is greater than the number of entries in the histogram

   // do we need to compute the bin size?
   if (!fBuffer) return 0;
   Int_t nbentries = (Int_t)fBuffer[0];
   if (!nbentries) return 0;
   Double_t *buffer = fBuffer;
   if (nbentries < 0) {
      if (action == 0) return 0;
      nbentries  = -nbentries;
      fBuffer=0;
      Reset();
      fBuffer = buffer;
   }
   if (TestBit(kCanRebin) || fXaxis.GetXmax() <= fXaxis.GetXmin() || fYaxis.GetXmax() <= fYaxis.GetXmin()) {
      //find min, max of entries in buffer
      Double_t xmin = fBuffer[2];
      Double_t xmax = xmin;
      Double_t ymin = fBuffer[3];
      Double_t ymax = ymin;
      for (Int_t i=1;i<nbentries;i++) {
         Double_t x = fBuffer[3*i+2];
         if (x < xmin) xmin = x;
         if (x > xmax) xmax = x;
         Double_t y = fBuffer[3*i+3];
         if (y < ymin) ymin = y;
         if (y > ymax) ymax = y;
      }
      if (fXaxis.GetXmax() <= fXaxis.GetXmin() || fYaxis.GetXmax() <= fYaxis.GetXmin()) {
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(this,xmin,xmax,ymin,ymax);
      } else {
         fBuffer = 0;
         Int_t keep = fBufferSize; fBufferSize = 0;
         if (xmin <  fXaxis.GetXmin()) RebinAxis(xmin,"X");
         if (xmax >= fXaxis.GetXmax()) RebinAxis(xmax,"X");
         if (ymin <  fYaxis.GetXmin()) RebinAxis(ymin,"Y");
         if (ymax >= fYaxis.GetXmax()) RebinAxis(ymax,"Y");
         fBuffer = buffer;
         fBufferSize = keep;
      }
   }

   fBuffer = 0;
   for (Int_t i=0;i<nbentries;i++) {
      Fill(buffer[3*i+2],buffer[3*i+3],buffer[3*i+1]);
   }
   fBuffer = buffer;

   if (action > 0) { delete [] fBuffer; fBuffer = 0; fBufferSize = 0;}
   else {
      if (nbentries == (Int_t)fEntries) fBuffer[0] = -nbentries;
      else                              fBuffer[0] = 0;
   }
   return nbentries;
}

//______________________________________________________________________________
Int_t TH2::BufferFill(Double_t x, Double_t y, Double_t w)
{
// accumulate arguments in buffer. When buffer is full, empty the buffer
// fBuffer[0] = number of entries in buffer
// fBuffer[1] = w of first entry
// fBuffer[2] = x of first entry
// fBuffer[3] = y of first entry

   if (!fBuffer) return -3;
   Int_t nbentries = (Int_t)fBuffer[0];
   if (nbentries < 0) {
      nbentries  = -nbentries;
      fBuffer[0] =  nbentries;
      if (fEntries > 0) {
         Double_t *buffer = fBuffer; fBuffer=0;
         Reset();
         fBuffer = buffer;
      }
   }
   if (3*nbentries+3 >= fBufferSize) {
      BufferEmpty(1);
      return Fill(x,y,w);
   }
   fBuffer[3*nbentries+1] = w;
   fBuffer[3*nbentries+2] = x;
   fBuffer[3*nbentries+3] = y;
   fBuffer[0] += 1;
   return -3;
}

//______________________________________________________________________________
void TH2::Copy(TObject &obj) const
{
   // Copy.

   TH1::Copy(obj);
   ((TH2&)obj).fScalefactor = fScalefactor;
   ((TH2&)obj).fTsumwy      = fTsumwy;
   ((TH2&)obj).fTsumwy2     = fTsumwy2;
   ((TH2&)obj).fTsumwxy     = fTsumwxy;
}

//______________________________________________________________________________
Int_t TH2::Fill(Double_t x,Double_t y)
{
   //*-*-*-*-*-*-*-*-*-*-*Increment cell defined by x,y by 1*-*-*-*-*-*-*-*-*-*
   //*-*                  ==================================
   //*-*
   //*-* if x or/and y is less than the low-edge of the corresponding axis first bin,
   //*-*   the Underflow cell is incremented.
   //*-* if x or/and y is greater than the upper edge of corresponding axis last bin,
   //*-*   the Overflow cell is incremented.
   //*-*
   //*-* If the storage of the sum of squares of weights has been triggered,
   //*-* via the function Sumw2, then the sum of the squares of weights is incremented
   //*-* by 1in the cell corresponding to x,y.
   //*-*
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (fBuffer) return BufferFill(x,y,1);

   Int_t binx, biny, bin;
   fEntries++;
   binx = fXaxis.FindBin(x);
   biny = fYaxis.FindBin(y);
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin);
   if (fSumw2.fN) ++fSumw2.fArray[bin];
   if (binx == 0 || binx > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   if (biny == 0 || biny > fYaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   ++fTsumw;
   ++fTsumw2;
   fTsumwx  += x;
   fTsumwx2 += x*x;
   fTsumwy  += y;
   fTsumwy2 += y*y;
   fTsumwxy += x*y;
   return bin;
}

//______________________________________________________________________________
Int_t TH2::Fill(Double_t x, Double_t y, Double_t w)
{
   //*-*-*-*-*-*-*-*-*-*-*Increment cell defined by x,y by a weight w*-*-*-*-*-*
   //*-*                  ===========================================
   //*-*
   //*-* if x or/and y is less than the low-edge of the corresponding axis first bin,
   //*-*   the Underflow cell is incremented.
   //*-* if x or/and y is greater than the upper edge of corresponding axis last bin,
   //*-*   the Overflow cell is incremented.
   //*-*
   //*-* If the storage of the sum of squares of weights has been triggered,
   //*-* via the function Sumw2, then the sum of the squares of weights is incremented
   //*-* by w^2 in the cell corresponding to x,y.
   //*-*
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (fBuffer) return BufferFill(x,y,w);

   Int_t binx, biny, bin;
   fEntries++;
   binx = fXaxis.FindBin(x);
   biny = fYaxis.FindBin(y);
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin,w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (binx == 0 || binx > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   if (biny == 0 || biny > fYaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   Double_t z= (w > 0 ? w : -w);
   fTsumw   += z;
   fTsumw2  += z*z;
   fTsumwx  += z*x;
   fTsumwx2 += z*x*x;
   fTsumwy  += z*y;
   fTsumwy2 += z*y*y;
   fTsumwxy += z*x*y;
   return bin;
}

//______________________________________________________________________________
Int_t TH2::Fill(const char *namex, const char *namey, Double_t w)
{
   // Increment cell defined by namex,namey by a weight w
   //
   // if x or/and y is less than the low-edge of the corresponding axis first bin,
   //   the Underflow cell is incremented.
   // if x or/and y is greater than the upper edge of corresponding axis last bin,
   //   the Overflow cell is incremented.
   //
   // If the storage of the sum of squares of weights has been triggered,
   // via the function Sumw2, then the sum of the squares of weights is incremented
   // by w^2 in the cell corresponding to x,y.
   //

   Int_t binx, biny, bin;
   fEntries++;
   binx = fXaxis.FindBin(namex);
   biny = fYaxis.FindBin(namey);
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin,w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   Double_t x = fXaxis.GetBinCenter(binx);
   Double_t y = fYaxis.GetBinCenter(biny);
   Double_t z= (w > 0 ? w : -w);
   fTsumw   += z;
   fTsumw2  += z*z;
   fTsumwx  += z*x;
   fTsumwx2 += z*x*x;
   fTsumwy  += z*y;
   fTsumwy2 += z*y*y;
   fTsumwxy += z*x*y;
   return bin;
}

//______________________________________________________________________________
Int_t TH2::Fill(const char *namex, Double_t y, Double_t w)
{
   // Increment cell defined by namex,y by a weight w
   //
   // if x or/and y is less than the low-edge of the corresponding axis first bin,
   //   the Underflow cell is incremented.
   // if x or/and y is greater than the upper edge of corresponding axis last bin,
   //   the Overflow cell is incremented.
   //
   // If the storage of the sum of squares of weights has been triggered,
   // via the function Sumw2, then the sum of the squares of weights is incremented
   // by w^2 in the cell corresponding to x,y.
   //

   Int_t binx, biny, bin;
   fEntries++;
   binx = fXaxis.FindBin(namex);
   biny = fYaxis.FindBin(y);
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin,w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   Double_t x = fXaxis.GetBinCenter(binx);
   Double_t z= (w > 0 ? w : -w);
   fTsumw   += z;
   fTsumw2  += z*z;
   fTsumwx  += z*x;
   fTsumwx2 += z*x*x;
   fTsumwy  += z*y;
   fTsumwy2 += z*y*y;
   fTsumwxy += z*x*y;
   return bin;
}

//______________________________________________________________________________
Int_t TH2::Fill(Double_t x, const char *namey, Double_t w)
{
   // Increment cell defined by x,namey by a weight w
   //
   // if x or/and y is less than the low-edge of the corresponding axis first bin,
   //   the Underflow cell is incremented.
   // if x or/and y is greater than the upper edge of corresponding axis last bin,
   //   the Overflow cell is incremented.
   //
   // If the storage of the sum of squares of weights has been triggered,
   // via the function Sumw2, then the sum of the squares of weights is incremented
   // by w^2 in the cell corresponding to x,y.
   //

   Int_t binx, biny, bin;
   fEntries++;
   binx = fXaxis.FindBin(x);
   biny = fYaxis.FindBin(namey);
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin,w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (binx == 0 || binx > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   Double_t y = fYaxis.GetBinCenter(biny);
   Double_t z= (w > 0 ? w : -w);
   fTsumw   += z;
   fTsumw2  += z*z;
   fTsumwx  += z*x;
   fTsumwx2 += z*x*x;
   fTsumwy  += z*y;
   fTsumwy2 += z*y*y;
   fTsumwxy += z*x*y;
   return bin;
}

//______________________________________________________________________________
void TH2::FillN(Int_t ntimes, const Double_t *x, const Double_t *y, const Double_t *w, Int_t stride)
{
   //*-*-*-*-*-*-*Fill a 2-D histogram with an array of values and weights*-*-*-*
   //*-*          ========================================================
   //*-*
   //*-* ntimes:  number of entries in arrays x and w (array size must be ntimes*stride)
   //*-* x:       array of x values to be histogrammed
   //*-* y:       array of y values to be histogrammed
   //*-* w:       array of weights
   //*-* stride:  step size through arrays x, y and w
   //*-*
   //*-* If the storage of the sum of squares of weights has been triggered,
   //*-* via the function Sumw2, then the sum of the squares of weights is incremented
   //*-* by w[i]^2 in the cell corresponding to x[i],y[i].
   //*-* if w is NULL each entry is assumed a weight=1
   //*-*
   //*-* NB: function only valid for a TH2x object
   //*-*
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Int_t binx, biny, bin, i;
   fEntries += ntimes;
   Double_t ww = 1;
   ntimes *= stride;
   for (i=0;i<ntimes;i+=stride) {
      binx = fXaxis.FindBin(x[i]);
      biny = fYaxis.FindBin(y[i]);
      bin  = biny*(fXaxis.GetNbins()+2) + binx;
      if (w) ww = w[i];
      AddBinContent(bin,ww);
      if (fSumw2.fN) fSumw2.fArray[bin] += ww*ww;
      if (binx == 0 || binx > fXaxis.GetNbins()) {
         if (!fgStatOverflows) continue;
      }
      if (biny == 0 || biny > fYaxis.GetNbins()) {
         if (!fgStatOverflows) continue;
      }
      Double_t z= (ww > 0 ? ww : -ww);
      fTsumw   += z;
      fTsumw2  += z*z;
      fTsumwx  += z*x[i];
      fTsumwx2 += z*x[i]*x[i];
      fTsumwy  += z*y[i];
      fTsumwy2 += z*y[i]*y[i];
      fTsumwxy += z*x[i]*y[i];
   }
}

//______________________________________________________________________________
void TH2::FillRandom(const char *fname, Int_t ntimes)
{
   //*-*-*-*-*-*-*Fill histogram following distribution in function fname*-*-*-*
   //*-*          =======================================================
   //*-*
   //*-*   The distribution contained in the function fname (TF2) is integrated
   //*-*   over the channel contents.
   //*-*   It is normalized to 1.
   //*-*   Getting one random number implies:
   //*-*     - Generating a random number between 0 and 1 (say r1)
   //*-*     - Look in which bin in the normalized integral r1 corresponds to
   //*-*     - Fill histogram channel
   //*-*   ntimes random numbers are generated
   //*-*
   //*-*  One can also call TF2::GetRandom2 to get a random variate from a function.
   //*-*
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*

   Int_t bin, binx, biny, ibin, loop;
   Double_t r1, x, y, xv[2];
   //*-*- Search for fname in the list of ROOT defined functions
   TF1 *f1 = (TF1*)gROOT->GetFunction(fname);
   if (!f1) { Error("FillRandom", "Unknown function: %s",fname); return; }

   //*-*- Allocate temporary space to store the integral and compute integral
   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbins  = nbinsx*nbinsy;

   Double_t *integral = new Double_t[nbins+1];
   ibin = 0;
   integral[ibin] = 0;
   for (biny=1;biny<=nbinsy;biny++) {
      xv[1] = fYaxis.GetBinCenter(biny);
      for (binx=1;binx<=nbinsx;binx++) {
         xv[0] = fXaxis.GetBinCenter(binx);
         ibin++;
         integral[ibin] = integral[ibin-1] + f1->Eval(xv[0],xv[1]);
      }
   }

   //*-*- Normalize integral to 1
   if (integral[nbins] == 0 ) {
      Error("FillRandom", "Integral = zero"); return;
   }
   for (bin=1;bin<=nbins;bin++)  integral[bin] /= integral[nbins];

   //*-*--------------Start main loop ntimes
   for (loop=0;loop<ntimes;loop++) {
      r1 = gRandom->Rndm(loop);
      ibin = TMath::BinarySearch(nbins,&integral[0],r1);
      biny = ibin/nbinsx;
      binx = 1 + ibin - nbinsx*biny;
      biny++;
      x    = fXaxis.GetBinCenter(binx);
      y    = fYaxis.GetBinCenter(biny);
      Fill(x,y, 1.);
   }
   delete [] integral;
}

//______________________________________________________________________________
void TH2::FillRandom(TH1 *h, Int_t ntimes)
{
   //*-*-*-*-*-*-*Fill histogram following distribution in histogram h*-*-*-*
   //*-*          ====================================================
   //*-*
   //*-*   The distribution contained in the histogram h (TH2) is integrated
   //*-*   over the channel contents.
   //*-*   It is normalized to 1.
   //*-*   Getting one random number implies:
   //*-*     - Generating a random number between 0 and 1 (say r1)
   //*-*     - Look in which bin in the normalized integral r1 corresponds to
   //*-*     - Fill histogram channel
   //*-*   ntimes random numbers are generated
   //*-*
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*

   if (!h) { Error("FillRandom", "Null histogram"); return; }
   if (fDimension != h->GetDimension()) {
      Error("FillRandom", "Histograms with different dimensions"); return;
   }

   if (h->ComputeIntegral() == 0) return;

   Int_t loop;
   Double_t x,y;
   TH2 *h2 = (TH2*)h;
   for (loop=0;loop<ntimes;loop++) {
      h2->GetRandom2(x,y);
      Fill(x,y,1.);
   }
}


//______________________________________________________________________________
void TH2::FitSlicesX(TF1 *f1, Int_t binmin, Int_t binmax, Int_t cut, Option_t *option)
{
   // Project slices along X in case of a 2-D histogram, then fit each slice
   // with function f1 and make a histogram for each fit parameter
   // Only bins along Y between binmin and binmax are considered.
   // if f1=0, a gaussian is assumed
   // Before invoking this function, one can set a subrange to be fitted along X
   // via f1->SetRange(xmin,xmax)
   // The argument option (default="QNR") can be used to change the fit options.
   //     "Q"  means Quiet mode
   //     "N"  means do not show the result of the fit
   //     "R"  means fit the function in the specified function range
   //     "G2" merge 2 consecutive bins along X
   //     "G3" merge 3 consecutive bins along X
   //     "G4" merge 4 consecutive bins along X
   //     "G5" merge 5 consecutive bins along X
   //
   // Note that the generated histograms are added to the list of objects
   // in the current directory. It is the user's responsability to delete
   // these histograms.
   //
   //  Example: Assume a 2-d histogram h2
   //   Root > h2->FitSlicesX(); produces 4 TH1D histograms
   //          with h2_0 containing parameter 0(Constant) for a Gaus fit
   //                    of each bin in Y projected along X
   //          with h2_1 containing parameter 1(Mean) for a gaus fit
   //          with h2_2 containing parameter 2(RMS)  for a gaus fit
   //          with h2_chi2 containing the chisquare/number of degrees of freedom for a gaus fit
   //
   //   Root > h2->FitSlicesX(0,15,22,10);
   //          same as above, but only for bins 15 to 22 along Y
   //          and only for bins in Y for which the corresponding projection
   //          along X has more than cut bins filled.
   //
   //  NOTE: To access the generated histograms in the current directory, do eg:
   //     TH1D *h2_1 = (TH1D*)gDirectory->Get("h2_1");

   Int_t nbins  = fYaxis.GetNbins();
   if (binmin < 1) binmin = 1;
   if (binmax > nbins) binmax = nbins;
   if (binmax < binmin) {binmin = 1; binmax = nbins;}
   TString opt = option;
   opt.ToLower();
   Int_t ngroup = 1;
   if (opt.Contains("g2")) {ngroup = 2; opt.ReplaceAll("g2","");}
   if (opt.Contains("g3")) {ngroup = 3; opt.ReplaceAll("g3","");}
   if (opt.Contains("g4")) {ngroup = 4; opt.ReplaceAll("g4","");}
   if (opt.Contains("g5")) {ngroup = 5; opt.ReplaceAll("g5","");}

   //default is to fit with a gaussian
   if (f1 == 0) {
      f1 = (TF1*)gROOT->GetFunction("gaus");
      if (f1 == 0) f1 = new TF1("gaus","gaus",fXaxis.GetXmin(),fXaxis.GetXmax());
      else         f1->SetRange(fXaxis.GetXmin(),fXaxis.GetXmax());
   }
   Int_t npar = f1->GetNpar();
   if (npar <= 0) return;
   Double_t *parsave = new Double_t[npar];
   f1->GetParameters(parsave);

   //Create one histogram for each function parameter
   Int_t ipar;
   TH1D **hlist = new TH1D*[npar];
   char *name   = new char[2000];
   char *title  = new char[2000];
   const TArrayD *bins = fYaxis.GetXbins();
   for (ipar=0;ipar<npar;ipar++) {
      sprintf(name,"%s_%d",GetName(),ipar);
      sprintf(title,"Fitted value of par[%d]=%s",ipar,f1->GetParName(ipar));
      delete gDirectory->FindObject(name);
      if (bins->fN == 0) {
         hlist[ipar] = new TH1D(name,title, nbins, fYaxis.GetXmin(), fYaxis.GetXmax());
      } else {
         hlist[ipar] = new TH1D(name,title, nbins,bins->fArray);
      }
      hlist[ipar]->GetXaxis()->SetTitle(fYaxis.GetTitle());
   }
   sprintf(name,"%s_chi2",GetName());
   delete gDirectory->FindObject(name);
   TH1D *hchi2 = new TH1D(name,"chisquare", nbins, fYaxis.GetXmin(), fYaxis.GetXmax());
   hchi2->GetXaxis()->SetTitle(fYaxis.GetTitle());

   //Loop on all bins in Y, generate a projection along X
   Int_t bin;
   Int_t nentries;
   for (bin=binmin;bin<=binmax;bin += ngroup) {
      TH1D *hpx = ProjectionX("_temp",bin,bin+ngroup-1,"e");
      if (hpx == 0) continue;
      nentries = Int_t(hpx->GetEntries());
      if (nentries == 0 || nentries < cut) {delete hpx; continue;}
      f1->SetParameters(parsave);
      hpx->Fit(f1,opt.Data());
      Int_t npfits = f1->GetNumberFitPoints();
      if (npfits > npar && npfits >= cut) {
         Int_t binx = bin + ngroup/2;
         for (ipar=0;ipar<npar;ipar++) {
            hlist[ipar]->Fill(fYaxis.GetBinCenter(binx),f1->GetParameter(ipar));
            hlist[ipar]->SetBinError(binx,f1->GetParError(ipar));
         }
         hchi2->Fill(fYaxis.GetBinCenter(binx),f1->GetChisquare()/(npfits-npar));
      }
      delete hpx;
   }
   delete [] parsave;
   delete [] name;
   delete [] title;
   delete [] hlist;
}

//______________________________________________________________________________
void TH2::FitSlicesY(TF1 *f1, Int_t binmin, Int_t binmax, Int_t cut, Option_t *option)
{
   // Project slices along Y in case of a 2-D histogram, then fit each slice
   // with function f1 and make a histogram for each fit parameter
   // Only bins along X between binmin and binmax are considered.
   // if f1=0, a gaussian is assumed
   // Before invoking this function, one can set a subrange to be fitted along Y
   // via f1->SetRange(ymin,ymax)
   // The argument option (default="QNR") can be used to change the fit options.
   //     "Q"  means Quiet mode
   //     "N"  means do not show the result of the fit
   //     "R"  means fit the function in the specified function range
   //     "G2" merge 2 consecutive bins along Y
   //     "G3" merge 3 consecutive bins along Y
   //     "G4" merge 4 consecutive bins along Y
   //     "G5" merge 5 consecutive bins along Y
   //
   // Note that the generated histograms are added to the list of objects
   // in the current directory. It is the user's responsability to delete
   // these histograms.
   //
   //  Example: Assume a 2-d histogram h2
   //   Root > h2->FitSlicesY(); produces 4 TH1D histograms
   //          with h2_0 containing parameter 0(Constant) for a Gaus fit
   //                    of each bin in X projected along Y
   //          with h2_1 containing parameter 1(Mean) for a gaus fit
   //          with h2_2 containing parameter 2(RMS)  for a gaus fit
   //          with h2_chi2 containing the chisquare/number of degrees of freedom for a gaus fit
   //
   //   Root > h2->FitSlicesY(0,15,22,10);
   //          same as above, but only for bins 15 to 22 along X
   //          and only for bins in X for which the corresponding projection
   //          along Y has more than cut bins filled.
   //
   //  NOTE: To access the generated histograms in the current directory, do eg:
   //     TH1D *h2_1 = (TH1D*)gDirectory->Get("h2_1");
   //
   // A complete example of this function is given in begin_html <a href="examples/fitslicesy.C.html">tutorial:fitslicesy.C</a> end_html
   // with the following output:
   //Begin_Html
   /*
   <img src="gif/fitslicesy.gif">
   */
   //End_Html

   Int_t nbins  = fXaxis.GetNbins();
   if (binmin < 1) binmin = 1;
   if (binmax > nbins) binmax = nbins;
   if (binmax < binmin) {binmin = 1; binmax = nbins;}
   TString opt = option;
   opt.ToLower();
   Int_t ngroup = 1;
   if (opt.Contains("g2")) {ngroup = 2; opt.ReplaceAll("g2","");}
   if (opt.Contains("g3")) {ngroup = 3; opt.ReplaceAll("g3","");}
   if (opt.Contains("g4")) {ngroup = 4; opt.ReplaceAll("g4","");}
   if (opt.Contains("g5")) {ngroup = 5; opt.ReplaceAll("g5","");}

   //default is to fit with a gaussian
   if (f1 == 0) {
      f1 = (TF1*)gROOT->GetFunction("gaus");
      if (f1 == 0) f1 = new TF1("gaus","gaus",fYaxis.GetXmin(),fYaxis.GetXmax());
      else         f1->SetRange(fYaxis.GetXmin(),fYaxis.GetXmax());
   }
   Int_t npar = f1->GetNpar();
   if (npar <= 0) return;
   Double_t *parsave = new Double_t[npar];
   f1->GetParameters(parsave);

   //Create one histogram for each function parameter
   Int_t ipar;
   TH1D **hlist = new TH1D*[npar];
   char *name   = new char[2000];
   char *title  = new char[2000];
   const TArrayD *bins = fXaxis.GetXbins();
   for (ipar=0;ipar<npar;ipar++) {
      sprintf(name,"%s_%d",GetName(),ipar);
      sprintf(title,"Fitted value of par[%d]=%s",ipar,f1->GetParName(ipar));
      delete gDirectory->FindObject(name);
      if (bins->fN == 0) {
         hlist[ipar] = new TH1D(name,title, nbins, fXaxis.GetXmin(), fXaxis.GetXmax());
      } else {
         hlist[ipar] = new TH1D(name,title, nbins,bins->fArray);
      }
      hlist[ipar]->GetXaxis()->SetTitle(fXaxis.GetTitle());
   }
   sprintf(name,"%s_chi2",GetName());
   delete gDirectory->FindObject(name);
   TH1D *hchi2 = new TH1D(name,"chisquare", nbins, fXaxis.GetXmin(), fXaxis.GetXmax());
   hchi2->GetXaxis()->SetTitle(fXaxis.GetTitle());

   //Loop on all bins in X, generate a projection along Y
   Int_t bin;
   Int_t nentries;
   for (bin=binmin;bin<=binmax;bin += ngroup) {
      TH1D *hpy = ProjectionY("_temp",bin,bin+ngroup-1,"e");
      if (hpy == 0) continue;
      nentries = Int_t(hpy->GetEntries());
      if (nentries == 0 || nentries < cut) {delete hpy; continue;}
      f1->SetParameters(parsave);
      hpy->Fit(f1,opt.Data());
      Int_t npfits = f1->GetNumberFitPoints();
      if (npfits > npar && npfits >= cut) {
         Int_t biny = bin + ngroup/2;
         for (ipar=0;ipar<npar;ipar++) {
            hlist[ipar]->Fill(fXaxis.GetBinCenter(biny),f1->GetParameter(ipar));
            hlist[ipar]->SetBinError(biny,f1->GetParError(ipar));
         }
         hchi2->Fill(fXaxis.GetBinCenter(biny),f1->GetChisquare()/(npfits-npar));
      }
      delete hpy;
   }
   delete [] parsave;
   delete [] name;
   delete [] title;
   delete [] hlist;
}

//______________________________________________________________________________
Double_t TH2::GetBinWithContent2(Double_t c, Int_t &binx, Int_t &biny, Int_t firstx, Int_t lastx, Int_t firsty, Int_t lasty, Double_t maxdiff) const
{
   // compute first cell (binx,biny) in the range [firstx,lastx](firsty,lasty] for which
   // diff = abs(cell_content-c) <= maxdiff
   // In case several cells in the specified range with diff=0 are found
   // the first cell found is returned in binx,biny.
   // In case several cells in the specified range satisfy diff <=maxdiff
   // the cell with the smallest difference is returned in binx,biny.
   // In all cases the function returns the smallest difference.
   //
   // NOTE1: if firstx <= 0, firstx is set to bin 1
   //        if (lastx < firstx then firstx is set to the number of bins in X
   //        ie if firstx=0 and lastx=0 (default) the search is on all bins in X.
   //        if firsty <= 0, firsty is set to bin 1
   //        if (lasty < firsty then firsty is set to the number of bins in Y
   //        ie if firsty=0 and lasty=0 (default) the search is on all bins in Y.
   // NOTE2: if maxdiff=0 (default), the first cell with content=c is returned.

   if (fDimension != 2) {
      binx = 0;
      biny = 0;
      Error("GetBinWithContent2","function is only valid for 2-D histograms");
      return 0;
   }
   if (firstx <= 0) firstx = 1;
   if (lastx < firstx) lastx = fXaxis.GetNbins();
   if (firsty <= 0) firsty = 1;
   if (lasty < firsty) lasty = fYaxis.GetNbins();
   Int_t binminx = 0, binminy=0;
   Double_t diff, curmax = 1.e240;
   for (Int_t j=firsty;j<=lasty;j++) {
      for (Int_t i=firstx;i<=lastx;i++) {
         diff = TMath::Abs(GetBinContent(i,j)-c);
         if (diff <= 0) {binx = i; biny=j; return diff;}
         if (diff < curmax && diff <= maxdiff) {curmax = diff, binminx=i; binminy=j;}
      }
   }
   binx = binminx;
   biny = binminy;
   return curmax;
}

//______________________________________________________________________________
Double_t TH2::GetCorrelationFactor(Int_t axis1, Int_t axis2) const
{
   //*-*-*-*-*-*-*-*Return correlation factor between axis1 and axis2*-*-*-*-*
   //*-*            ====================================================
   if (axis1 < 1 || axis2 < 1 || axis1 > 2 || axis2 > 2) {
      Error("GetCorrelationFactor","Wrong parameters");
      return 0;
   }
   if (axis1 == axis2) return 1;
   Double_t rms1 = GetRMS(axis1);
   if (rms1 == 0) return 0;
   Double_t rms2 = GetRMS(axis2);
   if (rms2 == 0) return 0;
   return GetCovariance(axis1,axis2)/rms1/rms2;
}

//______________________________________________________________________________
Double_t TH2::GetCovariance(Int_t axis1, Int_t axis2) const
{
   //*-*-*-*-*-*-*-*Return covariance between axis1 and axis2*-*-*-*-*
   //*-*            ====================================================

   if (axis1 < 1 || axis2 < 1 || axis1 > 2 || axis2 > 2) {
      Error("GetCovariance","Wrong parameters");
      return 0;
   }
   Double_t stats[7];
   GetStats(stats);
   Double_t sumw   = stats[0];
   //Double_t sumw2  = stats[1];
   Double_t sumwx  = stats[2];
   Double_t sumwx2 = stats[3];
   Double_t sumwy  = stats[4];
   Double_t sumwy2 = stats[5];
   Double_t sumwxy = stats[6];

   if (sumw == 0) return 0;
   if (axis1 == 1 && axis2 == 1) {
      return TMath::Abs(sumwx2/sumw - sumwx/sumw*sumwx/sumw);
   }
   if (axis1 == 2 && axis2 == 2) {
      return TMath::Abs(sumwy2/sumw - sumwy/sumw*sumwy/sumw);
   }
   return sumwxy/sumw - sumwx/sumw*sumwy/sumw;
}

//______________________________________________________________________________
void TH2::GetRandom2(Double_t &x, Double_t &y)
{
   // return 2 random numbers along axis x and y distributed according
   // the cellcontents of a 2-dim histogram

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbins  = nbinsx*nbinsy;
   Double_t integral;
   if (fIntegral) {
      if (fIntegral[nbins+1] != fEntries) integral = ComputeIntegral();
   } else {
      integral = ComputeIntegral();
      if (integral == 0 || fIntegral == 0) return;
   }
   Float_t r1 = gRandom->Rndm();
   Int_t ibin = TMath::BinarySearch(nbins,fIntegral,r1);
   Int_t biny = ibin/nbinsx;
   Int_t binx = ibin - nbinsx*biny;
   x = fXaxis.GetBinLowEdge(binx+1);
   if (r1 > fIntegral[ibin]) x +=
      fXaxis.GetBinWidth(binx+1)*(r1-fIntegral[ibin])/(fIntegral[ibin+1] - fIntegral[ibin]);
   y = fYaxis.GetBinLowEdge(biny+1) + fYaxis.GetBinWidth(biny+1)*gRandom->Rndm();
}

//______________________________________________________________________________
void TH2::GetStats(Double_t *stats) const
{
   // fill the array stats from the contents of this histogram
   // The array stats must be correctly dimensionned in the calling program.
   // stats[0] = sumw
   // stats[1] = sumw2
   // stats[2] = sumwx
   // stats[3] = sumwx2
   // stats[4] = sumwy
   // stats[5] = sumwy2
   // stats[6] = sumwxy

   if (fBuffer) ((TH2*)this)->BufferEmpty();

   Int_t bin, binx, biny;
   Double_t w,err;
   Double_t x,y;
   if (fTsumw == 0 || fXaxis.TestBit(TAxis::kAxisRange) || fYaxis.TestBit(TAxis::kAxisRange)) {
      for (bin=0;bin<7;bin++) stats[bin] = 0;
      for (biny=fYaxis.GetFirst();biny<=fYaxis.GetLast();biny++) {
         y = fYaxis.GetBinCenter(biny);
         for (binx=fXaxis.GetFirst();binx<=fXaxis.GetLast();binx++) {
            bin = GetBin(binx,biny);
            x   = fXaxis.GetBinCenter(binx);
            w   = TMath::Abs(GetBinContent(bin));
            err = TMath::Abs(GetBinError(bin));
            stats[0] += w;
            stats[1] += err*err;
            stats[2] += w*x;
            stats[3] += w*x*x;
            stats[4] += w*y;
            stats[5] += w*y*y;
            stats[6] += w*x*y;
         }
      }
   } else {
      stats[0] = fTsumw;
      stats[1] = fTsumw2;
      stats[2] = fTsumwx;
      stats[3] = fTsumwx2;
      stats[4] = fTsumwy;
      stats[5] = fTsumwy2;
      stats[6] = fTsumwxy;
   }
}

//______________________________________________________________________________
Double_t TH2::Integral(Option_t *option) const
{
   //Return integral of bin contents. Only bins in the bins range are considered.
   // By default the integral is computed as the sum of bin contents in the range.
   // if option "width" is specified, the integral is the sum of
   // the bin contents multiplied by the bin width in x and in y.

   return Integral(fXaxis.GetFirst(),fXaxis.GetLast(),
      fYaxis.GetFirst(),fYaxis.GetLast(),option);
}

//______________________________________________________________________________
Double_t TH2::Integral(Int_t binx1, Int_t binx2, Int_t biny1, Int_t biny2, Option_t *option) const
{
   //Return integral of bin contents in range [binx1,binx2],[biny1,biny2]
   // for a 2-D histogram
   // By default the integral is computed as the sum of bin contents in the range.
   // if option "width" is specified, the integral is the sum of
   // the bin contents multiplied by the bin width in x and in y.

   if (fBuffer) ((TH2*)this)->BufferEmpty();

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   if (binx1 < 0) binx1 = 0;
   if (binx2 > nbinsx+1) binx2 = nbinsx+1;
   if (binx2 < binx1)    binx2 = nbinsx;
   if (biny1 < 0) biny1 = 0;
   if (biny2 > nbinsy+1) biny2 = nbinsy+1;
   if (biny2 < biny1)    biny2 = nbinsy;
   Double_t integral = 0;

   //*-*- Loop on bins in specified range
   TString opt = option;
   opt.ToLower();
   Bool_t width = kFALSE;
   if (opt.Contains("width")) width = kTRUE;
   Int_t bin, binx, biny;
   for (biny=biny1;biny<=biny2;biny++) {
      for (binx=binx1;binx<=binx2;binx++) {
         bin = binx +(nbinsx+2)*biny;
         if (width) integral += GetBinContent(bin)*fXaxis.GetBinWidth(binx)*fYaxis.GetBinWidth(biny);
         else       integral += GetBinContent(bin);
      }
   }
   return integral;
}

//______________________________________________________________________________
Double_t TH2::KolmogorovTest(const TH1 *h2, Option_t *option) const
{
   //  Statistical test of compatibility in shape between
   //  THIS histogram and h2, using Kolmogorov test.
   //     Default: Ignore under- and overflow bins in comparison
   //
   //     option is a character string to specify options
   //         "U" include Underflows in test
   //         "O" include Overflows
   //         "N" include comparison of normalizations
   //         "D" Put out a line of "Debug" printout
   //
   //   The returned function value is the probability of test
   //       (much less than one means NOT compatible)
   //
   //  Code adapted by Rene Brun from original HBOOK routine HDIFF

   TString opt = option;
   opt.ToUpper();

   Double_t prb = 0;
   TH1 *h1 = (TH1*)this;
   if (h2 == 0) return 0;
   TAxis *xaxis1 = h1->GetXaxis();
   TAxis *xaxis2 = h2->GetXaxis();
   TAxis *yaxis1 = h1->GetYaxis();
   TAxis *yaxis2 = h2->GetYaxis();
   Int_t ncx1   = xaxis1->GetNbins();
   Int_t ncx2   = xaxis2->GetNbins();
   Int_t ncy1   = yaxis1->GetNbins();
   Int_t ncy2   = yaxis2->GetNbins();

   // Check consistency of dimensions
   if (h1->GetDimension() != 2 || h2->GetDimension() != 2) {
      Error("KolmogorovTest","Histograms must be 2-D\n");
      return 0;
   }

   // Check consistency in number of channels
   if (ncx1 != ncx2) {
      Error("KolmogorovTest","Number of channels in X is different, %d and %d\n",ncx1,ncx2);
      return 0;
   }
   if (ncy1 != ncy2) {
      Error("KolmogorovTest","Number of channels in Y is different, %d and %d\n",ncy1,ncy2);
      return 0;
   }

   // Check consistency in channel edges
   Bool_t afunc1 = kFALSE;
   Bool_t afunc2 = kFALSE;
   Double_t difprec = 1e-5;
   Double_t diff1 = TMath::Abs(xaxis1->GetXmin() - xaxis2->GetXmin());
   Double_t diff2 = TMath::Abs(xaxis1->GetXmax() - xaxis2->GetXmax());
   if (diff1 > difprec || diff2 > difprec) {
      Error("KolmogorovTest","histograms with different binning along X");
      return 0;
   }
   diff1 = TMath::Abs(yaxis1->GetXmin() - yaxis2->GetXmin());
   diff2 = TMath::Abs(yaxis1->GetXmax() - yaxis2->GetXmax());
   if (diff1 > difprec || diff2 > difprec) {
      Error("KolmogorovTest","histograms with different binning along Y");
      return 0;
   }

   //   Should we include Uflows, Oflows?
   Int_t ibeg = 1, jbeg = 1;
   Int_t iend = ncx1, jend = ncy1;
   if (opt.Contains("U")) {ibeg = 0; jbeg = 0;}
   if (opt.Contains("O")) {iend = ncx1+1; jend = ncy1+1;}

   Int_t i,j;
   Double_t hsav;
   Double_t sum1  = 0;
   Double_t tsum1 = 0;
   for (i=0;i<=ncx1+1;i++) {
      for (j=0;j<=ncy1+1;j++) {
         hsav = h1->GetCellContent(i,j);
         tsum1 += hsav;
         if (i >= ibeg && i <= iend && j >= jbeg && j <= jend) sum1 += hsav;
      }
   }
   Double_t sum2  = 0;
   Double_t tsum2 = 0;
   for (i=0;i<=ncx1+1;i++) {
      for (j=0;j<=ncy1+1;j++) {
         hsav = h2->GetCellContent(i,j);
         tsum2 += hsav;
         if (i >= ibeg && i <= iend && j >= jbeg && j <= jend) sum2 += hsav;
      }
   }

   //    Check that both scatterplots contain events
   if (sum1 == 0) {
      Error("KolmogorovTest","Integral is zero for h1=%s\n",h1->GetName());
      return 0;
   }
   if (sum2 == 0) {
      Error("KolmogorovTest","Integral is zero for h2=%s\n",h2->GetName());
      return 0;
   }

   //    Check that scatterplots are not weighted or saturated
   Double_t num1 = h1->GetEntries();
   Double_t num2 = h2->GetEntries();
   if (num1 != tsum1) {
      Warning("KolmogorovTest","Saturation or weighted events for h1=%s, num1=%g, tsum1=%g\n",h1->GetName(),num1,tsum1);
   }
   if (num2 != tsum2) {
      Warning("KolmogorovTest","Saturation or weighted events for h2=%s, num2=%g, tsum2=%g\n",h2->GetName(),num2,tsum2);
   }

   //   Find first Kolmogorov distance
   Double_t s1 = 1/sum1;
   Double_t s2 = 1/sum2;
   Double_t dfmax = 0;
   Double_t rsum1=0, rsum2=0;
   for (i=ibeg;i<=iend;i++) {
      for (j=jbeg;j<=jend;j++) {
         rsum1 += s1*h1->GetCellContent(i,j);
         rsum2 += s2*h2->GetCellContent(i,j);
         dfmax  = TMath::Max(dfmax, TMath::Abs(rsum1-rsum2));
      }
   }

   //   Find second Kolmogorov distance
   Double_t dfmax2 = 0;
   rsum1=0, rsum2=0;
   for (j=jbeg;j<=jend;j++) {
      for (i=ibeg;i<=iend;i++) {
         rsum1 += s1*h1->GetCellContent(i,j);
         rsum2 += s2*h2->GetCellContent(i,j);
         dfmax2 = TMath::Max(dfmax2, TMath::Abs(rsum1-rsum2));
      }
   }

   //    Get Kolmogorov probability
   Double_t factnm;
   if (afunc1)      factnm = TMath::Sqrt(sum2);
   else if (afunc2) factnm = TMath::Sqrt(sum1);
   else             factnm = TMath::Sqrt(sum1*sum2/(sum1+sum2));
   Double_t z  = dfmax*factnm;
   Double_t z2 = dfmax2*factnm;

   prb = TMath::KolmogorovProb(0.5*(z+z2));

   Double_t prb1=0, prb2=0;
   Double_t resum1, resum2, chi2, d12;
   if (opt.Contains("N")) { //Combine probabilities for shape and normalization,
      prb1   = prb;
      resum1 = sum1; if (afunc1) resum1 = 0;
      resum2 = sum2; if (afunc2) resum2 = 0;
      d12    = sum1-sum2;
      chi2   = d12*d12/(resum1+resum2);
      prb2   = TMath::Prob(chi2,1);
      //     see Eadie et al., section 11.6.2
      if (prb > 0 && prb2 > 0) prb = prb*prb2*(1-TMath::Log(prb*prb2));
      else                     prb = 0;
   }

   //    debug printout
   if (opt.Contains("D")) {
      printf(" Kolmo Prob  h1 = %s, sum1=%g\n",h1->GetName(),sum1);
      printf(" Kolmo Prob  h2 = %s, sum2=%g\n",h2->GetName(),sum2);
      printf(" Kolmo Probabil = %f, Max Dist = %g\n",prb,dfmax);
      if (opt.Contains("N"))
         printf(" Kolmo Probabil = %f for shape alone, =%f for normalisation alone\n",prb1,prb2);
   }
   // This numerical error condition should never occur:
   if (TMath::Abs(rsum1-1) > 0.002) Warning("KolmogorovTest","Numerical problems with h1=%s\n",h1->GetName());
   if (TMath::Abs(rsum2-1) > 0.002) Warning("KolmogorovTest","Numerical problems with h2=%s\n",h2->GetName());

   return prb;
}

//______________________________________________________________________________
Long64_t TH2::Merge(TCollection *list)
{
   //Add all histograms in the collection to this histogram.
   //This function computes the min/max for the axes,
   //compute a new number of bins, if necessary,
   //add bin contents, errors and statistics.
   //If overflows are present and limits are different the function will fail.
   //The function returns the total number of entries in the result histogram
   //if the merge is successfull, -1 otherwise.
   //
   //IMPORTANT remark. The 2 axis x and y may have different number
   //of bins and different limits, BUT the largest bin width must be
   //a multiple of the smallest bin width and the upper limit must also
   //be a multiple of the bin width.

   if (!list) return 0;
   if (list->IsEmpty()) return (Int_t) GetEntries();

   TList inlist;
   TH1* hclone = (TH1*)Clone("FirstClone");
   R__ASSERT(hclone);
   BufferEmpty(1);         // To remove buffer.
   Reset();                // BufferEmpty sets limits so we can't use it later.
   SetEntries(0);
   inlist.Add(hclone);
   inlist.AddAll(list);

   TAxis newXAxis;
   TAxis newYAxis;
   Bool_t initialLimitsFound = kFALSE;
   Bool_t same = kTRUE;
   Bool_t allHaveLimits = kTRUE;

   TIter next(&inlist);
   while (TObject *o = next()) {
      TH2* h = dynamic_cast<TH2*> (o);
      if (!h) {
         Error("Add","Attempt to add object of class: %s to a %s",
            o->ClassName(),this->ClassName());
         return -1;
      }
      Bool_t hasLimits = h->GetXaxis()->GetXmin() < h->GetXaxis()->GetXmax();
      allHaveLimits = allHaveLimits && hasLimits;

      if (hasLimits) {
         h->BufferEmpty();
         if (!initialLimitsFound) {
            initialLimitsFound = kTRUE;
            newXAxis.Set(h->GetXaxis()->GetNbins(), h->GetXaxis()->GetXmin(),
               h->GetXaxis()->GetXmax());
            newYAxis.Set(h->GetYaxis()->GetNbins(), h->GetYaxis()->GetXmin(),
               h->GetYaxis()->GetXmax());
         }
         else {
            if (!RecomputeAxisLimits(newXAxis, *(h->GetXaxis()))) {
               Error("Merge", "Cannot merge histograms - limits are inconsistent:\n "
                  "first: (%d, %f, %f), second: (%d, %f, %f)",
                  newXAxis.GetNbins(), newXAxis.GetXmin(), newXAxis.GetXmax(),
                  h->GetXaxis()->GetNbins(), h->GetXaxis()->GetXmin(),
                  h->GetXaxis()->GetXmax());
            }
            if (!RecomputeAxisLimits(newYAxis, *(h->GetYaxis()))) {
               Error("Merge", "Cannot merge histograms - limits are inconsistent:\n "
                  "first: (%d, %f, %f), second: (%d, %f, %f)",
                  newYAxis.GetNbins(), newYAxis.GetXmin(), newYAxis.GetXmax(),
                  h->GetYaxis()->GetNbins(), h->GetYaxis()->GetXmin(),
                  h->GetYaxis()->GetXmax());
            }
         }
      }
   }
   next.Reset();

   same = same && SameLimitsAndNBins(newXAxis, *GetXaxis())
      && SameLimitsAndNBins(newYAxis, *GetYaxis());
   if (!same && initialLimitsFound)
      SetBins(newXAxis.GetNbins(), newXAxis.GetXmin(), newXAxis.GetXmax(),
      newYAxis.GetNbins(), newYAxis.GetXmin(), newYAxis.GetXmax());

   if (!allHaveLimits) {
      // fill this histogram with all the data from buffers of histograms without limits
      while (TH2* h = dynamic_cast<TH2*> (next())) {
         if (h->GetXaxis()->GetXmin() >= h->GetXaxis()->GetXmax() && h->fBuffer) {
            // no limits
            Int_t nbentries = (Int_t)h->fBuffer[0];
            for (Int_t i = 0; i < nbentries; i++)
               Fill(h->fBuffer[3*i + 2], h->fBuffer[3*i + 3], h->fBuffer[3*i + 1]);
            // Entries from buffers have to be filled one by one
            // because FillN doesn't resize histograms.
         }
      }
      if (!initialLimitsFound)
         return (Int_t) GetEntries();  // all histograms have been processed
      next.Reset();
   }

   //merge bin contents and errors
   const Int_t kNstat = 7;
   Double_t stats[kNstat], totstats[kNstat];
   for (Int_t i=0;i<kNstat;i++) {totstats[i] = stats[i] = 0;}
   GetStats(totstats);
   Double_t nentries = GetEntries();
   Int_t binx, biny, ix, iy, nx, ny, bin, ibin;
   Double_t cu;
   Int_t nbix = fXaxis.GetNbins();
   Bool_t canRebin=TestBit(kCanRebin);
   ResetBit(kCanRebin); // reset, otherwise setting the under/overflow will rebin

   while (TH1* h=(TH1*)next()) {
      // process only if the histogram has limits; otherwise it was processed before
      if (h->GetXaxis()->GetXmin() < h->GetXaxis()->GetXmax()) {
         // import statistics
         h->GetStats(stats);
         for (Int_t i = 0; i < kNstat; i++)
            totstats[i] += stats[i];
         nentries += h->GetEntries();

         nx = h->GetXaxis()->GetNbins();
         ny = h->GetYaxis()->GetNbins();

         for (biny = 0; biny <= ny + 1; biny++) {
            iy = fYaxis.FindBin(h->GetYaxis()->GetBinCenter(biny));
            for (binx = 0; binx <= nx + 1; binx++) {
               ix = fXaxis.FindBin(h->GetXaxis()->GetBinCenter(binx));
               bin = binx +(nx+2)*biny;
               ibin = ix +(nbix+2)*iy;
               cu = h->GetBinContent(bin);
               if ((!same) && (binx == 0 || binx == nx + 1
                  || biny == 0 || biny == ny + 1)) {
                     if (cu != 0) {
                        Error("Merge", "Cannot merge histograms - the histograms have"
                           " different limits and undeflows/overflows are present."
                           " The initial histogram is now broken!");
                        return -1;
                     }
               }
               AddBinContent(ibin,cu);
               if (fSumw2.fN) {
                  Double_t error1 = h->GetBinError(bin);
                  fSumw2.fArray[ibin] += error1*error1;
               }
            }
         }
      }
   }
   if (canRebin) SetBit(kCanRebin);

   //copy merged stats
   PutStats(totstats);
   SetEntries(nentries);
   inlist.Remove(hclone);
   delete hclone;
   return (Long64_t)nentries;
}

//______________________________________________________________________________
TH2 *TH2::RebinX(Int_t ngroup, const char *newname)
{
   // Rebin only the X axis
   // see Rebin2D

   return Rebin2D(ngroup, 1, newname);
}

//______________________________________________________________________________
TH2 *TH2::RebinY(Int_t ngroup, const char *newname)
{
   // Rebin only the Y axis
   // see Rebin2D

   return Rebin2D(1, ngroup, newname);
}


//______________________________________________________________________________
TH2 *TH2::Rebin2D(Int_t nxgroup, Int_t nygroup, const char *newname)
{
   //   -*-*-*Rebin this histogram grouping nxgroup/nygroup bins along the xaxis/yaxis together*-*-*-*-
   //         =================================================================================
   //   if newname is not blank a new temporary histogram hnew is created.
   //   else the current histogram is modified (default)
   //   The parameter nxgroup/nygroup indicate how many bins along the xaxis/yaxis of this
   //   have to me merged into one bin of hnew
   //   If the original histogram has errors stored (via Sumw2), the resulting
   //   histograms has new errors correctly calculated.
   //
   //   examples: if hpxpy is an existing TH2 histogram with 40 x 40 bins
   //     hpxpy->Rebin();  // merges two bins along the xaxis and yaxis in one in hpxpy
   //                      // Carefull: previous contents of hpxpy are lost
   //     hpxpy->RebinX(5); //merges five bins along the xaxis in one in hpxpy
   //     TH2 *hnew = hpxpy->RebinY(5,"hnew"); // creates a new histogram hnew
   //                                          // merging 5 bins of h1 along the yaxis in one bin
   //
   //   NOTE : If nxgroup/nygroup is not an exact divider of the number of bins,
   //          along the xaxis/yaxis the top limit(s) of the rebinned histogram
   //          is changed to the upper edge of the xbin=newxbins*nxgroup resp.
   //          ybin=newybins*nygroup and the corresponding bins are added to
   //          the overflow bin.
   //          Statistics will be recomputed from the new bin contents.

   Int_t i,j,xbin,ybin;
   Int_t nxbins  = fXaxis.GetNbins();
   Int_t nybins  = fYaxis.GetNbins();
   Double_t xmin  = fXaxis.GetXmin();
   Double_t xmax  = fXaxis.GetXmax();
   Double_t ymin  = fYaxis.GetXmin();
   Double_t ymax  = fYaxis.GetXmax();
   if ((nxgroup <= 0) || (nxgroup > nxbins)) {
      Error("Rebin", "Illegal value of nxgroup=%d",nxgroup);
      return 0;
   }
   if ((nygroup <= 0) || (nygroup > nybins)) {
      Error("Rebin", "Illegal value of nygroup=%d",nygroup);
      return 0;
   }

   Int_t newxbins = nxbins/nxgroup;
   Int_t newybins = nybins/nygroup;

   // Save old bin contents into a new array
   Double_t entries = fEntries;
   Double_t *oldBins = new Double_t[(nxbins+2)*(nybins+2)];
   for (xbin = 0; xbin < nxbins+2; xbin++) {
      for (ybin = 0; ybin < nybins+2; ybin++) {
         oldBins[xbin*(nybins+2)+ybin] = GetBinContent(xbin, ybin);
      }
   }
   Double_t *oldErrors = 0;
   if (fSumw2.fN != 0) {
      oldErrors = new Double_t[(nxbins+2)*(nybins+2)];
      for (xbin = 0; xbin < nxbins+2; xbin++) {
         for (ybin = 0; ybin < nybins+2; ybin++) {
            oldErrors[xbin*(nybins+2)+ybin] = GetBinError(xbin, ybin);
         }
      }
   }

   // create a clone of the old histogram if newname is specified
   TH2 *hnew = this;
   if (newname && strlen(newname)) {
      hnew = (TH2*)Clone();
      hnew->SetName(newname);
   }

   // change axis specs and rebuild bin contents array
   if(newxbins*nxgroup != nxbins) {
      xmax = fXaxis.GetBinUpEdge(newxbins*nxgroup);
      hnew->fTsumw = 0; //stats must be reset because top bins will be moved to overflow bin
   }
   if(newybins*nygroup != nybins) {
      ymax = fYaxis.GetBinUpEdge(newybins*nygroup);
      hnew->fTsumw = 0; //stats must be reset because top bins will be moved to overflow bin
   }
   // save the TAttAxis members (reset by SetBins) for x axis
   Int_t    nXdivisions  = fXaxis.GetNdivisions();
   Color_t  xAxisColor   = fXaxis.GetAxisColor();
   Color_t  xLabelColor  = fXaxis.GetLabelColor();
   Style_t  xLabelFont   = fXaxis.GetLabelFont();
   Float_t  xLabelOffset = fXaxis.GetLabelOffset();
   Float_t  xLabelSize   = fXaxis.GetLabelSize();
   Float_t  xTickLength  = fXaxis.GetTickLength();
   Float_t  xTitleOffset = fXaxis.GetTitleOffset();
   Float_t  xTitleSize   = fXaxis.GetTitleSize();
   Color_t  xTitleColor  = fXaxis.GetTitleColor();
   Style_t  xTitleFont   = fXaxis.GetTitleFont();
   // save the TAttAxis members (reset by SetBins) for y axis
   Int_t    nYdivisions  = fYaxis.GetNdivisions();
   Color_t  yAxisColor   = fYaxis.GetAxisColor();
   Color_t  yLabelColor  = fYaxis.GetLabelColor();
   Style_t  yLabelFont   = fYaxis.GetLabelFont();
   Float_t  yLabelOffset = fYaxis.GetLabelOffset();
   Float_t  yLabelSize   = fYaxis.GetLabelSize();
   Float_t  yTickLength  = fYaxis.GetTickLength();
   Float_t  yTitleOffset = fYaxis.GetTitleOffset();
   Float_t  yTitleSize   = fYaxis.GetTitleSize();
   Color_t  yTitleColor  = fYaxis.GetTitleColor();
   Style_t  yTitleFont   = fYaxis.GetTitleFont();


   // copy merged bin contents (ignore under/overflows)
   if (nxgroup != 1 || nygroup != 1) {
      if(fXaxis.GetXbins()->GetSize() > 0 || fYaxis.GetXbins()->GetSize() > 0){
         // variable bin sizes in x or y, don't treat both cases separately
         Double_t *xbins = new Double_t[newxbins+1];
         for(i = 0; i <= newxbins; ++i) xbins[i] = fXaxis.GetBinLowEdge(1+i*nxgroup);
         Double_t *ybins = new Double_t[newybins+1];
         for(i = 0; i <= newybins; ++i) ybins[i] = fYaxis.GetBinLowEdge(1+i*nygroup);
         hnew->SetBins(newxbins,xbins, newybins, ybins);//changes also errors array (if any)
         delete [] xbins;
         delete [] ybins;
      } else {
         hnew->SetBins(newxbins, xmin, xmax, newybins, ymin, ymax);//changes also errors array
      }

      Double_t binContent, binError;
      Int_t oldxbin = 1;
      for (xbin = 1; xbin <= newxbins; xbin++) {
         Int_t oldybin = 1;
         for (ybin = 1; ybin <= newybins; ybin++) {
            binContent = 0;
            binError   = 0;
            for (i = 0; i < nxgroup; i++) {
               if (oldxbin+i > nxbins) break;
               for (j =0; j < nygroup; j++) {
                  if (oldybin+j > nybins) break;
                  binContent += oldBins[oldybin+j + (oldxbin+i)*(nybins+2)];
                  if (oldErrors) binError += oldErrors[oldybin+ j + (oldxbin+i)*(nybins+2)]*oldErrors[oldybin + j + (oldxbin+i)*(nybins+2)];
               }
            }
            hnew->SetBinContent(xbin,ybin, binContent);
            if (oldErrors) hnew->SetBinError(xbin,ybin,TMath::Sqrt(binError));
            oldybin += nygroup;
         }
         oldxbin += nxgroup;
      }

      // Recompute correct underflows and overflows.
      hnew->SetBinContent(newxbins+1,newybins+1,oldBins[(nxbins+1)*(nybins+2)+(nybins+1)]);
      hnew->SetBinContent(0,0,oldBins[0]);
      hnew->SetBinContent(0,newybins+1,oldBins[nybins+1]);
      hnew->SetBinContent(newxbins+1,0,oldBins[(nxbins+1)*(nybins+2)]);

      Double_t binContent0, binContent2;
      oldxbin = 1;
      for (xbin = 1; xbin<=newxbins; xbin++) {
         binContent0 = binContent2 = 0;
         for (i=0; i<nxgroup; i++) {
            if (oldxbin+i > nxbins) break;
            binContent0 += oldBins[(oldxbin+i)*(nybins+2)];
            binContent2 += oldBins[(oldxbin+i)*(nybins+2)+(nybins+1)];
         }
         hnew->SetBinContent(xbin,0,binContent0);
         hnew->SetBinContent(xbin,newybins+1,binContent2);
         oldxbin += nxgroup;
      }

      Int_t oldybin = 1;
      for (ybin = 1; ybin<=newybins; ybin++) {
         binContent0 = binContent2 = 0;
         for (i=0; i<nygroup; i++) {
            if (oldybin+i > nybins) break;
            binContent0 += oldBins[(oldybin+i)];
            binContent2 += oldBins[(nxbins+1)*(nybins+2)+(oldybin+i)];
         }
         hnew->SetBinContent(0,ybin,binContent0);
         hnew->SetBinContent(newxbins+1,ybin,binContent2);
         oldybin += nygroup;
      }
   }

   // Restore x axis attributes
   fXaxis.SetNdivisions(nXdivisions);
   fXaxis.SetAxisColor(xAxisColor);
   fXaxis.SetLabelColor(xLabelColor);
   fXaxis.SetLabelFont(xLabelFont);
   fXaxis.SetLabelOffset(xLabelOffset);
   fXaxis.SetLabelSize(xLabelSize);
   fXaxis.SetTickLength(xTickLength);
   fXaxis.SetTitleOffset(xTitleOffset);
   fXaxis.SetTitleSize(xTitleSize);
   fXaxis.SetTitleColor(xTitleColor);
   fXaxis.SetTitleFont(xTitleFont);
   // Restore y axis attributes
   fYaxis.SetNdivisions(nYdivisions);
   fYaxis.SetAxisColor(yAxisColor);
   fYaxis.SetLabelColor(yLabelColor);
   fYaxis.SetLabelFont(yLabelFont);
   fYaxis.SetLabelOffset(yLabelOffset);
   fYaxis.SetLabelSize(yLabelSize);
   fYaxis.SetTickLength(yTickLength);
   fYaxis.SetTitleOffset(yTitleOffset);
   fYaxis.SetTitleSize(yTitleSize);
   fYaxis.SetTitleColor(yTitleColor);
   fYaxis.SetTitleFont(yTitleFont);

   hnew->SetEntries(entries); //was modified by SetBinContent

   delete [] oldBins;
   if (oldErrors) delete [] oldErrors;
   return hnew;
}

//______________________________________________________________________________
TProfile *TH2::ProfileX(const char *name, Int_t firstybin, Int_t lastybin, Option_t *option) const
{
   //*-*-*-*-*Project a 2-D histogram into a profile histogram along X*-*-*-*-*-*
   //*-*      ========================================================
   //
   //   The projection is made from the channels along the Y axis
   //   ranging from firstybin to lastybin included.
   //   By default, bins 1 to ny are included
   //   When all bins are included, the number of entries in the projection
   //   is set to the number of entries of the 2-D histogram, otherwise
   //   the number of entries is incremented by 1 for all non empty cells.
   //
   //   if option "d" is specified, the profile is drawn in the current pad.
   //
   //   Using a TCutG object, it is possible to select a sub-range of a 2-D histogram.
   //   One must create a graphical cut (mouse or C++) and specify the name
   //   of the cut between [] in the option.
   //   For example, with a TCutG named "cutg", one can call:
   //      myhist->ProfileX(" ",firstybin,lastybin,"[cutg]");
   //   To invert the cut, it is enough to put a "-" in front of its name:
   //      myhist->ProfileX(" ",firstybin,lastybin,"[-cutg]");
   //   It is possible to apply several cuts ("," means logical AND):
   //      myhist->ProfileX(" ",firstybin,lastybin,[cutg1,cutg2]");
   //
   //   NOTE that if a TProfile named name exists in the current directory or pad,
   //   the histogram is reset and filled again with the current contents of the TH2.
   //   The X axis attributes of the TH2 are copied to the X axis of the profile.

   TString opt = option;
   Int_t nx = fXaxis.GetNbins();
   Int_t ny = fYaxis.GetNbins();
   if (firstybin < 0) firstybin = 1;
   if (lastybin  < 0) lastybin  = ny;
   if (lastybin  > ny+1) lastybin  = ny;

   // Create the profile histogram
   char *pname = (char*)name;
   if (name && strcmp(name,"_pfx") == 0) {
      Int_t nch = strlen(GetName()) + 5;
      pname = new char[nch];
      sprintf(pname,"%s%s",GetName(),name);
   }
   TProfile *h1=0;
   //check if a profile with identical name exist
   TObject *h1obj = gROOT->FindObject(pname);
   if (h1obj && h1obj->InheritsFrom("TProfile")) {
      h1 = (TProfile*)h1obj;
      h1->Reset();
   }

   Int_t ncuts = 0;
   if (opt.Contains("[")) {
      ((TH2 *)this)->GetPainter();
      if (fPainter) ncuts = fPainter->MakeCuts((char*)opt.Data());
   }
   opt.ToLower();  //must be called after MakeCuts

   if (!h1) {
      const TArrayD *bins = fXaxis.GetXbins();
      if (bins->fN == 0) {
         h1 = new TProfile(pname,GetTitle(),nx,fXaxis.GetXmin(),fXaxis.GetXmax(),option);
      } else {
         h1 = new TProfile(pname,GetTitle(),nx,bins->fArray,option);
      }
   }
   if (pname != name)  delete [] pname;

   // Copy attributes
   h1->GetXaxis()->ImportAttributes(this->GetXaxis());
   h1->SetLineColor(this->GetLineColor());
   h1->SetFillColor(this->GetFillColor());
   h1->SetMarkerColor(this->GetMarkerColor());
   h1->SetMarkerStyle(this->GetMarkerStyle());

   //
   // Fill the profile histogram
   Double_t cont, err;
   for (Int_t binx =0;binx<=nx+1;binx++) {
      for (Int_t biny=firstybin;biny<=lastybin;biny++) {
         if (ncuts) {
            if (!fPainter->IsInside(binx,biny)) continue;
         }
         cont =  GetCellContent(binx,biny);
         err = GetCellError(binx, biny);

         if (cont) {
            h1->Fill(fXaxis.GetBinCenter(binx),fYaxis.GetBinCenter(biny), err*err);
         }
      }
   }
   if ((firstybin <=1 && lastybin >= ny) && !ncuts) h1->SetEntries(fEntries);

   if (opt.Contains("d")) {
      TVirtualPad *padsav = gPad;
      TVirtualPad *pad = gROOT->GetSelectedPad();
      if (pad) pad->cd();
      char optin[100];
      strcpy(optin,opt.Data());
      char *d = (char*)strstr(optin,"d"); if (d) {*d = ' '; if (*(d+1) == 0) *d=0;}
      char *e = (char*)strstr(optin,"e"); if (e) {*e = ' '; if (*(e+1) == 0) *e=0;}
      if (!gPad->FindObject(h1)) {
         h1->Draw(optin);
      } else {
         h1->Paint(optin);
      }
      if (padsav) padsav->cd();
   }
   return h1;
}

//______________________________________________________________________________
TProfile *TH2::ProfileY(const char *name, Int_t firstxbin, Int_t lastxbin, Option_t *option) const
{
   //*-*-*-*-*Project a 2-D histogram into a profile histogram along Y*-*-*-*-*-*
   //*-*      ========================================================
   //
   //   The projection is made from the channels along the X axis
   //   ranging from firstxbin to lastxbin included.
   //   By default, bins 1 to nx are included
   //   When all bins are included, the number of entries in the projection
   //   is set to the number of entries of the 2-D histogram, otherwise
   //   the number of entries is incremented by 1 for all non empty cells.
   //
   //   if option "d" is specified, the profile is drawn in the current pad.
   //
   //   Using a TCutG object, it is possible to select a sub-range of a 2-D histogram.
   //   One must create a graphical cut (mouse or C++) and specify the name
   //   of the cut between [] in the option.
   //   For example, with a TCutG named "cutg", one can call:
   //      myhist->ProfileY(" ",firstybin,lastybin,"[cutg]");
   //   To invert the cut, it is enough to put a "-" in front of its name:
   //      myhist->ProfileY(" ",firstybin,lastybin,"[-cutg]");
   //   It is possible to apply several cuts:
   //      myhist->ProfileY(" ",firstybin,lastybin,[cutg1,cutg2]");
   //
   //   NOTE that if a TProfile named name exists in the current directory or pad,
   //   the histogram is reset and filled again with the current contents of the TH2.
   //   The Y axis attributes of the TH2 are copied to the X axis of the profile.

   TString opt = option;
   Int_t nx = fXaxis.GetNbins();
   Int_t ny = fYaxis.GetNbins();
   if (firstxbin < 0) firstxbin = 1;
   if (lastxbin  < 0) lastxbin  = nx;
   if (lastxbin  > nx+1) lastxbin  = nx;

   // Create the projection histogram
   char *pname = (char*)name;
   if (name && strcmp(name,"_pfy") == 0) {
      Int_t nch = strlen(GetName()) + 5;
      pname = new char[nch];
      sprintf(pname,"%s%s",GetName(),name);
   }
   TProfile *h1=0;
   //check if a profile with identical name exist
   TObject *h1obj = gROOT->FindObject(pname);
   if (h1obj && h1obj->InheritsFrom("TProfile")) {
      h1 = (TProfile*)h1obj;
      h1->Reset();
   }

   Int_t ncuts = 0;
   if (opt.Contains("[")) {
      ((TH2 *)this)->GetPainter();
      if (fPainter) ncuts = fPainter->MakeCuts((char*)opt.Data());
   }
   opt.ToLower();  //must be called after MakeCuts

   if (!h1) {
      const TArrayD *bins = fYaxis.GetXbins();
      if (bins->fN == 0) {
         h1 = new TProfile(pname,GetTitle(),ny,fYaxis.GetXmin(),fYaxis.GetXmax(),option);
      } else {
         h1 = new TProfile(pname,GetTitle(),ny,bins->fArray,option);
      }
   }
   if (pname != name)  delete [] pname;

   // Copy attributes
   h1->GetXaxis()->ImportAttributes(this->GetYaxis());
   h1->SetLineColor(this->GetLineColor());
   h1->SetFillColor(this->GetFillColor());
   h1->SetMarkerColor(this->GetMarkerColor());
   h1->SetMarkerStyle(this->GetMarkerStyle());

   // Fill the profile histogram
   Double_t cont, err;
   for (Int_t biny =0;biny<=ny+1;biny++) {
      for (Int_t binx=firstxbin;binx<=lastxbin;binx++) {
         if (ncuts) {
            if (!fPainter->IsInside(binx,biny)) continue;
         }
         cont =  GetCellContent(binx,biny);
         err = GetCellError(binx, biny);
         if (cont) {
            h1->Fill(fYaxis.GetBinCenter(biny),fXaxis.GetBinCenter(binx), err*err);
         }
      }
   }
   if ((firstxbin <=1 && lastxbin >= nx) && !ncuts) h1->SetEntries(fEntries);

   if (opt.Contains("d")) {
      TVirtualPad *padsav = gPad;
      TVirtualPad *pad = gROOT->GetSelectedPad();
      if (pad) pad->cd();
      char optin[100];
      strcpy(optin,opt.Data());
      char *d = (char*)strstr(optin,"d"); if (d) {*d = ' '; if (*(d+1) == 0) *d=0;}
      char *e = (char*)strstr(optin,"e"); if (e) {*e = ' '; if (*(e+1) == 0) *e=0;}
      if (!gPad->FindObject(h1)) {
         h1->Draw(optin);
      } else {
         h1->Paint(optin);
      }
      if (padsav) padsav->cd();
   }
   return h1;
}

//______________________________________________________________________________
TH1D *TH2::ProjectionX(const char *name, Int_t firstybin, Int_t lastybin, Option_t *option) const
{
   //*-*-*-*-*Project a 2-D histogram into a 1-D histogram along X*-*-*-*-*-*-*
   //*-*      ====================================================
   //
   //   The projection is always of the type TH1D.
   //   The projection is made from the channels along the Y axis
   //   ranging from firstybin to lastybin included.
   //   By default, bins 1 to ny are included
   //   When all bins are included, the number of entries in the projection
   //   is set to the number of entries of the 2-D histogram, otherwise
   //   the number of entries is incremented by 1 for all non empty cells.
   //
   //   To make the projection in X of the underflow bin in Y, use firstybin=lastybin=0;
   //   To make the projection in X of the overflow bin in Y, use firstybin=lastybin=ny+1;
   //
   //   if option "e" is specified, the errors are computed.
   //   if option "d" is specified, the projection is drawn in the current pad.
   //
   //   Using a TCutG object, it is possible to select a sub-range of a 2-D histogram.
   //   One must create a graphical cut (mouse or C++) and specify the name
   //   of the cut between [] in the option.
   //   For example, with a TCutG named "cutg", one can call:
   //      myhist->ProjectionX(" ",firstybin,lastybin,"[cutg]");
   //   To invert the cut, it is enough to put a "-" in front of its name:
   //      myhist->ProjectionX(" ",firstybin,lastybin,"[-cutg]");
   //   It is possible to apply several cuts:
   //      myhist->ProjectionX(" ",firstybin,lastybin,[cutg1,cutg2]");
   //
   //   NOTE that if a TH1D named name exists in the current directory or pad,
   //   the histogram is reset and filled again with the current contents of the TH2.
   //   The X axis attributes of the TH2 are copied to the X axis of the projection.

   TString opt = option;
   Int_t nx = fXaxis.GetNbins();
   Int_t ny = fYaxis.GetNbins();
   if (firstybin < 0) firstybin = 1;
   if (lastybin  < 0) lastybin  = ny;
   if (lastybin  > ny+1) lastybin  = ny;

   // Create the projection histogram
   char *pname = (char*)name;
   if (name && strcmp(name,"_px") == 0) {
      Int_t nch = strlen(GetName()) + 4;
      pname = new char[nch];
      sprintf(pname,"%s%s",GetName(),name);
   }
   TH1D *h1=0;
   //check if histogram with identical name exist
   TObject *h1obj = gROOT->FindObject(pname);
   if (h1obj && h1obj->InheritsFrom("TH1D")) {
      h1 = (TH1D*)h1obj;
      h1->Reset();
   }

   Int_t ncuts = 0;
   if (opt.Contains("[")) {
      ((TH2 *)this)->GetPainter();
      if (fPainter) ncuts = fPainter->MakeCuts((char*)opt.Data());
   }
   opt.ToLower();  //must be called after MakeCuts

   if (!h1) {
      const TArrayD *bins = fXaxis.GetXbins();
      if (bins->fN == 0) {
         h1 = new TH1D(pname,GetTitle(),nx,fXaxis.GetXmin(),fXaxis.GetXmax());
      } else {
         h1 = new TH1D(pname,GetTitle(),nx,bins->fArray);
      }
      if (opt.Contains("e")) h1->Sumw2();
   }
   if (pname != name)  delete [] pname;

   // Copy the axis attributes and the axis labels if needed.
   h1->GetXaxis()->ImportAttributes(this->GetXaxis());
   THashList* labels=GetXaxis()->GetLabels();
   if (labels) {
      TIter iL(labels);
      TObjString* lb;
      Int_t i = 1;
      while ((lb=(TObjString*)iL())) {
         h1->GetXaxis()->SetBinLabel(i,lb->String().Data());
         i++;
      }
   }
      
   h1->SetLineColor(this->GetLineColor());
   h1->SetFillColor(this->GetFillColor());
   h1->SetMarkerColor(this->GetMarkerColor());
   h1->SetMarkerStyle(this->GetMarkerStyle());

   // Fill the projected histogram
   Double_t cont,err,err2;
   for (Int_t binx =0;binx<=nx+1;binx++) {
      err2 = 0;
      for (Int_t biny=firstybin;biny<=lastybin;biny++) {
         if (ncuts) {
            if (!fPainter->IsInside(binx,biny)) continue;
         }
         cont  = GetCellContent(binx,biny);
         err   = GetCellError(binx,biny);
         err2 += err*err;
         if (cont) {
            h1->Fill(fXaxis.GetBinCenter(binx), cont);
         }
      }
      if (h1->GetSumw2N()) h1->SetBinError(binx,TMath::Sqrt(err2));
   }
   if ((firstybin <=1 && lastybin >= ny) && !ncuts) h1->SetEntries(fEntries);

   if (opt.Contains("d")) {
      TVirtualPad *padsav = gPad;
      TVirtualPad *pad = gROOT->GetSelectedPad();
      if (pad) pad->cd();
      char optin[100];
      strcpy(optin,opt.Data());
      char *d = (char*)strstr(optin,"d"); if (d) {*d = ' '; if (*(d+1) == 0) *d=0;}
      char *e = (char*)strstr(optin,"e"); if (e) {*e = ' '; if (*(e+1) == 0) *e=0;}
      if (!gPad->FindObject(h1)) {
         h1->Draw(optin);
      } else {
         h1->Paint(optin);
      }
      if (padsav) padsav->cd();
   }

   return h1;
}

//______________________________________________________________________________
TH1D *TH2::ProjectionY(const char *name, Int_t firstxbin, Int_t lastxbin, Option_t *option) const
{
   //*-*-*-*-*Project a 2-D histogram into a 1-D histogram along Y*-*-*-*-*-*-*
   //*-*      ====================================================
   //
   //   The projection is always of the type TH1D.
   //   The projection is made from the channels along the X axis
   //   ranging from firstxbin to lastxbin included.
   //   By default, bins 1 to nx are included
   //   When all bins are included, the number of entries in the projection
   //   is set to the number of entries of the 2-D histogram, otherwise
   //   the number of entries is incremented by 1 for all non empty cells.
   //
   //   To make the projection in Y of the underflow bin in X, use firstxbin=lastxbin=0;
   //   To make the projection in Y of the overflow bin in X, use firstxbin=lastxbin=nx+1;
   //
   //   if option "e" is specified, the errors are computed.
   //   if option "d" is specified, the projection is drawn in the current pad.
   //
   //   Using a TCutG object, it is possible to select a sub-range of a 2-D histogram.
   //   One must create a graphical cut (mouse or C++) and specify the name
   //   of the cut between [] in the option.
   //   For example, with a TCutG named "cutg", one can call:
   //      myhist->ProjectionY(" ",firstxbin,lastxbin,"[cutg]");
   //   To invert the cut, it is enough to put a "-" in front of its name:
   //      myhist->ProjectionY(" ",firstxbin,lastxbin,"[-cutg]");
   //   It is possible to apply several cuts:
   //      myhist->ProjectionY(" ",firstxbin,lastxbin,[cutg1,cutg2]");
   //
   //   NOTE that if a TH1D named name exists in the current directory or pad,
   //   the histogram is reset and filled again with the current contents of the TH2.
   //   The Y axis attributes of the TH2 are copied to the X axis of the projection.

   TString opt = option;
   Int_t nx = fXaxis.GetNbins();
   Int_t ny = fYaxis.GetNbins();
   if (firstxbin < 0) firstxbin = 1;
   if (lastxbin  < 0) lastxbin  = nx;
   if (lastxbin  > nx+1) lastxbin  = nx;

   // Create the projection histogram
   char *pname = (char*)name;
   if (name && strcmp(name,"_py") == 0) {
      Int_t nch = strlen(GetName()) + 4;
      pname = new char[nch];
      sprintf(pname,"%s%s",GetName(),name);
   }
   TH1D *h1=0;
   //check if histogram with identical name exist
   TObject *h1obj = gROOT->FindObject(pname);
   if (h1obj && h1obj->InheritsFrom("TH1D")) {
      h1 = (TH1D*)h1obj;
      h1->Reset();
   }

   Int_t ncuts = 0;
   if (opt.Contains("[")) {
      ((TH2 *)this)->GetPainter();
      if (fPainter) ncuts = fPainter->MakeCuts((char*)opt.Data());
   }
   opt.ToLower();  //must be called after MakeCuts

   if (!h1) {
      const TArrayD *bins = fYaxis.GetXbins();
      if (bins->fN == 0) {
         h1 = new TH1D(pname,GetTitle(),ny,fYaxis.GetXmin(),fYaxis.GetXmax());
      } else {
         h1 = new TH1D(pname,GetTitle(),ny,bins->fArray);
      }
      if (opt.Contains("e")) h1->Sumw2();
   }
   if (pname != name)  delete [] pname;

   // Copy the axis attributes and the axis labels if needed.
   h1->GetXaxis()->ImportAttributes(this->GetYaxis());
   THashList* labels=GetYaxis()->GetLabels();
   if (labels) {
      TIter iL(labels);
      TObjString* lb;
      Int_t i = 1;
      while ((lb=(TObjString*)iL())) {
         h1->GetXaxis()->SetBinLabel(i,lb->String().Data());
         i++;
      }
   }
      
   h1->SetLineColor(this->GetLineColor());
   h1->SetFillColor(this->GetFillColor());
   h1->SetMarkerColor(this->GetMarkerColor());
   h1->SetMarkerStyle(this->GetMarkerStyle());

   // Fill the projected histogram
   Double_t cont,err,err2;
   for (Int_t biny =0;biny<=ny+1;biny++) {
      err2 = 0;
      for (Int_t binx=firstxbin;binx<=lastxbin;binx++) {
         if (ncuts) {
            if (!fPainter->IsInside(binx,biny)) continue;
         }
         cont  = GetCellContent(binx,biny);
         err   = GetCellError(binx,biny);
         err2 += err*err;
         if (cont) {
            h1->Fill(fYaxis.GetBinCenter(biny), cont);
         }
      }
      if (h1->GetSumw2N()) h1->SetBinError(biny,TMath::Sqrt(err2));
   }
   if ((firstxbin <=1 && lastxbin >= nx) && !ncuts) h1->SetEntries(fEntries);

   if (opt.Contains("d")) {
      TVirtualPad *padsav = gPad;
      TVirtualPad *pad = gROOT->GetSelectedPad();
      if (pad) pad->cd();
      char optin[100];
      strcpy(optin,opt.Data());
      char *d = (char*)strstr(optin,"d"); if (d) {*d = ' '; if (*(d+1) == 0) *d=0;}
      char *e = (char*)strstr(optin,"e"); if (e) {*e = ' '; if (*(e+1) == 0) *e=0;}
      if (!gPad->FindObject(h1)) {
         h1->Draw(optin);
      } else {
         h1->Paint(optin);
      }
      if (padsav) padsav->cd();
   }

   return h1;
}

//______________________________________________________________________________
void TH2::PutStats(Double_t *stats)
{
   // Replace current statistics with the values in array stats

   TH1::PutStats(stats);
   fTsumwy  = stats[4];
   fTsumwy2 = stats[5];
   fTsumwxy = stats[6];
}

//______________________________________________________________________________
void TH2::Reset(Option_t *option)
{
   //*-*-*-*-*-*-*-*Reset this histogram: contents, errors, etc*-*-*-*-*-*-*-*
   //*-*            ===========================================

   TH1::Reset(option);
   TString opt = option;
   opt.ToUpper();
   if (opt.Contains("ICE")) return;
   fTsumwy  = 0;
   fTsumwy2 = 0;
   fTsumwxy = 0;
}

//______________________________________________________________________________
void TH2::SetShowProjectionX(Int_t nbins)
{
   // When the mouse is moved in a pad containing a 2-d view of this histogram
   // a second canvas shows the projection along X corresponding to the
   // mouse position along Y.
   // To stop the generation of the projections, delete the canvas
   // containing the projection.

   GetPainter();

   if (fPainter) fPainter->SetShowProjection("x",nbins);
}


//______________________________________________________________________________
void TH2::SetShowProjectionY(Int_t nbins)
{
   // When the mouse is moved in a pad containing a 2-d view of this histogram
   // a second canvas shows the projection along Y corresponding to the
   // mouse position along X.
   // To stop the generation of the projections, delete the canvas
   // containing the projection.

   GetPainter();

   if (fPainter) fPainter->SetShowProjection("y",nbins);
}

//______________________________________________________________________________
TH1 *TH2::ShowBackground(Int_t niter, Option_t *option)
{
//   This function calculates the background spectrum in this histogram.
//   The background is returned as a histogram. 
//   to be implemented (may be)
   
   
   return (TH1*)gROOT->ProcessLineFast(Form("TSpectrum2::StaticBackground((TH1*)0x%x,%d,\"%s\")",this,niter,option));
}

//______________________________________________________________________________
Int_t TH2::ShowPeaks(Double_t sigma, Option_t *option, Double_t threshold)
{
   //Interface to TSpectrum2::Search
   //the function finds peaks in this histogram where the width is > sigma
   //and the peak maximum greater than threshold*maximum bin content of this.
   //for more detauils see TSpectrum::Search.
   //note the difference in the default value for option compared to TSpectrum2::Search
   //option="" by default (instead of "goff")
   
   
   return (Int_t)gROOT->ProcessLineFast(Form("TSpectrum2::StaticSearch((TH1*)0x%x,%g,\"%s\",%g)",this,sigma,option,threshold));
}
   

//______________________________________________________________________________
void TH2::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH2.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TH2::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TH1::Streamer(R__b);
      R__b >> fScalefactor;
      R__b >> fTsumwy;
      R__b >> fTsumwy2;
      R__b >> fTsumwxy;
      //====end of old versions

   } else {
      TH2::Class()->WriteBuffer(R__b,this);
   }
}

ClassImp(TH2C)

//______________________________________________________________________________
//                     TH2C methods
//______________________________________________________________________________
TH2C::TH2C(): TH2()
{
   // Constructor.
}

//______________________________________________________________________________
TH2C::~TH2C()
{
   // Destructor.
}

//______________________________________________________________________________
TH2C::TH2C(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
           ,Int_t nbinsy,Double_t ylow,Double_t yup)
           :TH2(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup)
{
   // Constructor.

   TArrayC::Set(fNcells);

   //if (xlow >= xup || ylow >= yup) SetBuffer(fgBufferSize);
}

//______________________________________________________________________________
TH2C::TH2C(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
           ,Int_t nbinsy,Double_t ylow,Double_t yup)
           :TH2(name,title,nbinsx,xbins,nbinsy,ylow,yup)
{
   // Constructor.

   TArrayC::Set(fNcells);
}

//______________________________________________________________________________
TH2C::TH2C(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
           ,Int_t nbinsy,const Double_t *ybins)
           :TH2(name,title,nbinsx,xlow,xup,nbinsy,ybins)
{
   // Constructor.

   TArrayC::Set(fNcells);
}

//______________________________________________________________________________
TH2C::TH2C(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
           ,Int_t nbinsy,const Double_t *ybins)
           :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   // Constructor.

   TArrayC::Set(fNcells);
}

//______________________________________________________________________________
TH2C::TH2C(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
           ,Int_t nbinsy,const Float_t *ybins)
           :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   // Constructor.

   TArrayC::Set(fNcells);
}

//______________________________________________________________________________
TH2C::TH2C(const TH2C &h2c) : TH2(), TArrayC()
{
   // Copy constructor.

   ((TH2C&)h2c).Copy(*this);
}

//______________________________________________________________________________
void TH2C::AddBinContent(Int_t bin)
{
   //*-*-*-*-*-*-*-*-*-*Increment bin content by 1*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                ==========================

   if (fArray[bin] < 127) fArray[bin]++;
}

//______________________________________________________________________________
void TH2C::AddBinContent(Int_t bin, Double_t w)
{
   //*-*-*-*-*-*-*-*-*-*Increment bin content by w*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                ==========================

   Int_t newval = fArray[bin] + Int_t(w);
   if (newval > -128 && newval < 128) {fArray[bin] = Char_t(newval); return;}
   if (newval < -127) fArray[bin] = -127;
   if (newval >  127) fArray[bin] =  127;
}

//______________________________________________________________________________
void TH2C::Copy(TObject &newth2) const
{
   // Copy.

   TH2::Copy((TH2C&)newth2);
   TArrayC::Copy((TH2C&)newth2);
}

//______________________________________________________________________________
TH1 *TH2C::DrawCopy(Option_t *option) const
{
   // Draw copy.

   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH2C *newth2 = (TH2C*)Clone();
   newth2->SetDirectory(0);
   newth2->SetBit(kCanDelete);
   newth2->AppendPad(option);
   return newth2;
}

//______________________________________________________________________________
Double_t TH2C::GetBinContent(Int_t bin) const
{
   // Get bin content.

   if (fBuffer) ((TH2C*)this)->BufferEmpty();
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   if (!fArray) return 0;
   return Double_t (fArray[bin]);
}

//______________________________________________________________________________
void TH2C::Reset(Option_t *option)
{
   //*-*-*-*-*-*-*-*Reset this histogram: contents, errors, etc*-*-*-*-*-*-*-*
   //*-*            ===========================================

   TH2::Reset(option);
   TArrayC::Reset();
}

//______________________________________________________________________________
void TH2C::SetBinContent(Int_t bin, Double_t content)
{
   // Set bin content
   if (bin < 0) return;
   if (bin >= fNcells) return;
   fArray[bin] = Char_t (content);
   fEntries++;
}

//______________________________________________________________________________
void TH2C::SetBinsLength(Int_t n)
{
   // Set total number of bins including under/overflow
   // Reallocate bin contents array

   if (n < 0) n = (fXaxis.GetNbins()+2)*(fYaxis.GetNbins()+2);
   fNcells = n;
   TArrayC::Set(n);
}

//______________________________________________________________________________
void TH2C::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH2C.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TH2C::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      if (R__v < 2) {
         R__b.ReadVersion();
         TH1::Streamer(R__b);
         TArrayC::Streamer(R__b);
         R__b.ReadVersion();
         R__b >> fScalefactor;
         R__b >> fTsumwy;
         R__b >> fTsumwy2;
         R__b >> fTsumwxy;
      } else {
         TH2::Streamer(R__b);
         TArrayC::Streamer(R__b);
         R__b.CheckByteCount(R__s, R__c, TH2C::IsA());
      }
      //====end of old versions

   } else {
      TH2C::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
TH2C& TH2C::operator=(const TH2C &h1)
{
   // Operator =

   if (this != &h1)  ((TH2C&)h1).Copy(*this);
   return *this;
}


//______________________________________________________________________________
TH2C operator*(Float_t c1, TH2C &h1)
{
   // Operator *

   TH2C hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2C operator+(TH2C &h1, TH2C &h2)
{
   // Operator +

   TH2C hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2C operator-(TH2C &h1, TH2C &h2)
{
   // Operator -

   TH2C hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2C operator*(TH2C &h1, TH2C &h2)
{
   // Operator *

   TH2C hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2C operator/(TH2C &h1, TH2C &h2)
{
   // Operator /

   TH2C hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

ClassImp(TH2S)

//______________________________________________________________________________
//                     TH2S methods
//______________________________________________________________________________
TH2S::TH2S(): TH2()
{
   // Constructor.
}

//______________________________________________________________________________
TH2S::~TH2S()
{
   // Destructor.
}

//______________________________________________________________________________
TH2S::TH2S(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
           ,Int_t nbinsy,Double_t ylow,Double_t yup)
           :TH2(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup)
{
   // Constructor.

   TArrayS::Set(fNcells);

   //if (xlow >= xup || ylow >= yup) SetBuffer(fgBufferSize);
}

//______________________________________________________________________________
TH2S::TH2S(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
           ,Int_t nbinsy,Double_t ylow,Double_t yup)
           :TH2(name,title,nbinsx,xbins,nbinsy,ylow,yup)
{
   // Constructor.

   TArrayS::Set(fNcells);
}

//______________________________________________________________________________
TH2S::TH2S(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
           ,Int_t nbinsy,const Double_t *ybins)
           :TH2(name,title,nbinsx,xlow,xup,nbinsy,ybins)
{
   // Constructor.

   TArrayS::Set(fNcells);
}

//______________________________________________________________________________
TH2S::TH2S(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
           ,Int_t nbinsy,const Double_t *ybins)
           :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   // Constructor.

   TArrayS::Set(fNcells);
}

//______________________________________________________________________________
TH2S::TH2S(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
           ,Int_t nbinsy,const Float_t *ybins)
           :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   // Constructor.

   TArrayS::Set(fNcells);
}

//______________________________________________________________________________
TH2S::TH2S(const TH2S &h2s) : TH2(), TArrayS()
{
   // Copy constructor.

   ((TH2S&)h2s).Copy(*this);
}

//______________________________________________________________________________
void TH2S::AddBinContent(Int_t bin)
{
   //*-*-*-*-*-*-*-*-*-*Increment bin content by 1*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                ==========================

   if (fArray[bin] < 32767) fArray[bin]++;
}

//______________________________________________________________________________
void TH2S::AddBinContent(Int_t bin, Double_t w)
{
   //*-*-*-*-*-*-*-*-*-*Increment bin content by w*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                ==========================

   Int_t newval = fArray[bin] + Int_t(w);
   if (newval > -32768 && newval < 32768) {fArray[bin] = Short_t(newval); return;}
   if (newval < -32767) fArray[bin] = -32767;
   if (newval >  32767) fArray[bin] =  32767;
}

//______________________________________________________________________________
void TH2S::Copy(TObject &newth2) const
{
   // Copy.
   
   TH2::Copy((TH2S&)newth2);
   TArrayS::Copy((TH2S&)newth2);
}

//______________________________________________________________________________
TH1 *TH2S::DrawCopy(Option_t *option) const
{
   // Draw copy.

   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH2S *newth2 = (TH2S*)Clone();
   newth2->SetDirectory(0);
   newth2->SetBit(kCanDelete);
   newth2->AppendPad(option);
   return newth2;
}

//______________________________________________________________________________
Double_t TH2S::GetBinContent(Int_t bin) const
{
   // Get bin content.

   if (fBuffer) ((TH2C*)this)->BufferEmpty();
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   if (!fArray) return 0;
   return Double_t (fArray[bin]);
}

//______________________________________________________________________________
void TH2S::Reset(Option_t *option)
{
   //*-*-*-*-*-*-*-*Reset this histogram: contents, errors, etc*-*-*-*-*-*-*-*
   //*-*            ===========================================

   TH2::Reset(option);
   TArrayS::Reset();
}

//______________________________________________________________________________
void TH2S::SetBinContent(Int_t bin, Double_t content)
{
   // Set bin content
   if (bin < 0) return;
   if (bin >= fNcells) return;
   fArray[bin] = Short_t (content);
   fEntries++;
}

//______________________________________________________________________________
void TH2S::SetBinsLength(Int_t n)
{
   // Set total number of bins including under/overflow
   // Reallocate bin contents array

   if (n < 0) n = (fXaxis.GetNbins()+2)*(fYaxis.GetNbins()+2);
   fNcells = n;
   TArrayS::Set(n);
}

//______________________________________________________________________________
void TH2S::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH2S.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TH2S::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      if (R__v < 2) {
         R__b.ReadVersion();
         TH1::Streamer(R__b);
         TArrayS::Streamer(R__b);
         R__b.ReadVersion();
         R__b >> fScalefactor;
         R__b >> fTsumwy;
         R__b >> fTsumwy2;
         R__b >> fTsumwxy;
      } else {
         TH2::Streamer(R__b);
         TArrayS::Streamer(R__b);
         R__b.CheckByteCount(R__s, R__c, TH2S::IsA());
      }
      //====end of old versions

   } else {
      TH2S::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
TH2S& TH2S::operator=(const TH2S &h1)
{
   // Operator =

   if (this != &h1)  ((TH2S&)h1).Copy(*this);
   return *this;
}


//______________________________________________________________________________
TH2S operator*(Float_t c1, TH2S &h1)
{
   // Operator *

   TH2S hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2S operator+(TH2S &h1, TH2S &h2)
{
   // Operator +

   TH2S hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2S operator-(TH2S &h1, TH2S &h2)
{
   // Operator -

   TH2S hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2S operator*(TH2S &h1, TH2S &h2)
{
   // Operator *

   TH2S hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2S operator/(TH2S &h1, TH2S &h2)
{
   // Operator /

   TH2S hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

ClassImp(TH2I)

//______________________________________________________________________________
//                     TH2I methods
//______________________________________________________________________________
TH2I::TH2I(): TH2()
{
   // Constructor.
}

//______________________________________________________________________________
TH2I::~TH2I()
{
   // Destructor.
}

//______________________________________________________________________________
TH2I::TH2I(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
           ,Int_t nbinsy,Double_t ylow,Double_t yup)
           :TH2(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup)
{
   // Constructor.

   TArrayI::Set(fNcells);

   //if (xlow >= xup || ylow >= yup) SetBuffer(fgBufferSize);
}

//______________________________________________________________________________
TH2I::TH2I(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
           ,Int_t nbinsy,Double_t ylow,Double_t yup)
           :TH2(name,title,nbinsx,xbins,nbinsy,ylow,yup)
{
   // Constructor.

   TArrayI::Set(fNcells);
}

//______________________________________________________________________________
TH2I::TH2I(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
           ,Int_t nbinsy,const Double_t *ybins)
           :TH2(name,title,nbinsx,xlow,xup,nbinsy,ybins)
{
   // Constructor.

   TArrayI::Set(fNcells);
}

//______________________________________________________________________________
TH2I::TH2I(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
           ,Int_t nbinsy,const Double_t *ybins)
           :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   // Constructor.

   TArrayI::Set(fNcells);
}

//______________________________________________________________________________
TH2I::TH2I(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
           ,Int_t nbinsy,const Float_t *ybins)
           :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   // Constructor.

   TArrayI::Set(fNcells);
}

//______________________________________________________________________________
TH2I::TH2I(const TH2I &h2i) : TH2(), TArrayI()
{
   // Copy constructor.

   ((TH2I&)h2i).Copy(*this);
}

//______________________________________________________________________________
void TH2I::AddBinContent(Int_t bin)
{
   //*-*-*-*-*-*-*-*-*-*Increment bin content by 1*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                ==========================

   if (fArray[bin] < 2147483647) fArray[bin]++;
}

//______________________________________________________________________________
void TH2I::AddBinContent(Int_t bin, Double_t w)
{
   //*-*-*-*-*-*-*-*-*-*Increment bin content by w*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                ==========================

   Int_t newval = fArray[bin] + Int_t(w);
   if (newval > -2147483647 && newval < 2147483647) {fArray[bin] = Int_t(newval); return;}
   if (newval < -2147483647) fArray[bin] = -2147483647;
   if (newval >  2147483647) fArray[bin] =  2147483647;
}

//______________________________________________________________________________
void TH2I::Copy(TObject &newth2) const
{
   // Copy.

   TH2::Copy((TH2I&)newth2);
   TArrayI::Copy((TH2I&)newth2);
}

//______________________________________________________________________________
TH1 *TH2I::DrawCopy(Option_t *option) const
{
   // Draw copy.

   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH2I *newth2 = (TH2I*)Clone();
   newth2->SetDirectory(0);
   newth2->SetBit(kCanDelete);
   newth2->AppendPad(option);
   return newth2;
}

//______________________________________________________________________________
Double_t TH2I::GetBinContent(Int_t bin) const
{
   // Get bin content.

   if (fBuffer) ((TH2C*)this)->BufferEmpty();
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   if (!fArray) return 0;
   return Double_t (fArray[bin]);
}

//______________________________________________________________________________
void TH2I::Reset(Option_t *option)
{
   //*-*-*-*-*-*-*-*Reset this histogram: contents, errors, etc*-*-*-*-*-*-*-*
   //*-*            ===========================================

   TH2::Reset(option);
   TArrayI::Reset();
}

//______________________________________________________________________________
void TH2I::SetBinContent(Int_t bin, Double_t content)
{
   // Set bin content
   if (bin < 0) return;
   if (bin >= fNcells) return;
   fArray[bin] = Int_t (content);
   fEntries++;
}

//______________________________________________________________________________
void TH2I::SetBinsLength(Int_t n)
{
   // Set total number of bins including under/overflow
   // Reallocate bin contents array

   if (n < 0) n = (fXaxis.GetNbins()+2)*(fYaxis.GetNbins()+2);
   fNcells = n;
   TArrayI::Set(n);
}

//______________________________________________________________________________
TH2I& TH2I::operator=(const TH2I &h1)
{
   // Operator =

   if (this != &h1)  ((TH2I&)h1).Copy(*this);
   return *this;
}


//______________________________________________________________________________
TH2I operator*(Float_t c1, TH2I &h1)
{
   // Operator *

   TH2I hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2I operator+(TH2I &h1, TH2I &h2)
{
   // Operator +

   TH2I hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2I operator-(TH2I &h1, TH2I &h2)
{
   // Operator -

   TH2I hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2I operator*(TH2I &h1, TH2I &h2)
{
   // Operator *

   TH2I hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2I operator/(TH2I &h1, TH2I &h2)
{
   // Operator /

   TH2I hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

ClassImp(TH2F)

//______________________________________________________________________________
//                     TH2F methods
//______________________________________________________________________________
TH2F::TH2F(): TH2()
{
   // Constructor.
}

//______________________________________________________________________________
TH2F::~TH2F()
{
   // Destructor.
}

//______________________________________________________________________________
TH2F::TH2F(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
           ,Int_t nbinsy,Double_t ylow,Double_t yup)
           :TH2(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup)
{
   // Constructor.

   TArrayF::Set(fNcells);

   //if (xlow >= xup || ylow >= yup) SetBuffer(fgBufferSize);
}

//______________________________________________________________________________
TH2F::TH2F(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
           ,Int_t nbinsy,Double_t ylow,Double_t yup)
           :TH2(name,title,nbinsx,xbins,nbinsy,ylow,yup)
{
   // Constructor.

   TArrayF::Set(fNcells);
}

//______________________________________________________________________________
TH2F::TH2F(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
           ,Int_t nbinsy,const Double_t *ybins)
           :TH2(name,title,nbinsx,xlow,xup,nbinsy,ybins)
{
   // Constructor.

   TArrayF::Set(fNcells);
}

//______________________________________________________________________________
TH2F::TH2F(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
           ,Int_t nbinsy,const Double_t *ybins)
           :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   // Constructor.

   TArrayF::Set(fNcells);
}

//______________________________________________________________________________
TH2F::TH2F(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
           ,Int_t nbinsy,const Float_t *ybins)
           :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   // Constructor.

   TArrayF::Set(fNcells);
}

//______________________________________________________________________________
TH2F::TH2F(const TMatrixFBase &m)
:TH2("TMatrixFBase","",m.GetNcols(),m.GetColLwb(),1+m.GetColUpb(),m.GetNrows(),m.GetRowLwb(),1+m.GetRowUpb())
{
   // Constructor.

   TArrayF::Set(fNcells);
   Int_t ilow = m.GetRowLwb();
   Int_t iup  = m.GetRowUpb();
   Int_t jlow = m.GetColLwb();
   Int_t jup  = m.GetColUpb();
   for (Int_t i=ilow;i<=iup;i++) {
      for (Int_t j=jlow;j<=jup;j++) {
         SetCellContent(j-jlow+1,i-ilow+1,m(i,j));
      }
   }
}

//______________________________________________________________________________
TH2F::TH2F(const TH2F &h2f) : TH2(), TArrayF()
{
   // Copy constructor.

   ((TH2F&)h2f).Copy(*this);
}

//______________________________________________________________________________
void TH2F::Copy(TObject &newth2) const
{
   // Copy.

   TH2::Copy((TH2F&)newth2);
   TArrayF::Copy((TH2F&)newth2);
}

//______________________________________________________________________________
TH1 *TH2F::DrawCopy(Option_t *option) const
{
   // Draw copy.

   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH2F *newth2 = (TH2F*)Clone();
   newth2->SetDirectory(0);
   newth2->SetBit(kCanDelete);
   newth2->AppendPad(option);
   return newth2;
}

//______________________________________________________________________________
Double_t TH2F::GetBinContent(Int_t bin) const
{
   // Get bin content.

   if (fBuffer) ((TH2C*)this)->BufferEmpty();
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   if (!fArray) return 0;
   return Double_t (fArray[bin]);
}

//______________________________________________________________________________
void TH2F::Reset(Option_t *option)
{
   //*-*-*-*-*-*-*-*Reset this histogram: contents, errors, etc*-*-*-*-*-*-*-*
   //*-*            ===========================================

   TH2::Reset(option);
   TArrayF::Reset();
}

//______________________________________________________________________________
void TH2F::SetBinContent(Int_t bin, Double_t content)
{
   // Set bin content
   if (bin < 0) return;
   if (bin >= fNcells) return;
   fArray[bin] = Float_t (content);
   fEntries++;
}

//______________________________________________________________________________
void TH2F::SetBinsLength(Int_t n)
{
   // Set total number of bins including under/overflow
   // Reallocate bin contents array

   if (n < 0) n = (fXaxis.GetNbins()+2)*(fYaxis.GetNbins()+2);
   fNcells = n;
   TArrayF::Set(n);
}

//______________________________________________________________________________
void TH2F::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH2F.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TH2F::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      if (R__v < 2) {
         R__b.ReadVersion();
         TH1::Streamer(R__b);
         TArrayF::Streamer(R__b);
         R__b.ReadVersion();
         R__b >> fScalefactor;
         R__b >> fTsumwy;
         R__b >> fTsumwy2;
         R__b >> fTsumwxy;
      } else {
         TH2::Streamer(R__b);
         TArrayF::Streamer(R__b);
         R__b.CheckByteCount(R__s, R__c, TH2F::IsA());
      }
      //====end of old versions

   } else {
      TH2F::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
TH2F& TH2F::operator=(const TH2F &h1)
{
   // Operator =

   if (this != &h1)  ((TH2F&)h1).Copy(*this);
   return *this;
}


//______________________________________________________________________________
TH2F operator*(Float_t c1, TH2F &h1)
{
   // Operator *

   TH2F hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}


//______________________________________________________________________________
TH2F operator*(TH2F &h1, Float_t c1)
{
   // Operator *

   TH2F hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2F operator+(TH2F &h1, TH2F &h2)
{
   // Operator +

   TH2F hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2F operator-(TH2F &h1, TH2F &h2)
{
   // Operator -

   TH2F hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2F operator*(TH2F &h1, TH2F &h2)
{
   // Operator *

   TH2F hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2F operator/(TH2F &h1, TH2F &h2)
{
   // Operator /

   TH2F hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

ClassImp(TH2D)

//______________________________________________________________________________
//                     TH2D methods
//______________________________________________________________________________
TH2D::TH2D(): TH2()
{
   // Constructor.
}

//______________________________________________________________________________
TH2D::~TH2D()
{
   // Destructor.
}

//______________________________________________________________________________
TH2D::TH2D(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
           ,Int_t nbinsy,Double_t ylow,Double_t yup)
           :TH2(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup)
{
   // Constructor.

   TArrayD::Set(fNcells);

   //if (xlow >= xup || ylow >= yup) SetBuffer(fgBufferSize);
}

//______________________________________________________________________________
TH2D::TH2D(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
           ,Int_t nbinsy,Double_t ylow,Double_t yup)
           :TH2(name,title,nbinsx,xbins,nbinsy,ylow,yup)
{
   // Constructor.

   TArrayD::Set(fNcells);
}

//______________________________________________________________________________
TH2D::TH2D(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
           ,Int_t nbinsy,const Double_t *ybins)
           :TH2(name,title,nbinsx,xlow,xup,nbinsy,ybins)
{
   // Constructor.

   TArrayD::Set(fNcells);
}

//______________________________________________________________________________
TH2D::TH2D(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
           ,Int_t nbinsy,const Double_t *ybins)
           :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   // Constructor.

   TArrayD::Set(fNcells);
}

//______________________________________________________________________________
TH2D::TH2D(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
           ,Int_t nbinsy,const Float_t *ybins)
           :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   // Constructor.

   TArrayD::Set(fNcells);
}

//______________________________________________________________________________
TH2D::TH2D(const TMatrixDBase &m)
:TH2("TMatrixDBase","",m.GetNcols(),m.GetColLwb(),1+m.GetColUpb(),m.GetNrows(),m.GetRowLwb(),1+m.GetRowUpb())
{
   // Constructor.

   TArrayD::Set(fNcells);
   Int_t ilow = m.GetRowLwb();
   Int_t iup  = m.GetRowUpb();
   Int_t jlow = m.GetColLwb();
   Int_t jup  = m.GetColUpb();
   for (Int_t i=ilow;i<=iup;i++) {
      for (Int_t j=jlow;j<=jup;j++) {
         SetCellContent(j-jlow+1,i-ilow+1,m(i,j));
      }
   }
}

//______________________________________________________________________________
TH2D::TH2D(const TH2D &h2d) : TH2(), TArrayD()
{
   // Copy constructor.

   ((TH2D&)h2d).Copy(*this);
}

//______________________________________________________________________________
void TH2D::Copy(TObject &newth2) const
{
   // Copy.

   TH2::Copy((TH2D&)newth2);
   TArrayD::Copy((TH2D&)newth2);
}

//______________________________________________________________________________
TH1 *TH2D::DrawCopy(Option_t *option) const
{
   // Draw copy.

   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH2D *newth2 = (TH2D*)Clone();
   newth2->SetDirectory(0);
   newth2->SetBit(kCanDelete);
   newth2->AppendPad(option);
   return newth2;
}

//______________________________________________________________________________
Double_t TH2D::GetBinContent(Int_t bin) const
{
   // Get bin content.

   if (fBuffer) ((TH2C*)this)->BufferEmpty();
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   if (!fArray) return 0;
   return Double_t (fArray[bin]);
}

//______________________________________________________________________________
void TH2D::Reset(Option_t *option)
{
   //*-*-*-*-*-*-*-*Reset this histogram: contents, errors, etc*-*-*-*-*-*-*-*
   //*-*            ===========================================

   TH2::Reset(option);
   TArrayD::Reset();
}

//______________________________________________________________________________
void TH2D::SetBinContent(Int_t bin, Double_t content)
{
   // Set bin content
   if (bin < 0) return;
   if (bin >= fNcells) return;
   fArray[bin] = Double_t (content);
   fEntries++;
}

//______________________________________________________________________________
void TH2D::SetBinsLength(Int_t n)
{
   // Set total number of bins including under/overflow
   // Reallocate bin contents array

   if (n < 0) n = (fXaxis.GetNbins()+2)*(fYaxis.GetNbins()+2);
   fNcells = n;
   TArrayD::Set(n);
}

//______________________________________________________________________________
void TH2D::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH2D.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TH2D::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      if (R__v < 2) {
         R__b.ReadVersion();
         TH1::Streamer(R__b);
         TArrayD::Streamer(R__b);
         R__b.ReadVersion();
         R__b >> fScalefactor;
         R__b >> fTsumwy;
         R__b >> fTsumwy2;
         R__b >> fTsumwxy;
      } else {
         TH2::Streamer(R__b);
         TArrayD::Streamer(R__b);
         R__b.CheckByteCount(R__s, R__c, TH2D::IsA());
      }
      //====end of old versions

   } else {
      TH2D::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
TH2D& TH2D::operator=(const TH2D &h1)
{
   // Operator =

   if (this != &h1)  ((TH2D&)h1).Copy(*this);
   return *this;
}


//______________________________________________________________________________
TH2D operator*(Float_t c1, TH2D &h1)
{
   // Operator *

   TH2D hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2D operator+(TH2D &h1, TH2D &h2)
{
   // Operator +

   TH2D hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2D operator-(TH2D &h1, TH2D &h2)
{
   // Operator -

   TH2D hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2D operator*(TH2D &h1, TH2D &h2)
{
   // Operator *

   TH2D hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2D operator/(TH2D &h1, TH2D &h2)
{
   // Operator /

   TH2D hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}
