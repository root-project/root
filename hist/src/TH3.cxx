// @(#)root/hist:$Name:  $:$Id: TH3.cxx,v 1.20 2002/02/25 16:05:37 brun Exp $
// Author: Rene Brun   27/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TH3.h"
#include "TH2.h"
#include "TF1.h"
#include "TVirtualPad.h"
#include "THLimitsFinder.h"
#include "TRandom.h"
#include "TFile.h"

ClassImp(TH3)

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-*  The 3-D histogram classes derived from the 1-D histogram classes.
//*-*  all operations are supported (fill, fit).
//*-*  Drawing is currently restricted to one single option.
//*-*  A cloud of points is drawn. The number of points is proportional to
//*-*  cell content.
//*-*
//
//  TH3C a 3-D histogram with one byte per cell (char)
//  TH3S a 3-D histogram with two bytes per cell (short integer)
//  TH3F a 3-D histogram with four bytes per cell (float)
//  TH3D a 3-D histogram with eight bytes per cell (double)
//
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//______________________________________________________________________________
TH3::TH3()
{
   fDimension   = 3;
}

//______________________________________________________________________________
TH3::TH3(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup
                                     ,Int_t nbinsz,Axis_t zlow,Axis_t zup)
     :TH1(name,title,nbinsx,xlow,xup),
      TAtt3D()
{
//*-*-*-*-*-*-*-*-*Normal constructor for fix bin size 3-D histograms*-*-*-*-*
//*-*              ==================================================

   fDimension   = 3;
   if (nbinsy <= 0) nbinsy = 1;
   if (nbinsz <= 0) nbinsz = 1;
   fYaxis.Set(nbinsy,ylow,yup);
   fZaxis.Set(nbinsz,zlow,zup);
   fNcells      = (nbinsx+2)*(nbinsy+2)*(nbinsz+2);
}

//______________________________________________________________________________
TH3::TH3(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                           ,Int_t nbinsy,const Float_t *ybins
                                           ,Int_t nbinsz,const Float_t *zbins)
     :TH1(name,title,nbinsx,xbins),
      TAtt3D()
{
//*-*-*-*-*-*-*-*Normal constructor for variable bin size 3-D histograms*-*-*-*
//*-*            =======================================================
   fDimension   = 3;
   if (nbinsy <= 0) nbinsy = 1;
   if (nbinsz <= 0) nbinsz = 1;
   if (ybins) fYaxis.Set(nbinsy,ybins);
   else       fYaxis.Set(nbinsy,0,1);
   if (zbins) fZaxis.Set(nbinsz,zbins);
   else       fZaxis.Set(nbinsz,0,1);
   fNcells      = (nbinsx+2)*(nbinsy+2)*(nbinsz+2);
}

//______________________________________________________________________________
TH3::TH3(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                           ,Int_t nbinsy,const Double_t *ybins
                                           ,Int_t nbinsz,const Double_t *zbins)
     :TH1(name,title,nbinsx,xbins),
      TAtt3D()
{
//*-*-*-*-*-*-*-*Normal constructor for variable bin size 3-D histograms*-*-*-*
//*-*            =======================================================
   fDimension   = 3;
   if (nbinsy <= 0) nbinsy = 1;
   if (nbinsz <= 0) nbinsz = 1;
   if (ybins) fYaxis.Set(nbinsy,ybins);
   else       fYaxis.Set(nbinsy,0,1);
   if (zbins) fZaxis.Set(nbinsz,zbins);
   else       fZaxis.Set(nbinsz,0,1);
   fNcells      = (nbinsx+2)*(nbinsy+2)*(nbinsz+2);
}

//______________________________________________________________________________
TH3::~TH3()
{
}

//______________________________________________________________________________
void TH3::Copy(TObject &obj)
{
   TH1::Copy(obj);
}
      
//______________________________________________________________________________
Int_t TH3::BufferEmpty(Bool_t deleteBuffer)
{
// Fill histogram with all entries in the buffer.

   // do we need to compute the bin size?
   Int_t nbentries = (Int_t)fBuffer[0];
   if (!nbentries) return 0;
   if (fXaxis.GetXmax() <= fXaxis.GetXmin() ||
       fYaxis.GetXmax() <= fYaxis.GetXmin() ||
       fZaxis.GetXmax() <= fZaxis.GetXmin()) {
     //find min, max of entries in buffer
      Double_t xmin = fBuffer[2];
      Double_t xmax = xmin;
      Double_t ymin = fBuffer[3];
      Double_t ymax = ymin;
      Double_t zmin = fBuffer[4];
      Double_t zmax = zmin;
      for (Int_t i=1;i<nbentries;i++) {
         Double_t x = fBuffer[4*i+2];
         if (x < xmin) xmin = x;
         if (x > xmax) xmax = x;
         Double_t y = fBuffer[4*i+3];
         if (y < ymin) ymin = y;
         if (y > ymax) ymax = y;
         Double_t z = fBuffer[4*i+4];
         if (z < zmin) zmin = z;
         if (z > zmax) zmax = z;
     }
     THLimitsFinder::GetLimitsFinder()->FindGoodLimits(this,xmin,xmax,ymin,ymax,zmin,zmax);
   }
   Double_t *buffer = fBuffer; fBuffer = 0;
   
   for (Int_t i=0;i<nbentries;i++) {
      Fill(buffer[4*i+2],buffer[4*i+3],buffer[4*i+4],buffer[4*i+1]);
   }
   
   if (deleteBuffer) { delete buffer;    fBufferSize = 0;}
   else              { fBuffer = buffer; fBuffer[0] = 0;}
   return nbentries;
}
 
//______________________________________________________________________________
Int_t TH3::BufferFill(Axis_t x, Axis_t y, Axis_t z, Stat_t w)
{
// accumulate arguments in buffer. When buffer is full, empty the buffer
// fBuffer[0] = number of entries in buffer
// fBuffer[1] = w of first entry
// fBuffer[2] = x of first entry
// fBuffer[3] = y of first entry
// fBuffer[4] = z of first entry

   Int_t nbentries = (Int_t)fBuffer[0];
   if (4*nbentries+4 >= fBufferSize) {
      BufferEmpty(kTRUE);
      return Fill(x,y,z,w);
   }
   fBuffer[4*nbentries+1] = w;
   fBuffer[4*nbentries+2] = x;
   fBuffer[4*nbentries+3] = y;
   fBuffer[4*nbentries+4] = y;
   fBuffer[0] += 1;
   return -3;
}

//______________________________________________________________________________
Int_t TH3::Fill(Axis_t x, Axis_t y, Axis_t z)
{
//*-*-*-*-*-*-*-*-*-*-*Increment cell defined by x,y,z by 1 *-*-*-*-*
//*-*                  ====================================
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (fBuffer) return BufferFill(x,y,z,1);

   Int_t binx, biny, binz, bin;
   fEntries++;
   binx = fXaxis.FindBin(x);
   biny = fYaxis.FindBin(y);
   binz = fZaxis.FindBin(z);
   bin  =  binx + (fXaxis.GetNbins()+2)*(biny + (fYaxis.GetNbins()+2)*binz);
   AddBinContent(bin);
   if (fSumw2.fN) ++fSumw2.fArray[bin];
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   if (binz == 0 || binz > fZaxis.GetNbins()) return -1;
   return bin;
}

//______________________________________________________________________________
Int_t TH3::Fill(Axis_t x, Axis_t y, Axis_t z, Stat_t w)
{
//*-*-*-*-*-*-*-*-*-*-*Increment cell defined by x,y,z by a weight w*-*-*-*-*
//*-*                  =============================================
//*-*
//*-* If the storage of the sum of squares of weights has been triggered,
//*-* via the function Sumw2, then the sum of the squares of weights is incremented
//*-* by w^2 in the cell corresponding to x,y,z.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (fBuffer) return BufferFill(x,y,z,w);

   Int_t binx, biny, binz, bin;
   fEntries++;
   binx = fXaxis.FindBin(x);
   biny = fYaxis.FindBin(y);
   binz = fZaxis.FindBin(z);
   bin  =  binx + (fXaxis.GetNbins()+2)*(biny + (fYaxis.GetNbins()+2)*binz);
   AddBinContent(bin,w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   if (binz == 0 || binz > fZaxis.GetNbins()) return -1;
   return bin;
}

//______________________________________________________________________________
Int_t TH3::Fill(const char *namex, const char *namey, const char *namez, Stat_t w)
{
// Increment cell defined by namex,namey,namez by a weight w
//
// If the storage of the sum of squares of weights has been triggered,
// via the function Sumw2, then the sum of the squares of weights is incremented
// by w^2 in the cell corresponding to x,y,z.
//
   Int_t binx, biny, binz, bin;
   fEntries++;
   binx = fXaxis.FindBin(namex);
   biny = fYaxis.FindBin(namey);
   binz = fZaxis.FindBin(namez);
   bin  =  binx + (fXaxis.GetNbins()+2)*(biny + (fYaxis.GetNbins()+2)*binz);
   AddBinContent(bin,w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   if (binz == 0 || binz > fZaxis.GetNbins()) return -1;
   return bin;
}

//______________________________________________________________________________
Int_t TH3::Fill(const char *namex, Axis_t y, const char *namez, Stat_t w)
{
// Increment cell defined by namex,y,namez by a weight w
//
// If the storage of the sum of squares of weights has been triggered,
// via the function Sumw2, then the sum of the squares of weights is incremented
// by w^2 in the cell corresponding to x,y,z.
//
   Int_t binx, biny, binz, bin;
   fEntries++;
   binx = fXaxis.FindBin(namex);
   biny = fYaxis.FindBin(y);
   binz = fZaxis.FindBin(namez);
   bin  =  binx + (fXaxis.GetNbins()+2)*(biny + (fYaxis.GetNbins()+2)*binz);
   AddBinContent(bin,w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   if (binz == 0 || binz > fZaxis.GetNbins()) return -1;
   return bin;
}

//______________________________________________________________________________
Int_t TH3::Fill(const char *namex, const char *namey, Axis_t z, Stat_t w)
{
// Increment cell defined by namex,namey,z by a weight w
//
// If the storage of the sum of squares of weights has been triggered,
// via the function Sumw2, then the sum of the squares of weights is incremented
// by w^2 in the cell corresponding to x,y,z.
//
   Int_t binx, biny, binz, bin;
   fEntries++;
   binx = fXaxis.FindBin(namex);
   biny = fYaxis.FindBin(namey);
   binz = fZaxis.FindBin(z);
   bin  =  binx + (fXaxis.GetNbins()+2)*(biny + (fYaxis.GetNbins()+2)*binz);
   AddBinContent(bin,w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   if (binz == 0 || binz > fZaxis.GetNbins()) return -1;
   return bin;
}

//______________________________________________________________________________
Int_t TH3::Fill(Axis_t x, const char *namey, const char *namez, Stat_t w)
{
// Increment cell defined by x,namey,namezz by a weight w
//
// If the storage of the sum of squares of weights has been triggered,
// via the function Sumw2, then the sum of the squares of weights is incremented
// by w^2 in the cell corresponding to x,y,z.
//
   Int_t binx, biny, binz, bin;
   fEntries++;
   binx = fXaxis.FindBin(x);
   biny = fYaxis.FindBin(namey);
   binz = fZaxis.FindBin(namez);
   bin  =  binx + (fXaxis.GetNbins()+2)*(biny + (fYaxis.GetNbins()+2)*binz);
   AddBinContent(bin,w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   if (binz == 0 || binz > fZaxis.GetNbins()) return -1;
   return bin;
}

//______________________________________________________________________________
Int_t TH3::Fill(Axis_t x, const char *namey, Axis_t z, Stat_t w)
{
// Increment cell defined by x,namey,z by a weight w
//
// If the storage of the sum of squares of weights has been triggered,
// via the function Sumw2, then the sum of the squares of weights is incremented
// by w^2 in the cell corresponding to x,y,z.
//
   Int_t binx, biny, binz, bin;
   fEntries++;
   binx = fXaxis.FindBin(x);
   biny = fYaxis.FindBin(namey);
   binz = fZaxis.FindBin(z);
   bin  =  binx + (fXaxis.GetNbins()+2)*(biny + (fYaxis.GetNbins()+2)*binz);
   AddBinContent(bin,w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   if (binz == 0 || binz > fZaxis.GetNbins()) return -1;
   return bin;
}

//______________________________________________________________________________
Int_t TH3::Fill(Axis_t x, Axis_t y, const char *namez, Stat_t w)
{
// Increment cell defined by x,y,namez by a weight w
//
// If the storage of the sum of squares of weights has been triggered,
// via the function Sumw2, then the sum of the squares of weights is incremented
// by w^2 in the cell corresponding to x,y,z.
//
   Int_t binx, biny, binz, bin;
   fEntries++;
   binx = fXaxis.FindBin(x);
   biny = fYaxis.FindBin(y);
   binz = fZaxis.FindBin(namez);
   bin  =  binx + (fXaxis.GetNbins()+2)*(biny + (fYaxis.GetNbins()+2)*binz);
   AddBinContent(bin,w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   if (binz == 0 || binz > fZaxis.GetNbins()) return -1;
   return bin;
}

//______________________________________________________________________________
void TH3::FillRandom(const char *fname, Int_t ntimes)
{
//*-*-*-*-*-*-*Fill histogram following distribution in function fname*-*-*-*
//*-*          =======================================================
//*-*
//*-*   The distribution contained in the function fname (TF1) is integrated
//*-*   over the channel contents.
//*-*   It is normalized to 1.
//*-*   Getting one random number implies:
//*-*     - Generating a random number between 0 and 1 (say r1)
//*-*     - Look in which bin in the normalized integral r1 corresponds to
//*-*     - Fill histogram channel
//*-*   ntimes random numbers are generated
//*-*
//*-*  One can also call TF1::GetRandom to get a random variate from a function.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*

   Int_t bin, binx, biny, binz, ibin, loop;
   Double_t r1, x, y,z, xv[3];
//*-*- Search for fname in the list of ROOT defined functions
   TF1 *f1 = (TF1*)gROOT->GetFunction(fname);
   if (!f1) { Error("FillRandom", "Unknown function: %s",fname); return; }

//*-*- Allocate temporary space to store the integral and compute integral
   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
   Int_t nxy    = nbinsx*nbinsy;
   Int_t nbins  = nxy*nbinsz;

   Double_t *integral = new Double_t[nbins+1];
   ibin = 0;
   integral[ibin] = 0;
   for (binz=1;binz<=nbinsz;binz++) {
      xv[2] = fZaxis.GetBinCenter(binz);
      for (biny=1;biny<=nbinsy;biny++) {
         xv[1] = fYaxis.GetBinCenter(biny);
         for (binx=1;binx<=nbinsx;binx++) {
            xv[0] = fXaxis.GetBinCenter(binx);
            ibin++;
            integral[ibin] = integral[ibin-1] + f1->Eval(xv[0],xv[1],xv[2]);
         }
      }
   }

//*-*- Normalize integral to 1
   if (integral[nbins] == 0 ) {
      Error("FillRandom", "Integral = zero"); return;
   }
   for (bin=1;bin<=nbins;bin++)  integral[bin] /= integral[nbins];

//*-*--------------Start main loop ntimes
   if (fDimension < 2) nbinsy = -1;
   if (fDimension < 3) nbinsz = -1;
   for (loop=0;loop<ntimes;loop++) {
      r1 = gRandom->Rndm(loop);
      ibin = TMath::BinarySearch(nbins,&integral[0],r1);
      binz = ibin/nxy;
      biny = (ibin - nxy*binz)/nbinsx;
      binx = 1 + ibin - nbinsx*(biny + nbinsy*binz);
      if (nbinsz) binz++;
      if (nbinsy) biny++;
      x    = fXaxis.GetBinCenter(binx);
      y    = fYaxis.GetBinCenter(biny);
      z    = fZaxis.GetBinCenter(binz);
      Fill(x,y,z, 1.);
  }
  delete [] integral;
}

//______________________________________________________________________________
void TH3::FillRandom(TH1 *h, Int_t ntimes)
{
//*-*-*-*-*-*-*Fill histogram following distribution in histogram h*-*-*-*
//*-*          ====================================================
//*-*
//*-*   The distribution contained in the histogram h (TH3) is integrated
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

   TH3 *h3 = (TH3*)h;
   Int_t loop;
   Axis_t x,y,z;
   for (loop=0;loop<ntimes;loop++) {
      h3->GetRandom3(x,y,z);
      Fill(x,y,z,1.);
   }
}


//______________________________________________________________________________
void TH3::FitSlicesZ(TF1 *f1, Int_t binminx, Int_t binmaxx, Int_t binminy, Int_t binmaxy, Int_t cut, Option_t *option)
{
// Project slices along Z in case of a 3-D histogram, then fit each slice
// with function f1 and make a 2-d histogram for each fit parameter
// Only cells in the bin range [binminx,binmaxx] and [binminy,binmaxy] are considered.
// if f1=0, a gaussian is assumed
// Before invoking this function, one can set a subrange to be fitted along Z
// via f1->SetRange(zmin,zmax)
// The argument option (default="QNR") can be used to change the fit options.
//     "Q" means Quiet mode
//     "N" means do not show the result of the fit
//     "R" means fit the function in the specified function range
//
// Note that the generated histograms are added to the list of objects
// in the current directory. It is the user's responsability to delete
// these histograms.
//
//  Example: Assume a 3-d histogram h3
//   Root > h3->FitSlicesZ(); produces 4 TH2D histograms
//          with h3_0 containing parameter 0(Constant) for a Gaus fit
//                    of each cell in X,Y projected along Z
//          with h3_1 containing parameter 1(Mean) for a gaus fit
//          with h3_2 containing parameter 2(RMS)  for a gaus fit
//          with h3_chi2 containing the chisquare/number of degrees of freedom for a gaus fit
//
//   Root > h3->Fit(0,15,22,0,0,10);
//          same as above, but only for bins 15 to 22 along X
//          and only for cells in X,Y for which the corresponding projection
//          along Z has more than cut bins filled.
//
//  NOTE: To access the generated histograms in the current directory, do eg:
//     TH2D *h3_1 = (TH2D*)gDirectory->Get("h3_1");

   Int_t nbinsx  = fXaxis.GetNbins();
   Int_t nbinsy  = fYaxis.GetNbins();
   Int_t nbinsz  = fZaxis.GetNbins();
   if (binminx < 1) binminx = 1;
   if (binmaxx > nbinsx) binmaxx = nbinsx;
   if (binmaxx < binminx) {binminx = 1; binmaxx = nbinsx;}
   if (binminy < 1) binminy = 1;
   if (binmaxy > nbinsy) binmaxy = nbinsy;
   if (binmaxy < binminy) {binminy = 1; binmaxy = nbinsy;}

   //default is to fit with a gaussian
   if (f1 == 0) {
      f1 = (TF1*)gROOT->GetFunction("gaus");
      if (f1 == 0) f1 = new TF1("gaus","gaus",fZaxis.GetXmin(),fZaxis.GetXmax());
      else         f1->SetRange(fZaxis.GetXmin(),fZaxis.GetXmax());
   }
   const char *fname = f1->GetName();
   Int_t npar = f1->GetNpar();
   Double_t *parsave = new Double_t[npar];
   f1->GetParameters(parsave);

   //Create one 2-d histogram for each function parameter
   Int_t ipar;
   char name[80], title[80];
   TH2D *hlist[25];
   for (ipar=0;ipar<npar;ipar++) {
      sprintf(name,"%s_%d",GetName(),ipar);
      sprintf(title,"Fitted value of par[%d]=%s",ipar,f1->GetParName(ipar));
      hlist[ipar] = new TH2D(name,title, nbinsx, fXaxis.GetXmin(), fXaxis.GetXmax()
                                       , nbinsy, fYaxis.GetXmin(), fYaxis.GetXmax());
   }
   sprintf(name,"%s_chi2",GetName());
   TH2D *hchi2 = new TH2D(name,"chisquare", nbinsx, fXaxis.GetXmin(), fXaxis.GetXmax()
                                          , nbinsy, fYaxis.GetXmin(), fYaxis.GetXmax());

   //Loop on all cells in X,Y generate a projection along Z
   TH1D *hpz = new TH1D("R_temp","_temp",nbinsz, fZaxis.GetXmin(), fZaxis.GetXmax());
   Int_t bin,binx,biny,binz;
   for (biny=binminy;biny<=binmaxy;biny++) {
      Float_t y = fYaxis.GetBinCenter(biny);
      for (binx=binminx;binx<=binmaxx;binx++) {
         Float_t x = fXaxis.GetBinCenter(binx);
         hpz->Reset();
         Int_t nfill = 0;
         for (binz=1;binz<=nbinsz;binz++) {
            bin = GetBin(binx,biny,binz);
            Float_t w = GetBinContent(bin);
            if (w == 0) continue;
            hpz->Fill(fZaxis.GetBinCenter(binz),w);
            hpz->SetBinError(binz,GetBinError(bin));
            nfill++;
         }
         if (nfill < cut) continue;
         f1->SetParameters(parsave);
         hpz->Fit(fname,option);
         Int_t npfits = f1->GetNumberFitPoints();
         if (npfits > npar && npfits >= cut) {
            for (ipar=0;ipar<npar;ipar++) {
               hlist[ipar]->Fill(x,y,f1->GetParameter(ipar));
               hlist[ipar]->SetCellError(binx,biny,f1->GetParError(ipar));
            }
            hchi2->Fill(x,y,f1->GetChisquare()/(npfits-npar));
         }
      }
   }
   delete [] parsave;
   delete hpz;
}

//______________________________________________________________________________
void TH3::GetRandom3(Axis_t &x, Axis_t &y, Axis_t &z)
{
// return 3 random numbers along axis x , y and z distributed according
// the cellcontents of a 3-dim histogram

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
   Int_t nxy    = nbinsx*nbinsy;
   Int_t nbins  = nxy*nbinsz;
   Double_t integral;
   if (fIntegral) {
      if (fIntegral[nbins+1] != fEntries) integral = ComputeIntegral();
   } else {
      integral = ComputeIntegral();
      if (integral == 0 || fIntegral == 0) return;
   }
   Float_t r1 = gRandom->Rndm();
   Int_t ibin = TMath::BinarySearch(nbins,&fIntegral[0],r1);
   Int_t binz = ibin/nxy;
   Int_t biny = (ibin - nxy*binz)/nbinsx;
   Int_t binx = ibin - nbinsx*(biny + nbinsy*binz);
   x = fXaxis.GetBinLowEdge(binx+1)
      +fXaxis.GetBinWidth(binx+1)*(fIntegral[ibin+1]-r1)/(fIntegral[ibin+1] - fIntegral[ibin]);
   y = fYaxis.GetBinLowEdge(biny+1) + fYaxis.GetBinWidth(biny+1)*gRandom->Rndm();
   z = fZaxis.GetBinLowEdge(binz+1) + fZaxis.GetBinWidth(binz+1)*gRandom->Rndm();
}

//______________________________________________________________________________
void TH3::GetStats(Stat_t *stats) const
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
   // stats[7] = sumwz
   // stats[8] = sumwz2

   if (fBuffer) ((TH3*)this)->BufferEmpty();
   
   Int_t bin, binx, biny, binz;
   Stat_t w;
   Float_t x,y,z;
   for (bin=0;bin<9;bin++) stats[bin] = 0;
   for (binz=fZaxis.GetFirst();binz<=fZaxis.GetLast();binz++) {
      z = fZaxis.GetBinCenter(binz);
      for (biny=fYaxis.GetFirst();biny<=fYaxis.GetLast();biny++) {
         y = fYaxis.GetBinCenter(biny);
         for (binx=fXaxis.GetFirst();binx<=fXaxis.GetLast();binx++) {
            bin = GetBin(binx,biny,binz);
            x = fXaxis.GetBinCenter(binx);
            w = TMath::Abs(GetBinContent(bin));
            stats[0] += w;
            stats[1] += w*w;
            stats[2] += w*x;
            stats[3] += w*x*x;
            stats[4] += w*y;
            stats[5] += w*y*y;
            stats[6] += w*x*y;
            stats[7] += w*z;
            stats[8] += w*z*z;
         }
      }
   }
}

//______________________________________________________________________________
Stat_t TH3::Integral(Option_t *option) const
{
//Return integral of bin contents. Only bins in the bins range are considered.
// By default the integral is computed as the sum of bin contents in the range.
// if option "width" is specified, the integral is the sum of
// the bin contents multiplied by the bin width in x, y and in z.

   return Integral(fXaxis.GetFirst(),fXaxis.GetLast(),
                   fYaxis.GetFirst(),fYaxis.GetLast(),
                   fZaxis.GetFirst(),fZaxis.GetLast(),option);
}

//______________________________________________________________________________
Stat_t TH3::Integral(Int_t binx1, Int_t binx2, Int_t biny1, Int_t biny2, Int_t binz1, Int_t binz2, Option_t *option) const
{
//Return integral of bin contents in range [binx1,binx2],[biny1,biny2],[binz1,binz2]
// for a 3-D histogram
// By default the integral is computed as the sum of bin contents in the range.
// if option "width" is specified, the integral is the sum of
// the bin contents multiplied by the bin width in x, y and in z.

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
   if (binx1 < 0) binx1 = 0;
   if (binx2 > nbinsx+1) binx2 = nbinsx+1;
   if (biny1 < 0) biny1 = 0;
   if (biny2 > nbinsy+1) biny2 = nbinsy+1;
   if (binz1 < 0) binz1 = 0;
   if (binz2 > nbinsz+1) binz2 = nbinsz+1;
   Stat_t integral = 0;

//*-*- Loop on bins in specified range
   TString opt = option;
   opt.ToLower();
   Bool_t width = kFALSE;
   if (opt.Contains("width")) width = kTRUE;
   Int_t bin, binx, biny, binz;
   for (binz=binz1;binz<=binz2;binz++) {
      for (biny=biny1;biny<=biny2;biny++) {
         for (binx=binx1;binx<=binx2;binx++) {
            bin = binx +(nbinsx+2)*(biny + (nbinsy+2)*binz);
            if (width) integral += GetBinContent(bin)*fXaxis.GetBinWidth(binx)*fYaxis.GetBinWidth(biny)*fZaxis.GetBinWidth(binz);
            else       integral += GetBinContent(bin);
         }
      }
   }
   return integral;
}
        
//______________________________________________________________________________
Double_t TH3::KolmogorovTest(TH1 *h2, Option_t *option)
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
//        WARNING !!!! THIS FUNCTION NOT YET TESTED
//  I started from TH2::KolmogorovTest, but changes are probably required
//  when invoking KolmogorovProb to take into account the 3rd dimension
//  It would be nice if a mathematician could look into this.
//
//  Code adapted by Rene Brun from original HBOOK routine HDIFF

   TString opt = option;
   opt.ToUpper();
   
   Double_t prb = 0;
   TH1 *h1 = this;
   if (h2 == 0) return 0;
   TAxis *xaxis1 = h1->GetXaxis();
   TAxis *xaxis2 = h2->GetXaxis();
   TAxis *yaxis1 = h1->GetYaxis();
   TAxis *yaxis2 = h2->GetYaxis();
   TAxis *zaxis1 = h1->GetZaxis();
   TAxis *zaxis2 = h2->GetZaxis();
   Int_t ncx1   = xaxis1->GetNbins();
   Int_t ncx2   = xaxis2->GetNbins();
   Int_t ncy1   = yaxis1->GetNbins();
   Int_t ncy2   = yaxis2->GetNbins();
   Int_t ncz1   = zaxis1->GetNbins();
   Int_t ncz2   = zaxis2->GetNbins();

     // Check consistency of dimensions
   if (h1->GetDimension() != 3 || h2->GetDimension() != 3) {
      Error("KolmogorovTest","Histograms must be 3-D\n");
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
   if (ncz1 != ncz2) {
      Error("KolmogorovTest","Number of channels in Z is different, %d and %d\n",ncz1,ncz2);
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
   diff1 = TMath::Abs(zaxis1->GetXmin() - zaxis2->GetXmin());
   diff2 = TMath::Abs(zaxis1->GetXmax() - zaxis2->GetXmax());
   if (diff1 > difprec || diff2 > difprec) {
      Error("KolmogorovTest","histograms with different binning along Z");
      return 0;
   }

   //   Should we include Uflows, Oflows?
   Int_t ibeg = 1, jbeg = 1, kbeg = 1;
   Int_t iend = ncx1, jend = ncy1, kend = ncz1;
   if (opt.Contains("U")) {ibeg = 0; jbeg = 0; kbeg = 0;}
   if (opt.Contains("O")) {iend = ncx1+1; jend = ncy1+1; kend = ncz1+1;}
   
   Int_t i,j,k,bin;
   Double_t hsav;
   Double_t sum1  = 0;
   Double_t tsum1 = 0;
   for (i=0;i<=ncx1+1;i++) {
      for (j=0;j<=ncy1+1;j++) {
         for (k=0;k<=ncz1+1;k++) {
            bin = h1->GetBin(i,j,k);
            hsav = h1->GetBinContent(bin);
            tsum1 += hsav;
            if (i >= ibeg && i <= iend && j >= jbeg && j <= jend && k >= kbeg && k <= kend) sum1 += hsav;
         }
      }
   }
   Double_t sum2  = 0;
   Double_t tsum2 = 0;
   for (i=0;i<=ncx1+1;i++) {
      for (j=0;j<=ncy1+1;j++) {
         for (k=0;k<=ncz1+1;k++) {
            bin = h1->GetBin(i,j,k);
            hsav = h1->GetBinContent(bin);
            tsum2 += hsav;
            if (i >= ibeg && i <= iend && j >= jbeg && j <= jend&& k >= kbeg && k <= kend) sum2 += hsav;
         }
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
         for (k=kbeg;k<=kend;k++) {
            bin = h1->GetBin(i,j,k);
            rsum1 += s1*h1->GetBinContent(bin);
            rsum2 += s2*h2->GetBinContent(bin);
            dfmax  = TMath::Max(dfmax, TMath::Abs(rsum1-rsum2));
         }
      }
   }

   //   Find second Kolmogorov distance
   Double_t dfmax2 = 0;
   rsum1=0, rsum2=0;
   for (k=kbeg;k<=kend;k++) {
      for (j=jbeg;j<=jend;j++) {
         for (i=ibeg;i<=iend;i++) {
            bin = h1->GetBin(i,j,k);
            rsum1 += s1*h1->GetBinContent(bin);
            rsum2 += s2*h2->GetBinContent(bin);
            dfmax2 = TMath::Max(dfmax2, TMath::Abs(rsum1-rsum2));
         }
      }
   }

   //  Probably one should compute a third distance <======
   
   //    Get Kolmogorov probability
   Double_t factnm;
   if (afunc1)      factnm = TMath::Sqrt(sum2);
   else if (afunc2) factnm = TMath::Sqrt(sum1);
   else             factnm = TMath::Sqrt(sum1*sum2/(sum1+sum2));
   Double_t z  = dfmax*factnm;
   Double_t z2 = dfmax2*factnm;
   
   prb = TMath::KolmogorovProb(0.5*(z+z2)); //<==this should probably be updated

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
Int_t TH3::Merge(TCollection *list)
{
   //Merge all histograms in the collection in this histogram.
   //This function computes the min/max for the axes,
   //compute a new number of bins, if necessary,
   //add bin contents, errors and statistics.
   //The function returns the merged number of entries if the merge is 
   //successfull, -1 otherwise.
   //
   //IMPORTANT remark. The 3 axis x,y and z may have different number
   //of bins and different limits, BUT the largest bin width must be
   //a multiple of the smallest bin width.
   
   if (!list) return 0;
   TIter next(list);
   Double_t umin,umax,vmin,vmax,wmin,wmax;
   Int_t nx,ny,nz;
   Double_t xmin  = fXaxis.GetXmin();
   Double_t xmax  = fXaxis.GetXmax();
   Double_t ymin  = fYaxis.GetXmin();
   Double_t ymax  = fYaxis.GetXmax();
   Double_t zmin  = fZaxis.GetXmin();
   Double_t zmax  = fZaxis.GetXmax();
   Double_t bwix  = fXaxis.GetBinWidth(1);
   Double_t bwiy  = fYaxis.GetBinWidth(1);
   Double_t bwiz  = fZaxis.GetBinWidth(1);
   Int_t    nbix  = fXaxis.GetNbins();
   Int_t    nbiy  = fYaxis.GetNbins();
   Int_t    nbiz  = fZaxis.GetNbins();

   const Int_t kNstat = 10;
   Stat_t stats[kNstat], totstats[kNstat];
   TH3 *h;
   Int_t i, nentries=0;
   for (i=0;i<kNstat;i++) {totstats[i] = stats[i] = 0;}
   Bool_t same = kTRUE;
   while ((h=(TH3*)next())) {     
      if (!h->InheritsFrom(TH3::Class())) {
         Error("Add","Attempt to add object of class: %s to a %s",h->ClassName(),this->ClassName());
         return -1;
      }
      //import statistics
      h->GetStats(stats);
      for (i=0;i<kNstat;i++) totstats[i] += stats[i];
      nentries += (Int_t)h->GetEntries();
      
      // find min/max of the axes
      umin = h->GetXaxis()->GetXmin();
      umax = h->GetXaxis()->GetXmax();
      vmin = h->GetYaxis()->GetXmin();
      vmax = h->GetYaxis()->GetXmax();
      wmin = h->GetZaxis()->GetXmin();
      wmax = h->GetZaxis()->GetXmax();
      nx   = h->GetXaxis()->GetNbins();
      ny   = h->GetYaxis()->GetNbins();
      nz   = h->GetZaxis()->GetNbins();
      if (nx != nbix || ny != nbiy || nz != nbiz ||
                    umin != xmin || umax != xmax ||
                    vmin != ymin || vmax != ymax ||
                    wmin != zmin || wmax != zmax) {
         same = kFALSE;
         if (umin < xmin) xmin = umin;  
         if (umax > xmax) xmax = umax;  
         if (vmin < ymin) ymin = vmin;  
         if (vmax > ymax) ymax = vmax;  
         if (wmin < zmin) zmin = wmin;  
         if (wmax > zmax) zmax = wmax;  
         if (h->GetXaxis()->GetBinWidth(1) > bwix) bwix = h->GetXaxis()->GetBinWidth(1);     
         if (h->GetYaxis()->GetBinWidth(1) > bwiy) bwiy = h->GetYaxis()->GetBinWidth(1);     
         if (h->GetZaxis()->GetBinWidth(1) > bwiz) bwiz = h->GetZaxis()->GetBinWidth(1);     
      }
   }
   
   //  if different binning compute best binning
   if (!same) {
      nbix = (Int_t) ((xmax-xmin)/bwix +0.1); while(nbix > 100) nbix /= 2;
      nbiy = (Int_t) ((ymax-ymin)/bwiy +0.1); while(nbiy > 100) nbiy /= 2;
      nbiz = (Int_t) ((zmax-zmin)/bwiz +0.1); while(nbiz > 100) nbiz /= 2;
      SetBins(nbix,xmin,xmax,nbiy,ymin,ymax,nbiz,zmin,zmax);
   }
   
   //merge bin contents and errors
   next.Reset();
   Int_t ibin, bin, binx, biny, binz, ix, iy, iz;
   Double_t cu;
   while ((h=(TH3*)next())) {     
      nx   = h->GetXaxis()->GetNbins();
      ny   = h->GetYaxis()->GetNbins();
      nz   = h->GetZaxis()->GetNbins();
      for (binz=0;binz<=nz+1;binz++) {
         iz = fZaxis.FindBin(h->GetBinCenter(binz));
         for (biny=0;biny<=ny+1;biny++) {
            iy = fYaxis.FindBin(h->GetBinCenter(biny));
            for (binx=0;binx<=nx+1;binx++) {
               ix = fXaxis.FindBin(h->GetBinCenter(binx));
               bin = binx +(nx+2)*(biny + (ny+2)*binz);
               ibin = ix +(nbix+2)*(iy + (nbiy+2)*iz);
               cu  = h->GetBinContent(bin);
               AddBinContent(ibin,cu);
               if (fSumw2.fN) {
                  Double_t error1 = h->GetBinError(bin);
                  fSumw2.fArray[ibin] += error1*error1;
               }
            }
         }
      }
   }
   
   //copy merged stats
   PutStats(totstats);
   SetEntries(nentries);
   
   return nentries;
}   

//______________________________________________________________________________
TH1D *TH3::ProjectionZ(const char *name, Int_t ixmin, Int_t ixmax, Int_t iymin, Int_t iymax, Option_t *option) const
{
//*-*-*-*-*Project a 3-D histogram into a 1-D histogram along Z*-*-*-*-*-*-*
//*-*      ====================================================
//
//   The projection is always of the type TH1D.
//   The projection is made from the cells along the X axis
//   ranging from ixmin to ixmax and iymin to iymax included.
//   By default, bins 1 to nx and 1 to ny  are included
//
//   if option "E" is specified, the errors are computed.
//
//   code from Paola Collins & Hans Dijkstra

  TString opt = option;
  opt.ToLower();
  Int_t nx = GetNbinsX();
  Int_t ny = GetNbinsY();
  Int_t nz = GetNbinsZ();
  if (ixmin < 0)  ixmin = 1;
  if (ixmax > nx) ixmax = nx;
  if (iymin < 0)  iymin = 1;
  if (iymax > ny) iymax = ny;

// Create the projection histogram
  char *pname = (char*)name;
  if (strcmp(name,"_pz") == 0) {
     Int_t nch = strlen(GetName()) + 4;
     pname = new char[nch];
     sprintf(pname,"%s%s",GetName(),name);
  }
  TH1D *h1;
  const TArrayD *bins = fZaxis.GetXbins();
  if (bins->fN == 0) {
     h1 = new TH1D(pname,GetTitle(),nz,fZaxis.GetXmin(),fZaxis.GetXmax());
  } else {
     h1 = new TH1D(pname,GetTitle(),nz,bins->fArray);
  }
  Bool_t computeErrors = kFALSE;
  if (opt.Contains("e")) {h1->Sumw2(); computeErrors = kTRUE;}
  if (pname != name)  delete [] pname;

// Fill the projected histogram
  Float_t cont,e,e1;
  Double_t entries  = 0;
  Double_t newerror = 0;
  for (Int_t ixbin=ixmin;ixbin<=ixmax;ixbin++){
     for (Int_t iybin=iymin;iybin<=iymax;iybin++){
        for (Int_t binz=0;binz<=(nz+1);binz++){
           Int_t bin = GetBin(ixbin,iybin,binz);
           cont = GetBinContent(bin);
           if (computeErrors) {
              e        = GetBinError(bin);
              e1       = h1->GetBinError(binz);
              newerror = TMath::Sqrt(e*e + e1*e1);
           }
           if (cont) {
              h1->Fill(fZaxis.GetBinCenter(binz), cont);
              entries += cont;
           }
           if (computeErrors) h1->SetBinError(binz,newerror);
        }
     }
  }
  if (iymin <=1 && iymax >= ny && ixmin <=1 && ixmax >= nx) h1->SetEntries(fEntries);
  else h1->SetEntries(entries);
  return h1;
}

//______________________________________________________________________________
TH1 *TH3::Project3D(Option_t *option) const
{
   // Project a 3-d histogram into 1 or 2-d histograms depending on the
   // option parameter
   // option may contain a combination of the characters x,y,z,e
   // option = "x" return the x projection into a TH1D histogram
   // option = "y" return the y projection into a TH1D histogram
   // option = "z" return the z projection into a TH1D histogram
   // option = "xy" return the x versus y projection into a TH2D histogram
   // option = "yx" return the y versus x projection into a TH2D histogram
   // option = "xz" return the x versus z projection into a TH2D histogram
   // option = "zx" return the z versus x projection into a TH2D histogram
   // option = "yz" return the y versus z projection into a TH2D histogram
   // option = "zy" return the z versus y projection into a TH2D histogram
   //
   // If option contains the string "e", errors are computed
   //
   // The projection is made for the selected bins only.
   // To select a bin range along an axis, use TAxis::SetRange, eg
   //    h3.GetYaxis()->SetRange(23,56);

  TString opt = option; opt.ToLower();
  Int_t ixmin = fXaxis.GetFirst();
  Int_t ixmax = fXaxis.GetLast();
  Int_t iymin = fYaxis.GetFirst();
  Int_t iymax = fYaxis.GetLast();
  Int_t izmin = fZaxis.GetFirst();
  Int_t izmax = fZaxis.GetLast();
  Int_t nx = ixmax-ixmin+1;
  Int_t ny = iymax-iymin+1;
  Int_t nz = izmax-izmin+1;
  Int_t pcase = 0;
  if (opt.Contains("x"))  pcase = 1;
  if (opt.Contains("y"))  pcase = 2;
  if (opt.Contains("z"))  pcase = 3;
  if (opt.Contains("xy")) pcase = 4;
  if (opt.Contains("yx")) pcase = 5;
  if (opt.Contains("xz")) pcase = 6;
  if (opt.Contains("zx")) pcase = 7;
  if (opt.Contains("yz")) pcase = 8;
  if (opt.Contains("zy")) pcase = 9;

// Create the projection histogram
  TH1D *h1 = 0;
  TH2D *h2 = 0;
  Int_t nch = strlen(GetName()) +opt.Length() +2;
  char *name = new char[nch];
  sprintf(name,"%s_%s",GetName(),option);
  nch = strlen(GetTitle()) +opt.Length() +2;
  char *title = new char[nch];
  sprintf(title,"%s_%s",GetTitle(),option);
  const TArrayD *bins;
  const TArrayD *xbins;
  const TArrayD *ybins;
  const TArrayD *zbins;
  switch (pcase) {
     case 1:
        // "x"
        bins = fXaxis.GetXbins();
        if (bins->fN == 0) {
           h1 = new TH1D(name,title,nx,fXaxis.GetBinLowEdge(ixmin),fXaxis.GetBinUpEdge(ixmax));
        } else {
           h1 = new TH1D(name,title,nx,&bins->fArray[ixmin-1]);
        }
        break;

     case 2:
        // "y"
        bins = fYaxis.GetXbins();
        if (bins->fN == 0) {
           h1 = new TH1D(name,title,ny,fYaxis.GetBinLowEdge(iymin),fYaxis.GetBinUpEdge(iymax));
        } else {
           h1 = new TH1D(name,title,ny,&bins->fArray[iymin-1]);
        }
        break;

     case 3:
        // "z"
        bins = fZaxis.GetXbins();
        if (bins->fN == 0) {
           h1 = new TH1D(name,title,nz,fZaxis.GetBinLowEdge(izmin),fZaxis.GetBinUpEdge(izmax));
        } else {
           h1 = new TH1D(name,title,nz,&bins->fArray[izmin-1]);
        }
        break;
     case 4:
        // "xy"
        xbins = fXaxis.GetXbins();
        ybins = fYaxis.GetXbins();
        if (xbins->fN == 0 && ybins->fN == 0) {
           h2 = new TH2D(name,title,ny,fYaxis.GetBinLowEdge(iymin),fYaxis.GetBinUpEdge(iymax)                                  
                        ,nx,fXaxis.GetBinLowEdge(ixmin),fXaxis.GetBinUpEdge(ixmax));
        } else if (ybins->fN == 0) {
           h2 = new TH2D(name,title,ny,fYaxis.GetBinLowEdge(iymin),fYaxis.GetBinUpEdge(iymax)
                        ,nx,&xbins->fArray[ixmin-1]);
        } else if (xbins->fN == 0) {
           h2 = new TH2D(name,title,ny,&ybins->fArray[iymin-1]                                  
                        ,nx,fXaxis.GetBinLowEdge(ixmin),fXaxis.GetBinUpEdge(ixmax));
        } else {
           h2 = new TH2D(name,title,ny,&ybins->fArray[iymin-1],nx,&xbins->fArray[ixmin-1]);
        }
        break;

     case 5:
        // "yx"
        xbins = fXaxis.GetXbins();
        ybins = fYaxis.GetXbins();
        if (xbins->fN == 0 && ybins->fN == 0) {
           h2 = new TH2D(name,title,nx,fXaxis.GetBinLowEdge(ixmin),fXaxis.GetBinUpEdge(ixmax)
                        ,ny,fYaxis.GetBinLowEdge(iymin),fYaxis.GetBinUpEdge(iymax));
        } else if (xbins->fN == 0) {
           h2 = new TH2D(name,title,nx,fXaxis.GetBinLowEdge(ixmin),fXaxis.GetBinUpEdge(ixmax)
                                   ,ny,&ybins->fArray[iymin-1]);
        } else if (ybins->fN == 0) {
           h2 = new TH2D(name,title,nx,&xbins->fArray[ixmin-1]
                                  ,ny,fYaxis.GetBinLowEdge(iymin),fYaxis.GetBinUpEdge(iymax));
        } else {
           h2 = new TH2D(name,title,nx,&xbins->fArray[ixmin-1],ny,&ybins->fArray[iymin-1]);
        }
        break;

     case 6:
        // "xz"
        xbins = fXaxis.GetXbins();
        zbins = fZaxis.GetXbins();
        if (xbins->fN == 0 && zbins->fN == 0) {
           h2 = new TH2D(name,title,nz,fZaxis.GetBinLowEdge(izmin),fZaxis.GetBinUpEdge(izmax)
                                  ,nx,fXaxis.GetBinLowEdge(ixmin),fXaxis.GetBinUpEdge(ixmax));
        } else if (zbins->fN == 0) {
           h2 = new TH2D(name,title,nz,fZaxis.GetBinLowEdge(izmin),fZaxis.GetBinUpEdge(izmax)
                                   ,nx,&xbins->fArray[ixmin-1]);
        } else if (xbins->fN == 0) {
           h2 = new TH2D(name,title,nz,&zbins->fArray[izmin-1]
                                  ,nx,fXaxis.GetBinLowEdge(ixmin),fXaxis.GetBinUpEdge(ixmax));
        } else {
           h2 = new TH2D(name,title,nz,&zbins->fArray[izmin-1],nx,&xbins->fArray[ixmin-1]);
        }
        break;

     case 7:
        // "zx"
        xbins = fXaxis.GetXbins();
        zbins = fZaxis.GetXbins();
        if (xbins->fN == 0 && zbins->fN == 0) {
           h2 = new TH2D(name,title,nx,fXaxis.GetBinLowEdge(ixmin),fXaxis.GetBinUpEdge(ixmax)
                                  ,nz,fZaxis.GetBinLowEdge(izmin),fZaxis.GetBinUpEdge(izmax));
        } else if (xbins->fN == 0) {
           h2 = new TH2D(name,title,nx,fXaxis.GetBinLowEdge(ixmin),fXaxis.GetBinUpEdge(ixmax)
                                   ,nz,&zbins->fArray[izmin-1]);
        } else if (zbins->fN == 0) {
           h2 = new TH2D(name,title,nx,&xbins->fArray[ixmin-1]
                                  ,nz,fZaxis.GetBinLowEdge(izmin),fZaxis.GetBinUpEdge(izmax));
        } else {
           h2 = new TH2D(name,title,nx,&xbins->fArray[ixmin-1],nz,&zbins->fArray[izmin-1]);
        }
        break;

     case 8:
        // "yz"
        ybins = fYaxis.GetXbins();
        zbins = fZaxis.GetXbins();
        if (ybins->fN == 0 && zbins->fN == 0) {
           h2 = new TH2D(name,title,nz,fZaxis.GetBinLowEdge(izmin),fZaxis.GetBinUpEdge(izmax)
                                  ,ny,fYaxis.GetBinLowEdge(iymin),fYaxis.GetBinUpEdge(iymax));
        } else if (zbins->fN == 0) {
           h2 = new TH2D(name,title,nz,fZaxis.GetBinLowEdge(izmin),fZaxis.GetBinUpEdge(izmax)
                                   ,ny,&ybins->fArray[iymin-1]);
        } else if (ybins->fN == 0) {
           h2 = new TH2D(name,title,nz,&zbins->fArray[izmin-1]
                                  ,ny,fYaxis.GetBinLowEdge(iymin),fYaxis.GetBinUpEdge(iymax));
        } else {
           h2 = new TH2D(name,title,nz,&zbins->fArray[izmin-1],ny,&ybins->fArray[iymin-1]);
        }
        break;

     case 9:
        // "zy"
        ybins = fYaxis.GetXbins();
        zbins = fZaxis.GetXbins();
        if (ybins->fN == 0 && zbins->fN == 0) {
           h2 = new TH2D(name,title,ny,fYaxis.GetBinLowEdge(iymin),fYaxis.GetBinUpEdge(iymax)
                                  ,nz,fZaxis.GetBinLowEdge(izmin),fZaxis.GetBinUpEdge(izmax));
        } else if (ybins->fN == 0) {
           h2 = new TH2D(name,title,ny,fYaxis.GetBinLowEdge(iymin),fYaxis.GetBinUpEdge(iymax)
                                   ,nz,&zbins->fArray[izmin-1]);
        } else if (zbins->fN == 0) {
           h2 = new TH2D(name,title,ny,&ybins->fArray[iymin-1]
                                  ,nz,fZaxis.GetBinLowEdge(izmin),fZaxis.GetBinUpEdge(izmax));
        } else {
           h2 = new TH2D(name,title,ny,&ybins->fArray[iymin-1],nz,&zbins->fArray[izmin-1]);
        }
        break;

     }

  delete [] name;
  delete [] title;
  TH1 *h = h1;
  if (h2) h = h2;
  if (h == 0) return 0;

  Bool_t computeErrors = kFALSE;
  if (opt.Contains("e")) {h->Sumw2(); computeErrors = kTRUE;}

// Fill the projected histogram
  Float_t cont,e,e1;
  Double_t entries  = 0;
  Double_t newerror = 0;
  for (Int_t ixbin=ixmin;ixbin<=ixmax;ixbin++){
     Int_t ix = ixbin-ixmin+1;
     for (Int_t iybin=iymin;iybin<=iymax;iybin++){
        Int_t iy = iybin-iymin+1;
        for (Int_t izbin=izmin;izbin<=izmax;izbin++){
           Int_t iz = izbin-izmin+1;
           Int_t bin = GetBin(ixbin,iybin,izbin);
           cont = GetBinContent(bin);
           switch (pcase) {
           case 1:
              e1       = h1->GetBinError(ix);
              if (cont) h1->Fill(fXaxis.GetBinCenter(ixbin), cont);
              if (computeErrors) {
                 e        = GetBinError(bin);
                 newerror = TMath::Sqrt(e*e + e1*e1);
                 h1->SetBinError(ix,newerror);
              }
              break;

           case 2:
              e1       = h1->GetBinError(iy);
              if (cont) h1->Fill(fYaxis.GetBinCenter(iybin), cont);
              if (computeErrors) {
                 e        = GetBinError(bin);
                 newerror = TMath::Sqrt(e*e + e1*e1);
                 h1->SetBinError(iy,newerror);
              }
              break;

           case 3:
              e1       = h1->GetBinError(iz);
              if (cont) h1->Fill(fZaxis.GetBinCenter(izbin), cont);
              if (computeErrors) {
                 e        = GetBinError(bin);
                 newerror = TMath::Sqrt(e*e + e1*e1);
                 h1->SetBinError(iz,newerror);
              }
              break;

           case 4:
              e1       = h2->GetCellError(iy,ix);
              if (cont) h2->Fill(fYaxis.GetBinCenter(iybin),fXaxis.GetBinCenter(ixbin), cont);
              if (computeErrors) {
                 e        = GetBinError(bin);
                 newerror = TMath::Sqrt(e*e + e1*e1);
                 h2->SetCellError(iy,ix,newerror);
              }
              break;

           case 5:
              e1       = h2->GetCellError(ix,iy);
              if (cont) h2->Fill(fXaxis.GetBinCenter(ixbin),fYaxis.GetBinCenter(iybin), cont);
              if (computeErrors) {
                 e        = GetBinError(bin);
                 newerror = TMath::Sqrt(e*e + e1*e1);
                 h2->SetCellError(ix,iy,newerror);
              }
              break;

           case 6:
              e1       = h2->GetCellError(iz,ix);
              if (cont) h2->Fill(fZaxis.GetBinCenter(izbin),fXaxis.GetBinCenter(ixbin), cont);
              if (computeErrors) {
                 e        = GetBinError(bin);
                 newerror = TMath::Sqrt(e*e + e1*e1);
                 h2->SetCellError(iz,ix,newerror);
              }
              break;

           case 7:
              e1       = h2->GetCellError(ix,iz);
              if (cont) h2->Fill(fXaxis.GetBinCenter(ixbin),fZaxis.GetBinCenter(izbin), cont);
              if (computeErrors) {
                 e        = GetBinError(bin);
                 newerror = TMath::Sqrt(e*e + e1*e1);
                 h2->SetCellError(ix,iz,newerror);
              }
              break;

           case 8:
              e1       = h2->GetCellError(iz,iy);
              if (cont) h2->Fill(fZaxis.GetBinCenter(izbin),fYaxis.GetBinCenter(iybin), cont);
              if (computeErrors) {
                 e        = GetBinError(bin);
                 newerror = TMath::Sqrt(e*e + e1*e1);
                 h2->SetCellError(iz,iy,newerror);
              }
              break;

           case 9:
              e1       = h2->GetCellError(iy,iz);
              if (cont) h2->Fill(fYaxis.GetBinCenter(iybin),fZaxis.GetBinCenter(izbin), cont);
              if (computeErrors) {
                 e        = GetBinError(bin);
                 newerror = TMath::Sqrt(e*e + e1*e1);
                 h2->SetCellError(iy,iz,newerror);
              }
              break;
           }
           if (cont) {
              entries += cont;
           }
        }
     }
  }
  h->SetEntries(entries);
  return h;
}

//______________________________________________________________________________
void TH3::PutStats(Stat_t *stats)
{
   // Replace current statistics with the values in array stats

   TH1::PutStats(stats);
}

//______________________________________________________________________________
void TH3::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*Reset this histogram: contents, errors, etc*-*-*-*-*-*-*-*
//*-*            ===========================================

   TH1::Reset(option);
   // should also reset statistics once statistics are implemented for TH3
}

//______________________________________________________________________________
void TH3::Sizeof3D() const
{
//*-*-*-*-*-*-*Return total size of this 3-D shape with its attributes*-*-*
//*-*          ==========================================================

   char *cmd;
   if (GetDrawOption() && strstr(GetDrawOption(),"box")) {
      cmd = Form("TMarker3DBox::SizeofH3((TH3 *)0x%lx);",(Long_t)this);
   } else {
      cmd = Form("TPolyMarker3D::SizeofH3((TH3 *)0x%lx);",(Long_t)this);
   }
   gROOT->ProcessLine(cmd);
}

//______________________________________________________________________________
void TH3::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH3.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TH3::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TH1::Streamer(R__b);
      TAtt3D::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TH3::IsA());
      //====end of old versions
      
   } else {
      TH3::Class()->WriteBuffer(R__b,this);
   }
}


ClassImp(TH3C)

//______________________________________________________________________________
//                     TH3C methods
//______________________________________________________________________________
TH3C::TH3C(): TH3()
{
}

//______________________________________________________________________________
TH3C::~TH3C()
{
}

//______________________________________________________________________________
TH3C::TH3C(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup
                                     ,Int_t nbinsz,Axis_t zlow,Axis_t zup)
     :TH3(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup,nbinsz,zlow,zup)
{
//*-*-*-*-*-*-*-*-*Normal constructor for fix bin size 3-D histograms*-*-*-*-*
//*-*              ==================================================

   TArrayC::Set(fNcells);
}

//______________________________________________________________________________
TH3C::TH3C(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                             ,Int_t nbinsy,const Float_t *ybins
                                             ,Int_t nbinsz,const Float_t *zbins)
     :TH3(name,title,nbinsx,xbins,nbinsy,ybins,nbinsz,zbins)
{
//*-*-*-*-*-*-*-*Normal constructor for variable bin size 3-D histograms*-*-*-*
//*-*            =======================================================
   TArrayC::Set(fNcells);
}

//______________________________________________________________________________
TH3C::TH3C(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                             ,Int_t nbinsy,const Double_t *ybins
                                             ,Int_t nbinsz,const Double_t *zbins)
     :TH3(name,title,nbinsx,xbins,nbinsy,ybins,nbinsz,zbins)
{
//*-*-*-*-*-*-*-*Normal constructor for variable bin size 3-D histograms*-*-*-*
//*-*            =======================================================
   TArrayC::Set(fNcells);
}

//______________________________________________________________________________
TH3C::TH3C(const TH3C &h3c)
{
   ((TH3C&)h3c).Copy(*this);
}

//______________________________________________________________________________
void TH3C::AddBinContent(Int_t bin)
{
//*-*-*-*-*-*-*-*-*-*Increment bin content by 1*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==========================

   if (fArray[bin] < 127) fArray[bin]++;
}

//______________________________________________________________________________
void TH3C::AddBinContent(Int_t bin, Stat_t w)
{
//*-*-*-*-*-*-*-*-*-*Increment bin content by w*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==========================

   Int_t newval = fArray[bin] + Int_t(w);
   if (newval > -128 && newval < 128) {fArray[bin] = Char_t(newval); return;}
   if (newval < -127) fArray[bin] = -127;
   if (newval >  127) fArray[bin] =  127;
}

//______________________________________________________________________________
void TH3C::Copy(TObject &newth3)
{
//*-*-*-*-*-*-*Copy this 3-D histogram structure to newth3*-*-*-*-*-*-*-*-*-*
//*-*          ===========================================

   TH3::Copy((TH3C&)newth3);
   TArrayC::Copy((TH3C&)newth3);
}

//______________________________________________________________________________
TH1 *TH3C::DrawCopy(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH3C *newth3 = (TH3C*)Clone();
   newth3->SetDirectory(0);
   newth3->SetBit(kCanDelete);
   newth3->AppendPad(option);
   return newth3;
}

//______________________________________________________________________________
Stat_t TH3C::GetBinContent(Int_t bin) const
{
   if (fBuffer) ((TH3C*)this)->BufferEmpty();
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   return Stat_t (fArray[bin]);
}

//______________________________________________________________________________
void TH3C::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*Reset this histogram: contents, errors, etc*-*-*-*-*-*-*-*
//*-*            ===========================================

   TH3::Reset(option);
   TArrayC::Reset();
   // should also reset statistics once statistics are implemented for TH3
}

//______________________________________________________________________________
void TH3C::SetBinsLength(Int_t n)
{
// Set total number of bins including under/overflow
// Reallocate bin contents array
   
   if (n < 0) n = (fXaxis.GetNbins()+2)*(fYaxis.GetNbins()+2)*(fZaxis.GetNbins()+2);
   fNcells = n;
   TArrayC::Set(n);
}

//______________________________________________________________________________
void TH3C::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH3C.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      TFile *file = (TFile*)R__b.GetParent();
      if (file && file->GetVersion() < 22300) return;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TH3C::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      if (R__v < 2) {
         R__b.ReadVersion();
         TH1::Streamer(R__b);
         TArrayC::Streamer(R__b);
         R__b.ReadVersion(&R__s, &R__c);
         TAtt3D::Streamer(R__b);
      } else {
         TH3::Streamer(R__b);
         TArrayC::Streamer(R__b);
         R__b.CheckByteCount(R__s, R__c, TH3C::IsA());
      }
      //====end of old versions
      
   } else {
      TH3C::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
TH3C& TH3C::operator=(const TH3C &h1)
{
   if (this != &h1)  ((TH3C&)h1).Copy(*this);
   return *this;
}

//______________________________________________________________________________
TH3C operator*(Float_t c1, TH3C &h1)
{
   TH3C hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3C operator+(TH3C &h1, TH3C &h2)
{
   TH3C hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3C operator-(TH3C &h1, TH3C &h2)
{
   TH3C hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3C operator*(TH3C &h1, TH3C &h2)
{
   TH3C hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3C operator/(TH3C &h1, TH3C &h2)
{
   TH3C hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

ClassImp(TH3S)

//______________________________________________________________________________
//                     TH3S methods
//______________________________________________________________________________
TH3S::TH3S(): TH3()
{
}

//______________________________________________________________________________
TH3S::~TH3S()
{
}

//______________________________________________________________________________
TH3S::TH3S(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup
                                     ,Int_t nbinsz,Axis_t zlow,Axis_t zup)
     :TH3(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup,nbinsz,zlow,zup)
{
//*-*-*-*-*-*-*-*-*Normal constructor for fix bin size 3-D histograms*-*-*-*-*
//*-*              ==================================================
   TArrayS::Set(fNcells);
}

//______________________________________________________________________________
TH3S::TH3S(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                             ,Int_t nbinsy,const Float_t *ybins
                                             ,Int_t nbinsz,const Float_t *zbins)
     :TH3(name,title,nbinsx,xbins,nbinsy,ybins,nbinsz,zbins)
{
//*-*-*-*-*-*-*-*Normal constructor for variable bin size 3-D histograms*-*-*-*
//*-*            =======================================================
   TArrayS::Set(fNcells);
}

//______________________________________________________________________________
TH3S::TH3S(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                             ,Int_t nbinsy,const Double_t *ybins
                                             ,Int_t nbinsz,const Double_t *zbins)
     :TH3(name,title,nbinsx,xbins,nbinsy,ybins,nbinsz,zbins)
{
//*-*-*-*-*-*-*-*Normal constructor for variable bin size 3-D histograms*-*-*-*
//*-*            =======================================================
   TArrayS::Set(fNcells);
}

//______________________________________________________________________________
TH3S::TH3S(const TH3S &h3s)
{
   ((TH3S&)h3s).Copy(*this);
}

//______________________________________________________________________________
void TH3S::AddBinContent(Int_t bin)
{
//*-*-*-*-*-*-*-*-*-*Increment bin content by 1*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==========================

   if (fArray[bin] < 32767) fArray[bin]++;
}

//______________________________________________________________________________
void TH3S::AddBinContent(Int_t bin, Stat_t w)
{
//*-*-*-*-*-*-*-*-*-*Increment bin content by w*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==========================

   Int_t newval = fArray[bin] + Int_t(w);
   if (newval > -32768 && newval < 32768) {fArray[bin] = Short_t(newval); return;}
   if (newval < -32767) fArray[bin] = -32767;
   if (newval >  32767) fArray[bin] =  32767;
}

//______________________________________________________________________________
void TH3S::Copy(TObject &newth3)
{
//*-*-*-*-*-*-*Copy this 3-D histogram structure to newth3*-*-*-*-*-*-*-*-*-*
//*-*          ===========================================

   TH3::Copy((TH3S&)newth3);
   TArrayS::Copy((TH3S&)newth3);
}

//______________________________________________________________________________
TH1 *TH3S::DrawCopy(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH3S *newth3 = (TH3S*)Clone();
   newth3->SetDirectory(0);
   newth3->SetBit(kCanDelete);
   newth3->AppendPad(option);
   return newth3;
}

//______________________________________________________________________________
Stat_t TH3S::GetBinContent(Int_t bin) const
{
   if (fBuffer) ((TH3S*)this)->BufferEmpty();
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   return Stat_t (fArray[bin]);
}

//______________________________________________________________________________
void TH3S::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*Reset this histogram: contents, errors, etc*-*-*-*-*-*-*-*
//*-*            ===========================================

   TH3::Reset(option);
   TArrayS::Reset();
   // should also reset statistics once statistics are implemented for TH3
}

//______________________________________________________________________________
void TH3S::SetBinsLength(Int_t n)
{
// Set total number of bins including under/overflow
// Reallocate bin contents array
   
   if (n < 0) n = (fXaxis.GetNbins()+2)*(fYaxis.GetNbins()+2)*(fZaxis.GetNbins()+2);
   fNcells = n;
   TArrayS::Set(n);
}

//______________________________________________________________________________
void TH3S::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH3S.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      TFile *file = (TFile*)R__b.GetParent();
      if (file && file->GetVersion() < 22300) return;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TH3S::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      if (R__v < 2) {
         R__b.ReadVersion();
         TH1::Streamer(R__b);
         TArrayS::Streamer(R__b);
         R__b.ReadVersion(&R__s, &R__c);
         TAtt3D::Streamer(R__b);
      } else {
         TH3::Streamer(R__b);
         TArrayS::Streamer(R__b);
         R__b.CheckByteCount(R__s, R__c, TH3S::IsA());
      }
      //====end of old versions
      
   } else {
      TH3S::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
TH3S& TH3S::operator=(const TH3S &h1)
{
   if (this != &h1)  ((TH3S&)h1).Copy(*this);
   return *this;
}

//______________________________________________________________________________
TH3S operator*(Float_t c1, TH3S &h1)
{
   TH3S hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3S operator+(TH3S &h1, TH3S &h2)
{
   TH3S hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3S operator-(TH3S &h1, TH3S &h2)
{
   TH3S hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3S operator*(TH3S &h1, TH3S &h2)
{
   TH3S hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3S operator/(TH3S &h1, TH3S &h2)
{
   TH3S hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

ClassImp(TH3F)

//______________________________________________________________________________
//                     TH3F methods
//______________________________________________________________________________
TH3F::TH3F(): TH3()
{
}

//______________________________________________________________________________
TH3F::~TH3F()
{
}

//______________________________________________________________________________
TH3F::TH3F(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup
                                     ,Int_t nbinsz,Axis_t zlow,Axis_t zup)
     :TH3(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup,nbinsz,zlow,zup)
{
//*-*-*-*-*-*-*-*-*Normal constructor for fix bin size 3-D histograms*-*-*-*-*
//*-*              ==================================================
   TArrayF::Set(fNcells);
}

//______________________________________________________________________________
TH3F::TH3F(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                             ,Int_t nbinsy,const Float_t *ybins
                                             ,Int_t nbinsz,const Float_t *zbins)
     :TH3(name,title,nbinsx,xbins,nbinsy,ybins,nbinsz,zbins)
{
//*-*-*-*-*-*-*-*Normal constructor for variable bin size 3-D histograms*-*-*-*
//*-*            =======================================================
   TArrayF::Set(fNcells);
}

//______________________________________________________________________________
TH3F::TH3F(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                             ,Int_t nbinsy,const Double_t *ybins
                                             ,Int_t nbinsz,const Double_t *zbins)
     :TH3(name,title,nbinsx,xbins,nbinsy,ybins,nbinsz,zbins)
{
//*-*-*-*-*-*-*-*Normal constructor for variable bin size 3-D histograms*-*-*-*
//*-*            =======================================================
   TArrayF::Set(fNcells);
}

//______________________________________________________________________________
TH3F::TH3F(const TH3F &h3f)
{
   ((TH3F&)h3f).Copy(*this);
}

//______________________________________________________________________________
void TH3F::Copy(TObject &newth3)
{
//*-*-*-*-*-*-*Copy this 3-D histogram structure to newth3*-*-*-*-*-*-*-*-*-*
//*-*          ===========================================

   TH3::Copy((TH3F&)newth3);
   TArrayF::Copy((TH3F&)newth3);
}

//______________________________________________________________________________
TH1 *TH3F::DrawCopy(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH3F *newth3 = (TH3F*)Clone();
   newth3->SetDirectory(0);
   newth3->SetBit(kCanDelete);
   newth3->AppendPad(option);
   return newth3;
}

//______________________________________________________________________________
Stat_t TH3F::GetBinContent(Int_t bin) const
{
   if (fBuffer) ((TH3F*)this)->BufferEmpty();
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   return Stat_t (fArray[bin]);
}

//______________________________________________________________________________
void TH3F::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*Reset this histogram: contents, errors, etc*-*-*-*-*-*-*-*
//*-*            ===========================================

   TH3::Reset(option);
   TArrayF::Reset();
   // should also reset statistics once statistics are implemented for TH3
}

//______________________________________________________________________________
void TH3F::SetBinsLength(Int_t n)
{
// Set total number of bins including under/overflow
// Reallocate bin contents array
   
   if (n < 0) n = (fXaxis.GetNbins()+2)*(fYaxis.GetNbins()+2)*(fZaxis.GetNbins()+2);
   fNcells = n;
   TArrayF::Set(n);
}

//______________________________________________________________________________
void TH3F::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH3F.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      TFile *file = (TFile*)R__b.GetParent();
      if (file && file->GetVersion() < 22300) return;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TH3F::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      if (R__v < 2) {
         R__b.ReadVersion();
         TH1::Streamer(R__b);
         TArrayF::Streamer(R__b);
         R__b.ReadVersion(&R__s, &R__c);
         TAtt3D::Streamer(R__b);
      } else {
         TH3::Streamer(R__b);
         TArrayF::Streamer(R__b);
         R__b.CheckByteCount(R__s, R__c, TH3F::IsA());
      }
      //====end of old versions
      
   } else {
      TH3F::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
TH3F& TH3F::operator=(const TH3F &h1)
{
   if (this != &h1)  ((TH3F&)h1).Copy(*this);
   return *this;
}

//______________________________________________________________________________
TH3F operator*(Float_t c1, TH3F &h1)
{
   TH3F hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3F operator+(TH3F &h1, TH3F &h2)
{
   TH3F hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3F operator-(TH3F &h1, TH3F &h2)
{
   TH3F hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3F operator*(TH3F &h1, TH3F &h2)
{
   TH3F hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3F operator/(TH3F &h1, TH3F &h2)
{
   TH3F hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

ClassImp(TH3D)

//______________________________________________________________________________
//                     TH3D methods
//______________________________________________________________________________
TH3D::TH3D(): TH3()
{
}

//______________________________________________________________________________
TH3D::~TH3D()
{
}

//______________________________________________________________________________
TH3D::TH3D(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup
                                     ,Int_t nbinsz,Axis_t zlow,Axis_t zup)
     :TH3(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup,nbinsz,zlow,zup)
{
//*-*-*-*-*-*-*-*-*Normal constructor for fix bin size 3-D histograms*-*-*-*-*
//*-*              ==================================================
   TArrayD::Set(fNcells);
}

//______________________________________________________________________________
TH3D::TH3D(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                             ,Int_t nbinsy,const Float_t *ybins
                                             ,Int_t nbinsz,const Float_t *zbins)
     :TH3(name,title,nbinsx,xbins,nbinsy,ybins,nbinsz,zbins)
{
//*-*-*-*-*-*-*-*Normal constructor for variable bin size 3-D histograms*-*-*-*
//*-*            =======================================================
   TArrayD::Set(fNcells);
}

//______________________________________________________________________________
TH3D::TH3D(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                             ,Int_t nbinsy,const Double_t *ybins
                                             ,Int_t nbinsz,const Double_t *zbins)
     :TH3(name,title,nbinsx,xbins,nbinsy,ybins,nbinsz,zbins)
{
//*-*-*-*-*-*-*-*Normal constructor for variable bin size 3-D histograms*-*-*-*
//*-*            =======================================================
   TArrayD::Set(fNcells);
}

//______________________________________________________________________________
TH3D::TH3D(const TH3D &h3d)
{
   ((TH3D&)h3d).Copy(*this);
}

//______________________________________________________________________________
void TH3D::Copy(TObject &newth3)
{
//*-*-*-*-*-*-*Copy this 3-D histogram structure to newth3*-*-*-*-*-*-*-*-*-*
//*-*          ===========================================

   TH3::Copy((TH3D&)newth3);
   TArrayD::Copy((TH3D&)newth3);
}

//______________________________________________________________________________
TH1 *TH3D::DrawCopy(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH3D *newth3 = (TH3D*)Clone();
   newth3->SetDirectory(0);
   newth3->SetBit(kCanDelete);
   newth3->AppendPad(option);
   return newth3;
}

//______________________________________________________________________________
Stat_t TH3D::GetBinContent(Int_t bin) const
{
   if (fBuffer) ((TH3D*)this)->BufferEmpty();
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   return Stat_t (fArray[bin]);
}

//______________________________________________________________________________
void TH3D::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*Reset this histogram: contents, errors, etc*-*-*-*-*-*-*-*
//*-*            ===========================================

   TH3::Reset(option);
   TArrayD::Reset();
   // should also reset statistics once statistics are implemented for TH3
}

//______________________________________________________________________________
void TH3D::SetBinsLength(Int_t n)
{
// Set total number of bins including under/overflow
// Reallocate bin contents array
   
   if (n < 0) n = (fXaxis.GetNbins()+2)*(fYaxis.GetNbins()+2)*(fZaxis.GetNbins()+2);
   fNcells = n;
   TArrayD::Set(n);
}

//______________________________________________________________________________
void TH3D::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH3D.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      TFile *file = (TFile*)R__b.GetParent();
      if (file && file->GetVersion() < 22300) return;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TH3D::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      if (R__v < 2) {
         R__b.ReadVersion();
         TH1::Streamer(R__b);
         TArrayD::Streamer(R__b);
         R__b.ReadVersion(&R__s, &R__c);
         TAtt3D::Streamer(R__b);
      } else {
         TH3::Streamer(R__b);
         TArrayD::Streamer(R__b);
         R__b.CheckByteCount(R__s, R__c, TH3D::IsA());
      }
      //====end of old versions
      
   } else {
      TH3D::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
TH3D& TH3D::operator=(const TH3D &h1)
{
   if (this != &h1)  ((TH3D&)h1).Copy(*this);
   return *this;
}

//______________________________________________________________________________
TH3D operator*(Float_t c1, TH3D &h1)
{
   TH3D hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3D operator+(TH3D &h1, TH3D &h2)
{
   TH3D hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3D operator-(TH3D &h1, TH3D &h2)
{
   TH3D hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3D operator*(TH3D &h1, TH3D &h2)
{
   TH3D hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH3D operator/(TH3D &h1, TH3D &h2)
{
   TH3D hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}
