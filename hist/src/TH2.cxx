// @(#)root/hist:$Name:  $:$Id: TH2.cxx,v 1.4 2000/06/15 06:51:49 brun Exp $
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TH2.h"
#include "TVirtualPad.h"
#include "TF2.h"
#include "TProfile.h"
#include "TRandom.h"

ClassImp(TH2)

//______________________________________________________________________________
//
// Service class for 2-Dim histogram classes
//
//  TH2C a 2-D histogram with one byte per cell (char)
//  TH2S a 2-D histogram with two bytes per cell (short integer)
//  TH2F a 2-D histogram with four bytes per cell (float)
//  TH2D a 2-D histogram with eight bytes per cell (double)
//

//______________________________________________________________________________
TH2::TH2()
{
   fDimension   = 2;
   fScalefactor = 1;
   fTsumwy      = fTsumwy2 = fTsumwxy = 0;
}

//______________________________________________________________________________
TH2::TH2(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup)
     :TH1(name,title,nbinsx,xlow,xup)
{
   fDimension   = 2;
   fScalefactor = 1;
   fTsumwy      = fTsumwy2 = fTsumwxy = 0;
   if (nbinsy <= 0) nbinsy = 1;
   fYaxis.Set(nbinsy,ylow,yup);
   fNcells      = (nbinsx+2)*(nbinsy+2);
}

//______________________________________________________________________________
TH2::TH2(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup)
     :TH1(name,title,nbinsx,xbins)
{
   fDimension   = 2;
   fScalefactor = 1;
   fTsumwy      = fTsumwy2 = fTsumwxy = 0;
   if (nbinsy <= 0) nbinsy = 1;
   fYaxis.Set(nbinsy,ylow,yup);
   fNcells      = (nbinsx+2)*(nbinsy+2);
}

//______________________________________________________________________________
TH2::TH2(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Double_t *ybins)
     :TH1(name,title,nbinsx,xlow,xup)
{
   fDimension   = 2;
   fScalefactor = 1;
   fTsumwy      = fTsumwy2 = fTsumwxy = 0;
   if (nbinsy <= 0) nbinsy = 1;
   if (ybins) fYaxis.Set(nbinsy,ybins);
   else       fYaxis.Set(nbinsy,0,1);
   fNcells      = (nbinsx+2)*(nbinsy+2);
}

//______________________________________________________________________________
TH2::TH2(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                           ,Int_t nbinsy,Double_t *ybins)
     :TH1(name,title,nbinsx,xbins)
{
   fDimension   = 2;
   fScalefactor = 1;
   fTsumwy      = fTsumwy2 = fTsumwxy = 0;
   if (nbinsy <= 0) nbinsy = 1;
   if (ybins) fYaxis.Set(nbinsy,ybins);
   else       fYaxis.Set(nbinsy,0,1);
   fNcells      = (nbinsx+2)*(nbinsy+2);
}

//______________________________________________________________________________
TH2::TH2(const char *name,const char *title,Int_t nbinsx,Float_t *xbins
                                           ,Int_t nbinsy,Float_t *ybins)
     :TH1(name,title,nbinsx,xbins)
{
   fDimension   = 2;
   fScalefactor = 1;
   fTsumwy      = fTsumwy2 = fTsumwxy = 0;
   if (nbinsy <= 0) nbinsy = 1;
   if (ybins) fYaxis.Set(nbinsy,ybins);
   else       fYaxis.Set(nbinsy,0,1);
   fNcells      = (nbinsx+2)*(nbinsy+2);
}

//______________________________________________________________________________
TH2::~TH2()
{
}

//______________________________________________________________________________
void TH2::Copy(TObject &obj)
{
   TH1::Copy(obj);
   ((TH2&)obj).fScalefactor = fScalefactor;
   ((TH2&)obj).fTsumwy      = fTsumwy;
   ((TH2&)obj).fTsumwy2     = fTsumwy2;
   ((TH2&)obj).fTsumwxy     = fTsumwxy;
}

//______________________________________________________________________________
Int_t TH2::Fill(Axis_t x,Axis_t y)
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
   Int_t binx, biny, bin;
   fEntries++;
   binx = fXaxis.FindBin(x);
   biny = fYaxis.FindBin(y);
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin);
   if (fSumw2.fN) ++fSumw2.fArray[bin];
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
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
Int_t TH2::Fill(Axis_t x, Axis_t y, Stat_t w)
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

   Int_t binx, biny, bin;
   fEntries++;
   binx = fXaxis.FindBin(x);
   biny = fYaxis.FindBin(y);
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin,w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   Stat_t z= (w > 0 ? w : -w);
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
void TH2::FillN(Int_t ntimes, Axis_t *x, Axis_t *y, Double_t *w, Int_t stride)
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
      if (binx == 0 || binx > fXaxis.GetNbins()) continue;
      if (biny == 0 || biny > fYaxis.GetNbins()) continue;
      Stat_t z= (ww > 0 ? ww : -ww);
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
//*-*   The distribution contained in the histogram h (TH1) is integrated
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
   Axis_t x,y;
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
//     "Q" means Quiet mode
//     "N" means do not show the result of the fit
//     "R" means fit the function in the specified function range
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

   Int_t nbins  = fYaxis.GetNbins();
   if (binmin < 1) binmin = 1;
   if (binmax > nbins) binmax = nbins;
   if (binmax < binmin) {binmin = 1; binmax = nbins;}

   //default is to fit with a gaussian
   if (f1 == 0) {
      f1 = (TF1*)gROOT->GetFunction("gaus");
      if (f1 == 0) f1 = new TF1("gaus","gaus",fXaxis.GetXmin(),fXaxis.GetXmax());
      else         f1->SetRange(fXaxis.GetXmin(),fXaxis.GetXmax());
   }
   const char *fname = f1->GetName();
   Int_t npar = f1->GetNpar();
   Double_t *parsave = new Double_t[npar];
   f1->GetParameters(parsave);

   //Create one histogram for each function parameter
   Int_t ipar;
   char name[80], title[80];
   TH1D *hlist[25];
   TArrayD *bins = fYaxis.GetXbins();
   for (ipar=0;ipar<npar;ipar++) {
      sprintf(name,"%s_%d",GetName(),ipar);
      sprintf(title,"Fitted value of par[%d]=%s",ipar,f1->GetParName(ipar));
      if (bins->fN == 0) {
         hlist[ipar] = new TH1D(name,title, nbins, fYaxis.GetXmin(), fYaxis.GetXmax());
      } else {
         hlist[ipar] = new TH1D(name,title, nbins,bins->fArray);
      }
      hlist[ipar]->GetXaxis()->SetTitle(fYaxis.GetTitle());
   }
   sprintf(name,"%s_chi2",GetName());
   TH1D *hchi2 = new TH1D(name,"chisquare", nbins, fYaxis.GetXmin(), fYaxis.GetXmax());
   hchi2->GetXaxis()->SetTitle(fYaxis.GetTitle());

   //Loop on all bins in Y, generate a projection along X
   Int_t bin;
   Int_t nentries;
   for (bin=binmin;bin<=binmax;bin++) {
      TH1D *hpx = ProjectionX("_temp",bin,bin,"e");
      if (hpx == 0) continue;
      nentries = Int_t(hpx->GetEntries());
      if (nentries == 0 || nentries < cut) {delete hpx; continue;}
      f1->SetParameters(parsave);
      hpx->Fit(fname,option);
      Int_t npfits = f1->GetNumberFitPoints();
      if (npfits > npar && npfits >= cut) {
         for (ipar=0;ipar<npar;ipar++) {
            hlist[ipar]->Fill(fYaxis.GetBinCenter(bin),f1->GetParameter(ipar));
            hlist[ipar]->SetBinError(bin,f1->GetParError(ipar));
         }
         hchi2->Fill(fYaxis.GetBinCenter(bin),f1->GetChisquare()/(npfits-npar));
      }
      delete hpx;
   }
   delete [] parsave;
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
//     "Q" means Quiet mode
//     "N" means do not show the result of the fit
//     "R" means fit the function in the specified function range
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

   //default is to fit with a gaussian
   if (f1 == 0) {
      f1 = (TF1*)gROOT->GetFunction("gaus");
      if (f1 == 0) f1 = new TF1("gaus","gaus",fYaxis.GetXmin(),fYaxis.GetXmax());
      else         f1->SetRange(fYaxis.GetXmin(),fYaxis.GetXmax());
   }
   const char *fname = f1->GetName();
   Int_t npar = f1->GetNpar();
   Double_t *parsave = new Double_t[npar];
   f1->GetParameters(parsave);

   //Create one histogram for each function parameter
   Int_t ipar;
   char name[80], title[80];
   TH1D *hlist[25];
   TArrayD *bins = fXaxis.GetXbins();
   for (ipar=0;ipar<npar;ipar++) {
      sprintf(name,"%s_%d",GetName(),ipar);
      sprintf(title,"Fitted value of par[%d]=%s",ipar,f1->GetParName(ipar));
      if (bins->fN == 0) {
         hlist[ipar] = new TH1D(name,title, nbins, fXaxis.GetXmin(), fXaxis.GetXmax());
      } else {
         hlist[ipar] = new TH1D(name,title, nbins,bins->fArray);
      }
      hlist[ipar]->GetXaxis()->SetTitle(fXaxis.GetTitle());
   }
   sprintf(name,"%s_chi2",GetName());
   TH1D *hchi2 = new TH1D(name,"chisquare", nbins, fXaxis.GetXmin(), fXaxis.GetXmax());
   hchi2->GetXaxis()->SetTitle(fXaxis.GetTitle());

   //Loop on all bins in X, generate a projection along Y
   Int_t bin;
   Int_t nentries;
   for (bin=binmin;bin<=binmax;bin++) {
      TH1D *hpy = ProjectionY("_temp",bin,bin,"e");
      if (hpy == 0) continue;
      nentries = Int_t(hpy->GetEntries());
      if (nentries == 0 || nentries < cut) {delete hpy; continue;}
      f1->SetParameters(parsave);
      hpy->Fit(fname,option);
      Int_t npfits = f1->GetNumberFitPoints();
      if (npfits > npar && npfits >= cut) {
         for (ipar=0;ipar<npar;ipar++) {
            hlist[ipar]->Fill(fXaxis.GetBinCenter(bin),f1->GetParameter(ipar));
            hlist[ipar]->SetBinError(bin,f1->GetParError(ipar));
         }
         hchi2->Fill(fXaxis.GetBinCenter(bin),f1->GetChisquare()/(npfits-npar));
      }
      delete hpy;
   }
   delete [] parsave;
}

//______________________________________________________________________________
Stat_t TH2::GetCorrelationFactor(Int_t axis1, Int_t axis2)
{
//*-*-*-*-*-*-*-*Return correlation factor between axis1 and axis2*-*-*-*-*
//*-*            ====================================================
  if (axis1 < 1 || axis2 < 1 || axis1 > 2 || axis2 > 2) {
     Error("GetCorrelationFactor","Wrong parameters");
     return 0;
  }
  if (axis1 == axis2) return 1;
  Stat_t rms1 = GetRMS(axis1);
  if (rms1 == 0) return 0;
  Stat_t rms2 = GetRMS(axis2);
  if (rms2 == 0) return 0;
  return GetCovariance(axis1,axis2)/rms1/rms2;
}

//______________________________________________________________________________
Stat_t TH2::GetCovariance(Int_t axis1, Int_t axis2)
{
//*-*-*-*-*-*-*-*Return covariance between axis1 and axis2*-*-*-*-*
//*-*            ====================================================

  if (axis1 < 1 || axis2 < 1 || axis1 > 2 || axis2 > 2) {
     Error("GetCovariance","Wrong parameters");
     return 0;
  }
  Stat_t stats[7];
  GetStats(stats);
  Stat_t sumw   = stats[0];
  Stat_t sumw2  = stats[1];
  Stat_t sumwx  = stats[2];
  Stat_t sumwx2 = stats[3];
  Stat_t sumwy  = stats[4];
  Stat_t sumwy2 = stats[5];
  Stat_t sumwxy = stats[6];

  if (sumw == 0) return 0;
  if (axis1 == 1 && axis2 == 1) {
     return TMath::Abs(sumwx2/sumw - sumwx*sumwx/sumw2);
  }
  if (axis1 == 2 && axis2 == 2) {
     return TMath::Abs(sumwy2/sumw - sumwy*sumwy/sumw2);
  }
  return sumwxy/sumw - sumwx/sumw*sumwy/sumw;
}

//______________________________________________________________________________
void TH2::GetRandom2(Axis_t &x, Axis_t &y)
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
   Int_t ibin = TMath::BinarySearch(nbins,&fIntegral[0],r1);
   Int_t biny = ibin/nbinsx;
   Int_t binx = ibin - nbinsx*biny;
   x = fXaxis.GetBinLowEdge(binx+1)
      +fXaxis.GetBinWidth(binx+1)*(fIntegral[ibin+1]-r1)/(fIntegral[ibin+1] - fIntegral[ibin]);
   y = fYaxis.GetBinLowEdge(biny+1) + fYaxis.GetBinWidth(biny+1)*gRandom->Rndm();
}

//______________________________________________________________________________
void TH2::GetStats(Stat_t *stats)
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

   Int_t bin, binx, biny;
   Stat_t w;
   Float_t x,y;
   if (fTsumw == 0 || fXaxis.TestBit(TAxis::kAxisRange) || fYaxis.TestBit(TAxis::kAxisRange)) {
      for (bin=0;bin<7;bin++) stats[bin] = 0;
      for (biny=fYaxis.GetFirst();biny<=fYaxis.GetLast();biny++) {
         y = fYaxis.GetBinCenter(biny);
         for (binx=fXaxis.GetFirst();binx<=fXaxis.GetLast();binx++) {
            bin = GetBin(binx,biny);
            x = fXaxis.GetBinCenter(binx);
            w = TMath::Abs(GetBinContent(bin));
            stats[0] += w;
            stats[1] += w*w;
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
Stat_t TH2::Integral()
{
//Return integral of bin contents. Only bins in the bins range are considered.

   return Integral(fXaxis.GetFirst(),fXaxis.GetLast(),
                   fYaxis.GetFirst(),fYaxis.GetLast());
}

//______________________________________________________________________________
Stat_t TH2::Integral(Int_t binx1, Int_t binx2, Int_t biny1, Int_t biny2)
{
//Return integral of bin contents in range [binx1,binx2],[biny1,biny2]
// for a 2-D histogram

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
   if (binx1 < 0) binx1 = 0;
   if (binx2 > nbinsx+1) binx2 = nbinsx+1;
   if (binx2 < binx1)    binx2 = nbinsx;
   if (biny1 < 0) biny1 = 0;
   if (biny2 > nbinsy+1) biny2 = nbinsy+1;
   if (biny2 < biny1)    biny2 = nbinsy;
   Stat_t integral = 0;

//*-*- Loop on bins in specified range
   Int_t bin, binx, biny, binz;
   for (binz=0;binz<=nbinsz+1;binz++) {
      for (biny=biny1;biny<=biny2;biny++) {
         for (binx=binx1;binx<=binx2;binx++) {
            bin = binx +(nbinsx+2)*(biny + (nbinsy+2)*binz);
            integral += GetBinContent(bin);
         }
      }
   }
   return integral;
}
        
//______________________________________________________________________________
Double_t TH2::KolmogorovTest(TH1 *h2, Option_t *option)
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
//         "L"=Left:   include x-underflows
//         "R"=Right:  include x-overflows
//         "T"=Top:    include y-overflows
//         "B"=Bottom: include y-underflows
//   for example: "OB" means x- and y-overflows and y-underflows !!
//
//   The returned function value is the probability of test
//       (much less than one means NOT compatible)
//
//  Code adapted by Rene Brun from original HBOOK routine HDIFF

   TString opt = option;
   opt.ToUpper();
   
   Double_t prb = 0;
   TH1 *h1 = this;
   if (h2 == 0) return 0;
   TAxis *axis1 = h1->GetXaxis();
   TAxis *axis2 = h2->GetXaxis();
   Int_t ncx1   = axis1->GetNbins();
   Int_t ncx2   = axis2->GetNbins();

     // Check consistency of dimensions
   if (h1->GetDimension() != 2 || h2->GetDimension() != 2) {
      Error("KolmogorovTest","Histograms must be 2-D\n");
      return 0;
   }

   printf(" NOT YET IMPLEMENTED\n");
   return 0;
}   
   
//______________________________________________________________________________
TProfile *TH2::ProfileX(const char *name, Int_t firstybin, Int_t lastybin, Option_t *option)
{
//*-*-*-*-*Project a 2-D histogram into a profile histogram along X*-*-*-*-*-*
//*-*      ========================================================
//
//   The projection is made from the channels along the Y axis
//   ranging from firstybin to lastybin included.
//

  TString opt = option;
  opt.ToLower();
  Int_t nx = fXaxis.GetNbins();
  Int_t ny = fYaxis.GetNbins();
  if (firstybin < 0) firstybin = 0;
  if (lastybin > ny) lastybin = ny+1;

// Create the profile histogram
  char *pname = (char*)name;
  if (strcmp(name,"_pfx") == 0) {
     Int_t nch = strlen(GetName()) + 5;
     pname = new char[nch];
     sprintf(pname,"%s%s",GetName(),name);
  }
  TProfile *h1;
  TArrayD *bins = fYaxis.GetXbins();
  if (bins->fN == 0) {
     h1 = new TProfile(pname,GetTitle(),nx,fXaxis.GetXmin(),fXaxis.GetXmax(),option);
  } else {
     h1 = new TProfile(pname,GetTitle(),nx,bins->fArray,option);
  }
  if (pname != name)  delete [] pname;

// Fill the profile histogram
  Double_t cont;
  for (Int_t binx =0;binx<=nx+1;binx++) {
     for (Int_t biny=firstybin;biny<=lastybin;biny++) {
        cont =  GetCellContent(binx,biny);
        if (cont) {
           h1->Fill(fXaxis.GetBinCenter(binx),fYaxis.GetBinCenter(biny), cont);
        }
     }
  }
  return h1;
}

//______________________________________________________________________________
TProfile *TH2::ProfileY(const char *name, Int_t firstxbin, Int_t lastxbin, Option_t *option)
{
//*-*-*-*-*Project a 2-D histogram into a profile histogram along Y*-*-*-*-*-*
//*-*      ========================================================
//
//   The projection is made from the channels along the X axis
//   ranging from firstxbin to lastxbin included.
//

  TString opt = option;
  opt.ToLower();
  Int_t nx = fXaxis.GetNbins();
  Int_t ny = fYaxis.GetNbins();
  if (firstxbin < 0) firstxbin = 0;
  if (lastxbin > nx) lastxbin = nx+1;

// Create the projection histogram
  char *pname = (char*)name;
  if (strcmp(name,"_pfy") == 0) {
     Int_t nch = strlen(GetName()) + 5;
     pname = new char[nch];
     sprintf(pname,"%s%s",GetName(),name);
  }
  TProfile *h1;
  TArrayD *bins = fYaxis.GetXbins();
  if (bins->fN == 0) {
     h1 = new TProfile(pname,GetTitle(),ny,fYaxis.GetXmin(),fYaxis.GetXmax(),option);
  } else {
     h1 = new TProfile(pname,GetTitle(),ny,bins->fArray,option);
  }
  if (pname != name)  delete [] pname;

// Fill the profile histogram
  Double_t cont;
  for (Int_t biny =0;biny<=ny+1;biny++) {
     for (Int_t binx=firstxbin;binx<=lastxbin;binx++) {
        cont =  GetCellContent(binx,biny);
        if (cont) {
           h1->Fill(fYaxis.GetBinCenter(biny),fXaxis.GetBinCenter(binx), cont);
        }
     }
  }
  return h1;
}

//______________________________________________________________________________
TH1D *TH2::ProjectionX(const char *name, Int_t firstybin, Int_t lastybin, Option_t *option)
{
//*-*-*-*-*Project a 2-D histogram into a 1-D histogram along X*-*-*-*-*-*-*
//*-*      ====================================================
//
//   The projection is always of the type TH1D.
//   The projection is made from the channels along the Y axis
//   ranging from firstybin to lastybin included.
//
//   if option "E" is specified, the errors are computed.
//

  TString opt = option;
  opt.ToLower();
  Int_t nx = fXaxis.GetNbins();
  Int_t ny = fYaxis.GetNbins();
  if (firstybin < 0) firstybin = 0;
  if (lastybin > ny) lastybin = ny+1;

// Create the projection histogram
  char *pname = (char*)name;
  if (strcmp(name,"_px") == 0) {
     Int_t nch = strlen(GetName()) + 4;
     pname = new char[nch];
     sprintf(pname,"%s%s",GetName(),name);
  }
  TH1D *h1;
  TArrayD *bins = fXaxis.GetXbins();
  if (bins->fN == 0) {
     h1 = new TH1D(pname,GetTitle(),nx,fXaxis.GetXmin(),fXaxis.GetXmax());
  } else {
     h1 = new TH1D(pname,GetTitle(),nx,bins->fArray);
  }
  Bool_t computeErrors = kFALSE;
  if (opt.Contains("e")) {h1->Sumw2(); computeErrors = kTRUE;}
  if (pname != name)  delete [] pname;

// Fill the projected histogram
  Double_t cont,err,err2;
  for (Int_t binx =0;binx<=nx+1;binx++) {
     err2 = 0;
     for (Int_t biny=firstybin;biny<=lastybin;biny++) {
        cont  = GetCellContent(binx,biny);
        err   = GetCellError(binx,biny);
        err2 += err*err;
        if (cont) {
           h1->Fill(fXaxis.GetBinCenter(binx), cont);
        }
     }
     if (computeErrors) h1->SetBinError(binx,TMath::Sqrt(err2));
  }
  return h1;
}

//______________________________________________________________________________
TH1D *TH2::ProjectionY(const char *name, Int_t firstxbin, Int_t lastxbin, Option_t *option)
{
//*-*-*-*-*Project a 2-D histogram into a 1-D histogram along Y*-*-*-*-*-*-*
//*-*      ====================================================
//
//   The projection is always of the type TH1D.
//   The projection is made from the channels along the X axis
//   ranging from firstxbin to lastxbin included.
//
//   if option "E" is specified, the errors are computed.
//

  TString opt = option;
  opt.ToLower();
  Int_t nx = fXaxis.GetNbins();
  Int_t ny = fYaxis.GetNbins();
  if (firstxbin < 0) firstxbin = 0;
  if (lastxbin > nx) lastxbin = nx+1;

// Create the projection histogram
  char *pname = (char*)name;
  if (strcmp(name,"_py") == 0) {
     Int_t nch = strlen(GetName()) + 4;
     pname = new char[nch];
     sprintf(pname,"%s%s",GetName(),name);
  }
  TH1D *h1;
  TArrayD *bins = fYaxis.GetXbins();
  if (bins->fN == 0) {
     h1 = new TH1D(pname,GetTitle(),ny,fYaxis.GetXmin(),fYaxis.GetXmax());
  } else {
     h1 = new TH1D(pname,GetTitle(),ny,bins->fArray);
  }
  Bool_t computeErrors = kFALSE;
  if (opt.Contains("e")) {h1->Sumw2(); computeErrors = kTRUE;}
  if (pname != name)  delete [] pname;

// Fill the projected histogram
  Double_t cont,err,err2;
  for (Int_t biny =0;biny<=ny+1;biny++) {
     err2 = 0;
     for (Int_t binx=firstxbin;binx<=lastxbin;binx++) {
        cont  = GetCellContent(binx,biny);
        err   = GetCellError(binx,biny);
        err2 += err*err;
        if (cont) {
           h1->Fill(fYaxis.GetBinCenter(biny), cont);
        }
     }
     if (computeErrors) h1->SetBinError(biny,TMath::Sqrt(err2));
  }
  return h1;
}

//______________________________________________________________________________
void TH2::PutStats(Stat_t *stats)
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
   fTsumwy  = 0;
   fTsumwy2 = 0;
   fTsumwxy = 0;
}

//______________________________________________________________________________
void TH2::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH2.

   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(); if (R__v) { }
      TH1::Streamer(R__b);
      R__b >> fScalefactor;
      R__b >> fTsumwy;
      R__b >> fTsumwy2;
      R__b >> fTsumwxy;
   } else {
      R__b.WriteVersion(TH2::IsA());
      TH1::Streamer(R__b);
      R__b << fScalefactor;
      R__b << fTsumwy;
      R__b << fTsumwy2;
      R__b << fTsumwxy;
   }
}

ClassImp(TH2C)

//______________________________________________________________________________
//                     TH2C methods
//______________________________________________________________________________
TH2C::TH2C(): TH2()
{
}

//______________________________________________________________________________
TH2C::~TH2C()
{
}

//______________________________________________________________________________
TH2C::TH2C(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup)
     :TH2(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup)
{
   TArrayC::Set(fNcells);
}

//______________________________________________________________________________
TH2C::TH2C(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup)
     :TH2(name,title,nbinsx,xbins,nbinsy,ylow,yup)
{
   TArrayC::Set(fNcells);
}

//______________________________________________________________________________
TH2C::TH2C(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Double_t *ybins)
     :TH2(name,title,nbinsx,xlow,xup,nbinsy,ybins)
{
   TArrayC::Set(fNcells);
}

//______________________________________________________________________________
TH2C::TH2C(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                             ,Int_t nbinsy,Double_t *ybins)
     :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   TArrayC::Set(fNcells);
}

//______________________________________________________________________________
TH2C::TH2C(const char *name,const char *title,Int_t nbinsx,Float_t *xbins
                                             ,Int_t nbinsy,Float_t *ybins)
     :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   TArrayC::Set(fNcells);
}

//______________________________________________________________________________
TH2C::TH2C(const TH2C &h2c)
{
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
void TH2C::AddBinContent(Int_t bin, Stat_t w)
{
//*-*-*-*-*-*-*-*-*-*Increment bin content by w*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==========================

   Int_t newval = fArray[bin] + Int_t(w);
   if (newval > -128 && newval < 128) {fArray[bin] = Char_t(newval); return;}
   if (newval < -127) fArray[bin] = -127;
   if (newval >  127) fArray[bin] =  127;
}

//______________________________________________________________________________
void TH2C::Copy(TObject &newth2)
{
   TH2::Copy((TH2C&)newth2);
   TArrayC::Copy((TH2C&)newth2);
}

//______________________________________________________________________________
TH1 *TH2C::DrawCopy(Option_t *option)
{
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
Stat_t TH2C::GetBinContent(Int_t bin)
{
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   return Stat_t (fArray[bin]);
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
void TH2C::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH2C.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
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
   } else {
      R__c = R__b.WriteVersion(TH2C::IsA(), kTRUE);
      TH2::Streamer(R__b);
      TArrayC::Streamer(R__b);
      R__b.SetByteCount(R__c, kTRUE);
   }
}

//______________________________________________________________________________
TH2C& TH2C::operator=(const TH2C &h1)
{
   if (this != &h1)  ((TH2C&)h1).Copy(*this);
   return *this;
}


//______________________________________________________________________________
TH2C operator*(Float_t c1, TH2C &h1)
{
   TH2C hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2C operator+(TH2C &h1, TH2C &h2)
{
   TH2C hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2C operator-(TH2C &h1, TH2C &h2)
{
   TH2C hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2C operator*(TH2C &h1, TH2C &h2)
{
   TH2C hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2C operator/(TH2C &h1, TH2C &h2)
{
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
}

//______________________________________________________________________________
TH2S::~TH2S()
{

}

//______________________________________________________________________________
TH2S::TH2S(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup)
     :TH2(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup)
{
   TArrayS::Set(fNcells);
}

//______________________________________________________________________________
TH2S::TH2S(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup)
     :TH2(name,title,nbinsx,xbins,nbinsy,ylow,yup)
{
   TArrayS::Set(fNcells);
}

//______________________________________________________________________________
TH2S::TH2S(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Double_t *ybins)
     :TH2(name,title,nbinsx,xlow,xup,nbinsy,ybins)
{
   TArrayS::Set(fNcells);
}

//______________________________________________________________________________
TH2S::TH2S(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                             ,Int_t nbinsy,Double_t *ybins)
     :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   TArrayS::Set(fNcells);
}

//______________________________________________________________________________
TH2S::TH2S(const char *name,const char *title,Int_t nbinsx,Float_t *xbins
                                             ,Int_t nbinsy,Float_t *ybins)
     :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   TArrayS::Set(fNcells);
}

//______________________________________________________________________________
TH2S::TH2S(const TH2S &h2s)
{
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
void TH2S::AddBinContent(Int_t bin, Stat_t w)
{
//*-*-*-*-*-*-*-*-*-*Increment bin content by w*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==========================

   Int_t newval = fArray[bin] + Int_t(w);
   if (newval > -32768 && newval < 32768) {fArray[bin] = Short_t(newval); return;}
   if (newval < -32767) fArray[bin] = -32767;
   if (newval >  32767) fArray[bin] =  32767;
}

//______________________________________________________________________________
void TH2S::Copy(TObject &newth2)
{
   TH2::Copy((TH2S&)newth2);
   TArrayS::Copy((TH2S&)newth2);
}

//______________________________________________________________________________
TH1 *TH2S::DrawCopy(Option_t *option)
{
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
Stat_t TH2S::GetBinContent(Int_t bin)
{
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   return Stat_t (fArray[bin]);
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
void TH2S::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH2S.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
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
   } else {
      R__c = R__b.WriteVersion(TH2S::IsA(), kTRUE);
      TH2::Streamer(R__b);
      TArrayS::Streamer(R__b);
      R__b.SetByteCount(R__c, kTRUE);
   }
}

//______________________________________________________________________________
TH2S& TH2S::operator=(const TH2S &h1)
{
   if (this != &h1)  ((TH2S&)h1).Copy(*this);
   return *this;
}


//______________________________________________________________________________
TH2S operator*(Float_t c1, TH2S &h1)
{
   TH2S hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2S operator+(TH2S &h1, TH2S &h2)
{
   TH2S hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2S operator-(TH2S &h1, TH2S &h2)
{
   TH2S hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2S operator*(TH2S &h1, TH2S &h2)
{
   TH2S hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2S operator/(TH2S &h1, TH2S &h2)
{
   TH2S hnew = h1;
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
}

//______________________________________________________________________________
TH2F::~TH2F()
{
}

//______________________________________________________________________________
TH2F::TH2F(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup)
     :TH2(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup)
{
   TArrayF::Set(fNcells);
}

//______________________________________________________________________________
TH2F::TH2F(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup)
     :TH2(name,title,nbinsx,xbins,nbinsy,ylow,yup)
{

   TArrayF::Set(fNcells);
}

//______________________________________________________________________________
TH2F::TH2F(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Double_t *ybins)
     :TH2(name,title,nbinsx,xlow,xup,nbinsy,ybins)
{

   TArrayF::Set(fNcells);
}

//______________________________________________________________________________
TH2F::TH2F(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                             ,Int_t nbinsy,Double_t *ybins)
     :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   TArrayF::Set(fNcells);
}

//______________________________________________________________________________
TH2F::TH2F(const char *name,const char *title,Int_t nbinsx,Float_t *xbins
                                             ,Int_t nbinsy,Float_t *ybins)
     :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   TArrayF::Set(fNcells);
}

//______________________________________________________________________________
TH2F::TH2F(const TH2F &h2f)
{
   ((TH2F&)h2f).Copy(*this);
}

//______________________________________________________________________________
void TH2F::Copy(TObject &newth2)
{
   TH2::Copy((TH2F&)newth2);
   TArrayF::Copy((TH2F&)newth2);
}

//______________________________________________________________________________
TH1 *TH2F::DrawCopy(Option_t *option)
{
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
Stat_t TH2F::GetBinContent(Int_t bin)
{
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   return Stat_t (fArray[bin]);
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
void TH2F::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH2F.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
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
   } else {
      R__c = R__b.WriteVersion(TH2F::IsA(), kTRUE);
      TH2::Streamer(R__b);
      TArrayF::Streamer(R__b);
      R__b.SetByteCount(R__c, kTRUE);
   }
}

//______________________________________________________________________________
TH2F& TH2F::operator=(const TH2F &h1)
{
   if (this != &h1)  ((TH2F&)h1).Copy(*this);
   return *this;
}


//______________________________________________________________________________
TH2F operator*(Float_t c1, TH2F &h1)
{
   TH2F hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}


//______________________________________________________________________________
TH2F operator*(TH2F &h1, Float_t c1)
{
   TH2F hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2F operator+(TH2F &h1, TH2F &h2)
{
   TH2F hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2F operator-(TH2F &h1, TH2F &h2)
{
   TH2F hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2F operator*(TH2F &h1, TH2F &h2)
{
   TH2F hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2F operator/(TH2F &h1, TH2F &h2)
{
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
}

//______________________________________________________________________________
TH2D::~TH2D()
{
}

//______________________________________________________________________________
TH2D::TH2D(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup)
     :TH2(name,title,nbinsx,xlow,xup,nbinsy,ylow,yup)
{
   TArrayD::Set(fNcells);
}

//______________________________________________________________________________
TH2D::TH2D(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                     ,Int_t nbinsy,Axis_t ylow,Axis_t yup)
     :TH2(name,title,nbinsx,xbins,nbinsy,ylow,yup)
{
   TArrayD::Set(fNcells);
}

//______________________________________________________________________________
TH2D::TH2D(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                     ,Int_t nbinsy,Double_t *ybins)
     :TH2(name,title,nbinsx,xlow,xup,nbinsy,ybins)
{
   TArrayD::Set(fNcells);
}

//______________________________________________________________________________
TH2D::TH2D(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                             ,Int_t nbinsy,Double_t *ybins)
     :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   TArrayD::Set(fNcells);
}

//______________________________________________________________________________
TH2D::TH2D(const char *name,const char *title,Int_t nbinsx,Float_t *xbins
                                             ,Int_t nbinsy,Float_t *ybins)
     :TH2(name,title,nbinsx,xbins,nbinsy,ybins)
{
   TArrayD::Set(fNcells);
}

//______________________________________________________________________________
TH2D::TH2D(const TH2D &h2d)
{
   ((TH2D&)h2d).Copy(*this);
}

//______________________________________________________________________________
void TH2D::Copy(TObject &newth2)
{
   TH2::Copy((TH2D&)newth2);
   TArrayD::Copy((TH2D&)newth2);
}

//______________________________________________________________________________
TH1 *TH2D::DrawCopy(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH2D *newth2 = (TH2D*)Clone();
   newth2->SetDirectory(0);
   newth2->SetBit(kCanDelete);
   newth2->AppendPad();
   return newth2;
}

//______________________________________________________________________________
Stat_t TH2D::GetBinContent(Int_t bin)
{
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   return Stat_t (fArray[bin]);
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
void TH2D::Streamer(TBuffer &R__b)
{
   // Stream an object of class TH2D.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
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
   } else {
      R__c = R__b.WriteVersion(TH2D::IsA(), kTRUE);
      TH2::Streamer(R__b);
      TArrayD::Streamer(R__b);
      R__b.SetByteCount(R__c, kTRUE);
   }
}

//______________________________________________________________________________
TH2D& TH2D::operator=(const TH2D &h1)
{
   if (this != &h1)  ((TH2D&)h1).Copy(*this);
   return *this;
}


//______________________________________________________________________________
TH2D operator*(Float_t c1, TH2D &h1)
{
   TH2D hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2D operator+(TH2D &h1, TH2D &h2)
{
   TH2D hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2D operator-(TH2D &h1, TH2D &h2)
{
   TH2D hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2D operator*(TH2D &h1, TH2D &h2)
{
   TH2D hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH2D operator/(TH2D &h1, TH2D &h2)
{
   TH2D hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}
