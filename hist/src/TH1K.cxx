// @(#)root/hist:$Name:  $:$Id: TH1K.cxx,v 1.1 2001/02/21 15:23:16 brun Exp $
// Author: Victor Perevoztchikov <perev@bnl.gov>  21/02/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdlib.h>
   
#include "TROOT.h"
#include "TH1K.h"
#include "TMath.h"

//*-**-*-*-*-*-*-*-*-*-*-*The T H 1 K   Class*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*-*-*
//*-*                  ===============================
//*-*
//*-*  TH1K class supports the nearest K Neighbours method,
//*-*       widely used in cluster analysis. 
//*-*       This method is especially useful for small statistics.
//*-*
//*-*       In this method :
//*-*         DensityOfProbability ~ 1/DistanceToNearestKthNeighbour
//*-*
//*-*      Ctr TH1K::TH1K(name,title,nbins,xlow,xup,K=0)
//*-*      differs from TH1F only by "K"
//*-*      K - is the order of K Neighbours method, usually >=3
//*-*      K = 0, means default, where K is selected by TH1K in such a way
//*-*             that DistanceToNearestKthNeighbour > BinWidth and K >=3
//*-*
//*-*  This class has been implemented by Victor Perevoztchikov <perev@bnl.gov>
//*-*  
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

ClassImp(TH1K)

//______________________________________________________________________________
TH1K::TH1K(): TH1(), TArrayF()
{
   fDimension = 1;
   fNIn   = 0;
   fReady = 0;
   fKOrd  = 3;
}

//______________________________________________________________________________
TH1K::TH1K(const char *name,const char *title,Int_t nbins,Axis_t xlow,Axis_t xup,Int_t k)
     : TH1(name,title,nbins,xlow,xup), TArrayF(100)
{
//
//    Create a 1-Dim histogram with fix bins of type float
//    ====================================================
//           (see TH1K::TH1 for explanation of parameters)
//
   fDimension = 1;
   fNIn = 0;
   fReady = 0;
   fKOrd = k;
}

//______________________________________________________________________________
TH1K::~TH1K()
{
}

//______________________________________________________________________________
void   TH1K::Reset(Option_t *option)
{
  fNIn   =0; 
  fReady = 0; 
  TH1::Reset(option);
} 
 
//______________________________________________________________________________
Int_t TH1K::Fill(Axis_t x)
{
//*-*-*-*-*-*-*-*-*-*Increment bin with abscissa X by 1*-*-*-*-*-*-*-*-*-*-*
//*-*                ==================================
//*-*
//*-* if x is less than the low-edge of the first bin, the Underflow bin is incremented
//*-* if x is greater than the upper edge of last bin, the Overflow bin is incremented
//*-*
//*-* If the storage of the sum of squares of weights has been triggered,
//*-* via the function Sumw2, then the sum of the squares of weights is incremented
//*-* by 1 in the bin corresponding to x.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fReady = 0;
   Int_t bin;
   fEntries++;
   bin =fXaxis.FindBin(x);
   if (bin == 0 || bin > fXaxis.GetNbins()) return -1;
   ++fTsumw;
   ++fTsumw2;
   fTsumwx  += x;
   fTsumwx2 += x*x;
   fReady = 0;
   if (fNIn == fN) Set(fN*2);
   AddAt(x,fNIn++);
   return bin;
}

//______________________________________________________________________________
Stat_t TH1K::GetBinContent(Int_t bin) const
{
//*-*-*-*-*-*-*Return content of global bin number bin*-*-*-*-*-*-*-*-*-*-*-*
//*-*          =======================================
   if (!fReady) {
      ((TH1K*)this)->Sort();
      ((TH1K*)this)->fReady=1;
   }
   if (!fNIn)   return 0.;
   float x = GetBinCenter(bin);
   int left = TMath::BinarySearch(fNIn,fArray,x);
   int jl=left,jr=left+1,nk,nkmax =fKOrd;
//   assert(jl==-1   || fArray[jl] < x);
//   assert(jr==fNIn || fArray[jr] > x);
   float fl,fr,ff=0.,ffmin=1.e-6;
   if (!nkmax) {nkmax = 3; ffmin = GetBinWidth(bin);}
   if (nkmax >= fNIn) nkmax = fNIn-1;
   for (nk = 1; nk <= nkmax || ff <= ffmin; nk++) {
     fl = (jl>=0  ) ? TMath::Abs(fArray[jl]-x) : 1.e+20;
     fr = (jr<fNIn) ? TMath::Abs(fArray[jr]-x) : 1.e+20;
     if (jl<0 && jr>=fNIn) break;
     if (fl < fr) { ff = fl; jl--;}
     else         { ff = fr; jr++;}
   }
   ((TH1K*)this)->fKCur = nk - 1;
   return fNIn * 0.5*fKCur/((float)(fNIn+1))*GetBinWidth(bin)/ff;
}

//______________________________________________________________________________
Stat_t TH1K::GetBinError(Int_t bin) const
{
//*-*-*-*-*-*-*Return content of global bin error *-*-*-*-*-*-*-*-*-*-*-*
//*-*          ===================================
   return TMath::Sqrt(((double)(fNIn-fKCur+1))/((fNIn+1)*(fKCur-1)))*GetBinContent(bin);
}

//______________________________________________________________________________
static int TH1K_fcompare(const void *f1,const void *f2)
{
  if (*((float*)f1) < *((float*)f2)) return -1;
  if (*((float*)f1) > *((float*)f2)) return  1;
  return 0;
}

//______________________________________________________________________________
void TH1K::Sort()
{
  if (fNIn<2) return;
  qsort(GetArray(),fNIn,sizeof(Float_t),&TH1K_fcompare);
}






