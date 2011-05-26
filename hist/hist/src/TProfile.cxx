// @(#)root/hist:$Id$
// Author: Rene Brun   29/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TProfile.h"
#include "TMath.h"
#include "TF1.h"
#include "THLimitsFinder.h"
#include "Riostream.h"
#include "TVirtualPad.h"
#include "TError.h"
#include "TClass.h"

#include "TProfileHelper.h"

Bool_t TProfile::fgApproximate = kFALSE;

ClassImp(TProfile)

//______________________________________________________________________________
//
//  Profile histograms are used to display the mean
//  value of Y and its RMS for each bin in X. Profile histograms are in many cases an
//  elegant replacement of two-dimensional histograms : the inter-relation of two
//  measured quantities X and Y can always be visualized by a two-dimensional
//  histogram or scatter-plot; its representation on the line-printer is not particularly
//  satisfactory, except for sparse data. If Y is an unknown (but single-valued)
//  approximate function of X, this function is displayed by a profile histogram with
//  much better precision than by a scatter-plot.
//
//  The following formulae show the cumulated contents (capital letters) and the values
//  displayed by the printing or plotting routines (small letters) of the elements for bin J.
//
//                                                    2
//      H(J)  =  sum Y                  E(J)  =  sum Y
//      l(J)  =  sum l                  L(J)  =  sum l
//      h(J)  =  H(J)/L(J)              s(J)  =  sqrt(E(J)/L(J)- h(J)**2)
//      e(J)  =  s(J)/sqrt(L(J))
//
//  In the special case where s(J) is zero (eg, case of 1 entry only in one bin)
//  e(J) is computed from the average of the s(J) for all bins if the static function
//  TProfile::Approximate has been called.
//  This simple/crude approximation was suggested in order to keep the bin
//  during a fit operation. But note that this approximation is not the default behaviour.
//
//           Example of a profile histogram with its graphics output
//{
//  TCanvas *c1 = new TCanvas("c1","Profile histogram example",200,10,700,500);
//  hprof  = new TProfile("hprof","Profile of pz versus px",100,-4,4,0,20);
//  Float_t px, py, pz;
//  for ( Int_t i=0; i<25000; i++) {
//     gRandom->Rannor(px,py);
//     pz = px*px + py*py;
//     hprof->Fill(px,pz,1);
//  }
//  hprof->Draw();
//}
//Begin_Html
/*
<img src="gif/profile.gif">
*/
//End_Html
//

//______________________________________________________________________________
TProfile::TProfile() : TH1D()
{
//*-*-*-*-*-*Default constructor for Profile histograms*-*-*-*-*-*-*-*-*
//*-*        ==========================================

   BuildOptions(0,0,"");
}

//______________________________________________________________________________
TProfile::~TProfile()
{
//*-*-*-*-*-*Default destructor for Profile histograms*-*-*-*-*-*-*-*-*
//*-*        =========================================

}

//______________________________________________________________________________
TProfile::TProfile(const char *name,const char *title,Int_t nbins,Double_t xlow,Double_t xup,Option_t *option)
    : TH1D(name,title,nbins,xlow,xup)
{
//*-*-*-*-*-*Normal Constructor for Profile histograms*-*-*-*-*-*-*-*-*-*
//*-*        ==========================================
//
//  The first five parameters are similar to TH1D::TH1D.
//  All values of y are accepted at filling time.
//  To fill a profile histogram, one must use TProfile::Fill function.
//
//  Note that when filling the profile histogram the function Fill
//  checks if the variable y is betyween fYmin and fYmax.
//  If a minimum or maximum value is set for the Y scale before filling,
//  then all values below ymin or above ymax will be discarded.
//  Setting the minimum or maximum value for the Y scale before filling
//  has the same effect as calling the special TProfile constructor below
//  where ymin and ymax are specified.
//
//  H(J) is printed as the channel contents. The errors displayed are s(J) if CHOPT='S'
//  (spread option), or e(J) if CHOPT=' ' (error on mean).
//
//        See TProfile::BuildOptions for explanation of parameters
//
// see also comments in the TH1 base class constructors

   BuildOptions(0,0,option);
}

//______________________________________________________________________________
TProfile::TProfile(const char *name,const char *title,Int_t nbins,const Float_t *xbins,Option_t *option)
    : TH1D(name,title,nbins,xbins)
{
//*-*-*-*-*-*Constructor for Profile histograms with variable bin size*-*-*-*-*
//*-*        =========================================================
//
//        See TProfile::BuildOptions for more explanations on errors
//
// see also comments in the TH1 base class constructors

   BuildOptions(0,0,option);
}

//______________________________________________________________________________
TProfile::TProfile(const char *name,const char *title,Int_t nbins,const Double_t *xbins,Option_t *option)
    : TH1D(name,title,nbins,xbins)
{
//*-*-*-*-*-*Constructor for Profile histograms with variable bin size*-*-*-*-*
//*-*        =========================================================
//
//        See TProfile::BuildOptions for more explanations on errors
//
// see also comments in the TH1 base class constructors

   BuildOptions(0,0,option);
}

//______________________________________________________________________________
TProfile::TProfile(const char *name,const char *title,Int_t nbins,const Double_t *xbins,Double_t ylow,Double_t yup,Option_t *option)
    : TH1D(name,title,nbins,xbins)
{
//*-*-*-*-*-*Constructor for Profile histograms with variable bin size*-*-*-*-*
//*-*        =========================================================
//
//        See TProfile::BuildOptions for more explanations on errors
//
// see also comments in the TH1 base class constructors

   BuildOptions(ylow,yup,option);
}

//______________________________________________________________________________
TProfile::TProfile(const char *name,const char *title,Int_t nbins,Double_t xlow,Double_t xup,Double_t ylow,Double_t yup,Option_t *option)
    : TH1D(name,title,nbins,xlow,xup)
{
//*-*-*-*-*-*Constructor for Profile histograms with range in y*-*-*-*-*-*
//*-*        ==================================================
//  The first five parameters are similar to TH1D::TH1D.
//  Only the values of Y between ylow and yup will be considered at filling time.
//  ylow and yup will also be the maximum and minimum values
//  on the y scale when drawing the profile.
//
//        See TProfile::BuildOptions for more explanations on errors
//
// see also comments in the TH1 base class constructors

   BuildOptions(ylow,yup,option);
}


//______________________________________________________________________________
void TProfile::BuildOptions(Double_t ymin, Double_t ymax, Option_t *option)
{
//*-*-*-*-*-*-*Set Profile histogram structure and options*-*-*-*-*-*-*-*-*
//*-*          ===========================================
//
//    If a bin has N data points all with the same value Y (especially
//    possible when dealing with integers), the spread in Y for that bin
//    is zero, and the uncertainty assigned is also zero, and the bin is
//    ignored in making subsequent fits. If SQRT(Y) was the correct error
//    in the case above, then SQRT(Y)/SQRT(N) would be the correct error here.
//    In fact, any bin with non-zero number of entries N but with zero spread
//    should have an uncertainty SQRT(Y)/SQRT(N).
//
//    Now, is SQRT(Y)/SQRT(N) really the correct uncertainty?
//    that it is only in the case where the Y variable is some sort
//    of counting statistics, following a Poisson distribution. This should
//    probably be set as the default case. However, Y can be any variable
//    from an original NTUPLE, not necessarily distributed "Poissonly".
//    The computation of errors is based on the parameter option:
//    option:
//     ' '  (Default) Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  SQRT(Y)/SQRT(N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     's'            Errors are Spread  for Spread.ne.0. ,
//                      "     "  SQRT(Y)  for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     'i'            Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  1./SQRT(12.*N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//
//    The third case above corresponds to Integer Y values for which the
//    uncertainty is +-0.5, with the assumption that the probability that Y
//    takes any value between Y-0.5 and Y+0.5 is uniform (the same argument
//    goes for Y uniformly distributed between Y and Y+1); this would be
//    useful if Y is an ADC measurement, for example. 
//     Other, fancier options
//    would be possible, at the cost of adding one more parameter to the PROFILE
//    command. For example, if all Y variables are distributed according to some
//    known Gaussian of standard deviation Sigma (which can be different for each measurement), 
//    and the profile has been filled  with a weight equal to 1/Sigma**2, 
//    then one cam use the following option: 
// 
//     'G'            Errors are 1./SQRT(Sum(1/sigma**2)) 
//    For example, this would be useful when all Y's are experimental quantities
//    measured with different precision Sigma_Y.
//
//

   SetErrorOption(option);

   fBinEntries.Set(fNcells);  //*-* create number of entries per bin array

   // TH1::Sumw2 create sum of square of weights array times y (fSumw2) . This is always created for a TProfile
   TH1::Sumw2();                   //*-* create sum of squares of weights array times y
   // TProfile::Sumw2 create sum of square of weight2 (fBinSumw2). This is needed only for profile filled with weights not 1
   if (fgDefaultSumw2) Sumw2();    // optionally create sum of squares of weights / bin 

   fYmin = ymin;
   fYmax = ymax;
   fScaling = kFALSE;
   fTsumwy = fTsumwy2 = 0;

}
 
//______________________________________________________________________________
TProfile::TProfile(const TProfile &profile) : TH1D()
{
   // Copy constructor.

   ((TProfile&)profile).Copy(*this);
}


//______________________________________________________________________________
void TProfile::Add(TF1 *, Double_t, Option_t * )
{
   // Performs the operation: this = this + c1*f1

   Error("Add","Function not implemented for TProfile");
   return;
}


//______________________________________________________________________________
void TProfile::Add(const TH1 *h1, Double_t c1)
{
   // Performs the operation: this = this + c1*h1

   if (!h1) {
      Error("Add","Attempt to add a non-existing profile");
      return;
   }
   if (!h1->InheritsFrom(TProfile::Class())) {
      Error("Add","Attempt to add a non-profile object");
      return;
   }
   
   TProfileHelper::Add(this, this, h1, 1, c1);
}

//______________________________________________________________________________
void TProfile::Add(const TH1 *h1, const TH1 *h2, Double_t c1, Double_t c2)
{
//*-*-*-*-*Replace contents of this profile by the addition of h1 and h2*-*-*
//*-*      =============================================================
//
//   this = c1*h1 + c2*h2
//
//   c1 and c2 are considered as weights applied to the two summed profiles. 
//   The operation acts therefore like merging the two profiles with a weight c1 and c2
//

   if (!h1 || !h2) {
      Error("Add","Attempt to add a non-existing profile");
      return;
   }
   if (!h1->InheritsFrom(TProfile::Class())) {
      Error("Add","Attempt to add a non-profile object");
      return;
   }
   if (!h2->InheritsFrom(TProfile::Class())) {
      Error("Add","Attempt to add a non-profile object");
      return;
   }
   TProfileHelper::Add(this, h1, h2, c1, c2);
}


//______________________________________________________________________________
void TProfile::Approximate(Bool_t approx)
{
//     static function
// set the fgApproximate flag. When the flag is true, the function GetBinError
// will approximate the bin error with the average profile error on all bins
// in the following situation only
//  - the number of bins in the profile is less than 1002
//  - the bin number of entries is small ( <5)
//  - the estimated bin error is extremely small compared to the bin content
//  (see TProfile::GetBinError)

   fgApproximate = approx;
}

//______________________________________________________________________________
Int_t TProfile::BufferEmpty(Int_t action)
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
      Reset("ICES"); // reset without deleting the functions
      fBuffer = buffer;
   }
   if (TestBit(kCanRebin) || fXaxis.GetXmax() <= fXaxis.GetXmin()) {
      //find min, max of entries in buffer
      Double_t xmin = fBuffer[2];
      Double_t xmax = xmin;
      for (Int_t i=1;i<nbentries;i++) {
         Double_t x = fBuffer[3*i+2];
         if (x < xmin) xmin = x;
         if (x > xmax) xmax = x;
      }
      if (fXaxis.GetXmax() <= fXaxis.GetXmin()) {
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(this,xmin,xmax);
      } else {
         fBuffer = 0;
         Int_t keep = fBufferSize; fBufferSize = 0;
         if (xmin <  fXaxis.GetXmin()) RebinAxis(xmin,&fXaxis);
         if (xmax >= fXaxis.GetXmax()) RebinAxis(xmax,&fXaxis);
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
Int_t TProfile::BufferFill(Double_t x, Double_t y, Double_t w)
{
// accumulate arguments in buffer. When buffer is full, empty the buffer
// fBuffer[0] = number of entries in buffer
// fBuffer[1] = w of first entry
// fBuffer[2] = x of first entry
// fBuffer[3] = y of first entry

   if (!fBuffer) return -2;
   Int_t nbentries = (Int_t)fBuffer[0];
   if (nbentries < 0) {
      nbentries  = -nbentries;
      fBuffer[0] =  nbentries;
      if (fEntries > 0) {
         Double_t *buffer = fBuffer; fBuffer=0;
         Reset("ICES");  // reset without deleting the functions
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
   return -2;
}

//______________________________________________________________________________
void TProfile::Copy(TObject &obj) const
{
//*-*-*-*-*-*-*-*Copy a Profile histogram to a new profile histogram*-*-*-*-*
//*-*            ===================================================

   TH1D::Copy(((TProfile&)obj));
   fBinEntries.Copy(((TProfile&)obj).fBinEntries);
   fBinSumw2.Copy(((TProfile&)obj).fBinSumw2);
   for (int bin=0;bin<fNcells;bin++) {
      ((TProfile&)obj).fArray[bin]        = fArray[bin];
      ((TProfile&)obj).fSumw2.fArray[bin] = fSumw2.fArray[bin];
   }

   ((TProfile&)obj).fYmin = fYmin;
   ((TProfile&)obj).fYmax = fYmax;
   ((TProfile&)obj).fScaling   = fScaling;
   ((TProfile&)obj).fErrorMode = fErrorMode;
   ((TProfile&)obj).fTsumwy    = fTsumwy;
   ((TProfile&)obj).fTsumwy2   = fTsumwy2;
}


//______________________________________________________________________________
void TProfile::Divide(TF1 *, Double_t )
{
   // Performs the operation: this = this/(c1*f1)

   Error("Divide","Function not implemented for TProfile");
   return;
}

//______________________________________________________________________________
void TProfile::Divide(const TH1 *h1)
{
//*-*-*-*-*-*-*-*-*-*-*Divide this profile by h1*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//
//   this = this/h1
// This function accepts to divide a TProfile by a histogram
//

   if (!h1) {
      Error("Divide","Attempt to divide a non-existing profile");
      return;
   }
   if (!h1->InheritsFrom(TH1::Class())) {
      Error("Divide","Attempt to divide by a non-profile or non-histogram object");
      return;
   }
   TProfile *p1 = (TProfile*)h1;

   // delete buffer if it is there since it will become invalid
   if (fBuffer) BufferEmpty(1);


   Int_t nbinsx = GetNbinsX();
//*-*- Check profile compatibility
   if (nbinsx != p1->GetNbinsX()) {
      Error("Divide","Attempt to divide profiles with different number of bins");
      return;
   }

//*-*- Reset statistics
   fEntries = fTsumw   = fTsumw2 = fTsumwx = fTsumwx2 = fTsumwy = fTsumwy2 = 0;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin;
   Double_t *cu1=0, *er1=0, *en1=0;
   Double_t e0,e1,c12;
   if (h1->InheritsFrom(TProfile::Class())) {
      cu1 = p1->GetW();
      er1 = p1->GetW2();
      en1 = p1->GetB();
   }
   Double_t c0,c1,w,z,x;
   for (bin=0;bin<=nbinsx+1;bin++) {
      c0  = fArray[bin];
      if (cu1) c1  = cu1[bin];
      else     c1  = h1->GetBinContent(bin);
      if (c1) w = c0/c1;
      else    w = 0;
      fArray[bin] = w;
      z = TMath::Abs(w);
      x = fXaxis.GetBinCenter(bin);
      fEntries++;
      fTsumw   += z;
      fTsumw2  += z*z;
      fTsumwx  += z*x;
      fTsumwx2 += z*x*x;
      fTsumwy  += z*c1;
      fTsumwx2 += z*c1*c1;
      e0 = fSumw2.fArray[bin];
      if (er1) e1 = er1[bin];
      else    {e1 = h1->GetBinError(bin); e1*=e1;}
      c12= c1*c1;
      if (!c1) fSumw2.fArray[bin] = 0;
      else     fSumw2.fArray[bin] = (e0*c1*c1 + e1*c0*c0)/(c12*c12);
      if (!en1) continue;
      if (!en1[bin]) fBinEntries.fArray[bin] = 0;
      else           fBinEntries.fArray[bin] /= en1[bin];
   }
   // mantaining the correct sum of weights square is not supported when dividing
   // bin error resulting from division of profile needs to be checked 
   if (fBinSumw2.fN) { 
      Warning("Divide","Cannot preserve during the division of profiles the sum of bin weight square");
      fBinSumw2 = TArrayD();
   }
   
}


//______________________________________________________________________________
void TProfile::Divide(const TH1 *h1, const TH1 *h2, Double_t c1, Double_t c2, Option_t *option)
{
//*-*-*-*-*Replace contents of this profile by the division of h1 by h2*-*-*
//*-*      ============================================================
//
//   this = c1*h1/(c2*h2)
//

   TString opt = option;
   opt.ToLower();
   Bool_t binomial = kFALSE;
   if (opt.Contains("b")) binomial = kTRUE;
   if (!h1 || !h2) {
      Error("Divide","Attempt to divide a non-existing profile");
      return;
   }
   if (!h1->InheritsFrom(TProfile::Class())) {
      Error("Divide","Attempt to divide a non-profile object");
      return;
   }
   TProfile *p1 = (TProfile*)h1;
   if (!h2->InheritsFrom(TProfile::Class())) {
      Error("Divide","Attempt to divide by a non-profile object");
      return;
   }
   TProfile *p2 = (TProfile*)h2;

   // delete buffer if it is there since it will become invalid
   if (fBuffer) BufferEmpty(1);

   Int_t nbinsx = GetNbinsX();
//*-*- Check histogram compatibility
   if (nbinsx != p1->GetNbinsX() || nbinsx != p2->GetNbinsX()) {
      Error("Divide","Attempt to divide profiles with different number of bins");
      return;
   }
   if (!c2) {
      Error("Divide","Coefficient of dividing profile cannot be zero");
      return;
   }

   //THE ALGORITHM COMPUTING THE ERRORS IS WRONG. HELP REQUIRED
   printf("WARNING!!: The algorithm in TProfile::Divide computing the errors is not accurate\n");
   printf(" Instead of Divide(TProfile *h1, TProfile *h2, do:\n");
   printf("   TH1D *p1 = h1->ProjectionX();\n");
   printf("   TH1D *p2 = h2->ProjectionX();\n");
   printf("   p1->Divide(p2);\n");

//*-*- Reset statistics
   fEntries = fTsumw   = fTsumw2 = fTsumwx = fTsumwx2 = 0;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin;
   Double_t *cu1 = p1->GetW();
   Double_t *cu2 = p2->GetW();
   Double_t *er1 = p1->GetW2();
   Double_t *er2 = p2->GetW2();
   Double_t *en1 = p1->GetB();
   Double_t *en2 = p2->GetB();
   Double_t b1,b2,w,z,x,ac1,ac2;
   //d1 = c1*c1;
   //d2 = c2*c2;
   ac1 = TMath::Abs(c1);
   ac2 = TMath::Abs(c2);
   for (bin=0;bin<=nbinsx+1;bin++) {
      b1  = cu1[bin];
      b2  = cu2[bin];
      if (b2) w = c1*b1/(c2*b2);
      else    w = 0;
      fArray[bin] = w;
      z = TMath::Abs(w);
      x = fXaxis.GetBinCenter(bin);
      fEntries++;
      fTsumw   += z;
      fTsumw2  += z*z;
      fTsumwx  += z*x;
      fTsumwx2 += z*x*x;
      //fTsumwy  += z*x;
      //fTsumwy2 += z*x*x;
      Double_t e1 = er1[bin];
      Double_t e2 = er2[bin];
    //Double_t b22= b2*b2*d2;
      Double_t b22= b2*b2*TMath::Abs(c2);
      if (!b2) fSumw2.fArray[bin] = 0;
      else {
         if (binomial) {
            fSumw2.fArray[bin] = TMath::Abs(w*(1-w)/b2);
         } else {
          //fSumw2.fArray[bin] = d1*d2*(e1*b2*b2 + e2*b1*b1)/(b22*b22);
            fSumw2.fArray[bin] = ac1*ac2*(e1*b2*b2 + e2*b1*b1)/(b22*b22);
         }
      }
      if (en2[bin]) fBinEntries.fArray[bin] = en1[bin]/en2[bin];
      else          fBinEntries.fArray[bin] = 0;
   }

   // mantaining the correct sum of weights square is not supported when dividing
   // bin error resulting from division of profile needs to be checked 
   if (fBinSumw2.fN) { 
      Warning("Divide","Cannot preserve during the division of profiles the sum of bin weight square");
      fBinSumw2 = TArrayD();
   }

}

//______________________________________________________________________________
TH1 *TProfile::DrawCopy(Option_t *option) const
{
//*-*-*-*-*-*-*-*Draw a copy of this profile histogram*-*-*-*-*-*-*-*-*-*-*-*
//*-*            =====================================
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TProfile *newpf = (TProfile*)Clone();
   newpf->SetDirectory(0);
   newpf->SetBit(kCanDelete);
   newpf->AppendPad(option);
   return newpf;
}

//______________________________________________________________________________
Int_t TProfile::Fill(Double_t x, Double_t y)
{
//*-*-*-*-*-*-*-*-*-*-*Fill a Profile histogram (no weights)*-*-*-*-*-*-*-*
//*-*                  =====================================

   if (fBuffer) return BufferFill(x,y,1);

   Int_t bin;
   if (fYmin != fYmax) {
      if (y <fYmin || y> fYmax || TMath::IsNaN(y) ) return -1;
   }

   fEntries++;
   bin =fXaxis.FindBin(x);
   AddBinContent(bin, y);
   fSumw2.fArray[bin] += (Double_t)y*y;
   fBinEntries.fArray[bin] += 1;
   if (fBinSumw2.fN)  fBinSumw2.fArray[bin] += 1;
   if (bin == 0 || bin > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   fTsumw++;
   fTsumw2++;
   fTsumwx  += x;
   fTsumwx2 += x*x;
   fTsumwy  += y;
   fTsumwy2 += y*y;
   return bin;
}

//______________________________________________________________________________
Int_t TProfile::Fill(const char *namex, Double_t y)
{
// Fill a Profile histogram (no weights)
//
   Int_t bin;
   if (fYmin != fYmax) {
      if (y <fYmin || y> fYmax || TMath::IsNaN(y) ) return -1;
   }

   fEntries++;
   bin =fXaxis.FindBin(namex);
   AddBinContent(bin, y);
   fSumw2.fArray[bin] += (Double_t)y*y;
   fBinEntries.fArray[bin] += 1;
   if (fBinSumw2.fN)  fBinSumw2.fArray[bin] += 1;
   if (bin == 0 || bin > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   Double_t x = fXaxis.GetBinCenter(bin);
   fTsumw++;
   fTsumw2++;
   fTsumwx  += x;
   fTsumwx2 += x*x;
   fTsumwy  += y;
   fTsumwy2 += y*y;
   return bin;
}

//______________________________________________________________________________
Int_t TProfile::Fill(Double_t x, Double_t y, Double_t w)
{
//*-*-*-*-*-*-*-*-*-*-*Fill a Profile histogram with weights*-*-*-*-*-*-*-*
//*-*                  =====================================

   if (fBuffer) return BufferFill(x,y,w);

   Int_t bin;
   if (fYmin != fYmax) {
      if (y <fYmin || y> fYmax || TMath::IsNaN(y) ) return -1;
   }

   Double_t u= w; // (w > 0 ? w : -w);
   fEntries++;
   bin =fXaxis.FindBin(x);
   AddBinContent(bin, u*y);
   fSumw2.fArray[bin] += u*y*y;
   fBinEntries.fArray[bin] += u;
   if (fBinSumw2.fN)  fBinSumw2.fArray[bin] += u*u;
   if (bin == 0 || bin > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   fTsumw   += u;
   fTsumw2  += u*u;
   fTsumwx  += u*x;
   fTsumwx2 += u*x*x;
   fTsumwy  += u*y;
   fTsumwy2 += u*y*y;
   return bin;
}

//______________________________________________________________________________
Int_t TProfile::Fill(const char *namex, Double_t y, Double_t w)
{
// Fill a Profile histogram with weights
//
   Int_t bin;

   if (fYmin != fYmax) {
      if (y <fYmin || y> fYmax || TMath::IsNaN(y) ) return -1;
   }

   Double_t u= w; // (w > 0 ? w : -w);
   fEntries++;
   bin =fXaxis.FindBin(namex);
   AddBinContent(bin, u*y);
   fSumw2.fArray[bin] += u*y*y;
   fBinEntries.fArray[bin] += u;
   if (fBinSumw2.fN)  fBinSumw2.fArray[bin] += u*u;
   if (bin == 0 || bin > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   Double_t x = fXaxis.GetBinCenter(bin);
   fTsumw   += u;
   fTsumw2  += u*u;
   fTsumwx  += u*x;
   fTsumwx2 += u*x*x;
   fTsumwy  += u*y;
   fTsumwy2 += u*y*y;
   return bin;
}


//______________________________________________________________________________
void TProfile::FillN(Int_t ntimes, const Double_t *x, const Double_t *y, const Double_t *w, Int_t stride)
{
//*-*-*-*-*-*-*-*-*-*-*Fill a Profile histogram with weights*-*-*-*-*-*-*-*
//*-*                  =====================================
   Int_t bin,i;
   ntimes *= stride;
   for (i=0;i<ntimes;i+=stride) {
      if (fYmin != fYmax) {
         if (y[i] <fYmin || y[i]> fYmax || TMath::IsNaN(y[i])) continue;
      }

      Double_t u = (w) ? w[i] : 1; // (w[i] > 0 ? w[i] : -w[i]);
      fEntries++;
      bin =fXaxis.FindBin(x[i]);
      AddBinContent(bin, u*y[i]);
      fSumw2.fArray[bin] += u*y[i]*y[i];
      fBinEntries.fArray[bin] += u;
      if (fBinSumw2.fN)  fBinSumw2.fArray[bin] += u*u;
      if (bin == 0 || bin > fXaxis.GetNbins()) {
         if (!fgStatOverflows) continue;
      }
      fTsumw   += u;
      fTsumw2  += u*u;
      fTsumwx  += u*x[i];
      fTsumwx2 += u*x[i]*x[i];
      fTsumwy  += u*y[i];
      fTsumwy2 += u*y[i]*y[i];
   }
}

//______________________________________________________________________________
Double_t TProfile::GetBinContent(Int_t bin) const
{
//*-*-*-*-*-*-*Return bin content of a Profile histogram*-*-*-*-*-*-*-*-*-*
//*-*          =========================================

   if (fBuffer) ((TProfile*)this)->BufferEmpty();

   if (bin < 0 || bin >= fNcells) return 0;
   if (fBinEntries.fArray[bin] == 0) return 0;
   if (!fArray) return 0;
   return fArray[bin]/fBinEntries.fArray[bin];
}

//______________________________________________________________________________
Double_t TProfile::GetBinEntries(Int_t bin) const
{
//*-*-*-*-*-*-*Return bin entries of a Profile histogram*-*-*-*-*-*-*-*-*-*
//*-*          =========================================

   if (fBuffer) ((TProfile*)this)->BufferEmpty();

   if (bin < 0 || bin >= fNcells) return 0;
   return fBinEntries.fArray[bin];
}

//______________________________________________________________________________
Double_t TProfile::GetBinEffectiveEntries(Int_t bin) const
{
//            Return bin effective entries for a weighted filled Profile histogram. 
//            In case of an unweighted profile, it is equivalent to the number of entries per bin   
//            The effective entries is defined as the square of the sum of the weights divided by the 
//            sum of the weights square. 
//            TProfile::Sumw2() must be called before filling the profile with weights. 
//            Only by calling this method the  sum of the square of the weights per bin is stored. 
//  
//*-*          =========================================

   return TProfileHelper::GetBinEffectiveEntries((TProfile*)this, bin);
}

//______________________________________________________________________________
Double_t TProfile::GetBinError(Int_t bin) const
{
// *-*-*-*-*-*-*Return bin error of a Profile histogram*-*-*-*-*-*-*-*-*-*
// *-*          =======================================
//
// Computing errors: A moving field
// =================================
// The computation of errors for a TProfile has evolved with the versions
// of ROOT. The difficulty is in computing errors for bins with low statistics.
// - prior to version 3.00, we had no special treatment of low statistic bins.
//   As a result, these bins had huge errors. The reason is that the
//   expression eprim2 is very close to 0 (rounding problems) or 0.
// - in version 3.00 (18 Dec 2000), the algorithm is protected for values of
//   eprim2 very small and the bin errors set to the average bin errors, following
//   recommendations from a group of users.
// - in version 3.01 (19 Apr 2001), it is realized that the algorithm above
//   should be applied only to low statistic bins.
// - in version 3.02 (26 Sep 2001), the same group of users recommend instead
//   to take two times the average error on all bins for these low
//   statistics bins giving a very small value for eprim2.
// - in version 3.04 (Nov 2002), the algorithm is modified/protected for the case
//   when a TProfile is projected (ProjectionX). The previous algorithm
//   generated a N^2 problem when projecting a TProfile with a large number of
//   bins (eg 100000).
// - in version 3.05/06, a new static function TProfile::Approximate
//   is introduced to enable or disable (default) the approximation.
//
// Ideas for improvements of this algorithm are welcome. No suggestions
// received since our call for advice to roottalk in Jul 2002.
// see for instance: http://root.cern.ch/root/roottalk/roottalk02/2916.html

   return TProfileHelper::GetBinError((TProfile*)this, bin);
}

//______________________________________________________________________________
Option_t *TProfile::GetErrorOption() const
{
//*-*-*-*-*-*-*-*-*-*Return option to compute profile errors*-*-*-*-*-*-*-*-*
//*-*                =======================================

   if (fErrorMode == kERRORSPREAD)  return "s";
   if (fErrorMode == kERRORSPREADI) return "i";
   if (fErrorMode == kERRORSPREADG) return "g";
   return "";
}

//______________________________________________________________________________
char* TProfile::GetObjectInfo(Int_t px, Int_t py) const
{
   //   Redefines TObject::GetObjectInfo.
   //   Displays the profile info (bin number, contents, eroor, entries per bin
   //   corresponding to cursor position px,py
   //
   if (!gPad) return (char*)"";
   static char info[200];
   Double_t x  = gPad->PadtoX(gPad->AbsPixeltoX(px));
   Double_t y  = gPad->PadtoY(gPad->AbsPixeltoY(py));
   Int_t binx   = GetXaxis()->FindFixBin(x);
   snprintf(info,200,"(x=%g, y=%g, binx=%d, binc=%g, bine=%g, binn=%d)", x, y, binx, GetBinContent(binx), GetBinError(binx), (Int_t)GetBinEntries(binx));
   return info;
}

//______________________________________________________________________________
void TProfile::GetStats(Double_t *stats) const
{
   // fill the array stats from the contents of this profile
   // The array stats must be correctly dimensionned in the calling program.
   // stats[0] = sumw
   // stats[1] = sumw2
   // stats[2] = sumwx
   // stats[3] = sumwx2
   // stats[4] = sumwy
   // stats[5] = sumwy2
   //
   // If no axis-subrange is specified (via TAxis::SetRange), the array stats
   // is simply a copy of the statistics quantities computed at filling time.
   // If a sub-range is specified, the function recomputes these quantities
   // from the bin contents in the current axis range.

   if (fBuffer) ((TProfile*)this)->BufferEmpty();

   // Loop on bins
   Int_t bin, binx;
   if (fTsumw == 0 || fXaxis.TestBit(TAxis::kAxisRange)) {
      for (bin=0;bin<6;bin++) stats[bin] = 0;
      if (!fBinEntries.fArray) return;
      Int_t firstBinX = fXaxis.GetFirst();
      Int_t lastBinX  = fXaxis.GetLast();
      // include underflow/overflow if TH1::StatOverflows(kTRUE) in case no range is set on the axis
      if (fgStatOverflows && !fXaxis.TestBit(TAxis::kAxisRange)) {
         if (firstBinX == 1) firstBinX = 0;
         if (lastBinX ==  fXaxis.GetNbins() ) lastBinX += 1;
      }
      for (binx = firstBinX; binx <= lastBinX; binx++) {
         Double_t w   = fBinEntries.fArray[binx];
         Double_t w2  = (fBinSumw2.fN ? fBinSumw2.fArray[binx] : w);  
         Double_t x   = fXaxis.GetBinCenter(binx);
         stats[0] += w;
         stats[1] += w2;
         stats[2] += w*x;
         stats[3] += w*x*x;
         stats[4] += fArray[binx];
         stats[5] += fSumw2.fArray[binx];
      }
   } else {
      if (fTsumwy == 0 && fTsumwy2 == 0) {
         //this case may happen when processing TProfiles with version <=3
         TProfile *p = (TProfile*)this; // chheting with const
         for (binx=fXaxis.GetFirst();binx<=fXaxis.GetLast();binx++) {
            p->fTsumwy  += fArray[binx];
            p->fTsumwy2 += fSumw2.fArray[binx];
         }
      }
      stats[0] = fTsumw;
      stats[1] = fTsumw2;
      stats[2] = fTsumwx;
      stats[3] = fTsumwx2;
      stats[4] = fTsumwy;
      stats[5] = fTsumwy2;
   }
}

//___________________________________________________________________________
void TProfile::LabelsDeflate(Option_t *option)
{
// Reduce the number of bins for this axis to the number of bins having a label.

   TProfileHelper::LabelsDeflate(this, option);
}

//___________________________________________________________________________
void TProfile::LabelsInflate(Option_t *options)
{
// Double the number of bins for axis.
// Refill histogram
// This function is called by TAxis::FindBin(const char *label)

  TProfileHelper::LabelsInflate(this, options); 
}

//___________________________________________________________________________
void TProfile::LabelsOption(Option_t *option, Option_t * /*ax */)
{
//  Set option(s) to draw axis with labels
//  option = "a" sort by alphabetic order
//         = ">" sort by decreasing values
//         = "<" sort by increasing values
//         = "h" draw labels horizonthal
//         = "v" draw labels vertical
//         = "u" draw labels up (end of label right adjusted)
//         = "d" draw labels down (start of label left adjusted)

   THashList *labels = fXaxis.GetLabels();
   if (!labels) {
      Warning("LabelsOption","Cannot sort. No labels");
      return;
   }
   TString opt = option;
   opt.ToLower();
   if (opt.Contains("h")) {
      fXaxis.SetBit(TAxis::kLabelsHori);
      fXaxis.ResetBit(TAxis::kLabelsVert);
      fXaxis.ResetBit(TAxis::kLabelsDown);
      fXaxis.ResetBit(TAxis::kLabelsUp);
   }
   if (opt.Contains("v")) {
      fXaxis.SetBit(TAxis::kLabelsVert);
      fXaxis.ResetBit(TAxis::kLabelsHori);
      fXaxis.ResetBit(TAxis::kLabelsDown);
      fXaxis.ResetBit(TAxis::kLabelsUp);
   }
   if (opt.Contains("u")) {
      fXaxis.SetBit(TAxis::kLabelsUp);
      fXaxis.ResetBit(TAxis::kLabelsVert);
      fXaxis.ResetBit(TAxis::kLabelsDown);
      fXaxis.ResetBit(TAxis::kLabelsHori);
   }
   if (opt.Contains("d")) {
      fXaxis.SetBit(TAxis::kLabelsDown);
      fXaxis.ResetBit(TAxis::kLabelsVert);
      fXaxis.ResetBit(TAxis::kLabelsHori);
      fXaxis.ResetBit(TAxis::kLabelsUp);
   }
   Int_t sort = -1;
   if (opt.Contains("a")) sort = 0;
   if (opt.Contains(">")) sort = 1;
   if (opt.Contains("<")) sort = 2;
   if (sort < 0) return;

   Int_t n = TMath::Min(fXaxis.GetNbins(), labels->GetSize());
   Int_t *a = new Int_t[n+2];
   Int_t i,j;
   Double_t *cont   = new Double_t[n+2];
   Double_t *sumw   = new Double_t[n+2];
   Double_t *errors = new Double_t[n+2];
   Double_t *ent    = new Double_t[n+2];
   THashList *labold = new THashList(labels->GetSize(),1);
   TIter nextold(labels);
   TObject *obj;
   while ((obj=nextold())) {
      labold->Add(obj);
   }
   labels->Clear();
   if (sort > 0) {
      //---sort by values of bins
      for (i=1;i<=n;i++) {
         sumw[i-1]   = fArray[i];
         errors[i-1] = fSumw2.fArray[i];
         ent[i-1]    = fBinEntries.fArray[i];
         if (fBinEntries.fArray[i] == 0) cont[i-1] = 0;
         else cont[i-1] = fArray[i]/fBinEntries.fArray[i];
      }
      if (sort ==1) TMath::Sort(n,cont,a,kTRUE);  //sort by decreasing values
      else          TMath::Sort(n,cont,a,kFALSE); //sort by increasing values
      for (i=1;i<=n;i++) {
         fArray[i] = sumw[a[i-1]];
         fSumw2.fArray[i] = errors[a[i-1]];
         fBinEntries.fArray[i] = ent[a[i-1]];
      }
      for (i=1;i<=n;i++) {
         obj = labold->At(a[i-1]);
         labels->Add(obj);
         obj->SetUniqueID(i);
      }
   } else {
      //---alphabetic sort
      const UInt_t kUsed = 1<<18;
      TObject *objk=0;
      a[0] = 0;
      a[n+1] = n+1;
      for (i=1;i<=n;i++) {
         const char *label = "zzzzzzzzzzzz";
         for (j=1;j<=n;j++) {
            obj = labold->At(j-1);
            if (!obj) continue;
            if (obj->TestBit(kUsed)) continue;
            //use strcasecmp for case non-sensitive sort (may be an option)
            if (strcmp(label,obj->GetName()) < 0) continue;
            objk = obj;
            a[i] = j;
            label = obj->GetName();
         }
         if (objk) {
            objk->SetUniqueID(i);
            labels->Add(objk);
            objk->SetBit(kUsed);
         }
      }
      for (i=1;i<=n;i++) {
         obj = labels->At(i-1);
         if (!obj) continue;
         obj->ResetBit(kUsed);
      }

      for (i=1;i<=n;i++) {
         sumw[i]   = fArray[a[i]];
         errors[i] = fSumw2.fArray[a[i]];
         ent[i]    = fBinEntries.fArray[a[i]];
      }
      for (i=1;i<=n;i++) {
         fArray[i] = sumw[i];
         fSumw2.fArray[i] = errors[i];
         fBinEntries.fArray[i] = ent[i];
      }
   }
   delete labold;
   if (a)      delete [] a;
   if (sumw)   delete [] sumw;
   if (cont)   delete [] cont;
   if (errors) delete [] errors;
   if (ent)    delete [] ent;
}

//______________________________________________________________________________
Long64_t TProfile::Merge(TCollection *li)
{
   //Merge all histograms in the collection in this histogram.
   //This function computes the min/max for the x axis,
   //compute a new number of bins, if necessary,
   //add bin contents, errors and statistics.
   //If overflows are present and limits are different the function will fail.
   //The function returns the total number of entries in the result histogram
   //if the merge is successfull, -1 otherwise.
   //
   //IMPORTANT remark. The axis x may have different number
   //of bins and different limits, BUT the largest bin width must be
   //a multiple of the smallest bin width and the upper limit must also
   //be a multiple of the bin width.

   return TProfileHelper::Merge(this, li);
}


//______________________________________________________________________________
void TProfile::Multiply(TF1 *f1, Double_t c1)
{
   // Performs the operation: this = this*c1*f1

   if (!f1) {
      Error("Multiply","Attempt to multiply by a null function");
      return;
   }

   Int_t nbinsx = GetNbinsX();

//*-*- Add statistics
   Double_t xx[1], cf1, ac1 = TMath::Abs(c1);
   Double_t s1[10];
   Int_t i;
   for (i=0;i<10;i++) {s1[i] = 0;}
   PutStats(s1);

   SetMinimum();
   SetMaximum();

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin;
   for (bin=0;bin<=nbinsx+1;bin++) {
      xx[0] = fXaxis.GetBinCenter(bin);
      if (!f1->IsInside(xx)) continue;
      TF1::RejectPoint(kFALSE);
      cf1 = f1->EvalPar(xx);
      if (TF1::RejectedPoint()) continue;
      fArray[bin]             *= c1*cf1;
      //see http://savannah.cern.ch/bugs/?func=detailitem&item_id=14851
      //fSumw2.fArray[bin]      *= c1*c1*cf1*cf1;
      fSumw2.fArray[bin]      *= ac1*cf1*cf1;
      //fBinEntries.fArray[bin] *= ac1*TMath::Abs(cf1);
   }
}

//______________________________________________________________________________
void TProfile::Multiply(const TH1 *)
{
//*-*-*-*-*-*-*-*-*-*-*Multiply this profile by h1*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =============================
//
//   this = this*h1
//
   Error("Multiply","Multiplication of profile histograms not implemented");
}


//______________________________________________________________________________
void TProfile::Multiply(const TH1 *, const TH1 *, Double_t, Double_t, Option_t *)
{
//*-*-*-*-*Replace contents of this profile by multiplication of h1 by h2*-*
//*-*      ================================================================
//
//   this = (c1*h1)*(c2*h2)
//

   Error("Multiply","Multiplication of profile histograms not implemented");
}

//______________________________________________________________________________
TH1D *TProfile::ProjectionX(const char *name, Option_t *option) const
{
//*-*-*-*-*Project this profile into a 1-D histogram along X*-*-*-*-*-*-*
//*-*      =================================================
//
//   The projection is always of the type TH1D.
//
//   if option "E" is specified the errors of the projected histogram are computed and set 
//      to be equal to the errors of the profile.
//      Option "E" is defined as the default one in the header file. 
//   if option "" is specified the histogram errors are simply the sqrt of its content
//   if option "B" is specified, the content of bin of the returned histogram
//      will be equal to the GetBinEntries(bin) of the profile,
//      otherwise (default) it will be equal to GetBinContent(bin)
//   if option "C=E" the bin contents of the projection are set to the
//       bin errors of the profile
//   if option "W" is specified the bin content of the projected histogram  is set to the 
//       product of the bin content of the profile and the entries. 
//       With this option the returned histogram will be equivalent to the one obtained by 
//       filling directly a TH1D using the 2-nd value as a weight. 
//       This makes sense only for profile filled with weights =1. If not, the error of the 
//        projected histogram obtained with this option will not be correct.


   TString opt = option;
   opt.ToLower();
   Int_t nx = fXaxis.GetNbins();

// Create the projection histogram
   TString pname = name; 
   if (pname == "_px") { 
      pname = GetName(); 
      pname.Append("_px");
   }
   TH1D *h1;
   const TArrayD *bins = fXaxis.GetXbins();
   if (bins->fN == 0) {
      h1 = new TH1D(pname,GetTitle(),nx,fXaxis.GetXmin(),fXaxis.GetXmax());
   } else {
      h1 = new TH1D(pname,GetTitle(),nx,bins->fArray);
   }
   Bool_t computeErrors = kFALSE;
   Bool_t cequalErrors  = kFALSE;
   Bool_t binEntries    = kFALSE;
   Bool_t binWeight     = kFALSE;
   if (opt.Contains("b")) binEntries = kTRUE;
   if (opt.Contains("e")) computeErrors = kTRUE;
   if (opt.Contains("w")) binWeight = kTRUE;
   if (opt.Contains("c=e")) {cequalErrors = kTRUE; computeErrors=kFALSE;}
   if (computeErrors || binWeight ) h1->Sumw2();

   // Fill the projected histogram
   Double_t cont;
   for (Int_t bin =0;bin<=nx+1;bin++) {

      if (binEntries)         cont = GetBinEntries(bin);
      else if (cequalErrors)  cont = GetBinError(bin);
      else if (binWeight)     cont = fArray[bin];  // bin content * bin entries
      else                    cont = GetBinContent(bin);    // default case
      
      h1->SetBinContent(bin ,cont);

      // if option E projected histogram errors are same as profile
      if (computeErrors ) h1->SetBinError(bin , GetBinError(bin) );
      // in case of option W bin error is deduced from bin sum of z**2 values of profile
      // this is correct only if the profile is filled with weights =1
      if (binWeight) h1->SetBinError(bin , TMath::Sqrt(fSumw2.fArray[bin] ) );
      // in case of bin entries and h1 has sumw2 set, we need to set also the bin error
      if (binEntries && h1->GetSumw2() ) {
         Double_t err2;
         if (fBinSumw2.fN) 
            err2 = fBinSumw2.fArray[bin]; 
         else 
            err2 = cont; // this is fBinEntries.fArray[bin]
         h1->SetBinError(bin, TMath::Sqrt(err2 ) ); 
      }

   }

   // Copy the axis attributes and the axis labels if needed.
   h1->GetXaxis()->ImportAttributes(this->GetXaxis());
   h1->GetYaxis()->ImportAttributes(this->GetYaxis());
   THashList* labels=this->GetXaxis()->GetLabels();
   if (labels) {
      TIter iL(labels);
      TObjString* lb;
      Int_t i = 1;
      while ((lb=(TObjString*)iL())) {
         h1->GetXaxis()->SetBinLabel(i,lb->String().Data());
         i++;
      }
   }
   
   h1->SetEntries(fEntries);
   return h1;
}

//______________________________________________________________________________
void TProfile::PutStats(Double_t *stats)
{
   // Replace current statistics with the values in array stats

   fTsumw   = stats[0];
   fTsumw2  = stats[1];
   fTsumwx  = stats[2];
   fTsumwx2 = stats[3];
   fTsumwy  = stats[4];
   fTsumwy2 = stats[5];
}

//______________________________________________________________________________
TH1 *TProfile::Rebin(Int_t ngroup, const char*newname, const Double_t *xbins)
{
//*-*-*-*-*Rebin this profile grouping ngroup bins together*-*-*-*-*-*-*-*-*
//*-*      ================================================
//  -case 1  xbins=0
//   if newname is not blank a new temporary profile hnew is created.
//   else the current profile is modified (default)
//   The parameter ngroup indicates how many bins of this have to me merged
//   into one bin of hnew
//   If the original profile has errors stored (via Sumw2), the resulting
//   profile has new errors correctly calculated.
//
//   examples: if hp is an existing TProfile histogram with 100 bins
//     hp->Rebin();  //merges two bins in one in hp: previous contents of hp are lost
//     hp->Rebin(5); //merges five bins in one in hp
//     TProfile *hnew = hp->Rebin(5,"hnew"); // creates a new profile hnew
//                                       //merging 5 bins of hp in one bin
//
//   NOTE:  If ngroup is not an exact divider of the number of bins,
//          the top limit of the rebinned profile is changed
//          to the upper edge of the bin=newbins*ngroup and the corresponding
//          bins are added to the overflow bin.
//          Statistics will be recomputed from the new bin contents.
//
//  -case 2  xbins!=0
//   a new profile is created (you should specify newname).
//   The parameter ngroup is the number of variable size bins in the created profile
//   The array xbins must contain ngroup+1 elements that represent the low-edge
//   of the bins.
//   The data of the old bins are added to the new bin which contains the bin center
//   of the old bins. It is possible that information from the old binning are attached
//   to the under-/overflow bins of the new binning.
//
//   examples: if hp is an existing TProfile with 100 bins
//     Double_t xbins[25] = {...} array of low-edges (xbins[25] is the upper edge of last bin
//     hp->Rebin(24,"hpnew",xbins);  //creates a new variable bin size profile hpnew

   Int_t nbins    = fXaxis.GetNbins();
   Double_t xmin  = fXaxis.GetXmin();
   Double_t xmax  = fXaxis.GetXmax();
   if ((ngroup <= 0) || (ngroup > nbins)) {
      Error("Rebin", "Illegal value of ngroup=%d",ngroup);
      return 0;
   }
   if (!newname && xbins) {
      Error("Rebin","if xbins is specified, newname must be given");
      return 0;
   }

   Int_t newbins = nbins/ngroup;
   if (!xbins) { 
      Int_t nbg = nbins/ngroup;
      if (nbg*ngroup != nbins) {
         Warning("Rebin", "ngroup=%d must be an exact divider of nbins=%d",ngroup,nbins);
      }
   }
   else {   
   // in the case of xbins given (rebinning in variable bins) ngroup is the new number of bins.
   //  and number of grouped bins is not constant. 
   // when looping for setting the contents for the new histogram we 
   // need to loop on all bins of original histogram. Set then ngroup=nbins
      newbins = ngroup;
      ngroup = nbins;
   }

   // Save old bin contents into a new array
   Double_t *oldBins   = new Double_t[nbins+2];
   Double_t *oldCount  = new Double_t[nbins+2];
   Double_t *oldErrors = new Double_t[nbins+2];
   Double_t *oldBinw2  = (fBinSumw2.fN ? new Double_t[nbins+2] : 0  ); 
   Int_t bin, i;
   Double_t *cu1 = GetW();
   Double_t *er1 = GetW2();
   Double_t *en1 = GetB();
   Double_t *ew1 = GetB2();
   
   for (bin=0;bin<=nbins+1;bin++) {
      oldBins[bin]   = cu1[bin];
      oldCount[bin]  = en1[bin];
      oldErrors[bin] = er1[bin];
      if (ew1 && fBinSumw2.fN) oldBinw2[bin]  = ew1[bin];
   }

   // create a clone of the old histogram if newname is specified
   TProfile *hnew = this;
   if ((newname && strlen(newname) > 0) || xbins) {
      hnew = (TProfile*)Clone(newname);
   }

   // in case of ngroup not an excat divider of nbins, 
   // top limit is changed (see NOTE in method comment) 
   if(!xbins && (newbins*ngroup != nbins)) {
      xmax = fXaxis.GetBinUpEdge(newbins*ngroup);
      hnew->fTsumw = 0; //stats must be reset because top bins will be moved to overflow bin
   }

   // set correctly the axis and resizes the bin arrays
   if(!xbins && (fXaxis.GetXbins()->GetSize() > 0)){ 
      // for rebinning of variable bins in a constant group 
      Double_t *bins = new Double_t[newbins+1];
      for(i = 0; i <= newbins; ++i) bins[i] = fXaxis.GetBinLowEdge(1+i*ngroup);
      hnew->SetBins(newbins,bins); //this also changes the bin array's
      delete [] bins;
   } else if (xbins) { 
      // when rebinning in variable bins
      hnew->SetBins(newbins,xbins);
   } else {
      hnew->SetBins(newbins,xmin,xmax);
   }

   // merge bin contents ignoring now underflow/overflows
   if (fBinSumw2.fN) hnew->Sumw2();

   // Start merging only once the new lowest edge is reached
   Int_t startbin = 1;
   const Double_t newxmin = hnew->GetXaxis()->GetBinLowEdge(1);
   while( fXaxis.GetBinCenter(startbin) < newxmin && startbin <= nbins ) {
      startbin++;
   }
   
   Double_t *cu2 = hnew->GetW();
   Double_t *er2 = hnew->GetW2();
   Double_t *en2 = hnew->GetB();
   Double_t *ew2 = hnew->GetB2();
   Int_t oldbin = startbin;
   Double_t binContent, binCount, binError, binSumw2;
   for (bin = 1;bin<=newbins;bin++) {
      binContent = 0;
      binCount   = 0;
      binError   = 0;
      binSumw2   = 0;

      //for xbins != 0: ngroup == nbins
      Int_t imax = ngroup;
      Double_t xbinmax = hnew->GetXaxis()->GetBinUpEdge(bin);
      for (i=0;i<ngroup;i++) {
	 if((hnew == this && (oldbin+i > nbins)) ||
	    (hnew != this && (fXaxis.GetBinCenter(oldbin+i) > xbinmax)))
	 {
	    imax = i;
	    break;
	 }

	 binContent += oldBins[oldbin+i];
	 binCount   += oldCount[oldbin+i];
	 binError   += oldErrors[oldbin+i];
	 if (fBinSumw2.fN) binSumw2 += oldBinw2[oldbin+i];
      }
   
      cu2[bin] = binContent;
      er2[bin] = binError;
      en2[bin] = binCount;
      if (fBinSumw2.fN) ew2[bin] = binSumw2;
      oldbin += imax;
   }
   // set bin statistics for underflow bin
   binContent = 0;
   binCount   = 0;
   binError   = 0;
   binSumw2   = 0;
   for(i=0;i<startbin;i++)
   {
      binContent += oldBins[i];
      binCount   += oldCount[i];
      binError   += oldErrors[i];
      if (fBinSumw2.fN) binSumw2 += oldBinw2[i];
   }
   hnew->fArray[0] = binContent;
   hnew->fBinEntries[0] = binCount;
   hnew->fSumw2[0] = binError;
   if ( fBinSumw2.fN ) hnew->fBinSumw2[0] = binSumw2;

   // set bin statistics for overflow bin
   binContent = 0;
   binCount   = 0;
   binError   = 0;
   binSumw2   = 0;
   for(i=oldbin;i<=nbins+1;i++)
   {
      binContent += oldBins[i];
      binCount   += oldCount[i];
      binError   += oldErrors[i];
      if (fBinSumw2.fN) binSumw2 += oldBinw2[i];
   }
   hnew->fArray[newbins+1] = binContent;
   hnew->fBinEntries[newbins+1] = binCount;   
   hnew->fSumw2[newbins+1] = binError;
   if ( fBinSumw2.fN ) hnew->fBinSumw2[newbins+1] = binSumw2;


   delete [] oldBins;
   delete [] oldCount;
   delete [] oldErrors;
   if (oldBinw2) delete [] oldBinw2; 
   return hnew;
}

//______________________________________________________________________________
void TProfile::RebinAxis(Double_t x, TAxis *axis)
{
// Profile histogram is resized along x axis such that x is in the axis range.
// The new axis limits are recomputed by doubling iteratively
// the current axis range until the specified value x is within the limits.
// The algorithm makes a copy of the histogram, then loops on all bins
// of the old histogram to fill the rebinned histogram.
// Takes into account errors (Sumw2) if any.
// The bit kCanRebin must be set before invoking this function.
//  Ex:  h->SetBit(TH1::kCanRebin);

   TProfile*  hold = TProfileHelper::RebinAxis(this, x, axis);
   if ( hold ) {
      fTsumwy  = hold->fTsumwy;
      fTsumwy2 = hold->fTsumwy2;
      
      delete hold;
   }
}

//______________________________________________________________________________
void TProfile::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Reset contents of a Profile histogram*-*-*-*-*-*-*-*-*
//*-*                =====================================
   TH1D::Reset(option);
   fBinEntries.Reset();
   fBinSumw2.Reset();
   TString opt = option;
   opt.ToUpper();
   if (opt.Contains("ICE") && !opt.Contains("S")) return;
   fTsumwy  = 0;
   fTsumwy2 = 0;
}

//______________________________________________________________________________
void TProfile::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
    // Save primitive as a C++ statement(s) on output stream out

   //Note the following restrictions in the code generated:
   // - variable bin size not implemented
   // - SetErrorOption not implemented

   Bool_t nonEqiX = kFALSE;
   Int_t i;
   // Check if the profile has equidistant X bins or not.  If not, we
   // create an array holding the bins.
   if (GetXaxis()->GetXbins()->fN && GetXaxis()->GetXbins()->fArray) {
      nonEqiX = kTRUE;
      out << "   Double_t xAxis[" << GetXaxis()->GetXbins()->fN
          << "] = {";
      for (i = 0; i < GetXaxis()->GetXbins()->fN; i++) {
         if (i != 0) out << ", ";
         out << GetXaxis()->GetXbins()->fArray[i];
      }
      out << "}; " << endl;
   }

   char quote = '"';
   out<<"   "<<endl;
   out<<"   "<<ClassName()<<" *";

   //histogram pointer has by default teh histogram name.
   //however, in case histogram has no directory, it is safer to add a incremental suffix
   static Int_t hcounter = 0;
   TString histName = GetName();
   if (!fDirectory) {
      hcounter++;
      histName += "__";
      histName += hcounter;
   }
   const char *hname = histName.Data();
   
   out<<hname<<" = new "<<ClassName()<<"("<<quote<<GetName()<<quote<<","<<quote<<GetTitle()<<quote
                 <<","<<GetXaxis()->GetNbins();
   if (nonEqiX)
      out << ", xAxis";
   else
      out << "," << GetXaxis()->GetXmin()
          << "," << GetXaxis()->GetXmax()
          <<","<<quote<<GetErrorOption()<<quote<<");"<<endl;

   // save bin entries
   Int_t bin;
   for (bin=0;bin<fNcells;bin++) {
      Double_t bi = GetBinEntries(bin);
      if (bi) {
         out<<"   "<<hname<<"->SetBinEntries("<<bin<<","<<bi<<");"<<endl;
      }
   }
   //save bin contents
   for (bin=0;bin<fNcells;bin++) {
      Double_t bc = fArray[bin];
      if (bc) {
         out<<"   "<<hname<<"->SetBinContent("<<bin<<","<<bc<<");"<<endl;
      }
   }
   // save bin errors
   if (fSumw2.fN) {
      for (bin=0;bin<fNcells;bin++) {
         Double_t be = TMath::Sqrt(fSumw2.fArray[bin]);
         if (be) {
            out<<"   "<<hname<<"->SetBinError("<<bin<<","<<be<<");"<<endl;
         }
      }
   }

   TH1::SavePrimitiveHelp(out, hname, option);
}

//______________________________________________________________________________
void TProfile::Scale(Double_t c1, Option_t * option)
{
// *-*-*-*-*Multiply this profile by a constant c1*-*-*-*-*-*-*-*-*
// *-*      ======================================
//
//   this = c1*this
//
// This function uses the services of TProfile::Add
//
   
   TProfileHelper::Scale(this, c1, option);
}

//______________________________________________________________________________
void TProfile::SetBinEntries(Int_t bin, Double_t w)
{
//*-*-*-*-*-*-*-*-*Set the number of entries in bin*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ================================

   if (bin < 0 || bin >= fNcells) return;
   fBinEntries.fArray[bin] = w;
}

//______________________________________________________________________________
void TProfile::SetBins(Int_t nx, Double_t xmin, Double_t xmax)
{
//*-*-*-*-*-*-*-*-*Redefine  x axis parameters*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ===========================

   fXaxis.Set(nx,xmin,xmax);
   fNcells = nx+2;
   SetBinsLength(fNcells);
   fBinEntries.Set(fNcells);
   fSumw2.Set(fNcells);
   if (fBinSumw2.fN) fBinSumw2.Set(fNcells);
}

//______________________________________________________________________________
void TProfile::SetBins(Int_t nx, const Double_t *xbins)
{
//*-*-*-*-*-*-*-*-*Redefine  x axis parameters*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ===========================

   fXaxis.Set(nx,xbins);
   fNcells = nx+2;
   SetBinsLength(fNcells);
   fBinEntries.Set(fNcells);
   fSumw2.Set(fNcells);
   if (fBinSumw2.fN) fBinSumw2.Set(fNcells);
}


//______________________________________________________________________________
void TProfile::SetBuffer(Int_t buffersize, Option_t *)
{
// set the buffer size in units of 8 bytes (double)

   if (fBuffer) {
      BufferEmpty();
      delete [] fBuffer;
      fBuffer = 0;
   }
   if (buffersize <= 0) {
      fBufferSize = 0;
      return;
   }
   if (buffersize < 100) buffersize = 100;
   fBufferSize = 1 + 3*buffersize;
   fBuffer = new Double_t[fBufferSize];
   memset(fBuffer,0,8*fBufferSize);
}

//______________________________________________________________________________
void TProfile::SetErrorOption(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Set option to compute profile errors*-*-*-*-*-*-*-*-*
//*-*                =====================================
//
//    The computation of errors is based on the parameter option:
//    option:
//     ' '  (Default) Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  SQRT(Y)/SQRT(N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     's'            Errors are Spread  for Spread.ne.0. ,
//                      "     "  SQRT(Y)  for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     'i'            Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  1./SQRT(12.*N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     'g'            Errors are 1./SQRT(W) for Spread.ne.0. , 
//                      "     "  0.  for N.eq.0
//                    W is the sum of weights of the profile. 
//                    This option is for measurements y +/ dy and  the profile is filled with 
//                    weights w = 1/dy**2
//
//   See TProfile::BuildOptions for explanation of all options

   TString opt = option;
   opt.ToLower();
   fErrorMode = kERRORMEAN;
   if (opt.Contains("s")) fErrorMode = kERRORSPREAD;
   if (opt.Contains("i")) fErrorMode = kERRORSPREADI;
   if (opt.Contains("g")) fErrorMode = kERRORSPREADG;
}

//______________________________________________________________________________
void TProfile::Streamer(TBuffer &R__b)
{
   // Stream an object of class TProfile.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         R__b.ReadClassBuffer(TProfile::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TH1D::Streamer(R__b);
      fBinEntries.Streamer(R__b);
      Int_t errorMode;
      R__b >> errorMode;
      fErrorMode = (EErrorType)errorMode;
      if (R__v < 2) {
         Float_t ymin,ymax;
         R__b >> ymin; fYmin = ymin;
         R__b >> ymax; fYmax = ymax;
      } else {
         R__b >> fYmin;
         R__b >> fYmax;
      }
      R__b.CheckByteCount(R__s, R__c, TProfile::IsA());
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TProfile::Class(),this);
   }
}
//______________________________________________________________________________
void TProfile::Sumw2()
{
   // Create structure to store sum of squares of weights per bin  *-*-*-*-*-*-*-*
   //   This is needed to compute  the correct statistical quantities  
   //    of a profile filled with weights 
   //  
   //
   //  This function is automatically called when the histogram is created
   //  if the static function TH1::SetDefaultSumw2 has been called before.

   TProfileHelper::Sumw2(this);
}
