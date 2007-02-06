// @(#)root/hist:$Name:  $:$Id: TProfile.cxx,v 1.84 2007/02/01 14:58:44 brun Exp $
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
#include "THashList.h"
#include "TF1.h"
#include "THLimitsFinder.h"
#include "Riostream.h"
#include "TVirtualPad.h"
#include "TError.h"
#include "TClass.h"

const Int_t kNstat = 11;
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
//  e(J) is computed from the average of the s(J) for all bins.
//  This simple/crude approximation was suggested in order to keep the bin
//  during a fit operation.
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
//  Only the values of Y between YMIN and YMAX will be considered at filling time.
//  ymin and ymax will also be the maximum and minimum values
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
//    useful if Y is an ADC measurement, for example. Other, fancier options
//    would be possible, at the cost of adding one more parameter to the PROFILE
//    command. For example, if all Y variables are distributed according to some
//    known Gaussian of standard deviation Sigma, then:
//     'G'            Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  Sigma/SQRT(N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//    For example, this would be useful when all Y's are experimental quantities
//    measured with the same instrument with precision Sigma.
//
//

   SetErrorOption(option);

   fBinEntries.Set(fNcells);  //*-* create number of entries per bin array

   Sumw2();                   //*-* create sum of squares of weights array

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
   if (!h1->InheritsFrom("TProfile")) {
      Error("Add","Attempt to add a non-profile object");
      return;
   }
   TProfile *p1 = (TProfile*)h1;

   Int_t nbinsx = GetNbinsX();
//*-*- Check profile compatibility
   if (nbinsx != p1->GetNbinsX()) {
      Error("Add","Attempt to add profiles with different number of bins");
      return;
   }

//*-*- Add statistics
   Double_t ac1 = TMath::Abs(c1);
   fEntries += ac1*p1->GetEntries();
   fTsumw   += ac1*p1->fTsumw;
   fTsumw2  += c1*c1*p1->fTsumw2;
   fTsumwx  += ac1*p1->fTsumwx;
   fTsumwx2 += ac1*p1->fTsumwx2;
   fTsumwy  += ac1*p1->fTsumwy;
   fTsumwy2 += ac1*p1->fTsumwy2;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin;
   Double_t *cu1 = p1->GetW();
   Double_t *er1 = p1->GetW2();
   Double_t *en1 = p1->GetB();
   for (bin=0;bin<=nbinsx+1;bin++) {
      fArray[bin]             += c1*cu1[bin];
      //see http://savannah.cern.ch/bugs/?func=detailitem&item_id=14851
      fSumw2.fArray[bin]      += ac1*er1[bin];
      fBinEntries.fArray[bin] += ac1*en1[bin];
   }
}

//______________________________________________________________________________
void TProfile::Add(const TH1 *h1, const TH1 *h2, Double_t c1, Double_t c2)
{
//*-*-*-*-*Replace contents of this profile by the addition of h1 and h2*-*-*
//*-*      =============================================================
//
//   this = c1*h1 + c2*h2
//

   if (!h1 || !h2) {
      Error("Add","Attempt to add a non-existing profile");
      return;
   }
   if (!h1->InheritsFrom("TProfile")) {
      Error("Add","Attempt to add a non-profile object");
      return;
   }
   TProfile *p1 = (TProfile*)h1;
   if (!h2->InheritsFrom("TProfile")) {
      Error("Add","Attempt to add a non-profile object");
      return;
   }
   TProfile *p2 = (TProfile*)h2;

   Int_t nbinsx = GetNbinsX();
//*-*- Check profile compatibility
   if (nbinsx != p1->GetNbinsX() || nbinsx != p2->GetNbinsX()) {
      Error("Add","Attempt to add profiles with different number of bins");
      return;
   }

//*-*- Add statistics
   Double_t ac1 = TMath::Abs(c1);
   Double_t ac2 = TMath::Abs(c2);
   fEntries = ac1*p1->GetEntries() + ac2*p2->GetEntries();
   fTsumw   = ac1*p1->fTsumw       + ac2*p2->fTsumw;
   fTsumw2  = c1*c1*p1->fTsumw2    + c2*c2*p2->fTsumw2;
   fTsumwx  = ac1*p1->fTsumwx      + ac2*p2->fTsumwx;
   fTsumwx2 = ac1*p1->fTsumwx2     + ac2*p2->fTsumwx2;
   fTsumwy  = ac1*p1->fTsumwy      + ac2*p2->fTsumwy;
   fTsumwy2 = ac1*p1->fTsumwy2     + ac2*p2->fTsumwy2;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin;
   Double_t *cu1 = p1->GetW();
   Double_t *cu2 = p2->GetW();
   Double_t *er1 = p1->GetW2();
   Double_t *er2 = p2->GetW2();
   Double_t *en1 = p1->GetB();
   Double_t *en2 = p2->GetB();
   for (bin=0;bin<=nbinsx+1;bin++) {
      fArray[bin]             = c1*cu1[bin] +  c2*cu2[bin];
      if (fScaling) {
         //see http://savannah.cern.ch/bugs/?func=detailitem&item_id=14851
         fSumw2.fArray[bin]      = ac1*ac1*er1[bin] + ac2*ac2*er2[bin];
         fBinEntries.fArray[bin] = en1[bin];
      } else {
         fSumw2.fArray[bin]      = ac1*er1[bin] + ac2*er2[bin];
         fBinEntries.fArray[bin] = ac1*en1[bin] + ac2*en2[bin];
      }
   }
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
      Reset();
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
         if (xmin <  fXaxis.GetXmin()) RebinAxis(xmin,"X");
         if (xmax >= fXaxis.GetXmax()) RebinAxis(xmax,"X");
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
   return -2;
}

//______________________________________________________________________________
void TProfile::Copy(TObject &obj) const
{
//*-*-*-*-*-*-*-*Copy a Profile histogram to a new profile histogram*-*-*-*-*
//*-*            ===================================================

   TH1D::Copy(((TProfile&)obj));
   fBinEntries.Copy(((TProfile&)obj).fBinEntries);
   ((TProfile&)obj).fYmin = fYmin;
   ((TProfile&)obj).fYmax = fYmax;
   ((TProfile&)obj).fScaling = fScaling;
   ((TProfile&)obj).fErrorMode = fErrorMode;
   ((TProfile&)obj).fTsumwy      = fTsumwy;
   ((TProfile&)obj).fTsumwy2     = fTsumwy2;
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
   if (!h1->InheritsFrom("TH1")) {
      Error("Divide","Attempt to divide by a non-profile or non-histogram object");
      return;
   }
   TProfile *p1 = (TProfile*)h1;

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
   if (h1->InheritsFrom("TProfile")) {
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
   if (!h1->InheritsFrom("TProfile")) {
      Error("Divide","Attempt to divide a non-profile object");
      return;
   }
   TProfile *p1 = (TProfile*)h1;
   if (!h2->InheritsFrom("TProfile")) {
      Error("Divide","Attempt to divide by a non-profile object");
      return;
   }
   TProfile *p2 = (TProfile*)h2;

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
}

//______________________________________________________________________________
TH1 *TProfile::DrawCopy(Option_t *option) const
{
//*-*-*-*-*-*-*-*Draw a copy of this profile histogram*-*-*-*-*-*-*-*-*-*-*-*
//*-*            =====================================
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TProfile *newpf = new TProfile();
   Copy(*newpf);
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
      if (y <fYmin || y> fYmax) return -1;
   }

   fEntries++;
   bin =fXaxis.FindBin(x);
   AddBinContent(bin, y);
   fSumw2.fArray[bin] += (Double_t)y*y;
   fBinEntries.fArray[bin] += 1;
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
      if (y <fYmin || y> fYmax) return -1;
   }

   fEntries++;
   bin =fXaxis.FindBin(namex);
   AddBinContent(bin, y);
   fSumw2.fArray[bin] += (Double_t)y*y;
   fBinEntries.fArray[bin] += 1;
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
      if (y <fYmin || y> fYmax) return -1;
   }

   Double_t z= (w > 0 ? w : -w);
   fEntries++;
   bin =fXaxis.FindBin(x);
   AddBinContent(bin, z*y);
   fSumw2.fArray[bin] += z*y*y;
   fBinEntries.fArray[bin] += z;
   if (bin == 0 || bin > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   fTsumw   += z;
   fTsumw2  += z*z;
   fTsumwx  += z*x;
   fTsumwx2 += z*x*x;
   fTsumwy  += z*y;
   fTsumwy2 += z*y*y;
   return bin;
}

//______________________________________________________________________________
Int_t TProfile::Fill(const char *namex, Double_t y, Double_t w)
{
// Fill a Profile histogram with weights
//
   Int_t bin;

   if (fYmin != fYmax) {
      if (y <fYmin || y> fYmax) return -1;
   }

   Double_t z= (w > 0 ? w : -w);
   fEntries++;
   bin =fXaxis.FindBin(namex);
   AddBinContent(bin, z*y);
   fSumw2.fArray[bin] += z*y*y;
   fBinEntries.fArray[bin] += z;
   if (bin == 0 || bin > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   Double_t x = fXaxis.GetBinCenter(bin);
   fTsumw   += z;
   fTsumw2  += z*z;
   fTsumwx  += z*x;
   fTsumwx2 += z*x*x;
   fTsumwy  += z*y;
   fTsumwy2 += z*y*y;
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
         if (y[i] <fYmin || y[i]> fYmax) continue;
      }

      Double_t z= (w[i] > 0 ? w[i] : -w[i]);
      fEntries++;
      bin =fXaxis.FindBin(x[i]);
      AddBinContent(bin, z*y[i]);
      fSumw2.fArray[bin] += z*y[i]*y[i];
      fBinEntries.fArray[bin] += z;
      if (bin == 0 || bin > fXaxis.GetNbins()) {
         if (!fgStatOverflows) continue;
      }
      fTsumw   += z;
      fTsumw2  += z*z;
      fTsumwx  += z*x[i];
      fTsumwx2 += z*x[i]*x[i];
      fTsumwy  += z*y[i];
      fTsumwy2 += z*y[i]*y[i];
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
Double_t TProfile::GetBinError(Int_t bin) const
{
//*-*-*-*-*-*-*Return bin error of a Profile histogram*-*-*-*-*-*-*-*-*-*
//*-*          =======================================
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

   if (fBuffer) ((TProfile*)this)->BufferEmpty();

   if (bin < 0 || bin >= fNcells) return 0;
   Double_t cont = fArray[bin];
   Double_t sum  = fBinEntries.fArray[bin];
   Double_t err2 = fSumw2.fArray[bin];
   if (sum == 0) return 0;
   Double_t eprim;
   Double_t contsum = cont/sum;
   Double_t eprim2  = TMath::Abs(err2/sum - contsum*contsum);
   eprim          = TMath::Sqrt(eprim2);
   Double_t test = 1;
   if (err2 != 0 && sum < 5) test = eprim2*sum/err2;
//printf("bin=%d, cont=%g, sum=%g, err2=%g, eprim2=%g, test=%g\n",bin,cont,sum,err2,eprim2,test);
   //if statistics is unsufficient, take approximation.
   // error is set to the (average error on all bins) * 2
   //if (eprim <= 0) {   test in version 2.25/03
   //if (test < 1.e-4) { test in version 3.01/06
   //if (test < 1.e-4 || eprim2 < 1e-6) { test in version 3.03/09
   if (fgApproximate && fNcells <=1002 && (test < 1.e-4 || eprim2 < 1e-6)) { //3.04
      Double_t scont, ssum, serr2;
      scont = ssum = serr2 = 0;
      for (Int_t i=1;i<fNcells;i++) {
         if (fSumw2.fArray[i] <= 0) continue; //added in 3.10/02
         scont += fArray[i];
         ssum  += fBinEntries.fArray[i];
         serr2 += fSumw2.fArray[i];
      }
      Double_t scontsum = scont/ssum;
      Double_t seprim2  = TMath::Abs(serr2/ssum - scontsum*scontsum);
      eprim           = 2*TMath::Sqrt(seprim2);
      sum = ssum;
   }
   sum = TMath::Abs(sum);
   if (fErrorMode == kERRORMEAN) return eprim/TMath::Sqrt(sum);
   else if (fErrorMode == kERRORSPREAD) return eprim;
   else if (fErrorMode == kERRORSPREADI) {
      if (eprim != 0) return eprim/TMath::Sqrt(sum);
      return 1/TMath::Sqrt(12*sum);
   }
   else if (fErrorMode == kERRORSPREADG) {
      return eprim/TMath::Sqrt(sum);
   }
   else return eprim;
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
   static char info[64];
   Double_t x  = gPad->PadtoX(gPad->AbsPixeltoX(px));
   Double_t y  = gPad->PadtoY(gPad->AbsPixeltoY(py));
   Int_t binx   = GetXaxis()->FindFixBin(x);
   sprintf(info,"(x=%g, y=%g, binx=%d, binc=%g, bine=%g, binn=%d)", x, y, binx, GetBinContent(binx), GetBinError(binx), (Int_t)GetBinEntries(binx));
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
      Double_t w;
      Double_t x;
      for (bin=0;bin<6;bin++) stats[bin] = 0;
      if (!fBinEntries.fArray) return;
      for (binx=fXaxis.GetFirst();binx<=fXaxis.GetLast();binx++) {
         w         = fBinEntries.fArray[binx];
         x         = fXaxis.GetBinCenter(binx);
         stats[0] += w;
         stats[1] += w*w;
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
void TProfile::LabelsDeflate(Option_t *)
{
// Reduce the number of bins for this axis to the number of bins having a label.

   if (!fXaxis.GetLabels()) return;
   TIter next(fXaxis.GetLabels());
   TObject *obj;
   Int_t nbins = 0;
   while ((obj = next())) {
      if (obj->GetUniqueID()) nbins++;
   }
   if (nbins < 2) nbins = 2;
   TProfile *hold = (TProfile*)Clone();
   hold->SetDirectory(0);

   Double_t xmin = fXaxis.GetXmin();
   Double_t xmax = fXaxis.GetBinUpEdge(nbins);
   fXaxis.SetRange(0,0);
   fXaxis.Set(nbins,xmin,xmax);
   Int_t ncells = nbins+2;
   SetBinsLength(ncells);
   fBinEntries.Set(ncells);
   fSumw2.Set(ncells);

   //now loop on all bins and refill
   Int_t bin;
   for (bin=1;bin<=nbins;bin++) {
      fArray[bin] = hold->fArray[bin];
      fBinEntries.fArray[bin] = hold->fBinEntries.fArray[bin];
      fSumw2.fArray[bin] = hold->fSumw2.fArray[bin];
   }
   delete hold;
}

//___________________________________________________________________________
void TProfile::LabelsInflate(Option_t *)
{
// Double the number of bins for axis.
// Refill histogram
// This function is called by TAxis::FindBin(const char *label)

   TProfile *hold = (TProfile*)Clone();
   hold->SetDirectory(0);

   Int_t  nbold  = fXaxis.GetNbins();
   Int_t nbins   = nbold;
   Double_t xmin = fXaxis.GetXmin();
   Double_t xmax = fXaxis.GetXmax();
   xmax = xmin + 2*(xmax-xmin);
   fXaxis.SetRange(0,0);
   fXaxis.Set(2*nbins,xmin,xmax);
   nbins *= 2;
   Int_t ncells = nbins+2;
   SetBinsLength(ncells);
   fBinEntries.Set(ncells);
   fSumw2.Set(ncells);

   //now loop on all bins and refill
   Int_t bin;
   for (bin=1;bin<=nbins;bin++) {
      if (bin <= nbold) {
         fArray[bin] = hold->fArray[bin];
         fBinEntries.fArray[bin] = hold->fBinEntries.fArray[bin];
         fSumw2.fArray[bin] = hold->fSumw2.fArray[bin];
      } else {
         fArray[bin] = 0;
         fBinEntries.fArray[bin] = 0;
         fSumw2.fArray[bin] = 0;
      }
   }
   delete hold;
}

//___________________________________________________________________________
void TProfile::LabelsOption(Option_t *option, Option_t * /*ax*/)
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
         fArray[i] = sumw[a[i]];
         fSumw2.fArray[i] = errors[a[i]];
         fBinEntries.fArray[i] = ent[a[i]];
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

   if (!li) return 0;
   if (li->IsEmpty()) return (Int_t) GetEntries();

   TList inlist;
   TH1* hclone = (TH1*)Clone("FirstClone");
   R__ASSERT(hclone);
   BufferEmpty(1);         // To remove buffer.
   Reset();                // BufferEmpty sets limits so we can't use it later.
   SetEntries(0);
   inlist.Add(hclone);
   inlist.AddAll(li);

   TAxis newXAxis;
   Bool_t initialLimitsFound = kFALSE;
   Bool_t same = kTRUE;
   Bool_t allHaveLimits = kTRUE;

   TIter next(&inlist);
   while (TObject *o = next()) {
      TProfile* h = dynamic_cast<TProfile*> (o);
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
         }
         else {
            if (!RecomputeAxisLimits(newXAxis, *(h->GetXaxis()))) {
               Error("Merge", "Cannot merge histograms - limits are inconsistent:\n "
                     "first: (%d, %f, %f), second: (%d, %f, %f)",
                     newXAxis.GetNbins(), newXAxis.GetXmin(), newXAxis.GetXmax(),
                     h->GetXaxis()->GetNbins(), h->GetXaxis()->GetXmin(),
                     h->GetXaxis()->GetXmax());
            }
         }
      }
   }
   next.Reset();

   same = same && SameLimitsAndNBins(newXAxis, *GetXaxis());
   if (!same && initialLimitsFound)
      SetBins(newXAxis.GetNbins(), newXAxis.GetXmin(), newXAxis.GetXmax());

   if (!allHaveLimits) {
      // fill this histogram with all the data from buffers of histograms without limits
      while (TProfile* h = (TProfile*)next()) {
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
   Double_t stats[kNstat], totstats[kNstat];
   for (Int_t i=0;i<kNstat;i++) {totstats[i] = stats[i] = 0;}
   GetStats(totstats);
   Double_t nentries = GetEntries();
   Int_t binx, ix, nx;
   Bool_t canRebin=TestBit(kCanRebin);
   ResetBit(kCanRebin); // reset, otherwise setting the under/overflow will rebin
   while (TProfile* h=(TProfile*)next()) {
      // process only if the histogram has limits; otherwise it was processed before
      if (h->GetXaxis()->GetXmin() < h->GetXaxis()->GetXmax()) {
         // import statistics
         h->GetStats(stats);
         for (Int_t i=0;i<kNstat;i++)
            totstats[i] += stats[i];
         nentries += h->GetEntries();

         nx = h->GetXaxis()->GetNbins();
         for (binx = 0; binx <= nx + 1; binx++) {
            if ((!same) && (binx == 0 || binx == nx + 1)) {
               if (h->GetW()[binx] != 0) {
                  Error("Merge", "Cannot merge histograms - the histograms have"
                                 " different limits and undeflows/overflows are present."
                                 " The initial histogram is now broken!");
                  return -1;
               }
            }
            ix = fXaxis.FindBin(h->GetBinCenter(binx));
            fArray[ix]             += h->GetW()[binx];
            fSumw2.fArray[ix]      += h->GetW2()[binx];
            fBinEntries.fArray[ix] += h->GetB()[binx];
         }
         fEntries += h->GetEntries();
         fTsumw   += h->fTsumw;
         fTsumw2  += h->fTsumw2;
         fTsumwx  += h->fTsumwx;
         fTsumwx2 += h->fTsumwx2;
         fTsumwy  += h->fTsumwy;
      }
   }
   if (canRebin) SetBit(kCanRebin);

   PutStats(totstats);
   SetEntries(nentries);
   inlist.Remove(hclone);
   delete hclone;
   return (Long64_t) nentries;
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
//   if option "E" is specified, the errors are computed. (default)
//   if option "B" is specified, the content of bin of the returned histogram
//      will be equal to the GetBinEntries(bin) of the profile,
//      otherwise (default) it will be equal to GetBinContent(bin)
//   if option "C=E" the bin contents of the projection are set to the
//       bin errors of the profile

   TString opt = option;
   opt.ToLower();
   Int_t nx = fXaxis.GetNbins();

// Create the projection histogram
   char *pname = (char*)name;
   if (strcmp(name,"_px") == 0) {
      Int_t nch = strlen(GetName()) + 4;
      pname = new char[nch];
      sprintf(pname,"%s%s",GetName(),name);
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
   if (opt.Contains("b")) binEntries = kTRUE;
   if (opt.Contains("e")) computeErrors = kTRUE;
   if (opt.Contains("c=e")) {cequalErrors = kTRUE; computeErrors=kFALSE;}
   if (computeErrors) h1->Sumw2();
   if (pname != name)  delete [] pname;

   // Fill the projected histogram
   Double_t cont,err;
   for (Int_t binx =0;binx<=nx+1;binx++) {
      if (binEntries)    cont = GetBinEntries(binx);
      else               cont = GetBinContent(binx);
      err = GetBinError(binx);
      if (cequalErrors)  h1->SetBinContent(binx, err);
      else               h1->SetBinContent(binx, cont);
      if (computeErrors) h1->SetBinError(binx,err);
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
//   The parameter is the number of variable size bins in the created profile
//   The array xbins must contain ngroup+1 elements that represent the low-edge
//   of the bins.
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
   Int_t newbins = nbins/ngroup;
   if (xbins) newbins = ngroup;

   // Save old bin contents into a new array
   Double_t entries = fEntries;
   Double_t *oldBins   = new Double_t[nbins+1];
   Double_t *oldCount  = new Double_t[nbins+1];
   Double_t *oldErrors = new Double_t[nbins+1];
   Int_t bin, i;
   Double_t *cu1 = GetW();
   Double_t *er1 = GetW2();
   Double_t *en1 = GetB();
   for (bin=1;bin<=nbins;bin++) {
      oldBins[bin]   = cu1[bin];
      oldCount[bin]  = en1[bin];
      oldErrors[bin] = er1[bin];
   }

   // create a clone of the old histogram if newname is specified
   TProfile *hnew = this;
   if ((newname && strlen(newname) > 0) || xbins) {
      hnew = (TProfile*)Clone(newname);
   }

   // change axis specs and rebuild bin contents array
   if(!xbins && (newbins*ngroup != nbins)) {
      xmax = fXaxis.GetBinUpEdge(newbins*ngroup);
      hnew->fTsumw = 0; //stats must be reset because top bins will be moved to overflow bin
   }

   if(!xbins && (fXaxis.GetXbins()->GetSize() > 0)){ // variable bin sizes
      Double_t *bins = new Double_t[newbins+1];
      for(Int_t i = 0; i <= newbins; ++i) bins[i] = fXaxis.GetBinLowEdge(1+i*ngroup);
      hnew->SetBins(newbins,bins); //this also changes errors array (if any)
      delete [] bins;
   } else if (xbins) {
      ngroup = 1;
      hnew->SetBins(newbins,xbins);
   } else {
      hnew->SetBins(newbins,xmin,xmax);
   }

   // copy merged bin contents (ignore under/overflows)
   Double_t *cu2 = hnew->GetW();
   Double_t *er2 = hnew->GetW2();
   Double_t *en2 = hnew->GetB();
   Int_t oldbin = 1;
   Double_t binContent, binCount, binError;
   for (bin = 1;bin<=newbins;bin++) {
      binContent = 0;
      binCount   = 0;
      binError   = 0;
      for (i=0;i<ngroup;i++) {
         if (oldbin+i > nbins) break;
         binContent += oldBins[oldbin+i];
         binCount   += oldCount[oldbin+i];
         binError   += oldErrors[oldbin+i];
      }
      cu2[bin] = binContent;
      er2[bin] = binError;
      en2[bin] = binCount;
      oldbin += ngroup;
   }
   hnew->SetEntries(entries); //was modified by SetBinContent

   delete [] oldBins;
   delete [] oldCount;
   delete [] oldErrors;
   return hnew;
}

//______________________________________________________________________________
void TProfile::RebinAxis(Double_t x, const char* /* ax */)
{
// Profile histogram is resized along x axis such that x is in the axis range.
// The new axis limits are recomputed by doubling iteratively
// the current axis range until the specified value x is within the limits.
// The algorithm makes a copy of the histogram, then loops on all bins
// of the old histogram to fill the rebinned histogram.
// Takes into account errors (Sumw2) if any.
// The bit kCanRebin must be set before invoking this function.
//  Ex:  h->SetBit(TH1::kCanRebin);

   if (!TestBit(kCanRebin)) return;
   TAxis *axis = &fXaxis;
   if (axis->GetXmin() >= axis->GetXmax()) return;
   if (axis->GetNbins() <= 0) return;

   Double_t xmin, xmax;
   if (!FindNewAxisLimits(axis, x, xmin, xmax))
      return;

   //save a copy of this histogram
   TProfile *hold = (TProfile*)Clone();
   hold->SetDirectory(0);
   //set new axis limits
   axis->SetLimits(xmin,xmax);

   Int_t  nbinsx = fXaxis.GetNbins();

   //now loop on all bins and refill
   Reset("ICE"); //reset only Integral, contents and Errors
   for (Int_t binx = 1; binx <= nbinsx; binx++) {
      Int_t destinationBin = fXaxis.FindFixBin(hold->GetXaxis()->GetBinCenter(binx));
      Int_t sourceBin = binx;
      AddBinContent(destinationBin, hold->fArray[sourceBin]);
      fBinEntries.fArray[destinationBin] += hold->fBinEntries.fArray[sourceBin];
      fSumw2.fArray[destinationBin] += hold->fSumw2.fArray[sourceBin];
   }
   fTsumwy = hold->fTsumwy;
   fTsumwy2 = hold->fTsumwy2;

   delete hold;
}

//______________________________________________________________________________
void TProfile::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Reset contents of a Profile histogram*-*-*-*-*-*-*-*-*
//*-*                =====================================
   TH1D::Reset(option);
   fBinEntries.Reset();
   TString opt = option;
   opt.ToUpper();
   if (opt.Contains("ICE")) return;
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

   out<<GetName()<<" = new "<<ClassName()<<"("<<quote<<GetName()<<quote<<","<<quote<<GetTitle()<<quote
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
         out<<"   "<<GetName()<<"->SetBinEntries("<<bin<<","<<bi<<");"<<endl;
      }
   }
   //save bin contents
   for (bin=0;bin<fNcells;bin++) {
      Double_t bc = fArray[bin];
      if (bc) {
         out<<"   "<<GetName()<<"->SetBinContent("<<bin<<","<<bc<<");"<<endl;
      }
   }
   // save bin errors
   if (fSumw2.fN) {
      for (bin=0;bin<fNcells;bin++) {
         Double_t be = TMath::Sqrt(fSumw2.fArray[bin]);
         if (be) {
            out<<"   "<<GetName()<<"->SetBinError("<<bin<<","<<be<<");"<<endl;
         }
      }
   }

   TH1::SavePrimitiveHelp(out, option);
}

//______________________________________________________________________________
void TProfile::Scale(Double_t c1)
{
//*-*-*-*-*Multiply this profile by a constant c1*-*-*-*-*-*-*-*-*
//*-*      ======================================
//
//   this = c1*this
//
// This function uses the services of TProfile::Add
//

   Double_t ent = fEntries;
   fScaling = kTRUE;
   Add(this,this,c1,0);
   fScaling = kFALSE;
   fEntries = ent;
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
      R__b >> (Int_t&)fErrorMode;
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
