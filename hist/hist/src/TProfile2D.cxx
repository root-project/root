// @(#)root/hist:$Id$
// Author: Rene Brun   16/04/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TProfile2D.h"
#include "TMath.h"
#include "THLimitsFinder.h"
#include "Riostream.h"
#include "TVirtualPad.h"
#include "TError.h"
#include "TClass.h"

#include "TProfileHelper.h"

Bool_t TProfile2D::fgApproximate = kFALSE;

ClassImp(TProfile2D)

//______________________________________________________________________________
//
//  Profile2D histograms are used to display the mean
//  value of Z and its RMS for each cell in X,Y.
//  Profile2D histograms are in many cases an
//  elegant replacement of three-dimensional histograms : the inter-relation of three
//  measured quantities X, Y and Z can always be visualized by a three-dimensional
//  histogram or scatter-plot; its representation on the line-printer is not particularly
//  satisfactory, except for sparse data. If Z is an unknown (but single-valued)
//  approximate function of X,Y this function is displayed by a profile2D histogram with
//  much better precision than by a scatter-plot.
//
//  The following formulae show the cumulated contents (capital letters) and the values
//  displayed by the printing or plotting routines (small letters) of the elements for cell I, J.
//
//                                                        2
//      H(I,J)  =  sum Z                  E(I,J)  =  sum Z
//      l(I,J)  =  sum l                  L(I,J)  =  sum l
//      h(I,J)  =  H(I,J)/L(I,J)          s(I,J)  =  sqrt(E(I,J)/L(I,J)- h(I,J)**2)
//      e(I,J)  =  s(I,J)/sqrt(L(I,J))
//
//  In the special case where s(I,J) is zero (eg, case of 1 entry only in one cell)
//  e(I,J) is computed from the average of the s(I,J) for all cells.
//  This simple/crude approximation was suggested in order to keep the cell
//  during a fit operation.
//
//           Example of a profile2D histogram
//{
//  TCanvas *c1 = new TCanvas("c1","Profile histogram example",200,10,700,500);
//  hprof2d  = new TProfile2D("hprof2d","Profile of pz versus px and py",40,-4,4,40,-4,4,0,20);
//  Float_t px, py, pz;
//  for ( Int_t i=0; i<25000; i++) {
//     gRandom->Rannor(px,py);
//     pz = px*px + py*py;
//     hprof2d->Fill(px,py,pz,1);
//  }
//  hprof2d->Draw();
//}
//

//______________________________________________________________________________
TProfile2D::TProfile2D() : TH2D()
{
//*-*-*-*-*-*Default constructor for Profile2D histograms*-*-*-*-*-*-*-*-*
//*-*        ============================================
   fTsumwz = fTsumwz2 = 0;
   fScaling = kFALSE;
   BuildOptions(0,0,"");
}

//______________________________________________________________________________
TProfile2D::~TProfile2D()
{
//*-*-*-*-*-*Default destructor for Profile2D histograms*-*-*-*-*-*-*-*-*
//*-*        ===========================================

}

//______________________________________________________________________________
TProfile2D::TProfile2D(const char *name,const char *title,Int_t nx,Double_t xlow,Double_t xup,Int_t ny,Double_t ylow,Double_t yup,Option_t *option)
    : TH2D(name,title,nx,xlow,xup,ny,ylow,yup)
{
//*-*-*-*-*-*Normal Constructor for Profile histograms*-*-*-*-*-*-*-*-*-*
//*-*        ==========================================
//
//  The first eight parameters are similar to TH2D::TH2D.
//  All values of z are accepted at filling time.
//  To fill a profile2D histogram, one must use TProfile2D::Fill function.
//
//  Note that when filling the profile histogram the function Fill
//  checks if the variable z is betyween fZmin and fZmax.
//  If a minimum or maximum value is set for the Z scale before filling,
//  then all values below zmin or above zmax will be discarded.
//  Setting the minimum or maximum value for the Z scale before filling
//  has the same effect as calling the special TProfile2D constructor below
//  where zmin and zmax are specified.
//
//  H(I,J) is printed as the cell contents. The errors computed are s(I,J) if CHOPT='S'
//  (spread option), or e(I,J) if CHOPT=' ' (error on mean).
//
//        See TProfile2D::BuildOptions for explanation of parameters
//
//   see other constructors below with all possible combinations of
//   fix and variable bin size like in TH2D.

   BuildOptions(0,0,option);
   if (xlow >= xup || ylow >= yup) SetBuffer(fgBufferSize);
}

//______________________________________________________________________________
TProfile2D::TProfile2D(const char *name,const char *title,Int_t nx,const Double_t *xbins,Int_t ny,Double_t ylow,Double_t yup,Option_t *option)
    : TH2D(name,title,nx,xbins,ny,ylow,yup)
{
//  Create a 2-D Profile with variable bins in X and fix bins in Y

   BuildOptions(0,0,option);
}

//______________________________________________________________________________
TProfile2D::TProfile2D(const char *name,const char *title,Int_t nx,Double_t xlow,Double_t xup,Int_t ny,const Double_t *ybins,Option_t *option)
    : TH2D(name,title,nx,xlow,xup,ny,ybins)
{
//  Create a 2-D Profile with fix bins in X and variable bins in Y

   BuildOptions(0,0,option);
}

//______________________________________________________________________________
TProfile2D::TProfile2D(const char *name,const char *title,Int_t nx,const Double_t *xbins,Int_t ny,const Double_t *ybins,Option_t *option)
    : TH2D(name,title,nx,xbins,ny,ybins)
{
//  Create a 2-D Profile with variable bins in X and variable bins in Y

   BuildOptions(0,0,option);
}


//______________________________________________________________________________
TProfile2D::TProfile2D(const char *name,const char *title,Int_t nx,Double_t xlow,Double_t xup,Int_t ny, Double_t ylow,Double_t yup,Double_t zlow,Double_t zup,Option_t *option)
    : TH2D(name,title,nx,xlow,xup,ny,ylow,yup)
{
//*-*-*-*-*-*Constructor for Profile2D histograms with range in z*-*-*-*-*-*
//*-*        ====================================================
//  The first eight parameters are similar to TH2D::TH2D.
//  Only the values of Z between ZMIN and ZMAX will be considered at filling time.
//  zmin and zmax will also be the maximum and minimum values
//  on the z scale when drawing the profile2D.
//
//        See TProfile2D::BuildOptions for more explanations on errors
//

   BuildOptions(zlow,zup,option);
   if (xlow >= xup || ylow >= yup) SetBuffer(fgBufferSize);
}


//______________________________________________________________________________
void TProfile2D::BuildOptions(Double_t zmin, Double_t zmax, Option_t *option)
{
//*-*-*-*-*-*-*Set Profile2D histogram structure and options*-*-*-*-*-*-*-*-*
//*-*          =============================================
//
//    If a cell has N data points all with the same value Z (especially
//    possible when dealing with integers), the spread in Z for that cell
//    is zero, and the uncertainty assigned is also zero, and the cell is
//    ignored in making subsequent fits. If SQRT(Z) was the correct error
//    in the case above, then SQRT(Z)/SQRT(N) would be the correct error here.
//    In fact, any cell with non-zero number of entries N but with zero spread
//    should have an uncertainty SQRT(Z)/SQRT(N).
//
//    Now, is SQRT(Z)/SQRT(N) really the correct uncertainty?
//    that it is only in the case where the Z variable is some sort
//    of counting statistics, following a Poisson distribution. This should
//    probably be set as the default case. However, Z can be any variable
//    from an original NTUPLE, not necessarily distributed "Poissonly".
//    The computation of errors is based on the parameter option:
//    option:
//     ' '  (Default) Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  SQRT(Z)/SQRT(N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     's'            Errors are Spread  for Spread.ne.0. ,
//                      "     "  SQRT(Z)  for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     'i'            Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  1./SQRT(12.*N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//
//    The third case above corresponds to Integer Z values for which the
//    uncertainty is +-0.5, with the assumption that the probability that Z
//    takes any value between Z-0.5 and Z+0.5 is uniform (the same argument
//    goes for Z uniformly distributed between Z and Z+1); this would be
//    useful if Z is an ADC measurement, for example. Other, fancier options
//    would be possible, at the cost of adding one more parameter to the PROFILE2D
//    For example, if all Z variables are distributed according to some
//    known Gaussian of standard deviation Sigma, then:
//     'G'            Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  Sigma/SQRT(N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//    For example, this would be useful when all Z's are experimental quantities
//    measured with the same instrument with precision Sigma.
//
//

   SetErrorOption(option);

   fBinEntries.Set(fNcells);  //*-* create number of entries per cell array

   TH1::Sumw2();                   //*-* create sum of squares of weights array times y
   if (fgDefaultSumw2) Sumw2();    // optionally create sum of squares of weights

   fZmin = zmin;
   fZmax = zmax;
   fScaling = kFALSE;
   fTsumwz  = fTsumwz2 = 0;
}

//______________________________________________________________________________
TProfile2D::TProfile2D(const TProfile2D &profile) : TH2D()
{
   // Copy constructor.

   ((TProfile2D&)profile).Copy(*this);
}


//______________________________________________________________________________
void TProfile2D::Add(TF1 *, Double_t , Option_t*)
{
   // Performs the operation: this = this + c1*f1

   Error("Add","Function not implemented for TProfile2D");
   return;
}


//______________________________________________________________________________
void TProfile2D::Add(const TH1 *h1, Double_t c1)
{
   // Performs the operation: this = this + c1*h1

   if (!h1) {
      Error("Add","Attempt to add a non-existing profile");
      return;
   }
   if (!h1->InheritsFrom(TProfile2D::Class())) {
      Error("Add","Attempt to add a non-profile2D object");
      return;
   }

   TProfileHelper::Add(this, this, h1, 1, c1);
}

//______________________________________________________________________________
void TProfile2D::Add(const TH1 *h1, const TH1 *h2, Double_t c1, Double_t c2)
{
//*-*-*-*-*Replace contents of this profile2D by the addition of h1 and h2*-*-*
//*-*      ===============================================================
//
//   this = c1*h1 + c2*h2
//

   if (!h1 || !h2) {
      Error("Add","Attempt to add a non-existing profile");
      return;
   }
   if (!h1->InheritsFrom(TProfile2D::Class())) {
      Error("Add","Attempt to add a non-profile2D object");
      return;
   }
   if (!h2->InheritsFrom(TProfile2D::Class())) {
      Error("Add","Attempt to add a non-profile2D object");
      return;
   }
   TProfileHelper::Add(this, h1, h2, c1, c2);
}


//______________________________________________________________________________
void TProfile2D::Approximate(Bool_t approx)
{
//     static function
// set the fgApproximate flag. When the flag is true, the function GetBinError
// will approximate the bin error with the average profile error on all bins
// in the following situation only
//  - the number of bins in the profile2D is less than 10404 (eg 100x100)
//  - the bin number of entries is small ( <5)
//  - the estimated bin error is extremely small compared to the bin content
//  (see TProfile2D::GetBinError)

   fgApproximate = approx;
}


//______________________________________________________________________________
Int_t TProfile2D::BufferEmpty(Int_t action)
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
         Double_t x = fBuffer[4*i+2];
         if (x < xmin) xmin = x;
         if (x > xmax) xmax = x;
         Double_t y = fBuffer[4*i+3];
         if (y < ymin) ymin = y;
         if (y > ymax) ymax = y;
      }
      if (fXaxis.GetXmax() <= fXaxis.GetXmin() || fYaxis.GetXmax() <= fYaxis.GetXmin()) {
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(this,xmin,xmax,ymin,ymax);
      } else {
         fBuffer = 0;
         Int_t keep = fBufferSize; fBufferSize = 0;
         if (xmin <  fXaxis.GetXmin()) RebinAxis(xmin,&fXaxis);
         if (xmax >= fXaxis.GetXmax()) RebinAxis(xmax,&fXaxis);
         if (ymin <  fYaxis.GetXmin()) RebinAxis(ymin,&fYaxis);
         if (ymax >= fYaxis.GetXmax()) RebinAxis(ymax,&fYaxis);
         fBuffer = buffer;
         fBufferSize = keep;
      }
   }

   fBuffer = 0;
   for (Int_t i=0;i<nbentries;i++) {
      Fill(buffer[4*i+2],buffer[4*i+3],buffer[4*i+4],buffer[4*i+1]);
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
Int_t TProfile2D::BufferFill(Double_t x, Double_t y, Double_t z, Double_t w)
{
// accumulate arguments in buffer. When buffer is full, empty the buffer
// fBuffer[0] = number of entries in buffer
// fBuffer[1] = w of first entry
// fBuffer[2] = x of first entry
// fBuffer[3] = y of first entry
// fBuffer[4] = z of first entry

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
   if (4*nbentries+4 >= fBufferSize) {
      BufferEmpty(1);
      return Fill(x,y,z,w);
   }
   fBuffer[4*nbentries+1] = w;
   fBuffer[4*nbentries+2] = x;
   fBuffer[4*nbentries+3] = y;
   fBuffer[4*nbentries+4] = z;
   fBuffer[0] += 1;
   return -2;
}

//______________________________________________________________________________
void TProfile2D::Copy(TObject &obj) const
{
//*-*-*-*-*-*-*-*Copy a Profile2D histogram to a new profile2D histogram*-*-*-*
//*-*            =======================================================

   TH2D::Copy(((TProfile2D&)obj));
   fBinEntries.Copy(((TProfile2D&)obj).fBinEntries);
   fBinSumw2.Copy(((TProfile2D&)obj).fBinSumw2);
   for (int bin=0;bin<fNcells;bin++) {
      ((TProfile2D&)obj).fArray[bin]        = fArray[bin];
      ((TProfile2D&)obj).fSumw2.fArray[bin] = fSumw2.fArray[bin];
   }
   ((TProfile2D&)obj).fZmin = fZmin;
   ((TProfile2D&)obj).fZmax = fZmax;
   ((TProfile2D&)obj).fScaling   = fScaling;
   ((TProfile2D&)obj).fErrorMode = fErrorMode;
   ((TProfile2D&)obj).fTsumwz    = fTsumwz;
   ((TProfile2D&)obj).fTsumwz2   = fTsumwz2;
}


//______________________________________________________________________________
void TProfile2D::Divide(TF1 *, Double_t )
{
   // Performs the operation: this = this/(c1*f1)

   Error("Divide","Function not implemented for TProfile2D");
   return;
}

//______________________________________________________________________________
void TProfile2D::Divide(const TH1 *h1)
{
//*-*-*-*-*-*-*-*-*-*-*Divide this profile2D by h1*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===========================
//
//   this = this/h1
//

   if (!h1) {
      Error("Divide","Attempt to divide a non-existing profile2D");
      return;
   }
   if (!h1->InheritsFrom(TProfile2D::Class())) {
      Error("Divide","Attempt to divide a non-profile2D object");
      return;
   }
   TProfile2D *p1 = (TProfile2D*)h1;

//*-*- Check profile compatibility
   Int_t nx = GetNbinsX();
   if (nx != p1->GetNbinsX()) {
      Error("Divide","Attempt to divide profiles with different number of bins");
      return;
   }
   Int_t ny = GetNbinsY();
   if (ny != p1->GetNbinsY()) {
      Error("Divide","Attempt to divide profiles with different number of bins");
      return;
   }

//*-*- Reset statistics
   fEntries = fTsumw   = fTsumw2 = fTsumwx = fTsumwx2 = 0;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin,binx,biny;
   Double_t *cu1 = p1->GetW();
   Double_t *er1 = p1->GetW2();
   Double_t *en1 = p1->GetB();
   Double_t c0,c1,w,z,x,y;
   for (binx =0;binx<=nx+1;binx++) {
      for (biny =0;biny<=ny+1;biny++) {
         bin   = biny*(fXaxis.GetNbins()+2) + binx;
         c0  = fArray[bin];
         c1  = cu1[bin];
         if (c1) w = c0/c1;
         else    w = 0;
         fArray[bin] = w;
         z = TMath::Abs(w);
         x = fXaxis.GetBinCenter(binx);
         y = fYaxis.GetBinCenter(biny);
         fEntries++;
         fTsumw   += z;
         fTsumw2  += z*z;
         fTsumwx  += z*x;
         fTsumwx2 += z*x*x;
         fTsumwy  += z*y;
         fTsumwy2 += z*y*y;
         fTsumwxy += z*x*y;
         fTsumwz  += z;
         fTsumwz2 += z*z;
         Double_t e0 = fSumw2.fArray[bin];
         Double_t e1 = er1[bin];
         Double_t c12= c1*c1;
         if (!c1) fSumw2.fArray[bin] = 0;
         else     fSumw2.fArray[bin] = (e0*c1*c1 + e1*c0*c0)/(c12*c12);
         if (!en1[bin]) fBinEntries.fArray[bin] = 0;
         else           fBinEntries.fArray[bin] /= en1[bin];
      }
   }
   // mantaining the correct sum of weights square is not supported when dividing
   // bin error resulting from division of profile needs to be checked 
   if (fBinSumw2.fN) { 
      Warning("Divide","Cannot preserve during the division of profiles the sum of bin weight square");
      fBinSumw2 = TArrayD();
   }
}


//______________________________________________________________________________
void TProfile2D::Divide(const TH1 *h1, const TH1 *h2, Double_t c1, Double_t c2, Option_t *option)
{
//*-*-*-*-*Replace contents of this profile2D by the division of h1 by h2*-*-*
//*-*      ==============================================================
//
//   this = c1*h1/(c2*h2)
//

   TString opt = option;
   opt.ToLower();
   Bool_t binomial = kFALSE;
   if (opt.Contains("b")) binomial = kTRUE;
   if (!h1 || !h2) {
      Error("Divide","Attempt to divide a non-existing profile2D");
      return;
   }
   if (!h1->InheritsFrom(TProfile2D::Class())) {
      Error("Divide","Attempt to divide a non-profile2D object");
      return;
   }
   TProfile2D *p1 = (TProfile2D*)h1;
   if (!h2->InheritsFrom(TProfile2D::Class())) {
      Error("Divide","Attempt to divide a non-profile2D object");
      return;
   }
   TProfile2D *p2 = (TProfile2D*)h2;

//*-*- Check histogram compatibility
   Int_t nx = GetNbinsX();
   if (nx != p1->GetNbinsX() || nx != p2->GetNbinsX()) {
      Error("Divide","Attempt to divide profiles with different number of bins");
      return;
   }
   Int_t ny = GetNbinsY();
   if (ny != p1->GetNbinsY() || ny != p2->GetNbinsY()) {
      Error("Divide","Attempt to divide profiles with different number of bins");
      return;
   }
   if (!c2) {
      Error("Divide","Coefficient of dividing profile cannot be zero");
      return;
   }

//*-*- Reset statistics
   fEntries = fTsumw   = fTsumw2 = fTsumwx = fTsumwx2 = 0;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin,binx,biny;
   Double_t *cu1 = p1->GetW();
   Double_t *cu2 = p2->GetW();
   Double_t *er1 = p1->GetW2();
   Double_t *er2 = p2->GetW2();
   Double_t *en1 = p1->GetB();
   Double_t *en2 = p2->GetB();
   Double_t b1,b2,w,z,x,y,ac1,ac2;
   ac1 = TMath::Abs(c1);
   ac2 = TMath::Abs(c2);
   for (binx =0;binx<=nx+1;binx++) {
      for (biny =0;biny<=ny+1;biny++) {
         bin   = biny*(fXaxis.GetNbins()+2) + binx;
         b1  = cu1[bin];
         b2  = cu2[bin];
         if (b2) w = c1*b1/(c2*b2);
         else    w = 0;
         fArray[bin] = w;
         z = TMath::Abs(w);
         x = fXaxis.GetBinCenter(binx);
         y = fYaxis.GetBinCenter(biny);
         fEntries++;
         fTsumw   += z;
         fTsumw2  += z*z;
         fTsumwx  += z*x;
         fTsumwx2 += z*x*x;
         fTsumwy  += z*y;
         fTsumwy2 += z*y*y;
         fTsumwxy += z*x*y;
         fTsumwz  += z;
         fTsumwz2 += z*z;
         Double_t e1 = er1[bin];
         Double_t e2 = er2[bin];
       //Double_t b22= b2*b2*d2;
         Double_t b22= b2*b2*TMath::Abs(c2);
         if (!b2) fSumw2.fArray[bin] = 0;
         else {
            if (binomial) {
               fSumw2.fArray[bin] = TMath::Abs(w*(1-w)/(c2*b2));
            } else {
               fSumw2.fArray[bin] = ac1*ac2*(e1*b2*b2 + e2*b1*b1)/(b22*b22);
            }
         }
         if (!en2[bin]) fBinEntries.fArray[bin] = 0;
         else           fBinEntries.fArray[bin] = en1[bin]/en2[bin];
      }
   }
}

//______________________________________________________________________________
TH1 *TProfile2D::DrawCopy(Option_t *option) const
{
//*-*-*-*-*-*-*-*Draw a copy of this profile2D histogram*-*-*-*-*-*-*-*-*-*-*
//*-*            =======================================
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TProfile2D *newpf = (TProfile2D*)Clone();
   newpf->SetDirectory(0);
   newpf->SetBit(kCanDelete);
   newpf->AppendPad(option);
   return newpf;
}

//______________________________________________________________________________
Int_t TProfile2D::Fill(Double_t x, Double_t y, Double_t z)
{
//*-*-*-*-*-*-*-*-*-*-*Fill a Profile2D histogram (no weights)*-*-*-*-*-*-*-*
//*-*                  =======================================

   if (fBuffer) return BufferFill(x,y,z,1);

   Int_t bin,binx,biny;

   if (fZmin != fZmax) {
      if (z <fZmin || z> fZmax || TMath::IsNaN(z) ) return -1;
   }

   fEntries++;
   binx =fXaxis.FindBin(x);
   biny =fYaxis.FindBin(y);
   if (binx <0 || biny <0) return -1;
   bin = GetBin(binx, biny); 
   fArray[bin] += z; 
   fSumw2.fArray[bin] += z*z;
   fBinEntries.fArray[bin] += 1;
   if (fBinSumw2.fN)  fBinSumw2.fArray[bin] += 1;
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
   fTsumwz  += z;
   fTsumwz2 += z*z;
   return bin;
}

//______________________________________________________________________________
Int_t TProfile2D::Fill(Double_t x, const char *namey, Double_t z)
{
// Fill a Profile2D histogram (no weights)
//
   Int_t bin,binx,biny;

   if (fZmin != fZmax) {
      if (z <fZmin || z> fZmax || TMath::IsNaN(z)) return -1;
   }

   fEntries++;
   binx =fXaxis.FindBin(x);
   biny =fYaxis.FindBin(namey);
   if (binx <0 || biny <0) return -1;
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin, z);
   fSumw2.fArray[bin] += (Double_t)z*z;
   fBinEntries.fArray[bin] += 1;
   if (fBinSumw2.fN)  fBinSumw2.fArray[bin] += 1;
   if (binx == 0 || binx > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   Double_t y = fYaxis.GetBinCenter(biny);
   ++fTsumw;
   ++fTsumw2;
   fTsumwx  += x;
   fTsumwx2 += x*x;
   fTsumwy  += y;
   fTsumwy2 += y*y;
   fTsumwxy += x*y;
   fTsumwz  += z;
   fTsumwz2 += z*z;
   return bin;
}

//______________________________________________________________________________
Int_t TProfile2D::Fill(const char *namex, const char *namey, Double_t z)
{
// Fill a Profile2D histogram (no weights)
//
   Int_t bin,binx,biny;

   if (fZmin != fZmax) {
      if (z <fZmin || z> fZmax || TMath::IsNaN(z) ) return -1;
   }

   fEntries++;
   binx =fXaxis.FindBin(namex);
   biny =fYaxis.FindBin(namey);
   if (binx <0 || biny <0) return -1;
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin, z);
   fSumw2.fArray[bin] += (Double_t)z*z;
   fBinEntries.fArray[bin] += 1;
   if (fBinSumw2.fN)  fBinSumw2.fArray[bin] += 1;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   Double_t x = fYaxis.GetBinCenter(binx);
   Double_t y = fYaxis.GetBinCenter(biny);
   ++fTsumw;
   ++fTsumw2;
   fTsumwx  += x;
   fTsumwx2 += x*x;
   fTsumwy  += y;
   fTsumwy2 += y*y;
   fTsumwxy += x*y;
   fTsumwz  += z;
   fTsumwz2 += z*z;
   return bin;
}

//______________________________________________________________________________
Int_t TProfile2D::Fill(const char *namex, Double_t y, Double_t z)
{
// Fill a Profile2D histogram (no weights)
//
   Int_t bin,binx,biny;

   if (fZmin != fZmax) {
      if (z <fZmin || z> fZmax || TMath::IsNaN(z)) return -1;
   }

   fEntries++;
   binx =fXaxis.FindBin(namex);
   biny =fYaxis.FindBin(y);
   if (binx <0 || biny <0) return -1;
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin, z);
   fSumw2.fArray[bin] += (Double_t)z*z;
   fBinEntries.fArray[bin] += 1;
   if (fBinSumw2.fN)  fBinSumw2.fArray[bin] += 1;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   Double_t x = fYaxis.GetBinCenter(binx);
   ++fTsumw;
   ++fTsumw2;
   fTsumwx  += x;
   fTsumwx2 += x*x;
   fTsumwy  += y;
   fTsumwy2 += y*y;
   fTsumwxy += x*y;
   fTsumwz  += z;
   fTsumwz2 += z*z;
   return bin;
}

//______________________________________________________________________________
Int_t TProfile2D::Fill(Double_t x, Double_t y, Double_t z, Double_t w)
{
//*-*-*-*-*-*-*-*-*-*-*Fill a Profile2D histogram with weights*-*-*-*-*-*-*-*
//*-*                  =======================================

   if (fBuffer) return BufferFill(x,y,z,w);

   Int_t bin,binx,biny;

   if (fZmin != fZmax) {
      if (z <fZmin || z> fZmax || TMath::IsNaN(z)) return -1;
   }

   Double_t u= (w > 0 ? w : -w);
   fEntries++;
   binx =fXaxis.FindBin(x);
   biny =fYaxis.FindBin(y);
   if (binx <0 || biny <0) return -1;
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin, u*z);
   fSumw2.fArray[bin] += u*z*z;
   fBinEntries.fArray[bin] += u;
   if (fBinSumw2.fN)  fBinSumw2.fArray[bin] += u*u;
   if (binx == 0 || binx > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   if (biny == 0 || biny > fYaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   fTsumw   += u;
   fTsumw2  += u*u;
   fTsumwx  += u*x;
   fTsumwx2 += u*x*x;
   fTsumwy  += u*y;
   fTsumwy2 += u*y*y;
   fTsumwxy += u*x*y;
   fTsumwz  += u*z;
   fTsumwz2 += u*z*z;
   return bin;
}

//______________________________________________________________________________
Double_t TProfile2D::GetBinContent(Int_t bin) const
{
//*-*-*-*-*-*-*Return bin content of a Profile2D histogram*-*-*-*-*-*-*-*-*
//*-*          ===========================================

   if (fBuffer) ((TProfile2D*)this)->BufferEmpty();

   if (bin < 0 || bin >= fNcells) return 0;
   if (fBinEntries.fArray[bin] == 0) return 0;
   if (!fArray) return 0;
   return fArray[bin]/fBinEntries.fArray[bin];
}

//______________________________________________________________________________
Double_t TProfile2D::GetBinEntries(Int_t bin) const
{
//*-*-*-*-*-*-*Return bin entries of a Profile2D histogram*-*-*-*-*-*-*-*-*
//*-*          ===========================================

   if (fBuffer) ((TProfile2D*)this)->BufferEmpty();

   if (bin < 0 || bin >= fNcells) return 0;
   return fBinEntries.fArray[bin];
}

//______________________________________________________________________________
Double_t TProfile2D::GetBinEffectiveEntries(Int_t bin)
{
//            Return bin effective entries for a weighted filled Profile histogram. 
//            In case of an unweighted profile, it is equivalent to the number of entries per bin   
//            The effective entries is defined as the square of the sum of the weights divided by the 
//            sum of the weights square. 
//            TProfile::Sumw2() must be called before filling the profile with weights. 
//            Only by calling this method the  sum of the square of the weights per bin is stored. 
//  
//*-*          =========================================

   return TProfileHelper::GetBinEffectiveEntries(this, bin);
}

//______________________________________________________________________________
Double_t TProfile2D::GetBinError(Int_t bin) const
{
// *-*-*-*-*-*-*Return bin error of a Profile2D histogram*-*-*-*-*-*-*-*-*
//
// Computing errors: A moving field
// =================================
// The computation of errors for a TProfile2D has evolved with the versions
// of ROOT. The difficulty is in computing errors for bins with low statistics.
// - prior to version 3.10, we had no special treatment of low statistic bins.
//   As a result, these bins had huge errors. The reason is that the
//   expression eprim2 is very close to 0 (rounding problems) or 0.
// - The algorithm is modified/protected for the case
//   when a TProfile2D is projected (ProjectionX). The previous algorithm
//   generated a N^2 problem when projecting a TProfile2D with a large number of
//   bins (eg 100000).
// - in version 3.10/02, a new static function TProfile::Approximate
//   is introduced to enable or disable (default) the approximation.
//   (see also comments in TProfile::GetBinError)

   return TProfileHelper::GetBinError((TProfile2D*)this, bin);
}

//______________________________________________________________________________
Option_t *TProfile2D::GetErrorOption() const
{
//*-*-*-*-*-*-*-*-*-*Return option to compute profile2D errors*-*-*-*-*-*-*-*
//*-*                =========================================

   if (fErrorMode == kERRORSPREAD)  return "s";
   if (fErrorMode == kERRORSPREADI) return "i";
   if (fErrorMode == kERRORSPREADG) return "g";
   return "";
}

//______________________________________________________________________________
void TProfile2D::GetStats(Double_t *stats) const
{
   // fill the array stats from the contents of this profile
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
   //
   // If no axis-subrange is specified (via TAxis::SetRange), the array stats
   // is simply a copy of the statistics quantities computed at filling time.
   // If a sub-range is specified, the function recomputes these quantities
   // from the bin contents in the current axis range.

   if (fBuffer) ((TProfile2D*)this)->BufferEmpty();

   // Loop on bins
   if (fTsumw == 0 || fXaxis.TestBit(TAxis::kAxisRange) || fYaxis.TestBit(TAxis::kAxisRange)) {
      Int_t bin, binx, biny;
      Double_t w, w2;
      Double_t x,y;
      for (bin=0;bin<9;bin++) stats[bin] = 0;
      if (!fBinEntries.fArray) return;
      Int_t firstBinX = fXaxis.GetFirst();
      Int_t lastBinX  = fXaxis.GetLast();
      Int_t firstBinY = fYaxis.GetFirst();
      Int_t lastBinY  = fYaxis.GetLast();
      // include underflow/overflow if TH1::StatOverflows(kTRUE) in case no range is set on the axis
      if (fgStatOverflows) {
        if ( !fXaxis.TestBit(TAxis::kAxisRange) ) {
            if (firstBinX == 1) firstBinX = 0;
            if (lastBinX ==  fXaxis.GetNbins() ) lastBinX += 1;
         }
         if ( !fYaxis.TestBit(TAxis::kAxisRange) ) {
            if (firstBinY == 1) firstBinY = 0;
            if (lastBinY ==  fYaxis.GetNbins() ) lastBinY += 1;
         }
      }
      for (biny = firstBinY; biny <= lastBinY; biny++) {
         y = fYaxis.GetBinCenter(biny);
         for (binx = firstBinX; binx <= lastBinX; binx++) {
            bin = GetBin(binx,biny);
            w         = fBinEntries.fArray[bin];
            w2        = (fBinSumw2.fN ? fBinSumw2.fArray[bin] : w );  
            x         = fXaxis.GetBinCenter(binx);
            stats[0] += w;
            stats[1] += w2;
            stats[2] += w*x;
            stats[3] += w*x*x;
            stats[4] += w*y;
            stats[5] += w*y*y;
            stats[6] += w*x*y;
            stats[7] += fArray[bin];
            stats[8] += fSumw2.fArray[bin];
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
      stats[7] = fTsumwz;
      stats[8] = fTsumwz2;
   }
}

//___________________________________________________________________________
void TProfile2D::LabelsDeflate(Option_t *ax)
{
// Reduce the number of bins for this axis to the number of bins having a label.

   TProfileHelper::LabelsDeflate(this, ax);
}

//___________________________________________________________________________
void TProfile2D::LabelsInflate(Option_t *ax)
{
// Double the number of bins for axis.
// Refill histogram
// This function is called by TAxis::FindBin(const char *label)

   TProfileHelper::LabelsInflate(this, ax);
}

//___________________________________________________________________________
void TProfile2D::LabelsOption(Option_t *option, Option_t *ax)
{
//  Set option(s) to draw axis with labels
//  option = "a" sort by alphabetic order
//         = ">" sort by decreasing values
//         = "<" sort by increasing values
//         = "h" draw labels horizonthal
//         = "v" draw labels vertical
//         = "u" draw labels up (end of label right adjusted)
//         = "d" draw labels down (start of label left adjusted)


   TAxis *axis = GetXaxis();
   if (ax[0] == 'y' || ax[0] == 'Y') axis = GetYaxis();
   THashList *labels = axis->GetLabels();
   if (!labels) {
      Warning("LabelsOption","Cannot sort. No labels");
      return;
   }
   TString opt = option;
   opt.ToLower();
   if (opt.Contains("h")) {
      axis->SetBit(TAxis::kLabelsHori);
      axis->ResetBit(TAxis::kLabelsVert);
      axis->ResetBit(TAxis::kLabelsDown);
      axis->ResetBit(TAxis::kLabelsUp);
   }
   if (opt.Contains("v")) {
      axis->SetBit(TAxis::kLabelsVert);
      axis->ResetBit(TAxis::kLabelsHori);
      axis->ResetBit(TAxis::kLabelsDown);
      axis->ResetBit(TAxis::kLabelsUp);
   }
   if (opt.Contains("u")) {
      axis->SetBit(TAxis::kLabelsUp);
      axis->ResetBit(TAxis::kLabelsVert);
      axis->ResetBit(TAxis::kLabelsDown);
      axis->ResetBit(TAxis::kLabelsHori);
   }
   if (opt.Contains("d")) {
      axis->SetBit(TAxis::kLabelsDown);
      axis->ResetBit(TAxis::kLabelsVert);
      axis->ResetBit(TAxis::kLabelsHori);
      axis->ResetBit(TAxis::kLabelsUp);
   }
   Int_t sort = -1;
   if (opt.Contains("a")) sort = 0;
   if (opt.Contains(">")) sort = 1;
   if (opt.Contains("<")) sort = 2;
   if (sort < 0) return;

   Int_t nx = fXaxis.GetNbins()+2;
   Int_t ny = fYaxis.GetNbins()+2;
   Int_t n = TMath::Min(axis->GetNbins(), labels->GetSize());
   Int_t *a = new Int_t[n+2];
   Int_t i,j,k,bin;
   Double_t *sumw   = new Double_t[nx*ny];
   Double_t *errors = new Double_t[nx*ny];
   Double_t *ent    = new Double_t[nx*ny];
   THashList *labold = new THashList(labels->GetSize(),1);
   TIter nextold(labels);
   TObject *obj;
   while ((obj=nextold())) {
      labold->Add(obj);
   }
   labels->Clear();
   if (sort > 0) {
      //---sort by values of bins
      Double_t *pcont = new Double_t[n+2];
      for (i=0;i<=n;i++) pcont[i] = 0;
      for (i=1;i<nx;i++) {
         for (j=1;j<ny;j++) {
            bin = i+nx*j;
            sumw[bin]   = fArray[bin];
            errors[bin] = fSumw2.fArray[bin];
            ent[bin]    = fBinEntries.fArray[bin];
            if (axis == GetXaxis()) k = i;
            else                    k = j;
            if (fBinEntries.fArray[bin] != 0) pcont[k-1] += fArray[bin]/fBinEntries.fArray[bin];
         }
      }
      if (sort ==1) TMath::Sort(n,pcont,a,kTRUE);  //sort by decreasing values
      else          TMath::Sort(n,pcont,a,kFALSE); //sort by increasing values
      delete [] pcont;
      for (i=0;i<n;i++) {
         obj = labold->At(a[i]);
         labels->Add(obj);
         obj->SetUniqueID(i+1);
      }
      for (i=1;i<nx;i++) {
         for (j=1;j<ny;j++) {
            bin = i+nx*j;
            if (axis == GetXaxis()) {
               fArray[bin] = sumw[a[i-1]+1+nx*j];
               fSumw2.fArray[bin] = errors[a[i-1]+1+nx*j];
               fBinEntries.fArray[bin] = ent[a[i-1]+1+nx*j];
            } else {
               fArray[bin] = sumw[i+nx*(a[j-1]+1)];
               fSumw2.fArray[bin] = errors[i+nx*(a[j-1]+1)];
               fBinEntries.fArray[bin] = ent[i+nx*(a[j-1]+1)];
            }
         }
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
      for (i=0;i<nx;i++) {
         for (j=0;j<ny;j++) {
            bin = i+nx*j;
            sumw[bin]   = fArray[bin];
            errors[bin] = fSumw2.fArray[bin];
            ent[bin]    = fBinEntries.fArray[bin];
         }
      }
      for (i=0;i<nx;i++) {
         for (j=0;j<ny;j++) {
            bin = i+nx*j;
            if (axis == GetXaxis()) {
               fArray[bin] = sumw[a[i]+nx*j];
               fSumw2.fArray[bin] = errors[a[i]+nx*j];
               fBinEntries.fArray[bin] = ent[a[i]+nx*j];
            } else {
               fArray[bin] = sumw[i+nx*a[j]];
               fSumw2.fArray[bin] = errors[i+nx*a[j]];
               fBinEntries.fArray[bin] = ent[i+nx*a[j]];
            }
         }
      }
   }
   delete labold;
   if (a)      delete [] a;
   if (sumw)   delete [] sumw;
   if (errors) delete [] errors;
   if (ent)    delete [] ent;
}

//______________________________________________________________________________
Long64_t TProfile2D::Merge(TCollection *li)
{
   //Merge all histograms in the collection in this histogram.
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

   return TProfileHelper::Merge(this, li);
}

//______________________________________________________________________________
void TProfile2D::Multiply(TF1 *, Double_t )
{
   // Performs the operation: this = this*c1*f1

   Error("Multiply","Function not implemented for TProfile2D");
   return;
}

//______________________________________________________________________________
void TProfile2D::Multiply(const TH1 *)
{
//*-*-*-*-*-*-*-*-*-*-*Multiply this profile2D by h1*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =============================
//
//   this = this*h1
//
   Error("Multiply","Multiplication of profile2D histograms not implemented");
}


//______________________________________________________________________________
void TProfile2D::Multiply(const TH1 *, const TH1 *, Double_t, Double_t, Option_t *)
{
//*-*-*-*-*Replace contents of this profile2D by multiplication of h1 by h2*-*
//*-*      ================================================================
//
//   this = (c1*h1)*(c2*h2)
//

   Error("Multiply","Multiplication of profile2D histograms not implemented");
}

//______________________________________________________________________________
TH2D *TProfile2D::ProjectionXY(const char *name, Option_t *option) const
{
//*-*-*-*-*Project this profile2D into a 2-D histogram along X,Y*-*-*-*-*-*-*
//*-*      =====================================================
//
//   The projection is always of the type TH2D.
//
//   if option "E" is specified  the errors of the projected histogram are computed and set 
//      to be equal to the errors of the profile.
//      Option "E" is defined as the default one in the header file. 
//   if option "" is specified the histogram errors are simply the sqrt of its content
//   if option "B" is specified, the content of bin of the returned histogram
//      will be equal to the GetBinEntries(bin) of the profile,
//   if option "C=E" the bin contents of the projection are set to the
//       bin errors of the profile
//   if option "W" is specified the bin content of the projected histogram  is set to the 
//       product of the bin content of the profile and the entries. 
//       With this option the returned histogram will be equivalent to the one obtained by 
//       filling directly a TH2D using the 3-rd value as a weight. 
//       This option makes sense only for profile filled with all weights =1. 
//       When the profile is weighted (filled with weights different than 1) the  
//       bin error of the projected histogram (obtained using this option "W") cannot be 
//       correctly computed from the information stored in the profile. 


   TString opt = option;
   opt.ToLower();
   

   Int_t nx = fXaxis.GetNbins();
   Int_t ny = fYaxis.GetNbins();

   // Create the projection histogram
   char *pname = (char*)name;
   if (strcmp(name,"_px") == 0) {
      Int_t nch = strlen(GetName()) + 4;
      pname = new char[nch];
      snprintf(pname,nch,"%s%s",GetName(),name);
   }
   TH2D *h1 = new TH2D(pname,GetTitle(),nx,fXaxis.GetXmin(),fXaxis.GetXmax(),ny,fYaxis.GetXmin(),fYaxis.GetXmax());
   Bool_t computeErrors = kFALSE;
   Bool_t cequalErrors  = kFALSE;
   Bool_t binEntries    = kFALSE;
   Bool_t binWeight     = kFALSE;
   if (opt.Contains("b")) binEntries = kTRUE;
   if (opt.Contains("e")) computeErrors = kTRUE;
   if (opt.Contains("w")) binWeight = kTRUE;
   if (opt.Contains("c=e")) {cequalErrors = kTRUE; computeErrors=kFALSE;}
   if (computeErrors || binWeight ) h1->Sumw2();
   if (pname != name)  delete [] pname;

   // Fill the projected histogram
   Int_t bin,binx, biny;
   Double_t cont;
   for (binx =0;binx<=nx+1;binx++) {
      for (biny =0;biny<=ny+1;biny++) {
         bin = GetBin(binx,biny);
         
         if (binEntries)         cont = GetBinEntries(bin);
         else if (cequalErrors)  cont = GetBinError(bin);
         else if (binWeight)     cont = GetBinContent(bin) * GetBinEntries(bin);
         else                    cont = GetBinContent(bin);    // default case

         h1->SetBinContent(bin ,cont);

         // if option E projected histogram errors are same as profile
         if (computeErrors ) h1->SetBinError(bin , GetBinError(bin) );
         // in case of option W bin error is deduced from bin sum of z**2 values of profile
         // this is correct only if the profile is unweighted 
         if (binWeight)      h1->SetBinError(bin , TMath::Sqrt(fSumw2.fArray[bin] ) );
         // in case of bin entries and h1 has sumw2 set set, we need to set the also bin error
         if (binEntries && h1->GetSumw2() ) {
            Double_t err2;
            if (fBinSumw2.fN) 
               err2 = fBinSumw2.fArray[bin]; 
            else 
               err2 = cont;  // this is fBinEntries.fArray[bin]
            h1->SetBinError(bin, TMath::Sqrt(err2 ) ); 
         }
      }
   }
   h1->SetEntries(fEntries);
   return h1;
}

//______________________________________________________________________________
void TProfile2D::PutStats(Double_t *stats)
{
   // Replace current statistics with the values in array stats

   fTsumw   = stats[0];
   fTsumw2  = stats[1];
   fTsumwx  = stats[2];
   fTsumwx2 = stats[3];
   fTsumwy  = stats[4];
   fTsumwy2 = stats[5];
   fTsumwxy = stats[6];
   fTsumwz  = stats[7];
   fTsumwz2 = stats[8];
}

//______________________________________________________________________________
void TProfile2D::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Reset contents of a Profile2D histogram*-*-*-*-*-*-*-*
//*-*                =======================================
   TH2D::Reset(option);
   fBinEntries.Reset();
   fBinSumw2.Reset();
   TString opt = option;
   opt.ToUpper();
   if (opt.Contains("ICE")) return;
   fTsumwz = fTsumwz2 = 0;
}


//______________________________________________________________________________
void TProfile2D::RebinAxis(Double_t x, TAxis *axis)
{
// Profile histogram is resized along axis such that x is in the axis range.
// The new axis limits are recomputed by doubling iteratively
// the current axis range until the specified value x is within the limits.
// The algorithm makes a copy of the histogram, then loops on all bins
// of the old histogram to fill the rebinned histogram.
// Takes into account errors (Sumw2) if any.
// The bit kCanRebin must be set before invoking this function.
//  Ex:  h->SetBit(TH1::kCanRebin);

   TProfile2D* hold = TProfileHelper::RebinAxis(this, x, axis);
   if ( hold ) {
      fTsumwz  = hold->fTsumwz;
      fTsumwz2 = hold->fTsumwz2;
      delete hold;
   }
}

//______________________________________________________________________________
TProfile2D * TProfile2D::Rebin2D(Int_t nxgroup ,Int_t nygroup,const char * newname ) {
   //   -*-*-*Rebin this histogram grouping nxgroup/nygroup bins along the xaxis/yaxis together*-*-*-*-
   //         =================================================================================
   //   if newname is not blank a new profile hnew is created.
   //   else the current histogram is modified (default)
   //   The parameter nxgroup/nygroup indicate how many bins along the xaxis/yaxis of this
   //   have to be merged into one bin of hnew
   //   If the original profile has errors stored (via Sumw2), the resulting
   //   profile has new errors correctly calculated.
   //
   //   examples: if hpxpy is an existing TProfile2D profile with 40 x 40 bins
   //     hpxpy->Rebin2D();  // merges two bins along the xaxis and yaxis in one
   //                        // Carefull: previous contents of hpxpy are lost
   //     hpxpy->Rebin2D(3,5);  // merges 3 bins along the xaxis and 5 bins along the yaxis in one
   //                           // Carefull: previous contents of hpxpy are lost
   //     hpxpy->RebinX(5); //merges five bins along the xaxis in one in hpxpy
   //     TProfile2D *hnew = hpxpy->RebinY(5,"hnew"); // creates a new profile hnew
   //                                                 // merging 5 bins of hpxpy along the yaxis in one bin
   //
   //   NOTE : If nxgroup/nygroup is not an exact divider of the number of bins,
   //          along the xaxis/yaxis the top limit(s) of the rebinned profile
   //          is changed to the upper edge of the xbin=newxbins*nxgroup resp.
   //          ybin=newybins*nygroup and the remaining bins are added to
   //          the overflow bin.
   //          Statistics will be recomputed from the new bin contents.

   //something to do?
   if((nxgroup != 1) || (nygroup != 1)){
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

      //warning if bins are added to the overflow bin
      if(newxbins*nxgroup != nxbins) {
	 Warning("Rebin", "nxgroup=%d should be an exact divider of nxbins=%d",nxgroup,nxbins);
      }
      if(newybins*nygroup != nybins) {
	 Warning("Rebin", "nygroup=%d should be an exact divider of nybins=%d",nygroup,nybins);
      }

      //save old bin contents in new arrays
      Double_t *oldBins   = new Double_t[(nxbins+2)*(nybins+2)];
      Double_t *oldCount  = new Double_t[(nxbins+2)*(nybins+2)];
      Double_t *oldErrors = new Double_t[(nxbins+2)*(nybins+2)];
      Double_t *oldBinw2  = (fBinSumw2.fN ? new Double_t[(nxbins+2)*(nybins+2)] : 0  );
      Double_t *cu1 = GetW();
      Double_t *er1 = GetW2();
      Double_t *en1 = GetB();
      Double_t *ew1 = GetB2();
      for(Int_t ibin=0; ibin < (nxbins+2)*(nybins+2); ibin++){
	 oldBins[ibin]   = cu1[ibin];
	 oldCount[ibin]  = en1[ibin];
	 oldErrors[ibin] = er1[ibin];
	 if (ew1 && fBinSumw2.fN) oldBinw2[ibin]  = ew1[ibin];
      }

      // create a clone of the old profile if newname is specified
      TProfile2D *hnew = this;
      if(newname && strlen(newname) > 0) {
	 hnew = (TProfile2D*)Clone(newname);
      }

      // in case of nxgroup/nygroup not an exact divider of nxbins/nybins, 
      // top limit is changed (see NOTE in method comment) 
      if(newxbins*nxgroup != nxbins) {
	 xmax = fXaxis.GetBinUpEdge(newxbins*nxgroup);
	 hnew->fTsumw = 0; //stats must be reset because top bins will be moved to overflow bin
      }
      if(newybins*nygroup != nybins) {
	 ymax = fYaxis.GetBinUpEdge(newybins*nygroup);
	 hnew->fTsumw = 0; //stats must be reset because top bins will be moved to overflow bin
      }

      //rebin the axis
      if((fXaxis.GetXbins()->GetSize() > 0) || (fYaxis.GetXbins()->GetSize() > 0)){
	 Double_t* xbins = new Double_t[newxbins+1];
	 Double_t* ybins = new Double_t[newybins+1];
	 for(Int_t i=0; i < newxbins+1; i++)
	    xbins[i] = fXaxis.GetBinLowEdge(1+i*nxgroup);
	 for(Int_t j=0; j < newybins+1; j++)
	    ybins[j] = fYaxis.GetBinLowEdge(1+j*nygroup);
	 hnew->SetBins(newxbins,xbins,newybins,ybins);
	 delete [] xbins;
	 delete [] ybins;
      }
      //fixed bin size
      else{
	 hnew->SetBins(newxbins,xmin,xmax,newybins,ymin,ymax);
      }

      //merge bins
      Double_t *cu2 = hnew->GetW();
      Double_t *er2 = hnew->GetW2();
      Double_t *en2 = hnew->GetB();
      Double_t *ew2 = hnew->GetB2();
      Double_t binContent, binCount, binError, binSumw2;
      //connection between x and y bin number and linear global bin number:
      //global bin = xbin + (nxbins+2) * ybin
      Int_t oldxbin = 1;
      Int_t oldybin = 1;
      //global bin number
      Int_t bin;
      for(Int_t xbin = 1; xbin <= newxbins; xbin++){
	 oldybin = 1;
	 for(Int_t ybin = 1; ybin <= newybins; ybin++){
	    binContent = 0;
	    binCount   = 0;
	    binError   = 0;
	    binSumw2   = 0;
	    for(Int_t i=0; i < nxgroup; i++){
	       if(oldxbin + i > nxbins) break;
	       for(Int_t j=0; j < nygroup; j++){
		  if(oldybin + j > nybins) break;
		  bin = oldxbin + i + (nxbins+2)*(oldybin+j);
		  binContent += oldBins[bin];
		  binCount += oldCount[bin];
		  binError += oldErrors[bin];
		  if(fBinSumw2.fN) binSumw2 += oldBinw2[bin];
	       }
	    }
	    bin = xbin + (newxbins + 2)*ybin;
	    cu2[bin] = binContent;
	    er2[bin] = binError;
	    en2[bin] = binCount;
	    if(fBinSumw2.fN) ew2[bin] = binSumw2;
	    oldybin += nygroup;
	 }
	 oldxbin += nxgroup;
      }

      //copy the underflow bin in x and y (0,0)
      cu2[0] = oldBins[0];
      er2[0] = oldErrors[0];
      en2[0] = oldCount[0];
      if(fBinSumw2.fN) ew2[0] = oldBinw2[0];
      //calculate overflow bin in x and y (newxbins+1,newybins+1)
      //therefore the oldxbin and oldybin from above are needed!
      binContent = 0;
      binCount   = 0;
      binError   = 0;
      binSumw2   = 0;
      for(Int_t i=oldxbin; i <= nxbins+1; i++){
	 for(Int_t j=oldybin; j <= nybins+1; j++){
	    //global bin number
	    bin = i + (nxbins+2)*j;
	    binContent += oldBins[bin];
	    binCount += oldCount[bin];
	    binError += oldErrors[bin];
	    if(fBinSumw2.fN) binSumw2 += oldBinw2[bin];
	 }
      }
      bin = (newxbins+2)*(newybins+2)-1;
      cu2[bin] = binContent;
      er2[bin] = binError;
      en2[bin] = binCount;
      if(fBinSumw2.fN) ew2[bin] = binSumw2;
      //calculate overflow bin in x and underflow bin in y (newxbins+1,0)
      binContent = 0;
      binCount   = 0;
      binError   = 0;
      binSumw2   = 0;
      for(Int_t i=oldxbin; i <= nxbins+1; i++){
	 bin = i;
	 binContent += oldBins[bin];
	 binCount += oldCount[bin];
	 binError += oldErrors[bin];
	 if(fBinSumw2.fN) binSumw2 += oldBinw2[bin];
      }
      bin = newxbins + 1;
      cu2[bin] = binContent;
      er2[bin] = binError;
      en2[bin] = binCount;
      if(fBinSumw2.fN) ew2[bin] = binSumw2;
      //calculate underflow bin in x and overflow bin in y (0,newybins+1)
      binContent = 0;
      binCount   = 0;
      binError   = 0;
      binSumw2   = 0;
      for(Int_t i=oldybin; i <= nybins+1; i++){
	 bin = i*(nxbins + 2);
	 binContent += oldBins[bin];
	 binCount += oldCount[bin];
	 binError += oldErrors[bin];
	 if(fBinSumw2.fN) binSumw2 += oldBinw2[bin];
      }
      bin = (newxbins + 2)*(newybins + 1);
      cu2[bin] = binContent;
      er2[bin] = binError;
      en2[bin] = binCount;
      if(fBinSumw2.fN) ew2[bin] = binSumw2;
      //calculate under/overflow contents in y for the new x bins
      Double_t binContentuf, binCountuf, binErroruf, binSumw2uf;
      Double_t binContentof, binCountof, binErrorof, binSumw2of;
      Int_t ufbin, ofbin;
      Int_t oldxbin2 = 1;
      for(Int_t xbin = 1; xbin <= newxbins; xbin++){
	 binContentuf = 0;
	 binCountuf   = 0;
	 binErroruf   = 0;
	 binSumw2uf   = 0;
	 binContentof = 0;
	 binCountof   = 0;
	 binErrorof   = 0;
	 binSumw2of   = 0;
	 for(Int_t i = 0; i < nxgroup; i++){
	    //index of under/overflow bin for y in old binning
	    ufbin = (oldxbin2 + i);
	    binContentuf += oldBins[ufbin];
	    binCountuf   += oldCount[ufbin];
	    binErroruf   += oldErrors[ufbin];
	    if(fBinSumw2.fN) binSumw2uf   += oldBinw2[ufbin];
	    for(Int_t j = oldybin; j <= nybins+1; j++)
	    {
	       ofbin = ufbin + j*(nxbins + 2);
	       binContentof += oldBins[ofbin];
	       binCountof   += oldCount[ofbin];
	       binErrorof   += oldErrors[ofbin];
	       if(fBinSumw2.fN) binSumw2of   += oldBinw2[ofbin];
	    }
	 }
	 //index of under/overflow bin for y in new binning
	 ufbin = xbin;
	 ofbin = ufbin + (newybins + 1)*(newxbins + 2);
	 cu2[ufbin] = binContentuf;
	 er2[ufbin] = binErroruf;
	 en2[ufbin] = binCountuf;
	 if(fBinSumw2.fN) ew2[ufbin] = binSumw2uf;
	 cu2[ofbin] = binContentof;
	 er2[ofbin] = binErrorof;
	 en2[ofbin] = binCountof;
	 if(fBinSumw2.fN) ew2[ofbin] = binSumw2of;
	 
	 oldxbin2 += nxgroup;
      }
      //calculate under/overflow contents in x for the new y bins
      Int_t oldybin2 = 1;
      for(Int_t ybin = 1; ybin <= newybins; ybin++){
	 binContentuf = 0;
	 binCountuf   = 0;
	 binErroruf   = 0;
	 binSumw2uf   = 0;
	 binContentof = 0;
	 binCountof   = 0;
	 binErrorof   = 0;
	 binSumw2of   = 0;
	 for(Int_t i = 0; i < nygroup; i++){
	    //index of under/overflow bin for x in old binning
	    ufbin = (oldybin2 + i)*(nxbins+2);
	    binContentuf += oldBins[ufbin];
	    binCountuf   += oldCount[ufbin];
	    binErroruf   += oldErrors[ufbin];
	    if(fBinSumw2.fN) binSumw2uf   += oldBinw2[ufbin];
	    for(Int_t j = oldxbin; j <= nxbins+1; j++)
	    {
	       ofbin = j + ufbin;
	       binContentof += oldBins[ofbin];
	       binCountof   += oldCount[ofbin];
	       binErrorof   += oldErrors[ofbin];
	       if(fBinSumw2.fN) binSumw2of   += oldBinw2[ofbin];
	    }
	 }
	 //index of under/overflow bin for x in new binning
	 ufbin = ybin * (newxbins + 2);
	 ofbin = newxbins + 1 + ufbin;
	 cu2[ufbin] = binContentuf;
	 er2[ufbin] = binErroruf;
	 en2[ufbin] = binCountuf;
	 if(fBinSumw2.fN) ew2[ufbin] = binSumw2uf;
	 cu2[ofbin] = binContentof;
	 er2[ofbin] = binErrorof;
	 en2[ofbin] = binCountof;
	 if(fBinSumw2.fN) ew2[ofbin] = binSumw2of;

	 oldybin2 += nygroup;
	 }

      delete [] oldBins;
      delete [] oldCount;
      delete [] oldErrors;
      if (oldBinw2) delete [] oldBinw2;

      return hnew;
   }
   //nxgroup == nygroup == 1
   else{
      if((newname) && (strlen(newname) > 0))
	 return (TProfile2D*)Clone(newname);
      else
	 return this;
   }   
}

//______________________________________________________________________________
TProfile2D * TProfile2D::RebinX(Int_t ngroup,const char * newname ) {
// Rebin only the X axis
// see Rebin2D

   return Rebin2D(ngroup,1,newname);
}

//______________________________________________________________________________
TProfile2D * TProfile2D::RebinY(Int_t ngroup,const char * newname ) {
// Rebin only the Y axis
// see Rebin2D

   return Rebin2D(1,ngroup,newname);
}

//______________________________________________________________________________
void TProfile2D::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out

   //Note the following restrictions in the code generated:
   // - variable bin size not implemented
   // - SetErrorOption not implemented


   char quote = '"';
   out <<"   "<<endl;
   out <<"   "<<ClassName()<<" *";

   out << GetName() << " = new " << ClassName() << "(" << quote
       << GetName() << quote << "," << quote<< GetTitle() << quote
       << "," << GetXaxis()->GetNbins();
   out << "," << GetXaxis()->GetXmin()
       << "," << GetXaxis()->GetXmax();
   out << "," << GetYaxis()->GetNbins();
   out << "," << GetYaxis()->GetXmin()
       << "," << GetYaxis()->GetXmax();
   out << "," << fZmin
       << "," << fZmax;
   out << ");" << endl;


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

   TH1::SavePrimitiveHelp(out, GetName(), option);
}

//______________________________________________________________________________
void TProfile2D::Scale(Double_t c1, Option_t * option)
{
// *-*-*-*-*Multiply this profile2D by a constant c1*-*-*-*-*-*-*-*-*
// *-*      ========================================
//
//   this = c1*this
//
// This function uses the services of TProfile2D::Add
//

   TProfileHelper::Scale(this, c1, option);
}

//______________________________________________________________________________
void TProfile2D::SetBinEntries(Int_t bin, Double_t w)
{
//*-*-*-*-*-*-*-*-*Set the number of entries in bin*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ================================

   if (bin < 0 || bin >= fNcells) return;
   fBinEntries.fArray[bin] = w;
}

//______________________________________________________________________________
void TProfile2D::SetBins(Int_t nx, Double_t xmin, Double_t xmax, Int_t ny, Double_t ymin, Double_t ymax)
{
//*-*-*-*-*-*-*-*-*Redefine  x axis parameters*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ===========================

   fXaxis.Set(nx,xmin,xmax);
   fYaxis.Set(ny,ymin,ymax);
   fNcells = (nx+2)*(ny+2);
   SetBinsLength(fNcells);
   fBinEntries.Set(fNcells);
   fSumw2.Set(fNcells);
   if (fBinSumw2.fN) fBinSumw2.Set(fNcells);
}


//______________________________________________________________________________
void TProfile2D::SetBuffer(Int_t buffersize, Option_t *)
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
   fBufferSize = 1 + 4*buffersize;
   fBuffer = new Double_t[fBufferSize];
   memset(fBuffer,0,8*fBufferSize);
}

//______________________________________________________________________________
void TProfile2D::SetErrorOption(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Set option to compute profile2D errors*-*-*-*-*-*-*-*
//*-*                =======================================
//
//    The computation of errors is based on the parameter option:
//    option:
//     ' '  (Default) Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  SQRT(Z)/SQRT(N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     's'            Errors are Spread  for Spread.ne.0. ,
//                      "     "  SQRT(Z)  for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     'i'            Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  1./SQRT(12.*N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//   See TProfile2D::BuildOptions for explanation of all options

   TString opt = option;
   opt.ToLower();
   fErrorMode = kERRORMEAN;
   if (opt.Contains("s")) fErrorMode = kERRORSPREAD;
   if (opt.Contains("i")) fErrorMode = kERRORSPREADI;
   if (opt.Contains("g")) fErrorMode = kERRORSPREADG;
}

//______________________________________________________________________________
void TProfile2D::Streamer(TBuffer &R__b)
{
   // Stream an object of class TProfile2D.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         R__b.ReadClassBuffer(TProfile2D::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TH2D::Streamer(R__b);
      fBinEntries.Streamer(R__b);
      Int_t errorMode;
      R__b >> errorMode;
      fErrorMode = (EErrorType)errorMode;
      if (R__v < 2) {
         Float_t zmin,zmax;
         R__b >> zmin; fZmin = zmin;
         R__b >> zmax; fZmax = zmax;
      } else {
         R__b >> fZmin;
         R__b >> fZmax;
      }
      R__b.CheckByteCount(R__s, R__c, TProfile2D::IsA());
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TProfile2D::Class(),this);
   }
}

//______________________________________________________________________________
void TProfile2D::Sumw2()
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
