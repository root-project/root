// @(#)root/hist:$Id$
// Author: Rene Brun   17/05/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TProfile3D.h"
#include "THashList.h"
#include "TMath.h"
#include "THLimitsFinder.h"
#include "Riostream.h"
#include "TVirtualPad.h"
#include "TError.h"
#include "TClass.h"

#include "TProfileHelper.h"

Bool_t TProfile3D::fgApproximate = kFALSE;

ClassImp(TProfile3D)

//______________________________________________________________________________
//
//  Profile3D histograms are used to display the mean
//  value of T and its RMS for each cell in X,Y,Z.
//  Profile3D histograms are in many cases an
//  The inter-relation of three measured quantities X, Y, Z and T can always 
//  be visualized by a four-dimensional histogram or scatter-plot; 
//  its representation on the line-printer is not particularly
//  satisfactory, except for sparse data. If T is an unknown (but single-valued)
//  approximate function of X,Y,Z this function is displayed by a profile3D histogram with
//  much better precision than by a scatter-plot.
//
//  The following formulae show the cumulated contents (capital letters) and the values
//  displayed by the printing or plotting routines (small letters) of the elements for cell I, J.
//
//                                                                2
//      H(I,J,K)  =  sum T                      E(I,J,K)  =  sum T
//      l(I,J,K)  =  sum l                      L(I,J,K)  =  sum l
//      h(I,J,K)  =  H(I,J,K)/L(I,J,K)          s(I,J,K)  =  sqrt(E(I,J,K)/L(I,J,K)- h(I,J,K)**2)
//      e(I,J,K)  =  s(I,J,K)/sqrt(L(I,J,K))
//
//  In the special case where s(I,J,K) is zero (eg, case of 1 entry only in one cell)
//  e(I,J,K) is computed from the average of the s(I,J,K) for all cells.
//  This simple/crude approximation was suggested in order to keep the cell
//  during a fit operation.
//
//           Example of a profile3D histogram
//{
//  TCanvas *c1 = new TCanvas("c1","Profile histogram example",200,10,700,500);
//  hprof3d  = new TProfile3D("hprof3d","Profile of pt versus px, py and pz",40,-4,4,40,-4,4,40,0,20);
//  Double_t px, py, pz, pt;
//  TRandom3 r(0);
//  for ( Int_t i=0; i<25000; i++) {
//     r.Rannor(px,py);
//     pz = px*px + py*py;
//     pt = r.Landau(0,1);
//     hprof3d->Fill(px,py,pz,pt,1);
//  }
//  hprof3d->Draw();
//}
//
// NOTE: A TProfile3D is drawn as it was a simple TH3

//______________________________________________________________________________
TProfile3D::TProfile3D() : TH3D()
{
//*-*-*-*-*-*Default constructor for Profile3D histograms*-*-*-*-*-*-*-*-*
//*-*        ============================================
   fTsumwt = fTsumwt2 = 0;
   fScaling = kFALSE;
   BuildOptions(0,0,"");
}

//______________________________________________________________________________
TProfile3D::~TProfile3D()
{
//*-*-*-*-*-*Default destructor for Profile3D histograms*-*-*-*-*-*-*-*-*
//*-*        ===========================================

}

//______________________________________________________________________________
TProfile3D::TProfile3D(const char *name,const char *title,Int_t nx,Double_t xlow,Double_t xup,Int_t ny,Double_t ylow,Double_t yup,Int_t nz, Double_t zlow,Double_t zup,Option_t *option)
    : TH3D(name,title,nx,xlow,xup,ny,ylow,yup,nz,zlow,zup)
{
//*-*-*-*-*-*Normal Constructor for Profile histograms*-*-*-*-*-*-*-*-*-*
//*-*        ==========================================
//
//  The first eleven parameters are similar to TH3D::TH3D.
//  All values of t are accepted at filling time.
//  To fill a profile3D histogram, one must use TProfile3D::Fill function.
//
//  Note that when filling the profile histogram the function Fill
//  checks if the variable t is betyween fTmin and fTmax.
//  If a minimum or maximum value is set for the T scale before filling,
//  then all values below tmin or above tmax will be discarded.
//  Setting the minimum or maximum value for the T scale before filling
//  has the same effect as calling the special TProfile3D constructor below
//  where tmin and tmax are specified.
//
//  H(I,J,K) is printed as the cell contents. The errors computed are s(I,J,K) if CHOPT='S'
//  (spread option), or e(I,J,K) if CHOPT=' ' (error on mean).
//
//        See TProfile3D::BuildOptions for explanation of parameters
//
//   see other constructors below with all possible combinations of
//   fix and variable bin size like in TH3D.

   BuildOptions(0,0,option);
   if (xlow >= xup || ylow >= yup || zlow >= zup) SetBuffer(fgBufferSize);
}

//______________________________________________________________________________
TProfile3D::TProfile3D(const char *name,const char *title,Int_t nx,const Double_t *xbins,Int_t ny,const Double_t *ybins,Int_t nz,const Double_t *zbins,Option_t *option)
    : TH3D(name,title,nx,xbins,ny,ybins,nz,zbins)
{
//  Create a 3-D Profile with variable bins in X , Y and Z

   BuildOptions(0,0,option);
}

//______________________________________________________________________________
void TProfile3D::BuildOptions(Double_t tmin, Double_t tmax, Option_t *option)
{
//*-*-*-*-*-*-*Set Profile3D histogram structure and options*-*-*-*-*-*-*-*-*
//*-*          =============================================
//
//    If a cell has N data points all with the same value T (especially
//    possible when dealing with integers), the spread in T for that cell
//    is zero, and the uncertainty assigned is also zero, and the cell is
//    ignored in making subsequent fits. If SQRT(T) was the correct error
//    in the case above, then SQRT(T)/SQRT(N) would be the correct error here.
//    In fact, any cell with non-zero number of entries N but with zero spread
//    should have an uncertainty SQRT(T)/SQRT(N).
//
//    Now, is SQRT(T)/SQRT(N) really the correct uncertainty?
//    that it is only in the case where the T variable is some sort
//    of counting statistics, following a Poisson distribution. This should
//    probably be set as the default case. However, T can be any variable
//    from an original NTUPLE, not necessarily distributed "Poissonly".
//    The computation of errors is based on the parameter option:
//    option:
//     ' '  (Default) Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  SQRT(T)/SQRT(N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     's'            Errors are Spread  for Spread.ne.0. ,
//                      "     "  SQRT(T)  for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     'i'            Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  1./SQRT(12.*N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//
//    The third case above corresponds to Integer T values for which the
//    uncertainty is +-0.5, with the assumption that the probability that T
//    takes any value between T-0.5 and T+0.5 is uniform (the same argument
//    goes for T uniformly distributed between T and T+1); this would be
//    useful if T is an ADC measurement, for example. Other, fancier options
//    would be possible, at the cost of adding one more parameter to the PROFILE2D
//    For example, if all T variables are distributed according to some
//    known Gaussian of standard deviation Sigma, then:
//     'G'            Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  Sigma/SQRT(N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//    For example, this would be useful when all T's are experimental quantities
//    measured with the same instrument with precision Sigma.
//
//

   SetErrorOption(option);

   fBinEntries.Set(fNcells);  //*-* create number of entries per cell array

   TH1::Sumw2();                   //*-* create sum of squares of weights array times y 
   if (fgDefaultSumw2) Sumw2();    // optionally create sum of squares of weights

   fTmin = tmin;
   fTmax = tmax;
   fScaling = kFALSE;
   fTsumwt  = fTsumwt2 = 0;
}

//______________________________________________________________________________
TProfile3D::TProfile3D(const TProfile3D &profile) : TH3D()
{
   //copy constructor
   ((TProfile3D&)profile).Copy(*this);
}


//______________________________________________________________________________
void TProfile3D::Add(TF1 *, Double_t , Option_t*)
{
   // Performs the operation: this = this + c1*f1

   Error("Add","Function not implemented for TProfile3D");
   return;
}


//______________________________________________________________________________
void TProfile3D::Add(const TH1 *h1, Double_t c1)
{
   // Performs the operation: this = this + c1*h1

   if (!h1) {
      Error("Add","Attempt to add a non-existing profile");
      return;
   }
   if (!h1->InheritsFrom(TProfile3D::Class())) {
      Error("Add","Attempt to add a non-profile2D object");
      return;
   }

   TProfileHelper::Add(this, this, h1, 1, c1);
}

//______________________________________________________________________________
void TProfile3D::Add(const TH1 *h1, const TH1 *h2, Double_t c1, Double_t c2)
{
//*-*-*-*-*Replace contents of this profile3D by the addition of h1 and h2*-*-*
//*-*      ===============================================================
//
//   this = c1*h1 + c2*h2
//

   if (!h1 || !h2) {
      Error("Add","Attempt to add a non-existing profile");
      return;
   }
   if (!h1->InheritsFrom(TProfile3D::Class())) {
      Error("Add","Attempt to add a non-profile3D object");
      return;
   }
   if (!h2->InheritsFrom(TProfile3D::Class())) {
      Error("Add","Attempt to add a non-profile3D object");
      return;
   }

   TProfileHelper::Add(this, h1, h2, c1, c2);
}


//______________________________________________________________________________
void TProfile3D::Approximate(Bool_t approx)
{
//     static function
// set the fgApproximate flag. When the flag is true, the function GetBinError
// will approximate the bin error with the average profile error on all bins
// in the following situation only
//  - the number of bins in the profile3D is less than 10404 (eg 100x100x100)
//  - the bin number of entries is small ( <5)
//  - the estimated bin error is extremely small compared to the bin content
//  (see TProfile3D::GetBinError)

   fgApproximate = approx;
}


//______________________________________________________________________________
Int_t TProfile3D::BufferEmpty(Int_t action)
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
      Double_t zmin = fBuffer[4];
      Double_t zmax = zmin;
      for (Int_t i=1;i<nbentries;i++) {
         Double_t x = fBuffer[5*i+2];
         if (x < xmin) xmin = x;
         if (x > xmax) xmax = x;
         Double_t y = fBuffer[5*i+3];
         if (y < ymin) ymin = y;
         if (y > ymax) ymax = y;
         Double_t z = fBuffer[5*i+4];
         if (z < zmin) zmin = z;
         if (z > zmax) zmax = z;
     }
      if (fXaxis.GetXmax() <= fXaxis.GetXmin() || fYaxis.GetXmax() <= fYaxis.GetXmin() || fZaxis.GetXmax() <= fZaxis.GetXmin()) {
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(this,xmin,xmax,ymin,ymax,zmin,zmax);
      } else {
         fBuffer = 0;
         Int_t keep = fBufferSize; fBufferSize = 0;
         if (xmin <  fXaxis.GetXmin()) RebinAxis(xmin,&fXaxis);
         if (xmax >= fXaxis.GetXmax()) RebinAxis(xmax,&fXaxis);
         if (ymin <  fYaxis.GetXmin()) RebinAxis(ymin,&fYaxis);
         if (ymax >= fYaxis.GetXmax()) RebinAxis(ymax,&fYaxis);
         if (zmin <  fZaxis.GetXmin()) RebinAxis(zmin,&fZaxis);
         if (zmax >= fZaxis.GetXmax()) RebinAxis(zmax,&fZaxis);
         fBuffer = buffer;
         fBufferSize = keep;
      }
   }

   fBuffer = 0;
   for (Int_t i=0;i<nbentries;i++) {
      Fill(buffer[5*i+2],buffer[5*i+3],buffer[5*i+4],buffer[5*i+5],buffer[5*i+1]);
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
Int_t TProfile3D::BufferFill(Double_t x, Double_t y, Double_t z, Double_t t, Double_t w)
{
// accumulate arguments in buffer. When buffer is full, empty the buffer
// fBuffer[0] = number of entries in buffer
// fBuffer[1] = w of first entry
// fBuffer[2] = x of first entry
// fBuffer[3] = y of first entry
// fBuffer[4] = z of first entry
// fBuffer[5] = t of first entry

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
   if (5*nbentries+5 >= fBufferSize) {
      BufferEmpty(1);
      return Fill(x,y,z,t,w);
   }
   fBuffer[5*nbentries+1] = w;
   fBuffer[5*nbentries+2] = x;
   fBuffer[5*nbentries+3] = y;
   fBuffer[5*nbentries+4] = z;
   fBuffer[5*nbentries+5] = t;
   fBuffer[0] += 1;
   return -2;
}

//______________________________________________________________________________
void TProfile3D::Copy(TObject &obj) const
{
//*-*-*-*-*-*-*-*Copy a Profile3D histogram to a new profile2D histogram*-*-*-*
//*-*            =======================================================

   TH3D::Copy(((TProfile3D&)obj));
   fBinEntries.Copy(((TProfile3D&)obj).fBinEntries);
   fBinSumw2.Copy(((TProfile3D&)obj).fBinSumw2);
   for (int bin=0;bin<fNcells;bin++) {
      ((TProfile3D&)obj).fArray[bin]        = fArray[bin];
      ((TProfile3D&)obj).fSumw2.fArray[bin] = fSumw2.fArray[bin];
   }
   ((TProfile3D&)obj).fTmin = fTmin;
   ((TProfile3D&)obj).fTmax = fTmax;
   ((TProfile3D&)obj).fScaling = fScaling;
   ((TProfile3D&)obj).fErrorMode = fErrorMode;
   ((TProfile3D&)obj).fTsumwt  = fTsumwt;
   ((TProfile3D&)obj).fTsumwt2 = fTsumwt2;
}


//______________________________________________________________________________
void TProfile3D::Divide(TF1 *, Double_t )
{
   // Performs the operation: this = this/(c1*f1)

   Error("Divide","Function not implemented for TProfile3D");
   return;
}

//______________________________________________________________________________
void TProfile3D::Divide(const TH1 *h1)
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
   if (!h1->InheritsFrom(TProfile3D::Class())) {
      Error("Divide","Attempt to divide a non-profile3D object");
      return;
   }
   TProfile3D *p1 = (TProfile3D*)h1;

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
   Int_t nz = GetNbinsZ();
   if (nz != p1->GetNbinsZ()) {
      Error("Divide","Attempt to divide profiles with different number of bins");
      return;
   }

//*-*- Reset statistics
   fEntries = fTsumw   = fTsumw2 = fTsumwx = fTsumwx2 = 0;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin,binx,biny,binz;
   Double_t *cu1 = p1->GetW();
   Double_t *er1 = p1->GetW2();
   Double_t *en1 = p1->GetB();
   Double_t c0,c1,w,u,x,y,z;
   for (binx =0;binx<=nx+1;binx++) {
      for (biny =0;biny<=ny+1;biny++) {
         for (binz =0;binz<=nz+1;binz++) {
            bin   = GetBin(binx,biny,binz);
            c0  = fArray[bin];
            c1  = cu1[bin];
            if (c1) w = c0/c1;
            else    w = 0;
            fArray[bin] = w;
            u = TMath::Abs(w);
            x = fXaxis.GetBinCenter(binx);
            y = fYaxis.GetBinCenter(biny);
            z = fZaxis.GetBinCenter(binz);
            fEntries++;
            fTsumw   += u;
            fTsumw2  += u*u;
            fTsumwx  += u*x;
            fTsumwx2 += u*x*x;
            fTsumwy  += u*y;
            fTsumwy2 += u*y*y;
            fTsumwxy += u*x*y;
            fTsumwz  += u;
            fTsumwz2 += u*z;
            fTsumwxz += u*x*z;
            fTsumwyz += u*y*z;
            fTsumwt  += u;
            fTsumwt2 += u*u;
            Double_t e0 = fSumw2.fArray[bin];
            Double_t e1 = er1[bin];
            Double_t c12= c1*c1;
            if (!c1) fSumw2.fArray[bin] = 0;
            else     fSumw2.fArray[bin] = (e0*c1*c1 + e1*c0*c0)/(c12*c12);
            if (!en1[bin]) fBinEntries.fArray[bin] = 0;
            else           fBinEntries.fArray[bin] /= en1[bin];
         }
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
void TProfile3D::Divide(const TH1 *h1, const TH1 *h2, Double_t c1, Double_t c2, Option_t *option)
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
   if (!h1->InheritsFrom(TProfile3D::Class())) {
      Error("Divide","Attempt to divide a non-profile2D object");
      return;
   }
   TProfile3D *p1 = (TProfile3D*)h1;
   if (!h2->InheritsFrom(TProfile3D::Class())) {
      Error("Divide","Attempt to divide a non-profile2D object");
      return;
   }
   TProfile3D *p2 = (TProfile3D*)h2;

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
   Int_t nz = GetNbinsZ();
   if (nz != p1->GetNbinsZ() || nz != p2->GetNbinsZ()) {
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
   Int_t bin,binx,biny,binz;
   Double_t *cu1 = p1->GetW();
   Double_t *cu2 = p2->GetW();
   Double_t *er1 = p1->GetW2();
   Double_t *er2 = p2->GetW2();
   Double_t *en1 = p1->GetB();
   Double_t *en2 = p2->GetB();
   Double_t b1,b2,w,u,x,y,z,ac1,ac2;
   ac1 = TMath::Abs(c1);
   ac2 = TMath::Abs(c2);
   for (binx =0;binx<=nx+1;binx++) {
      for (biny =0;biny<=ny+1;biny++) {
         for (binz =0;binz<=nz+1;binz++) {
            bin   = GetBin(binx,biny,binz);
            b1  = cu1[bin];
            b2  = cu2[bin];
            if (b2) w = c1*b1/(c2*b2);
            else    w = 0;
            fArray[bin] = w;
            u = TMath::Abs(w);
            x = fXaxis.GetBinCenter(binx);
            y = fYaxis.GetBinCenter(biny);
            z = fZaxis.GetBinCenter(biny);
            fEntries++;
            fTsumw   += u;
            fTsumw2  += u*u;
            fTsumwx  += u*x;
            fTsumwx2 += u*x*x;
            fTsumwy  += u*y;
            fTsumwy2 += u*y*y;
            fTsumwxy += u*x*y;
            fTsumwz  += u*z;
            fTsumwz2 += u*z*z;
            fTsumwxz += u*x*z;
            fTsumwyz += u*y*z;
            fTsumwt  += u;
            fTsumwt2 += u*u;
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
}

//______________________________________________________________________________
TH1 *TProfile3D::DrawCopy(Option_t *option) const
{
//*-*-*-*-*-*-*-*Draw a copy of this profile3D histogram*-*-*-*-*-*-*-*-*-*-*
//*-*            =======================================
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TProfile3D *newpf = new TProfile3D();
   Copy(*newpf);
   newpf->SetDirectory(0);
   newpf->SetBit(kCanDelete);
   newpf->AppendPad(option);
   return newpf;
}

//______________________________________________________________________________
Int_t TProfile3D::Fill(Double_t x, Double_t y, Double_t z, Double_t t)
{
//*-*-*-*-*-*-*-*-*-*-*Fill a Profile3D histogram (no weights)*-*-*-*-*-*-*-*
//*-*                  =======================================

   if (fBuffer) return BufferFill(x,y,z,t,1);

   Int_t bin,binx,biny,binz;

   if (fTmin != fTmax) {
      if (t <fTmin || t> fTmax) return -1;
   }

   fEntries++;
   binx =fXaxis.FindBin(x);
   biny =fYaxis.FindBin(y);
   binz =fZaxis.FindBin(z);
   if (binx <0 || biny <0 || binz<0) return -1;
   bin  = GetBin(binx,biny,binz);
   AddBinContent(bin, t);
   fSumw2.fArray[bin] += (Double_t)t*t;
   fBinEntries.fArray[bin] += 1;
   if (fBinSumw2.fN)  fBinSumw2.fArray[bin] += 1;
   if (binx == 0 || binx > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   if (biny == 0 || biny > fYaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   if (binz == 0 || binz > fZaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
//printf("x=%g, y=%g, z=%g, t=%g, binx=%d, biny=%d, binz=%d, bin=%d\n",x,y,z,t,binx,biny,binz,bin);
   ++fTsumw;
   ++fTsumw2;
   fTsumwx  += x;
   fTsumwx2 += x*x;
   fTsumwy  += y;
   fTsumwy2 += y*y;
   fTsumwxy += x*y;
   fTsumwz  += z;
   fTsumwz2 += z*z;
   fTsumwxz += x*z;
   fTsumwyz += y*z;
   fTsumwt  += t;
   fTsumwt2 += t*t;
   return bin;
}

//______________________________________________________________________________
Int_t TProfile3D::Fill(Double_t x, Double_t y, Double_t z, Double_t t, Double_t w)
{
//*-*-*-*-*-*-*-*-*-*-*Fill a Profile3D histogram with weights*-*-*-*-*-*-*-*
//*-*                  =======================================

   if (fBuffer) return BufferFill(x,y,z,t,w);

   Int_t bin,binx,biny,binz;

   if (fTmin != fTmax) {
      if (t <fTmin || z> fTmax) return -1;
   }

   Double_t u= (w > 0 ? w : -w);
   fEntries++;
   binx =fXaxis.FindBin(x);
   biny =fYaxis.FindBin(y);
   binz =fZaxis.FindBin(z);
   if (binx <0 || biny <0 || binz<0) return -1;
   bin  = GetBin(binx,biny,binz);
   AddBinContent(bin, u*t);
   fSumw2.fArray[bin] += u*t*t;
   fBinEntries.fArray[bin] += u;
   if (fBinSumw2.fN)  fBinSumw2.fArray[bin] += u*u;
   if (binx == 0 || binx > fXaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   if (biny == 0 || biny > fYaxis.GetNbins()) {
      if (!fgStatOverflows) return -1;
   }
   if (binz == 0 || binz > fZaxis.GetNbins()) {
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
   fTsumwxz += u*x*z;
   fTsumwyz += u*y*z;
   fTsumwt  += u*t;
   fTsumwt2 += u*t*t;
   return bin;
}

//______________________________________________________________________________
Double_t TProfile3D::GetBinContent(Int_t bin) const
{
//*-*-*-*-*-*-*Return bin content of a Profile3D histogram*-*-*-*-*-*-*-*-*
//*-*          ===========================================

   if (fBuffer) ((TProfile3D*)this)->BufferEmpty();

   if (bin < 0 || bin >= fNcells) return 0;
   if (fBinEntries.fArray[bin] == 0) return 0;
   if (!fArray) return 0;
   return fArray[bin]/fBinEntries.fArray[bin];
}

//______________________________________________________________________________
Double_t TProfile3D::GetBinEntries(Int_t bin) const
{
//*-*-*-*-*-*-*Return bin entries of a Profile3D histogram*-*-*-*-*-*-*-*-*
//*-*          ===========================================

   if (fBuffer) ((TProfile3D*)this)->BufferEmpty();

   if (bin < 0 || bin >= fNcells) return 0;
   return fBinEntries.fArray[bin];
}

//______________________________________________________________________________
Double_t TProfile3D::GetBinEffectiveEntries(Int_t bin)
{
//            Return bin effective entries for a weighted filled Profile histogram. 
//            In case of an unweighted profile, it is equivalent to the number of entries per bin   
//            The effective entries is defined as the square of the sum of the weights divided by the 
//            sum of the weights square. 
//            TProfile::Sumw2() must be called before filling the profile with weights. 
//            Only by calling this method the  sum of the square of the weights per bin is stored. 
//  
//*-*          =========================================

   return TProfileHelper::GetBinEffectiveEntries((TProfile3D*)this, bin);
}

//______________________________________________________________________________
Double_t TProfile3D::GetBinError(Int_t bin) const
{
// *-*-*-*-*-*-*Return bin error of a Profile3D histogram*-*-*-*-*-*-*-*-*
//
// Computing errors: A moving field
// =================================
// The computation of errors for a TProfile3D has evolved with the versions
// of ROOT. The difficulty is in computing errors for bins with low statistics.
// - prior to version 3.10, we had no special treatment of low statistic bins.
//   As a result, these bins had huge errors. The reason is that the
//   expression eprim2 is very close to 0 (rounding problems) or 0.
// - The algorithm is modified/protected for the case
//   when a TProfile3D is projected (ProjectionX). The previous algorithm
//   generated a N^2 problem when projecting a TProfile3D with a large number of
//   bins (eg 100000).
// - in version 3.10/02, a new static function TProfile::Approximate
//   is introduced to enable or disable (default) the approximation.
//   (see also comments in TProfile::GetBinError)

   return TProfileHelper::GetBinError((TProfile3D*)this, bin);
}

//______________________________________________________________________________
Option_t *TProfile3D::GetErrorOption() const
{
//*-*-*-*-*-*-*-*-*-*Return option to compute profile2D errors*-*-*-*-*-*-*-*
//*-*                =========================================

   if (fErrorMode == kERRORSPREAD)  return "s";
   if (fErrorMode == kERRORSPREADI) return "i";
   if (fErrorMode == kERRORSPREADG) return "g";
   return "";
}

//______________________________________________________________________________
void TProfile3D::GetStats(Double_t *stats) const
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
   // stats[9] = sumwxz
   // stats[10]= sumwyz
   // stats[11]= sumwt
   // stats[12]= sumwt2
   //
   // If no axis-subrange is specified (via TAxis::SetRange), the array stats
   // is simply a copy of the statistics quantities computed at filling time.
   // If a sub-range is specified, the function recomputes these quantities
   // from the bin contents in the current axis range.

   if (fBuffer) ((TProfile3D*)this)->BufferEmpty();

   // Loop on bins
   if (fTsumw == 0 || fXaxis.TestBit(TAxis::kAxisRange) || fYaxis.TestBit(TAxis::kAxisRange)) {
      Int_t bin, binx, biny,binz;
      Double_t w, w2;
      Double_t x,y,z;
      for (bin=0;bin<kNstat;bin++) stats[bin] = 0;
      if (!fBinEntries.fArray) return;
      for (binz=fZaxis.GetFirst();binz<=fZaxis.GetLast();binz++) {
         z = fZaxis.GetBinCenter(binz);
         for (biny=fYaxis.GetFirst();biny<=fYaxis.GetLast();biny++) {
            y = fYaxis.GetBinCenter(biny);
            for (binx=fXaxis.GetFirst();binx<=fXaxis.GetLast();binx++) {
               bin = GetBin(binx,biny,binz);
               w         = fBinEntries.fArray[bin];
               w2        = (fBinSumw2.fN ? fBinSumw2.fArray[bin] : w );
               x         = fXaxis.GetBinCenter(binx);
               stats[0]  += w;
               stats[1]  += w2;
               stats[2]  += w*x;
               stats[3]  += w*x*x;
               stats[4]  += w*y;
               stats[5]  += w*y*y;
               stats[6]  += w*x*y;
               stats[7]  += w*z;
               stats[8]  += w*z*z;
               stats[9]  += w*x*z;
               stats[10] += w*y*z;
               stats[11] += fArray[bin];
               stats[12] += fSumw2.fArray[bin];
            }
         }
      }
   } else {
      stats[0]  = fTsumw;
      stats[1]  = fTsumw2;
      stats[2]  = fTsumwx;
      stats[3]  = fTsumwx2;
      stats[4]  = fTsumwy;
      stats[5]  = fTsumwy2;
      stats[6]  = fTsumwxy;
      stats[7]  = fTsumwz;
      stats[8]  = fTsumwz2;
      stats[9]  = fTsumwxz;
      stats[10] = fTsumwyz;
      stats[11] = fTsumwt;
      stats[12] = fTsumwt2;
   }
}

//______________________________________________________________________________
Long64_t TProfile3D::Merge(TCollection *li)
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

//    if (!li) return 0;
//    if (li->IsEmpty()) return (Int_t) GetEntries();

//    TList inlist;
//    TH1* hclone = (TH1*)Clone("FirstClone");
//    R__ASSERT(hclone);
//    BufferEmpty(1);         // To remove buffer.
//    Reset();                // BufferEmpty sets limits so we can't use it later.
//    SetEntries(0);
//    inlist.Add(hclone);
//    inlist.AddAll(li);

//    TAxis newXAxis;
//    TAxis newYAxis;
//    TAxis newZAxis;
//    Bool_t initialLimitsFound = kFALSE;
//    Bool_t same = kTRUE;
//    Bool_t allHaveLimits = kTRUE;

//    TIter next(&inlist);
//    while (TObject *o = next()) {
//       TProfile3D* h = dynamic_cast<TProfile3D*> (o);
//       if (!h) {
//          Error("Add","Attempt to add object of class: %s to a %s",
//                o->ClassName(),this->ClassName());
//          return -1;
//       }
//       Bool_t hasLimits = h->GetXaxis()->GetXmin() < h->GetXaxis()->GetXmax();
//       allHaveLimits = allHaveLimits && hasLimits;

//       if (hasLimits) {
//          h->BufferEmpty();
//          if (!initialLimitsFound) {
//             initialLimitsFound = kTRUE;
//             newXAxis.Set(h->GetXaxis()->GetNbins(), h->GetXaxis()->GetXmin(),
//                      h->GetXaxis()->GetXmax());
//             newYAxis.Set(h->GetYaxis()->GetNbins(), h->GetYaxis()->GetXmin(),
//                      h->GetYaxis()->GetXmax());
//             newZAxis.Set(h->GetZaxis()->GetNbins(), h->GetZaxis()->GetXmin(),
//                      h->GetZaxis()->GetXmax());
//          }
//          else {
//             if (!RecomputeAxisLimits(newXAxis, *(h->GetXaxis()))) {
//                Error("Merge", "Cannot merge histograms - limits are inconsistent:\n "
//                      "first: (%d, %f, %f), second: (%d, %f, %f)",
//                      newXAxis.GetNbins(), newXAxis.GetXmin(), newXAxis.GetXmax(),
//                      h->GetXaxis()->GetNbins(), h->GetXaxis()->GetXmin(),
//                      h->GetXaxis()->GetXmax());
//             }
//             if (!RecomputeAxisLimits(newYAxis, *(h->GetYaxis()))) {
//                Error("Merge", "Cannot merge histograms - limits are inconsistent:\n "
//                      "first: (%d, %f, %f), second: (%d, %f, %f)",
//                      newYAxis.GetNbins(), newYAxis.GetXmin(), newYAxis.GetXmax(),
//                      h->GetYaxis()->GetNbins(), h->GetYaxis()->GetXmin(),
//                      h->GetYaxis()->GetXmax());
//             }
//             if (!RecomputeAxisLimits(newZAxis, *(h->GetZaxis()))) {
//                Error("Merge", "Cannot merge histograms - limits are inconsistent:\n "
//                      "first: (%d, %f, %f), second: (%d, %f, %f)",
//                      newZAxis.GetNbins(), newZAxis.GetXmin(), newZAxis.GetXmax(),
//                      h->GetZaxis()->GetNbins(), h->GetZaxis()->GetXmin(),
//                      h->GetZaxis()->GetXmax());
//             }
//          }
//       }
//    }
//    next.Reset();

//    same = same && SameLimitsAndNBins(newXAxis, *GetXaxis())
//                && SameLimitsAndNBins(newYAxis, *GetYaxis())
//                && SameLimitsAndNBins(newZAxis, *GetZaxis());
//    if (!same && initialLimitsFound)
//       SetBins(newXAxis.GetNbins(), newXAxis.GetXmin(), newXAxis.GetXmax(),
//               newYAxis.GetNbins(), newYAxis.GetXmin(), newYAxis.GetXmax(),
//               newZAxis.GetNbins(), newZAxis.GetXmin(), newZAxis.GetXmax());

//    if (!allHaveLimits) {
//       // fill this histogram with all the data from buffers of histograms without limits
//       while (TProfile3D* h = dynamic_cast<TProfile3D*> (next())) {
//          if (h->GetXaxis()->GetXmin() >= h->GetXaxis()->GetXmax() && h->fBuffer) {
//              // no limits
//             Int_t nbentries = (Int_t)h->fBuffer[0];
//             for (Int_t i = 0; i < nbentries; i++)
//                Fill(h->fBuffer[5*i + 2], h->fBuffer[5*i + 3],
//                     h->fBuffer[5*i + 4], h->fBuffer[5*i + 5], h->fBuffer[5*i + 1]);
//          }
//       }
//       if (!initialLimitsFound)
//          return (Int_t) GetEntries();  // all histograms have been processed
//       next.Reset();
//    }

//    //merge bin contents and errors
//    Double_t stats[kNstat], totstats[kNstat];
//    for (Int_t i=0;i<kNstat;i++) {totstats[i] = stats[i] = 0;}
//    GetStats(totstats);
//    Double_t nentries = GetEntries();
//    Int_t binx, biny, binz, ix, iy, iz, nx, ny, nz, bin, ibin;
//    Int_t nbix = fXaxis.GetNbins();
//    Int_t nbiy = fYaxis.GetNbins();
//    Bool_t canRebin=TestBit(kCanRebin);
//    ResetBit(kCanRebin); // reset, otherwise setting the under/overflow will rebin

//    while (TProfile3D* h=(TProfile3D*)next()) {
//       // process only if the histogram has limits; otherwise it was processed before
//       if (h->GetXaxis()->GetXmin() < h->GetXaxis()->GetXmax()) {
//          // import statistics
//          h->GetStats(stats);
//          for (Int_t i = 0; i < kNstat; i++)
//             totstats[i] += stats[i];
//          nentries += h->GetEntries();

//          nx = h->GetXaxis()->GetNbins();
//          ny = h->GetYaxis()->GetNbins();
//          nz = h->GetZaxis()->GetNbins();

//          for (binz = 0; binz <= nz + 1; binz++) {
//             iz = fZaxis.FindBin(h->GetZaxis()->GetBinCenter(binz));
//             for (biny = 0; biny <= ny + 1; biny++) {
//                iy = fYaxis.FindBin(h->GetYaxis()->GetBinCenter(biny));
//                for (binx = 0; binx <= nx + 1; binx++) {
//                   ix = fXaxis.FindBin(h->GetXaxis()->GetBinCenter(binx));
//                   bin  = binx +(nx+2)*(biny + (ny+2)*binz);
//                   ibin = ix   +(nbix+2)*(iy + (nbiy+2)*iz);
//                   if ((!same) && (binx == 0 || binx == nx + 1
//                                || biny == 0 || biny == ny + 1 
//                                || binz == 0 || binz == nz + 1)) {
//                      if (h->GetW()[bin] != 0) {
//                         Error("Merge", "Cannot merge histograms - the histograms have"
//                                        " different limits and undeflows/overflows are present."
//                                        " The initial histogram is now broken!");
//                         return -1;
//                      }
//                   }
//                   fArray[ibin]             += h->GetW()[bin];
//                   fSumw2.fArray[ibin]      += h->GetW2()[bin];
//                   fBinEntries.fArray[ibin] += h->GetB()[bin];
//                }
//             }
//          }
//          fEntries += h->GetEntries();
//          fTsumw   += h->fTsumw;
//          fTsumw2  += h->fTsumw2;
//          fTsumwx  += h->fTsumwx;
//          fTsumwx2 += h->fTsumwx2;
//          fTsumwy  += h->fTsumwy;
//          fTsumwy2 += h->fTsumwy2;
//          fTsumwxy += h->fTsumwxy;
//          fTsumwz  += h->fTsumwz;
//          fTsumwz2 += h->fTsumwz2;
//          fTsumwxz += h->fTsumwxz;
//          fTsumwyz += h->fTsumwyz;
//          fTsumwt  += h->fTsumwt;
//          fTsumwt2 += h->fTsumwt2;
//       }
//    }
//    if (canRebin) SetBit(kCanRebin);

//    //copy merged stats
//    PutStats(totstats);
//    SetEntries(nentries);
//    inlist.Remove(hclone);
//    delete hclone;
//    return (Long64_t)nentries;
}

//______________________________________________________________________________
void TProfile3D::Multiply(TF1 *, Double_t )
{
   // Performs the operation: this = this*c1*f1

   Error("Multiply","Function not implemented for TProfile3D");
   return;
}

//______________________________________________________________________________
void TProfile3D::Multiply(const TH1 *)
{
//*-*-*-*-*-*-*-*-*-*-*Multiply this profile2D by h1*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =============================
//
//   this = this*h1
//
   Error("Multiply","Multiplication of profile2D histograms not implemented");
}


//______________________________________________________________________________
void TProfile3D::Multiply(const TH1 *, const TH1 *, Double_t, Double_t, Option_t *)
{
//*-*-*-*-*Replace contents of this profile2D by multiplication of h1 by h2*-*
//*-*      ================================================================
//
//   this = (c1*h1)*(c2*h2)
//

   Error("Multiply","Multiplication of profile2D histograms not implemented");
}

//______________________________________________________________________________
TH3D *TProfile3D::ProjectionXYZ(const char *name, Option_t *option) const
{
//*-*-*-*-*Project this profile3D into a 3-D histogram along X,Y,Z*-*-*-*-*-*-*
//*-*      =====================================================
//
//   The projection is always of the type TH3D.
//
//   if option "E" is specified, the errors are computed. (default)
//   if option "B" is specified, the content of bin of the returned histogram
//      will be equal to the GetBinEntries(bin) of the profile,
//   if option "C=E" the bin contents of the projection are set to the
//       bin errors of the profile
//

   TString opt = option;
   opt.ToLower();
   Int_t nx = fXaxis.GetNbins();
   Int_t ny = fYaxis.GetNbins();
   Int_t nz = fZaxis.GetNbins();

   // Create the projection histogram
   TString pname = name; 
   if (pname == "_px") { 
      pname = GetName();  pname.Append("_px");
   }
   TH3D *h1 = new TH3D(pname,GetTitle(),nx,fXaxis.GetXmin(),fXaxis.GetXmax(),ny,fYaxis.GetXmin(),fYaxis.GetXmax(),nz,fZaxis.GetXmin(),fZaxis.GetXmax());
   Bool_t computeErrors = kFALSE;
   Bool_t cequalErrors  = kFALSE;
   Bool_t binEntries    = kFALSE;
   if (opt.Contains("b")) binEntries = kTRUE;
   if (opt.Contains("e")) computeErrors = kTRUE;
   if (opt.Contains("c=e")) {cequalErrors = kTRUE; computeErrors=kFALSE;}
   if (computeErrors) h1->Sumw2();

   // Fill the projected histogram
   Int_t bin,binx,biny,binz;
   Double_t cont,err;
   for (binx =0;binx<=nx+1;binx++) {
      for (biny =0;biny<=ny+1;biny++) {
         for (binz =0;binz<=nz+1;binz++) {
            bin = GetBin(binx,biny,binz);
            if (binEntries)    cont = GetBinEntries(bin);
            else               cont = GetBinContent(bin);
            err   = GetBinError(bin);
            if (cequalErrors)  h1->SetBinContent(binx,biny,binz, err);
            else               h1->SetBinContent(binx,biny,binz,cont);
            if (computeErrors) h1->SetBinError(binx,biny,binz,err);
         }
      }
   }
   h1->SetEntries(fEntries);
   return h1;
}

//______________________________________________________________________________
void TProfile3D::PutStats(Double_t *stats)
{
   // Replace current statistics with the values in array stats

   TH3::PutStats(stats);
   fTsumwt  = stats[11];
   fTsumwt2 = stats[12];
}

//______________________________________________________________________________
void TProfile3D::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Reset contents of a Profile3D histogram*-*-*-*-*-*-*-*
//*-*                =======================================
   TH3D::Reset(option);
   fBinSumw2.Reset();
   fBinEntries.Reset();
   TString opt = option;
   opt.ToUpper();
   if (opt.Contains("ICE")) return;
   fTsumwt = fTsumwt2 = 0;
}

//______________________________________________________________________________
void TProfile3D::RebinAxis(Double_t x, TAxis *axis)
{
// Profile histogram is resized along axis such that x is in the axis range.
// The new axis limits are recomputed by doubling iteratively
// the current axis range until the specified value x is within the limits.
// The algorithm makes a copy of the histogram, then loops on all bins
// of the old histogram to fill the rebinned histogram.
// Takes into account errors (Sumw2) if any.
// The bit kCanRebin must be set before invoking this function.
//  Ex:  h->SetBit(TH1::kCanRebin);

   TProfile3D* hold = TProfileHelper::RebinAxis(this, x, axis);
   if ( hold ) {
      fTsumwt  = hold->fTsumwt;
      fTsumwt2 = hold->fTsumwt2;
      delete hold;
   }
}

//______________________________________________________________________________
void TProfile3D::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
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
   out << "," << GetZaxis()->GetNbins();
   out << "," << GetZaxis()->GetXmin()
       << "," << GetZaxis()->GetXmax();
   out << "," << fTmin
       << "," << fTmax;
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
void TProfile3D::Scale(Double_t c1, Option_t *option)
{
// *-*-*-*-*Multiply this profile2D by a constant c1*-*-*-*-*-*-*-*-*
// *-*      ========================================
//
//   this = c1*this
//
// This function uses the services of TProfile3D::Add
//

   TProfileHelper::Scale(this, c1, option);
}

//______________________________________________________________________________
void TProfile3D::SetBinEntries(Int_t bin, Double_t w)
{
//*-*-*-*-*-*-*-*-*Set the number of entries in bin*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ================================

   if (bin < 0 || bin >= fNcells) return;
   fBinEntries.fArray[bin] = w;
}

//______________________________________________________________________________
void TProfile3D::SetBins(Int_t nx, Double_t xmin, Double_t xmax, Int_t ny, Double_t ymin, Double_t ymax, Int_t nz, Double_t zmin, Double_t zmax)
{
//*-*-*-*-*-*-*-*-*Redefine  x axis parameters*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ===========================

   fXaxis.Set(nx,xmin,xmax);
   fYaxis.Set(ny,ymin,ymax);
   fZaxis.Set(ny,zmin,zmax);
   fNcells = (nx+2)*(ny+2)*(nz+2);
   fBinEntries.Set(fNcells);
   fSumw2.Set(fNcells);
   if (fBinSumw2.fN) fBinSumw2.Set(fNcells);
}


//______________________________________________________________________________
void TProfile3D::SetBuffer(Int_t buffersize, Option_t *)
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
   fBufferSize = 1 + 5*buffersize; 
   fBuffer = new Double_t[fBufferSize];
   memset(fBuffer,0,8*fBufferSize);
}

//______________________________________________________________________________
void TProfile3D::SetErrorOption(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Set option to compute profile2D errors*-*-*-*-*-*-*-*
//*-*                =======================================
//
//    The computation of errors is based on the parameter option:
//    option:
//     ' '  (Default) Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  SQRT(T)/SQRT(N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     's'            Errors are Spread  for Spread.ne.0. ,
//                      "     "  SQRT(T)  for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//     'i'            Errors are Spread/SQRT(N) for Spread.ne.0. ,
//                      "     "  1./SQRT(12.*N) for Spread.eq.0,N.gt.0 ,
//                      "     "  0.  for N.eq.0
//   See TProfile3D::BuildOptions for explanation of all options

   TString opt = option;
   opt.ToLower();
   fErrorMode = kERRORMEAN;
   if (opt.Contains("s")) fErrorMode = kERRORSPREAD;
   if (opt.Contains("i")) fErrorMode = kERRORSPREADI;
   if (opt.Contains("g")) fErrorMode = kERRORSPREADG;
}

//______________________________________________________________________________
void TProfile3D::Sumw2()
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
