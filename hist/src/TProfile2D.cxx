// @(#)root/hist:$Name:  $:$Id: TProfile2D.cxx,v 1.2 2000/06/13 10:36:47 brun Exp $
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
}

//______________________________________________________________________________
TProfile2D::~TProfile2D()
{
//*-*-*-*-*-*Default destructor for Profile2D histograms*-*-*-*-*-*-*-*-*
//*-*        ===========================================

}

//______________________________________________________________________________
TProfile2D::TProfile2D(const char *name,const char *title,Int_t nx,Axis_t xlow,Axis_t xup,Int_t ny,Axis_t ylow,Axis_t yup,Option_t *option)
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

   BuildOptions(0,0,option);
}

//______________________________________________________________________________
TProfile2D::TProfile2D(const char *name,const char *title,Int_t nx,Axis_t xlow,Axis_t xup,Int_t ny, Axis_t ylow,Axis_t yup,Axis_t zlow,Axis_t zup,Option_t *option)
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

   Sumw2();                   //*-* create sum of squares of weights array

   fZmin = zmin;
   fZmax = zmax;
}

//______________________________________________________________________________
TProfile2D::TProfile2D(const TProfile2D &profile)
{
   ((TProfile2D&)profile).Copy(*this);
}


//______________________________________________________________________________
void TProfile2D::Add(TF1 *, Double_t )
{
   // Performs the operation: this = this + c1*f1

   Error("Add","Function not implemented for TProfile2D");
   return;
}


//______________________________________________________________________________
void TProfile2D::Add(TH1 *h1, Double_t c1)
{
   // Performs the operation: this = this + c1*h1

   if (!h1) {
      Error("Add","Attempt to add a non-existing profile");
      return;
   }
   if (!h1->InheritsFrom("TProfile2D")) {
      Error("Add","Attempt to add a non-profile2D object");
      return;
   }
   TProfile2D *p1 = (TProfile2D*)h1;

//*-*- Check profile compatibility
   Int_t nx = GetNbinsX();
   if (nx != p1->GetNbinsX()) {
      Error("Add","Attempt to add profiles with different number of bins");
      return;
   }
   Int_t ny = GetNbinsY();
   if (ny != p1->GetNbinsY()) {
      Error("Add","Attempt to add profiles with different number of bins");
      return;
   }

//*-*- Add statistics
   Double_t ac1 = TMath::Abs(c1);
   fEntries += ac1*p1->GetEntries();
   fTsumw   += ac1*p1->fTsumw;
   fTsumw2  += ac1*p1->fTsumw2;
   fTsumwx  += ac1*p1->fTsumwx;
   fTsumwx2 += ac1*p1->fTsumwx2;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin,binx,biny;
   Double_t *cu1 = p1->GetW();
   Double_t *er1 = p1->GetW2();
   Double_t *en1 = p1->GetB();
   for (binx =0;binx<=nx+1;binx++) {
      for (biny =0;biny<=ny+1;biny++) {
         bin   = biny*(fXaxis.GetNbins()+2) + binx;
         fArray[bin]             +=  c1*cu1[bin];
         fSumw2.fArray[bin]      += ac1*er1[bin];
         fBinEntries.fArray[bin] += ac1*en1[bin];
      }
   }
}

//______________________________________________________________________________
void TProfile2D::Add(TH1 *h1, TH1 *h2, Double_t c1, Double_t c2)
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
   if (!h1->InheritsFrom("TProfile2D")) {
      Error("Add","Attempt to add a non-profile2D object");
      return;
   }
   TProfile2D *p1 = (TProfile2D*)h1;
   if (!h2->InheritsFrom("TProfile2D")) {
      Error("Add","Attempt to add a non-profile2D object");
      return;
   }
   TProfile2D *p2 = (TProfile2D*)h2;

//*-*- Check profile compatibility
   Int_t nx = GetNbinsX();
   if (nx != p1->GetNbinsX() || nx != p2->GetNbinsX()) {
      Error("Add","Attempt to add profiles with different number of bins");
      return;
   }
   Int_t ny = GetNbinsY();
   if (ny != p1->GetNbinsY() || ny != p2->GetNbinsY()) {
      Error("Add","Attempt to add profiles with different number of bins");
      return;
   }

//*-*- Add statistics
   Double_t ac1 = TMath::Abs(c1);
   Double_t ac2 = TMath::Abs(c2);
   fEntries = ac1*p1->GetEntries() + ac2*p2->GetEntries();
   fTsumw   = ac1*p1->fTsumw       + ac2*p2->fTsumw;
   fTsumw2  = ac1*p1->fTsumw2      + ac2*p2->fTsumw2;
   fTsumwx  = ac1*p1->fTsumwx      + ac2*p2->fTsumwx;
   fTsumwx2 = ac1*p1->fTsumwx2     + ac2*p2->fTsumwx2;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin,binx,biny;
   Double_t *cu1 = p1->GetW();
   Double_t *cu2 = p2->GetW();
   Double_t *er1 = p1->GetW2();
   Double_t *er2 = p2->GetW2();
   Double_t *en1 = p1->GetB();
   Double_t *en2 = p2->GetB();
   for (binx =0;binx<=nx+1;binx++) {
      for (biny =0;biny<=ny+1;biny++) {
         bin   = biny*(fXaxis.GetNbins()+2) + binx;
         fArray[bin]             =  c1*cu1[bin] +  c2*cu2[bin];
         fSumw2.fArray[bin]      = ac1*er1[bin] + ac2*er2[bin];
         fBinEntries.fArray[bin] = ac1*en1[bin] + ac2*en2[bin];
      }
   }
}

//______________________________________________________________________________
void TProfile2D::Copy(TObject &obj)
{
//*-*-*-*-*-*-*-*Copy a Profile2D histogram to a new profile2D histogram*-*-*-*
//*-*            =======================================================

   TH2D::Copy(((TProfile2D&)obj));
   fBinEntries.Copy(((TProfile2D&)obj).fBinEntries);
   ((TProfile2D&)obj).fZmin = fZmin;
   ((TProfile2D&)obj).fZmax = fZmax;
   ((TProfile2D&)obj).fErrorMode = fErrorMode;
}


//______________________________________________________________________________
void TProfile2D::Divide(TF1 *, Double_t )
{
   // Performs the operation: this = this/(c1*f1)

   Error("Divide","Function not implemented for TProfile2D");
   return;
}

//______________________________________________________________________________
void TProfile2D::Divide(TH1 *h1)
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
   if (!h1->InheritsFrom("TProfile2D")) {
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
   Double_t c0,c1,w,z,x;
   for (binx =0;binx<=nx+1;binx++) {
      for (biny =0;biny<=ny+1;biny++) {
         bin   = biny*(fXaxis.GetNbins()+2) + binx;
         c0  = fArray[bin];
         c1  = cu1[bin];
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
         Double_t e0 = fSumw2.fArray[bin];
         Double_t e1 = er1[bin];
         Double_t c12= c1*c1;
         if (!c1) fSumw2.fArray[bin] = 0;
         else     fSumw2.fArray[bin] = (e0*e0*c1*c1 + e1*e1*c0*c0)/(c12*c12);
         if (!en1[bin]) fBinEntries.fArray[bin] = 0;
         else           fBinEntries.fArray[bin] /= en1[bin];
      }
   }
}


//______________________________________________________________________________
void TProfile2D::Divide(TH1 *h1, TH1 *h2, Double_t c1, Double_t c2, Option_t *option)
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
   if (!h1->InheritsFrom("TProfile2D")) {
      Error("Divide","Attempt to divide a non-profile2D object");
      return;
   }
   TProfile2D *p1 = (TProfile2D*)h1;
   if (!h2->InheritsFrom("TProfile2D")) {
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
   Double_t b1,b2,w,z,x,d1,d2;
   d1 = c1*c1;
   d2 = c2*c2;
   for (binx =0;binx<=nx+1;binx++) {
      for (biny =0;biny<=ny+1;biny++) {
         bin   = biny*(fXaxis.GetNbins()+2) + binx;
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
         Double_t e1 = er1[bin];
         Double_t e2 = er2[bin];
         Double_t b22= b2*b2*d2;
         if (!b2) fSumw2.fArray[bin] = 0;
         else {
            if (binomial) {
               fSumw2.fArray[bin] = TMath::Abs(w*(1-w)/(c2*b2));
            } else {
               fSumw2.fArray[bin] = d1*d2*(e1*e1*b2*b2 + e2*e2*b1*b1)/(b22*b22);
            }
         }
         if (!en2[bin]) fBinEntries.fArray[bin] = 0;
         else           fBinEntries.fArray[bin] = en1[bin]/en2[bin];
      }
   }
}

//______________________________________________________________________________
TH1 *TProfile2D::DrawCopy(Option_t *option)
{
//*-*-*-*-*-*-*-*Draw a copy of this profile2D histogram*-*-*-*-*-*-*-*-*-*-*
//*-*            =======================================
   TProfile2D *newpf = new TProfile2D();
   Copy(*newpf);
   newpf->SetDirectory(0);
   newpf->SetBit(kCanDelete);
   newpf->AppendPad(option);
   return newpf;
}

//______________________________________________________________________________
Int_t TProfile2D::Fill(Axis_t x, Axis_t y, Axis_t z)
{
//*-*-*-*-*-*-*-*-*-*-*Fill a Profile2D histogram (no weights)*-*-*-*-*-*-*-*
//*-*                  =======================================
   Int_t bin,binx,biny;

   if (fZmin != fZmax) {
      if (z <fZmin || z> fZmax) return -1;
   }

   fEntries++;
   binx =fXaxis.FindBin(x);
   biny =fYaxis.FindBin(y);
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin, z);
   fSumw2.fArray[bin] += (Stat_t)z*z;
   fBinEntries.fArray[bin] += 1;
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
Int_t TProfile2D::Fill(Axis_t x, Axis_t y, Axis_t z, Stat_t w)
{
//*-*-*-*-*-*-*-*-*-*-*Fill a Profile2D histogram with weights*-*-*-*-*-*-*-*
//*-*                  =======================================
   Int_t bin,binx,biny;

   if (fZmin != fZmax) {
      if (z <fZmin || z> fZmax) return -1;
   }

   Stat_t u= (w > 0 ? w : -w);
   fEntries++;
   binx =fXaxis.FindBin(x);
   biny =fYaxis.FindBin(y);
   bin  = biny*(fXaxis.GetNbins()+2) + binx;
   AddBinContent(bin, u*z);
   fSumw2.fArray[bin] += u*z*z;
   fBinEntries.fArray[bin] += w;
   if (binx == 0 || binx > fXaxis.GetNbins()) return -1;
   if (biny == 0 || biny > fYaxis.GetNbins()) return -1;
   fTsumw   += u;
   fTsumw2  += u*u;
   fTsumwx  += u*x;
   fTsumwx2 += u*x*x;
   fTsumwy  += u*y;
   fTsumwy2 += u*y*y;
   fTsumwxy += u*x*y;
   return bin;
}

//______________________________________________________________________________
Stat_t TProfile2D::GetBinContent(Int_t bin)
{
//*-*-*-*-*-*-*Return bin content of a Profile2D histogram*-*-*-*-*-*-*-*-*
//*-*          ===========================================

   if (bin < 0 || bin >= fNcells) return 0;
   if (fBinEntries.fArray[bin] == 0) return 0;
   return fArray[bin]/fBinEntries.fArray[bin];
}

//______________________________________________________________________________
Stat_t TProfile2D::GetBinEntries(Int_t bin)
{
//*-*-*-*-*-*-*Return bin entries of a Profile2D histogram*-*-*-*-*-*-*-*-*
//*-*          ===========================================

   if (bin < 0 || bin >= fNcells) return 0;
   return fBinEntries.fArray[bin];
}

//______________________________________________________________________________
Stat_t TProfile2D::GetBinError(Int_t bin)
{
//*-*-*-*-*-*-*Return bin error of a Profile2D histogram*-*-*-*-*-*-*-*-*
//*-*          =========================================

   if (bin < 0 || bin >= fNcells) return 0;
   Stat_t cont = fArray[bin];
   Stat_t sum  = fBinEntries.fArray[bin];
   Stat_t err2 = fSumw2.fArray[bin];
   if (sum == 0) return 0;
   Stat_t eprim;
   Stat_t contsum = cont/sum;
   Stat_t eprim2  = TMath::Abs(err2/sum - contsum*contsum);
   eprim          = TMath::Sqrt(eprim2);
   if (eprim <= 0) {
      Stat_t scont, ssum, serr2;
      scont = ssum = serr2 = 0;
      for (Int_t i=1;i<fNcells;i++) {
         scont += fArray[i];
         ssum  += fBinEntries.fArray[i];
         serr2 += fSumw2.fArray[i];
      }
      Stat_t scontsum = scont/ssum;
      Stat_t seprim2  = TMath::Abs(serr2/ssum - scontsum*scontsum);
      eprim           = TMath::Sqrt(seprim2);
   }
   if (fErrorMode == kERRORMEAN) return eprim/TMath::Sqrt(sum);
   else if (fErrorMode == kERRORSPREAD) return eprim;
   else return eprim;
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
void TProfile2D::Multiply(TF1 *, Double_t )
{
   // Performs the operation: this = this*c1*f1

   Error("Multiply","Function not implemented for TProfile2D");
   return;
}

//______________________________________________________________________________
void TProfile2D::Multiply(TH1 *)
{
//*-*-*-*-*-*-*-*-*-*-*Multiply this profile2D by h1*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =============================
//
//   this = this*h1
//
   Error("Multiply","Multiplication of profile2D histograms not implemented");
}


//______________________________________________________________________________
void TProfile2D::Multiply(TH1 *, TH1 *, Double_t, Double_t, Option_t *)
{
//*-*-*-*-*Replace contents of this profile2D by multiplication of h1 by h2*-*
//*-*      ================================================================
//
//   this = (c1*h1)*(c2*h2)
//

   Error("Multiply","Multiplication of profile2D histograms not implemented");
}

//______________________________________________________________________________
TH2D *TProfile2D::ProjectionXY(const char *name, Option_t *option)
{
//*-*-*-*-*Project this profile2D into a 2-D histogram along X,Y*-*-*-*-*-*-*
//*-*      =====================================================
//
//   The projection is always of the type TH2D.
//
//   if option "E" is specified, the errors are computed. (default)
//
//

  TString opt = option;
  opt.ToLower();
  Int_t nx = fXaxis.GetNbins();
  Int_t ny = fYaxis.GetNbins();

// Create the projection histogram
  char *pname = (char*)name;
  if (strcmp(name,"_px") == 0) {
     Int_t nch = strlen(GetName()) + 4;
     pname = new char[nch];
     sprintf(pname,"%s%s",GetName(),name);
  }
  TH2D *h1 = new TH2D(pname,GetTitle(),nx,fXaxis.GetXmin(),fXaxis.GetXmax(),ny,fYaxis.GetXmin(),fYaxis.GetXmax());
  Bool_t computeErrors = kFALSE;
  if (opt.Contains("e")) {h1->Sumw2(); computeErrors = kTRUE;}
  if (pname != name)  delete [] pname;

// Fill the projected histogram
  Int_t bin, binx, biny;
  Double_t cont,err;
  for (binx =0;binx<=nx+1;binx++) {
     for (biny =0;biny<=ny+1;biny++) {
        bin   = biny*(fXaxis.GetNbins()+2) + binx;
        cont  = GetBinContent(bin);
        err   = GetBinError(bin);
        if (cont)          h1->Fill(fXaxis.GetBinCenter(binx),fYaxis.GetBinCenter(biny), cont);
        if (computeErrors) h1->SetBinError(bin,err);
     }
  }
  h1->SetEntries(fEntries);
  return h1;
}

//______________________________________________________________________________
void TProfile2D::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Reset contents of a Profile2D histogram*-*-*-*-*-*-*-*
//*-*                =======================================
  TH2D::Reset(option);
  fBinEntries.Reset();
}

//______________________________________________________________________________
void TProfile2D::Scale(Double_t c1)
{
//*-*-*-*-*Multiply this profile2D by a constant c1*-*-*-*-*-*-*-*-*
//*-*      ========================================
//
//   this = c1*this
//
// This function uses the services of TProfile2D::Add
//

   Double_t ent = fEntries;
   Add(this,this,c1,0);
   fEntries = ent;
}

//______________________________________________________________________________
void TProfile2D::SetBinEntries(Int_t bin, Stat_t w)
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
   fBinEntries.Set(fNcells);
   fSumw2.Set(fNcells);
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

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      TH2D::Streamer(R__b);
      fBinEntries.Streamer(R__b);
      R__b >> (Int_t&)fErrorMode;
      if (R__v < 2) {
         Float_t zmin,zmax;
         R__b >> zmin; fZmin = zmin;
         R__b >> zmax; fZmax = zmax;
      } else {
         R__b >> fZmin;
         R__b >> fZmax;
      }
      R__b.CheckByteCount(R__s, R__c, TProfile2D::IsA());
   } else {
      R__c = R__b.WriteVersion(TProfile2D::IsA(), kTRUE);
      TH2D::Streamer(R__b);
      fBinEntries.Streamer(R__b);
      R__b << (Int_t)fErrorMode;
      R__b << fZmin;
      R__b << fZmax;
      R__b.SetByteCount(R__c, kTRUE);
   }
}
