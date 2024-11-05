// Author: Stefan Schmitt
// DESY, 23/01/09

//  Version 17.9, parallel to changes in TUnfold
//
//  History:
//    Version 17.8, parallel to changes in TUnfold
//    Version 17.7, bug fix in GetBackground()
//    Version 17.6, with updated doxygen comments
//    Version 17.5, in parallel to changes in TUnfold
//    Version 17.4, in parallel to changes in TUnfoldBinning
//    Version 17.3, in parallel to changes in TUnfoldBinning
//    Version 17.2, add methods to find back systematic and background sources
//    Version 17.1, bug fix with background uncertainty
//    Version 17.0, possibility to specify an error matrix with SetInput
//    Version 16.1, parallel to changes in TUnfold
//    Version 16.0, parallel to changes in TUnfold
//    Version 15, fix bugs with uncorr. uncertainties, add backgnd subtraction
//    Version 14, remove some print-out, do not add unused sys.errors
//    Version 13, support for systematic errors

/** \class TUnfoldSys
\ingroup Unfold
An algorithm to unfold distributions from detector to truth level,
with background subtraction and propagation of systematic uncertainties

TUnfoldSys is used to decompose a measurement y into several sources x,
given the measurement uncertainties, background b and a matrix of migrations A.
The method can be applied to a large number of problems,
where the measured distribution y is a linear superposition
of several Monte Carlo shapes. Beyond such a simple template fit,
TUnfoldSys has an adjustable regularisation term and also supports an
optional constraint on the total number of events.
Background sources can be specified, with a normalisation constant and
normalisation uncertainty. In addition, variants of the response
matrix may be specified, these are taken to determine systematic
uncertainties.

<b>For most applications, it is better to use the derived class
TUnfoldDensity instead of TUnfoldSys. TUnfoldDensity adds
features to TUnfoldSys, related to possible complex multidimensional
arrangements of bins. For innocent
users, the most notable improvement of TUnfoldDensity over TUnfoldSys are
the getter functions. For TUnfoldSys, histograms have to be booked by the
user and the getter functions fill the histogram bins. TUnfoldDensity
simply returns a new, already filled histogram.</b>

If you use this software, please consider the following citation

<b>S.Schmitt, JINST 7 (2012) T10003 [arXiv:1205.6201]</b>

Detailed documentation and updates are available on
http://www.desy.de/~sschmitt

Brief recipy to use TUnfoldSys:
<ul>
<li>a matrix (truth,reconstructed) is given as a two-dimensional histogram
  as argument to the constructor of TUnfold</li>
<li>a vector of measurements is given as one-dimensional histogram using
  the SetInput() method</li>
<li>repeated calls to SubtractBackground() to specify background
sources</li>
<li>repeated calls to AddSysError() to specify systematic uncertainties
<li>The unfolding is performed
<ul>
<li>either once with a fixed parameter tau, method DoUnfold(tau)</li>
<li>or multiple times in a scan to determine the best chouce of tau,
     method ScanLCurve()</li>
</ul>
<li>Unfolding results are retrieved using various GetXXX() methods
</ul>

Description of (systematic) uncertainties available in
TUnfoldSys. There are covariance matrix contributions and there are
systematic shifts. Systematic shifts correspond to the variation of a
(buicance) parameter, for example a background normalisation or a
one-sigma variation of a correlated systematic error.
<table>
<tr><th> </th><th>Set by</th>
<th>Access covariance matrix</th>
<th>Access vector of shifts</th>
<th>Description</th>
</tr>
<tr>
<td>(a)</td><td>TUnfoldSys constructor</td>
<td>GetEmatrixSysUncorr()</td><td> n.a. </td>
<td>uncorrelated errors on the input matrix histA, taken as the errors
provided with the histogram. These are typically statistical errors
from finite Monte Carlo samples.</td>
</tr>
<tr>
<td>(b)</td><td>AddSysError()</td><td>GetEmatrixSysSource()</td>
<td>GetDeltaSysSource()</td>
<td>correlated shifts of the input matrix histA. These shifts are taken
as one-sigma effects when switchig on a given error soure. Several
such error sources may be defined</td>
</tr>
<tr>
<td>(c)</td><td>SetTauError()</td><td>GetEmatrixSysTau()</td>
<td>GetDeltaSysTau()</td>
<td>A systematic error on the regularisation parameter tau</td>
</tr>
<tr><td>(d)</td><td>SubtractBackground()</td>
<td>GetEmatrixSysBackgroundUncorr()</td><td>n.a.</td>
<td>uncorrelated errors on background sources, originating from the errors
 provided with the background histograms</td>
</tr>
<tr>
<td>(e)</td><td>SubtractBackground</td>
<td>GetEmatrixSysBackgroundScale()</td><td>GetDeltaSysBackgroundScale()</td>
<td>scale errors on background sources</td>
</tr>
<tr>
<td>(i)</td><td>SetInput()</td>
<td>GetEmatrixInput()</td><td>n.a.</td><td>statistical uncertainty of
the input (the measurement)</td>
</tr>
<tr>
<td>(i)+(d)+(e)</td><td>see above</td><td>GetEmatrix()</td><td>n.a.</td>
<td>Partial sun of uncertainties: all sources which are propagated to the covariance before unfolding</td>
</tr>
<tr>
<td>
(i)+(a)+(b)+(c)+(d)+(e)</td><td>see above</td><td>GetEmatrixTotal()</td>
<td>n.a.</td><td>All known error sources summed up</td>
</tr>
</table>

Note:  (a), (b), (c) are propagated to the result AFTER unfolding,
whereas the background errors (d) and (e) are added to the data errors
BEFORE unfolding. For this reason the errors of type (d) and (e) are
INCLUDED in the standard error matrix and other methods provided by
the base class TUnfold, whereas errors of type (a), (b), (c) are NOT
INCLUDED in the methods provided by the base class TUnfold.
*/

/*
  This file is part of TUnfold.

  TUnfold is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  TUnfold is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with TUnfold.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <TMap.h>
#include <TMath.h>
#include <TObjString.h>
#include <TSortedList.h>
#include <RVersion.h>
#include <cmath>

#include "TUnfoldSys.h"

ClassImp(TUnfoldSys)

TUnfoldSys::~TUnfoldSys(void)
{
   // delete all data members
   DeleteMatrix(&fDAinRelSq);
   DeleteMatrix(&fDAinColRelSq);
   delete fBgrIn;
   delete fBgrErrUncorrInSq;
   delete fBgrErrScaleIn;
   delete fSysIn;
   ClearResults();
   delete fDeltaCorrX;
   delete fDeltaCorrAx;
   DeleteMatrix(&fYData);
   DeleteMatrix(&fVyyData);
}

////////////////////////////////////////////////////////////////////////
/// only for use by root streamer or derived classes
///
TUnfoldSys::TUnfoldSys(void)
{
   // set all pointers to zero
   InitTUnfoldSys();
}

////////////////////////////////////////////////////////////////////////
/// set up response matrix A, uncorrelated uncertainties of A and
/// regularisation scheme
///
/// \param[in] hist_A matrix that describes the migrations
/// \param[in] histmap mapping of the histogram axes to the unfolding output
/// \param[in] regmode (default=kRegModeSize) global regularisation mode
/// \param[in] constraint (default=kEConstraintArea) type of constraint
///
/// For further details, consult the constructir of the class TUnfold.
/// The uncertainties of hist_A are taken to be uncorrelated and aper
/// propagated to the unfolding result, method GetEmatrixSysUncorr().
TUnfoldSys::TUnfoldSys
(const TH2 *hist_A, EHistMap histmap, ERegMode regmode,EConstraint constraint)
   : TUnfold(hist_A,histmap,regmode,constraint)
{
   // data members initialized to something different from zero:
   //    fDA2, fDAcol

   // initialize TUnfoldSys
   InitTUnfoldSys();

   // save underflow and overflow bins
   fAoutside = new TMatrixD(GetNx(),2);
   // save the normalized errors on hist_A
   // to the matrices fDAinRelSq and fDAinColRelSq
   fDAinColRelSq = new TMatrixD(GetNx(),1);

   Int_t nmax=GetNx()*GetNy();
   Int_t *rowDAinRelSq = new Int_t[nmax];
   Int_t *colDAinRelSq = new Int_t[nmax];
   Double_t *dataDAinRelSq = new Double_t[nmax];

   Int_t da_nonzero=0;
   for (Int_t ix = 0; ix < GetNx(); ix++) {
      Int_t ibinx = fXToHist[ix];
      Double_t sum_sq= fSumOverY[ix]*fSumOverY[ix];
      for (Int_t ibiny = 0; ibiny <= GetNy()+1; ibiny++) {
         Double_t dz;
         if (histmap == kHistMapOutputHoriz) {
            dz = hist_A->GetBinError(ibinx, ibiny);
         } else {
            dz = hist_A->GetBinError(ibiny, ibinx);
         }
         Double_t normerr_sq=dz*dz/sum_sq;
         // quadratic sum of all errors from all bins,
         //   including under/overflow bins
         (*fDAinColRelSq)(ix,0) += normerr_sq;

         if(ibiny==0) {
            // underflow bin
            if (histmap == kHistMapOutputHoriz) {
               (*fAoutside)(ix,0)=hist_A->GetBinContent(ibinx, ibiny);
            } else {
               (*fAoutside)(ix,0)=hist_A->GetBinContent(ibiny, ibinx);
            }
         } else if(ibiny==GetNy()+1) {
            // overflow bins
            if (histmap == kHistMapOutputHoriz) {
               (*fAoutside)(ix,1)=hist_A->GetBinContent(ibinx, ibiny);
            } else {
               (*fAoutside)(ix,1)=hist_A->GetBinContent(ibiny, ibinx);
            }
         } else {
            // error on this bin
            rowDAinRelSq[da_nonzero]=ibiny-1;
            colDAinRelSq[da_nonzero] = ix;
            dataDAinRelSq[da_nonzero] = normerr_sq;
            if(dataDAinRelSq[da_nonzero]>0.0) da_nonzero++;
         }
      }
   }
   if(da_nonzero) {
      fDAinRelSq = CreateSparseMatrix(GetNy(),GetNx(),da_nonzero,
                                      rowDAinRelSq,colDAinRelSq,dataDAinRelSq);
   } else {
      DeleteMatrix(&fDAinColRelSq);
   }
   delete[] rowDAinRelSq;
   delete[] colDAinRelSq;
   delete[] dataDAinRelSq;
}

////////////////////////////////////////////////////////////////////////
/// Specify a correlated systematic uncertainty
///
/// \param[in] sysError alternative matrix or matrix of absolute/relative shifts
/// \param[in] name identifier of the error source
/// \param[in] histmap mapping of the histogram axes
/// \param[in] mode format of the error source
///
/// <b>sysError</b> corresponds to a one-sigma variation. If
/// may be given in various forms, specified by <b>mode</b>
/// <ul>
/// <li><b>mode=kSysErrModeMatrix</b> the histogram <b>sysError</b>
/// corresponds to an alternative response matrix.</li>
/// <li><b>mode=kSysErrModeShift</b> the content of the histogram <b>sysError</b> are the absolute shifts of the response matrix
/// <li><b>mode=kSysErrModeRelative</b> the content of the histogram <b>sysError</b>
/// specifies the relative uncertainties
/// </ul>
/// Internally, all three cases are transformed to the case <b>mode=kSysErrModeMatrix</b>.
void TUnfoldSys::AddSysError
(const TH2 *sysError,const char *name,EHistMap histmap,ESysErrMode mode)
{

   if(fSysIn->FindObject(name)) {
      Error("AddSysError","Source %s given twice, ignoring 2nd call.\n",name);
   } else {
      // a copy of fA is made. It can be accessed inside the loop
      // without having to take care that the sparse structure of *fA
      // otherwise, *fA may be accidentally destroyed by asking
      // for an element which is zero.
      TMatrixD aCopy(*fA);

      Int_t nmax= GetNx()*GetNy();
      Double_t *data=new Double_t[nmax];
      Int_t *cols=new Int_t[nmax];
      Int_t *rows=new Int_t[nmax];
      nmax=0;
      for (Int_t ix = 0; ix < GetNx(); ix++) {
         Int_t ibinx = fXToHist[ix];
         Double_t sum=0.0;
         for(Int_t loop=0;loop<2;loop++) {
            for (Int_t ibiny = 0; ibiny <= GetNy()+1; ibiny++) {
               Double_t z;
               // get bin content, depending on histmap
               if (histmap == kHistMapOutputHoriz) {
                  z = sysError->GetBinContent(ibinx, ibiny);
               } else {
                  z = sysError->GetBinContent(ibiny, ibinx);
               }
               // correct to absolute numbers
               if(mode!=kSysErrModeMatrix) {
                  Double_t z0;
                  if((ibiny>0)&&(ibiny<=GetNy())) {
                     z0=aCopy(ibiny-1,ix)*fSumOverY[ix];
                  } else if(ibiny==0) {
                     z0=(*fAoutside)(ix,0);
                  } else {
                     z0=(*fAoutside)(ix,1);
                  }
                  if(mode==kSysErrModeShift) {
                     z += z0;
                  } else if(mode==kSysErrModeRelative) {
                     z = z0*(1.+z);
                  }
               }
               if(loop==0) {
                  // sum up all entries, including overflow bins
                  sum += z;
               } else {
                  if((ibiny>0)&&(ibiny<=GetNy())) {
                     // save normalized matrix of shifts,
                     // excluding overflow bins
                     rows[nmax]=ibiny-1;
                     cols[nmax]=ix;
                     if(sum>0.0) {
                        data[nmax]=z/sum-aCopy(ibiny-1,ix);
                     } else {
                        data[nmax]=0.0;
                     }
                     if(data[nmax] != 0.0) nmax++;
                  }
               }
            }
         }
      }
      if(nmax==0) {
         Error("AddSysError",
               "source %s has no influence and has not been added.\n",name);
      } else {
         TMatrixDSparse *dsys=CreateSparseMatrix(GetNy(),GetNx(),
                                                 nmax,rows,cols,data);
         fSysIn->Add(new TObjString(name),dsys);
      }
      delete[] data;
      delete[] rows;
      delete[] cols;
   }
}

////////////////////////////////////////////////////////////////////////
/// perform background subtraction
///
/// This prepares the data members for the base class TUnfold, such
/// that the background is properly taken into account.
void TUnfoldSys::DoBackgroundSubtraction(void)
{
   // fY = fYData - fBgrIn
   // fVyy = fVyyData + fBgrErrUncorr^2 + fBgrErrCorr * fBgrErrCorr#
   // fVyyinv = fVyy^(-1)

   if(fYData) {
      DeleteMatrix(&fY);
      DeleteMatrix(&fVyy);
      if(fBgrIn->GetEntries()<=0) {
         // simple copy
         fY=new TMatrixD(*fYData);
         fVyy=new TMatrixDSparse(*fVyyData);
      } else {
         if(GetVyyInv()) {
            Warning("DoBackgroundSubtraction",
                    "inverse error matrix from user input,"
                    " not corrected for background");
         }
         // copy of the data
         fY=new TMatrixD(*fYData);
         // subtract background from fY
         const TObject *key;
         {
            TMapIter bgrPtr(fBgrIn);
            for(key=bgrPtr.Next();key;key=bgrPtr.Next()) {
               const TMatrixD *bgr=(const TMatrixD *)((const TPair *)*bgrPtr)->Value();
               for(Int_t i=0;i<GetNy();i++) {
                  (*fY)(i,0) -= (*bgr)(i,0);
               }
            }
         }
         // copy original error matrix
         TMatrixD vyy(*fVyyData);
         // determine used bins
         Int_t ny=fVyyData->GetNrows();
         const Int_t *vyydata_rows=fVyyData->GetRowIndexArray();
         const Int_t *vyydata_cols=fVyyData->GetColIndexArray();
         const Double_t *vyydata_data=fVyyData->GetMatrixArray();
         Int_t *usedBin=new Int_t[ny];
         for(Int_t i=0;i<ny;i++) {
            usedBin[i]=0;
         }
         for(Int_t i=0;i<ny;i++) {
            for(Int_t k=vyydata_rows[i];k<vyydata_rows[i+1];k++) {
               if(vyydata_data[k]>0.0) {
                  usedBin[i]++;
                  usedBin[vyydata_cols[k]]++;
               }
            }
         }
         // add uncorrelated background errors
         {
            TMapIter bgrErrUncorrSqPtr(fBgrErrUncorrInSq);
            for(key=bgrErrUncorrSqPtr.Next();key;
                key=bgrErrUncorrSqPtr.Next()) {
               const TMatrixD *bgrerruncorrSquared=(TMatrixD const *)((const TPair *)*bgrErrUncorrSqPtr)->Value();
               for(Int_t yi=0;yi<ny;yi++) {
                  if(!usedBin[yi]) continue;
                  vyy(yi,yi) +=(*bgrerruncorrSquared)(yi,0);
               }
            }
         }
         // add correlated background errors
         {
            TMapIter bgrErrScalePtr(fBgrErrScaleIn);
            for(key=bgrErrScalePtr.Next();key;key=bgrErrScalePtr.Next()) {
               const TMatrixD *bgrerrscale=(const TMatrixD *)((const TPair *)*bgrErrScalePtr)->Value();
               for(Int_t yi=0;yi<ny;yi++) {
                  if(!usedBin[yi]) continue;
                  for(Int_t yj=0;yj<ny;yj++) {
                     if(!usedBin[yj]) continue;
                     vyy(yi,yj) +=(*bgrerrscale)(yi,0)* (*bgrerrscale)(yj,0);
                  }
               }
            }
         }
         delete[] usedBin;
         usedBin=nullptr;

         // convert to sparse matrix
         fVyy=new TMatrixDSparse(vyy);

      }
   } else {
      Fatal("DoBackgroundSubtraction","No input vector defined");
   }
}

Int_t TUnfoldSys::SetInput(const TH1 *hist_y,Double_t scaleBias,
                              Double_t oneOverZeroError,const TH2 *hist_vyy,
                              const TH2 *hist_vyy_inv)
{
   // Define the input data for subsequent calls to DoUnfold(Double_t)
   //  input:   input distribution with errors
   //  scaleBias:  scale factor applied to the bias
   //  oneOverZeroError: for bins with zero error, this number defines 1/error.
   // Return value: number of bins with bad error
   //                 +10000*number of unconstrained output bins
   //         Note: return values>=10000 are fatal errors,
   //               for the given input, the unfolding can not be done!
   // Calls the SetInput method of the base class, then renames the input
   // vectors fY and fVyy, then performs the background subtraction
   // Data members modified:
   //   fYData,fY,fVyyData,fVyy,fVyyinvData,fVyyinv
   // and those modified by TUnfold::SetInput()
   // and those modified by DoBackgroundSubtraction()

   Int_t r=TUnfold::SetInput(hist_y,scaleBias,oneOverZeroError,hist_vyy,
                             hist_vyy_inv);
   fYData=fY;
   fY=nullptr;
   fVyyData=fVyy;
   fVyy=nullptr;
   DoBackgroundSubtraction();

   return r;
}

////////////////////////////////////////////////////////////////////////
/// Specify a source of background
///
/// \param[in] bgr background distribution with uncorrelated errors
/// \param[in] name identifier for this background source
/// \param[in] scale normalisation factor applied to the background
/// \param[in] scaleError normalisation uncertainty
///
/// The contribution <b>scale</b>*<b>bgr</b> is subtracted from the
/// measurement prior to unfolding. The following contributions are
/// added to the input covarianc ematrix
/// <ul>
/// <li>using the uncorrelated histogram errors <b>dbgr</b>, the contribution
/// (<b>scale</b>*<b>dbgr<sub>i</sub></b>)<sup>2</sup> is added to the
/// diagonals of the covariance</li>
/// <li>using the histogram contents, the background normalisation uncertainty contribution
/// <b>dscale</b>*<b>bgr<sub>i</sub></b> <b>dscale</b>*<b>bgr<sub>j</sub></b>
/// is added to the covariance matrix</li>

void TUnfoldSys::SubtractBackground
(const TH1 *bgr,const char *name,Double_t scale,Double_t scale_error)
{
   // Data members modified:
   //   fBgrIn,fBgrErrUncorrInSq,fBgrErrScaleIn
   // and those modified by DoBackgroundSubtraction()

   // save background source
   if(fBgrIn->FindObject(name)) {
      Error("SubtractBackground","Source %s given twice, ignoring 2nd call.\n",
            name);
   } else {
      TMatrixD *bgrScaled=new TMatrixD(GetNy(),1);
      TMatrixD *bgrErrUncSq=new TMatrixD(GetNy(),1);
      TMatrixD *bgrErrCorr=new TMatrixD(GetNy(),1);
      for(Int_t row=0;row<GetNy();row++) {
         (*bgrScaled)(row,0) = scale*bgr->GetBinContent(row+1);
         (*bgrErrUncSq)(row,0) =
            TMath::Power(scale*bgr->GetBinError(row+1),2.);
         (*bgrErrCorr)(row,0) = scale_error*bgr->GetBinContent(row+1);
      }
      fBgrIn->Add(new TObjString(name),bgrScaled);
      fBgrErrUncorrInSq->Add(new TObjString(name),bgrErrUncSq);
      fBgrErrScaleIn->Add(new TObjString(name),bgrErrCorr);
      if(fYData) {
         DoBackgroundSubtraction();
      } else {
         Info("SubtractBackground",
              "Background subtraction prior to setting input data");
      }
   }
}

////////////////////////////////////////////////////////////////////////
/// get background into a histogram
///
/// \param[inout] bgrHist target histogram, content and errors will be altered
/// \param[in] bgrSource (default=nullptr) name of backgrond source or zero
/// to add all sources of background
/// \param[in] binMap (default=nullptr) remap histogram bins
/// \param[in] includeError (default=3) include uncorrelated(1),
/// correlated (2) or both (3) sources of uncertainty in the
/// histogram errors
/// \param[in] clearHist (default=true) reset histogram before adding
/// up the specified background sources
///
/// the array <b>binMap</b> is explained with the method GetOutput().
/// The flag <b>clearHist</b> may be used to add background from
/// several sources in successive calls to GetBackground().

void TUnfoldSys::GetBackground
(TH1 *bgrHist,const char *bgrSource,const Int_t *binMap,
 Int_t includeError,Bool_t clearHist) const
{
   if(clearHist) {
       ClearHistogram(bgrHist);
   }
   // add all background sources
   const TObject *key;
   {
      TMapIter bgrPtr(fBgrIn);
      for(key=bgrPtr.Next();key;key=bgrPtr.Next()) {
         TString bgrName=((const TObjString *)key)->GetString();
         if(bgrSource && bgrName.CompareTo(bgrSource)) continue;
         const TMatrixD *bgr=(const TMatrixD *)((const TPair *)*bgrPtr)->Value();
         for(Int_t i=0;i<GetNy();i++) {
            Int_t destBin=binMap[i+1];
            bgrHist->SetBinContent(destBin,bgrHist->GetBinContent(destBin)+
                                   (*bgr)(i,0));
         }
      }
   }
   // add uncorrelated background errors
   if(includeError &1) {
      TMapIter bgrErrUncorrSqPtr(fBgrErrUncorrInSq);
      for(key=bgrErrUncorrSqPtr.Next();key;key=bgrErrUncorrSqPtr.Next()) {
         TString bgrName=((const TObjString *)key)->GetString();
         if(bgrSource && bgrName.CompareTo(bgrSource)) continue;
         const TMatrixD *bgrerruncorrSquared=(TMatrixD const *)((const TPair *)*bgrErrUncorrSqPtr)->Value();
         for(Int_t i=0;i<GetNy();i++) {
            Int_t destBin=binMap[i+1];
            bgrHist->SetBinError
               (destBin,TMath::Sqrt
                ((*bgrerruncorrSquared)(i,0)+
                 TMath::Power(bgrHist->GetBinError(destBin),2.)));
         }
      }
   }
   if(includeError & 2) {
      TMapIter bgrErrScalePtr(fBgrErrScaleIn);
      for(key=bgrErrScalePtr.Next();key;key=bgrErrScalePtr.Next()) {
         TString bgrName=((const TObjString *)key)->GetString();
         if(bgrSource && bgrName.CompareTo(bgrSource)) continue;
         const TMatrixD *bgrerrscale=(TMatrixD const *)((const TPair *)*bgrErrScalePtr)->Value();
         for(Int_t i=0;i<GetNy();i++) {
            Int_t destBin=binMap[i+1];
            bgrHist->SetBinError(destBin,hypot((*bgrerrscale)(i,0),
                                               bgrHist->GetBinError(destBin)));
         }
      }
   }
}

void TUnfoldSys::InitTUnfoldSys(void)
{
   // initialize pointers and TMaps

   // input
   fDAinRelSq = nullptr;
   fDAinColRelSq = nullptr;
   fAoutside = nullptr;
   fBgrIn = new TMap();
   fBgrErrUncorrInSq = new TMap();
   fBgrErrScaleIn = new TMap();
   fSysIn = new TMap();
   fBgrIn->SetOwnerKeyValue();
   fBgrErrUncorrInSq->SetOwnerKeyValue();
   fBgrErrScaleIn->SetOwnerKeyValue();
   fSysIn->SetOwnerKeyValue();
   // results
   fEmatUncorrX = nullptr;
   fEmatUncorrAx = nullptr;
   fDeltaCorrX = new TMap();
   fDeltaCorrAx = new TMap();
   fDeltaCorrX->SetOwnerKeyValue();
   fDeltaCorrAx->SetOwnerKeyValue();
   fDeltaSysTau = nullptr;
   fDtau=0.0;
   fYData=nullptr;
   fVyyData=nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Clear all data members which depend on the unfolding results.

void TUnfoldSys::ClearResults(void)
{
   // clear all data members which depend on the unfolding results
   TUnfold::ClearResults();
   DeleteMatrix(&fEmatUncorrX);
   DeleteMatrix(&fEmatUncorrAx);
   fDeltaCorrX->Clear();
   fDeltaCorrAx->Clear();
   DeleteMatrix(&fDeltaSysTau);
}

////////////////////////////////////////////////////////////////////////
/// Matrix calculations required to propagate systematic errors
void TUnfoldSys::PrepareSysError(void)
{
   // data members modified
   //    fEmatUncorrX, fEmatUncorrAx, fDeltaCorrX, fDeltaCorrAx
   if(!fEmatUncorrX) {
      fEmatUncorrX=PrepareUncorrEmat(GetDXDAM(0),GetDXDAM(1));
   }
   TMatrixDSparse *AM0=nullptr,*AM1=nullptr;
   if(!fEmatUncorrAx) {
      if(!AM0) AM0=MultiplyMSparseMSparse(fA,GetDXDAM(0));
      if(!AM1) {
         AM1=MultiplyMSparseMSparse(fA,GetDXDAM(1));
         Int_t *rows_cols=new Int_t[GetNy()];
         Double_t *data=new Double_t[GetNy()];
         for(Int_t i=0;i<GetNy();i++) {
            rows_cols[i]=i;
            data[i]=1.0;
         }
         TMatrixDSparse *one=CreateSparseMatrix
            (GetNy(),GetNy(),GetNy(),rows_cols, rows_cols,data);
         delete[] data;
         delete[] rows_cols;
         AddMSparse(AM1,-1.,one);
         DeleteMatrix(&one);
         fEmatUncorrAx=PrepareUncorrEmat(AM0,AM1);
      }
   }
   if((!fDeltaSysTau )&&(fDtau>0.0)) {
      fDeltaSysTau=new TMatrixDSparse(*GetDXDtauSquared());
      Double_t scale=2.*TMath::Sqrt(fTauSquared)*fDtau;
      Int_t n=fDeltaSysTau->GetRowIndexArray() [fDeltaSysTau->GetNrows()];
      Double_t *data=fDeltaSysTau->GetMatrixArray();
      for(Int_t i=0;i<n;i++) {
         data[i] *= scale;
      }
   }

   TMapIter sysErrIn(fSysIn);
   const TObjString *key;

   // calculate individual systematic errors
   for(key=(const TObjString *)sysErrIn.Next();key;
       key=(const TObjString *)sysErrIn.Next()) {
      const TMatrixDSparse *dsys=
         (const TMatrixDSparse *)((const TPair *)*sysErrIn)->Value();
      const TPair *named_emat=(const TPair *)
         fDeltaCorrX->FindObject(key->GetString());
      if(!named_emat) {
         TMatrixDSparse *emat=PrepareCorrEmat(GetDXDAM(0),GetDXDAM(1),dsys);
         fDeltaCorrX->Add(new TObjString(*key),emat);
      }
      named_emat=(const TPair *)fDeltaCorrAx->FindObject(key->GetString());
      if(!named_emat) {
         if(!AM0) AM0=MultiplyMSparseMSparse(fA,GetDXDAM(0));
         if(!AM1) {
            AM1=MultiplyMSparseMSparse(fA,GetDXDAM(1));
            Int_t *rows_cols=new Int_t[GetNy()];
            Double_t *data=new Double_t[GetNy()];
            for(Int_t i=0;i<GetNy();i++) {
               rows_cols[i]=i;
               data[i]=1.0;
            }
            TMatrixDSparse *one=CreateSparseMatrix
               (GetNy(),GetNy(),GetNy(),rows_cols, rows_cols,data);
            delete[] data;
            delete[] rows_cols;
            AddMSparse(AM1,-1.,one);
            DeleteMatrix(&one);
            fEmatUncorrAx=PrepareUncorrEmat(AM0,AM1);
         }
         TMatrixDSparse *emat=PrepareCorrEmat(AM0,AM1,dsys);
         fDeltaCorrAx->Add(new TObjString(*key),emat);
      }
   }
   DeleteMatrix(&AM0);
   DeleteMatrix(&AM1);
}

////////////////////////////////////////////////////////////////////////
/// Covariance contribution from uncorrelated uncertainties of the
/// response matrix
///
/// \param[inout] ematrix covariance matrix histogram
/// \param[in] binMap mapping of histogram bins
/// \param[in] clearEmat if true, ematrix is cleared prior to adding
/// this covariance matrix contribution
///
/// This method propagates the uncertainties of the response matrix
/// histogram, specified with the constructor, to the unfolding
/// result. It is assumed that the entries of that histogram are
/// bin-to-bin uncorrelated. In many cases this corresponds to the
/// "Monte Carlo statistical uncertainties".
/// <br/>
/// The array <b>binMap</b> is explained with the method GetOutput().
/// The flag <b>clearEmat</b> may be used to add covariance matrices from
/// several uncertainty sources.
///
void TUnfoldSys::GetEmatrixSysUncorr
(TH2 *ematrix,const Int_t *binMap,Bool_t clearEmat)
{
   // data members modified:
   //   fVYAx, fESparse, fEAtV, fErrorAStat
   PrepareSysError();
   ErrorMatrixToHist(ematrix,fEmatUncorrX,binMap,clearEmat);
}

////////////////////////////////////////////////////////////////////////
/// propagate uncorrelated systematic errors to a covariance matrix
///
/// \param[in] m_0 coefficients for error propagation
/// \param[in] m_1 coefficients for error propagation
///
/// returns the covariance matrix

TMatrixDSparse *TUnfoldSys::PrepareUncorrEmat
(const TMatrixDSparse *m_0,const TMatrixDSparse *m_1)
{
   // propagate uncorrelated systematic errors to a covariance matrix
   //   m0,m1 : coefficients (matrices) for propagating the errors
   //
   // the error matrix is calculated by standard error propagation, where the
   // derivative of the result vector X wrt the matrix A is given by
   //
   //  dX_k / dA_ij  =  M0_kj * Z0_i  - M1_ki * Z1_j
   //
   // where:
   //   the matrices M0 and M1 are arguments to this function
   //   the vectors Z0, Z1 : GetDXDAZ()
   //
   // The matrix A is calculated from a matrix B as
   //
   //    A_ij = B_ij / sum_k B_kj
   //
   // where k runs over additional indices of B, not present in A.
   // (underflow and overflow bins, used for efficiency corrections)
   //
   // define:   Norm_j = sum_k B_kj   (data member fSumOverY)
   //
   // the derivative of A wrt this input matrix B is given by:
   //
   //   dA_ij / dB_kj = (  delta_ik - A_ij ) * 1/Norm_j
   //
   // The covariance matrix Vxx is:
   //
   //   Vxx_mn  = sum_ijlk [   (dX_m / dA_ij) * (dA_ij / dB_kj) * DB_kj
   //                        * (dX_n / dA_lj) * (dA_lj / dB_kj)  ]
   //
   // where DB_kj is the error on B_kj squared
   // Simplify the sum over k:
   //
   //   sum_k [ (dA_ij / dB_kj) * DB_kj * (dA_lj / dB_kj) ]
   //      =  sum_k [  ( delta_ik - A_ij ) * 1/Norm_j * DB_kj *
   //                * ( delta_lk - A_lj ) * 1/Norm_j ]
   //      =  sum_k [ ( delta_ik*delta_lk - delta_ik*A_lj - delta_lk*A_ij
   //                  + A_ij * A_lj ) * DB_kj / Norm_j^2 ]
   //
   // introduce normalized errors:  Rsq_kj = DB_kj / Norm_j^2
   // after summing over k:
   //   delta_ik*delta_lk*Rsq_kj  ->    delta_il*Rsq_ij
   //   delta_ik*A_lj*Rsq_kj      ->    A_lj*Rsq_ij
   //   delta_lk*A_ij*Rsq_kj      ->    A_ij*Rsq_lj
   //   A_ij*A_lj*Rsq_kj          ->    A_ij*A_lj*sum_k(Rsq_kj)
   //
   // introduce sum of normalized errors squared:   SRsq_j = sum_k(Rsq_kj)
   //
   // Note: Rsq_ij is stored as  fDAinRelSq     (excludes extra indices of B)
   //   and SRsq_j is stored as  fDAinColRelSq  (sum includes all indices of B)
   //
   //  Vxx_nm = sum_ijl [ (dX_m / dA_ij) * (dX_n / dA_lj)
   //     (delta_il*Rsq_ij - A_lj*Rsq_ij - A_ij*Rsq_lj + A_ij*A_lj *SRsq_j) ]
   //
   //  Vxx_nm =    sum_j [ F_mj * F_nj * SRsq_j
   //            - sum_j [ G_mj * F_nj ]
   //            - sum_j [ F_mj * G_nj ]
   //            + sum_ij [  (dX_m / dA_ij) * (dX_n / dA_lj) * Rsq_ij ]
   //
   // where:
   //    F_mj = sum_i [ (dX_m / dA_ij) * A_ij ]
   //    G_mj = sum_i [ (dX_m / dA_ij) * Rsq_ij ]
   //
   // In order to avoid explicitly calculating the 3-dimensional tensor
   // (dX_m/dA_ij) the sums are evaluated further, using
   //    dX_k / dA_ij  =  M0_kj * Z0_i  - M1_ki * Z1_j
   //
   //   F_mj = M0_mj * (A# Z0)_j - (M1 A)_mj Z1_j
   //   G_mj = M0_mj * (Rsq# Z0)_j - (M1 Rsq)_mj Z1_j
   //
   // and
   //
   //   sum_ij [ (dX_m/dA_ij) * (dX_n/dA_ij) * Rsq_ij ] =
   //      sum_j [ M0_mj * M0_nj *  [ sum_i (Z0_i)^2 * Rsq_ij ] ]
   //    + sum_i [ M1_mi * M1_ni *  [ sum_j (Z1_j)^2 * Rsq_ij ] ]
   //    - sum_i [ M1_mi * H_ni + M1_ni * H_mi]
   // where:
   //   H_mi = Z0_i * sum_j [ M0_mj * Z1_j * Rsq_ij ]
   //
   // collect all contributions:
   //   Vxx_nm = r0 -r1 -r2 +r3 +r4 -r5 -r6
   //      r0 = sum_j [ F_mj * F_nj * SRsq_j ]
   //      r1 = sum_j [ G_mj * F_nj ]
   //      r2 = sum_j [ F_mj * G_nj ]
   //      r3 = sum_j [ M0_mj * M0_nj *  [ sum_i (Z0_i)^2 * Rsq_ij ] ]
   //      r4 = sum_i [ M1_mi * M1_ni *  [ sum_j (Z1_j)^2 * Rsq_ij ] ]
   //      r5 = sum_i [ M1_mi * H_ni ]
   //      r6 = sum_i [ M1_ni * H_mi ]

   //======================================================
   // calculate contributions containing matrices F and G
   // r0,r1,r2
   TMatrixDSparse *r=nullptr;
   if(fDAinColRelSq && fDAinRelSq) {
      // calculate matrices (M1*A)_mj * Z1_j  and  (M1*Rsq)_mj * Z1_j
      TMatrixDSparse *M1A_Z1=MultiplyMSparseMSparse(m_1,fA);
      ScaleColumnsByVector(M1A_Z1,GetDXDAZ(1));
      TMatrixDSparse *M1Rsq_Z1=MultiplyMSparseMSparse(m_1,fDAinRelSq);
      ScaleColumnsByVector(M1Rsq_Z1,GetDXDAZ(1));
      // calculate vectors A#*Z0  and  Rsq#*Z0
      TMatrixDSparse *AtZ0 = MultiplyMSparseTranspMSparse(fA,GetDXDAZ(0));
      TMatrixDSparse *RsqZ0=
         MultiplyMSparseTranspMSparse(fDAinRelSq,GetDXDAZ(0));
      //calculate matrix F
      //   F_mj = M0_mj * (A# Z0)_j - (M1 A)_mj Z1_j
      TMatrixDSparse *F=new TMatrixDSparse(*m_0);
      ScaleColumnsByVector(F,AtZ0);
      AddMSparse(F,-1.0,M1A_Z1);
      //calculate matrix G
      //   G_mj = M0_mj * (Rsq# Z0)_j - (M1 Rsq)_mj Z1_j
      TMatrixDSparse *G=new TMatrixDSparse(*m_0);
      ScaleColumnsByVector(G,RsqZ0);
      AddMSparse(G,-1.0,M1Rsq_Z1);
      DeleteMatrix(&M1A_Z1);
      DeleteMatrix(&M1Rsq_Z1);
      DeleteMatrix(&AtZ0);
      DeleteMatrix(&RsqZ0);
      //      r0 = sum_j [ F_mj * F_nj * SRsq_j ]
      r=MultiplyMSparseMSparseTranspVector(F,F,fDAinColRelSq);
      //      r1 = sum_j [ G_mj * F_nj ]
      TMatrixDSparse *r1=MultiplyMSparseMSparseTranspVector(F,G,nullptr);
      //      r2 = sum_j [ F_mj * G_nj ]
      TMatrixDSparse *r2=MultiplyMSparseMSparseTranspVector(G,F,nullptr);
      // r = r0-r1-r2
      AddMSparse(r,-1.0,r1);
      AddMSparse(r,-1.0,r2);
      DeleteMatrix(&r1);
      DeleteMatrix(&r2);
      DeleteMatrix(&F);
      DeleteMatrix(&G);
   }
   //======================================================
   // calculate contribution
   //   sum_ij [ (dX_m/dA_ij) * (dX_n/dA_ij) * Rsq_ij ]
   //  (r3,r4,r5,r6)
   if(fDAinRelSq) {
      // (Z0_i)^2
      TMatrixDSparse Z0sq(*GetDXDAZ(0));
      const Int_t *Z0sq_rows=Z0sq.GetRowIndexArray();
      Double_t *Z0sq_data=Z0sq.GetMatrixArray();
      for(int index=0;index<Z0sq_rows[Z0sq.GetNrows()];index++) {
         Z0sq_data[index] *= Z0sq_data[index];
      }
      // Z0sqRsq =  sum_i (Z_i)^2 * Rsq_ij
      TMatrixDSparse *Z0sqRsq=MultiplyMSparseTranspMSparse(fDAinRelSq,&Z0sq);
      //      r3 = sum_j [ M0_mj * M0_nj *  [ sum_i (Z0_i)^2 * Rsq_ij ] ]
      TMatrixDSparse *r3=MultiplyMSparseMSparseTranspVector(m_0,m_0,Z0sqRsq);
      DeleteMatrix(&Z0sqRsq);

      // (Z1_j)^2
      TMatrixDSparse Z1sq(*GetDXDAZ(1));
      const Int_t *Z1sq_rows=Z1sq.GetRowIndexArray();
      Double_t *Z1sq_data=Z1sq.GetMatrixArray();
      for(int index=0;index<Z1sq_rows[Z1sq.GetNrows()];index++) {
         Z1sq_data[index] *= Z1sq_data[index];
      }
      // Z1sqRsq = sum_j (Z1_j)^2 * Rsq_ij ]
      TMatrixDSparse *Z1sqRsq=MultiplyMSparseMSparse(fDAinRelSq,&Z1sq);
      //      r4 = sum_i [ M1_mi * M1_ni *  [ sum_j (Z1_j)^2 * Rsq_ij ] ]
      TMatrixDSparse *r4=MultiplyMSparseMSparseTranspVector(m_1,m_1,Z1sqRsq);
      DeleteMatrix(&Z1sqRsq);

      // sum_j [ M0_mj * Z1_j * Rsq_ij ]
      TMatrixDSparse *H=MultiplyMSparseMSparseTranspVector
         (m_0,fDAinRelSq,GetDXDAZ(1));
      // H_mi = Z0_i * sum_j [ M0_mj * Z1_j * Rsq_ij ]
      ScaleColumnsByVector(H,GetDXDAZ(0));
      //      r5 = sum_i [ M1_mi * H_ni ]
      TMatrixDSparse *r5=MultiplyMSparseMSparseTranspVector(m_1,H,nullptr);
      //      r6 = sum_i [ H_mi * M1_ni ]
      TMatrixDSparse *r6=MultiplyMSparseMSparseTranspVector(H,m_1,nullptr);
      DeleteMatrix(&H);
      // r =  r0 -r1 -r2 +r3 +r4 -r5 -r6
      if(r) {
         AddMSparse(r,1.0,r3);
         DeleteMatrix(&r3);
      } else {
         r=r3;
         r3=nullptr;
      }
      AddMSparse(r,1.0,r4);
      AddMSparse(r,-1.0,r5);
      AddMSparse(r,-1.0,r6);
      DeleteMatrix(&r4);
      DeleteMatrix(&r5);
      DeleteMatrix(&r6);
   }
   return r;
}

////////////////////////////////////////////////////////////////////////
///  propagate correlated systematic shift to an output vector
///
/// \param[in] m1 coefficients
/// \param[in] m2 coeffiicients
/// \param[in] dsys matrix of correlated shifts from this source

TMatrixDSparse *TUnfoldSys::PrepareCorrEmat
(const TMatrixDSparse *m1,const TMatrixDSparse *m2,const TMatrixDSparse *dsys)
{
   // propagate correlated systematic shift to output vector
   //   m1,m2 : coefficients for propagating the errors
   //   dsys : matrix of correlated shifts from this source

   // delta_m =
   //   sum{i,j}   {
   //      ((*m1)(m,j) * (*fVYAx)(i) - (*m2)(m,i) * (*fX)(j))*dsys(i,j) }
   //   =    sum_j (*m1)(m,j)  sum_i dsys(i,j) * (*fVYAx)(i)
   //     -  sum_i (*m2)(m,i)  sum_j dsys(i,j) * (*fX)(j)

   TMatrixDSparse *dsysT_VYAx = MultiplyMSparseTranspMSparse(dsys,GetDXDAZ(0));
   TMatrixDSparse *delta =  MultiplyMSparseMSparse(m1,dsysT_VYAx);
   DeleteMatrix(&dsysT_VYAx);
   TMatrixDSparse *dsys_X = MultiplyMSparseMSparse(dsys,GetDXDAZ(1));
   TMatrixDSparse *delta2 = MultiplyMSparseMSparse(m2,dsys_X);
   DeleteMatrix(&dsys_X);
   AddMSparse(delta,-1.0,delta2);
   DeleteMatrix(&delta2);
   return delta;
}

////////////////////////////////////////////////////////////////////////
/// Specify an uncertainty on tau
///
/// \param[in] delta_tau new uncertainty on tau
///
/// The default is to have no uncertyainty on tau.
void TUnfoldSys::SetTauError(Double_t delta_tau)
{
   // set uncertainty on tau
   fDtau=delta_tau;
   DeleteMatrix(&fDeltaSysTau);
}

////////////////////////////////////////////////////////////////////////
/// correlated one-sigma shifts correspinding to a given systematic uncertainty
///
/// \param[out] hist_delta histogram to store shifts
/// \param[in] name  identifier of the background source
/// \param[in] binMap (default=nullptr) remapping of histogram bins
///
/// returns true if the error source was found.
/// <br>
/// This method returns the shifts of the unfolding result induced by
/// varying the identified systematic source by one sigma.
/// <br/>
/// the array <b>binMap</b> is explained with the method GetOutput().
Bool_t TUnfoldSys::GetDeltaSysSource(TH1 *hist_delta,const char *name,
                                   const Int_t *binMap)
{
   PrepareSysError();
   const TPair *named_emat=(const TPair *)fDeltaCorrX->FindObject(name);
   const TMatrixDSparse *delta=nullptr;
   if(named_emat) {
      delta=(TMatrixDSparse *)named_emat->Value();
   }
   VectorMapToHist(hist_delta,delta,binMap);
   return delta !=nullptr;
}

////////////////////////////////////////////////////////////////////////
/// correlated one-sigma shifts from background normalisation uncertainty
///
/// \param[out] hist_delta histogram to store shifts
/// \param[in] source  identifier of the background source
/// \param[in] binMap (default=nullptr) remapping of histogram bins
///
/// returns true if the background source was found.
/// <br>
/// This method returns the shifts of the unfolding result induced by
/// varying the normalisation of the identified background by one sigma.
/// <br/>
/// the array <b>binMap</b> is explained with the method GetOutput().

Bool_t TUnfoldSys::GetDeltaSysBackgroundScale
(TH1 *hist_delta,const char *source,const Int_t *binMap)
{
   PrepareSysError();
   const TPair *named_err=(const TPair *)fBgrErrScaleIn->FindObject(source);
   TMatrixDSparse *dx=nullptr;
   if(named_err) {
      const TMatrixD *dy=(TMatrixD *)named_err->Value();
      dx=MultiplyMSparseM(GetDXDY(),dy);
   }
   VectorMapToHist(hist_delta,dx,binMap);
   if(dx!=nullptr) {
      DeleteMatrix(&dx);
      return kTRUE;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////
/// correlated one-sigma shifts from shifting tau
///
/// \param[out] hist_delta histogram to store shifts
/// \param[in] source  identifier of the background source
/// \param[in] binMap (default=nullptr) remapping of histogram bins
///
/// returns true if the background source was found.
/// <br>
/// This method returns the shifts of the unfolding result induced by
/// varying the normalisation of the identified background by one sigma.
/// <br/>
/// the array <b>binMap</b> is explained with the method GetOutput().

Bool_t TUnfoldSys::GetDeltaSysTau(TH1 *hist_delta,const Int_t *binMap)
{
   // calculate systematic shift from tau variation
   //    ematrix: output
   //    binMap: see method GetEmatrix()
   PrepareSysError();
   VectorMapToHist(hist_delta,fDeltaSysTau,binMap);
   return fDeltaSysTau !=nullptr;
}

////////////////////////////////////////////////////////////////////////
/// covariance contribution from a systematic variation of the
/// response matrix
///
/// \param[inout] ematrix covariance matrix histogram
/// \param[in] name identifier of the systematic variation
/// \param[in] binMap (default=nullptr) remapping of histogram bins
/// \param[in] clearEmat (default=true) if true, clear the histogram
/// prior to adding the covariance matrix contribution
///
/// Returns the covariance matrix contribution from shifting the given
/// uncertainty source within one sigma
/// <br/>
/// the array <b>binMap</b> is explained with the method GetOutput().
/// The flag <b>clearEmat</b> may be used to add covariance matrices from
/// several uncertainty sources.

void TUnfoldSys::GetEmatrixSysSource
(TH2 *ematrix,const char *name,const Int_t *binMap,Bool_t clearEmat)
{
   PrepareSysError();
   const TPair *named_emat=(const TPair *)fDeltaCorrX->FindObject(name);
   TMatrixDSparse *emat=nullptr;
   if(named_emat) {
      const TMatrixDSparse *delta=(TMatrixDSparse *)named_emat->Value();
      emat=MultiplyMSparseMSparseTranspVector(delta,delta,nullptr);
   }
   ErrorMatrixToHist(ematrix,emat,binMap,clearEmat);
   DeleteMatrix(&emat);
}

////////////////////////////////////////////////////////////////////////
/// covariance contribution from background normalisation uncertainty
///
/// \param[inout] ematrix output histogram
/// \param[in] source identifier of the background source
/// \param[in] binMap (default=nullptr) remapping of histogram bins
/// \param[in] clearEmat (default=true) if true, clear the histogram
/// prior to adding the covariance matrix contribution
///
/// this method returns the uncertainties on the unfolding result
/// arising from the background source <b>source</b> and its normalisation
/// uncertainty. See method SubtractBackground() how to set the normalisation uncertainty
/// <br/>
/// the array <b>binMap</b> is explained with the method GetOutput().
/// The flag <b>clearEmat</b> may be used to add covariance matrices from
/// several uncertainty sources.

void TUnfoldSys::GetEmatrixSysBackgroundScale
(TH2 *ematrix,const char *name,const Int_t *binMap,Bool_t clearEmat)
{
   PrepareSysError();
   const TPair *named_err=(const TPair *)fBgrErrScaleIn->FindObject(name);
   TMatrixDSparse *emat=nullptr;
   if(named_err) {
      const TMatrixD *dy=(TMatrixD *)named_err->Value();
      TMatrixDSparse *dx=MultiplyMSparseM(GetDXDY(),dy);
      emat=MultiplyMSparseMSparseTranspVector(dx,dx,nullptr);
      DeleteMatrix(&dx);
   }
   ErrorMatrixToHist(ematrix,emat,binMap,clearEmat);
   DeleteMatrix(&emat);
}

////////////////////////////////////////////////////////////////////////
/// covariance matrix contribution from error on regularisation
/// parameter
///
/// \param[inout] ematrix output histogram
/// \param[in] binMap (default=nullptr) remapping of histogram bins
/// \param[in] clearEmat (default=true) if true, clear the histogram
///
/// this method returns the covariance contributions to the unfolding result
/// from the assigned uncertainty on the parameter tau, see method
/// SetTauError().
/// <br>
/// the array <b>binMap</b> is explained with the method GetOutput().
/// The flag <b>clearEmat</b> may be used to add covariance matrices from
/// several uncertainty sources.

void TUnfoldSys::GetEmatrixSysTau
(TH2 *ematrix,const Int_t *binMap,Bool_t clearEmat)
{
   // calculate error matrix from error in regularisation parameter
   //    ematrix: output
   //    binMap: see method GetEmatrix()
   //    clearEmat: set kTRUE to clear the histogram prior to adding the errors
   PrepareSysError();
   TMatrixDSparse *emat=nullptr;
   if(fDeltaSysTau) {
      emat=MultiplyMSparseMSparseTranspVector(fDeltaSysTau,fDeltaSysTau,nullptr);
   }
   ErrorMatrixToHist(ematrix,emat,binMap,clearEmat);
   DeleteMatrix(&emat);
}

////////////////////////////////////////////////////////////////////////
/// covariance matrix contribution from input measurement uncertainties
///
/// \param[inout] ematrix output histogram
/// \param[in] binMap (default=nullptr) remapping of histogram bins
/// \param[in] clearEmat (default=true) if true, clear the histogram
///
/// this method returns the covariance contributions to the unfolding result
/// from the uncertainties or covariance of the input
/// data. In many cases, these are the "statistical uncertainties".
/// <br>
/// The array <b>binMap</b> is explained with the method GetOutput().
/// The flag <b>clearEmat</b> may be used to add covariance matrices from
/// several uncertainty sources.

void TUnfoldSys::GetEmatrixInput
(TH2 *ematrix,const Int_t *binMap,Bool_t clearEmat)
{
   GetEmatrixFromVyy(fVyyData,ematrix,binMap,clearEmat);
}

////////////////////////////////////////////////////////////////////////
/// covariance contribution from background uncorrelated  uncertainty
///
/// \param[in] ematrix output histogram
/// \param[in] source identifier of the background source
/// \param[in] binMap (default=nullptr) remapping of histogram bins
/// \param[in] clearEmat (default=true) if true, clear the histogram
///
/// this method returns the covariance contributions to the unfolding result
/// arising from the background source <b>source</b> and the uncorrelated
/// (background histogram uncertainties). Also see method SubtractBackground()
/// <br/>
/// the array <b>binMap</b> is explained with the method GetOutput().
/// The flag <b>clearEmat</b> may be used to add covariance matrices from
/// several uncertainty sources.

void TUnfoldSys::GetEmatrixSysBackgroundUncorr
(TH2 *ematrix,const char *source,const Int_t *binMap,Bool_t clearEmat)
{
   const TPair *named_err=(const TPair *)fBgrErrUncorrInSq->FindObject(source);
   TMatrixDSparse *emat=nullptr;
   if(named_err) {
      TMatrixD const *dySquared=(TMatrixD const *)named_err->Value();
      emat=MultiplyMSparseMSparseTranspVector(GetDXDY(),GetDXDY(),dySquared);
   }
   ErrorMatrixToHist(ematrix,emat,binMap,clearEmat);
   DeleteMatrix(&emat);
}

////////////////////////////////////////////////////////////////////////
/// propagate an error matrix on the input vector to the unfolding result
///
/// \param[in] vyy input error matrix
/// \param[inout] ematrix histogram to be updated
/// \param[in] binMap  mapping of histogram bins
/// \param[in] clearEmat if set, clear histogram before adding this
/// covariance contribution
void TUnfoldSys::GetEmatrixFromVyy
(const TMatrixDSparse *vyy,TH2 *ematrix,const Int_t *binMap,Bool_t clearEmat)
{
   // propagate error matrix vyy to the result
   //    vyy: error matrix on input data fY
   //    ematrix: output
   //    binMap: see method GetEmatrix()
   //    clearEmat: set kTRUE to clear the histogram prior to adding the errors
   PrepareSysError();
   TMatrixDSparse *em=nullptr;
   if(vyy) {
      TMatrixDSparse *dxdyVyy=MultiplyMSparseMSparse(GetDXDY(),vyy);
      em=MultiplyMSparseMSparseTranspVector(dxdyVyy,GetDXDY(),nullptr);
      DeleteMatrix(&dxdyVyy);
   }
   ErrorMatrixToHist(ematrix,em,binMap,clearEmat);
   DeleteMatrix(&em);
}

////////////////////////////////////////////////////////////////////////
/// Get total error matrix, summing up all contributions
///
/// \param[out] ematrix histogram which will be filled
/// \param[in] binMap (default=nullptr) remapping of histogram bins
///
/// the array <b>binMap</b> is explained with the method GetOutput().
void TUnfoldSys::GetEmatrixTotal(TH2 *ematrix,const Int_t *binMap)
{
   // get total error including statistical error
   //    ematrix: output
   //    binMap: see method GetEmatrix()
   GetEmatrix(ematrix,binMap);  // (stat)+(d)+(e)
   GetEmatrixSysUncorr(ematrix,binMap,kFALSE); // (a)
   TMapIter sysErrPtr(fDeltaCorrX);
   const TObject *key;

   for(key=sysErrPtr.Next();key;key=sysErrPtr.Next()) {
      GetEmatrixSysSource(ematrix,
                          ((const TObjString *)key)->GetString(),
                          binMap,kFALSE); // (b)
   }
   GetEmatrixSysTau(ematrix,binMap,kFALSE); // (c)
}

////////////////////////////////////////////////////////////////////////
/// determine total error matrix on the vector Ax
TMatrixDSparse *TUnfoldSys::GetSummedErrorMatrixYY(void)
{
   PrepareSysError();

   // errors from input vector and from background subtraction
   TMatrixDSparse *emat_sum=new TMatrixDSparse(*fVyy);

   // uncorrelated systematic error
   if(fEmatUncorrAx) {
      AddMSparse(emat_sum,1.0,fEmatUncorrAx);
   }
   TMapIter sysErrPtr(fDeltaCorrAx);
   const TObject *key;

   // correlated systematic errors
   for(key=sysErrPtr.Next();key;key=sysErrPtr.Next()) {
      const TMatrixDSparse *delta=(TMatrixDSparse *)((const TPair *)*sysErrPtr)->Value();
      TMatrixDSparse *emat=MultiplyMSparseMSparseTranspVector(delta,delta,nullptr);
      AddMSparse(emat_sum,1.0,emat);
      DeleteMatrix(&emat);
   }
   // error on tau
   if(fDeltaSysTau) {
      TMatrixDSparse *Adx_tau=MultiplyMSparseMSparse(fA,fDeltaSysTau);
      TMatrixDSparse *emat_tau=
         MultiplyMSparseMSparseTranspVector(Adx_tau,Adx_tau,nullptr);
      DeleteMatrix(&Adx_tau);
      AddMSparse(emat_sum,1.0,emat_tau);
      DeleteMatrix(&emat_tau);
   }
   return emat_sum;
}

////////////////////////////////////////////////////////////////////////
/// determine total error matrix on the vector x
TMatrixDSparse *TUnfoldSys::GetSummedErrorMatrixXX(void)
{
   PrepareSysError();

   // errors from input vector and from background subtraction
   TMatrixDSparse *emat_sum=new TMatrixDSparse(*GetVxx());

   // uncorrelated systematic error
   if(fEmatUncorrX) {
      AddMSparse(emat_sum,1.0,fEmatUncorrX);
   }
   TMapIter sysErrPtr(fDeltaCorrX);
   const TObject *key;

   // correlated systematic errors
   for(key=sysErrPtr.Next();key;key=sysErrPtr.Next()) {
      const TMatrixDSparse *delta=(TMatrixDSparse *)((const TPair *)*sysErrPtr)->Value();
      TMatrixDSparse *emat=MultiplyMSparseMSparseTranspVector(delta,delta,nullptr);
      AddMSparse(emat_sum,1.0,emat);
      DeleteMatrix(&emat);
   }
   // error on tau
   if(fDeltaSysTau) {
      TMatrixDSparse *emat_tau=
         MultiplyMSparseMSparseTranspVector(fDeltaSysTau,fDeltaSysTau,nullptr);
      AddMSparse(emat_sum,1.0,emat_tau);
      DeleteMatrix(&emat_tau);
   }
   return emat_sum;
}


////////////////////////////////////////////////////////////////////////
/// calculate total chi**2 including all systematic errors

Double_t TUnfoldSys::GetChi2Sys(void)
{

   TMatrixDSparse *emat_sum=GetSummedErrorMatrixYY();

   Int_t rank=0;
   TMatrixDSparse *v=InvertMSparseSymmPos(emat_sum,&rank);
   TMatrixD dy(*fY, TMatrixD::kMinus, *GetAx());
   TMatrixDSparse *vdy=MultiplyMSparseM(v,&dy);
   DeleteMatrix(&v);
   Double_t r=0.0;
   const Int_t *vdy_rows=vdy->GetRowIndexArray();
   const Double_t *vdy_data=vdy->GetMatrixArray();
   for(Int_t i=0;i<vdy->GetNrows();i++) {
      if(vdy_rows[i+1]>vdy_rows[i]) {
         r += vdy_data[vdy_rows[i]] * dy(i,0);
      }
   }
   DeleteMatrix(&vdy);
   DeleteMatrix(&emat_sum);
   return r;
}

////////////////////////////////////////////////////////////////////////
/// Get global correlatiocn coefficients, summing up all contributions
///
/// \param[out] rhoi histogram which will be filled
/// \param[in] binMap (default=nullptr) remapping of histogram bins
/// \param[out] invEmat (default=nullptr) inverse of error matrix
///
/// return the global correlation coefficients, including all error
/// sources. If <b>invEmat</b> is nonzero, the inverse of the error
/// matrix is returned in that histogram
/// <br/>
/// the array <b>binMap</b> is explained with the method GetOutput().
void TUnfoldSys::GetRhoItotal(TH1 *rhoi,const Int_t *binMap,TH2 *invEmat)
{
   // get global correlation coefficients including systematic,statistical,background,tau errors
   //    rhoi: output histogram
   //    binMap: for each global bin, indicate in which histogram bin
   //            to store its content
   //    invEmat: output histogram for inverse of error matrix
   //              (pointer may zero if inverse is not requested)
   ClearHistogram(rhoi,-1.);
   TMatrixDSparse *emat_sum=GetSummedErrorMatrixXX();
   GetRhoIFromMatrix(rhoi,emat_sum,binMap,invEmat);

   DeleteMatrix(&emat_sum);
}

////////////////////////////////////////////////////////////////////////
/// scale columns of a matrix by the corresponding rows of a vector
///
/// \par[inout] m matrix
/// \par[in] v vector
///
/// the entries m<sub>ij</sub> are multiplied by v<sub>j</sub>.
void TUnfoldSys::ScaleColumnsByVector
(TMatrixDSparse *m,const TMatrixTBase<Double_t> *v) const
{
   // scale columns of m by the corresponding rows of v
   // input:
   //   m:  pointer to sparse matrix of dimension NxM
   //   v:  pointer to matrix of dimension Mx1
   if((m->GetNcols() != v->GetNrows())||(v->GetNcols()!=1)) {
      Fatal("ScaleColumnsByVector error",
            "matrix cols/vector rows %d!=%d OR vector cols %d !=1\n",
            m->GetNcols(),v->GetNrows(),v->GetNcols());
   }
   const Int_t *rows_m=m->GetRowIndexArray();
   const Int_t *cols_m=m->GetColIndexArray();
   Double_t *data_m=m->GetMatrixArray();
   const TMatrixDSparse *v_sparse=dynamic_cast<const TMatrixDSparse *>(v);
   if(v_sparse) {
      const Int_t *rows_v=v_sparse->GetRowIndexArray();
      const Double_t *data_v=v_sparse->GetMatrixArray();
      for(Int_t i=0;i<m->GetNrows();i++) {
         for(Int_t index_m=rows_m[i];index_m<rows_m[i+1];index_m++) {
            Int_t j=cols_m[index_m];
            Int_t index_v=rows_v[j];
            if(index_v<rows_v[j+1]) {
               data_m[index_m] *= data_v[index_v];
            } else {
               data_m[index_m] =0.0;
            }
         }
      }
   } else {
      for(Int_t i=0;i<m->GetNrows();i++) {
         for(Int_t index_m=rows_m[i];index_m<rows_m[i+1];index_m++) {
            Int_t j=cols_m[index_m];
            data_m[index_m] *= (*v)(j,0);
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////
/// map delta to hist_delta, possibly summing up bins
///
/// \param[out] hist_delta result histogram
/// \param[in] delta vector to be mapped to the histogram
/// \param[in] binMap  mapping of histogram bins
///
/// grooups of bins of <b>delta</b> are mapped to bins of
/// <b>hist_delta</b>. The histogram contents are set to the sum over
/// the group of bins. The histogram errors are reset to zero.
/// <br/>
/// The array <b>binMap</b> is explained with the method GetOutput()
void TUnfoldSys::VectorMapToHist
(TH1 *hist_delta,const TMatrixDSparse *delta,const Int_t *binMap)
{
   // sum over bins of *delta, as defined in binMap,fXToHist
   //   hist_delta: histogram to return summed vector
   //   delta: vector to sum and remap
   Int_t nbin=hist_delta->GetNbinsX();
   Double_t *c=new Double_t[nbin+2];
   for(Int_t i=0;i<nbin+2;i++) {
      c[i]=0.0;
   }
   if(delta) {
      Int_t binMapSize = fHistToX.GetSize();
      const Double_t *delta_data=delta->GetMatrixArray();
      const Int_t *delta_rows=delta->GetRowIndexArray();
      for(Int_t i=0;i<binMapSize;i++) {
         Int_t destBinI=binMap ? binMap[i] : i;
         Int_t srcBinI=fHistToX[i];
         if((destBinI>=0)&&(destBinI<nbin+2)&&(srcBinI>=0)) {
            Int_t index=delta_rows[srcBinI];
            if(index<delta_rows[srcBinI+1]) {
               c[destBinI]+=delta_data[index];
            }
         }
      }
   }
   for(Int_t i=0;i<nbin+2;i++) {
      hist_delta->SetBinContent(i,c[i]);
      hist_delta->SetBinError(i,0.0);
   }
   delete[] c;
}

////////////////////////////////////////////////////////////////////////////////
/// Get a new list of all systematic uuncertainty sources.
///
/// The user is responsible for deleting the list
TSortedList *TUnfoldSys::GetSysSources(void) const {
   // get list of names of systematic sources
   TSortedList *r=new TSortedList();
   TMapIter i(fSysIn);
   for(const TObject *key=i.Next();key;key=i.Next()) {
      r->Add(((TObjString *)key)->Clone());
   }
   return r;
}

////////////////////////////////////////////////////////////////////////////////
/// Get a new list of all background sources.
///
/// The user is responsible for deleting the list
/// get list of name of background sources

TSortedList *TUnfoldSys::GetBgrSources(void) const {
   TSortedList *r=new TSortedList();
   TMapIter i(fBgrIn);
   for(const TObject *key=i.Next();key;key=i.Next()) {
      r->Add(((TObjString *)key)->Clone());
   }
   return r;
}
