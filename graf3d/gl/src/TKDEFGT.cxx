// @(#)root/gl:$Id$
// Author: Timur Pocheptsov  2009
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include <climits>

#include "TError.h"
#include "TMath.h"

#include "TKDEFGT.h"
#include "TGL5D.h"

//   Kernel density estimator based on Fast Gauss Transform.
//
//   Nice implementation of FGT by Sebastien Paris
//   can be found here:
//      http://www.mathworks.com/matlabcentral/fileexchange/17438
//
//   This version is based on his work.

////////////////////////////////////////////////////////////////////////////////
///Constructor.

TKDEFGT::TKDEFGT()
         : fDim(0),
           fP(0),
           fK(0),
           fSigma(1.),
           fPD(0),
           fModelValid(kFALSE)
{
}

namespace {

UInt_t NChooseK(UInt_t n, UInt_t k);

}

////////////////////////////////////////////////////////////////////////////////
///Destructor.

TKDEFGT::~TKDEFGT()
{
}

////////////////////////////////////////////////////////////////////////////////
///Calculate coefficients for FGT.

void TKDEFGT::BuildModel(const std::vector<Double_t> &sources, Double_t sigma,
                         UInt_t dim, UInt_t p, UInt_t k)
{
   if (!sources.size()) {
      Warning("TKDEFGT::BuildModel", "Bad input - zero size vector");
      return;
   }

   if (!dim) {
      Warning("TKDEFGT::BuildModel", "Number of dimensions is zero");
      return;
   }

   if (!p) {
      Warning("TKDEFGT::BuildModel", "Order of truncation is zero, 8 will be used");
      p = 8;
   }

   fDim = dim;
   fP = p;
   const UInt_t nP = UInt_t(sources.size()) / fDim;
   fK = !k ? UInt_t(TMath::Sqrt(Double_t(nP))) : k;
   fSigma = sigma;
   fPD = NChooseK(fP + fDim - 1, fDim);

   fWeights.assign(nP, 1.);
   fXC.assign(fDim * fK, 0.);
   fA_K.assign(fPD * fK, 0.);
   fIndxc.assign(fK, 0);
   fIndx.assign(nP, 0);
   fXhead.assign(fK, 0);
   fXboxsz.assign(fK, 0);
   fDistC.assign(nP, 0.);
   fC_K.assign(fPD, 0.);
   fHeads.assign(fDim + 1, 0);
   fCinds.assign(fPD, 0);
   fDx.assign(fDim, 0.);
   fProds.assign(fPD, 0.);

   Kcenter(sources);
   Compute_C_k();
   Compute_A_k(sources);

   fModelValid = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
///Calculate coefficients for FGT.
///Alternative specialized version for data from TTree.

void TKDEFGT::BuildModel(const TGL5DDataSet *sources, Double_t sigma, UInt_t p, UInt_t k)
{
   if (!sources->SelectedSize()) {
      Warning("TKDEFGT::BuildModel", "Bad input - zero size vector");
      return;
   }

   fDim = 3;

   if (!p) {
      Warning("TKDEFGT::BuildModel", "Order of truncation is zero, 8 will be used");
      p = 8;
   }

   fP = p;
   const UInt_t nP = sources->SelectedSize();
   fK = !k ? UInt_t(TMath::Sqrt(Double_t(nP))) : k;
   fSigma = sigma;
   fPD = NChooseK(fP + fDim - 1, fDim);

   fWeights.assign(nP, 1.);
   fXC.assign(fDim * fK, 0.);
   fA_K.assign(fPD * fK, 0.);
   fIndxc.assign(fK, 0);
   fIndx.assign(nP, 0);
   fXhead.assign(fK, 0);
   fXboxsz.assign(fK, 0);
   fDistC.assign(nP, 0.);
   fC_K.assign(fPD, 0.);
   fHeads.assign(fDim + 1, 0);
   fCinds.assign(fPD, 0);
   fDx.assign(fDim, 0.);
   fProds.assign(fPD, 0.);

   Kcenter(sources);
   Compute_C_k();
   Compute_A_k(sources);

   fModelValid = kTRUE;
}

void BuildModel();

namespace {

Double_t DDist(const Double_t *x , const Double_t *y , Int_t d);
Double_t DDist(Double_t x1, Double_t y1, Double_t z1, Double_t x2, Double_t y2, Double_t z2);

UInt_t Idmax(const std::vector<Double_t> &x , UInt_t n);

}

////////////////////////////////////////////////////////////////////////////////
///Solve kcenter task.

void TKDEFGT::Kcenter(const std::vector<double> &x)
{
   //Randomly pick one node as the first center.
   const UInt_t nP = UInt_t(x.size()) / fDim;

   UInt_t *indxc = &fIndxc[0];
   UInt_t ind = 1;
   *indxc++ = ind;

   const Double_t *x_j   = &x[0];
   const Double_t *x_ind = &x[0] + ind * fDim;

   for (UInt_t j = 0; j < nP; x_j += fDim, ++j) {
      fDistC[j] = (j == ind) ? 0. : DDist(x_j , x_ind , fDim);
      fIndx[j] = 0;
   }

   for (UInt_t i = 1 ; i < fK; ++i) {
      ind = Idmax(fDistC, nP);
      *indxc++ = ind;
      x_j      = &x[0];
      x_ind    = &x[0] + ind * fDim;
      for (UInt_t j = 0; j < nP; x_j += fDim, ++j) {
         const Double_t temp = (j==ind) ? 0.0 : DDist(x_j, x_ind, fDim);
         if (temp < fDistC[j]) {
            fDistC[j] = temp;
            fIndx[j]   = i;
         }
      }
   }

   for (UInt_t i = 0, nd = 0 ; i < nP; i++, nd += fDim) {
      fXboxsz[fIndx[i]]++;
      Int_t ibase = fIndx[i] * fDim;
      for (UInt_t j = 0 ; j < fDim; ++j)
         fXC[j + ibase] += x[j + nd];
   }

   for (UInt_t i = 0, ibase = 0; i < fK ; ++i, ibase += fDim) {
      const Double_t temp = 1. / fXboxsz[i];
      for (UInt_t j = 0; j < fDim; ++j)
         fXC[j + ibase] *= temp;
   }
}

////////////////////////////////////////////////////////////////////////////////
///Solve kcenter task. Version for dim == 3 and data from TTree.
///Randomly pick one node as the first center.

void TKDEFGT::Kcenter(const TGL5DDataSet *sources)
{
   const UInt_t nP = sources->SelectedSize();

   UInt_t *indxc = &fIndxc[0];
   *indxc++ = 1;

   {
   //Block to limit the scope of x_ind etc.
   const Double_t x_ind = sources->V1(1);
   const Double_t y_ind = sources->V2(1);
   const Double_t z_ind = sources->V3(1);

   for (UInt_t j = 0; j < nP; ++j) {
      const Double_t x_j = sources->V1(j);
      const Double_t y_j = sources->V2(j);
      const Double_t z_j = sources->V3(j);
      fDistC[j] = (j == 1) ? 0. : DDist(x_j, y_j, z_j, x_ind, y_ind, z_ind);
      fIndx[j] = 0;
   }
   //Block to limit the scope of x_ind etc.
   }

   for (UInt_t i = 1 ; i < fK; ++i) {
      const UInt_t ind = Idmax(fDistC, nP);
      const Double_t x_ind = sources->V1(ind);
      const Double_t y_ind = sources->V2(ind);
      const Double_t z_ind = sources->V3(ind);

      *indxc++ = ind;
      for (UInt_t j = 0; j < nP; ++j) {
         const Double_t x_j = sources->V1(j);
         const Double_t y_j = sources->V2(j);
         const Double_t z_j = sources->V3(j);

         const Double_t temp = (j==ind) ? 0.0 : DDist(x_j, y_j, z_j, x_ind, y_ind, z_ind);
         if (temp < fDistC[j]) {
            fDistC[j] = temp;
            fIndx[j]   = i;
         }
      }
   }

   for (UInt_t i = 0, nd = 0 ; i < nP; i++, nd += fDim) {
      fXboxsz[fIndx[i]]++;
      UInt_t ibase = fIndx[i] * fDim;
      fXC[ibase]     += sources->V1(i);
      fXC[ibase + 1] += sources->V2(i);
      fXC[ibase + 2] += sources->V3(i);
   }

   for (UInt_t i = 0, ibase = 0; i < fK ; ++i, ibase += fDim) {
      const Double_t temp = 1. / fXboxsz[i];
      for (UInt_t j = 0; j < fDim; ++j)
         fXC[j + ibase] *= temp;
   }
}

////////////////////////////////////////////////////////////////////////////////
///Coefficients C_K.

void TKDEFGT::Compute_C_k()
{
   fHeads[fDim] = UINT_MAX;
   fCinds[0] = 0;
   fC_K[0] = 1.0;

   for (UInt_t k = 1, t = 1, tail = 1; k < fP; ++k, tail = t) {
      for (UInt_t i = 0; i < fDim; ++i) {
         const UInt_t head = fHeads[i];
         fHeads[i] = t;
         for (UInt_t j = head; j < tail; ++j, ++t) {
            fCinds[t] = (j < fHeads[i + 1]) ? fCinds[j] + 1 : 1;
            fC_K[t]   = 2.0 * fC_K[j];
            fC_K[t]  /= fCinds[t];
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
///Coefficients A_K.

void TKDEFGT::Compute_A_k(const std::vector<Double_t> &x)
{
   const Double_t ctesigma = 1. / fSigma;
   const UInt_t nP = UInt_t(x.size()) / fDim;

   for (UInt_t n = 0; n < nP; n++) {
      UInt_t nbase    = n * fDim;
      UInt_t ix2c     = fIndx[n];
      UInt_t ix2cbase = ix2c * fDim;
      UInt_t ind      = ix2c * fPD;
      Double_t temp   = fWeights[n];
      Double_t sum    = 0.0;

      for (UInt_t i = 0; i < fDim; ++i) {
         fDx[i]    = (x[i + nbase] - fXC[i + ix2cbase]) * ctesigma;
         sum      += fDx[i] * fDx[i];
         fHeads[i] = 0;
      }

      fProds[0] = TMath::Exp(-sum);

      for (UInt_t k = 1, t = 1, tail = 1; k < fP; ++k, tail = t) {
         for (UInt_t i = 0; i < fDim; ++i) {
            const UInt_t head = fHeads[i];
            fHeads[i] = t;
            const Double_t temp1 = fDx[i];
            for (UInt_t j = head; j < tail; ++j, ++t)
               fProds[t] = temp1 * fProds[j];
         }
      }

      for (UInt_t i = 0 ; i < fPD ; i++)
         fA_K[i + ind] += temp * fProds[i];
   }

   for (UInt_t k = 0; k < fK; ++k) {
      const UInt_t ind = k * fPD;
      for (UInt_t i = 0; i < fPD; ++i)
         fA_K[i + ind] *= fC_K[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///Coefficients A_K. Special case for TTree and dim == 3.

void TKDEFGT::Compute_A_k(const TGL5DDataSet *sources)
{
   const Double_t ctesigma = 1. / fSigma;
   const UInt_t nP = sources->SelectedSize();

   for (UInt_t n = 0; n < nP; n++) {
      UInt_t ix2c     = fIndx[n];
      UInt_t ix2cbase = ix2c * 3;
      UInt_t ind      = ix2c * fPD;
      Double_t temp   = fWeights[n];
      Double_t sum    = 0.0;

      fDx[0] = (sources->V1(n) - fXC[ix2cbase]) * ctesigma;
      fDx[1] = (sources->V2(n) - fXC[ix2cbase + 1]) * ctesigma;
      fDx[2] = (sources->V3(n) - fXC[ix2cbase + 2]) * ctesigma;

      sum += (fDx[0] * fDx[0] + fDx[1] * fDx[1] + fDx[2] * fDx[2]);
      fHeads[0] = fHeads[1] = fHeads[2] = 0;

      fProds[0] = TMath::Exp(-sum);

      for (UInt_t k = 1, t = 1, tail = 1; k < fP; ++k, tail = t) {
         for (UInt_t i = 0; i < 3; ++i) {
            const UInt_t head = fHeads[i];
            fHeads[i] = t;
            const Double_t temp1 = fDx[i];
            for (UInt_t j = head; j < tail; ++j, ++t)
               fProds[t] = temp1 * fProds[j];
         }
      }

      for (UInt_t i = 0 ; i < fPD ; i++)
         fA_K[i + ind] += temp * fProds[i];
   }

   for (UInt_t k = 0; k < fK; ++k) {
      const Int_t ind = k * fPD;
      for (UInt_t i = 0; i < fPD; ++i)
         fA_K[i + ind] *= fC_K[i];
   }
}

namespace {

UInt_t InvNChooseK(UInt_t d, UInt_t cnk);

}

////////////////////////////////////////////////////////////////////////////////
///Calculate densities.

void TKDEFGT::Predict(const std::vector<Double_t> &ts, std::vector<Double_t> &v, Double_t eval)const
{
   if (!fModelValid) {
      Error("TKDEFGT::Predict", "Call BuildModel first!");
      return;
   }

   if (!ts.size()) {
      Warning("TKDEFGT::Predict", "Empty targets vector.");
      return;
   }

   v.assign(ts.size() / fDim, 0.);

   fHeads.assign(fDim + 1, 0);
   fDx.assign(fDim, 0.);
   fProds.assign(fPD, 0.);

   const Double_t ctesigma = 1. / fSigma;
   const UInt_t p  = InvNChooseK(fDim , fPD);
   const UInt_t nP = UInt_t(ts.size()) / fDim;

   for (UInt_t m = 0; m < nP; ++m) {
      Double_t temp = 0.;
      const UInt_t mbase = m * fDim;

      for (UInt_t kn = 0 ; kn < fK ; ++kn) {
         const UInt_t xbase = kn * fDim;
         const UInt_t ind = kn * fPD;
         Double_t sum2  = 0.0;
         for (UInt_t i = 0; i < fDim ; ++i) {
            fDx[i] = (ts[i + mbase] - fXC[i + xbase]) * ctesigma;
            sum2 += fDx[i] * fDx[i];
            fHeads[i] = 0;
         }

         if (sum2 > eval) continue; //skip to next kn

         fProds[0] = TMath::Exp(-sum2);

         for (UInt_t k = 1, t = 1, tail = 1; k < p; ++k, tail = t) {
            for (UInt_t i = 0; i < fDim; ++i) {
               UInt_t head = fHeads[i];
               fHeads[i] = t;
               const Double_t temp1 = fDx[i];
               for (UInt_t j = head; j < tail; ++j, ++t)
                  fProds[t] = temp1 * fProds[j];
            }
         }

         for (UInt_t i = 0 ; i < fPD ; ++i)
            temp += fA_K[i + ind] * fProds[i];
      }

      v[m] = temp;
   }

   Double_t dMin = v[0], dMax = dMin;
   for (UInt_t i = 1; i < nP; ++i) {
      dMin = TMath::Min(dMin, v[i]);
      dMax = TMath::Max(dMax, v[i]);
   }

   const Double_t dRange = dMax - dMin;
   for (UInt_t i = 0; i < nP; ++i)
      v[i] = (v[i] - dMin) / dRange;

   dMin = v[0], dMax = dMin;
   for (UInt_t i = 1; i < nP; ++i) {
      dMin = TMath::Min(dMin, v[i]);
      dMax = TMath::Max(dMax, v[i]);
   }
}

namespace {

////////////////////////////////////////////////////////////////////////////////
///n is always >= k.

UInt_t NChooseK(UInt_t n, UInt_t k)
{
   UInt_t n_k = n - k;
   if (k < n_k) {
      k = n_k;
      n_k = n - k;
   }
   UInt_t nchsk = 1;
   for (UInt_t i = 1; i <= n_k; ++i) {
      nchsk *= ++k;
      nchsk /= i;
   }

   return nchsk;
}

////////////////////////////////////////////////////////////////////////////////

Double_t DDist(Double_t x1, Double_t y1, Double_t z1, Double_t x2, Double_t y2, Double_t z2)
{
   return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
}

////////////////////////////////////////////////////////////////////////////////

Double_t DDist(const Double_t *x , const Double_t *y , Int_t d)
{
   Double_t t = 0., s = 0.;

   for (Int_t i = 0 ; i < d ; i++) {
      t  = (x[i] - y[i]);
      s += (t * t);
   }

   return s;
}

////////////////////////////////////////////////////////////////////////////////

UInt_t Idmax(const std::vector<Double_t> &x , UInt_t n)
{
   UInt_t k = 0;
   Double_t t = -1.0;
   for (UInt_t i = 0; i < n; ++i) {
      if (t < x[i]) {
         t = x[i];
         k = i;
      }
   }

   return k;
}

////////////////////////////////////////////////////////////////////////////////

UInt_t InvNChooseK(UInt_t d, UInt_t cnk)
{
   UInt_t cted = 1;
   for (UInt_t i = 2 ; i <= d ; ++i)
      cted *= i;

   const UInt_t cte = cnk * cted;
   UInt_t p = 2, ctep = 2;

   for (UInt_t i = p + 1; i < p + d; ++i)
      ctep *= i;

   while (ctep != cte) {
      ctep = ((p + d) * ctep) / p;
      ++p;
   }

   return p;
}

}//anonymous namespace.
