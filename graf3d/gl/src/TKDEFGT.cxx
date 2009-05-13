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

//   Kernel density estimator based on Fast Gauss Transform.
//   
//   Nice implementation of FGT by Sebastien Paris
//   can be found here:
//      http://www.mathworks.com/matlabcentral/fileexchange/17438
//      
//   This version is based on his work.

ClassImp(TKDEFGT)

//______________________________________________________________________________
TKDEFGT::TKDEFGT()
         : fDim(0),
           fP(0),
           fK(0),
           fSigma(1.),
           fPD(0),
           fModelValid(kFALSE),
           fVerbose(kTRUE)
{
   //Constructor.
}

namespace {
Int_t NChooseK(Int_t n, Int_t k);
}

//______________________________________________________________________________
TKDEFGT::~TKDEFGT()
{
   //Destructor.
}

//______________________________________________________________________________
void TKDEFGT::BuildModel(const std::vector<Double_t> &sources, Double_t sigma, Int_t dim, Int_t p, Int_t k)
{
   //Calculate coefficients for FGT.
   if (!sources.size()) {
      Warning("TKDEFGT::BuildModel", "Bad input - zero size vector");
      return;
   }

   if (dim <= 0) {
      Warning("TKDEFGT::BuildModel", "Bad number of dimensions: %d", dim);
      return;
   }

   if (p <= 0) {
      Warning("TKDEFGT::BuildModel", "Bad order of truncation: %d, 8 will be used", p);
      p = 8;
   }
   
   fDim = dim;
   fP = p;
   const Int_t nP = sources.size() / fDim;
   fK = k <= 0 ? Int_t(TMath::Sqrt(Double_t(nP))) : k;
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

   if (fVerbose)
      Info("TKDEFGT::BuildModel", "Initializing ...");

   Kcenter(sources);
   Compute_C_k();
   Compute_A_k(sources);

   if (fVerbose)
      Info("TKDEFGT::BuildModel", "Done.");

   fModelValid = kTRUE;
}

namespace {
Double_t DDist(const Double_t *x , const Double_t *y , Int_t d);
Int_t Idmax(const std::vector<Double_t> &x , Int_t n);
}

//______________________________________________________________________________
void TKDEFGT::Kcenter(const std::vector<double> &x)
{
   //Solve kcenter task.
   
   //Randomly pick one node as the first center.
   const Int_t nP = Int_t(x.size() / fDim);
   
   Int_t *indxc = &fIndxc[0];
   Int_t ind = 1;
   *indxc++ = ind;
   
   const Double_t *x_j   = &x[0];
   const Double_t *x_ind = &x[0] + ind * fDim;
   
   for (Int_t j = 0; j < nP; x_j += fDim, ++j) {
      fDistC[j] = (j == ind) ? 0. : DDist(x_j , x_ind , fDim);
      fIndx[j] = 0;
   }

   for (Int_t i = 1 ; i < fK; ++i) {
      ind = Idmax(fDistC, nP);
      *indxc++ = ind;
      x_j      = &x[0];
      x_ind    = &x[0] + ind * fDim;
      for (Int_t j = 0; j < nP; x_j += fDim, ++j) {
         const Double_t temp = (j==ind) ? 0.0 : DDist(x_j, x_ind, fDim);
         if (temp < fDistC[j]) {
            fDistC[j] = temp;
            fIndx[j]   = i;
         }
      }
   }
   
   for (Int_t i = 0, nd = 0 ; i < nP; i++, nd += fDim) {
      fXboxsz[fIndx[i]]++;
      Int_t ibase = fIndx[i] * fDim;
      for (Int_t j = 0 ; j < fDim; ++j)
         fXC[j + ibase] += x[j + nd];
   }
   
   for (Int_t i = 0, ibase = 0; i < fK ; ++i, ibase += fDim) {
      const Double_t temp = 1.0/fXboxsz[i];
      for (Int_t j = 0; j < fDim; j++)
         fXC[j + ibase] *= temp;
   }
}

//______________________________________________________________________________
void TKDEFGT::Compute_C_k()
{
   //Coefficients C_K.
   fHeads[fDim] = INT_MAX;
   fCinds[0] = 0;
   fC_K[0] = 1.0;
   
   for (Int_t k = 1, t = 1, tail = 1; k < fP; ++k, tail = t) {
      for (Int_t i = 0; i < fDim; ++i) {
         const Int_t head = fHeads[i];
         fHeads[i] = t;
         for (Int_t j = head; j < tail; ++j, ++t) {
            fCinds[t] = (j < fHeads[i + 1]) ? fCinds[j] + 1 : 1;
            fC_K[t]   = 2.0 * fC_K[j];
            fC_K[t]  /= fCinds[t];
         }
      }
   }
}

//______________________________________________________________________________
void TKDEFGT::Compute_A_k(const std::vector<Double_t> &x)
{
   //Coefficients A_K.
   const Double_t ctesigma = 1. / fSigma;
   const Int_t nP = Int_t(x.size() / fDim);
   
   for (Int_t n = 0; n < nP; n++) {
      Int_t nbase    = n * fDim;
      Int_t ix2c     = fIndx[n];
      Int_t ix2cbase = ix2c * fDim;
      Int_t ind      = ix2c * fPD;
      Double_t temp  = fWeights[n];
      Double_t sum   = 0.0;
      
      for (Int_t i = 0; i < fDim; ++i) {
         fDx[i]    = (x[i + nbase] - fXC[i + ix2cbase]) * ctesigma;
         sum      += fDx[i] * fDx[i];
         fHeads[i] = 0;
      }

      fProds[0] = TMath::Exp(-sum);

      for (Int_t k = 1, t = 1, tail = 1; k < fP; ++k, tail = t) {
         for (Int_t i = 0; i < fDim; ++i) {
            const Int_t head = fHeads[i];
            fHeads[i] = t;
            const Double_t temp1 = fDx[i];
            for (Int_t j = head; j < tail; ++j, ++t)
               fProds[t] = temp1 * fProds[j];
         }
      }

      for (Int_t i = 0 ; i < fPD ; i++)
         fA_K[i + ind] += temp * fProds[i];
   }

   for (Int_t k = 0; k < fK; ++k) {
      const Int_t ind = k * fPD;
      for (Int_t i = 0; i < fPD; ++i)
         fA_K[i + ind] *= fC_K[i];
   }
}

namespace {
Int_t InvNChooseK(Int_t d, Int_t cnk);
}

//______________________________________________________________________________
void TKDEFGT::Predict(const std::vector<Double_t> &ts, std::vector<Double_t> &v, Double_t eval)const
{
   //Calculate densities.
   
   if (!fModelValid) {
      Error("TKDEFGT::Predict", "Call BuildModel first!");
      return;
   }
   
   if (!ts.size()) {
      Warning("TKDEFGT::Predict", "Empty targets vector.");
      return;
   }
 
   if (fVerbose)  
      Info("TKDEFGT::Predict", "Estimation started ...");
   
   v.assign(ts.size() / fDim, 0.);
   
   fHeads.assign(fDim + 1, 0);
   fDx.assign(fDim, 0.);
   fProds.assign(fPD, 0.);
   
   const Double_t ctesigma = 1. / fSigma;
   const Int_t p = InvNChooseK(fDim , fPD);
   const Int_t nP = Int_t(ts.size() / fDim);
   
   for (Int_t m = 0; m < nP; ++m) {
      Double_t temp = 0.;
      const Int_t mbase = m * fDim;
      
      for (Int_t kn = 0 ; kn < fK ; ++kn) {
         const Int_t xbase = kn * fDim;
         const Int_t ind = kn * fPD;
         Double_t sum2  = 0.0;
         for (Int_t i = 0; i < fDim ; ++i) {
            fDx[i] = (ts[i + mbase] - fXC[i + xbase]) * ctesigma;
            sum2 += fDx[i] * fDx[i];
            fHeads[i] = 0;
         }

         if (sum2 > eval) continue; //skip to next kn

         fProds[0] = TMath::Exp(-sum2);

         for (Int_t k = 1, t = 1, tail = 1; k < p; ++k, tail = t) {
            for (Int_t i = 0; i < fDim; ++i) {
               Int_t head = fHeads[i];
               fHeads[i] = t;
               const Double_t temp1 = fDx[i];
               for (Int_t j = head; j < tail; ++j, ++t)
                  fProds[t] = temp1 * fProds[j];
            }
         }

         for (Int_t i = 0 ; i < fPD ; ++i)
            temp += fA_K[i + ind] * fProds[i];
      }

      v[m] = temp;
   }
   
   Double_t dMin = v[0], dMax = dMin;
   for (Int_t i = 1, e = Int_t(v.size()); i < e; ++i) {
      dMin = TMath::Min(dMin, v[i]);
      dMax = TMath::Max(dMax, v[i]);
   }
   
   const Double_t dRange = dMax - dMin;
   for (Int_t i = 0, e = Int_t(v.size()); i < e; ++i)
      v[i] = (v[i] - dMin) / dRange;

   dMin = v[0], dMax = dMin;
   for (Int_t i = 1, e = Int_t(v.size()); i < e; ++i) {
      dMin = TMath::Min(dMin, v[i]);
      dMax = TMath::Max(dMax, v[i]);
   }
}

namespace {
//______________________________________________________________________________
Int_t NChooseK(Int_t n, Int_t k)
{
   Int_t n_k = n - k;
   if (k < n_k) {
      k = n_k;
      n_k = n - k;
   }
   Int_t nchsk = 1;
   for (Int_t i = 1; i <= n_k; ++i) {
      nchsk *= (++k);
      nchsk /= i;
   }
   return nchsk;
}

//______________________________________________________________________________
Double_t DDist(const Double_t *x , const Double_t *y , Int_t d)
{
   Double_t t = 0., s = 0.;

   for (Int_t i = 0 ; i < d ; i++) {
      t  = (x[i] - y[i]);
      s += (t * t);
   }

   return s;
}

//______________________________________________________________________________
Int_t Idmax(const std::vector<Double_t> &x , Int_t n)
{
   Int_t k = 0;
   Double_t t = -1.0;
   for (Int_t i = 0; i < n; ++i) {
      if (t < x[i]) {
         t = x[i];
         k = i;
      }
   }
   return k;
}

//______________________________________________________________________________
Int_t InvNChooseK(Int_t d, Int_t cnk)
{
   Int_t cted = 1;
   for (Int_t i = 2 ; i <= d ; ++i)
      cted *= i;

   const Int_t cte = cnk * cted;
   Int_t p = 2, ctep = 2;

   for (Int_t i = p + 1; i < p + d; ++i)
      ctep *= i;

   while (ctep != cte) {
      ctep = ((p + d) * ctep) / p;
      ++p;
   }

   return p;
}
}//anonymous namespace.
