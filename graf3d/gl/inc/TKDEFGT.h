// @(#)root/gl:$Id$
// Author: Timur Pocheptsov  2009
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TKDEFGT
#define ROOT_TKDEFGT

#include <vector>

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

//KDE - kenrel density estimator,
//based on fast Gauss transform.


class TKDEFGT {
private:
   //KDE-related stuff.   
   std::vector<Double_t>   fXC;     //Centers.
   std::vector<Double_t>   fWeights;//Weights.
   std::vector<Int_t>      fIndxc;  //Internal data.
   std::vector<Double_t>   fA_K;    //Polynomial coefficient (pd x K)
   std::vector<Int_t>      fIndx;   //Internal data.
   std::vector<Int_t>      fXhead;  //Internal data.
   std::vector<Int_t>      fXboxsz; //Internal data.
   std::vector<Double_t>   fDistC;  //Internal data.
   std::vector<Double_t>   fC_K;    //Internal data.
   std::vector<Int_t>      fCinds;  //Internal data.
   
   mutable std::vector<Int_t>      fHeads; //Internal data.
   mutable std::vector<Double_t>   fDx;    //Internal data.
   mutable std::vector<Double_t>   fProds; //Internal data.
   
   Int_t                   fDim;        //Number of dimensions.
   Int_t                   fP;          //Order of trancation.
   Int_t                   fK;          //Number of centers.
   Double_t                fSigma;      //Noise Standard deviation of the kernel (default sigma = 1)
   Int_t                   fPD;         //nchoosek(fP + fDim - 1, fDim); 
   Bool_t                  fModelValid; //Check, if coefficients are ok.
   Bool_t                  fVerbose;
   
public:
   TKDEFGT();
   
   virtual ~TKDEFGT();
   
   //Sources and targets must be a vector of packed points, if you have, say,
   //Dim == 3, vector will be [xyz|xyz|xyz|xyz|xyz].
   void BuildModel(const std::vector<Double_t> &sources, Double_t sigma = 1.,
                   Int_t dim = 3, Int_t p = 8, Int_t k = 0);
   void Predict(const std::vector<Double_t> &targets, std::vector<Double_t> &densities,
                Double_t e)const;
                
   void SetVerbose(Bool_t v) {fVerbose = v;}
private:
   void Kcenter(const std::vector<double> &x);
   void Compute_C_k();
   void Compute_A_k(const std::vector<Double_t> &x);

   TKDEFGT(const TKDEFGT &rhs);
   TKDEFGT &operator = (const TKDEFGT &rhs);
   
   ClassDef(TKDEFGT, 0)//FGT based kernel density estimator.
};

#endif
