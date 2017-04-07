// @(#)root/hist:$Id$
// Author: L. Moneta May 2015

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015  ROOT  Team, CERN/PH-SFT                        *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
//
//  TF1NormSum.h
//  
//
//  Created by Aurélie Flandi on 27.08.14.
//
//

#ifndef ROOT_TF1NormSum__
#define ROOT_TF1NormSum__

#include <iostream>
#include "TF1.h"
#include <memory>

//class adding two functions: c1*f1+c2*f2
class TF1NormSum {
   
protected:

   // vector market as temp data member could probably be removed 
   
   unsigned int fNOfFunctions;               //! Number of functions to add
   //int> p1(new int(5));
   Double_t     fScale;                      // Fixed Scale parameter to normalize function (e.g. bin width)
   std::vector < std::shared_ptr < TF1 > > fFunctions;     // Vector of size fNOfFunctions containing TF1 functions
   std::vector < Double_t  > fCoeffs;        // Vector of size afNOfFunctions containing coefficients in front of each function
//   std::vector < Int_t     > fNOfParams;     //! Vector of size fNOfFunctions containing number of parameters for each function (does not contain the coefficients!)
//   std::vector < Int_t     > fNOfNonCstParams;  // Vector of  size fNOfFunctions containing the indices of the non-const parameter of each function
//   std::vector < Double_t *  > fParams;     //! Vector of size [fNOfFunctions][Function_p] containing an array of all function parameters
                                             // (non including coefficients) for each function
   std::vector < Int_t     > fCstIndexes;    // vector with size of fNOfFunctions containing the index of the constant parameter/ function (the removed ones) 
   std::vector< TString >    fParNames;      // parameter names 
   
   void InitializeDataMembers(const std::vector <std::shared_ptr < TF1 >> &functions, const std::vector <Double_t> &coeffs, Double_t scale); // acts as a constrcutor
   
   //smart pointer
public:
   
   TF1NormSum();
   TF1NormSum(const std::vector <TF1*>&functions, const std::vector <Double_t> &coeffs, Double_t scale = 1.);
   TF1NormSum(TF1* function1, TF1* function2, Double_t coeff1 = 1., Double_t coeff2 = 1., Double_t scale = 1.);
   TF1NormSum(TF1* function1, TF1* function2, TF1*function3, Double_t coeff1 = 1., Double_t coeff2 = 1., Double_t coeff3 = 1., Double_t scale = 1.);
   TF1NormSum(const TString &formula, Double_t xmin, Double_t xmax);
   
   double  operator()(double* x, double* p);

   std::vector<double> GetParameters() const;

   void      SetScale(Double_t scale) { fScale = scale; };
   
   void      SetParameters(const double* params);
   
   void      SetParameters(Double_t p0, Double_t p1, Double_t p2=0., Double_t p3=0., Double_t p4=0.,
                                   Double_t p5=0., Double_t p6=0., Double_t p7=0., Double_t p8=0., Double_t p9=0., Double_t p10=0.);
   
   Int_t             GetNpar() const;
   
   Double_t             GetScale() const { return fScale; }

   const char *      GetParName(Int_t ipar) const { return fParNames.at(ipar).Data(); }
   
   //ClassDef(TF1NormSum,1)
   
};
#endif /* defined(ROOT_TF1NormSum__) */
