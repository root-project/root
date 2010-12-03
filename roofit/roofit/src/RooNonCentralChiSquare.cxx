/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id: RooNonCentralChiSquare                               *
 * Authors:                                                                  *
 *   Kyle Cranmer
 *                                                                           *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// The PDF of the Non-Central Chi Square distribution for n degrees of freedom.  
// It is the asymptotic distribution of the profile likeihood ratio test q_mu 
// when a different mu' is true.  It is Wald's generalization of Wilks' Theorem.
//
// See:
//  Asymptotic formulae for likelihood-based tests of new physics
//     By Glen Cowan, Kyle Cranmer, Eilam Gross, Ofer Vitells
//     http://arXiv.org/abs/arXiv:1007.1727
//
//  Wikipedia:
//    http://en.wikipedia.org/wiki/Noncentral_chi-square_distribution#Approximation
//
// It requires MathMore to evaluate for non-integer degrees of freedom, k.
//
// When the Mathmore library is available we can use the MathMore libraries impelmented using GSL. 
// It makes use of the modified Bessel function of the first kind (for k > 2). For k < 2 it uses 
// the hypergeometric function 0F1. 
// When is not available we use explicit summation of normal chi-squared distributions
// The usage of the sum can be forced by calling SetForceSum(true);
//
// This implementation could be improved.  BOOST has a nice implementation:
// http://live.boost.org/doc/libs/1_42_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/dists/nc_chi_squared_dist.html
// http://wesnoth.repositoryhosting.com/trac/wesnoth_wesnoth/browser/trunk/include/boost/math/distributions/non_central_chi_squared.hpp?rev=6
// END_HTML
//

#include "Riostream.h" 

#include "RooNonCentralChiSquare.h" 
#include "RooAbsReal.h" 
#include "RooAbsCategory.h" 
#include <math.h> 
#include "TMath.h" 
//#include "RooNumber.h"
#include "Math/DistFunc.h"


#include "RooMsgService.h"

ClassImp(RooNonCentralChiSquare) 

RooNonCentralChiSquare::RooNonCentralChiSquare(const char *name, const char *title, 
                                               RooAbsReal& _x,
                                               RooAbsReal& _k,
                                               RooAbsReal& _lambda) :
   RooAbsPdf(name,title), 
   x("x","x",this,_x),
   k("k","k",this,_k),
   lambda("lambda","lambda",this,_lambda),
   fErrorTol(1E-3),
   fMaxIters(10),
   fHasIssuedConvWarning(false),
   fHasIssuedSumWarning(false)
{ 
#ifdef R__HAS_MATHMORE  
   ccoutD(InputArguments) << "RooNonCentralChiSquare::ctor(" << GetName() << 
      "MathMore Available, will use Bessel function expressions unless SetForceSum(true) "<< endl ;
   fForceSum = false;
#else 
   fForceSum = true;
#endif
} 

RooNonCentralChiSquare::RooNonCentralChiSquare(const RooNonCentralChiSquare& other, const char* name) :  
   RooAbsPdf(other,name), 
   x("x",this,other.x),
   k("k",this,other.k),
   lambda("lambda",this,other.lambda),
   fErrorTol(other.fErrorTol),
   fMaxIters(other.fMaxIters),
   fHasIssuedConvWarning(false),
   fHasIssuedSumWarning(false)
{ 
#ifdef R__HAS_MATHMORE  
   ccoutD(InputArguments) << "RooNonCentralChiSquare::ctor(" << GetName() << 
     "MathMore Available, will use Bessel function expressions unless SetForceSum(true) "<< endl ;
   fForceSum = other.fForceSum;
#else 
   fForceSum = true;
#endif
} 


void RooNonCentralChiSquare::SetForceSum(Bool_t flag) { 
   fForceSum = flag; 
#ifndef R__HAS_MATHMORE
   if (!fForceSum) { 
      ccoutD(InputArguments) << "RooNonCentralChiSquare::SetForceSum" << GetName() << 
         "MathMore is not available- ForceSum must be on "<< endl ;
      fForceSum = true; 
   }
#endif

}


Double_t RooNonCentralChiSquare::evaluate() const 
{ 
   // ENTER EXPRESSION IN TERMS OF VARIABLE ARGUMENTS HERE 


   // chi^2(0,k) gives inf and causes various problems
   // truncate
   Double_t xmin = x.min(); 
   Double_t xmax = x.max();
   double _x = x;
   if(_x<=0){
     // options for dealing with this
     //     return 0; // gives a funny dip
     //     _x = 1./RooNumber::infinity(); // too tall
     _x = xmin + 1e-3*(xmax-xmin); // very small fraction of range
   }

   // special case (form below doesn't work when lambda==0)
   if(lambda==0){
      return ROOT::Math::chisquared_pdf(_x,k);
   }

   // three forms
   // FIRST FORM
   // \sum_i=0^\infty exp(-lambda/2) (\lamda/2)^i chi2(x,k+2i) / i!
   // could truncate sum

   if ( fForceSum  ){
      if(!fHasIssuedSumWarning){
         coutI(InputArguments) << "RooNonCentralChiSquare sum being forced" << endl ;
         fHasIssuedSumWarning=true;
      }
      double sum = 0;
      double ithTerm = 0;
      double errorTol = fErrorTol;
      int MaxIters = fMaxIters;
      int iDominant = (int) TMath::Floor(lambda/2);     
      //     cout <<"iDominant: " << iDominant << endl;
    
      // do 0th term last
      //     if(iDominant==0) iDominant = 1;
      for(int i = iDominant; ; ++i){
         ithTerm =exp(-lambda/2.)*pow(lambda/2.,i)*ROOT::Math::chisquared_pdf(_x,k+2*i)/TMath::Gamma(i+1);
         sum+=ithTerm;
         //       cout <<"progress: " << i << " " << ithTerm/sum << endl;
         if(ithTerm/sum < errorTol)
            break;

         if( i>iDominant+MaxIters){
            if(!fHasIssuedConvWarning){
               fHasIssuedConvWarning=true;
               coutW(Eval) << "RooNonCentralChiSquare did not converge: for x=" << x <<" k="<<k
                           << ", lambda="<<lambda << " fractional error = " << ithTerm/sum 
                           << "\n either adjust tolerance with SetErrorTolerance(tol) or max_iter with SetMaxIter(max_it)"  
                           << endl;
            }
            break;
         }
      }

      for(int i = iDominant - 1; i >= 0; --i){
         //       cout <<"Progress: " << i << " " << ithTerm/sum << endl;
         ithTerm =exp(-lambda/2.)*pow(lambda/2.,i)*ROOT::Math::chisquared_pdf(_x,k+2*i)/TMath::Gamma(i+1);
         sum+=ithTerm;
      }


      return sum;
   }

   // SECOND FORM (use MathMore function based on Bessel function (if k>2) or 
   // or  regularized confluent hypergeometric limit function.
#ifdef R__HAS_MATHMORE
   return  ROOT::Math::noncentral_chisquared_pdf(_x,k,lambda);
#else 
   coutF(Eval) << "RooNonCentralChisquare: ForceSum must be set" << endl;
   return 0; 
#endif

} 

//_____________________________________________________________________________
Int_t RooNonCentralChiSquare::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const 
{
   if (matchArgs(allVars,analVars,x)) return 1 ;
   return 0 ;
}



//_____________________________________________________________________________
Double_t RooNonCentralChiSquare::analyticalIntegral(Int_t code, const char* rangeName) const 
{
   assert(code==1 );
   //  cout << "evaluating analytic integral" << endl;
   Double_t xmin = x.min(rangeName); 
   Double_t xmax = x.max(rangeName);

   // if xmin~0 and xmax big, then can return 1. b/c evaluate is normalized.  

   // special case (form below doesn't work when lambda==0)
   if(lambda==0){
      return (ROOT::Math::chisquared_cdf(xmax,k) - ROOT::Math::chisquared_cdf(xmin,k));
   }

   // three forms
   // FIRST FORM
   // \sum_i=0^\infty exp(-lambda/2) (\lamda/2)^i chi2(x,k+2i) / i!
   // could truncate sum

  
   double sum = 0;
   double ithTerm = 0;
   double errorTol = fErrorTol; // for nomralization allow slightly larger error
   int MaxIters = fMaxIters; // for normalization use more terms

   int iDominant = (int) TMath::Floor(lambda/2);     
   //     cout <<"iDominant: " << iDominant << endl;
   //   iDominant=0;
   for(int i = iDominant; ; ++i){
      ithTerm =exp(-lambda/2.)*pow(lambda/2.,i)
         *( ROOT::Math::chisquared_cdf(xmax,k+2*i)/TMath::Gamma(i+1)
            - ROOT::Math::chisquared_cdf(xmin,k+2*i)/TMath::Gamma(i+1) );
      sum+=ithTerm;
      //     cout <<"progress: " << i << " " << ithTerm << " " << sum << endl;
      if(ithTerm/sum < errorTol)
         break;
     
      if( i>iDominant+MaxIters){
         if(!fHasIssuedConvWarning){
            fHasIssuedConvWarning=true;
            coutW(Eval) << "RooNonCentralChiSquare Normalization did not converge: for k="<<k
                        << ", lambda="<<lambda << " fractional error = " << ithTerm/sum 
                        << "\n either adjust tolerance with SetErrorTolerance(tol) or max_iter with SetMaxIter(max_it)"  
                        << endl;
         }
         break;
      }
   }
   
   for(int i = iDominant - 1; i >= 0; --i){
      ithTerm =exp(-lambda/2.)*pow(lambda/2.,i)
         *( ROOT::Math::chisquared_cdf(xmax,k+2*i)/TMath::Gamma(i+1)
            -ROOT::Math::chisquared_cdf(xmin,k+2*i)/TMath::Gamma(i+1) );
      sum+=ithTerm;
   }
   return sum;
}





