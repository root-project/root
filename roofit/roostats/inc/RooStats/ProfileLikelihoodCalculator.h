// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ProfileLikelihoodCalculator
#define ROOSTATS_ProfileLikelihoodCalculator

#ifndef ROOSTATS_CombinedCalculator
#include "RooStats/CombinedCalculator.h"
#endif

#include "RooStats/LikelihoodInterval.h"




namespace RooStats {
   
   class LikelihoodInterval;


   /**

ProfileLikelihoodCalculator is a concrete implementation of CombinedCalculator (the interface class for a tools which can produce both RooStats HypoTestResults and ConfIntervals).
The tool uses the profile likelihood ratio as a test statistic, and assumes that Wilks' theorem is valid. Wilks' theorem states that -2* log (profile likelihood ratio) is asymptotically distributed as a chi^2 distribution with N-dof, where N is the number of degrees of freedom. Thus, p-values can be constructed and the profile likelihood ratio can be used to construct a LikelihoodInterval. (In the future, this class could be extended to use toy Monte Carlo to calibrate the distribution of the test statistic).

Usage: It uses the interface of the CombinedCalculator, so that it can be configured by specifying:

*   a model common model (eg. a family of specific models which includes both the null and alternate),
*   a data set,
*   a set of parameters of interest. The nuisance parameters will be all other parameters of the model
*   a set of parameters of which specify the null hypothesis (including values and const/non-const status)

The interface allows one to pass the model, data, and parameters either directly or via a ModelConfig class. The alternate hypothesis leaves the parameter free to take any value other than those specified by the null hypotesis. There is therefore no need to specify the alternate parameters.

After configuring the calculator, one only needs to ask GetHypoTest() (which will return a HypoTestResult pointer) or GetInterval() (which will return an ConfInterval pointer).

   This calculator can work with both one-dimensional intervals or multi-dimensional ones (contours)

   Note that for hypothesis tests, it is oftern better to use the RooStats::AsymptoricCalculator class, which can compute in addition the expected p-value using an Asimov data set. 


   \ingroup Roostats
 */

 class ProfileLikelihoodCalculator : public CombinedCalculator {

   public:

      /// Default constructor (needed for I/O)
      ProfileLikelihoodCalculator();

      /// Constructor from data, from a full model pdf describing both parameter of interest and nuisance parameters 
      /// and from the set specifying the parameter of interest (POI).
      /// There is no need to specify the nuisance parameters since they are all other parameters of the model. 
      /// When using the calculator for performing an hypothesis test one needs to provide also a snapshot (a copy) 
      /// defining the null parameters and their value. There is no need to pass the alternate parameters. These  
      /// will be obtained by the value maximazing the likelihood function
      ProfileLikelihoodCalculator(RooAbsData& data, RooAbsPdf& pdf, const RooArgSet& paramsOfInterest, 
                                  Double_t size = 0.05, const RooArgSet* nullParams = 0 );


      /// Constructor from data and a model configuration
      /// If the ModelConfig defines a prior pdf for any of the parameters those will be included as constrained terms in the 
      /// likelihood function 
      ProfileLikelihoodCalculator(RooAbsData& data, ModelConfig & model, Double_t size = 0.05);


      virtual ~ProfileLikelihoodCalculator();
    
      /// Return a likelihood interval. A global fit to the likelihood is performed and 
      /// the interval is constructed using the the profile likelihood ratio function of the POI
      virtual LikelihoodInterval* GetInterval() const ; 

      /// Return the hypothesis test result obtained from the likelihood ratio of the 
      /// maximum likelihood value with the null parameters fixed to their values, with respect keeping all parameters 
      /// floating (global maximum likelihood value). 
      virtual HypoTestResult* GetHypoTest() const;   
    
      

   protected:

    // clear internal fit result
    void DoReset() const; 
    
    // perform a global fit 
    RooAbsReal * DoGlobalFit() const;
    
    // minimize likelihood
    static RooFitResult * DoMinimizeNLL(RooAbsReal * nll); 

    
    mutable RooFitResult * fFitResult;  // internal  result of gloabl fit
    mutable bool fGlobalFitDone;          // flag to control if a global fit has been done
    

    ClassDef(ProfileLikelihoodCalculator,2) // A concrete implementation of CombinedCalculator that uses the ProfileLikelihood ratio.

   };
}
#endif
