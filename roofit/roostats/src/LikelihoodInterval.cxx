// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*****************************************************************************
 * Project: RooStats
 * Package: RooFit/RooStats  
 * @(#)root/roofit/roostats:$Id$
 * Authors:                     
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
 *
 *****************************************************************************/



//_________________________________________________
/*
BEGIN_HTML
<p>
LikelihoodInterval is a concrete implementation of the RooStats::ConfInterval interface.  
It implements a connected N-dimensional intervals based on the contour of a likelihood ratio.
The boundary of the inteval is equivalent to a MINUIT/MINOS contour about the maximum likelihood estimator
[<a href="#minuit">1</a>].
The interval does not need to be an ellipse (eg. it is not the HESSE error matrix).
The level used to make the contour is the same as that used in MINOS, eg. it uses Wilks' theorem, 
which states that under certain regularity conditions the function -2* log (profile likelihood ratio) is asymptotically distributed as a chi^2 with N-dof, where 
N is the number of parameters of interest.  
</p>

<p>
Note, a boundary on the parameter space (eg. s>= 0) or a degeneracy (eg. mass of signal if Nsig = 0) can lead to violations of the conditions necessary for Wilks' theorem to be true.
</p>

<p>
Also note, one can use any RooAbsReal as the function that will be used in the contour; however, the level of the contour
is based on Wilks' theorem as stated above.
</p>

<P>References</P>

<p><A NAME="minuit">1</A>
F.&nbsp;James., Minuit.Long writeup D506, CERN, 1998.
</p>
  
END_HTML
*/
//
//

#ifndef RooStats_LikelihoodInterval
#include "RooStats/LikelihoodInterval.h"
#endif
#include "RooStats/RooStatsUtils.h"

#include "RooAbsReal.h"
#include "RooMsgService.h"

#include "Math/WrappedFunction.h"
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/MinimizerOptions.h"
#include "RooFunctor.h"
#include "RooProfileLL.h"

#include <string>

/*
// for debugging
#include "RooNLLVar.h"
#include "RooProfileLL.h"
#include "RooDataSet.h"
#include "RooAbsData.h"
*/

ClassImp(RooStats::LikelihoodInterval) ;

using namespace RooStats;


//____________________________________________________________________
LikelihoodInterval::LikelihoodInterval(const char* name) :
   ConfInterval(name), fBestFitParams(0), fLikelihoodRatio(0)
{
   // Default constructor with name and title
}

//____________________________________________________________________
LikelihoodInterval::LikelihoodInterval(const char* name, RooAbsReal* lr, const RooArgSet* params,  RooArgSet * bestParams) :
   ConfInterval(name), 
   fParameters(*params), 
   fBestFitParams(bestParams), 
   fLikelihoodRatio(lr) 
{
   // Alternate constructor taking a pointer to the profile likelihood ratio, parameter of interest and 
   // optionally a snaphot of best parameter of interest for interval
}


//____________________________________________________________________
LikelihoodInterval::~LikelihoodInterval()
{
   // Destructor
   if (fBestFitParams) delete fBestFitParams; 
   if (fLikelihoodRatio) delete fLikelihoodRatio;
}


//____________________________________________________________________
Bool_t LikelihoodInterval::IsInInterval(const RooArgSet &parameterPoint) const 
{  
  // This is the main method to satisfy the RooStats::ConfInterval interface.  
  // It returns true if the parameter point is in the interval.

  RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL) ;
  // Method to determine if a parameter point is in the interval
  if( !this->CheckParameters(parameterPoint) ) {
    std::cout << "parameters don't match" << std::endl;
    return false; 
  }

  // make sure likelihood ratio is set
  if(!fLikelihoodRatio) {
    std::cout << "likelihood ratio not set" << std::endl;
    return false;
  }

  

  // set parameters
  SetParameters(&parameterPoint, fLikelihoodRatio->getVariables() );


  // evaluate likelihood ratio, see if it's bigger than threshold
  if (fLikelihoodRatio->getVal()<0){
    std::cout << "The likelihood ratio is < 0, indicates a bad minimum or numerical precision problems.  Will return true" << std::endl;
    return true;
  }


  // here we use Wilks' theorem.
  if ( TMath::Prob( 2* fLikelihoodRatio->getVal(), parameterPoint.getSize()) < (1.-fConfidenceLevel) )
    return false;


  //RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;

  return true;
  
}

//____________________________________________________________________
RooArgSet* LikelihoodInterval::GetParameters() const
{  
  // returns list of parameters
   return new RooArgSet(fParameters); 
}

//____________________________________________________________________
Bool_t LikelihoodInterval::CheckParameters(const RooArgSet &parameterPoint) const
{  
  // check that the parameters are correct

  if (parameterPoint.getSize() != fParameters.getSize() ) {
    std::cout << "size is wrong, parameters don't match" << std::endl;
    return false;
  }
  if ( ! parameterPoint.equals( fParameters ) ) {
    std::cout << "size is ok, but parameters don't match" << std::endl;
    return false;
  }
  return true;
}



//____________________________________________________________________
Double_t LikelihoodInterval::LowerLimit(const RooRealVar& param) 
{  

   // compute upper limit, check first if limit has been computed 
   std::map<std::string, double>::const_iterator itr = fLowerLimits.find(param.GetName());
   if (itr != fLowerLimits.end() ) 
      return itr->second;

   // otherwise compute limit

   //RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING) ;

  RooAbsReal* newProfile = fLikelihoodRatio->createProfile(RooArgSet(param));

  RooArgSet* vars = newProfile->getVariables() ;
  RooRealVar* myarg = (RooRealVar *) vars->find(param.GetName());
  delete vars ;

  // I think here ndf must be 1 
  double target = TMath::ChisquareQuantile(fConfidenceLevel,1)/2.;

  //Double_t thisArgVal = param.getVal()*0.99;

  // initial point for the scan - 
  // start from  xmin - nsigma * error
  Double_t nsigma = target * 2; 
  RooRealVar * fitPar = (RooRealVar *) fBestFitParams->find(param.GetName() ); 
  Double_t thisArgVal = fitPar->getVal() - nsigma * fitPar->getError(); 

  myarg->setVal( thisArgVal);

  Double_t maxStep = (myarg->getMax()-myarg->getMin())/10 ;

  double step=-1;
  double L= newProfile->getVal();
  double L_old=0;
  double diff = L - target;
  //the parameters for the linear approximation
  double a=0, b=0;
  double x_app=0;
  int nIterations = 0, maxIterations = 100;
  while(fabs(diff)>0.01 && nIterations<maxIterations){

    nIterations++;
    L_old=L;
    thisArgVal=thisArgVal+step;
    if (thisArgVal<myarg->getMin())
      {
	thisArgVal=myarg->getMin(); 

	step=thisArgVal+step-x_app;
	if (fabs(step)<1E-5) {
          ccoutW(Eval) <<  "Lower limit is outside the parameters bounderies. Abort!" <<std::endl;
	  delete newProfile;
	  double ret = myarg->getMin();
	  return ret;
	}
      }
    
        
    myarg->setVal( thisArgVal );
    L=newProfile->getVal();

    // If L is below target and we are at boundary stop here
    if ((fabs(myarg->getVal()-myarg->getMin())<1e-5) && (L<target)) {
      ccoutW(Eval) <<"WARNING lower limit is outside the parameters bounderies (L at lower bound of " << myarg->GetName() << " is " << L 
	       << ", which is less than target value of " << target << "). Abort!"<<std::endl;
      delete newProfile;
      double ret = myarg->getMin();
      return ret;      
    }

    //Compute the linear function
    a=(L-L_old)/(step);
    if (fabs(a)<1E-3) {
       ccoutD(Eval) <<"WARNING: the slope of the Likelihood linear approximation is close to zero" <<std::endl;
    }
    b=L-a*thisArgVal;
    //approximate the position of the desired L value
    x_app=(target-b)/a;    
    step=x_app-thisArgVal;
    if (step>maxStep) step=maxStep ;
    if (step<-maxStep) step=-maxStep ;
    diff=L-target;

    if(a>0) {
      ccoutD(Eval) <<"WARNING: you are on the right of the MLE. Step towards MLE"<<std::endl;
      step=-(myarg->getMax()-myarg->getMin())/100; //Do a constant step, typical for the dimenions of the problem towards the minimum
     
      
    }
  }
    
  ccoutD(Eval) <<"LL search Iterations:"<< nIterations<<std::endl;
 
  //cout << "LL iterations " << nIterations << " value = " << myarg->getVal() << " PL = " << newProfile->getVal() << std::endl;

  delete newProfile;
  double ret=myarg->getVal();

  fLowerLimits[param.GetName()] = ret; 
  //RooMsgService::instance().setGlobalKillBelow(RooFit::INFO) ;

  return ret;
}



//____________________________________________________________________
Double_t LikelihoodInterval::UpperLimit(const RooRealVar& param) 
{  
   // compute upper limit, check first if limit has been computed 

   // otherwise compute limit using MINOS
   double lower = 0; 
   double upper = 0; 
   FindLimits(param, lower, upper); 
   return upper; 
}



void LikelihoodInterval::ResetLimits() { 
   // reset map with cached limits
   fLowerLimits.clear(); 
   fUpperLimits.clear(); 
}


bool LikelihoodInterval::CreateMinimizer() { 

   std::cout << "creating minimizer..........." << std::endl;

   // create minimizer object needed to find contours or interval limits
   // minimizer must be Minuit or Minuit2
   RooProfileLL * profilell = dynamic_cast<RooProfileLL*>(fLikelihoodRatio);
   if (!profilell) return false; 

   RooAbsReal & nll  = profilell->nll(); 
   // bind the nll function in the right interface for the Minimizer class 
   // as a function of only the parameters (poi + nuisance parameters) 

   RooArgSet * partmp = profilell->getVariables();
   RooArgList params(*partmp);
   delete partmp;

   // need to restore values and errors for POI
   if (fBestFitParams) { 
      for (int i = 0; i < params.getSize(); ++i) { 
         RooRealVar & par =  (RooRealVar &) params[i];
         RooRealVar * fitPar =  (RooRealVar *) (fBestFitParams->find(par.GetName() ) );
         if (fitPar) {
            par.setVal( fitPar->getVal() );
            par.setError( fitPar->getVal() );
         }
      }
   }

   // now do binding of NLL with a functor for Minimizer 
   fFunctor = std::auto_ptr<RooFunctor>(new RooFunctor(nll, RooArgSet(), params )); 

   std::string minimType =  ROOT::Math::MinimizerOptions::DefaultMinimizerType();

   if (minimType != "Minuit" && minimType != "Minuit2") { 
      ccoutE(InputArguments) << minimType << "is wrong type of minimizer for getting interval limits ir contours - must use Minuit or Minuit2" << std::endl;
      return false; 
   }
   // create minimizer class 
   fMinimizer = std::auto_ptr<ROOT::Math::Minimizer>(ROOT::Math::Factory::CreateMinimizer(minimType, "Migrad"));

   if (!fMinimizer.get()) return false;

   fMinFunc = std::auto_ptr<ROOT::Math::IMultiGenFunction>( new ROOT::Math::WrappedMultiFunction<RooFunctor &> (*fFunctor, fFunctor->nPar() ) );  
   fMinimizer->SetFunction(*fMinFunc); 

   // set minimizer parameters 
   assert( params.getSize() == int(fMinFunc->NDim()) ); 

   for (unsigned int i = 0; i < fMinFunc->NDim(); ++i) { 
      RooRealVar & v = (RooRealVar &) params[i]; 
      fMinimizer->SetLimitedVariable( i, v.GetName(), v.getVal(), v.getError(), v.getMin(), v.getMax() ); 
   }
   // for finding the contour need to find first global minimum
   bool iret = fMinimizer->Minimize();
   if (!iret || fMinimizer->X() == 0) { 
      ccoutE(Minimization) << "Error: Minimization failed  " << std::endl;
      return false; 
   }

   std::cout << "print minimizer result..........." << std::endl;

   fMinimizer->PrintResults();
   return true; 
}

bool LikelihoodInterval::FindLimits(const RooRealVar & param, double &lower, double & upper) 
{
   // find both lower and upper limits using MINOS

   // check first if cached values exist (limits have been already found) and return them in that case
   std::map<std::string, double>::const_iterator itrl = fLowerLimits.find(param.GetName());
   std::map<std::string, double>::const_iterator itru = fUpperLimits.find(param.GetName());
   if ( itrl != fLowerLimits.end() && itru != fUpperLimits.end() ) { 
      lower = itrl->second;
      upper = itru->second; 
      return true; 
   }
      

   RooArgSet * partmp = fLikelihoodRatio->getVariables();
   RooArgList params(*partmp);
   delete partmp;
   int ix = params.index(&param); 
   if (ix < 0 ) { 
      ccoutE(InputArguments) << "Error - invalid parameter " << param.GetName() << " specified for finding the interval limits " << std::endl;
      return false; 
   }

   bool ret = true;
   if (!fMinimizer.get()) ret = CreateMinimizer(); 
   if (!ret) { 
      ccoutE(Eval) << "Error returned creating minimizer for likelihood function - cannot find interval limits " << std::endl;
      return false; 
   }

   assert(fMinimizer.get());
        
   // getting a 1D interval so ndf = 1
   double err_level = TMath::ChisquareQuantile(ConfidenceLevel(),1); // level for -2log LR
   err_level = err_level/2; // since we are using -log LR
   fMinimizer->SetErrorDef(err_level);
   
   unsigned int ivarX = ix; 

   fMinimizer->SetPrintLevel(1);
   double elow = 0; 
   double eup = 0;
   ret = fMinimizer->GetMinosError(ivarX, elow, eup );
   // WHEN error is zero normally is at limit
   if (elow == 0) { 
      ccoutW(Minimization) << "Warning: lower value for " << param.GetName() << " is at limit " << std::endl; 
      lower = param.getMin();
   }
   else 
      lower = fMinimizer->X()[ivarX] + elow;  // elow is negative 

   if (eup == 0) { 
      ccoutW(Minimization) << "Warning: upper value for " << param.GetName() << " is at limit " << std::endl; 
      upper = param.getMax();
   }
   else 
      upper = fMinimizer->X()[ivarX] + eup;

   if (!ret)  ccoutE(Minimization) << "Error  running Minos for parameter " << param.GetName() << std::endl;
   else { 
      // store limits in the map 
      // minos return error limit = minValue +/- error
      fLowerLimits[param.GetName()] = lower; 
      fUpperLimits[param.GetName()] = upper; 
   }
      

   return ret; 
}


Int_t LikelihoodInterval::GetContourPoints(const RooRealVar & paramX, const RooRealVar & paramY, Double_t * x, Double_t *y, Int_t npoints ) { 
   // use Minuit to find the contour of the likelihood 
   // take first the nll function 
   // find contours 

   // check the parameters 
   // variable index in minimizer
   // is index in the RooArgList obtained from the profileLL variables
   RooArgSet * partmp = fLikelihoodRatio->getVariables();
   RooArgList params(*partmp);
   delete partmp;
   int ix = params.index(&paramX); 
   int iy = params.index(&paramY); 
   if (ix < 0 || iy < 0) { 
      ccoutE(InputArguments) << "Error - invalid parameters specified for finding the contours; parX = " << paramX.GetName() 
             << " parY = " << paramY.GetName() << std::endl;
         return 0; 
   }

   bool ret = true; 
   if (!fMinimizer.get()) ret = CreateMinimizer(); 
   if (!ret) { 
      ccoutE(Eval) << "Error returned creating minimizer for likelihood function - cannot find contour points " << std::endl;
      return 0; 
   }

   assert(fMinimizer.get());
        
   // getting a 2D contour so ndf = 2
   double cont_level = TMath::ChisquareQuantile(ConfidenceLevel(),2); // level for -2log LR
   cont_level = cont_level/2; // since we are using -log LR
   fMinimizer->SetErrorDef(cont_level);
  
   unsigned int ncp = npoints; 
   unsigned int ivarX = ix; 
   unsigned int ivarY = iy; 
   ret = fMinimizer->Contour(ivarX, ivarY, ncp, x, y );
   if (!ret) { 
      ccoutE(Minimization) << "Error finding contour for parameters " << paramX.GetName() << " and " << paramY.GetName()  << std::endl;
      return 0; 
   }
   if (int(ncp) < npoints) {
      ccoutW(Minimization) << "Warning - Less points calculated in contours np = " << ncp << " / " << npoints << std::endl;
   }

   return ncp;
 }
