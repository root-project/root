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
LikelihoodInterval::LikelihoodInterval(const char* name, const char* title) :
   ConfInterval(name,title), fBestFitParams(0), fLikelihoodRatio(0)
{
   // Default constructor with name and title
}

//____________________________________________________________________
LikelihoodInterval::LikelihoodInterval(const char* name, RooAbsReal* lr, const RooArgSet* params,  RooArgSet * bestParams) :
   ConfInterval(name,name), 
   fParameters(*params), 
   fBestFitParams(bestParams), 
   fLikelihoodRatio(lr)
{
   // Alternate constructor taking a pointer to the profile likelihood ratio, parameter of interest and 
   // optionally a snaphot of best parameter of interest for interval
}

//____________________________________________________________________
LikelihoodInterval::LikelihoodInterval(const char* name, const char* title, RooAbsReal* lr, const RooArgSet* params,   RooArgSet * bestParams) :
   ConfInterval(name,title), 
   fParameters(*params), 
   fBestFitParams( bestParams ), 
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
Bool_t LikelihoodInterval::IsInInterval(const RooArgSet &parameterPoint) 
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
  //return (RooArgSet*) fParameters->clone((std::string(fParameters->GetName())+"_clone").c_str());
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
Double_t LikelihoodInterval::LowerLimit(RooRealVar& param) 
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
Double_t LikelihoodInterval::UpperLimit(RooRealVar& param) 
{  
   // compute upper limit, check first if limit has been computed 
   std::map<std::string, double>::const_iterator itr = fUpperLimits.find(param.GetName());
   if (itr != fUpperLimits.end() ) 
      return itr->second;

   // otherwise compute limit

   //RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING) ;
  RooAbsReal* newProfile = fLikelihoodRatio->createProfile(RooArgSet(param));

  RooArgSet* vars = newProfile->getVariables() ;
  RooRealVar* myarg = (RooRealVar *)vars->find(param.GetName());
  delete vars ;

  double target = TMath::ChisquareQuantile(fConfidenceLevel, 1)/2.;
  //Double_t thisArgVal = param.getVal()*1.01 ;

  // start from  xmin + nsigma * error
  Double_t nsigma = target * 2; 
  RooRealVar * fitPar = (RooRealVar *) fBestFitParams->find(param.GetName() ); 
  Double_t thisArgVal = fitPar->getVal() + nsigma * fitPar->getError(); 

  //std::cout << " param UL - start value  " << thisArgVal << " val  " << myarg->getVal(); 

  myarg->setVal( thisArgVal );

  Double_t maxStep = (myarg->getMax()-myarg->getMin())/10 ;

  double step=1;
  double L= newProfile->getVal();
  double L_old=0;
  double diff = L - target;

  //std::cout << " L =   " << L << " diff = " << diff << std::endl;

  //the parameters for the linear approximation
  double a=0, b=0;
  double x_app=0;
  int nIterations = 0, maxIterations = 100;
  while(fabs(diff)>0.01 && nIterations<maxIterations){
    nIterations++;

//     std::cout << " iter " << nIterations << " step " << step << " val " << thisArgVal 
//               << "  L_old " << L_old << "  L " << L << std::endl; 

    L_old=L;
    thisArgVal=thisArgVal+step;

 
    if (thisArgVal>myarg->getMax())
    {
        ccoutD(Eval) <<"WARNING: near the upper boundery"<<std::endl;
	thisArgVal=myarg->getMax(); 
	step=thisArgVal+step-x_app;
	//std::cout<<"DEBUG: step:"<<step<<" thistArgVal:"<<thisArgVal<<std::endl;
	if (fabs(step)<1E-5) {
	  ccoutW(Eval) <<"Upper limit is outside the parameters bounderies. Abort!"<<std::endl;
	  delete newProfile;
	  double ret=myarg->getMax();
	  return ret;
	}
    }
    
    myarg->setVal( thisArgVal );
    L=newProfile->getVal();

    // If L is below target and we are at boundary stop here
    if ((fabs(myarg->getVal()-myarg->getMax())<1e-5) && (L<target)) {
       ccoutW(Eval) <<"WARNING upper limit is outside the parameters bounderies (L at upper bound of " << myarg->GetName() << " is " << L 
	       << ", which is less than target value of " << target << "). Abort!"<<std::endl;
       delete newProfile;
       double ret = myarg->getMax();
       return ret;      
    }


    //Compute the linear approximation
    a=(L-L_old)/(step);
   
    if (fabs(a)<1E-3){
       ccoutD(Eval) <<"WARNING: the slope of the Likelihood linear approximation is close to zero."<<std::endl;
    }
    b=L-a*thisArgVal;
    //approximate the position of the desired L value
    x_app=(target-b)/a;
    step=x_app-thisArgVal;
    if (step>maxStep) step=maxStep ;
    if (step<-maxStep) step=-maxStep ;
    diff=L-target;

    //If slope is negative you are below the minimum
    if(a<0) {
      ccoutD(Eval)<<"WARNING: you are on the left of the MLE. Step towards MLE"<<std::endl;
      step=(myarg->getMax()-myarg->getMin())/100; //Do a constant step, typical for the dimenions of the problem towards the minimum
      //L_old=0;
      
    }
    
  }
    
  ccoutD(Eval) <<"UL search Iterations:"<< nIterations<<std::endl;

  //cout << "UL iterations " << nIterations << " value = " << myarg->getVal() << " PL = " << newProfile->getVal() << std::endl;

  // restore ROOT reporting message level
  //RooMsgService::instance().setGlobalKillBelow(RooFit::INFO) ;

  delete newProfile;
  double ret=myarg->getVal();

  fUpperLimits[param.GetName()] = ret; 
  return ret;

}

void LikelihoodInterval::ResetLimits() { 
   // reset map with cached limits
   fLowerLimits.clear(); 
   fUpperLimits.clear(); 
}
