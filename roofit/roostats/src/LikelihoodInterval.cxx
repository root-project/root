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
LikelihoodInterval::LikelihoodInterval() : fLikelihoodRatio(0)
{
   // Default constructor
}

//____________________________________________________________________
LikelihoodInterval::LikelihoodInterval(const char* name) :
   ConfInterval(name,name), fLikelihoodRatio(0)
{
   // Alternate constructor
}

//____________________________________________________________________
LikelihoodInterval::LikelihoodInterval(const char* name, const char* title) :
   ConfInterval(name,title), fLikelihoodRatio(0)
{
   // Alternate constructor
}

//____________________________________________________________________
LikelihoodInterval::LikelihoodInterval(const char* name, RooAbsReal* lr, RooArgSet* params) :
   ConfInterval(name,name)
{
   // Alternate constructor
  fLikelihoodRatio = lr;
  fParameters = params;
}

//____________________________________________________________________
LikelihoodInterval::LikelihoodInterval(const char* name, const char* title, RooAbsReal* lr, RooArgSet* params) :
   ConfInterval(name,title)
{
   // Alternate constructor
  fLikelihoodRatio = lr;
  fParameters = params;
}

//____________________________________________________________________
LikelihoodInterval::~LikelihoodInterval()
{
   // Destructor
  if(fLikelihoodRatio) delete fLikelihoodRatio;

}


//____________________________________________________________________
Bool_t LikelihoodInterval::IsInInterval(RooArgSet &parameterPoint) 
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

  
  /*
  ///////////////////////////
  // Debugging
  RooProfileLL* profile = (RooProfileLL*) fLikelihoodRatio;
  profile->nll().Print();
  //  profile->nll().printCompactTree();
  RooNLLVar* nll = (RooNLLVar*) (profile->getComponents()->find("nll_modelWithConstraints_modelWithConstraintsData"));
  RooDataSet* tmpData = (RooDataSet*) &(nll->data());
  nll->Print();
  std::cout << "nll = " << nll << " data = " << &(nll->data()) <<  " " << tmpData << std::endl;
  tmpData->Print();
  for(int i=0; i<tmpData->numEntries(); ++i)
    tmpData->get(i)->Print("v");

  std::cout<< "best fit params = " << std::endl;
  profile->bestFitParams().Print("v");

  SetParameters(&(profile->bestFitParams()), fLikelihoodRatio->getVariables() );
  //////////////////////////
  */


  // set parameters
  SetParameters(&parameterPoint, fLikelihoodRatio->getVariables() );


  // evaluate likelihood ratio, see if it's bigger than threshold
  if (fLikelihoodRatio->getVal()<0){
    std::cout << "The likelihood ratio is < 0, indicates a bad minimum or numerical precision problems.  Will return true" << std::endl;
    return true;
  }


  /*  
    std::cout << "in likelihood interval: LR = " <<
      fLikelihoodRatio->getVal() << " " << 
    " ndof = " << parameterPoint.getSize() << 
    " alpha = " << 1.-fConfidenceLevel << " cl = " << fConfidenceLevel <<
    " with P = " <<
    TMath::Prob( 2* fLikelihoodRatio->getVal(), parameterPoint.getSize())  <<
    " and CL = " << fConfidenceLevel << std::endl;

    parameterPoint.Print("v");
    fLikelihoodRatio->getVariables()->Print("v");
    //    fLikelihoodRatio->printCompactTree();
    */
    

  // here we use Wilks' theorem.
  if ( TMath::Prob( 2* fLikelihoodRatio->getVal(), parameterPoint.getSize()) < (1.-fConfidenceLevel) )
    return false;


  RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
  return true;
  
}

//____________________________________________________________________
RooArgSet* LikelihoodInterval::GetParameters() const
{  
  // returns list of parameters
  return (RooArgSet*) fParameters->clone((std::string(fParameters->GetName())+"_clone").c_str());
}

//____________________________________________________________________
Bool_t LikelihoodInterval::CheckParameters(RooArgSet &parameterPoint) const
{  
  // check that the parameters are correct

  if (parameterPoint.getSize() != fParameters->getSize() ) {
    std::cout << "size is wrong, parameters don't match" << std::endl;
    return false;
  }
  if ( ! parameterPoint.equals( *fParameters ) ) {
    std::cout << "size is ok, but parameters don't match" << std::endl;
    return false;
  }
  return true;
}



//____________________________________________________________________
Double_t LikelihoodInterval::LowerLimit(RooRealVar& param) 
{  

  RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;

  RooAbsReal* newProfile = fLikelihoodRatio->createProfile(RooArgSet(param));
  RooRealVar* myarg = (RooRealVar *) newProfile->getVariables()->find(param.GetName());

  double target = TMath::ChisquareQuantile(fConfidenceLevel,fParameters->getSize())/2.;

  Double_t thisArgVal = param.getVal();
  myarg->setVal( thisArgVal );



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
           //std::cout<<"WARNING upper limit is outside the parameters bounderies. Abort!"<<std::endl;
	  delete newProfile;
	  double ret = myarg->getMax();
	  delete myarg;
	  return ret;
	}
      }
    
    
    myarg->setVal( thisArgVal );
    L=newProfile->getVal();
    //Compute the linear function
    a=(L-L_old)/(step);
    if (a<1E-3)
       //std::cout<<"WARNING: the slope of the Likelihood linear approximation is close to zero."<<std::endl;
    b=L-a*thisArgVal;
    //approximate the position of the desired L value
    x_app=(target-b)/a;
    step=x_app-thisArgVal;
    diff=L-target;

    if(a>0) {
       //std::cout<<"WARNING: you are on the right of the MLE. Step towards MLE"<<std::endl;
      step=-(myarg->getMax()-myarg->getMin())/100; //Do a constant step, typical for the dimenions of the problem towards the minimum
     
      
    }
  }
  
  
  //std::cout<<"LL search Iterations:"<< nIterations<<std::endl;
 
  delete newProfile;
  double ret=myarg->getVal();
  //delete myarg;
  return ret;
}



//____________________________________________________________________
Double_t LikelihoodInterval::UpperLimit(RooRealVar& param) 
{  
  RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
  RooAbsReal* newProfile = fLikelihoodRatio->createProfile(RooArgSet(param));
  RooRealVar* myarg = (RooRealVar *) newProfile->getVariables()->find(param.GetName());

  double target = TMath::ChisquareQuantile(fConfidenceLevel,fParameters->getSize())/2.;
  Double_t thisArgVal = param.getVal();
  myarg->setVal( thisArgVal );

  double step=1;
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
    if (thisArgVal>myarg->getMax())
      {
         //std::cout<<"WARNING: near the upper boundery"<<std::endl;
	thisArgVal=myarg->getMax(); 
	step=thisArgVal+step-x_app;
	//std::cout<<"DEBUG: step:"<<step<<" thistArgVal:"<<thisArgVal<<std::endl;
	if (fabs(step)<1E-5) {
	  std::cout<<"WARNING upper limit is outside the parameters bounderies. Abort!"<<std::endl;
	  delete newProfile;
	  double ret=myarg->getMax();
	  delete myarg;
	  return ret;
	}
      }
    
    myarg->setVal( thisArgVal );
    L=newProfile->getVal();
    //Compute the linear approximation
    a=(L-L_old)/(step);
   
    if (fabs(a)<1E-3){
       //std::cout<<"WARNING: the slope of the Likelihood linear approximation is close to zero."<<std::endl;
    }
    b=L-a*thisArgVal;
    //approximate the position of the desired L value
    x_app=(target-b)/a;
    step=x_app-thisArgVal;
    diff=L-target;

    //If slope is negative you are below the minimum
    if(a<0) {
       //std::cout<<"WARNING: you are on the left of the MLE. Step towards MLE"<<std::endl;
      step=(myarg->getMax()-myarg->getMin())/100; //Do a constant step, typical for the dimenions of the problem towards the minimum
      //L_old=0;
      
    }
    
  }
  
  
  //std::cout<<"UL search Iterations:"<< nIterations<<std::endl;
  

  delete newProfile;
  double ret=myarg->getVal();
  //delete myarg;
  return ret;
  //return x_app;

  RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
}



