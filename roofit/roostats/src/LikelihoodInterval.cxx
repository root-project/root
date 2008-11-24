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
#include "RooAbsReal.h"
#include "RooMsgService.h"
#include "RooStats/RooStatsUtils.h"
#include <string>


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

}


//____________________________________________________________________
Bool_t LikelihoodInterval::IsInInterval(RooArgSet &parameterPoint) 
{  
  // This is the main method to satisfy the RooStats::ConfInterval interface.  
  // It returns true if the parameter point is in the interval.

  RooMsgService::instance().setGlobalKillBelow(RooMsgService::FATAL) ;
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


  RooMsgService::instance().setGlobalKillBelow(RooMsgService::DEBUG) ;
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
Double_t LikelihoodInterval::LowerLimit(RooRealVar& param ) 
{  
// A binary search to get lower/upper limit for a given parameter.  Slow.

  RooAbsReal* newProfile = fLikelihoodRatio->createProfile(RooArgSet(param));
  RooRealVar* myarg = (RooRealVar *) newProfile->getVariables()->find(param.GetName());

  // to do this cleanly, should start at minimum.  Either need minimum from somewhere else, or need to find it via Minuit.
  // for testing, have param use min value

  // do binary search for minimum
  double target = TMath::ChisquareQuantile(fConfidenceLevel,fParameters->getSize());

  Double_t thisArgVal = param.getVal(); // need this to be MLE
  myarg->setVal( thisArgVal );
  //  std::cout << "lambda("<<thisArgVal<<") = " << newProfile->getVal() << std::endl;;


  double step = thisArgVal - myarg->getMin();
  double lastDiff = newProfile->getVal() - target, diff=lastDiff;
  int nIterations = 0, maxIterations = 20;
  std::cout << "about to do binary search" << std::endl;
  while(fabs(diff) > 0.01 && nIterations < maxIterations){
    nIterations++;
    if(diff<0)
      thisArgVal -= step; // LR too small, reduce myarg
    else
      thisArgVal += step; // LR too big, increase myarg

    if(lastDiff*diff < 0) // crossed target, reduce step size
      step /=2; 

    myarg->setVal( thisArgVal );
    //    myarg->Print();
    //    std::cout << "lambda("<<thisArgVal<<") = " << newProfile->getVal() << std::endl;;
    lastDiff = diff;
    // abs below to protect small negative numbers from numerical precision
    diff = 2.*(newProfile->getVal()) - target; 
    //    std::cout << "diff = " << diff << std::endl;
  }
  


  // put the parameters back to the way they were
  //  (*fParameters) = (*snapshot);

  return myarg->getVal();

}



//____________________________________________________________________
Double_t LikelihoodInterval::UpperLimit(RooRealVar& param ) 
{  

// A binary search to get lower/upper limit for a given parameter.  Slow.
  RooAbsReal* newProfile = fLikelihoodRatio->createProfile(RooArgSet(param));
  RooRealVar* myarg = (RooRealVar *) newProfile->getVariables()->find(param.GetName());

  // to do this cleanly, should start at minimum.  Either need minimum from somewhere else, or need to find it via Minuit.
  // for testing, have param use min value

  // do binary search for minimum
  double target = TMath::ChisquareQuantile(fConfidenceLevel,fParameters->getSize());

  Double_t thisArgVal = param.getVal(); // need this to be MLE
  myarg->setVal( thisArgVal );


  double step = thisArgVal - myarg->getMin();
  double lastDiff = newProfile->getVal() - target, diff=lastDiff;
  int nIterations = 0, maxIterations = 20;
  //  std::cout << "about to do binary search" << std::endl;
  while(fabs(diff) > 0.01 && nIterations < maxIterations){
    nIterations++;
    if(diff<0)
      thisArgVal += step; // LR too small, increase myarg
    else
      thisArgVal -= step; // LR too big, reduce myarg

    if(lastDiff*diff < 0) // crossed target, reduce step size
      step /=2; 

    myarg->setVal( thisArgVal );
    //    myarg->Print();
    //    std::cout << "lambda("<<thisArgVal<<") = " << newProfile->getVal() << std::endl;;
    lastDiff = diff;
    // abs below to protect small negative numbers from numerical precision
    diff = 2.*(newProfile->getVal()) - target; 
    //    std::cout << "diff = " << diff << std::endl;
  }
  


  // put the parameters back to the way they were
  //  (*fParameters) = (*snapshot);

  return myarg->getVal();

}



