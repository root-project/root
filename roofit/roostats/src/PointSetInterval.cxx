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
 * Original Author: Kyle Cranmer
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
 *
 *****************************************************************************/


/** \class RooStats::PointSetInterval
    \ingroup Roostats

PointSetInterval is a concrete implementation of the ConfInterval interface.
It implements simple general purpose interval of arbitrary dimensions and shape.
It does not assume the interval is connected.
It uses either a RooDataSet (eg. a list of parameter points in the interval) or
a RooDataHist (eg. a Histogram-like object for small regions of the parameter space) to
store the interval.

*/


#include "RooStats/PointSetInterval.h"

#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooDataHist.h"

using namespace std;

ClassImp(RooStats::PointSetInterval); ;

using namespace RooStats;


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

PointSetInterval::PointSetInterval(const char* name) :
   ConfInterval(name), fConfidenceLevel(0.95), fParameterPointsInInterval(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Alternate constructor passing the dataset

PointSetInterval::PointSetInterval(const char* name, RooAbsData& data) :
   ConfInterval(name), fConfidenceLevel(0.95), fParameterPointsInInterval(&data)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

PointSetInterval::~PointSetInterval()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Method to determine if a parameter point is in the interval

Bool_t PointSetInterval::IsInInterval(const RooArgSet &parameterPoint) const
{
  RooDataSet*  tree = dynamic_cast<RooDataSet*>(  fParameterPointsInInterval );
  RooDataHist* hist = dynamic_cast<RooDataHist*>( fParameterPointsInInterval );

  if( !this->CheckParameters(parameterPoint) ){
    //    std::cout << "problem with parameters" << std::endl;
    return false;
  }

  if( hist ) {
    if ( hist->weight( parameterPoint , 0 ) > 0 ) // positive value indicates point is in interval
      return true;
    else
      return false;
  }
  else if( tree ){
    const RooArgSet* thisPoint = 0;
    // need to check if the parameter point is the same as any point in tree.
    for(Int_t i = 0; i<tree->numEntries(); ++i){
      // This method is not complete
      thisPoint = tree->get(i);
      bool samePoint = true;
      for (auto const *myarg : static_range_cast<RooRealVar *>(parameterPoint)) {
        if(samePoint == false)
            break;
        if(myarg->getVal() != thisPoint->getRealValue(myarg->GetName()))
        samePoint = false;
      }
      if(samePoint)
        return true;
    }
    return false; // didn't find a good point
  }
  else {
      std::cout << "dataset is not initialized properly" << std::endl;
  }

   return true;

}

////////////////////////////////////////////////////////////////////////////////
/// returns list of parameters

RooArgSet* PointSetInterval::GetParameters() const
{
   return new RooArgSet(*(fParameterPointsInInterval->get()) );
}

////////////////////////////////////////////////////////////////////////////////

Bool_t PointSetInterval::CheckParameters(const RooArgSet &parameterPoint) const
{
   if (parameterPoint.getSize() != fParameterPointsInInterval->get()->getSize() ) {
     std::cout << "PointSetInterval: argument size is wrong, parameters don't match: arg=" << parameterPoint
          << " interval=" << (*fParameterPointsInInterval->get()) << std::endl;
      return false;
   }
   if ( ! parameterPoint.equals( *(fParameterPointsInInterval->get() ) ) ) {
      std::cout << "PointSetInterval: size is ok, but parameters don't match" << std::endl;
      return false;
   }
   return true;
}


////////////////////////////////////////////////////////////////////////////////

Double_t PointSetInterval::UpperLimit(RooRealVar& param )
{
  RooDataSet*  tree = dynamic_cast<RooDataSet*>(  fParameterPointsInInterval );
  Double_t low = 0, high = 0;
  if( tree ){
    tree->getRange(param, low, high);
    return high;
 }
  return param.getMax();
}

////////////////////////////////////////////////////////////////////////////////

Double_t PointSetInterval::LowerLimit(RooRealVar& param )
{
  RooDataSet*  tree = dynamic_cast<RooDataSet*>(  fParameterPointsInInterval );
  Double_t low = 0, high = 0;
  if( tree ){
    tree->getRange(param, low, high);
    return low;
 }
  return param.getMin();
}
