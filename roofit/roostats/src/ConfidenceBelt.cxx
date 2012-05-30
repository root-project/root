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



//_________________________________________________________
//
// BEGIN_HTML
// ConfidenceBelt is a concrete implementation of the ConfInterval interface.  
// It implements simple general purpose interval of arbitrary dimensions and shape.
// It does not assume the interval is connected.
// It uses either a RooDataSet (eg. a list of parameter points in the interval) or
// a RooDataHist (eg. a Histogram-like object for small regions of the parameter space) to
// store the interval.  
// END_HTML
//
//

#ifndef RooStats_ConfidenceBelt
#include "RooStats/ConfidenceBelt.h"
#endif

#include "RooDataSet.h"
#include "RooDataHist.h"

#include "RooStats/RooStatsUtils.h"

ClassImp(RooStats::ConfidenceBelt) ;

using namespace RooStats;
using namespace std;

//____________________________________________________________________
ConfidenceBelt::ConfidenceBelt() : 
   TNamed(), fParameterPoints(0)
{
   // Default constructor
}

//____________________________________________________________________
ConfidenceBelt::ConfidenceBelt(const char* name) :
  TNamed(name,name), fParameterPoints(0)
{
   // Alternate constructor
}

//____________________________________________________________________
ConfidenceBelt::ConfidenceBelt(const char* name, const char* title) :
   TNamed(name,title), fParameterPoints(0)
{
   // Alternate constructor
}

//____________________________________________________________________
ConfidenceBelt::ConfidenceBelt(const char* name, RooAbsData& data) :
  TNamed(name,name), fParameterPoints((RooAbsData*)data.Clone("PointsToTestForBelt"))
{
   // Alternate constructor
}

//____________________________________________________________________
ConfidenceBelt::ConfidenceBelt(const char* name, const char* title, RooAbsData& data) :
   TNamed(name,title), fParameterPoints((RooAbsData*)data.Clone("PointsToTestForBelt"))
{
   // Alternate constructor
}



//____________________________________________________________________
ConfidenceBelt::~ConfidenceBelt()
{
   // Destructor

}


//____________________________________________________________________
Double_t ConfidenceBelt::GetAcceptanceRegionMin(RooArgSet& parameterPoint, Double_t cl, Double_t leftside) {

  if(cl>0 || leftside > 0) cout <<"using default cl, leftside for now" <<endl;
  AcceptanceRegion * region = GetAcceptanceRegion(parameterPoint, cl,leftside); 
  return (region) ? region->GetLowerLimit() : TMath::QuietNaN();
}

//____________________________________________________________________
Double_t ConfidenceBelt::GetAcceptanceRegionMax(RooArgSet& parameterPoint, Double_t cl, Double_t leftside) {
  if(cl>0 || leftside > 0) cout <<"using default cl, leftside for now" <<endl;
  AcceptanceRegion * region = GetAcceptanceRegion(parameterPoint, cl,leftside); 
  return (region) ? region->GetUpperLimit() : TMath::QuietNaN();
}

//____________________________________________________________________
vector<Double_t> ConfidenceBelt::ConfidenceLevels() const {
  vector<Double_t> levels;
  return levels;
}

//____________________________________________________________________
void ConfidenceBelt::AddAcceptanceRegion(RooArgSet& parameterPoint, Int_t dsIndex,
					 Double_t lower, Double_t upper,
					 Double_t cl, Double_t leftside){
  
  if(cl>0 || leftside > 0) cout <<"using default cl, leftside for now" <<endl;

  RooDataSet*  tree = dynamic_cast<RooDataSet*>(  fParameterPoints );
  RooDataHist* hist = dynamic_cast<RooDataHist*>( fParameterPoints );  

  //  cout << "add: " << tree << " " << hist << endl;

  if( !this->CheckParameters(parameterPoint) )
    std::cout << "problem with parameters" << std::endl;

  Int_t luIndex = fSamplingSummaryLookup.GetLookupIndex(cl, leftside);
  //  cout << "lookup index = " << luIndex << endl;
  if(luIndex <0 ) {
    fSamplingSummaryLookup.Add(cl,leftside);
    luIndex = fSamplingSummaryLookup.GetLookupIndex(cl, leftside);
    cout << "lookup index = " << luIndex << endl;
  }
  AcceptanceRegion* thisRegion = new AcceptanceRegion(luIndex, lower, upper);
  
  if( hist ) {
    // need a way to get index for given point
    // Can do this by setting hist's internal parameters to desired values
    // need a better way
    //    RooStats::SetParameters(&parameterPoint, const_cast<RooArgSet*>(hist->get())); 
    //    int index = hist->calcTreeIndex(); // get index
    int index = hist->getIndex(parameterPoint); // get index
    //    cout << "hist index = " << index << endl;

    // allocate memory if necessary.  numEntries is overkill?
    if((Int_t)fSamplingSummaries.size() <= index) {
      fSamplingSummaries.reserve( hist->numEntries() ); 
      fSamplingSummaries.resize( hist->numEntries() ); 
    }

    // set the region for this point (check for duplicate?)
    fSamplingSummaries.at(index) = *thisRegion;
  }
  else if( tree ){
    //    int index = tree->getIndex(parameterPoint); 
    int index = dsIndex;
    //    cout << "tree index = " << index << endl;

    // allocate memory if necessary.  numEntries is overkill?
    if((Int_t)fSamplingSummaries.size() <= index){
      fSamplingSummaries.reserve( tree->numEntries()  ); 
      fSamplingSummaries.resize( tree->numEntries() ); 
    }

    // set the region for this point (check for duplicate?)
    fSamplingSummaries.at( index ) = *thisRegion;
  }
}

//____________________________________________________________________
void ConfidenceBelt::AddAcceptanceRegion(RooArgSet& parameterPoint, AcceptanceRegion region, 
					 Double_t cl, Double_t leftside){
  
  if(cl>0 || leftside > 0) cout <<"using default cl, leftside for now" <<endl;

  RooDataSet*  tree = dynamic_cast<RooDataSet*>(  fParameterPoints );
  RooDataHist* hist = dynamic_cast<RooDataHist*>( fParameterPoints );

  if( !this->CheckParameters(parameterPoint) )
    std::cout << "problem with parameters" << std::endl;
  
  
  if( hist ) {
    // need a way to get index for given point
    // Can do this by setting hist's internal parameters to desired values
    // need a better way
    //    RooStats::SetParameters(&parameterPoint, const_cast<RooArgSet*>(hist->get())); 
    //    int index = hist->calcTreeIndex(); // get index
    int index = hist->getIndex(parameterPoint); // get index

    // allocate memory if necessary.  numEntries is overkill?
    if((Int_t)fSamplingSummaries.size() < index) fSamplingSummaries.reserve( hist->numEntries() ); 

    // set the region for this point (check for duplicate?)
    fSamplingSummaries.at(index) = region;
  }
  else if( tree ){
    tree->add( parameterPoint ); // assume it's unique for now
    int index = tree->numEntries() - 1; //check that last point added has index nEntries -1
    // allocate memory if necessary.  numEntries is overkill?
    if((Int_t)fSamplingSummaries.size() < index) fSamplingSummaries.reserve( tree->numEntries()  ); 

    // set the region for this point (check for duplicate?)
    fSamplingSummaries.at( index ) = region;
  }
}

//____________________________________________________________________
AcceptanceRegion* ConfidenceBelt::GetAcceptanceRegion(RooArgSet &parameterPoint, Double_t cl, Double_t leftside) 
{  
   // Method to determine if a parameter point is in the interval

  if(cl>0 || leftside > 0) cout <<"using default cl, leftside for now" <<endl;

  RooDataSet*  tree = dynamic_cast<RooDataSet*>(  fParameterPoints );
  RooDataHist* hist = dynamic_cast<RooDataHist*>( fParameterPoints );
  
  if( !this->CheckParameters(parameterPoint) ){
    std::cout << "problem with parameters" << std::endl;
    return 0; 
  }
  
  if( hist ) {
    // need a way to get index for given point
    // Can do this by setting hist's internal parameters to desired values
    // need a better way
    //    RooStats::SetParameters(&parameterPoint, const_cast<RooArgSet*>(hist->get())); 
    //    Int_t index = hist->calcTreeIndex(); // get index
    int index = hist->getIndex(parameterPoint); // get index
    return &(fSamplingSummaries.at(index).GetAcceptanceRegion());
  }
  else if( tree ){
    // need a way to get index for given point
    //    RooStats::SetParameters(&parameterPoint, tree->get()); // set tree's parameters to desired values
    Int_t index = 0; //need something like tree->calcTreeIndex(); 
    const RooArgSet* thisPoint = 0;
    for(index=0; index<tree->numEntries(); ++index){
      thisPoint = tree->get(index); 
      bool samePoint = true;
      TIter it = parameterPoint.createIterator();
      RooRealVar *myarg; 
      while ( samePoint && (myarg = (RooRealVar *)it.Next())) { 
	if(myarg->getVal() != thisPoint->getRealValue(myarg->GetName()))
	  samePoint = false;
      }
      if(samePoint) 
	break;
    }

    return &(fSamplingSummaries.at(index).GetAcceptanceRegion());
  }
  else {
      std::cout << "dataset is not initialized properly" << std::endl;
  }

  return 0;
  
}

//____________________________________________________________________
RooArgSet* ConfidenceBelt::GetParameters() const
{  
   // returns list of parameters
   return new RooArgSet(*(fParameterPoints->get()));
}

//____________________________________________________________________
Bool_t ConfidenceBelt::CheckParameters(RooArgSet &parameterPoint) const
{  

   if (parameterPoint.getSize() != fParameterPoints->get()->getSize() ) {
      std::cout << "size is wrong, parameters don't match" << std::endl;
      return false;
   }
   if ( ! parameterPoint.equals( *(fParameterPoints->get() ) ) ) {
      std::cout << "size is ok, but parameters don't match" << std::endl;
      return false;
   }
   return true;
}
