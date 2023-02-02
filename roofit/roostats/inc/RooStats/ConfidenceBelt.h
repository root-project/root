// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_ConfidenceBelt
#define RooStats_ConfidenceBelt

#include "RooArgSet.h"
#include "RooAbsData.h"
#include "RooStats/ConfInterval.h"

#include "RooStats/SamplingDistribution.h"

#include "TRef.h"

#include <vector>
#include <map>


namespace RooStats {

////////////////////////////////////////////////////////////////////////////////

  class SamplingSummaryLookup : public TObject {

     typedef std::pair<double, double> AcceptanceCriteria; // defined by Confidence level, leftside tail probability
     typedef std::map<Int_t, AcceptanceCriteria> LookupTable; // map ( Index, ( CL, leftside tail prob) )

  public:
    SamplingSummaryLookup() {}
    ~SamplingSummaryLookup() override {}

    void Add(double cl, double leftside){
      // add cl,leftside pair to lookup table
      AcceptanceCriteria tmp(cl, leftside);

      // should check to see if this is already in the map
      if(GetLookupIndex(cl,leftside) >=0 ){
         std::cout<< "SamplingSummaryLookup::Add, already in lookup table" << std::endl;
      } else
   fLookupTable[fLookupTable.size()]= tmp;
    }

    Int_t GetLookupIndex(double cl, double leftside){
      // get index for cl,leftside pair
      AcceptanceCriteria tmp(cl, leftside);

      double tolerance = 1E-6; // some small number to protect floating point comparison.  What is better way?
      LookupTable::iterator it = fLookupTable.begin();
      Int_t index = -1;
      for(; it!= fLookupTable.end(); ++it) {
   index++;
   if( TMath::Abs( (*it).second.first - cl ) < tolerance &&
       TMath::Abs( (*it).second.second - leftside ) < tolerance )
     break; // exit loop, found
      }

      // check that it was found
      if(index == (Int_t)fLookupTable.size())
   index = -1;

      return index;
    }

  double GetConfidenceLevel(Int_t index){
    if(index<0 || index>(Int_t)fLookupTable.size()) {
       std::cout << "SamplingSummaryLookup::GetConfidenceLevel, index not in lookup table" << std::endl;
       return -1;
    }
    return fLookupTable[index].first;
  }

  double GetLeftSideTailFraction(Int_t index){
    if(index<0 || index>(Int_t)fLookupTable.size()) {
       std::cout << "SamplingSummaryLookup::GetLeftSideTailFraction, index not in lookup table" << std::endl;
       return -1;
    }
    return fLookupTable[index].second;
  }

  private:
    LookupTable fLookupTable; ///< map ( Index, ( CL, leftside tail prob) )

  protected:
    ClassDefOverride(SamplingSummaryLookup,1)  // A simple class used by ConfidenceBelt
  };


  ///////////////////////////
  class AcceptanceRegion : public TObject{
  public:
     AcceptanceRegion() : fLookupIndex(0), fLowerLimit(0), fUpperLimit(0) {}
    ~AcceptanceRegion() override {}

    AcceptanceRegion(Int_t lu, double ll, double ul){
      fLookupIndex = lu;
      fLowerLimit = ll;
      fUpperLimit = ul;
    }
    Int_t GetLookupIndex(){return fLookupIndex;}
    double GetLowerLimit(){return fLowerLimit;}
    double GetUpperLimit(){return fUpperLimit;}

  private:
    Int_t fLookupIndex; // want a small footprint reference to the RooArgSet for particular parameter point
    double fLowerLimit;  // lower limit on test statistic
    double fUpperLimit;  // upper limit on test statistic

  protected:
    ClassDefOverride(AcceptanceRegion,1)  // A simple class for acceptance regions used for ConfidenceBelt

  };

////////////////////////////////////////////////////////////////////////////////

  class SamplingSummary : public TObject {
  public:
     SamplingSummary() : fParameterPointIndex(0) {}
    ~SamplingSummary() override {}
     SamplingSummary(AcceptanceRegion& ar) : fParameterPointIndex(0) {
      AddAcceptanceRegion(ar);
    }
    Int_t GetParameterPointIndex(){return fParameterPointIndex;}
    SamplingDistribution* GetSamplingDistribution(){
      return (SamplingDistribution*) fSamplingDistribution.GetObject(); // dereference TRef
    }
    AcceptanceRegion& GetAcceptanceRegion(Int_t index=0){return fAcceptanceRegions[index];}

    void AddAcceptanceRegion(AcceptanceRegion& ar){
      Int_t index =  ar.GetLookupIndex();
      if( fAcceptanceRegions.count(index) !=0) {
   std::cout << "SamplingSummary::AddAcceptanceRegion, need to implement merging protocol" << std::endl;
      } else {
   fAcceptanceRegions[index]=ar;
      }
    }

  private:
     Int_t fParameterPointIndex; // want a small footprint reference to the RooArgSet for particular parameter point
     TRef fSamplingDistribution; // persistent pointer to a SamplingDistribution
     std::map<Int_t, AcceptanceRegion> fAcceptanceRegions;

  protected:
    ClassDefOverride(SamplingSummary,1)  // A summary of acceptance regions for confidence belt

  };

////////////////////////////////////////////////////////////////////////////////

 class ConfidenceBelt : public TNamed {

  private:
    SamplingSummaryLookup fSamplingSummaryLookup;
    std::vector<SamplingSummary> fSamplingSummaries; // composite of several AcceptanceRegions
    RooAbsData* fParameterPoints;  // either a histogram (RooDataHist) or a tree (RooDataSet)


  public:
    // constructors,destructors
    ConfidenceBelt();
    ConfidenceBelt(const char* name);
    ConfidenceBelt(const char* name, const char* title);
    ConfidenceBelt(const char* name, RooAbsData&);
    ConfidenceBelt(const char* name, const char* title, RooAbsData&);
    ~ConfidenceBelt() override;

    /// add after creating a region
    void AddAcceptanceRegion(RooArgSet&, AcceptanceRegion region, double cl=-1., double leftside=-1.);

    /// add without creating a region, more useful for clients
    void AddAcceptanceRegion(RooArgSet& point, Int_t dataSetIndex, double lower, double upper, double cl=-1., double leftside=-1.);

    AcceptanceRegion* GetAcceptanceRegion(RooArgSet&, double cl=-1., double leftside=-1.);
    double GetAcceptanceRegionMin(RooArgSet&, double cl=-1., double leftside=-1.);
    double GetAcceptanceRegionMax(RooArgSet&, double cl=-1., double leftside=-1.);
    std::vector<double> ConfidenceLevels() const ;

    /// do we want it to return list of parameters
    virtual RooArgSet* GetParameters() const;

    /// check if parameters are correct. (dummy implementation to start)
    bool CheckParameters(RooArgSet&) const ;

  protected:
    ClassDefOverride(ConfidenceBelt,1)  // A confidence belt for the Neyman Construction

  };
}

#endif
