// @(#)root/roostats:$Id$
// Author: George Lewis, Kyle Cranmer
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOBARLOWBEESTONLL
#define ROOBARLOWBEESTONLL

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace RooStats{
  namespace HistFactory{

class RooBarlowBeestonLL : public RooAbsReal {
public:

  RooBarlowBeestonLL() ;
  RooBarlowBeestonLL(const char *name, const char *title, RooAbsReal& nll /*, const RooArgSet& observables*/);
  RooBarlowBeestonLL(const RooBarlowBeestonLL& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooBarlowBeestonLL(*this,newname); }
  virtual ~RooBarlowBeestonLL() ;

  // A simple class to store the
  // necessary objects for a
  // single gamma in a single channel
  class BarlowCache {
  public:
    bool hasStatUncert = false;
    RooRealVar* gamma = nullptr;
    RooArgSet* observables = nullptr;
    RooArgSet* bin_center = nullptr; // Snapshot
    RooRealVar* tau = nullptr;
    RooAbsReal* nom_pois_mean = nullptr;
    RooAbsReal* sumPdf = nullptr;
    double nData = -1;
    double binVolume = 0;
    void SetBinCenter() const;
  };

  void initializeBarlowCache();
  bool getParameters(const RooArgSet* depList, RooArgSet& outputSet, bool stripDisconnected=true) const;
  RooAbsReal& nll() { return const_cast<RooAbsReal&>(_nll.arg()) ; }
  virtual bool redirectServersHook(const RooAbsCollection& /*newServerList*/,
                                   bool /*mustReplaceAll*/,
                                   bool /*nameChange*/,
                                   bool /*isRecursive*/) ;
  void setPdf(RooAbsPdf* pdf) { _pdf = pdf; }
  void setDataset(RooAbsData* data) { _data = data; }

protected:

  RooRealProxy _nll ;    ///< Input -log(L) function
  RooAbsPdf* _pdf;
  RooAbsData* _data;
  mutable std::map< std::string, std::vector< BarlowCache > > _barlowCache;
  mutable std::set< std::string > _statUncertParams;
  mutable std::map<std::string,bool> _paramFixed ; ///< Parameter constant status at last time of use
  Double_t evaluate() const ;

private:

  // Real-valued function representing a Barlow-Beeston minimized profile likelihood of external (likelihood) function
  ClassDef(RooStats::HistFactory::RooBarlowBeestonLL,0)
};

  }
}

#endif
