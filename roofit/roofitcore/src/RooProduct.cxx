/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   GR, Gerhard Raven,   VU Amsterdan,     graven@nikhef.nl                 *
 *                                                                           *
 * Copyright (c) 2000-2007, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooProduct.cxx
\class RooProduct
\ingroup Roofitcore

A RooProduct represents the product of a given set of RooAbsReal objects.

**/

#include "RooProduct.h"

#include "RooNameReg.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooMsgService.h"
#include "RooTrace.h"

#include <cmath>
#include <memory>

using namespace std ;

ClassImp(RooProduct);
;

class RooProduct::ProdMap : public  std::vector<std::pair<RooArgSet*,RooArgList*> > {} ;

// Namespace with helper functions that have STL stuff that we don't want to expose to CINT
namespace {
  typedef RooProduct::ProdMap::iterator RPPMIter ;
  std::pair<RPPMIter,RPPMIter> findOverlap2nd(RPPMIter i, RPPMIter end)  ;
  void dump_map(ostream& os, RPPMIter i, RPPMIter end) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooProduct::RooProduct() : _cacheMgr(this,10)
{
  TRACE_CREATE
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooProduct::~RooProduct()
{
  TRACE_DESTROY
}



////////////////////////////////////////////////////////////////////////////////
/// Construct function representing the product of functions in prodSet

RooProduct::RooProduct(const char* name, const char* title, const RooArgList& prodSet) :
  RooAbsReal(name, title),
  _compRSet("!compRSet","Set of real product components",this),
  _compCSet("!compCSet","Set of category product components",this),
  _cacheMgr(this,10)
{
  for (auto comp : prodSet) {
    addTerm(comp);
  }
  TRACE_CREATE
}


RooProduct::RooProduct(const char *name, const char *title, RooAbsReal& real1, RooAbsReal& real2) :
  RooProduct{name, title, {real1, real2}} {}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooProduct::RooProduct(const RooProduct& other, const char* name) :
  RooAbsReal(other, name),
  _compRSet("!compRSet",this,other._compRSet),
  _compCSet("!compCSet",this,other._compCSet),
  _cacheMgr(other._cacheMgr,this)
{
  TRACE_CREATE
}


////////////////////////////////////////////////////////////////////////////////
/// Add a term to this product.
void RooProduct::addTerm(RooAbsArg* term) {
  if (dynamic_cast<RooAbsReal*>(term)) {
    _compRSet.add(*term) ;
  } else if (dynamic_cast<RooAbsCategory*>(term)) {
    _compCSet.add(*term) ;
  } else {
    coutE(InputArguments) << "RooProduct::addTerm(" << GetName() << ") ERROR: component " << term->GetName()
        << " is not of type RooAbsReal or RooAbsCategory" << endl ;
    throw std::invalid_argument("RooProduct can only handle terms deriving from RooAbsReal or RooAbsCategory.");
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Force internal handling of integration of given observable if any
/// of the product terms depend on it.

bool RooProduct::forceAnalyticalInt(const RooAbsArg& dep) const
{
  // Force internal handling of integration of given observable if any
  // of the product terms depend on it.

  bool depends(false);
  for (auto const* rcomp : static_range_cast<RooAbsReal*>(_compRSet)) {
    if (depends) break;
    depends = rcomp->dependsOn(dep);
    }
  return depends ;
}



////////////////////////////////////////////////////////////////////////////////
/// Group observables into subsets in which the product factorizes
/// and that can thus be integrated separately

RooProduct::ProdMap* RooProduct::groupProductTerms(const RooArgSet& allVars) const
{
  ProdMap* map = new ProdMap ;

  // Do we have any terms which do not depend on the
  // on the variables we integrate over?
  RooArgList *indep = new RooArgList();
  for (auto const* rcomp : static_range_cast<RooAbsReal*>(_compRSet)) {
    if( !rcomp->dependsOn(allVars) ) indep->add(*rcomp);
  }
  if (indep->getSize()!=0) {
    map->push_back( std::make_pair(new RooArgSet(),indep) );
  } else {
     delete indep;
  }

  // Map observables -> functions ; start with individual observables
  for (auto const* var : static_range_cast<RooAbsReal*>(allVars)) {
    RooArgSet *vars  = new RooArgSet(); vars->add(*var);
    RooArgList *comps = new RooArgList();

    for (auto const* rcomp2 : static_range_cast<RooAbsReal*>(_compRSet)) {
      if( rcomp2->dependsOn(*var) ) comps->add(*rcomp2);
    }
    map->push_back( std::make_pair(vars,comps) );
  }

  // Merge groups with overlapping dependents
  bool overlap;
  do {
    std::pair<ProdMap::iterator,ProdMap::iterator> i = findOverlap2nd(map->begin(),map->end());
    overlap = (i.first!=i.second);
    if (overlap) {
      i.first->first->add(*i.second->first);

      // In the merging step, make sure not to duplicate
      for (auto const* targ : *(i.second->second)) {
        if (!i.first->second->find(*targ)) {
          i.first->second->add(*targ) ;
        }
      }
      //i.first->second->add(*i.second->second);

      delete i.second->first;
      delete i.second->second;
      map->erase(i.second);
    }
  } while (overlap);

#ifndef NDEBUG
  // check that we have all variables to be integrated over on the LHS
  // of the map, and all terms in the product do appear on the RHS
  int nVar=0; int nFunc=0;
  for (ProdMap::iterator i = map->begin();i!=map->end();++i) {
    nVar+=i->first->getSize();
    nFunc+=i->second->getSize();
  }
  assert(nVar==allVars.getSize());
  assert(nFunc==_compRSet.getSize());
#endif
  return map;
}



////////////////////////////////////////////////////////////////////////////////
/// Return list of (partial) integrals whose product defines the integral of this
/// RooProduct over the observables in iset in range isetRange. If no such list
/// exists, create it now and store it in the cache for future use.

Int_t RooProduct::getPartIntList(const RooArgSet* iset, const char *isetRange) const
{

  // check if we already have integrals for this combination of factors
  Int_t sterileIndex(-1);
  CacheElem* cache = (CacheElem*) _cacheMgr.getObj(iset,iset,&sterileIndex,RooNameReg::ptr(isetRange));
  if (cache!=0) {
    Int_t code = _cacheMgr.lastIndex();
    return code;
  }

  ProdMap* map = groupProductTerms(*iset);

  cxcoutD(Integration) << "RooProduct::getPartIntList(" << GetName() << ") groupProductTerms returned map" ;
  if (dologD(Integration)) {
    dump_map(ccoutD(Integration),map->begin(),map->end());
    ccoutD(Integration) << endl;
  }

  // did we find any factorizable terms?
  if (map->size()<2) {

    for (ProdMap::iterator iter = map->begin() ; iter != map->end() ; ++iter) {
      delete iter->first ;
      delete iter->second ;
    }

    delete map ;
    return -1; // RRI caller will zero analVars if return code = 0....
  }
  cache = new CacheElem();

  for (ProdMap::const_iterator i = map->begin();i!=map->end();++i) {
    RooAbsReal *term(0);
    if (i->second->getSize()>1) { // create a RooProd for this subexpression
      const char *name = makeFPName("SUBPROD_",*i->second);
      term = new RooProduct(name,name,*i->second);
      cache->_ownedList.addOwned(*term);
      cxcoutD(Integration) << "RooProduct::getPartIntList(" << GetName() << ") created subexpression " << term->GetName() << endl;
    } else {
      assert(i->second->getSize()==1);
      term = static_cast<RooAbsReal*>(i->second->at(0));
    }
    assert(term!=0);
    if (i->first->empty()) { // check whether we need to integrate over this term or not...
      cache->_prodList.add(*term);
      cxcoutD(Integration) << "RooProduct::getPartIntList(" << GetName() << ") adding simple factor " << term->GetName() << endl;
    } else {
      RooAbsReal *integral = term->createIntegral(*i->first,isetRange);
      cache->_prodList.add(*integral);
      cache->_ownedList.addOwned(*integral);
      cxcoutD(Integration) << "RooProduct::getPartIntList(" << GetName() << ") adding integral for " << term->GetName() << " : " << integral->GetName() << endl;
    }
  }
  // add current set-up to cache, and return index..
  Int_t code = _cacheMgr.setObj(iset,iset,(RooAbsCacheElement*)cache,RooNameReg::ptr(isetRange));

  cxcoutD(Integration) << "RooProduct::getPartIntList(" << GetName() << ") created list " << cache->_prodList << " with code " << code+1 << endl
             << " for iset=" << *iset << " @" << iset << " range: " << (isetRange?isetRange:"<none>") << endl ;

  for (ProdMap::iterator iter = map->begin() ; iter != map->end() ; ++iter) {
    delete iter->first ;
    delete iter->second ;
  }
  delete map ;
  return code;
}


////////////////////////////////////////////////////////////////////////////////
/// Declare that we handle all integrations internally

Int_t RooProduct::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars,
                 const RooArgSet* /*normSet*/,
                 const char* rangeName) const
{
  if (_forceNumInt) return 0 ;

  // Declare that we can analytically integrate all requested observables
  // (basically, we will take care of the problem, and delegate where required)
  //assert(normSet==0);
  assert(analVars.empty());
  analVars.add(allVars) ;
  Int_t code = getPartIntList(&analVars,rangeName)+1;
  return code ;
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate integral internally from appropriate partial integral cache

double RooProduct::analyticalIntegral(Int_t code, const char* rangeName) const
{
  // note: rangeName implicit encoded in code: see _cacheMgr.setObj in getPartIntList...
  CacheElem *cache = (CacheElem*) _cacheMgr.getObjByIndex(code-1);
  if (cache==0) {
    // cache got sterilized, trigger repopulation of this slot, then try again...
    std::unique_ptr<RooArgSet> vars( getParameters(RooArgSet()) );
    RooArgSet iset = _cacheMgr.selectFromSet2(*vars, code-1);
    Int_t code2 = getPartIntList(&iset,rangeName)+1;
    assert(code==code2); // must have revived the right (sterilized) slot...
    return analyticalIntegral(code2,rangeName);
  }
  assert(cache!=0);

  return calculate(cache->_prodList);
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate and return product of partial terms in partIntList

double RooProduct::calculate(const RooArgList& partIntList) const
{
  double val=1;
  for (const auto arg : partIntList) {
    const auto term = static_cast<const RooAbsReal*>(arg);
    double x = term->getVal();
    val*= x;
  }
  return val;
}


////////////////////////////////////////////////////////////////////////////////
/// Construct automatic name for internal product terms

const char* RooProduct::makeFPName(const char *pfx,const RooArgSet& terms) const
{
  static TString pname;
  pname = pfx;
  bool first(true);
  for (auto const* arg : terms) {
    if (first) { first=false;}
    else pname.Append("_X_");
    pname.Append(arg->GetName());
  }
  return pname.Data();
}



////////////////////////////////////////////////////////////////////////////////
/// Evaluate product of input functions

double RooProduct::evaluate() const
{
  double prod(1) ;

  const RooArgSet* nset = _compRSet.nset() ;
  for (const auto item : _compRSet) {
    auto rcomp = static_cast<const RooAbsReal*>(item);

    prod *= rcomp->getVal(nset) ;
  }

  for (const auto item : _compCSet) {
    auto ccomp = static_cast<const RooAbsCategory*>(item);

    prod *= ccomp->getCurrentIndex() ;
  }

  return prod ;
}


void RooProduct::computeBatch(cudaStream_t* /*stream*/, double* output, size_t nEvents, RooFit::Detail::DataMap const& dataMap) const
{
  for (unsigned int i = 0; i < nEvents; ++i) {
    output[i] = 1.;
  }

  for (const auto item : _compRSet) {
    auto rcomp = static_cast<const RooAbsReal*>(item);
    auto componentValues = dataMap.at(rcomp);

    for (unsigned int i = 0; i < nEvents; ++i) {
      output[i] *= componentValues.size() == 1 ? componentValues[0] : componentValues[i];
    }
  }

  for (const auto item : _compCSet) {
    auto ccomp = static_cast<const RooAbsCategory*>(item);
    const int catIndex = ccomp->getCurrentIndex();

    for (unsigned int i = 0; i < nEvents; ++i) {
      output[i] *= catIndex;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Forward the plot sampling hint from the p.d.f. that defines the observable obs

std::list<double>* RooProduct::binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  for (const auto item : _compRSet) {
    auto func = static_cast<const RooAbsReal*>(item);

    list<double>* binb = func->binBoundaries(obs,xlo,xhi) ;
    if (binb) {
      return binb ;
    }
  }

  return 0 ;
}


//_____________________________________________________________________________B
bool RooProduct::isBinnedDistribution(const RooArgSet& obs) const
{
  // If all components that depend on obs are binned that so is the product

  for (const auto item : _compRSet) {
    auto func = static_cast<const RooAbsReal*>(item);

    if (func->dependsOn(obs) && !func->isBinnedDistribution(obs)) {
      return false ;
    }
  }

  return true  ;
}



////////////////////////////////////////////////////////////////////////////////
/// Forward the plot sampling hint from the p.d.f. that defines the observable obs

std::list<double>* RooProduct::plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  for (const auto item : _compRSet) {
    auto func = static_cast<const RooAbsReal*>(item);

    list<double>* hint = func->plotSamplingHint(obs,xlo,xhi) ;
    if (hint) {
      return hint ;
    }
  }

  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooProduct::CacheElem::~CacheElem()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Return list of all RooAbsArgs in cache element

RooArgList RooProduct::CacheElem::containedArgs(Action)
{
  RooArgList ret(_ownedList) ;
  return ret ;
}




////////////////////////////////////////////////////////////////////////////////
/// Label OK'ed components of a RooProduct with cache-and-track

void RooProduct::setCacheAndTrackHints(RooArgSet& trackNodes)
{
  RooArgSet comp(components()) ;
  for (const auto parg : comp) {
    if (parg->isDerived()) {
      if (parg->canNodeBeCached()==Always) {
        trackNodes.add(*parg) ;
   //cout << "tracking node RooProduct component " << parg->ClassName() << "::" << parg->GetName() << endl ;
      }
    }
  }
}





////////////////////////////////////////////////////////////////////////////////
/// Customized printing of arguments of a RooProduct to more intuitively reflect the contents of the
/// product operator construction

void RooProduct::printMetaArgs(ostream& os) const
{
  bool first(true) ;

  for (const auto rcomp : _compRSet) {
    if (!first) {  os << " * " ; } else {  first = false ; }
    os << rcomp->GetName() ;
  }

  for (const auto item : _compCSet) {
    auto ccomp = static_cast<const RooAbsCategory*>(item);

    if (!first) {  os << " * " ; } else {  first = false ; }
    os << ccomp->GetName() ;
  }

  os << " " ;
}


void RooProduct::ioStreamerPass2() {
  RooAbsReal::ioStreamerPass2(); // call the baseclass method

  if(numProxies() < 2) {
    throw std::runtime_error("RooProduct::ioStreamerPass2(): the number of proxies in the proxy list should be at leat 2!");
  }

  // If the proxy data members are evolved by schema evolution, the proxy list
  // that references them will contain null pointers because the evolved
  // members are only created after the proxy list. That's why we have to set
  // them manually in that case.
  RooAbsProxy * p0 = getProxy(0);
  if(p0 == nullptr) {
    _proxyList.AddAt(&_compRSet, 0);
    p0 = &_compRSet;
  }
  RooAbsProxy * p1 = getProxy(1);
  if(p1 == nullptr) {
    _proxyList.AddAt(&_compCSet, 1);
    p1 = &_compCSet;
  }

  // If the proxies in the proxy list still don't correspond to _compRSet and
  // _compCSet, it's time to print errors. And try to recover.
  auto expectProxyIs = [this](std::size_t idx, RooAbsProxy * proxyInArg, RooListProxy * ourProxy, const char* memberName) {
    if(proxyInArg != ourProxy) {
      // From experience, it's rather the members of the RooProduct that is
      // still correct in these inconsistent cases. That's why we try to
      // recover by setting the proxy in the _proxyList to be equal to the
      // member proxy. But that might be wrong, so it's important to warn the
      // user anyway.
      _proxyList.RemoveAt(idx);
      _proxyList.AddAt(ourProxy, idx);
      std::stringstream ss;
      ss << "Problem when reading RooProduct instance \"" << GetName() << "\"!\n"
         << "     _proxyList[" << idx << "] was expected to be equal to " << memberName << ", but it's not.\n"
         << "         - proxyList[" << idx << "] : ";
      proxyInArg->print(ss, true);
      ss << "\n          - " << memberName << "   : " ;
      ourProxy->print(ss, true);
      ss << "\n    RooFit will resolve this inconsistency by making _proxyList[" << idx << "] point to " << memberName
         << ".";
      coutW(LinkStateMgmt) << ss.str() << std::endl;
    }
  };

  expectProxyIs(0, p0, &_compRSet, "_compRSet");
  expectProxyIs(1, p1, &_compCSet, "_compCSet");
}


namespace {

std::pair<RPPMIter,RPPMIter> findOverlap2nd(RPPMIter i, RPPMIter end)
{
  // Utility function finding pairs of overlapping input functions
  for (; i!=end; ++i) for ( RPPMIter j(i+1); j!=end; ++j) {
    if (i->second->overlaps(*j->second)) {
      return std::make_pair(i,j);
    }
  }
  return std::make_pair(end,end);
}


void dump_map(ostream& os, RPPMIter i, RPPMIter end)
{
  // Utility dump function for debugging
  bool first(true);
  os << " [ " ;
  for(; i!=end;++i) {
    if (first) { first=false; }
    else { os << " , " ; }
    os << *(i->first) << " -> " << *(i->second) ;
  }
  os << " ] " ;
}

}
