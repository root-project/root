 /*****************************************************************************
  * Project: RooFit                                                           *
  *                                                                           *
  * Copyright (c) 2000-2005, Regents of the University of California          *
  *                          and Stanford University. All rights reserved.    *
  *                                                                           *
  * Redistribution and use in source and binary forms,                        *
  * with or without modification, are permitted according to the terms        *
  * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
  *****************************************************************************/

/**
\file RooProjectedPdf.cxx
\class RooProjectedPdf
\ingroup Roofitcore

A RooAbsPdf implementation that represent a projection
of a given input p.d.f and the object returned by RooAbsPdf::createProjection.
The actual projection integral for it value and normalization are
calculated on the fly in getVal() once the normalization observables are known.
Class RooProjectedPdf can cache projected p.d.f.s for multiple normalization
observables simultaneously.
The createProjection() method of RooProjectedPdf is overloaded and will
return a new RooProjectedPdf that performs the projection of itself
and the requested additional projections in one integration step
The performance of <pre>f->createProjection(x)->createProjection(y)</pre>
is therefore identical to that of <pre>f->createProjection(RooArgSet(x,y))</pre>
**/

#include "Riostream.h"

#include "RooProjectedPdf.h"
#include "RooMsgService.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooNameReg.h"



////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooProjectedPdf::RooProjectedPdf() : _cacheMgr(this,10)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Construct projection of input pdf '_intpdf' over observables 'intObs'

 RooProjectedPdf::RooProjectedPdf(const char *name, const char *title, RooAbsReal& _intpdf, const RooArgSet& intObs) :
   RooAbsPdf(name,title),
   intpdf("!IntegratedPdf","intpdf",this,_intpdf,false,false),
   intobs("!IntegrationObservables","intobs",this,false,true),
   deps("!Dependents","deps",this,true,false),
   _cacheMgr(this,10)
 {
   // Since a projected PDF is an integral, we can use the same logic from
   // RooRealIntegral via the projection integral to figure out what the
   // servers are. Integration observables will be shape servers, the other
   // servers are value servers.
   int code;
   auto proj = getProjection(&intObs, nullptr, nullptr, code);
   for(RooAbsArg* server : proj->servers()) {
     if(server->isShapeServer(*proj)) {
       intobs.add(*server);
     } else if(server->isValueServer(*proj)) {
       deps.add(*server);
     }
   }
 }



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

 RooProjectedPdf::RooProjectedPdf(const RooProjectedPdf& other, const char* name) :
   RooAbsPdf(other,name),
   intpdf("!IntegratedPdf",this,other.intpdf),
   intobs("!IntegrationObservables",this,other.intobs),
   deps("!Dependents",this,other.deps),
   _cacheMgr(other._cacheMgr,this)
{
 }



////////////////////////////////////////////////////////////////////////////////
/// Evaluate projected p.d.f

double RooProjectedPdf::evaluate() const
{
  // Calculate current unnormalized value of object
  int code ;
  return getProjection(&intobs, _normSet, normRange(), code)->getVal() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Retrieve object representing projection integral of input p.d.f
/// over observables iset, while normalizing over observables
/// nset. The code argument returned by reference is the unique code
/// defining this particular projection configuration

const RooAbsReal* RooProjectedPdf::getProjection(const RooArgSet* iset, const RooArgSet* nset, const char* rangeName, int& code) const
{

  // Check if this configuration was created before
  Int_t sterileIdx(-1) ;
  if (auto cache = static_cast<CacheElem*>(_cacheMgr.getObj(iset,nset,&sterileIdx,RooNameReg::ptr(rangeName)))) {
    code = _cacheMgr.lastIndex() ;
    return static_cast<const RooAbsReal*>(cache->_projection.get());
  }

  RooArgSet nset2;
  intpdf.arg().getObservables(nset, nset2);

  if (iset) {
    nset2.add(*iset) ;
  }

  auto cache = new CacheElem ;
  cache->_projection = std::unique_ptr<RooAbsReal>{intpdf.arg().createIntegral(iset?*iset:RooArgSet(),&nset2,nullptr,rangeName)};

  code = _cacheMgr.setObj(iset, nset, cache, RooNameReg::ptr(rangeName)) ;

  coutI(Integration) << "RooProjectedPdf::getProjection(" << GetName() << ") creating new projection "
                     << cache->_projection->GetName() << " with code " << code << std::endl;

  return cache->_projection.get();
}



////////////////////////////////////////////////////////////////////////////////
/// Special version of RooAbsReal::createProjection that deals with
/// projections of projections. Instead of integrating twice, a new
/// RooProjectedPdf is returned that is configured to perform the
/// complete integration in one step

RooAbsPdf* RooProjectedPdf::createProjection(const RooArgSet& iset)
{
  RooArgSet combiset(iset) ;
  combiset.add(intobs) ;
  return static_cast<RooAbsPdf&>( const_cast<RooAbsReal&>(intpdf.arg()) ).createProjection(combiset) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Force RooRealIntegral to relegate integration of all observables to internal logic

bool RooProjectedPdf::forceAnalyticalInt(const RooAbsArg& /*dep*/) const
{
  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Mark all requested variables as internally integrated

Int_t RooProjectedPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName) const
{
  analVars.add(allVars) ;

  // Create the appropriate integral
  int code ;
  RooArgSet allVars2(allVars) ;
  allVars2.add(intobs) ;
  getProjection(&allVars2,normSet,rangeName,code) ;

  return code+1 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return analytical integral represent by appropriate element of projection cache

double RooProjectedPdf::analyticalIntegralWN(Int_t code, const RooArgSet* /*normSet*/, const char* rangeName) const
{
  if (auto cache = static_cast<CacheElem*>(_cacheMgr.getObjByIndex(code-1))) {
    return cache->_projection->getVal() ;
  } else {

    RooArgSet vars;
    getParameters(nullptr, vars);
    vars.add(intobs) ;
    RooArgSet iset = _cacheMgr.selectFromSet1(vars, code-1) ;
    RooArgSet nset = _cacheMgr.selectFromSet2(vars, code-1) ;

    int code2 = -1 ;

    return getProjection(&iset,&nset,rangeName,code2)->getVal() ;
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Intercept a server redirection all and update list of dependents if necessary
/// Specifically update the set proxy 'deps' which introduces the dependency
/// on server value dirty flags of ourselves

bool RooProjectedPdf::redirectServersHook(const RooAbsCollection& newServerList, bool mustReplaceAll,
                                          bool nameChange, bool isRecursive)
{
  // Redetermine explicit list of dependents if intPdf is being replaced
  if (RooAbsArg* newPdf = newServerList.find(intpdf.arg().GetName())) {

    // Determine if set of dependents of new p.d.f is different from old p.d.f.
    RooArgSet olddeps(deps) ;
    RooArgSet newdeps;
    newPdf->getParameters(&intobs, newdeps);
    RooArgSet common;
    newdeps.selectCommon(deps, common);
    newdeps.remove(common,true,true) ;
    olddeps.remove(common,true,true) ;

    // If so, adjust composition of deps Listproxy
    if (!newdeps.empty()) {
      deps.add(newdeps) ;
    }
    if (!olddeps.empty()) {
      deps.remove(olddeps,true,true) ;
    }
  }

  return RooAbsPdf::redirectServersHook(newServerList, mustReplaceAll, nameChange, isRecursive);
}



////////////////////////////////////////////////////////////////////////////////
/// Return RooAbsArg elements contained in projection cache element.

RooArgList RooProjectedPdf::CacheElem::containedArgs(Action)
{
  return *_projection;
}



////////////////////////////////////////////////////////////////////////////////
/// Customized printing of arguments of a RooRealIntegral to more intuitively reflect the contents of the
/// integration operation

void RooProjectedPdf::printMetaArgs(std::ostream& os) const
{
  os << "Int " << intpdf.arg().GetName() << " d"  << intobs << " ";
}




////////////////////////////////////////////////////////////////////////////////
/// Print contents of cache when printing self as part of object tree

void RooProjectedPdf::CacheElem::printCompactTreeHook(std::ostream& os, const char* indent, Int_t curElem, Int_t maxElem)
{
  if (curElem==0) {
    os << indent << "RooProjectedPdf begin projection cache" << std::endl ;
  }

  TString indent2(indent) ;
  indent2 += Form("[%d] ",curElem) ;

  _projection->printCompactTree(os,indent2) ;

  if(curElem==maxElem) {
    os << indent << "RooProjectedPdf end projection cache" << std::endl ;
  }
}


std::unique_ptr<RooAbsArg>
RooProjectedPdf::compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext &ctx) const
{
   RooArgSet nset2;
   intpdf->getObservables(&normSet, nset2);
   nset2.add(intobs);

   auto newArg = std::unique_ptr<RooAbsReal>{intpdf->createIntegral(intobs, &nset2)};
   ctx.markAsCompiled(*newArg);
   return newArg;
}
