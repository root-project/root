/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealIntegral.cc,v 1.43 2001/10/01 22:13:49 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooRealIntegral performs hybrid numerical/analytical integrals of RooAbsReal objects
// The class performs none of the actual integration, but only manages the logic
// of what variables can be integrated analytically, accounts for eventual jacobian
// terms and defines what numerical integrations needs to be done to complement the
// analytical integral.
//
// The actual analytical integrations (if any) are done in the PDF themselves, the numerical
// integration is performed in the various implemenations of the RooAbsIntegrator base class.

#include <iostream.h>
#include "TObjString.h"
#include "TH1.h"
#include "RooFitCore/RooRealIntegral.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooAbsRealLValue.hh"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooMultiCatIter.hh"
#include "RooFitCore/RooIntegrator1D.hh"
#include "RooFitCore/RooImproperIntegrator1D.hh"
#include "RooFitCore/RooMCIntegrator.hh"
#include "RooFitCore/RooRealBinding.hh"
#include "RooFitCore/RooRealAnalytic.hh"
#include "RooFitCore/RooInvTransform.hh"

ClassImp(RooRealIntegral) 
;


RooRealIntegral::RooRealIntegral(const char *name, const char *title, 
				 const RooAbsReal& function, const RooArgSet& depList,
				 const RooArgSet* funcNormSet) :
  RooAbsReal(name,title), _mode(0),
  _function("function","Function to be integrated",this,
	    const_cast<RooAbsReal&>(function),kFALSE,kFALSE), 
  _sumList("sumList","Categories to be summed numerically",this,kFALSE,kFALSE), 
  _intList("intList","Variables to be integrated numerically",this,kFALSE,kFALSE), 
  _anaList("anaList","Variables to be integrated analytically",this,kFALSE,kFALSE), 
  _jacList("jacList","Jacobian product term",this,kFALSE,kFALSE), 
  _facList("facList","Variables independent of function",this,kFALSE,kTRUE),
  _numIntEngine(0), _numIntegrand(0), _operMode(Hybrid), _valid(kTRUE)
{
  // Constructor - Performs structural analysis of the integrand

  //   A) Check that all dependents are lvalues 
  //
  //   B) Check if list of dependents can be re-expressed in        
  //      lvalues that are higher in the expression tree            
  //
  //   C) Check for dependents that the PDF insists on integrating  
  //      analytically iself                                        
  //
  //   D) Make list of servers that can be integrated analytically  
  //      Add all parameters/dependents as value/shape servers      
  //
  //   E) Interact with function to make list of objects actually integrated analytically   
  //
  //   F) Make list of numerical integration variables consisting of:               
  //     - Category dependents of RealLValues in analytical integration             
  //     - Leaf nodes server lists of function server that are not analytically integrated   
  //     - Make Jacobian list for analytically integrated RealLValues            
  //
  //   G) Split numeric list in integration list and summation list   
  //

  // Save private copy of funcNormSet, if supplied
  _funcNormSet = funcNormSet ? (RooArgSet*)funcNormSet->snapshot(kFALSE) : 0 ;
  
  // Delete all factorizing terms from funcNormSet
  if (_funcNormSet) {
    TIterator* iter = _funcNormSet->createIterator() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)iter->Next()) {
      if (!function.dependsOn(*arg)) _funcNormSet->remove(*arg) ;
    }
    delete iter ;
  }

  // Make internal copy of dependent list
  RooArgSet intDepList(depList) ;
  RooAbsArg *arg ;

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * A) Check that all dependents are lvalues and filter out any
  //      dependents that the PDF doesn't explicitly depend on
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  
  TIterator* depIter = intDepList.createIterator() ;
  while(arg=(RooAbsArg*)depIter->Next()) {
    if(!arg->isLValue()) {
      cout << ClassName() << "::" << GetName() << ": cannot integrate non-lvalue ";
      arg->Print("1");
      _valid= kFALSE;
    }
    if (!function.dependsOn(*arg)) {
      _facList.add(*arg) ;
      addServer(*arg,kFALSE,kTRUE) ;
    }
  }


  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * B) Check if list of dependents can be re-expressed in       *
  // *    lvalues that are higher in the expression tree           *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 


  // Initial fill of list of LValue branches
  RooArgSet exclLVBranches("exclLVBranches") ;
  RooArgSet branchList ;
  function.branchNodeServerList(&branchList) ;
  TIterator* bIter = branchList.createIterator() ;
  RooAbsArg* branch ;
  while(branch=(RooAbsArg*)bIter->Next()) {
    RooAbsRealLValue    *realArgLV = dynamic_cast<RooAbsRealLValue*>(branch) ;
    RooAbsCategoryLValue *catArgLV = dynamic_cast<RooAbsCategoryLValue*>(branch) ;
    if ((realArgLV && (realArgLV->isJacobianOK(intDepList)!=0)) || catArgLV) {
      exclLVBranches.add(*branch) ;
    }
  }
  delete bIter ;
  exclLVBranches.remove(depList,kTRUE,kTRUE) ;

  // Initial fill of list of LValue leaf servers (put in intDepList)
  RooArgSet exclLVServers("exclLVServers") ;
  exclLVServers.add(intDepList) ;

  // Obtain mutual exclusive dependence by iterative reduction
  TIterator *sIter = exclLVServers.createIterator() ;
  bIter = exclLVBranches.createIterator() ;
  RooAbsArg *server ;
  Bool_t converged(kFALSE) ;
  while(!converged) {
    converged=kTRUE ;

    // Reduce exclLVServers to only those serving exclusively exclLVBranches
    sIter->Reset() ;
    while (server=(RooAbsArg*)sIter->Next()) {
      if (!servesExclusively(server,exclLVBranches)) {
	exclLVServers.remove(*server) ;
	converged=kFALSE ;
      }
    }
    
    // Reduce exclLVBranches to only those depending exclusisvely on exclLVservers
    bIter->Reset() ;
    while(branch=(RooAbsArg*)bIter->Next()) {
      RooArgSet* brDepList = branch->getDependents(&intDepList) ;
      RooArgSet bsList(*brDepList,"bsList") ;
      delete brDepList ;
      bsList.remove(exclLVServers,kTRUE,kTRUE) ;
      if (bsList.getSize()>0) {
	exclLVBranches.remove(*branch,kTRUE,kTRUE) ;
	converged=kFALSE ;
      }
    }
  }
  delete sIter ;
  delete bIter ;
     
  // Replace exclusive lvalue branch servers with lvalue branches
  if (exclLVServers.getSize()>0) {
    intDepList.remove(exclLVServers) ;
    intDepList.add(exclLVBranches) ;
  }
     
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * C) Check for dependents that the PDF insists on integrating *
  //      analytically iself                                       *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  RooArgSet anIntOKDepList ;
  depIter = intDepList.createIterator() ;
  while(arg=(RooAbsArg*)depIter->Next()) {
    if (function.forceAnalyticalInt(*arg)) {
      anIntOKDepList.add(*arg) ;
    }
  }
  delete depIter ;

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * D) Make list of servers that can be integrated analytically *
  //      Add all parameters/dependents as value/shape servers     *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  
  sIter = function.serverIterator() ;
  while(arg=(RooAbsArg*)sIter->Next()) {

    // Dependent or parameter?
    if (!arg->dependsOn(intDepList)) {

      // Add parameter as value server
      addServer(*arg,kTRUE,kFALSE) ;
      continue ;

    } else {

      // Add final dependents of arg as shape servers
      RooArgSet argLeafServers ;
      arg->leafNodeServerList(&argLeafServers) ;

      TIterator* lIter = argLeafServers.createIterator() ;
      RooAbsArg* leaf ;
      while(leaf=(RooAbsArg*)lIter->Next()) {
	if (depList.find(leaf->GetName())) {
	  addServer(*leaf,kFALSE,kTRUE) ;
	} else {
	  addServer(*leaf,kTRUE,kFALSE) ;
	}	
      }
      delete lIter ;
    }

    // If this dependent arg is self-normalized, stop here
    //if (function.selfNormalized()) continue ;
    Bool_t depOK(kFALSE) ;
    // Check for integratable AbsRealLValue
    if (arg->isDerived()) {
      RooAbsRealLValue    *realArgLV = dynamic_cast<RooAbsRealLValue*>(arg) ;
      RooAbsCategoryLValue *catArgLV = dynamic_cast<RooAbsCategoryLValue*>(arg) ;
      if ((realArgLV && intDepList.find(realArgLV->GetName()) && (realArgLV->isJacobianOK(intDepList)!=0)) || catArgLV) {
	
	// Derived LValue with valid jacobian
	depOK = kTRUE ;
	
	// Now, check for overlaps
	Bool_t overlapOK = kTRUE ;
	RooAbsArg *otherArg ;
	TIterator* sIter2 = function.serverIterator() ;	
	while(otherArg=(RooAbsArg*)sIter2->Next()) {
	  // skip comparison with self
	  if (arg==otherArg) continue ;
	  if (arg->overlaps(*otherArg)) {
	    overlapOK=kFALSE ;
	  }
	}      	
	if (!overlapOK) depOK=kFALSE ;      

	delete sIter2 ;
      }
    } else {
      // Fundamental types are always OK
      depOK = kTRUE ;
    }
    
    // Add server to list of dependents that are OK for analytical integration
    if (depOK) {
      anIntOKDepList.add(*arg,kTRUE) ;      
    }
  }
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * E) interact with function to make list of objects actually integrated analytically  *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  RooArgSet anIntDepList ;
  _mode = ((RooAbsReal&)_function.arg()).getAnalyticalIntegralWN(anIntOKDepList,_anaList,_funcNormSet) ;    

  // WVE kludge: synchronize dset for use in analyticalIntegral
  function.getVal(funcNormSet) ;

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * F) Make list of numerical integration variables consisting of:            *  
  // *   - Category dependents of RealLValues in analytical integration          *  
  // *   - Expanded server lists of server that are not analytically integrated  *
  // *    Make Jacobian list with analytically integrated RealLValues            *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  RooArgSet numIntDepList ;

  // Loop over actually analytically integrated dependents
  TIterator* aiIter = _anaList.createIterator() ;
  while (arg=(RooAbsArg*)aiIter->Next()) {    

    // Process only derived RealLValues
    if (arg->IsA()->InheritsFrom(RooAbsRealLValue::Class()) && arg->isDerived()) {

      // Add to list of Jacobians to calculate
      _jacList.add(*arg) ;

      // Add category dependent of LValueReal used in integration
      RooAbsArg *argDep ;
      RooArgSet *argDepList = arg->getDependents(&intDepList) ;
      TIterator *adIter = argDepList->createIterator() ;
      while (argDep=(RooAbsArg*)adIter->Next()) {
	if (argDep->IsA()->InheritsFrom(RooAbsCategoryLValue::Class())) {
	  numIntDepList.add(*argDep) ;
	}
      }
      delete adIter ;
      delete argDepList ;
    }
  }
  delete aiIter ;

  // Loop again over function servers to add remaining numeric integrations
  sIter->Reset() ;
  while(arg=(RooAbsArg*)sIter->Next()) {

    // Process only servers that are not treated analytically
    if (!_anaList.find(arg->GetName()) && arg->dependsOn(intDepList)) {

      // Process only derived RealLValues
      if (dynamic_cast<RooAbsLValue*>(arg) && arg->isDerived()) {
	numIntDepList.add(*arg) ;	
      } else {
	
	// Expand server in final dependents 
	RooArgSet *argDeps = arg->getDependents(&intDepList) ;
	
	// Add final dependents, that are not forcibly integrated analytically, 
	// to numerical integration list      
	TIterator* iter = argDeps->createIterator() ;
	RooAbsArg* dep ;
	while(dep=(RooAbsArg*)iter->Next()) {
	  if (!_anaList.find(dep->GetName())) {
	    numIntDepList.add(*dep) ;
	  }
	}      
	delete iter ;
	delete argDeps ; 
      }

    }
  }
  delete sIter ;

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * G) Split numeric list in integration list and summation list  *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  // Split numeric integration list in summation and integration lists
  TIterator* numIter=numIntDepList.createIterator() ;
  while (arg=(RooAbsArg*)numIter->Next()) {

    if (arg->IsA()->InheritsFrom(RooAbsRealLValue::Class())) {
      _intList.add(*arg) ;
    } else if (arg->IsA()->InheritsFrom(RooAbsCategoryLValue::Class())) {
      _sumList.add(*arg) ;
    }
  }
  delete numIter ;

  // Determine operating mode
  if (numIntDepList.getSize()>0) {
    // Numerical and optional Analytical integration
    _operMode = Hybrid ;
  } else if (_anaList.getSize()>0) {
    // Purely analytical integration
    _operMode = Analytic ;    
  } else {
    // No integration performed
    _operMode = PassThrough ;
  }
}


Bool_t RooRealIntegral::servesExclusively(const RooAbsArg* server,const RooArgSet& exclLVBranches) const
{  
  // Determine if given server serves exclusively exactly one of the given nodes in exclLVBranches

  // Special case, no LV servers available
  if (exclLVBranches.getSize()==0) return kFALSE ;

  // If server has no clients and is not an LValue itself, return false
   if (server->_clientList.GetSize()==0 && exclLVBranches.find(server->GetName())) {
     return kFALSE ;
   }

   // Loop over all clients
   Bool_t ret(kTRUE) ;
   Int_t numLVServ(0) ;
   RooAbsArg* client ;
   TIterator* cIter = server->clientIterator() ;
   while(client=(RooAbsArg*)cIter->Next()) {
     // If client is not an LValue, recurse
     if (!exclLVBranches.find(client->GetName())) {
       if (!servesExclusively(client,exclLVBranches)) {
	 // Client is a non-LValue that doesn't have an exclusive LValue server
	 ret = kFALSE ;
       }
     } else {
       // Client is an LValue       
       numLVServ++ ;
     }
   }

   delete cIter ;
   return (numLVServ==1) ;
}




Bool_t RooRealIntegral::initNumIntegrator() const
{
  // (Re)Initialize numerical integration engine if necessary. Return kTRUE if
  // successful, or otherwise kFALSE.

  // if we already have an engine, check if it still works for the present limits.
  if(0 != _numIntEngine) {
    if(_numIntEngine->isValid() && _numIntEngine->checkLimits()) return kTRUE;
    // otherwise, cleanup the old engine
    delete _numIntEngine ;
    _numIntEngine= 0;
    if(0 != _numIntegrand) {
      delete _numIntegrand;
      _numIntegrand= 0;
    }
  }

  // All done if there are no arguments to integrate numerically
  if(0 == _intList.getSize()) return kTRUE;

  // Bind the appropriate analytic integral (specified by _mode) of our RooRealVar object to
  // those of its arguments that will be integrated out numerically.
  if(_mode != 0) {
    _numIntegrand= new RooRealAnalytic(_function.arg(),_intList,_mode,_funcNormSet);
  }
  else {
    _numIntegrand= new RooRealBinding(_function.arg(),_intList,_funcNormSet);
  }
  if(0 == _numIntegrand || !_numIntegrand->isValid()) {
    cout << ClassName() << "::" << GetName() << ": failed to create valid integrand." << endl;
    return kFALSE;
  }

  // Chose a suitable RooAbsIntegrator implementation
  if(_numIntegrand->getDimension() == 1) {
    if(RooNumber::isInfinite(_numIntegrand->getMinLimit(0)) ||
       RooNumber::isInfinite(_numIntegrand->getMaxLimit(0))) {
      _numIntEngine= new RooImproperIntegrator1D(*_numIntegrand);
    }
    else {
      _numIntEngine= new RooIntegrator1D(*_numIntegrand);
    }
  }
  else {
    // let the constructor check that the domain is finite
    _numIntEngine= new RooMCIntegrator(*_numIntegrand);
  }
  if(0 == _numIntEngine || !_numIntEngine->isValid()) {
    cout << ClassName() << "::" << GetName() << ": failed to create valid integrator." << endl;
    return kFALSE;
  }

  return kTRUE;
}

RooRealIntegral::RooRealIntegral(const RooRealIntegral& other, const char* name) : 
  RooAbsReal(other,name), _mode(other._mode),
 _function("function",this,other._function), 
  _intList("intList",this,other._intList), 
  _sumList("sumList",this,other._sumList),
  _anaList("anaList",this,other._anaList),
  _jacList("jacList",this,other._jacList),
  _facList("facList",this,other._facList),
  _operMode(other._operMode), _numIntEngine(0), _numIntegrand(0), _valid(other._valid)
{
  // Copy constructor
 _funcNormSet = other._funcNormSet ? (RooArgSet*)other._funcNormSet->snapshot(kFALSE) : 0 ;
}


RooRealIntegral::~RooRealIntegral()
  // Destructor
{
  if (_numIntEngine) delete _numIntEngine ;
  if (_numIntegrand) delete _numIntegrand ;
  if (_funcNormSet) delete _funcNormSet ;
}



Double_t RooRealIntegral::evaluate() const 
{
  if (_function.arg().operMode()==RooAbsArg::AClean) {
    if (RooAbsPdf::_verboseEval>1) 
      cout << "RooRealIntegral::evaluate(" << GetName() 
	   << ") integrand is AClean, returning cached value of " << _value << endl ;
    return _value ;
  }

  Double_t retVal ;
  switch (_operMode) {
    
  case Hybrid: 
    {
      // Find any function dependents that are AClean 
      // and switch them temporarily to ADirty
      prepareACleanFunc() ;

      // try to initialize our numerical integration engine
      if(!(_valid= initNumIntegrator())) {
	cout << ClassName() << "::" << GetName()
	     << ":evaluate: cannot initialize numerical integrator" << endl;
	return 0;
      }

      // Save current integral dependent values 
      RooArgSet *saveInt = (RooArgSet*) _intList.snapshot() ;
      RooArgSet *saveSum = (RooArgSet*) _sumList.snapshot() ;
      
      // Evaluate sum/integral
      retVal = sum() / jacobianProduct() ;
      
      // Restore integral dependent values
      _intList=*saveInt ;
      _sumList=*saveSum ;
      delete saveInt ;
      delete saveSum ;

      restoreACleanFunc() ;
      break ;
    }
  case Analytic:
    {
      retVal =  ((RooAbsReal&)_function.arg()).analyticalIntegralWN(_mode,_funcNormSet) ;
      if (RooAbsPdf::_verboseEval>0)
	cout << "RooRealIntegral::evaluate_analytic(" << GetName() 
	     << ")func = " << _function.arg().IsA()->GetName() << "::" << _function.arg().GetName()
	     << " raw = " << retVal << endl ;
      break ;
    }

  case PassThrough:
    {
      retVal= _function.arg().getVal(_funcNormSet) ;
      break ;
    }
  }

  

  // Multiply answer with integration ranges of factorized variables
  RooAbsArg *arg ;
  TIterator* fIter = _facList.createIterator() ; // WVE persist facList iterator
  while(arg=(RooAbsArg*)fIter->Next()) {
    // Multiply by fit range for 'real' dependents
    if (arg->IsA()->InheritsFrom(RooAbsRealLValue::Class())) {
      RooAbsRealLValue* argLV = (RooAbsRealLValue*)arg ;
      retVal *= (argLV->getFitMax() - argLV->getFitMin()) ;
    }
    // Multiply by number of states for category dependents
    if (arg->IsA()->InheritsFrom(RooAbsCategoryLValue::Class())) {
      RooAbsCategoryLValue* argLV = (RooAbsCategoryLValue*)arg ;
      retVal *= argLV->numTypes() ;
    }    
  }
  delete fIter ;
 

  if (RooAbsPdf::_verboseEval>0)
    cout << "RooRealIntegral::evaluate(" << GetName() << ") raw*fact = " << retVal << endl ;

  return retVal ;
}


void RooRealIntegral::prepareACleanFunc() const
{
  // Clear previous list contents
  _funcACleanBranchList.removeAll() ;

  // Get all function branch nodes
  _function.arg().branchNodeServerList(&_funcACleanBranchList) ;

  // Remove non-AClean branches from list 
  TIterator* iter = _funcACleanBranchList.createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()) {
    if (arg->operMode()!=RooAbsArg::AClean) {
      _funcACleanBranchList.remove(*arg) ;
    } else {
      arg->setOperMode(RooAbsArg::ADirty) ;
    }
  }
  delete iter ;
}



void RooRealIntegral::restoreACleanFunc() const
{
  // Restore formerly AClean branches to their AClean state
  TIterator* iter = _funcACleanBranchList.createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()) {
      arg->setOperMode(RooAbsArg::AClean) ;
  }
  delete iter ;
}


Double_t RooRealIntegral::jacobianProduct() const 
{
  // Return product of jacobian terms originating from analytical integration
  Double_t jacProd(1) ;

  TIterator *jIter = _jacList.createIterator() ;
  RooAbsRealLValue* arg ;
  while (arg=(RooAbsRealLValue*)jIter->Next()) {
    jacProd *= arg->jacobian() ;
  }
  delete jIter ;

  return jacProd ;
}


Double_t RooRealIntegral::sum() const
{
  // Perform summation of list of category dependents to be integrated
  if (_sumList.getSize()!=0) {
 
    // Add integrals for all permutations of categories summed over
    Double_t total(0) ;
    RooMultiCatIter sumIter(_sumList) ;
    Int_t counter(0) ;

    while(sumIter.Next()) {
      total += integrate() / jacobianProduct() ;
    }
    return total ;

  } else {

    // Simply return integral 
    Double_t ret = integrate() ;
    return ret ;
  }
}




Double_t RooRealIntegral::integrate() const
{
  // Perform hybrid numerical/analytical integration over all real-valued dependents

  if (!_numIntEngine) {
    // Trivial case, fully analytical integration
    return ((RooAbsReal&)_function.arg()).analyticalIntegralWN(_mode,_funcNormSet) ;
  }
  else {
    // Partial or complete numerical integration
//     if(_intList.getSize() > 1) {
//       cout << "RooRealIntegral: Integrating out ";
//       _intList.printToStream(cout,OneLine);
//     }
    return _numIntEngine->integral() ;
  }
}




Bool_t RooRealIntegral::isValidReal(Double_t value, Bool_t printError) const 
{
  // Check if current value is valid
  return kTRUE ;
}


Bool_t RooRealIntegral::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll)
{
  return kFALSE ;
}


void RooRealIntegral::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print the state of this object to the specified output stream.
  RooAbsReal::printToStream(os,opt,indent) ;
  if (opt==Verbose) {
    os << indent << "--- RooRealIntegral ---" << endl;
    os << indent << "  Integrates ";
    _function.arg().printToStream(os,Standard,indent);
    TString deeper(indent);
    deeper.Append("  ");
    os << indent << "  operating mode is " 
       << (_operMode==Hybrid?"Hybrid":(_operMode==Analytic?"Analytic":"PassThrough")) << endl ;
    os << indent << "  Summed discrete args are ";
    _sumList.printToStream(os,Standard,deeper);
    os << indent << "  Numerically integrated args are ";
    _intList.printToStream(os,Standard,deeper);
    os << indent << "  Analytically integrated args using mode " << _mode << " are ";
    _anaList.printToStream(os,Standard,deeper);
    os << indent << "  Arguments included in Jacobean are ";
    _jacList.printToStream(os,Standard,deeper);
    os << indent << "  Factorized arguments are ";
    _facList.printToStream(os,Standard,deeper);
    os << indent << "  Function normalization set " ;
    if (_funcNormSet) 
      _funcNormSet->Print("1") ; 
    else
      os << "<none>" ;

    os << endl ;
    return ;
  }
} 


