/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooConvolutedPdf.cc,v 1.20 2001/10/05 07:01:49 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// 
//  RooConvolutedPdf is the base class of for PDFs that represents a
//  physics model that can be convoluted with a resolution model
//  
//  To achieve factorization between the physics model and the resolution
//  model, each physics model must be able to be written in the form
//           _ _                 _              _ 
//    Phys(x,a,b) = Sum_k coef_k(a) * basis_k(x,b)
//  
//  where basis_k are a limited number of functions in terms of the variable
//  to be convoluted and coef_k are coefficients independent of the convolution
//  variable.
//  
//  Classes derived from RooResolutionModel implement 
//         _ _                        _                  _
//   R_k(x,b,c) = Int(dx') basis_k(x',b) * resModel(x-x',c)
// 
//  which RooConvolutedPdf uses to construct the pdf for [ Phys (x) R ] :
//          _ _ _                 _          _ _
//    PDF(x,a,b,c) = Sum_k coef_k(a) * R_k(x,b,c)
//
//  A minimal implementation of a RooConvolutedPdf physics model consists of
//  
//  - A constructor that declares the required basis functions using the declareBasis() method.
//    The declareBasis() function assigns a unique identifier code to each declare basis
//
//  - An implementation of coefficient(Int_t code) returning the coefficient value for each
//    declared basis function
//
//  Optionally, analytical integrals can be provided for the coefficient functions. The
//  interface for this is quite similar to that for integrals of regular PDFs. Two functions,
//
//   Int_t getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) 
//   Double_t coefAnalyticalIntegral(Int_t coef, Int_t code),
//
//  advertise the coefficient integration capabilities and implement them respectively.
//  Please see RooAbsPdf for additional details. Advertised analytical integrals must be
//  valid for all coefficients.


#include <iostream.h>
#include "RooFitCore/RooConvolutedPdf.hh"
#include "RooFitCore/RooResolutionModel.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooFormulaVar.hh"

ClassImp(RooConvolutedPdf) 
;


RooConvolutedPdf::RooConvolutedPdf(const char *name, const char *title, 
				   const RooResolutionModel& model, RooRealVar& convVar) :
  RooAbsPdf(name,title), _isCopy(kFALSE),
  _model((RooResolutionModel*)&model), _convVar((RooRealVar*)&convVar),
  _convSet("convSet","Set of resModel X basisFunc convolutions",this),
  _convNormSet(0), _convSetIter(_convSet.createIterator()),
  _codeReg(10)
{
  // Constructor. The supplied resolution model must be constructed with the same
  // convoluted variable as this physics model ('convVar')
  _convNormSet = new RooArgSet(convVar,"convNormSet") ;
}


RooConvolutedPdf::RooConvolutedPdf(const RooConvolutedPdf& other, const char* name) : 
  RooAbsPdf(other,name), _model(0), _convVar(0), _isCopy(kTRUE),
  _convSet("convSet",this,other._convSet),
  _convNormSet(new RooArgSet(*other._convNormSet)),
  _convSetIter(_convSet.createIterator()),
  _codeReg(other._codeReg)
{
  // Copy constructor
}



RooConvolutedPdf::~RooConvolutedPdf()
{
  // Destructor
  if (_convNormSet) {
    delete _convNormSet ;
  }
    
  delete _convSetIter ;

  if (!_isCopy) {
    TIterator* iter = _convSet.createIterator() ;
    RooAbsArg* arg ;
    while (arg = (RooAbsArg*)iter->Next()) {
      _convSet.remove(*arg) ;
      delete arg ;
    }
    delete iter ;
  }

  // Delete all basis functions we created 
  _basisList.Delete() ;
}


Int_t RooConvolutedPdf::declareBasis(const char* expression, const RooArgSet& params) 
{
  // Declare a basis function for use in this physics model. The string expression 
  // must be a valid RooFormulVar expression representing the basis function, referring
  // to the convolution variable as '@0', and any additional parameters (supplied in
  // 'params' as '@1','@2' etc.
  //
  // The return value is a unique identifier code, that will be passed to coefficient()
  // to identify the basis function for which the coefficient is requested. If the
  // resolution model used does not support the declared basis function, code -1 is
  // returned. 
  //

  // Sanity check
  if (!_model || !_convVar) {
    cout << "RooConvolutedPdf::declareBasis(" << GetName() << "): ERROR attempt to "
	 << " declare basis functions in a copied RooConvolutedPdf" << endl ;
    return -1 ;
  }

  // Resolution model must support declared basis
  if (!_model->isBasisSupported(expression)) {
    cout << "RooConvolutedPdf::declareBasis(" << GetName() << "): resolution model " 
	 << _model->GetName() 
	 << " doesn't support basis function " << expression << endl ;
    return -1 ;
  }

  // Instantiate basis function
  RooArgSet basisArgs(*_convVar) ;
  basisArgs.add(params) ;
  RooFormulaVar* basisFunc = new RooFormulaVar(expression,expression,basisArgs) ;
  _basisList.Add(basisFunc) ;

  // Instantiate resModel x basisFunc convolution
  RooAbsReal* conv = _model->convolution(basisFunc,this) ;
  if (!conv) {
    cout << "RooConvolutedPdf::declareBasis(" << GetName() << "): unable to construct convolution with basis function '" 
	 << expression << "'" << endl ;
    return -1 ;
  }
  _convSet.add(*conv) ;

  return _convSet.index(conv) ;
}



const RooRealVar* RooConvolutedPdf::convVar() const
{
  // Return a pointer to the convolution variable instance used in the resolution model
  RooResolutionModel* conv = (RooResolutionModel*) _convSet.at(0) ;
  if (!conv) return 0 ;  
  return &conv->convVar() ;
}



Double_t RooConvolutedPdf::evaluate() const
{
  // Calculate the current unnormalized value of the PDF
  //
  // PDF = sum_k coef_k * [ basis_k (x) ResModel ]
  //
  Double_t result(0) ;

  _convSetIter->Reset() ;
  RooAbsPdf* conv ;
  Int_t index(0) ;
  while(conv=(RooAbsPdf*)_convSetIter->Next()) {
    Double_t coef = coefficient(index++) ;
    if (coef!=0.) {
      result += conv->getVal(0)*coef ;
   }
  }
  
  return result ;
}


RooArgSet* RooConvolutedPdf::parseIntegrationRequest(const RooArgSet& intSet, Int_t& coefCode, RooArgSet* analVars) const
{
  // Auxiliary function for getAnalyticalIntegral parses integrated request represented by 'intSet' 
  // The integration request is split in a set of coefficient dependents and a set of convoluted dependents
  // From the coefficient set, the analytical integration code of the coefficient is determined.
  // A clone of convolution dependents set is passed as return value.

  // Split allVars in a list for coefficients and a list for convolutions
  RooArgSet allVarsCoef("allVarsCoef") ;
  RooArgSet allVarsConv("allVarsConvInt") ;

  RooAbsArg* arg ;
  RooAbsArg* conv ;

  TIterator* varIter  = intSet.createIterator() ;
  TIterator* convIter = _convSet.createIterator() ;
  while(arg=(RooAbsArg*) varIter->Next()) {
    Bool_t ok(kTRUE) ;
    convIter->Reset() ;
    while(conv=(RooAbsArg*) convIter->Next()) {
      
      if (conv->dependsOn(*arg)) ok=kFALSE ;
    }

    if (ok) {
      allVarsCoef.add(*arg) ;
    } else {
      allVarsConv.add(*arg) ;
    }

  }
  delete varIter ;
  delete convIter ;

  // Get analytical integration code for coefficients
  if (analVars) {
    coefCode = getCoefAnalyticalIntegral(allVarsCoef,*analVars) ;
  } else {
    RooArgSet tmp ;
    coefCode = getCoefAnalyticalIntegral(allVarsCoef,tmp) ;
  }

  // If convolution integration set is empty, return null ptr, otherwise return heap clone of set
  return allVarsConv.getSize()? new RooArgSet(allVarsConv) : 0 ;
}



Int_t RooConvolutedPdf::getAnalyticalIntegralWN(RooArgSet& allVars, 
	  				        RooArgSet& analVars, const RooArgSet* normSet) const 
{
  // Determine which part given integral can be performed analytically.
  //
  // A RooConvolutedPdf will always claim to analytically integrate all dependents of the
  // basis function (x) resolution model convolutions. These dependents are not actually
  // integrated in analyticalIntegralWN() but passed down the the resolution model objects
  // which will process these integrals internally.
  //
  // The ability to handle integration of coefficient depedents is determined from 
  // getCoefAnalyticalIntegral().
  //
  // The entire procedure is repeated twice: once for the integral requested and once for
  // the normalization of that integral. The output, 2 coefficient integration codes and
  // two sets of integration dependents to pass down to the resolution model objects,
  // are stored in an AICRegistry objects, which assigns a unique 'master' integration
  // code for each configuration.


  // Handle trivial no-integration scenario
  if (allVars.getSize()==0) return 0 ;

  // Process integration request
  Int_t intCoefCode(0) ;
  RooArgSet *intConvSet = parseIntegrationRequest(allVars,intCoefCode,&analVars) ;

  // Process normalization if integration request
  Int_t normCoefCode(0) ;    
  RooArgSet *normConvSet(0) ;
  if (normSet) {
    normConvSet = parseIntegrationRequest(*normSet,normCoefCode) ;
  }
  
  // Optional messaging
  if (_verboseEval>0) {
    cout << "RooConvolutedPdf::getAI(" << GetName() << ") coefficients integrate analytically " ; analVars.Print("1") ;
    cout << "RooConvolutedPdf::getAI(" << GetName() << ") intCoefCode  = " << intCoefCode << endl ;
    cout << "RooConvolutedPdf::getAI(" << GetName() << ") normCoefCode = " << normCoefCode << endl ;
    cout << "RooConvolutedPdf::getAI(" << GetName() << ") intConvSet  = " ; 
    if (intConvSet) intConvSet->Print("1") ; else cout << "<none>" << endl ;
    cout << "RooConvolutedPdf::getAI(" << GetName() << ") normConvSet  = " ; 
    if (normConvSet) normConvSet->Print("1") ; else cout << "<none>" << endl ;
  }

  // Register convolution dependents integrated as analytical
  if (intConvSet) analVars.add(*intConvSet) ;

  // Store integration configuration in registry
  Int_t masterCode(0) ;
  Int_t tmp[2] ;
  tmp[0] = intCoefCode ;
  tmp[1] = normCoefCode ;
  masterCode = _codeReg.store(tmp,2,intConvSet,normConvSet)+1 ; // takes ownership of intConvSet,normConvSet
  
  if (_verboseEval>0) {
    cout << "RooConvolutedPdf::getAI(" << GetName() << ") masterCode " << masterCode 
	 << " will integrate analytically " ; analVars.Print("1") ;  
  }
  return masterCode  ;
}



Double_t RooConvolutedPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const 
{
  // Return analytical integral defined by given scenario code.
  //
  // For unnormalized integrals this is
  //                    _                _     
  //   PDF = sum_k Int(dx) coef_k * Int(dy) [ basis_k (x) ResModel ].
  //       _
  // where x is the set of coefficient dependents to be integrated
  // and y the set of basis function dependents to be integrated. 
  //
  // For normalized integrals this becomes
  //
  //         sum_k Int(dx) coef_k * Int(dy) [ basis_k (x) ResModel ].
  //  PDF =  --------------------------------------------------------
  //         sum_k Int(dv) coef_k * Int(dw) [ basis_k (x) ResModel ].
  //
  // where x is the set of coefficient dependents to be integrated,
  // y the set of basis function dependents to be integrated,
  // v is the set of coefficient dependents over which is normalized and
  // w is the set of basis function dependents over which is normalized.
  //
  // Set x must be contained in v and set y must be contained in w.
  //

  // Handle trivial passthrough scenario
  if (code==0) return getVal(normSet) ;

  // Unpack master code
  RooArgSet *intConvSet, *normConvSet ;
  const Int_t* tmp = _codeReg.retrieve(code-1,intConvSet,normConvSet) ;
  Int_t intCoefCode = tmp[0] ;
  Int_t normCoefCode = tmp[1] ;
  
  RooResolutionModel* conv ;
  Int_t index(0) ;
  Double_t answer(0) ;
  _convSetIter->Reset() ;

//   cout << "RooConvolutedPdf::aI(" << GetName() << "): intCoefCode = " << intCoefCode << ", normCoefCode = " << normCoefCode << endl ;
//   cout << "          intConvSet = " ; if (intConvSet) intConvSet->Print("1") ; else cout << "<none>" << endl ;
//   cout << "         normConvSet = " ; if (normConvSet) normConvSet->Print("1") ; else cout << "<none>" << endl ;

  if (normSet==0) {

    // Integral over unnormalized function
    Double_t integral(0) ;
    while(conv=(RooResolutionModel*)_convSetIter->Next()) {
      Double_t coef = coefAnalyticalIntegral(index++,intCoefCode) ;
      if (coef!=0) {
	integral += coef*conv->getNorm(intConvSet) ;
      }
    }
    answer = integral ;
    
  } else {

    // Integral over normalized function
    Double_t integral(0) ;
    Double_t norm(0) ;
    while(conv=(RooResolutionModel*)_convSetIter->Next()) {

      Double_t coefInt = coefAnalyticalIntegral(index,intCoefCode) ;
      if (coefInt!=0) {
	integral += coefInt*conv->getNormSpecial(intConvSet) ;
      }
      Double_t coefNorm = (intCoefCode==normCoefCode)?coefInt:coefAnalyticalIntegral(index,normCoefCode) ;
      if (coefNorm!=0) {
	norm += coefNorm*conv->getNorm(normConvSet) ; 	
      }
      index++ ;
    }
    answer = integral/norm ;    
  }

  return answer ;
}



Int_t RooConvolutedPdf::getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  // Default implementation of function advertising integration capabilities: no integrals
  // are advertised.

  return 0 ;
}



Double_t RooConvolutedPdf::coefAnalyticalIntegral(Int_t coef, Int_t code) const 
{
  // Default implementation of function implementing advertised integrals. Only
  // the pass-through scenario (no integration) is implemented.

  if (code==0) return coefficient(coef) ;
  cout << "RooConvolutedPdf::coefAnalyticalIntegral(" << GetName() << ") ERROR: unrecognized integration code: " << code << endl ;
  assert(0) ;
  return 1 ;
}



Bool_t RooConvolutedPdf::forceAnalyticalInt(const RooAbsArg& dep) const
{
  // This function forces RooRealIntegral to offer all integration dependents
  // to RooConvolutedPdf::getAnalyticalIntegralWN() for consideration for
  // analytical integration, if RRI considers this to be unsafe (e.g. due
  // to hidden Jacobian terms). 
  //
  // RooConvolutedPdf will not attempt to actually integrate all these dependents
  // but feed them to the resolution models integration interface, which will
  // make the final determination on how to integrate these dependents.

  return kTRUE ;
}                                                                                                                         
               



Bool_t RooConvolutedPdf::syncNormalizationPreHook(RooAbsReal* norm,const RooArgSet* nset) const 
{
  // Overload of hook function in RooAbsPdf::syncNormalization(). This functions serves
  // two purposes: 
  //
  //   - Modify default normalization behaviour of RooAbsPdf: integration requests over
  //     unrelated variables are properly executed (introducing a trivial multiplication
  //     for each unrelated dependent). This is necessary if composite resolution models
  //     are used in which the components do not necessarily all have the same set
  //     of dependents.
  //
  //   - Built the sub set of normalization dependents that is contained the basis function/
  //     resolution model convolution (to be used in syncNormalizationPostHook().

  delete _convNormSet ;
  RooArgSet convNormArgs("convNormArgs") ;

  // Make iterator over data set arguments
  TIterator* dsIter = nset->createIterator() ;
  RooAbsArg* dsArg ;

  // Make iterator over convolution integrals
  TIterator* cvIter = _convSet.createIterator() ;
  RooResolutionModel* conv ;

  // Build integration list for convolutions
  while (dsArg = (RooAbsArg*) dsIter->Next()) {
    cvIter->Reset() ;
    while(conv = (RooResolutionModel*) cvIter->Next()) {
      if (conv->dependsOn(*dsArg)) {
	// Add any data set variable that occurs in any convolution integral
	convNormArgs.add(*dsArg) ;
      }
    }
  }
  delete dsIter ;
  delete cvIter ;
  _convNormSet = new RooArgSet(convNormArgs,"convNormSet") ;
  
  return kFALSE ;
}




void RooConvolutedPdf::syncNormalizationPostHook(RooAbsReal* norm,const RooArgSet* nset) const 
{
  // Overload of hook function in RooAbsPdf::syncNormalization(). This function propagates
  // the syncNormalization() call to all basis-function/resolution-model convolution component
  // objects and fixes the physics models client-server links by adding each variable that
  // serves any of the convolution objects normalizations. PDFs by default have all client-server
  // links that control the unnormalized value (as returned by evaluate()), but convoluted PDFs
  // have a non-trivial normalization term that may introduc dependencies on additional server
  // that exclusively appear in the normalization.

  TIterator* cvIter = _convSet.createIterator() ;
  RooResolutionModel* conv ;

  // Make convolution normalizations servers of the convoluted pdf normalization
  while(conv=(RooResolutionModel*)cvIter->Next()) {
    conv->syncNormalization(_convNormSet) ;

    // Add leaf node servers of convolution normalization integrals to our normalization
    // integral, except for the integrated variables

    RooArgSet leafList("leafNodeServerList") ;
    conv->normLeafServerList(leafList) ;
    TIterator* sIter = leafList.createIterator() ;

    RooAbsArg* server ;
    while(server=(RooAbsArg*)sIter->Next()) {
      if (!_norm->findServer(*server)) {
	_norm->addServer(*server,kTRUE,kFALSE) ;
      }
    }
    delete sIter ;

  }  
  delete cvIter ;

  return ;
}



void RooConvolutedPdf::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsPdf::printToStream() we add:
  //
  //   Verbose : detailed information on convolution integrals

  RooAbsPdf::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    os << indent << "--- RooConvolutedPdf ---" << endl;
    TIterator* iter = _convSet.createIterator() ;
    RooResolutionModel* conv ;
    while (conv=(RooResolutionModel*)iter->Next()) {
      conv->printToStream(os,Verbose,"    ") ;
    }
  }
}




