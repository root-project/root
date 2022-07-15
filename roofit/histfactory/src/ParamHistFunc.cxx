// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, George Lewis 
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////

/** \class ParamHistFunc
 * \ingroup HistFactory
 *   A class which maps the current values of a RooRealVar
 *  (or a set of RooRealVars) to one of a number of RooAbsReal
 *  (nominally RooRealVar):
 *
 *  `ParamHistFunc: {val1, val2, ...} -> {gamma (RooRealVar)}`
 *
 *  The intended interpretation is that each parameter in the
 *  range represent the height of a bin over the domain
 *  space.
 *
 *  The 'createParamSet' is an easy way to create these
 *  parameters from a set of observables. They are
 *  stored using the "TH1" ordering convention (as compared
 *  to the RooDataHist convention, which is used internally
 *  and one must map between the two).
 *
 *  All indices include '0':<br>
 *  \f$ \gamma_{i,j} \f$ = `paramSet[ size(i)*j + i ]`
 *
 *  ie assuming the dimensions are 5*5:<br>
 *  \f$ \gamma_{2,1} \f$ = `paramSet[ 5*1 + 2 ] = paramSet[7]`
 */


#include "RooStats/HistFactory/ParamHistFunc.h"

#include "RooConstVar.h"
#include "RooBinning.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooWorkspace.h"
#include "RunContext.h"

#include "TH1.h"

#include <sstream>
#include <stdexcept>
#include <iostream>

ClassImp(ParamHistFunc);


////////////////////////////////////////////////////////////////////////////////

ParamHistFunc::ParamHistFunc()
  : _normIntMgr(this), _numBins(0)
{
  _dataSet.removeSelfFromDir(); // files must not delete _dataSet.
}


////////////////////////////////////////////////////////////////////////////////
/// Create a function which returns binewise-values
/// This class contains N RooAbsReals's, one for each
/// bin from the given RooRealVar.
///
/// The value of the function in the ith bin is 
/// given by:
///
/// F(i) = gamma_i * nominal(i)
///
/// Where the nominal values are simply fixed
/// numbers (default = 1.0 for all i)
ParamHistFunc::ParamHistFunc(const char* name, const char* title, 
			     const RooArgList& vars, const RooArgList& paramSet) :
  RooAbsReal(name, title),
  _normIntMgr(this),
  _dataVars("!dataVars","data Vars",       this),
  _paramSet("!paramSet","bin parameters",  this),
  _numBins(0),
  _dataSet( (std::string(name)+"_dataSet").c_str(), "", vars)
{

  // Create the dataset that stores the binning info:
  
  //  _dataSet = RooDataSet("

  _dataSet.removeSelfFromDir(); // files must not delete _dataSet.

  // Set the binning
  // //_binning = var.getBinning().clone() ;
  
  // Create the set of parameters
  // controlling the height of each bin

  // Get the number of bins
  _numBins = GetNumBins( vars );

  // Add the parameters (with checking)
  addVarSet( vars );
  addParamSet( paramSet );
}


////////////////////////////////////////////////////////////////////////////////
/// Create a function which returns bin-wise values.
/// This class allows to multiply bin contents of histograms
/// with the values of a set of RooAbsReal.
///
/// The value of the function in the ith bin is 
/// given by:
/// \f[
///   F(i) = \gamma_{i} * \mathrm{nominal}(i)
/// \f]
///
/// Where the nominal values are taken from the histogram,
/// and the \f$ \gamma_{i} \f$ can be set from the outside.
ParamHistFunc::ParamHistFunc(const char* name, const char* title, 
			     const RooArgList& vars, const RooArgList& paramSet,
			     const TH1* Hist ) :
  RooAbsReal(name, title),
  _normIntMgr(this),
  //  _dataVar("!dataVar","data Var", this, (RooRealVar&) var),
  _dataVars("!dataVars","data Vars",       this),
  _paramSet("!paramSet","bin parameters",  this),
  _numBins(0),
  _dataSet( (std::string(name)+"_dataSet").c_str(), "", vars, Hist)
{

  _dataSet.removeSelfFromDir(); // files must not delete _dataSet.

  // Get the number of bins
  _numBins = GetNumBins( vars );

  // Add the parameters (with checking)
  addVarSet( vars );
  addParamSet( paramSet );
}


Int_t ParamHistFunc::GetNumBins( const RooArgSet& vars ) {

  // A helper method to get the number of bins

  if( vars.getSize() == 0 ) return 0;

  Int_t numBins = 1;

  for (auto comp : vars) {
    if (!dynamic_cast<RooRealVar*>(comp)) {
      auto errorMsg = std::string("ParamHistFunc::GetNumBins") + vars.GetName() + ") ERROR: component "
                      + comp->GetName() + " in vars list is not of type RooRealVar";
      oocoutE(static_cast<TObject*>(nullptr), InputArguments) <<  errorMsg << std::endl;
      throw std::runtime_error(errorMsg);
    }
    auto var = static_cast<RooRealVar*>(comp);

    Int_t varNumBins = var->numBins();
    numBins *= varNumBins;
  }

  return numBins;
}


////////////////////////////////////////////////////////////////////////////////

ParamHistFunc::ParamHistFunc(const ParamHistFunc& other, const char* name) :
  RooAbsReal(other, name), 
  _normIntMgr(other._normIntMgr, this),
  _dataVars("!dataVars", this, other._dataVars ),
  _paramSet("!paramSet", this, other._paramSet),
  _numBins( other._numBins ),
  _dataSet( other._dataSet )
{
  _dataSet.removeSelfFromDir(); // files must not delete _dataSet.

  // Copy constructor
  // Member _ownedList is intentionally not copy-constructed -- ownership is not transferred
}


////////////////////////////////////////////////////////////////////////////////

ParamHistFunc::~ParamHistFunc() 
{
  ;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the parameter associated with the index.
/// The index follows RooDataHist indexing conventions.
/// It uses the binMap to convert the RooDataSet style index
/// into the TH1 style index (which is how they are stored
/// internally in the '_paramSet' vector).
RooAbsReal& ParamHistFunc::getParameter( Int_t index ) const {

  auto const& n = _numBinsPerDim;

  // check if _numBins needs to be filled
  if(n.x == 0) {
    _numBinsPerDim = getNumBinsPerDim(_dataVars);
  }

  // Unravel the index to 3D coordinates. We can't use the index directly,
  // because in the parameter set the dimensions are ordered in reverse order
  // compared to the RooDataHist (for historical reasons).
  const int i = index / n.yz;
  const int tmp = index % n.yz;
  const int j = tmp / n.z;
  const int k = tmp % n.z;

  return static_cast<RooAbsReal&>(_paramSet[i + j * n.x + k * n.xy]);
}


////////////////////////////////////////////////////////////////////////////////

RooAbsReal& ParamHistFunc::getParameter() const {
  Int_t index = getCurrentBin();
  return getParameter( index );
}


void ParamHistFunc::setParamConst( Int_t index, Bool_t varConst ) {
  RooAbsReal& var = getParameter( index );
  var.setAttribute( "Constant", varConst );
}


void ParamHistFunc::setConstant( bool constant ) {
  for( int i=0; i < numBins(); ++i) {
    setParamConst(i, constant);
  }
}


////////////////////////////////////////////////////////////////////////////////

void ParamHistFunc::setShape( TH1* shape ) {
  int num_hist_bins = shape->GetNbinsX()*shape->GetNbinsY()*shape->GetNbinsZ();

  if( num_hist_bins != numBins() ) {
    std::cout << "Error - ParamHistFunc: cannot set Shape of ParamHistFunc: " << GetName()
	      << " using histogram: " << shape->GetName()
	      << ". Bins don't match" << std::endl;
    throw std::runtime_error("setShape");
  }


  Int_t TH1BinNumber = 0;
  for( Int_t i = 0; i < numBins(); ++i) {
    
    TH1BinNumber++;
    
    while( shape->IsBinUnderflow(TH1BinNumber) || shape->IsBinOverflow(TH1BinNumber) ){
      TH1BinNumber++;
    }

    RooRealVar* param = dynamic_cast<RooRealVar*>(&_paramSet[i]);
    if(!param) {
       std::cout << "Error - ParamHisFunc: cannot set Shape of ParamHistFunc: " << GetName()
                 << " - param is not RooRealVar" << std::endl;
       throw std::runtime_error("setShape");
    }
    param->setVal( shape->GetBinContent(TH1BinNumber) );
  }

}


////////////////////////////////////////////////////////////////////////////////
/// Create the list of RooRealVar
/// parameters which represent the
/// height of the histogram bins.
/// The list 'vars' represents the 
/// observables (corresponding to histogram bins)
/// that these newly created parameters will 
/// be mapped to. (ie, we create one parameter
/// per observable in vars and per bin in each observable)

/// Store them in a list using:
/// _paramSet.add( createParamSet() );
/// This list is stored in the "TH1" index order
RooArgList ParamHistFunc::createParamSet(RooWorkspace& w, const std::string& Prefix, 
					 const RooArgList& vars) {


  // Get the number of bins
  // in the nominal histogram

  RooArgList paramSet;

  Int_t numVars = vars.getSize();
  Int_t numBins = GetNumBins( vars );

  if( numVars == 0 ) {
    std::cout << "Warning - ParamHistFunc::createParamSet() :"
	 << " No Variables provided.  Not making constraint terms." 
	 << std::endl;
    return paramSet;
  }

  else if( numVars == 1 ) {

    // For each bin, create a RooRealVar
    for( Int_t i = 0; i < numBins; ++i) {

      std::stringstream VarNameStream;
      VarNameStream << Prefix << "_bin_" << i;
      std::string VarName = VarNameStream.str();

      RooRealVar gamma( VarName.c_str(), VarName.c_str(), 1.0 ); 
      // "Hard-Code" a minimum of 0.0
      gamma.setMin( 0.0 );
      gamma.setConstant( false );

      w.import( gamma, RooFit::RecycleConflictNodes() );
      RooRealVar* gamma_wspace = (RooRealVar*) w.var( VarName.c_str() );

      paramSet.add( *gamma_wspace );

    }
  }

  else if( numVars == 2 ) {

    // Create a vector of indices
    // all starting at 0
    std::vector< Int_t > Indices(numVars, 0);

    RooRealVar* varx = (RooRealVar*) vars.at(0);
    RooRealVar* vary = (RooRealVar*) vars.at(1);

    // For each bin, create a RooRealVar
    for( Int_t j = 0; j < vary->numBins(); ++j) {
      for( Int_t i = 0; i < varx->numBins(); ++i) {

        // Ordering is important:
        // To match TH1, list goes over x bins
        // first, then y

        std::stringstream VarNameStream;
        VarNameStream << Prefix << "_bin_" << i << "_" << j;
        std::string VarName = VarNameStream.str();

        RooRealVar gamma( VarName.c_str(), VarName.c_str(), 1.0 );
        // "Hard-Code" a minimum of 0.0
        gamma.setMin( 0.0 );
        gamma.setConstant( false );

        w.import( gamma, RooFit::RecycleConflictNodes() );
        RooRealVar* gamma_wspace = (RooRealVar*) w.var( VarName.c_str() );

        paramSet.add( *gamma_wspace );

      }
    }
  }

  else if( numVars == 3 ) {

    // Create a vector of indices
    // all starting at 0
    std::vector< Int_t > Indices(numVars, 0);

    RooRealVar* varx = (RooRealVar*) vars.at(0);
    RooRealVar* vary = (RooRealVar*) vars.at(1);
    RooRealVar* varz = (RooRealVar*) vars.at(2);

    // For each bin, create a RooRealVar
    for( Int_t k = 0; k < varz->numBins(); ++k) {
      for( Int_t j = 0; j < vary->numBins(); ++j) {
        for( Int_t i = 0; i < varx->numBins(); ++i) {

          // Ordering is important:
          // To match TH1, list goes over x bins
          // first, then y, then z

          std::stringstream VarNameStream;
          VarNameStream << Prefix << "_bin_" << i << "_" << j << "_" << k;
          std::string VarName = VarNameStream.str();

          RooRealVar gamma( VarName.c_str(), VarName.c_str(), 1.0 );
          // "Hard-Code" a minimum of 0.0
          gamma.setMin( 0.0 );
          gamma.setConstant( false );

          w.import( gamma, RooFit::RecycleConflictNodes() );
          RooRealVar* gamma_wspace = (RooRealVar*) w.var( VarName.c_str() );

          paramSet.add( *gamma_wspace );

        }
      }
    }
  }

  else {
    std::cout << " Error: ParamHistFunc doesn't support dimensions > 3D " <<  std::endl;
  }

  return paramSet;  

}


////////////////////////////////////////////////////////////////////////////////
/// Create the list of RooRealVar parameters which scale the
/// height of histogram bins.
/// The list `vars` represents the observables (corresponding to histogram bins)
/// that these newly created parameters will
/// be mapped to. *I.e.*, we create one parameter
/// per observable in `vars` and per bin in each observable.
///
/// The new parameters are initialised to 1 with an uncertainty of +/- 1.,
/// their range is set to the function arguments.
///
/// Store the parameters in a list using:
/// ```
/// _paramSet.add( createParamSet() );
/// ```
/// This list is stored in the "TH1" index order.
RooArgList ParamHistFunc::createParamSet(RooWorkspace& w, const std::string& Prefix, 
					 const RooArgList& vars, 
					 Double_t gamma_min, Double_t gamma_max) {



  RooArgList params = ParamHistFunc::createParamSet( w, Prefix, vars );

  for (auto comp : params) {
    auto var = static_cast<RooRealVar*>(comp);

    var->setMin( gamma_min );
    var->setMax( gamma_max );
  }

  return params;

}


////////////////////////////////////////////////////////////////////////////////
/// Create the list of RooRealVar
/// parameters which represent the
/// height of the histogram bins.
/// Store them in a list
RooArgList ParamHistFunc::createParamSet(const std::string& Prefix, Int_t numBins, 
					 Double_t gamma_min, Double_t gamma_max) {

  // Get the number of bins
  // in the nominal histogram

  RooArgList paramSet;

  if( gamma_max <= gamma_min ) {

    std::cout << "Warning: gamma_min <= gamma_max: Using default values (0, 10)" << std::endl;

    gamma_min = 0.0;
    gamma_max = 10.0;

  }

  Double_t gamma_nominal = 1.0;

  if( gamma_nominal < gamma_min ) {
    gamma_nominal = gamma_min;
  }

  if( gamma_nominal > gamma_max ) {
    gamma_nominal = gamma_max;
  }

  // For each bin, create a RooRealVar
  for( Int_t i = 0; i < numBins; ++i) {

    std::stringstream VarNameStream;
    VarNameStream << Prefix << "_bin_" << i;
    std::string VarName = VarNameStream.str();

    RooRealVar* gamma = new RooRealVar( VarName.c_str(), VarName.c_str(), 
					gamma_nominal, gamma_min, gamma_max );
    gamma->setConstant( false );
    paramSet.add( *gamma );

  }

  return paramSet;

}


ParamHistFunc::NumBins ParamHistFunc::getNumBinsPerDim(RooArgSet const& vars) {
  int numVars = vars.size();

  if (numVars > 3 || numVars < 1) {
    std::cout << "ParamHistFunc() - Only works for 1-3 variables (1d-3d)" << std::endl;
    throw -1;  
  }

  int numBinsX = numVars >= 1 ? static_cast<RooRealVar const&>(*vars[0]).numBins() : 1;
  int numBinsY = numVars >= 2 ? static_cast<RooRealVar const&>(*vars[1]).numBins() : 1;
  int numBinsZ = numVars >= 3 ? static_cast<RooRealVar const&>(*vars[2]).numBins() : 1;

  return {numBinsX, numBinsY, numBinsZ};
}


////////////////////////////////////////////////////////////////////////////////
/// Get the index of the gamma parameter associated with the current bin.
/// e.g. `RooRealVar& currentParam = getParameter( getCurrentBin() );`
Int_t ParamHistFunc::getCurrentBin() const {
  // We promise that our coordinates and the data hist coordinates have same layout.
  return _dataSet.getIndex(_dataVars, /*fast=*/true);
}


////////////////////////////////////////////////////////////////////////////////
/// return 0 for success
/// return 1 for failure
/// Check that the elements 
/// are actually RooRealVar's
/// If so, add them to the 
/// list of vars
Int_t ParamHistFunc::addVarSet( const RooArgList& vars ) {
  for(auto const& comp : vars) {
    if (!dynamic_cast<RooRealVar*>(comp)) {
      auto errorMsg = std::string("ParamHistFunc::(") + GetName() + ") ERROR: component "
                      + comp->GetName() + " in variables list is not of type RooRealVar";
      coutE(InputArguments) <<  errorMsg << std::endl;
      throw std::runtime_error(errorMsg);
    }
    _dataVars.add( *comp );
  }
  return 0;
}


////////////////////////////////////////////////////////////////////////////////

Int_t ParamHistFunc::addParamSet( const RooArgList& params ) {
  // return 0 for success
  // return 1 for failure

  // Check that the supplied list has
  // the right number of arguments:

  Int_t numVarBins  = GetNumBins(_dataVars);
  Int_t numElements = params.getSize();

  if( numVarBins != numElements ) {
    std::cout << "ParamHistFunc::addParamSet - ERROR - " 
	      << "Supplied list of parameters " << params.GetName()
	      << " has " << numElements << " elements but the ParamHistFunc"
	      << GetName() << " has " << numVarBins << " bins."
	      << std::endl;
    return 1;

  }

  // Check that the elements
  // are actually RooAbsreal's
  // If so, add them to the
  // list of params

  for (const auto comp : params) {
    if (!dynamic_cast<const RooAbsReal*>(comp)) {
      auto errorMsg = std::string("ParamHistFunc::(") + GetName() + ") ERROR: component "
                      + comp->GetName() + " in parameter list is not of type RooAbsReal.";
      coutE(InputArguments) <<  errorMsg << std::endl;
      throw std::runtime_error(errorMsg);
    }

    _paramSet.add( *comp );
  }
  
  return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Find the bin corresponding to the current value of the observable, and evaluate
/// the associated parameter.
Double_t ParamHistFunc::evaluate() const 
{
  return getParameter().getVal();
}


////////////////////////////////////////////////////////////////////////////////
/// Find all bins corresponding to the values of the observables in `evalData`, and evaluate
/// the associated parameters.
/// \param[in/out] evalData Input/output data for evaluating the ParamHistFunc.
/// \param[in] normSet Normalisation set passed on to objects that are serving values to us.
void ParamHistFunc::computeBatch(cudaStream_t*, double* output, size_t size, RooFit::Detail::DataMap const& dataMap) const {
  std::vector<double> oldValues;
  std::vector<RooSpan<const double>> data;
  oldValues.reserve(_dataVars.size());
  data.reserve(_dataVars.size());

  // Retrieve data for all variables
  for (auto arg : _dataVars) {
    const auto* var = static_cast<RooRealVar*>(arg);
    oldValues.push_back(var->getVal());
    data.push_back(dataMap.at(var));
  }

  // Run computation for each entry in the dataset
  for (std::size_t i = 0; i < size; ++i) {
    for (unsigned int j = 0; j < _dataVars.size(); ++j) {
      assert(i < data[j].size());
      auto& var = static_cast<RooRealVar&>(_dataVars[j]);
      var.setCachedValue(data[j][i], /*notifyClients=*/false);
    }

    const auto index = _dataSet.getIndex(_dataVars, /*fast=*/true);
    const RooAbsReal& param = getParameter(index);
    output[i] = param.getVal();
  }

  // Restore old values
  for (unsigned int j = 0; j < _dataVars.size(); ++j) {
    auto& var = static_cast<RooRealVar&>(_dataVars[j]);
    var.setCachedValue(oldValues[j], /*notifyClients=*/false);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise that all integrals can be handled internally.

Int_t ParamHistFunc::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, 
					     const RooArgSet* normSet, 
					     const char* /*rangeName*/) const 
{
  // Handle trivial no-integration scenario
  if (allVars.getSize()==0) return 0 ;
  if (_forceNumInt) return 0 ;


  // Select subset of allVars that are actual dependents
  analVars.add(allVars) ;

  // Check if this configuration was created before
  Int_t sterileIdx(-1) ;
  CacheElem* cache = (CacheElem*) _normIntMgr.getObj(normSet,&analVars,&sterileIdx,(const char*)0) ;
  if (cache) {
    return _normIntMgr.lastIndex()+1 ;
  }
  
  // Create new cache element
  cache = new CacheElem ;

  // Store cache element
  Int_t code = _normIntMgr.setObj(normSet,&analVars,(RooAbsCacheElement*)cache,0) ;

  return code+1 ; 

}


////////////////////////////////////////////////////////////////////////////////
/// Implement analytical integrations by doing appropriate weighting from  component integrals
/// functions to integrators of components

Double_t ParamHistFunc::analyticalIntegralWN(Int_t /*code*/, const RooArgSet* /*normSet2*/,
					     const char* /*rangeName*/) const 
{
  Double_t value(0) ;

  // Simply loop over bins, 
  // get the height, and
  // multiply by the bind width
  auto binVolumes = _dataSet.binVolumes(0, _dataSet.numEntries());

  for (unsigned int i=0; i < _paramSet.size(); ++i) {
    const auto& param = static_cast<const RooAbsReal&>(_paramSet[i]);
    assert(static_cast<Int_t>(i) == _dataSet.getIndex(param)); // We assume that each parameter i belongs to bin i

    // Get the gamma's value
    const double paramVal = param.getVal();
    
    // Finally, get the subtotal
    value += paramVal * binVolumes[i];
  }

  return value;

}



////////////////////////////////////////////////////////////////////////////////
/// Return sampling hint for making curves of (projections) of this function
/// as the recursive division strategy of RooCurve cannot deal efficiently
/// with the vertical lines that occur in a non-interpolated histogram

std::list<Double_t>* ParamHistFunc::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, 
						Double_t xhi) const
{
  // copied and edited from RooHistFunc
  RooAbsLValue* lvarg = &obs;

  // Retrieve position of all bin boundaries
  const RooAbsBinning* binning = lvarg->getBinningPtr(0) ;
  Double_t* boundaries = binning->array() ;

  std::list<Double_t>* hint = new std::list<Double_t> ;

  // Widen range slighty
  xlo = xlo - 0.01*(xhi-xlo) ;
  xhi = xhi + 0.01*(xhi-xlo) ;

  Double_t delta = (xhi-xlo)*1e-8 ;
 
  // Construct array with pairs of points positioned epsilon to the left and
  // right of the bin boundaries
  for (Int_t i=0 ; i<binning->numBoundaries() ; i++) {
    if (boundaries[i]>=xlo && boundaries[i]<=xhi) {
      hint->push_back(boundaries[i]-delta) ;
      hint->push_back(boundaries[i]+delta) ;
    }
  }
  return hint ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return sampling hint for making curves of (projections) of this function
/// as the recursive division strategy of RooCurve cannot deal efficiently
/// with the vertical lines that occur in a non-interpolated histogram

std::list<Double_t>* ParamHistFunc::binBoundaries(RooAbsRealLValue& obs, Double_t xlo, 
						  Double_t xhi) const 
{
  // copied and edited from RooHistFunc
  RooAbsLValue* lvarg = &obs;

  // Retrieve position of all bin boundaries
  const RooAbsBinning* binning = lvarg->getBinningPtr(0) ;
  Double_t* boundaries = binning->array() ;

  std::list<Double_t>* hint = new std::list<Double_t> ;

  // Construct array with pairs of points positioned epsilon to the left and
  // right of the bin boundaries
  for (Int_t i=0 ; i<binning->numBoundaries() ; i++) {
    if (boundaries[i]>=xlo && boundaries[i]<=xhi) {
      hint->push_back(boundaries[i]) ;
    }
  }

  return hint ;
}
