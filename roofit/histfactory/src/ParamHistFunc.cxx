/*****************************************************************************

 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// ParamHistFunc 
// END_HTML
//


#include <sstream>
#include "TMath.h"
#include "TH1.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooFit.h"
#include "RooStats/HistFactory/ParamHistFunc.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"

#include "RooConstVar.h"
#include "RooBinning.h"
#include "RooErrorHandler.h"

#include "RooGaussian.h"
#include "RooHistFunc.h"
#include "RooArgSet.h"
#include "RooNLLVar.h"
#include "RooChi2Var.h"
#include "RooMsgService.h"

// Forward declared:
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooWorkspace.h"
#include "RooBinning.h"

ClassImp(ParamHistFunc);


//_____________________________________________________________________________
ParamHistFunc::ParamHistFunc() 
{
  ;
}


//_____________________________________________________________________________
ParamHistFunc::ParamHistFunc(const char* name, const char* title, 
			     const RooArgList& vars, const RooArgList& paramSet) :
  RooAbsReal(name, title),
  _dataVars("!dataVars","data Vars",       this),
  _paramSet("!paramSet","bin paramaters",  this),
  _numBins(0),
  _dataSet( (string(name)+"_dataSet").c_str(), "", vars) 
{
  
  // Create a function which returns binewise-values
  // This class contains N RooRealVar's, one for each
  // bin from the given RooRealVar.
  //
  // The value of the function in the ith bin is 
  // given by:
  //
  // F(i) = gamma_i * nominal(i)
  //
  // Where the nominal values are simply fixed
  // numbers (default = 1.0 for all i)

  // Create the dataset that stores the binning info:
  
  //  _dataSet = RooDataSet("

  // Set the binning
  // //_binning = var.getBinning().clone() ;
  
  // Create the set of paramaters
  // controlling the height of each bin

  // Get the number of bins
  _numBins = GetNumBins( vars );

  // Add the paramaters (with checking)
  addVarSet( vars );
  addParamSet( paramSet );

  
}


//_____________________________________________________________________________
ParamHistFunc::ParamHistFunc(const char* name, const char* title, 
			     const RooArgList& vars, const RooArgList& paramSet,
			     const TH1* Hist ) :
  RooAbsReal(name, title),
  //  _dataVar("!dataVar","data Var", this, (RooRealVar&) var),
  _dataVars("!dataVars","data Vars",       this),
  _paramSet("!paramSet","bin paramaters",  this),
  _numBins(0),
  _dataSet( (string(name)+"_dataSet").c_str(), "", vars, Hist) 
{

  // Create a function which returns binewise-values
  // This class contains N RooRealVar's, one for each
  // bin from the given RooRealVar.
  //
  // The value of the function in the ith bin is 
  // given by:
  //
  // F(i) = gamma_i * nominal(i)
  //
  // Where the nominal values are simply fixed
  // numbers (default = 1.0 for all i)

  // Get the number of bins
  _numBins = GetNumBins( vars );

  // Add the paramaters (with checking)
  addVarSet( vars );
  addParamSet( paramSet );
 
}


Int_t ParamHistFunc::GetNumBins( const RooArgSet& vars ) {
  
  
  // A helper method to get the number of bins
  
  if( vars.getSize() == 0 ) return 0;
    
  Int_t numBins = 1;

  RooFIter varIter = vars.fwdIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*) varIter.next())) {
    if (!dynamic_cast<RooRealVar*>(comp)) {
      cout << "ParamHistFunc::GetNumBins" << vars.GetName() << ") ERROR: component " << comp->GetName() 
	   << " in vars list is not of type RooRealVar" << std::endl ;
      RooErrorHandler::softAbort() ;
      return -1;
    }
    RooRealVar* var = (RooRealVar*) comp;

    Int_t varNumBins = var->numBins();
    numBins *= varNumBins;
  }
    
  return numBins;

}


//_____________________________________________________________________________
ParamHistFunc::ParamHistFunc(const ParamHistFunc& other, const char* name) :
  RooAbsReal(other, name), 
  _dataVars("!dataVars", this, other._dataVars ),
  _paramSet("!paramSet", this, other._paramSet),
  _numBins( other._numBins ),
  _dataSet( other._dataSet )
{
  ;
  // Copy constructor
  // Member _ownedList is intentionally not copy-constructed -- ownership is not transferred
}


//_____________________________________________________________________________
ParamHistFunc::~ParamHistFunc() 
{
  ;
}


//_____________________________________________________________________________
Int_t ParamHistFunc::getCurrentBin() const {

  //  Int_t currentIndex = _dataSet.calcTreeIndex();

  //  const RooArgSet* currentBinCenters = _dataSet.get();
  //  Int_t currentIndex = _dataSet.getIndex( *currentBinCenters ); // calcTreeIndex();

  //cout << "ParamHistFunc::getCurrentBin() - Vars: " << _dataVars << std::endl;
  Int_t currentIndex = _dataSet.getIndex( _dataVars ); // calcTreeIndex();

  return currentIndex;

  /*
  RooArgSet* currentBinCenters = _dataSet.get();

  Double_t varVal = _dataVar.arg().getVal(); //.getVal();
  Int_t numBins   = _binning->numBins();

  Double_t low    = _binning->lowBound();
  Double_t high   = _binning->highBound();

  if( (varVal<low) || (varVal>high) ) {
    std::cout << "Current value of variable " << _dataVar.arg().GetName()
	      << ": " << varVal << " is out of range: "
	      << "[" << low << ", " << high << "]" 
	      << std::endl;
    return -1;
  }

  Double_t length = (high-low > 0) ? high-low : 0;  
  
  if( length == 0 ) {
    std::cout << "Current binning has length of 0" << std::endl;
    return -1;
  }

  Int_t currentBin = -1;

  if( _binning->isUniform() ) {

    // If binning is uniform, use
    // faster method:

    Double_t width = _binning->averageBinWidth();

    if( width == 0 ) {
      std::cout << "Error: Bins have 0 width" << std::endl;
      return -1;
    }

    // First bin  == 0
    // Second bin == 1
    // etc
    currentBin = TMath::Floor( (varVal-low) / width );


  } else {


    // RooBinning::rawBinNumber method allows for
    // the value to be in the underflow.  We want
    // to explicitely not allow that possability

    for(Int_t i = 0; i < numBins; ++i ) {

      Double_t binLow  = _binning->binLow( i );
      Double_t binHigh = _binning->binHigh( i );

      if( binLow <= varVal ) {
	if( varVal < binHigh ) {
	  currentBin = i;
	}
      }

    } // end loop over bins

    // Check maximum bound:
    if( varVal == high ) {
      currentBin = numBins - 1;
    }
  }

  // Ensure the current bin
  
  if( (currentBin < 0) || (currentBin >= _binning->numBins()) ) {
    std::cout << "Error: Bad value for current bin: " << currentBin << std::endl;
    throw -1;
  }

  return currentBin;
  */

}

//_____________________________________________________________________________
RooRealVar& ParamHistFunc::getParameter( Int_t index ) const {
  return (RooRealVar&) _paramSet[index];
}

//_____________________________________________________________________________
RooRealVar& ParamHistFunc::getParameter() const {
  Int_t index = getCurrentBin();
  return getParameter( index );
}

void ParamHistFunc::setParamConst( Int_t index, Bool_t varConst ) {

  RooRealVar& var = getParameter( index );
  var.setConstant( varConst );
}



//_____________________________________________________________________________
RooArgList ParamHistFunc::createParamSet(RooWorkspace& w, const std::string& Prefix, const RooArgList& vars) {
  
  // Create the list of RooRealVar
  // parameters which represent the
  // height of the histogram bins.
  // Store them in a list

  // _paramSet.add( createParamSet() );

  // Get the number of bins
  // in the nominal histogram


  RooArgList paramSet;

  Int_t numVars = vars.getSize();
  Int_t numBins = GetNumBins( vars );


  if( numVars == 0 ) {
    cout << "Warning - ParamHistFunc::createParamSet() :"
	 << " No Variables provided.  Not making constraint terms." 
	 << endl;
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
    for( Int_t i = 0; i < varx->numBins(); ++i) {
      for( Int_t j = 0; j < vary->numBins(); ++j) {

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
    for( Int_t i = 0; i < varx->numBins(); ++i) {
      for( Int_t j = 0; j < vary->numBins(); ++j) {
	for( Int_t k = 0; k < varz->numBins(); ++k) {

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
 
    cout << " Error: ParamHistFunc doesn't support dimensions > 3D " <<  endl;
    
    /*
    // Create a vector of indices
    // all starting at 0
    std::vector< Int_t > Indices(numVars, 0);

    // Loop over vars:
    RooFIter varIter = vars.fwdIterator() ;
    Int_t VarIndex = 0;
    RooAbsArg* comp ;
    while((comp = (RooAbsArg*) varIter.next())) {
    
      RooRealVar* var = (RooRealVar*) comp;

      // For each bin, create a RooRealVar
      for( Int_t i = 0; i < var->numBins(); ++i) {

	if( i != 0 ) Indices.at(VarIndex)++;
	
	// Make the name of the var:
	// Varname_bin_0_2_1  where x=0, y=2, z=1 (etc)
	std::stringstream VarNameStream;
	VarNameStream << Prefix << "_bin";
	for(Int_t j = 0; j < numVars; ++j) {
	  VarNameStream << "_" << Indices.at(j);;
	}
	std::string VarName = VarNameStream.str();
	
	RooRealVar gamma( VarName.c_str(), VarName.c_str(), 1.0 ); 
	// "Hard-Code" a minimum of 0.0
	gamma.setMin( 0.0 );
	gamma.setConstant( false );
	
	w.import( gamma, RooFit::RecycleConflictNodes() );
	RooRealVar* gamma_wspace = (RooRealVar*) w.var( VarName.c_str() );
	
	paramSet.add( *gamma_wspace );
	
	// Increase the bin index on this var
	// (Used in naming)

      }

      // Increase the Int_t iterator
      // over variables
      VarIndex++;

    }
    */
  }

  return paramSet;  

}


//_____________________________________________________________________________
RooArgList ParamHistFunc::createParamSet(RooWorkspace& w, const std::string& Prefix, const RooArgList& vars, 
					 Double_t gamma_min, Double_t gamma_max) {


  RooArgList params = ParamHistFunc::createParamSet( w, Prefix, vars );

  RooFIter paramIter = params.fwdIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*) paramIter.next())) {
    
    RooRealVar* var = (RooRealVar*) comp;

    var->setMin( gamma_min );
    var->setMax( gamma_max );
  }

  return params;

  /*  

  // Create the list of RooRealVar
  // parameters which represent the
  // height of the histogram bins.
  // Store them in a list

  // _paramSet.add( createParamSet() );

  // Get the number of bins
  // in the nominal histogram
 

  RooArgList paramSet;

  if( gamma_max <= gamma_min ) {

    std::cout << "Warming: gamma_min <= gamma_max: Using default values (0, 10)" << std::endl;

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

    RooRealVar gamma( VarName.c_str(), VarName.c_str(), 
		      gamma_nominal, gamma_min, gamma_max );
    gamma.setConstant( false );

    w.import( gamma, RooFit::RecycleConflictNodes() );
    RooRealVar* gamma_wspace = (RooRealVar*) w.var( VarName.c_str() );

    paramSet.add( *gamma_wspace );

  }


  return paramSet;
  */

}



//_____________________________________________________________________________
RooArgList ParamHistFunc::createParamSet(const std::string& Prefix, Int_t numBins, Double_t gamma_min, Double_t gamma_max) {

  // Create the list of RooRealVar
  // parameters which represent the
  // height of the histogram bins.
  // Store them in a list

  // _paramSet.add( createParamSet() );

  // Get the number of bins
  // in the nominal histogram
 

  RooArgList paramSet;

  if( gamma_max <= gamma_min ) {

    std::cout << "Warming: gamma_min <= gamma_max: Using default values (0, 10)" << std::endl;

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

//_____________________________________________________________________________
Int_t ParamHistFunc::addVarSet( const RooArgList& vars ) {
  
  // return 0 for success
  // return 1 for failure
  
  // Check that the elements 
  // are actually RooRealVar's
  // If so, add them to the 
  // list of vars

  RooFIter varIter = vars.fwdIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*) varIter.next())) {
    if (!dynamic_cast<RooRealVar*>(comp)) {
      coutE(InputArguments) << "ParamHistFunc::(" << GetName() << ") ERROR: component " << comp->GetName() 
			    << " in variables list is not of type RooRealVar" << std::endl ;
      RooErrorHandler::softAbort() ;
      return 1;
    }

    _dataVars.add( *comp );

  }
  
  return 0;

}

//_____________________________________________________________________________
Int_t ParamHistFunc::addParamSet( const RooArgList& params ) {
  
  // return 0 for success
  // return 1 for failure

  // Check that the supplied list has
  // the right number of arguments:

  Int_t numVarBins  = _numBins;
  Int_t numElements = params.getSize();

  if( numVarBins != numElements ) {
    std::cout << "ParamHistFunc::addParamSet - ERROR - " 
	      << "Supplied list of paramaters " << params.GetName()
	      << " has " << numElements << " elements but the ParamHistFunc"
	      << GetName() << " has " << numVarBins << " bins."
	      << std::endl;
    return 1;

  }
  
  // Check that the elements 
  // are actually RooRealVar's
  // If so, add them to the 
  // list of params

  RooFIter paramIter = params.fwdIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*) paramIter.next())) {
    if (!dynamic_cast<RooRealVar*>(comp)) {
      coutE(InputArguments) << "ParamHistFunc::(" << GetName() << ") ERROR: component " << comp->GetName() 
			    << " in paramater list is not of type RooRealVar" << std::endl ;
      RooErrorHandler::softAbort() ;
      return 1;
    }

    _paramSet.add( *comp );

  }
  
  return 0;

}


//_____________________________________________________________________________
Double_t ParamHistFunc::evaluate() const 
{

  // Find the bin cooresponding to the current
  // value of the RooRealVar:

  /*
  if( ! &_dataVar.arg() ) {
    std::cout << "ERROR: Proxy " << _dataVar.GetName() 
	      << " is invalid!" << std::endl;
  }

  Double_t varVal = _dataVar.arg().getVal(); //.getVal();

  Int_t currentBin = getCurrentBin();
  
  if( currentBin == -1 ) {
    std::cout << "Error: Current value of " << _dataVar.arg().GetName()
	      << " " << varVal << " "
	      << " is not in range of saved Binning."
	      << " [" << _binning->lowBound() << ", " << _binning->highBound() << "] "
	      << " Returning: -1.0" << std::endl;

    return 0.0;
  }



  if( currentBin >= _paramSet.getSize() ) {
    std::cout << "Error: Trying to get out-of-range bin" << std::endl;
    return 0.0;
  }
  */
  Int_t currentBin = getCurrentBin();

  RooRealVar* param = (RooRealVar*) &(_paramSet[currentBin]);

  Double_t value = param->getVal();

  /*
  cout << "ParamHistFunc - Current Bin: " << currentBin
       << " param pointer: " << param
       << " param val: " << paramVal
       << " val: " << value
       << endl;
  */
  return value;
  
}


//_____________________________________________________________________________
Int_t ParamHistFunc::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, 
						      const RooArgSet* normSet, const char* /*rangeName*/) const 
{
  // Advertise that all integrals can be handled internally.

  // Handle trivial no-integration scenario
  if (allVars.getSize()==0) return 0 ;
  if (_forceNumInt) return 0 ;


  // Select subset of allVars that are actual dependents
  analVars.add(allVars) ;
  //  RooArgSet* normSet = normSet2 ? getObservables(normSet2) : 0 ;
  //  RooArgSet* normSet = getObservables();
  //  RooArgSet* normSet = 0;


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


  // Make list of function projection and normalization integrals 
  //  RooAbsReal* param ;
  // RooAbsReal *func ;
  //  const RooArgSet* nset = _paramList.nset() ;

  // do nominal
  /*
  func = (RooAbsReal*) &( _dataVar.arg() );
  RooAbsReal* funcInt = func->createIntegral(analVars) ;
  cache->_funcIntList.addOwned(*funcInt) ;


  // Implement integration here

  // do variations
  //_lowIter->Reset() ;
  //_highIter->Reset() ;
  _paramIter->Reset() ;
  int i=0;
  while((param=(RooAbsReal*)_paramIter->Next())) {

    / *
    func = (RooAbsReal*)_lowIter->Next() ;
    funcInt = func->createIntegral(analVars) ;
    cache->_lowIntList.addOwned(*funcInt) ;

    func = (RooAbsReal*)_highIter->Next() ;
    funcInt = func->createIntegral(analVars) ;
    cache->_highIntList.addOwned(*funcInt) ;
    * /
    ++i;

  }
*/

}




//_____________________________________________________________________________
Double_t ParamHistFunc::analyticalIntegralWN(Int_t /*code*/, const RooArgSet* /*normSet2*/,const char* /*rangeName*/) const 
{
  // Implement analytical integrations by doing appropriate weighting from  component integrals
  // functions to integrators of components

  Double_t value(0) ;

  // Simply loop over bins, 
  // get the height, and
  // multiply by the bind width
  
  RooFIter paramIter = _paramSet.fwdIterator();
  RooRealVar* param = NULL;
  Int_t nominalItr = 0;
  while((param = (RooRealVar*) paramIter.next())) {

    // Get the gamma's value
    Double_t paramVal  = (*param).getVal();
    
    // Get the bin volume
    _dataSet.get( nominalItr );
    Double_t binVolume  = _dataSet.binVolume(); //_binning->binWidth( nominalItr );
    
    // Finally, get the subtotal
    value += paramVal*binVolume;

    ++nominalItr;

    /*
    std::cout << "Integrating : "
	      << " bin: "  << nomValue
	      << " binVolume:  "  << binVolume
	      << " paramValue:  "  << paramVal
	      << " nomValue:  "  << nomValue
	      << " subTotal:  "  << value
	      << std::endl;
    */

  }


  /*
  Int_t numBins = _binning->numBins(); 
  // Deal with chaching / value propagation
  RooAbsArg::setDirtyInhibit(kTRUE) ;

  for( Int_t i = 0; i < numBins; ++i ) {
    
    Double_t binCenter = _binning->binCenter( i );
    RooRealVar* dataVar = (RooRealVar*) &( _dataVar.arg() );
    dataVar->setVal( binCenter );
		
    Double_t binWidth = _binning->binWidth( i );
      
    Double_t funcValue = getVal();

    value += funcValue * binWidth;

    std::cout << "Integrating : "
	      << " binCenter: "  << binCenter
	      << " binWidth:  "  << binWidth
	      << " binValue:  "  << funcValue
	      << " subTotal:  "  << value
	      << std::endl;

  }

  RooAbsArg::setDirtyInhibit(kFALSE) ;
  */

  return value;

}



//_____________________________________________________________________________
list<Double_t>* ParamHistFunc::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  // Return sampling hint for making curves of (projections) of this function
  // as the recursive division strategy of RooCurve cannot deal efficiently
  // with the vertical lines that occur in a non-interpolated histogram

  return 0;

  // copied and edited from RooHistFunc
  RooAbsLValue* lvarg = &obs;

  // Retrieve position of all bin boundaries
  const RooAbsBinning* binning = lvarg->getBinningPtr(0) ;
  Double_t* boundaries = binning->array() ;

  list<Double_t>* hint = new list<Double_t> ;

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


//______________________________________________________________________________
std::list<Double_t>* ParamHistFunc::binBoundaries(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const 
{
  // Return sampling hint for making curves of (projections) of this function
  // as the recursive division strategy of RooCurve cannot deal efficiently
  // with the vertical lines that occur in a non-interpolated histogram

  return 0;

  // copied and edited from RooHistFunc
  RooAbsLValue* lvarg = &obs;

  // Retrieve position of all bin boundaries
  const RooAbsBinning* binning = lvarg->getBinningPtr(0) ;
  Double_t* boundaries = binning->array() ;

  list<Double_t>* hint = new list<Double_t> ;

  // Construct array with pairs of points positioned epsilon to the left and
  // right of the bin boundaries
  for (Int_t i=0 ; i<binning->numBoundaries() ; i++) {
    if (boundaries[i]>=xlo && boundaries[i]<=xhi) {
      hint->push_back(boundaries[i]) ;
    }
  }

  return hint ;
}
