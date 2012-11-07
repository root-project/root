
#include "RooRealVar.h"
#include "RooStats/HistFactory/HistFactoryException.h"

#include "RooStats/HistFactory/Asimov.h"

void RooStats::HistFactory::Asimov::ConfigureWorkspace(RooWorkspace* wspace) {

  // Here is where we set the values, and constantness
  // of all parameters in the workspace before creating
  // an asimov dataset

  /*
  // Okay, y'all, first we're going to create a snapshot
  // of the current state of the variables in the workspace
  
  std::string ListOfVariableNames = "";
  for( std::map< std::string, double >::iterator itr = fParamValsToSet.begin(); 
       itr != fParamValsToSet.end(); ++itr) {
    // Extend the Variable Name list
    ListOfVariableNames += "," + itr->first;
  }  
  for( std::map< std::string, bool >::iterator itr = fParamsToFix.begin(); 
       itr != fParamsToFix.end(); ++itr) {
    // Extend the Variable Name list
    ListOfVariableNames += "," + itr->first;
  }  
  
  // Save a snapshot
  std::string SnapShotName = "NominalParamValues";
  wspace->saveSnapshot(SnapShotName.c_str(), ListOfVariableNames.c_str());
  */

  //
  // First we set all parameters to their given values
  //


  for( std::map< std::string, double >::iterator itr = fParamValsToSet.begin(); 
       itr != fParamValsToSet.end(); ++itr) {

    std::string param = itr->first;
    double val  = itr->second;

    // Try to get the variable in the workspace
    RooRealVar* var = wspace->var(param.c_str());
    if( !var ) {
      std::cout << "Error: Trying to set variable: " << var
		<< " to a specific value in creation of asimov dataset: " << fName
		<< " but this variable doesn't appear to exist in the workspace"
		<< std::endl;
      throw hf_exc();
    }

    // Check that the desired value is in the range of the variable
    double inRange = var->inRange(val, NULL);
    if( !inRange ) {
      std::cout << "Error: Attempting to set variable: " << var
		<< " to value: " << val << ", however it appears"
		<< " that this is not withn the variable's range: " 
		<< "[" << var->getMin() << ", " << var->getMax() << "]"
		<< std::endl;
      throw hf_exc();
    }

    // Set its value
    std::cout << "Configuring Asimov Dataset: Setting " << param
	      << " = " << val << std::endl;
    var->setVal( val );
  }


  //
  // Then, we set any variables to constant
  //
  
  for( std::map< std::string, bool >::iterator itr = fParamsToFix.begin(); 
       itr != fParamsToFix.end(); ++itr) {

    std::string param = itr->first;
    bool isConstant  = itr->second;

    // Try to get the variable in the workspace
    RooRealVar* var = wspace->var(param.c_str());
    if( !var ) {
      std::cout << "Error: Trying to set variable: " << var
		<< " constant in creation of asimov dataset: " << fName
		<< " but this variable doesn't appear to exist in the workspace"
		<< std::endl;
      throw hf_exc();
    }

    std::cout << "Configuring Asimov Dataset: Setting " << param
	      << " to constant " << std::endl;
    var->setConstant( isConstant );
    
  }
  
  return;

}
