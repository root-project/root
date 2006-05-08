// @(#)root/tmva $Id: TMVA_MethodVariable.cpp,v 1.6 2006/05/03 08:31:10 helgevoss Exp $
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodVariable                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Wrapper class for a single variable "MVA"; this is required for      
// the evaluation of the single variable discrimination performance     
//
//_______________________________________________________________________

#include "TMVA_MethodVariable.h"
#include <algorithm>

#define DEBUG_TMVA_MethodVariable kFALSE

ClassImp(TMVA_MethodVariable)
 
//_______________________________________________________________________
TMVA_MethodVariable::TMVA_MethodVariable( TString jobName, vector<TString>* theVariables,  
					  TTree* theTree, TString theOption, 
					  TDirectory* theTargetDir )
  : TMVA_MethodBase( jobName, theVariables, theTree, theOption, theTargetDir )
{
  fMethodName    = "Variable";
  fTestvarPrefix ="";
  fTestvar       = fTestvarPrefix+GetMethodName(); 
  if (Verbose())
    cout << "--- " << GetName() << " <verbose>: uses as discriminating variable just "
	 << fOptions << " as specified in the option" <<endl;
  
  // option string contains variable name - but not only ! 
  // there is a "Var_" prefix, which is useful in the context of later root plotting
  // so, remove this part
  if (0 == theTree->FindBranch(fOptions)) {
    cout << "--- " << GetName() << ": variable " << fOptions <<" not found "<<endl;
    theTree->Print();
    exit(1);
  }
  else{
    fMethodName += "_";
    fMethodName += fOptions;
    fTestvar    =  fOptions;
    if (Verbose())
      cout << "--- " << GetName() << " <verbose>: sucessfully initialized as " 
	   << GetMethodName() <<endl;
  }
}

//_______________________________________________________________________
TMVA_MethodVariable::~TMVA_MethodVariable( void )
{}

//_______________________________________________________________________
void TMVA_MethodVariable::Train( void )
{
  //--------------------------------------------------------------

  // default sanity checks
  if (!CheckSanity()) { 
    cout << "--- " << GetName() << ": Error: sanity check failed" << endl;
    exit(1);
  }
}


//_______________________________________________________________________
Double_t TMVA_MethodVariable::GetMvaValue(TMVA_Event *e){
  return e->GetData(0);
}

//_______________________________________________________________________
void  TMVA_MethodVariable::WriteWeightsToFile( void )
{  
  // write coefficients to file
  cout << "--- " << GetName() << ": no weights to write " << endl;
}
  
//_______________________________________________________________________
void  TMVA_MethodVariable::ReadWeightsFromFile( void )
{
  // read coefficients from file
  cout << "--- " << GetName() << ": no weights to read "  <<  endl;
}


//_______________________________________________________________________
void  TMVA_MethodVariable::WriteHistosToFile( void )
{
  cout << "--- " << GetName() << ": no histograms to write " << endl;
}
