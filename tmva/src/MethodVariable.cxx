// @(#)root/tmva $Id: MethodVariable.cxx,v 1.4 2006/05/22 08:04:39 andreas.hoecker Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodVariable                                                  *
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
//_______________________________________________________________________

#include "TMVA/MethodVariable.h"
#include <algorithm>

ClassImp(TMVA::MethodVariable)
 
//_______________________________________________________________________
TMVA::MethodVariable::MethodVariable( TString jobName, vector<TString>* theVariables,  
				      TTree* theTree, TString theOption, 
				      TDirectory* theTargetDir )
  : TMVA::MethodBase( jobName, theVariables, theTree, theOption, theTargetDir )
{
  // standard constructor
  // option string contains variable name - but not only ! 
  // there is a "Var_" prefix, which is useful in the context of later root plotting
  // so, remove this part

  fMethodName    = "Variable";
  fTestvarPrefix ="";
  fTestvar       = fTestvarPrefix+GetMethodName(); 
  if (Verbose())
    cout << "--- " << GetName() << " <verbose>: uses as discriminating variable just "
         << fOptions << " as specified in the option" << endl;
  
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
TMVA::MethodVariable::~MethodVariable( void )
{
  // destructor
}

//_______________________________________________________________________
void TMVA::MethodVariable::Train( void )
{
  // no training required

  // default sanity checks
  if (!CheckSanity()) { 
    cout << "--- " << GetName() << ": Error: sanity check failed" << endl;
    exit(1);
  }
}


//_______________________________________________________________________
Double_t TMVA::MethodVariable::GetMvaValue( TMVA::Event *e )
{
  // "MVA" value is variable value
  return e->GetData(0);
}

//_______________________________________________________________________
void  TMVA::MethodVariable::WriteWeightsToFile( void )
{  
  // nothing to write
  cout << "--- " << GetName() << ": no weights to write " << endl;
}
  
//_______________________________________________________________________
void  TMVA::MethodVariable::ReadWeightsFromFile( void )
{
  // nothing to read
  cout << "--- " << GetName() << ": no weights to read "  <<  endl;
}

//_______________________________________________________________________
void  TMVA::MethodVariable::WriteHistosToFile( void )
{
  // write special monitoring histograms to file - not implemented for Variable
  cout << "--- " << GetName() << ": no histograms to write " << endl;
}
