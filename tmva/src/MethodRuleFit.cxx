// @(#)root/tmva $Id: MethodRuleFit.cxx,v 1.3 2006/05/20 12:31:44 andreas.hoecker Exp $    
// Author: Andreas Hoecker, Fredrik Tegenfeldt, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodRuleFit                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
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
// Friedman's RuleFit method -- not yet implemented -- dummy class --   
//_______________________________________________________________________

#include "TMVA/MethodRuleFit.h"
#include "TMVA/Tools.h"
#include "TMatrix.h"
#include "Riostream.h"
#include <algorithm>

ClassImp(TMVA::MethodRuleFit)
 
//_______________________________________________________________________
TMVA::MethodRuleFit::MethodRuleFit( TString jobName, vector<TString>* theVariables,  
				    TTree* theTree, TString theOption, TDirectory* theTargetDir )
  : TMVA::MethodBase( jobName, theVariables, theTree, theOption, theTargetDir )
{
  // standard constructor
  InitRuleFit();
}

//_______________________________________________________________________
TMVA::MethodRuleFit::MethodRuleFit( vector<TString> *theVariables, 
				    TString theWeightFile,  
				    TDirectory* theTargetDir )
  : TMVA::MethodBase( theVariables, theWeightFile, theTargetDir ) 
{
  // constructor from weight file
  InitRuleFit();
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::InitRuleFit( void )
{
  // default initialisation
  fMethodName         = "RuleFit";
  fMethod             = TMVA::Types::RuleFit;
  fTestvar            = fTestvarPrefix+GetMethodName();
}

//_______________________________________________________________________
TMVA::MethodRuleFit::~MethodRuleFit( void )
{
  // destructor
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::Train( void )
{
  // training of rules

  // default sanity checks
  if (!CheckSanity()) { 
    cout << "--- " << GetName() << ": Error: sanity check failed" << endl;
    exit(1);
  }

  // write weights to file
  WriteWeightsToFile();
}

//_______________________________________________________________________
void  TMVA::MethodRuleFit::WriteWeightsToFile( void )
{  
  // write rules to file
  TString fname = GetWeightFileName();
  cout << "--- " << GetName() << ": creating weight file: " << fname << endl;
  ofstream fout( fname );
  if (!fout.good( )) { // file not found --> Error
    cout << "--- " << GetName() << ": Error in ::WriteWeightsToFile: "
         << "unable to open output  weight file: " << fname << endl;
    exit(1);
  }
  fout.close();    
}
  
//_______________________________________________________________________
void  TMVA::MethodRuleFit::ReadWeightsFromFile( void )
{
  // read rules from file
  TString fname = GetWeightFileName();
  cout << "--- " << GetName() << ": reading weight file: " << fname << endl;
  ifstream fin( fname );

  if (!fin.good( )) { // file not found --> Error
    cout << "--- " << GetName() << ": Error in ::ReadWeightsFromFile: "
         << "unable to open input file: " << fname << endl;
    exit(1);
  }


  fin.close();    
}

//_______________________________________________________________________
Double_t TMVA::MethodRuleFit::GetMvaValue( TMVA::Event * /*e*/ )
{
  // returns MVA value for given event
  Double_t myMVA = 0;

  return myMVA;
}

//_______________________________________________________________________
void  TMVA::MethodRuleFit::WriteHistosToFile( void )
{
  // write special monitoring histograms to file - not implemented for RuleFit
  cout << "--- " << GetName() << ": write " << GetName() 
       <<" special histos to file: " << fBaseDir->GetPath() << endl;
}
