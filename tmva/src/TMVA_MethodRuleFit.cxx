// @(#)root/tmva $Id: TMVA_MethodRuleFit.cpp,v 1.6 2006/05/02 23:27:40 helgevoss Exp $    
// Author: Andreas Hoecker, Fredrik Tegenfeldt, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodRuleFit                                                    *
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
//                                                                      
//_______________________________________________________________________

#include "TMVA_MethodRuleFit.h"
#include "TMVA_Tools.h"
#include "TMatrix.h"
#include "Riostream.h"
#include <algorithm>

#define DEBUG_TMVA_MethodRuleFit kFALSE

ClassImp(TMVA_MethodRuleFit)
 
//_______________________________________________________________________
TMVA_MethodRuleFit::TMVA_MethodRuleFit( TString jobName, vector<TString>* theVariables,  
					TTree* theTree, TString theOption, TDirectory* theTargetDir )
  : TMVA_MethodBase( jobName, theVariables, theTree, theOption, theTargetDir )
{
  InitRuleFit();
}

//_______________________________________________________________________
TMVA_MethodRuleFit::TMVA_MethodRuleFit( vector<TString> *theVariables, 
					TString theWeightFile,  
					TDirectory* theTargetDir )
  : TMVA_MethodBase( theVariables, theWeightFile, theTargetDir ) 
{
  InitRuleFit();
}

//_______________________________________________________________________
void TMVA_MethodRuleFit::InitRuleFit( void )
{
  fMethodName         = "RuleFit";
  fMethod             = TMVA_Types::RuleFit;
  fTestvar            = fTestvarPrefix+GetMethodName();
}

//_______________________________________________________________________
TMVA_MethodRuleFit::~TMVA_MethodRuleFit( void )
{}

//_______________________________________________________________________
void TMVA_MethodRuleFit::Train( void )
{
  //--------------------------------------------------------------

  // default sanity checks
  if (!CheckSanity()) { 
    cout << "--- " << GetName() << ": Error: sanity check failed" << endl;
    exit(1);
  }

  // write weights to file
  WriteWeightsToFile();
}

//_______________________________________________________________________
void  TMVA_MethodRuleFit::WriteWeightsToFile( void )
{  
  // write coefficients to file
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
void  TMVA_MethodRuleFit::ReadWeightsFromFile( void )
{
  // read coefficients from file
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
Double_t TMVA_MethodRuleFit::GetMvaValue( TMVA_Event * /*e*/ )
{
  Double_t myMVA = 0;

  return myMVA;
}

//_______________________________________________________________________
void  TMVA_MethodRuleFit::WriteHistosToFile( void )
{
  cout << "--- " << GetName() << ": write " << GetName() 
       <<" special histos to file: " << fBaseDir->GetPath() << endl;
}
