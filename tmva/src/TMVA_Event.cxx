// @(#)root/tmva $Id: TMVA_Event.cpp,v 1.8 2006/05/03 19:45:38 helgevoss Exp $     
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_Event                                                            *
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
//                                                                      //
// Variables of an event as used for the Binary Tree                    //
//                                                                      //
//_______________________________________________________________________

#include <string>
#include "TMVA_Event.h"
#include "TMVA_Tools.h"
#include "TObjString.h"
#include "Riostream.h"
#include "TTree.h"
#include "TString.h"
#include <stdexcept>

ClassImp(TMVA_Event)

//_______________________________________________________________________
TMVA_Event::TMVA_Event(TTree* tree, Int_t ievt, std::vector<TString>* fInputVars)
{
  for (UInt_t ivar=0; ivar<fInputVars->size(); ivar++) 
    fVar.push_back(TMVA_Tools::GetValue( tree, ievt, (*fInputVars)[ivar] ));

  if (tree->GetBranchStatus("weight"))
    fWeight = Double_t(TMVA_Tools::GetValue( tree, ievt, "weight"));
  else
    fWeight = 1.;

  if (fWeight > 10) cout << "Weight in TMVA_Event " << fWeight <<endl;
  fType = Int_t(TMVA_Tools::GetValue( tree, ievt, "type" ));
}

//_______________________________________________________________________
const Double_t&  TMVA_Event::GetData(Int_t i) const 
{
  if (i<0 || i>(Int_t)fVar.size()) {
    cout<<"--- TMVA_Event::Data(Int i ... ERROR! i="<<i<<" out of range \n";
    exit(1);
  }
  else return fVar[i];
}

//_______________________________________________________________________
void TMVA_Event::Print(ostream& os) const 
{
  os << "Event with " << this->GetEventSize() << " variables and  weight " << this->GetWeight()<<endl;
  for (int i=0; i<this->GetEventSize(); i++){
    os << this->GetData(i) << "  "; 
  }
}

//_______________________________________________________________________
TMVA_Event* TMVA_Event::Read(ifstream& is)
{
  std::string tmp;
  Double_t dtmp;
  Int_t nvar;
  is >> tmp >> tmp >> nvar >> tmp >> tmp >> tmp >> dtmp;
  this->SetWeight(dtmp);
  for (int i=0; i<nvar; i++){
    is >> dtmp; this->Insert(dtmp);
  }
  return this;
}

//_______________________________________________________________________
ostream& operator<<(ostream& os, const TMVA_Event& event){ 
   //Outputs the data of an event
   
  event.Print(os);
  return os;
}

//_______________________________________________________________________
ostream& operator<<(ostream& os, const TMVA_Event* event){
  //Outputs the data of an event
   
  if (event!=NULL)event->Print(os);
  else os << "There is no TMVA_Event to print. Pointer == NULL";
  return os;
}







