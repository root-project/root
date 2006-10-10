// @(#)root/tmva $Id: Event.cxx,v 1.3 2006/05/23 19:35:06 brun Exp $     
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::Event                                                           *
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
//______________________________________________________________________//

#include <string>
#include "TMVA/Event.h"
#include "TMVA/Tools.h"
#include "TObjString.h"
#include "Riostream.h"
#include "TTree.h"
#include "TString.h"
#include <stdexcept>

ClassImp(TMVA::Event)

//_______________________________________________________________________
TMVA::Event::Event(TTree* tree, Int_t ievt, std::vector<TString>* fInputVars)
{
   //event constructor reading variables from a ROOT tree
   for (UInt_t ivar=0; ivar<fInputVars->size(); ivar++) 
      fVar.push_back(TMVA::Tools::GetValue( tree, ievt, (*fInputVars)[ivar] ));

   if (tree->GetBranchStatus("weight"))
      fWeight = Double_t(TMVA::Tools::GetValue( tree, ievt, "weight"));
   else
      fWeight = 1.;

   if (fWeight > 10) cout << "Weight in TMVA::Event " << fWeight <<endl;
   fType = Int_t(TMVA::Tools::GetValue( tree, ievt, "type" ));
}

//_______________________________________________________________________
const Double_t&  TMVA::Event::GetData(Int_t i) const 
{
   // return reference to "i-th" event variable
   if (i<0 || i>(Int_t)fVar.size()) {
      cout<<"--- TMVA::Event::Data(Int i ... ERROR! i="<<i<<" out of range \n";
      exit(1);
   }
   else return fVar[i];
}

//_______________________________________________________________________
void TMVA::Event::Print(ostream& os) const 
{
   //print event variables and event weight
   os << "Event with " << this->GetEventSize() << " variables and  weight " << this->GetWeight()<<endl;
   for (int i=0; i<this->GetEventSize(); i++){
      os << this->GetData(i) << "  "; 
   }
}

//_______________________________________________________________________
TMVA::Event* TMVA::Event::Read(ifstream& is)
{
   //read event e.g. from a text file
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
ostream& TMVA::operator<<(ostream& os, const TMVA::Event& event){ 
   //Outputs the data of an event
   
   event.Print(os);
   return os;
}

//_______________________________________________________________________
ostream& TMVA::operator<<(ostream& os, const TMVA::Event* event){
   //Outputs the data of an event
   
   if (event!=NULL)event->Print(os);
   else os << "There is no TMVA::Event to print. Pointer == NULL";
   return os;
}
