// @(#)root/tmva $Id: Event.cxx,v 1.10 2006/11/20 15:35:28 brun Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Event                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#include "TMVA/Event.h"
#include "TMVA/Tools.h"
#include "TTree.h"
#include "TBranch.h"
#include <iostream>
#include <iomanip>
 
Int_t TMVA::Event::fgCount = 0;

//____________________________________________________________
TMVA::Event::Event(const std::vector<VariableInfo>& varinfo, Bool_t AllowExternalLinks) 
   : fVariables(varinfo),
     fVarPtr(new void*[varinfo.size()]), // array to hold pointers to the integer or float array
     fVarPtrI(0),                        // array to hold all float variables 
     fVarPtrF(0),                        // array to hold all integer variables
     fType(0),
     fWeight(0),
     fBoostWeight(1.0),
     fCountI(0),
     fCountF(0)
{
   // constructor

   fgCount++; 
   for (UInt_t ivar=0; ivar<fVariables.size(); ivar++) {
      if      (fVariables[ivar].VarType()=='I') fCountI++;
      else if (fVariables[ivar].VarType()=='F') fCountF++;
   }
   InitPointers(AllowExternalLinks);
}

TMVA::Event::Event( const Event& event ) 
   : fVariables(event.fVariables),
     fVarPtr(new void*[event.fVariables.size()]), // array to hold pointers to the integer or float array
     fVarPtrI(0),                                 // array to hold all float variables 
     fVarPtrF(0),                                 // array to hold all integer variables
     fType(event.fType),
     fWeight(event.fWeight),
     fBoostWeight(event.fBoostWeight),
     fCountI(event.fCountI),
     fCountF(event.fCountF)
{
   // constructor

   fgCount++; 
   InitPointers(kFALSE); // this constructor is used in the BinarySearchTree
   // where we don't want to have externaly linked variables
   for (UInt_t ivar = 0; ivar< fCountI; ivar++){
      fVarPtrI[ivar] = *((Int_t*)event.fVarPtr[ivar]);
   }
   for (UInt_t ivar = 0; ivar< fCountF; ivar++){
      fVarPtrF[ivar] = *((Float_t*)event.fVarPtr[ivar]);
   }
}
 
void TMVA::Event::InitPointers(bool AllowExternalLink)
{
   // sets the links of fVarPtr to the internal arrays that hold the
   // integer and float variables

   fVarPtrI = new Int_t[fCountI];
   fVarPtrF = new Float_t[fCountF];
   
   UInt_t ivar(0), ivarI(0), ivarF(0);
   std::vector<VariableInfo>::const_iterator varIt = fVariables.begin();
   // for each variable,
   for (; varIt != fVariables.end(); varIt++, ivar++) {
      const VariableInfo& var = *varIt;
      // set the void pointer (which are used to access the data) to the proper field
      // if external field is given
      if (AllowExternalLink&& var.GetExternalLink()!=0) {
         fVarPtr[ivar] = var.GetExternalLink();
         // or if its type is I(int) or F(float)
      } 
      else if (var.VarType()=='F') {
         // set the void pointer to the float field
         fVarPtr[ivar] = fVarPtrF+ivarF++;
      } 
      else if (var.VarType()=='I') {
         // set the void pointer to the int field
         fVarPtr[ivar] = fVarPtrI+ivarI++;
      } 
   }
}


//____________________________________________________________
TMVA::Event::~Event() {
   delete[] fVarPtr;
   delete[] fVarPtrI;
   delete[] fVarPtrF;
   fgCount--;
}



//____________________________________________________________
void TMVA::Event::SetBranchAddresses(TTree *tr)
{
   // sets the branch addresses of the associated
   // tree to the local memory as given by fVarPtr

   fBranches.clear();
   Int_t ivar(0);
   TBranch * br(0);
   std::vector<VariableInfo>::const_iterator varIt;
   for (varIt = fVariables.begin(); varIt != fVariables.end(); varIt++) {
      const VariableInfo& var = *varIt;
      br = tr->GetBranch(var.GetInternalVarName());
      br->SetAddress(fVarPtr[ivar++]);
      fBranches.push_back(br);
   }
   br = tr->GetBranch("type");        br->SetAddress(&fType);        fBranches.push_back(br);
   br = tr->GetBranch("weight");      br->SetAddress(&fWeight);      fBranches.push_back(br);
   br = tr->GetBranch("boostweight"); br->SetAddress(&fBoostWeight); fBranches.push_back(br);
}

//____________________________________________________________
void TMVA::Event::CopyVarValues( const Event& other )
{
   // copies only the variable values
   for (UInt_t ivar=0; ivar<GetNVars(); ivar++) SetVal( ivar, other.GetVal( ivar ) );    
   SetType(other.Type());
   SetWeight(other.GetWeight());
   SetBoostWeight(other.GetBoostWeight());
}

//____________________________________________________________
void TMVA::Event::SetVal(UInt_t ivar, Float_t val) 
{
   // set variable ivar to val
   *((Float_t*)fVarPtr[ivar]) = val;
}

//____________________________________________________________
Float_t TMVA::Event::GetValueNormalized(Int_t ivar) const 
{
   // returns the value of variable ivar, normalized to [-1,1]

   return Tools::NormVariable(GetVal(ivar),fVariables[ivar].GetMin(),fVariables[ivar].GetMax());
}

//____________________________________________________________
void TMVA::Event::Print(std::ostream& o) const
{
   // print method

   o << fVariables.size() << " vars: ";
   for(UInt_t ivar=0; ivar<fVariables.size(); ivar++)
      o << " " << std::setw(10) << GetVal(ivar);
   o << "  EvtWeight " << std::setw(10) << GetWeight();
   o << std::setw(10) << (IsSignal()?" Signal":" Background");
   o << std::endl;
}

//_______________________________________________________________________
ostream& TMVA::operator<<(ostream& os, const TMVA::Event& event)
{ 
   // Outputs the data of an event
   
   event.Print(os);
   return os;
}

//_______________________________________________________________________
ostream& TMVA::operator<<(ostream& os, const TMVA::Event* event)
{
   // Outputs the data of an event
   return os << *event;
}
