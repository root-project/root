// @(#)root/tmva $Id: Event.cxx,v 1.19 2006/09/29 23:27:15 andreas.hoecker Exp $   
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
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
 **********************************************************************************/

#include "TMVA/Event.h"
#include "TMVA/Tools.h"
#include "TTree.h"
#include "TBranch.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;
using std::setw;
 
//____________________________________________________________
TMVA::Event::Event(const std::vector<VariableInfo> & varinfo) 
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
   for (UInt_t ivar=0; ivar<fVariables.size(); ivar++) {
      if      (fVariables[ivar].VarType()=='I') fCountI++;
      else if (fVariables[ivar].VarType()=='F') fCountF++;
      else {
         std::cout << "ERROR: Unknown variable type encountered in constructor of Event" << std::endl;
         exit(1);
      }
   }
   InitPointers();
}

TMVA::Event::Event(const Event & event) 
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
   InitPointers(kFALSE); // this constructor is (miss)used in the BinarySearchTree
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
   fVarPtrI = new Int_t[fCountI];
   fVarPtrF = new Float_t[fCountF];
   
   UInt_t ivar(0), ivarI(0), ivarF(0);
   std::vector<VariableInfo>::const_iterator varIt = fVariables.begin();
   // for each variable,
   for (; varIt != fVariables.end(); varIt++, ivar++) {
      const VariableInfo & var = *varIt;
      // set the void pointer (which are used to access the data) to the proper field
      // if external field is given
      if (AllowExternalLink && var.GetExternalLink()!=0) {
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
		else {
         std::cout << "ERROR: Unknown variable type encountered in constructor of Event" << std::endl;
         exit(1);
      }
   }
}

//____________________________________________________________
void TMVA::Event::SetBranchAddresses(TTree *tr) 
{
   fBranches.clear();
   Int_t ivar(0);
   TBranch * br(0);
   std::vector<VariableInfo>::const_iterator varIt;
   for (varIt = fVariables.begin(); varIt != fVariables.end(); varIt++) {
      const VariableInfo & var = *varIt;
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

   // sanity check
   if (GetNVars() != other.GetNVars()) {
      cout << "--- Event::CopyVarValues: Error: mismatch in events ==> abort" << endl;
      exit(1);
   }

   for (UInt_t ivar=0; ivar<GetNVars(); ivar++) SetVal( ivar, other.GetVal( ivar ) );    
}

//____________________________________________________________
void TMVA::Event::SetVal(UInt_t ivar, Float_t val) 
{
   if (ivar>=GetNVars()) {
      cout << "ERROR: Cannot set value for variable index " << ivar << ", exceeds max index " << GetNVars()-1 << endl;
   }
   *((Float_t*)fVarPtr[ivar]) = val;
}

//____________________________________________________________
Float_t TMVA::Event::GetValueNormalized(Int_t ivar) const 
{
   return Tools::NormVariable(GetVal(ivar),fVariables[ivar].GetMin(),fVariables[ivar].GetMax());
}

//____________________________________________________________
void TMVA::Event::Print(std::ostream & o) const
{
   o << fVariables.size() << " vars: ";
   for(UInt_t ivar=0; ivar<fVariables.size(); ivar++)
      o << setw(10) << GetVal(ivar);
   o << endl;
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
