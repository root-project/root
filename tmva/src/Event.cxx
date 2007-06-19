// @(#)root/tmva $Id: Event.cxx,v 1.12 2007/04/21 14:20:46 brun Exp $   
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
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
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
     fVarPtrF(0),                        // array to hold all integer variables
     fType(0),
     fWeight(0),
     fBoostWeight(1.0),
     fCountI(0),
     fCountF(0)
{
   // constructor

   fgCount++; // static member: counts number of Event instances (check for memory leak)

   fCountF = fVariables.size();

   InitPointers(AllowExternalLinks);
}

TMVA::Event::Event( const Event& event ) 
   : fVariables(event.fVariables),
     fVarPtr(new void*[event.fVariables.size()]), // array to hold pointers to the integer or float array
     fVarPtrF(0),                                 // array to hold all float variables
     fType(event.fType),
     fWeight(event.fWeight),
     fBoostWeight(event.fBoostWeight),
     fCountI(event.fCountI),
     fCountF(event.fCountF)
{
   // copy constructor

   fgCount++; 
   InitPointers(kFALSE); // this constructor is used in the BinarySearchTree

   // where we don't want to have externaly linked variables
   for (UInt_t ivar=0; ivar<GetNVars(); ivar++) {
      *(Float_t*)fVarPtr[ivar] = *(Float_t*)event.fVarPtr[ivar];
   }
}

//____________________________________________________________
TMVA::Event::~Event() 
{
   // Event destructor
   delete[] fVarPtr;
   delete[] fVarPtrF;
   fgCount--;
}
 
void TMVA::Event::InitPointers(bool AllowExternalLink)
{
   // sets the links of fVarPtr to the internal arrays that hold the
   // integer and float variables

   fVarPtrF = new Float_t[fCountF];
   
   UInt_t ivar(0), ivarF(0);
   std::vector<VariableInfo>::const_iterator varIt = fVariables.begin();

   // for each variable,
   for (; varIt != fVariables.end(); varIt++, ivar++) {
      const VariableInfo& var = *varIt;
      // set the void pointer (which are used to access the data) to the proper field
      // if external field is given
      if (AllowExternalLink && var.GetExternalLink()!=0) {
         fVarPtr[ivar] = var.GetExternalLink();
         // or if its type is I(int) or F(float)
      } 
      else {
         // set the void pointer to the float field
         fVarPtr[ivar] = fVarPtrF+ivarF++;
      } 
   }
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
   for (UInt_t ivar=0; ivar<GetNVars(); ivar++)
      SetVal(ivar, other.GetVal(ivar));
   SetType(other.Type());
   SetWeight(other.GetWeight());
   SetBoostWeight(other.GetBoostWeight());
}

//____________________________________________________________
void TMVA::Event::SetVal(UInt_t ivar, Float_t val) 
{
   // set variable ivar to val
   Bool_t isInternallyLinked = (fVarPtr[ivar] >= fVarPtrF && fVarPtr[ivar] < fVarPtrF+fCountF);

   if(isInternallyLinked) {
      fVarPtrF[ivar] = val;
   } else { // external variable, have to go with type
      if(fVariables[ivar].GetVarType()=='F') {
         *(Float_t*)fVarPtr[ivar] = val;
      } else {
         *(Int_t*)fVarPtr[ivar] = (Int_t)val;
      }
   }
}

//____________________________________________________________
Float_t TMVA::Event::GetVal(UInt_t ivar) const
{
   // return value of variable ivar
   Bool_t isInternallyLinked = (fVarPtr[ivar] >= fVarPtrF && fVarPtr[ivar] < fVarPtrF+fCountF);
   if(isInternallyLinked) {
      return fVarPtrF[ivar];
   } else { // external variable, have to go with type
      if(fVariables[ivar].GetVarType()=='F') {
         return *(Float_t*)fVarPtr[ivar];
      } else {
         return *(Int_t*)fVarPtr[ivar];
      }
   }
}

//____________________________________________________________
Float_t TMVA::Event::GetValueNormalized(UInt_t ivar) const 
{
   // returns the value of variable ivar, normalized to [-1,1]
   return Tools::NormVariable(GetVal(ivar),fVariables[ivar].GetMin(),fVariables[ivar].GetMax());
}

//____________________________________________________________
void TMVA::Event::Print(std::ostream& o) const
{
   // print method
   o << fVariables.size() << " variables: ";
   for (UInt_t ivar=0; ivar<fVariables.size(); ivar++)
      o << " " << std::setw(10) << GetVal(ivar) << '(' << fVariables[ivar].GetVarType() << ')';
   o << "  weight = " << GetWeight();
   o << std::setw(10) << (IsSignal()?" signal":" background");
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
