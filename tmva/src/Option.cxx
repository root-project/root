// @(#)root/tmva $Id: Option.cxx,v 1.9 2006/10/06 16:18:42 andreas.hoecker Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Option                                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
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

#include "Riostream.h"
#include <iomanip>
#include "TMVA/Option.h"
#include "TString.h"

//______________________________________________________________________
Bool_t TMVA::OptionBase::SetValue(const TString& vs) 
{
   fIsSet = kTRUE;
   setValue(vs);
   return kTRUE;
}

//______________________________________________________________________
template<>
Bool_t TMVA::Option<Bool_t>::isPreDefinedVal(const Bool_t& val) const 
{
   // template specialization for Bool_t 
   if (val); // dummy
   return kTRUE;
}

template<>
Bool_t TMVA::Option<Float_t>::isPreDefinedVal(const Float_t& val) const 
{
   // template specialization for Float_t 
   if (val); // dummy
   return kTRUE;
}

template<>
Bool_t TMVA::Option<TString>::isPreDefinedVal(const TString& val) const 
{
   TString tVal(val);
   tVal.ToLower();
   if (fPreDefs.size()==0) return kFALSE; // if nothing pre-defined then allow everything
   Bool_t foundPreDef = kFALSE;   
   std::vector<TString>::const_iterator predefIt;
   predefIt = fPreDefs.begin();
   for (;predefIt!=fPreDefs.end(); predefIt++) {
      TString s(*predefIt);
      s.ToLower();
      if ( s==tVal ) { foundPreDef = kTRUE; break; }
   }
   return foundPreDef;
}

//______________________________________________________________________
template<>
void TMVA::Option<TString>::setValue(const TString& val) {
   TString valToSet(val);
   if (fPreDefs.size()!=0) {
      TString tVal(val);
      tVal.ToLower();
      std::vector<TString>::const_iterator predefIt;
      predefIt = fPreDefs.begin();
      for (;predefIt!=fPreDefs.end(); predefIt++) {
         TString s(*predefIt);
         s.ToLower();
         if ( s==tVal ) { valToSet = *predefIt; break; }
      }
   }

   std::stringstream str(valToSet.Data());
   str >> fValue;
   if (fRefPtr!=0) *fRefPtr = fValue;
}


template<>
void TMVA::Option<Bool_t>::setValue(const TString& val) {
   TString valToSet(val);
   valToSet.ToLower();
   if(valToSet=="1" || valToSet=="true" || valToSet=="ktrue" || valToSet=="t") {
      fValue = true;
   } else if(valToSet=="0" || valToSet=="false" || valToSet=="kfalse" || valToSet=="f") {
      fValue = false;
   } else {
      cerr << "Value \'" << val << "\' can not be interpreted as boolean" << endl;
      exit(1);
   }
   if (fRefPtr!=0) *fRefPtr = fValue;
}


//______________________________________________________________________
template<>
void TMVA::Option<Bool_t>::AddPreDefVal(const Bool_t&) 
{
   // template specialization for Bool_t 
   std::cout << "--- " << GetName() << "::AddPreDefVal: WARNING: "
             << "predefined values for Option<Bool_t> don't make sense" << std::endl;
}

template<>
void TMVA::Option<Float_t>::AddPreDefVal(const Float_t&) 
{
   // template specialization for Float_t 
   std::cout << "--- " << GetName() << "::AddPreDefVal: WARNING: "
             << "predefined values for Option<Float_t> don't make sense" << std::endl;
}

//______________________________________________________________________
template<>
void TMVA::Option<Bool_t>::Print(ostream& os, Int_t levelofdetail) const 
{
   if (levelofdetail); // dummy to avoid compiler warnings
   os << TheName() << ": " << (Value() ? "True" : "False") << " [" << Description() << "]" << endl;
}

template<>
void TMVA::Option<TString>::Print(ostream& os, Int_t levelofdetail) const 
{
   os << TheName() << ": " << "\"" << Value() << "\"" << " [" << Description() << "]" << endl;
   if (levelofdetail>0) {
      os << "    possible values are";
      std::vector<TString>::const_iterator predefIt;
      predefIt = fPreDefs.begin();
      for (;predefIt!=fPreDefs.end(); predefIt++) {
         if (predefIt != fPreDefs.begin()) os << "                       ";
         os << "  - " << (*predefIt) << endl;
      }
   }
}

//______________________________________________________________________
template<>
const TString TMVA::Option<Bool_t>::Type() const 
{
   return "bool";
}

template<>
const TString TMVA::Option<Float_t>::Type() const 
{
   return "float";
}

template<>
const TString TMVA::Option<Int_t>::Type() const 
{
   return "int";
}

template<>
const TString TMVA::Option<TString>::Type() const 
{
   return "TString";
}

template<>
const TString TMVA::Option<Double_t>::Type() const 
{
   return "double";
}
