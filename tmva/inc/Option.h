// @(#)root/tmva $Id: Option.h,v 1.9 2006/10/06 16:18:42 andreas.hoecker Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Option                                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Option container                                                          *
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

#ifndef ROOT_TMVA_Option
#define ROOT_TMVA_Option

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Option                                                               //
//                                                                      //
// Class for MVA-option handling                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <sstream>

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif

namespace TMVA {

   class OptionBase : public TObject {

   public:

      OptionBase( const TString& name, const TString& desc ) :
         fName(name),
         fDescription(desc),
         fIsSet(false)
      {}
      virtual ~OptionBase(){};
         
      virtual const char* GetName() const { TString s(fName); s.ToLower(); return s.Data(); }
      virtual const char* TheName() const { return fName; }
      virtual TString GetValue() const = 0;

      Bool_t IsSet() const { return fIsSet; }
      const TString& Description() const { return fDescription; }
      virtual Bool_t  IsPreDefinedVal(const TString&) const = 0;
      virtual Bool_t HasPreDefinedVal() const = 0;
      Bool_t SetValue(const TString& vs);

      virtual void Print(const Option_t* o) const { return TObject::Print(o); }
      virtual void Print(ostream&, Int_t levelofdetail=0) const = 0;
      virtual const TString Type() const = 0;

   private:

      virtual void setValue(const TString& vs) = 0;         

      const TString fName;
      const TString fDescription;
      Bool_t        fIsSet;
   };
      
   // ---------------------------------------------------------------------------

   template <class T>
   class Option : public OptionBase {

   public:

      Option(const TString& name, const TString& desc ) :
         OptionBase(name, desc),
         fRefPtr(0)
      {}
      Option(T& ref, const TString& name, const TString& desc ) :
         OptionBase(name, desc),
         fValue(ref),
         fRefPtr(&ref)
      {}
      virtual ~Option(){};

      // getters
      virtual TString GetValue() const {
         std::stringstream str;
         str << fValue;
         return str.str();
      }
      const T& Value() const { return fValue; }
      Bool_t HasPreDefinedVal() const;
      virtual Bool_t IsPreDefinedVal(const TString&) const; 

      // setters
      virtual void AddPreDefVal(const T&);
      virtual void Print(const Option_t* o) const { return OptionBase::Print(o); }
      virtual void Print(ostream&, Int_t levelofdetail=0) const;
      virtual const TString Type() const;

   private:

     virtual void setValue(const TString& val);
      virtual Bool_t isPreDefinedVal(const T&) const;
      T fValue;
      T* fRefPtr;

      std::vector<T> fPreDefs;
   };      
} // namespace

//______________________________________________________________________
template<class T>
Bool_t TMVA::Option<T>::HasPreDefinedVal() const 
{
   return (fPreDefs.size()!=0);
}

template<class T>
Bool_t TMVA::Option<T>::IsPreDefinedVal(const TString& val) const 
{
   T tmpVal;
   std::stringstream str(val.Data());
   str >> tmpVal;
   return isPreDefinedVal(tmpVal);
}

//______________________________________________________________________
template<class T>
Bool_t TMVA::Option<T>::isPreDefinedVal(const T& val) const 
{
   if (fPreDefs.size()==0) return kFALSE; // if nothing pre-defined then allow everything
   Bool_t foundPreDef = kFALSE;   
   typename std::vector<T>::const_iterator predefIt;
   predefIt = fPreDefs.begin();
   for (;predefIt!=fPreDefs.end(); predefIt++) {
      if ( (*predefIt)==val ) { foundPreDef = kTRUE; break; }
   }
   return foundPreDef;
}

//______________________________________________________________________
template<class T>
void TMVA::Option<T>::AddPreDefVal(const T&val) 
{
   fPreDefs.push_back(val);
}

//______________________________________________________________________
template<class T>
void TMVA::Option<T>::Print(ostream& os, Int_t levelofdetail) const 
{
   if (levelofdetail); // dummy to avoid compiler warnings
   os << TheName() << ": " << GetValue() << " [" << Description() << "]" << endl;
}

//______________________________________________________________________
template<class T>
const TString TMVA::Option<T>::Type() const 
{
   return "undefined type";
}

//______________________________________________________________________
template<class T>
void TMVA::Option<T>::setValue(const TString& val) {
  std::stringstream str(val.Data());
  str >> fValue;
  if (fRefPtr!=0) *fRefPtr = fValue;
}


#endif
