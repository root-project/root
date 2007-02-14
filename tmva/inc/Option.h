// @(#)root/tmva $Id: Option.h,v 1.6 2006/11/20 15:35:28 brun Exp $   
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
#include <vector>

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

namespace TMVA {

   class OptionBase : public TObject {

   public:

      OptionBase( const TString& name, const TString& desc ) 
         : fName(name), fDescription(desc), fIsSet(false), fLogger(this) {}
      virtual ~OptionBase() {}
         
      virtual const char* GetName() const { TString s(fName); s.ToLower(); return s.Data(); }
      virtual const char* TheName() const { return fName; }
      virtual TString     GetValue() const = 0;

      Bool_t IsSet() const { return fIsSet; }
      const TString& Description() const { return fDescription; }
      virtual Bool_t IsPreDefinedVal(const TString&) const = 0;
      virtual Bool_t HasPreDefinedVal() const = 0;
      Bool_t         SetValue(const TString& vs);

      virtual void Print(const Option_t* o) const { return TObject::Print(o); }
      virtual void Print(ostream&, Int_t levelofdetail=0) const = 0;
      virtual const TString Type() const = 0;

   private:

      virtual void SetValueLocal(const TString& vs) = 0;         

      const TString fName;        // name of variable 
      const TString fDescription; // its description
      Bool_t        fIsSet;       // set by user ?

   protected:

      mutable MsgLogger fLogger;  // message logger

   };
      
   // ---------------------------------------------------------------------------

   template <class T>
      class Option : public OptionBase {

   public:

      Option( const TString& name, const TString& desc ) :
      OptionBase( name, desc ),
         fRefPtr(0)
         {}
      Option( T& ref, const TString& name, const TString& desc ) 
         : OptionBase(name, desc), fValue(ref), fRefPtr(&ref) {}
      virtual ~Option() {}

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

      virtual void   SetValueLocal(const TString& val);
      virtual Bool_t IsPreDefinedValLocal(const T&) const;

      T fValue;
      T* fRefPtr;

      std::vector<T> fPreDefs;  // templated vector
   };      
} // namespace

//______________________________________________________________________
template<class T>
Bool_t TMVA::Option<T>::HasPreDefinedVal() const 
{
   // template 
   return (fPreDefs.size()!=0);
}

template<class T>
Bool_t TMVA::Option<T>::IsPreDefinedVal(const TString& val) const 
{
   // template 
   T tmpVal;
   std::stringstream str(val.Data());
   str >> tmpVal;
   return IsPreDefinedValLocal(tmpVal);
}

namespace TMVA {
//______________________________________________________________________
template<class T>
inline Bool_t TMVA::Option<T>::IsPreDefinedValLocal(const T& val) const 
{
   // template 
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
template<>
inline Bool_t TMVA::Option<Bool_t>::IsPreDefinedValLocal(const Bool_t& val) const 
{
   // template specialization for Bool_t 
   if (val); // dummy
   return kTRUE;
}

template<>
inline Bool_t TMVA::Option<Float_t>::IsPreDefinedValLocal(const Float_t& val) const 
{
   // template specialization for Float_t 
   if (val); // dummy
   return kTRUE;
}

template<>
inline Bool_t TMVA::Option<TString>::IsPreDefinedValLocal(const TString& val) const 
{
   // template specialization for Bool_t 
   TString tVal(val);
   tVal.ToLower();
   if (fPreDefs.size()==0) return kFALSE; // if nothing pre-defined then allow everything
   Bool_t foundPreDef = kFALSE;   
   std::vector<TString>::const_iterator predefIt;
   predefIt = fPreDefs.begin();
   for (;predefIt!=fPreDefs.end(); predefIt++) {
      TString s(*predefIt);
      s.ToLower();
      if (s==tVal) { foundPreDef = kTRUE; break; }
   }
   return foundPreDef;
}



//______________________________________________________________________
template<class T>
inline void TMVA::Option<T>::AddPreDefVal(const T&val) 
{
   // template 
   fPreDefs.push_back(val);
}

template<>
inline void TMVA::Option<Bool_t>::AddPreDefVal(const Bool_t&) 
{
   // template specialization for Bool_t 
   
   fLogger << kWARNING << "<AddPreDefVal> predefined values for Option<Bool_t> don't make sense" 
           << Endl;
}

template<>
inline void TMVA::Option<Float_t>::AddPreDefVal(const Float_t&) 
{
   // template specialization for Float_t 

   fLogger << kWARNING << "<AddPreDefVal> predefined values for Option<Float_t> don't make sense" 
           << Endl;
}


//______________________________________________________________________
template<class T>
inline void TMVA::Option<T>::Print(ostream& os, Int_t levelofdetail) const 
{
   // template 
   if (levelofdetail); // dummy to avoid compiler warnings
   os << TheName() << ": " << GetValue() << " [" << Description() << "]";
}

template<>
inline void TMVA::Option<Bool_t>::Print(ostream& os, Int_t levelofdetail) const 
{
   // template specialization for Bool_t printing
   if (levelofdetail); // dummy to avoid compiler warnings
   os << TheName() << ": " << (Value() ? "True" : "False") << " [" << Description() << "]";
}

template<>
inline void TMVA::Option<TString>::Print(ostream& os, Int_t levelofdetail) const 
{
   // template specialization for TString printing
   os << TheName() << ": " << "\"" << Value() << "\"" << " [" << Description() << "]";
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
template<class T>
inline const TString TMVA::Option<T>::Type() const 
{
   // template 
   return "undefined type";
}

template<>
inline const TString TMVA::Option<Bool_t>::Type() const 
{
   // template specialization for Bool_t query
   return "bool";
}

template<>
inline const TString TMVA::Option<Float_t>::Type() const 
{
   // template specialization for Float_t query
   return "float";
}

template<>
inline const TString TMVA::Option<Int_t>::Type() const 
{
   // template specialization for Int_t query
   return "int";
}

template<>
inline const TString TMVA::Option<TString>::Type() const 
{
   // template specialization for TString query
   return "TString";
}

template<>
inline const TString TMVA::Option<Double_t>::Type() const 
{
   // template specialization for Double_t query
   return "double";
}

//______________________________________________________________________
template<class T>
inline void TMVA::Option<T>::SetValueLocal(const TString& val) 
{
   // template 
   std::stringstream str(val.Data());
   str >> fValue;
   if (fRefPtr!=0) *fRefPtr = fValue;
}

template<>
inline void TMVA::Option<TString>::SetValueLocal(const TString& val) 
{
   // set TString value
   TString valToSet(val);
   if (fPreDefs.size()!=0) {
      TString tVal(val);
      tVal.ToLower();
      std::vector<TString>::const_iterator predefIt;
      predefIt = fPreDefs.begin();
      for (;predefIt!=fPreDefs.end(); predefIt++) {
         TString s(*predefIt);
         s.ToLower();
         if (s==tVal) { valToSet = *predefIt; break; }
      }
   }

   std::stringstream str(valToSet.Data());
   str >> fValue;
   if (fRefPtr!=0) *fRefPtr = fValue;
}

template<>
inline void TMVA::Option<Bool_t>::SetValueLocal(const TString& val) 
{
   // set Bool_t value
   TString valToSet(val);
   valToSet.ToLower();
   if(valToSet=="1" || valToSet=="true" || valToSet=="ktrue" || valToSet=="t") {
      fValue = true;
   } 
   else if(valToSet=="0" || valToSet=="false" || valToSet=="kfalse" || valToSet=="f") {
      fValue = false;
   } 
   else {
      fLogger << kFATAL << "<SetValueLocal> value \'" << val 
              << "\' can not be interpreted as boolean" << Endl;
   }
   if (fRefPtr!=0) *fRefPtr = fValue;
}

}
#endif
