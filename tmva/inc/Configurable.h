// @(#)root/tmva $\Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Configurable                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Base class for all classes with option parsing                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_Configurable
#define ROOT_TMVA_Configurable

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Configurable                                                         //
//                                                                      //
// Base class for all classes with option parsing                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Riostream.h"
#include "TObject.h"
#include "TList.h"
#include <sstream>
#include <vector>
#include <iostream>

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_Option
#include "TMVA/Option.h"
#endif

namespace TMVA {

   class MsgLogger;

   class Configurable : public virtual TObject {

   public:

      // constructur
      Configurable( const TString & theOption = "" );
      
      // default destructur
      virtual ~Configurable();

      // parse the internal option string
      void ParseOptions( Bool_t verbose = kTRUE);

      // print list of defined options
      void PrintOptions() const;

      virtual const char* GetName() const { return fName; }
      void SetName( const char* n ) { fName = TString(n); }

      // Declare option and bind it to a variable
      template<class T> 
      OptionBase* DeclareOptionRef( T& ref, const TString& name, const TString& desc = "" );

      template<class T> 
      OptionBase* DeclareOptionRef( T*& ref, Int_t size, const TString& name, const TString& desc = "" );

      // Add a predefined value to the last declared option
      template<class T>
      void AddPreDefVal(const T&);

   protected:
      
      const TString & GetOptions() const { return fOptions; }
      Bool_t    LooseOptionCheckingEnabled() const { return fLooseOptionCheckingEnabled; }
      void      EnableLooseOptions( Bool_t b = kTRUE ) { fLooseOptionCheckingEnabled = b; }

      void WriteOptionsToStream(ostream& o) const;
      void ReadOptionsFromStream(istream& istr);

   private:

      // splits the option string at ':' and fills the list 'loo' with the primitive strings
      void SplitOptions(TString& theOpt, TList& loo);

      TString     fOptions;                          // options string
      Bool_t      fLooseOptionCheckingEnabled;       // checker for option string

      // classes and method related to easy and flexible option parsing
      OptionBase* fLastDeclaredOption;  // last declared option
      TList       fListOfOptions;       // option list

      TString     fName;                // the name of this configurable
      
      // the mutable declaration is needed to use the logger in const methods
      mutable MsgLogger fLogger; // message logger

   private:

      template <class T>
      void AssignOpt( const TString& name, T& valAssign ) const;
      
   public:
      ClassDef(Configurable,0)  // Virtual base class for all TMVA method

   };
} // namespace TMVA

// Template Declarations go here

//______________________________________________________________________
template <class T>
TMVA::OptionBase* TMVA::Configurable::DeclareOptionRef( T& ref, const TString& name, const TString& desc) 
{
   // set the reference for an option
   OptionBase* o = new Option<T>(ref, name, desc);
   fListOfOptions.Add(o);
   fLastDeclaredOption = o;
   return o;
}

template <class T>
TMVA::OptionBase* TMVA::Configurable::DeclareOptionRef( T*& ref, Int_t size, const TString& name, const TString& desc) 
{
   // set the reference for an option
   OptionBase* o = new Option<T*>(ref, size, name, desc);
   fListOfOptions.Add(o);
   fLastDeclaredOption = o;
   return o;
}

//______________________________________________________________________
template<class T>
void TMVA::Configurable::AddPreDefVal(const T& val) 
{
   // add predefined option value
   Option<T>* oc = dynamic_cast<Option<T>*>(fLastDeclaredOption);
   if(oc!=0) oc->AddPreDefVal(val);
}

//______________________________________________________________________
template <class T>
void TMVA::Configurable::AssignOpt(const TString& name, T& valAssign) const 
{
   // assign an option
   TObject* opt = fListOfOptions.FindObject(name);
   if (opt!=0) valAssign = ((Option<T>*)opt)->Value();
   else 
      fLogger << kFATAL << "Option \"" << name 
              << "\" not declared, please check the syntax of your option string" << endl;
}

#endif

