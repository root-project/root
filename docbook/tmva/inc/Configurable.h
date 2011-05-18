// @(#)root/tmva $Id$   
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
 *      CERN, Switzerland                                                         * 
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

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif

#ifndef ROOT_TMVA_Option
#include "TMVA/Option.h"
#endif

namespace TMVA {

   class Configurable : public TObject {

   public:

      // constructur
      Configurable( const TString& theOption = "" );
      
      // default destructur
      virtual ~Configurable();

      // parse the internal option string
      virtual void ParseOptions();

      // print list of defined options
      void PrintOptions() const;

      virtual const char* GetName()      const { return GetConfigName(); }
      const char* GetConfigName()        const { return fConfigName; }
      const char* GetConfigDescription() const { return fConfigDescription; }
      void SetConfigName       ( const char* n ) { fConfigName        = TString(n); }
      void SetConfigDescription( const char* d ) { fConfigDescription = TString(d); }

      // Declare option and bind it to a variable
      template<class T> 
      OptionBase* DeclareOptionRef( T& ref, const TString& name, const TString& desc = "" );

      template<class T> 
      OptionBase* DeclareOptionRef( T*& ref, Int_t size, const TString& name, const TString& desc = "" );

      // Add a predefined value to the last declared option
      template<class T>
      void AddPreDefVal(const T&);
      
      void CheckForUnusedOptions() const;

      const TString& GetOptions() const { return fOptions; }
      void SetOptions(const TString& s) { fOptions = s; }

      void WriteOptionsToStream ( std::ostream& o, const TString& prefix ) const;
      void ReadOptionsFromStream( istream& istr );

      void AddOptionsXMLTo( void* parent ) const;
      void ReadOptionsFromXML( void* node );

   protected:
      
      Bool_t LooseOptionCheckingEnabled() const { return fLooseOptionCheckingEnabled; }
      void   EnableLooseOptions( Bool_t b = kTRUE ) { fLooseOptionCheckingEnabled = b; }

      void   WriteOptionsReferenceToFile();

      void   ResetSetFlag();

      const TString& GetReferenceFile() const { return fReferenceFile; }

   private:

      // splits the option string at ':' and fills the list 'loo' with the primitive strings
      void SplitOptions(const TString& theOpt, TList& loo) const;

      TString     fOptions;                          //! options string
      Bool_t      fLooseOptionCheckingEnabled;       //! checker for option string

      // classes and method related to easy and flexible option parsing
      OptionBase* fLastDeclaredOption;  //! last declared option
      TList       fListOfOptions;       //! option list

      TString     fConfigName;          // the name of this configurable
      TString     fConfigDescription;   // description of this configurable
      TString     fReferenceFile;       // reference file for options writing

   protected:

      // the mutable declaration is needed to use the logger in const methods
      MsgLogger& Log() const { return *fLogger; }                       

   public:

      // set message type
      void SetMsgType( EMsgType t ) { fLogger->SetMinType(t); }


   private:

      mutable MsgLogger* fLogger;                     //! message logger

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
      Log() << kFATAL << "Option \"" << name 
            << "\" not declared, please check the syntax of your option string" << Endl;
}

#endif

