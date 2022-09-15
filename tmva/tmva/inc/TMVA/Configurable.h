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

#include "TNamed.h"
#include "TList.h"

#include "TMVA/Option.h"

namespace TMVA {

   class Configurable : public TNamed {

   public:

      // constructor
      Configurable( const TString& theOption = "" );

      // default destructor
      virtual ~Configurable();

      // parse the internal option string
      virtual void ParseOptions();

      // print list of defined options
      void PrintOptions() const;

      const char* GetConfigName()        const { return GetName(); }
      const char* GetConfigDescription() const { return fConfigDescription; }
      void SetConfigName       ( const char* n ) { SetName(n); }
      void SetConfigDescription( const char* d ) { fConfigDescription = TString(d); }

      // Declare option and bind it to a variable
      template<class T>
         OptionBase* DeclareOptionRef( T& ref, const TString& name, const TString& desc = "" );

      template<class T>
         OptionBase* DeclareOptionRef( T*& ref, Int_t size, const TString& name, const TString& desc = "" );

      // Add a predefined value to the last declared option
      template<class T>
         void AddPreDefVal(const T&);

      // Add a predefined value to the option named optname
      template<class T>
         void AddPreDefVal(const TString&optname ,const T&);


      void CheckForUnusedOptions() const;

      const TString& GetOptions() const { return fOptions; }
      void SetOptions(const TString& s) { fOptions = s; }

      void WriteOptionsToStream ( std::ostream& o, const TString& prefix ) const;
      void ReadOptionsFromStream( std::istream& istr );

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

      TString     fOptions;                          ///< options string
      Bool_t      fLooseOptionCheckingEnabled;       ///< checker for option string

      // classes and method related to easy and flexible option parsing
      OptionBase* fLastDeclaredOption;  ///<! last declared option
      TList       fListOfOptions;       ///< option list

      TString     fConfigDescription;   ///< description of this configurable
      TString     fReferenceFile;       ///< reference file for options writing

   public:

      // the mutable declaration is needed to use the logger in const methods
      MsgLogger& Log() const { return *fLogger; }

      // set message type
      void SetMsgType( EMsgType t ) { fLogger->SetMinType(t); }

   protected:
      mutable MsgLogger* fLogger;                     ///<! message logger

   private:


      template <class T>
         void AssignOpt( const TString& name, T& valAssign ) const;

   public:

      ClassDef(Configurable,1);  // Virtual base class for all TMVA method

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
   // add predefined option value to the last declared option
   Option<T>* oc = dynamic_cast<Option<T>*>(fLastDeclaredOption);
   if(oc) oc->AddPreDefVal(val);
}

//______________________________________________________________________
template<class T>
void TMVA::Configurable::AddPreDefVal(const TString &optname, const T& val)
{
   // add predefined option value to the option named optname

   TListIter optIt( &fListOfOptions );
   while (OptionBase * op = (OptionBase *) optIt()) {
      if (optname == TString(op->TheName())){
         Option<T>* oc = dynamic_cast<Option<T>*>(op);
         if(oc){
            oc->AddPreDefVal(val);
            return;
         }
         else{
            Log() << kFATAL << "Option \"" << optname
                  << "\" was found, but somehow I could not convert the pointer properly.. please check the syntax of your option declaration" << Endl;
            return;
         }

      }
   }
   Log() << kFATAL << "Option \"" << optname
         << "\" is not declared, hence cannot add predefined value, please check the syntax of your option declaration" << Endl;

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

