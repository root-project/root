// @(#)root/tmva $Id: Configurable.cxx,v 1.25 2007/06/15 22:01:31 andreas.hoecker Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Configurable                                                          *
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
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
//Begin_Html
/*
  Base Class for all classes that would like to habe option parsing enabled
*/
//End_Html
//_______________________________________________________________________

#include <string>
#include <fstream>
#include <stdlib.h>

#include "TROOT.h"
#include "TSystem.h"
#include "TObjString.h"
#include "TQObject.h"
#include "TSpline.h"
#include "TMatrix.h"
#include "TMath.h"
#include "TFile.h"
#include "TKey.h" 

#ifndef ROOT_TMVA_Configurable
#include "TMVA/Configurable.h"
#endif

// don't change this flag without a good reason ! The FitterBase code won't work anymore !!!
// #define TMVA_Configurable_SanctionUnknownOption kTRUE 

ClassImp(TMVA::Configurable)

//_______________________________________________________________________
TMVA::Configurable::Configurable( const TString & theOption)  
   : fOptions                    ( theOption ),
     fLooseOptionCheckingEnabled ( kTRUE ),
     fLastDeclaredOption         ( 0 ),
     fLogger                     ( this )
{
   // constructor
}

//_______________________________________________________________________
TMVA::Configurable::~Configurable()
{
   // default destructur
}

//_______________________________________________________________________
void TMVA::Configurable::SplitOptions(const TString& theOpt, TList& loo) const
{
   // splits the option string at ':' and fills the list 'loo' with the primitive strings
   TString splitOpt(theOpt);
   while (splitOpt.Length()>0) {
      if ( ! splitOpt.Contains(':') ) {
         loo.Add(new TObjString(splitOpt));
         splitOpt = "";
      } 
      else {
         TString toSave = splitOpt(0,splitOpt.First(':'));
         loo.Add(new TObjString(toSave.Data()));
         splitOpt = splitOpt(splitOpt.First(':')+1,splitOpt.Length());
      }
   }
}

//_______________________________________________________________________
void TMVA::Configurable::ResetSetFlag() {
   // resets the IsSet falg for all declare options
   // to be called before options are read from stream

   TListIter decOptIt(&fListOfOptions); // declared options
   while (OptionBase* decOpt = (OptionBase*) decOptIt()) { // loop over declared options
      decOpt->fIsSet = kFALSE;
   }
}

//_______________________________________________________________________
void TMVA::Configurable::ParseOptions( Bool_t verbose ) 
{
   // options parser
   if (verbose) {
      fLogger << kINFO << "Parsing option string: " << Endl;
      TString optionsWithoutTilde(fOptions);
      optionsWithoutTilde.ReplaceAll(TString("~"),TString(""));
      fLogger << kINFO << "\"" << optionsWithoutTilde << "\"" << Endl;
   }
   
   TList loo; // the List Of Options in the parsed string
   
   fOptions = fOptions.Strip(TString::kLeading, ':');
   
   // separate the options by the ':' marker
   SplitOptions(fOptions, loo);
   fOptions = "";

   // loop over the declared options and check for their availability
   std::map<TString, std::vector<std::pair<Int_t, TString> > > arrayTypeOptions;

   TListIter decOptIt(&fListOfOptions); // declared options
   TListIter setOptIt(&loo);   // parsed options
   while (TObjString * os = (TObjString*) setOptIt()) { // loop over parsed options

      TString s = os->GetString();
      // the tilde in the beginning is an indication that the option
      // has been accepted during previous parsing
      //
      // while parsing this option string eventual appearances of the
      // tilde will be preserved, for correctly parsed options a new
      // one will be added (in the end it will be checked if all
      // options were parsed
      Bool_t preserveTilde = s.BeginsWith('~');
      s = s.Strip(TString::kLeading, '~');

      Bool_t paramParsed = kFALSE;
      if (s.Contains('=')) { // desired way of setting an option: "...:optname=optvalue:..."
         TString optname = s(0,s.First('=')); optname.ToLower(); 
         TString optval = s(s.First('=')+1,s.Length());
         Int_t idx = -1;

         // deal with array specification
         if (optname.Contains('[')) {
            TString s = optname(optname.First('[')+1,100);
            s.Remove(s.First(']'));
            std::stringstream str(s.Data());
            str >> idx;                              // save the array index
            optname.Remove(optname.First('['));      // and remove [idx] from the option name
         }

         OptionBase * decOpt = (OptionBase *)fListOfOptions.FindObject(optname);
         TListIter optIt(&fListOfOptions);
         if (decOpt!=0) {
            if (decOpt->IsSet())
               fLogger << kWARNING << "Value for option " << decOpt->GetName() 
                       << " was previously set to " << decOpt->GetValue() << Endl;

            if (!decOpt->HasPreDefinedVal() || (decOpt->HasPreDefinedVal() && decOpt->IsPreDefinedVal(optval)) ) {
               if (decOpt->IsArrayOpt()) {
                  // if no index was found then we assume the value is to be set for the entire array
                  if (idx==-1) {
                     decOpt->SetValue(optval);
                  } 
                  else {
                     // since we don't know what else is comming we just put everthing into a map
                     if (!decOpt->SetValue(optval, idx))
                        fLogger << kFATAL << "Index " << idx << " too large for option " << decOpt->TheName()
                                << ", allowed range is [0," << decOpt->GetArraySize()-1 << "]" << Endl;
                  }
               } 
               else {
                  if (idx!=-1) 
                     fLogger << kFATAL << "Option " << decOpt->TheName()
                             << " is not an array, but you specified an index" << Endl;
                  decOpt->SetValue(optval);
               }
               paramParsed = kTRUE;
            }
            else fLogger << kFATAL << "Option " << decOpt->TheName() 
                         << " has no predefined value " << optval << Endl;               
         }
      }

      // boolean variables can be specified by just their name (!name), 
      // which will set the to true (false):  ...:V:...:!S:..
      Bool_t preserveNotSign = kFALSE;
      if (!paramParsed) {
         Bool_t hasNotSign = kFALSE;
         if (s.BeginsWith("!")) { s.Remove(0,1); preserveNotSign = hasNotSign = kTRUE; }
         TString optname(s); optname.ToLower();
         OptionBase* decOpt = 0;
         Bool_t optionExists = kFALSE;
         TListIter optIt(&fListOfOptions);
         while ( (decOpt = (OptionBase*)optIt()) !=0) {
            TString predOptName(decOpt->GetName());
            predOptName.ToLower();
            if (predOptName == optname) optionExists = kTRUE;
            if (dynamic_cast<Option<bool>*>(decOpt)==0) continue; // not a boolean option
            if (predOptName == optname) break;
         }
        
         if (decOpt != 0) {
            decOpt->SetValue( hasNotSign ? "0" : "1" );
            paramParsed = kTRUE;
         } else {
            if (optionExists && hasNotSign) {
               fLogger << kFATAL << "Negating a non-boolean variable " << optname
                       << ", please check the opions for method " << GetName() << Endl;
            }
         }
      }

      if (!paramParsed && LooseOptionCheckingEnabled()) {
         // loose options specification, loops through the possible string 
         // values any parameter can have not applicable for boolean or floats
         decOptIt.Reset();
         while (OptionBase* decOpt = (OptionBase*) decOptIt()) {
            if (decOpt->HasPreDefinedVal() && decOpt->IsPreDefinedVal(s) ) {
               paramParsed = decOpt->SetValue(s);
               break;
            }
         }
      }
   
      if(fOptions!="") fOptions += ":";
      if(paramParsed || preserveTilde) fOptions += '~';
      if(paramParsed || preserveNotSign) fOptions += '!';
      fOptions += s;

   }

   if (verbose) PrintOptions();
}

//______________________________________________________________________
void TMVA::Configurable::CheckForUnusedOptions() const 
{
   // checks for unused options in option string
   TString theOpt(fOptions);
   theOpt = theOpt.Strip(TString::kLeading, ':');
   
   // separate the options by the ':' marker
   TList loo; // the List of Options in the parsed string
   SplitOptions(theOpt, loo);

   TListIter setOptIt(&loo);   // options in a list
   TString unusedOptions("");
   while (TObjString * os = (TObjString*) setOptIt()) { // loop over parsed options

      TString s = os->GetString();
      if( !s.BeginsWith('~') ) {
         if(unusedOptions!="") unusedOptions += ':';
         unusedOptions += s;
      }
   }
   if(unusedOptions!="")
      fLogger << kFATAL
              << "The following options were specified, but could not be interpreted: \'"
              << unusedOptions << "\', please check!" << Endl;
}

//______________________________________________________________________
void TMVA::Configurable::PrintOptions() const 
{
   // prints out the options set in the options string and the defaults

   fLogger << kINFO << "The following options are set:" << Endl;

   TListIter optIt( & fListOfOptions );
   fLogger << kINFO << "- By User:" << Endl;
   Bool_t found = kFALSE;
   while (OptionBase* opt = (OptionBase *) optIt()) {
      if (opt->IsSet()) { fLogger << kINFO << "    "; opt->Print(fLogger); fLogger << Endl; found = kTRUE; }
   }
   if (!found) fLogger << kINFO << "    <none>" << Endl;

   optIt.Reset();
   fLogger << kINFO << "- Default:" << Endl;
   found = kFALSE;
   while (OptionBase* opt = (OptionBase *) optIt()) {
      if (!opt->IsSet()) { fLogger << kINFO << "    "; opt->Print(fLogger); fLogger << Endl; found = kTRUE; }
   }
   if (!found) fLogger << kINFO << "    <none>" << Endl;
}

//______________________________________________________________________
void TMVA::Configurable::WriteOptionsToStream( ostream& o, const TString& prefix ) const 
{
   // write options to output stream (e.g. in writing the MVA weight files

   TListIter optIt( &fListOfOptions );
   o << prefix << "# Set by User:" << endl;
   while (OptionBase * opt = (OptionBase *) optIt()) 
      if (opt->IsSet()) { o << prefix; opt->Print(o); o << endl; }
   optIt.Reset();
   o << prefix << "# Default:" << endl;
   while (OptionBase * opt = (OptionBase *) optIt()) 
      if (!opt->IsSet()) { o << prefix; opt->Print(o); o << endl; }
   o << prefix << "##" << endl;
}

//______________________________________________________________________
void TMVA::Configurable::ReadOptionsFromStream(istream& istr)
{
   // read option back from the weight file

   // first set the IsSet flag of all declared options to false
   // that is only necessary in our factory, when we test right
   // after the training
   ResetSetFlag();

   fOptions = "";
   char buf[512];
   istr.getline(buf,512);
   TString stropt, strval;
   while (istr.good() && !istr.eof() && !(buf[0]=='#' && buf[1]=='#')) { // if line starts with ## return
      char *p = buf;
      while (*p==' ' || *p=='\t') p++; // 'remove' leading whitespace
      if (*p=='#' || *p=='\0') {
         istr.getline(buf,512); // reading the next line
         continue; // if comment or empty line, read the next line
      }
      std::stringstream sstr(buf);
      sstr >> stropt >> strval;
      stropt.ReplaceAll(':','=');
      strval.ReplaceAll("\"","");
      if (fOptions.Length()!=0) fOptions += ":";
      fOptions += stropt;
      fOptions += strval;
      istr.getline(buf,512); // reading the next line
   }
}
