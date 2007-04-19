// @(#)root/tmva $\Id$
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
 *      CERN, Switzerland,                                                        *
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
void TMVA::Configurable::SplitOptions(TString& theOpt, TList& loo)
{
   // splits the option string at ':' and fills the list 'loo' with the primitive strings
   while (theOpt.Length()>0) {
      if ( ! theOpt.Contains(':') ) {
         loo.Add(new TObjString(theOpt));
         theOpt = "";
      } 
      else {
         TString toSave = theOpt(0,theOpt.First(':'));
         loo.Add(new TObjString(toSave.Data()));
         theOpt = theOpt(theOpt.First(':')+1,theOpt.Length());
      }
   }  
}

//_______________________________________________________________________
void TMVA::Configurable::ParseOptions( Bool_t verbose ) 
{
   // options parser

   if (verbose) {
      fLogger << kINFO << "Parsing option string: " << Endl;
      fLogger << kINFO << "\"" << fOptions << "\"" << Endl;
   }
   
   TString theOpt(fOptions);
   TList loo; // the List Of Options in the parsed string
   
   theOpt = theOpt.Strip(TString::kLeading, ':');
   
   // separate the options by the ':' marker
   SplitOptions(theOpt, loo);

   // loop over the declared options and check for their availability
   std::map<TString, std::vector<std::pair<Int_t, TString> > > arrayTypeOptions;

   TListIter decOptIt(&fListOfOptions); // declared options
   TListIter setOptIt(&loo);   // parsed options
   while (TObjString * os = (TObjString*) setOptIt()) { // loop over parsed options
      TString s = os->GetString();
      bool paramParsed = false;
      if (s.Contains('=')) { // desired way of setting an option: "...:optname=optvalue:..."
         TString optname = s(0,s.First('=')); optname.ToLower();
         TString optval = s(s.First('=')+1,s.Length());
         Int_t idx = -1;

         // deal with array specification
         if(optname.Contains('[')) {
            TString s = optname(optname.First('[')+1,100);
            s.Remove(s.First(']'));
            std::stringstream str(s.Data());
            str >> idx;                              // save the array index
            optname.Remove(optname.First('['));      // and remove [idx] from the option name
         }

         OptionBase * decOpt = (OptionBase *)fListOfOptions.FindObject(optname);;
         TListIter optIt(&fListOfOptions);
         if (decOpt!=0) {
            if (decOpt->IsSet() && verbose)
               fLogger << kWARNING << "Value for option " << decOpt->GetName() 
                       << " was previously set to " << decOpt->GetValue() << Endl;

            if (!decOpt->HasPreDefinedVal() || (decOpt->HasPreDefinedVal() && decOpt->IsPreDefinedVal(optval)) ) {
               if(decOpt->IsArrayOpt()) {
                  // if no index was found then we assume the value is to be set for the entire array
                  if(idx==-1) {
                     decOpt->SetValue(optval);
                  } else {
                     // since we don't know what else is comming we just put everthing into a map
                     if(!decOpt->SetValue(optval, idx))
                        fLogger << kFATAL << "Index " << idx << " too large for option " << decOpt->TheName()
                                << ", allowed range is [0," << decOpt->GetArraySize()-1 << "]" << Endl;
                  }
               } else {
                  if(idx!=-1) 
                     fLogger << kFATAL << "Option " << decOpt->TheName()
                             << " is not an array, but you specified an index" << Endl;
                  decOpt->SetValue(optval);
               }
               paramParsed = kTRUE;
            }
            else fLogger << kFATAL << "Option " << decOpt->TheName() 
                         << " has no predefined value " << optval << Endl;               
         }
         else fLogger << kFATAL << "Option " << optname << " not found!" << Endl;
      }

      // boolean variables can be specified by just their name (!name), 
      // which will set the to true (false):  ...:V:...:!S:..
      if (!paramParsed) {
         bool hasNot = false;
         if (s.BeginsWith("!")) { s.Remove(0,1); hasNot=true; }
         TString optname(s);optname.ToLower();
         OptionBase * decOpt = 0;
         TListIter optIt(&fListOfOptions);
         while ( (decOpt = (OptionBase*)optIt()) !=0) {
            if (dynamic_cast<Option<bool>*>(decOpt)==0) continue; // not a boolean option
            TString predOptName(decOpt->GetName());
            predOptName.ToLower();
            if (predOptName == optname) break;
         }
        
         if (decOpt!=0) {
            decOpt->SetValue(hasNot?"0":"1");
            paramParsed = true;
         } 
         else {
            if (hasNot) {
               fLogger << kFATAL << "Negating a non-boolean variable " << optname
                       << ", please check the opions for method " << GetName() << Endl;
            }
         }
      }

      if (!paramParsed && LooseOptionCheckingEnabled()) {
         // loose options specification, loops through the possible string 
         // values any parameter can have not applicable for boolean or floats
         decOptIt.Reset();
         while (OptionBase * decOpt = (OptionBase*) decOptIt()) {
            if (decOpt->HasPreDefinedVal() && decOpt->IsPreDefinedVal(s) ) {
               paramParsed = decOpt->SetValue(s);
               break;
            }
         }
      }
   
      if (!paramParsed) {
         fLogger << kFATAL << "Cannot interpret option \"" << s << "\" for method " 
                 << GetName() << ", please check" << Endl;
      } 
   }

   // now we have to go through the map of arrar-type options and set them
//    std::map<TString, std::vector<std::pair<Int_t, TString> > >::iterator mIt = arrayTypeOptions.begin();
//    for(;mIt != arrayTypeOptions.end(); mIt++) {
//       cout << "UUUUUU " << mIt->first << " => " << flush;
//       std::vector<std::pair<Int_t, TString> >::iterator = mIt->second
//    }
   if(verbose) PrintOptions();
}

//______________________________________________________________________
void TMVA::Configurable::PrintOptions() const 
{
   // prints out the options set in the options string and the defaults

   fLogger << kINFO << "The following options are set:" << Endl;
   TListIter optIt( & fListOfOptions );
   fLogger << kINFO << "- By User:" << Endl;
   while (OptionBase * opt = (OptionBase *) optIt()) {
      if (opt->IsSet()) { fLogger << kINFO << "    "; opt->Print(fLogger); fLogger << Endl; }
   }
   optIt.Reset();
   fLogger << kINFO << "- Default:" << Endl;
   while (OptionBase * opt = (OptionBase *) optIt()) {
      if (!opt->IsSet()) { fLogger << kINFO << "    "; opt->Print(fLogger); fLogger << Endl; }
   }
}

//______________________________________________________________________
void TMVA::Configurable::WriteOptionsToStream(ostream& o) const 
{
   // write options to output stream (e.g. in writing the MVA weight files

   TListIter optIt( & fListOfOptions );
   o << "# Set by User:" << endl;
   while (OptionBase * opt = (OptionBase *) optIt()) if (opt->IsSet()) { opt->Print(o); o << endl; }
   optIt.Reset();
   o << "# Default:" << endl;
   while (OptionBase * opt = (OptionBase *) optIt()) if (!opt->IsSet()) { opt->Print(o); o << endl; }
   o << "##" << endl;
}

//______________________________________________________________________
void TMVA::Configurable::ReadOptionsFromStream(istream& istr)
{
   // read option back from the weight file

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
