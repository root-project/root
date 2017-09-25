// @(#)root/alien:$Id$
// Author: Jan Fiete Grosse-Oetringhaus   28/9/2004
//         Lucia.Jancurova@cern.ch Slovakia 2007

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienJDL                                                            //
//                                                                      //
// Class which creates JDL files for the alien middleware               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAlienJDL.h"
#include "TGrid.h"
#include "TGridJob.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TObjString.h"
#include "TObjArray.h"

ClassImp(TAlienJDL);

////////////////////////////////////////////////////////////////////////////////
/// Sets the executable.

void TAlienJDL::SetExecutable(const char *value, const char *description)
{
   if (value)
      SetValue("Executable", AddQuotes(value));
   if (description)
      SetDescription("Executable", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the arguments.

void TAlienJDL::SetArguments(const char *value, const char *description)
{
   if (value)
      SetValue("Arguments", AddQuotes(value));
   if (description)
      SetDescription("Arguments", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets eMail address.

void TAlienJDL::SetEMail(const char *value, const char *description)
{
   if (value)
      SetValue("Email", AddQuotes(value));
   if (description)
      SetDescription("Email", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets OutputDirectory.

void TAlienJDL::SetOutputDirectory(const char *value, const char *description)
{
   if (value)
      SetValue ("OutputDir", AddQuotes(value));
   if (description)
      SetDescription("OutputDir", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets merged OutputDirectory.

void TAlienJDL::SetMergedOutputDirectory ( const char * value,const char* description)
{
   if (value)
      SetValue ("MergeOutputDir", AddQuotes(value));
   if (description)
      SetDescription("MergeOutputDir", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets OutputDirectory.

void TAlienJDL:: SetPrice(UInt_t price,const char* description)
{
   TString pricestring(Form("%d",price));
   SetValue("Price", AddQuotes(pricestring.Data()));

   if (description)
      SetDescription("Price", description);
}

////////////////////////////////////////////////////////////////////////////////
/// To inform AliEn master about estimated Time-To-Live of included nodes.

void TAlienJDL:: SetTTL(UInt_t ttl, const char *description)
{
   TString ttlstring;
   ttlstring+= ttl;
   SetValue("TTL", ttlstring.Data());

   if (description)
      SetDescription("TTL", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets Job Tag

void TAlienJDL::SetJobTag(const char* value,const char* description)
{
   if (value)
      SetValue("JobTag", AddQuotes(value));
   if (description)
      SetDescription("JobTag", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets InputDataListFormat - can be "xml-single" or "xml-multi"

void TAlienJDL::SetInputDataListFormat(const char* value,const char* description)
{
   if (value)
      SetValue("InputDataListFormat", AddQuotes(value));
   if (description)
      SetDescription("InputDataListFormat", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets InputDataList name

void TAlienJDL::SetInputDataList(const char* value,const char* description)
{
   if (value)
      SetValue("InputDataList", AddQuotes(value));
   if (description)
      SetDescription("InputDataList", description);
}


////////////////////////////////////////////////////////////////////////////////
/// Sets the split mode.

void TAlienJDL::SetSplitMode(const char *value, UInt_t maxnumberofinputfiles,
                             UInt_t maxinputfilesize, const char *d1, const char *d2,
                             const char *d3)
{
   if (value && !strcasecmp(value, "SE")) {

      SetSplitArguments(value,d1);
      if (maxnumberofinputfiles) {
         SetSplitModeMaxNumOfFiles(maxnumberofinputfiles,d2);
      }
      if (maxinputfilesize) {
         SetSplitModeMaxInputFileSize(maxinputfilesize,d3);
      }
   } else {
      if (value)
         SetSplitArguments(value,d1);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the SplitMaxNumOfFiles.

void TAlienJDL::SetSplitModeMaxNumOfFiles(UInt_t maxnumberofinputfiles, const char *description)
{
   TString val;
   val += maxnumberofinputfiles;
   SetValue ( "SplitMaxInputFileNumber", AddQuotes ( val.Data() ) );

   if (description)
      SetDescription("SplitMaxInputFileNumber", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the SplitMaxInputFileSize.

void TAlienJDL::SetSplitModeMaxInputFileSize(UInt_t maxinputfilesize, const char *description)
{
   TString val;
   val += maxinputfilesize;
   SetValue ( "SplitMaxInputFileSize", AddQuotes ( val.Data() ) );
   if (description)
      SetDescription("SplitMaxInputFileSize", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the split.

void TAlienJDL::SetSplitArguments(const char *splitarguments, const char *description)
{
   if (splitarguments)
      SetValue("Split", AddQuotes(splitarguments));
   if (description)
      SetDescription("Split", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the validation command.

void TAlienJDL::SetValidationCommand(const char *value, const char *description)
{
   SetValue("Validationcommand", AddQuotes(value));
   if (description)
      SetDescription("Validationcommand", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the Maxium init failed

void TAlienJDL::SetMaxInitFailed(Int_t maxInitFailed, const char *description)
{
   TString str;
   str += maxInitFailed;
   SetValue("MaxInitFailed", AddQuotes(str.Data()));
   if (description)
      SetDescription("MaxInitFailed", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the Own Command

void TAlienJDL::SetOwnCommand(const char *command, const char *value, const char *description)
{
   if ((command) && (value))
     SetValue(command, AddQuotes(value));
   if ((command) && (description))
     SetDescription(command, description);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a requirement.

void TAlienJDL::AddToRequirements(const char *value, const char *description)
{
   if (value)
      AddToReqSet("Requirements", value);
   if (description)
      AddToSetDescription("Requirements", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a file to the input sandbox.

void TAlienJDL::AddToInputSandbox(const char *value, const char *description)
{
   if (value)
      AddToSet("InputFile", value);
   if (description)
      AddToSetDescription("InputFile", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a file to the output sandbox.

void TAlienJDL::AddToOutputSandbox(const char *value, const char *description)
{
   if (value)
      AddToSet("OutputFile", value);
   if (description)
      AddToSetDescription("OutputFile", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a file to the input data.

void TAlienJDL::AddToInputData(const char *value, const char *description)
{
   if (value)
      AddToSet("InputData", value);
   if (description)
      AddToSetDescription("InputData", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a file to the input data collection.

void TAlienJDL::AddToInputDataCollection(const char *value, const char *description)
{
   if (value)
      AddToSet("InputDataCollection", value);
   if (description)
      AddToSetDescription("InputDataCollection", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a package name to the package section.

void TAlienJDL::AddToPackages(const char *name, const char *version,
                              const char *type, const char *description)
{
   if (name) {
      TString packagename = type;
      packagename += "@";
      packagename += name;
      packagename += "::";
      packagename += version;

      AddToSet("Packages", packagename.Data());
   }

   if (description)
      AddToSetDescription("Packages", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a package.

void TAlienJDL::AddToPackages(const char *name, const char *description)
{
   AddToSet("Packages", name);
   if (description)
      AddToSetDescription("Packages", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds an output archive definition

void TAlienJDL::AddToOutputArchive(const char* value,const char* description)
{
   if (value)
      AddToSet("OutputArchive", value);
   if (description)
      AddToSetDescription("OutputArchive", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a value to a key value which hosts a set of values.
/// E.g. InputSandbox: {"file1","file2"}

void TAlienJDL::AddToReqSet(const char *key, const char *value)
{
   const char *oldValue = GetValue(key);
   TString newString;
   if (oldValue)
      newString = oldValue;
   if (newString.IsNull()) {
      newString = "(";
   } else {
      newString.Remove(newString.Length()-1);
      newString += " && ";
   }

   newString += value;
   newString += ")";

   SetValue(key, newString);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a package name to the package section.

void TAlienJDL::AddToMerge(const char *filenameToMerge, const char *jdlToSubmit,
                           const char *mergedFile, const char *description )
{
   TString mergename ( filenameToMerge );
   mergename += ":";
   mergename += jdlToSubmit;
   mergename += ":";
   mergename += mergedFile;
   AddToSet ( "Merge", mergename.Data() );
   if (description)
      AddToSetDescription("Merge", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a package name the the package section.

void TAlienJDL::AddToMerge(const char *merge, const char *description)
{
   AddToSet("Merge", merge);
   if (description)
      AddToSetDescription("Merge", description);
}

////////////////////////////////////////////////////////////////////////////////
/// Tests the submission of a simple job.

Bool_t TAlienJDL::SubmitTest()
{
   Info("SubmitTest", "submitting test job /bin/date");

   if (!gGrid) {
      Error("SubmitTest", "you must have a proper GRID environment initialized");
      return kFALSE;
   }

   Clear();
   SetExecutable("/bin/date");
   SetArguments("-R");
   TGridJob* job = gGrid->Submit(Generate());
   printf("Sending:\n%s\n",Generate().Data());
   if (job == 0) {
      Error("SubmitTest", "submitting failed");
      return kFALSE;
   }

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the specified value to the specified command.

void TAlienJDL::SetValueByCmd(TString cmd, TString value)
{
   if ( !cmd.CompareTo ( "Executable" ) ) SetExecutable ( value.Data() );
   else if ( !cmd.CompareTo ( "Arguments" ) ) SetArguments ( value.Data() );
   else if ( !cmd.CompareTo ( "Email" ) ) SetEMail ( value.Data() );
   else if ( !cmd.CompareTo ( "OutputDir" ) ) SetOutputDirectory ( value.Data() );
   else if ( !cmd.CompareTo ( "Merge" ) ) AddToMerge ( value.Data() );
   else if ( !cmd.CompareTo ( "MergeOutputDir" ) ) SetMergedOutputDirectory ( value.Data() );
   else if ( !cmd.CompareTo ( "Price" ) ) SetPrice ( value.Atoi() );
   else if ( !cmd.CompareTo ( "TTL" ) ) SetTTL ( value.Atoi() );
   else if ( !cmd.CompareTo ( "JobTag" ) ) SetJobTag ( value.Data() );
   else if ( !cmd.CompareTo ( "InputDataListFormat" ) ) SetInputDataListFormat ( value.Data() );
   else if ( !cmd.CompareTo ( "InputDataList" ) ) SetInputDataList ( value.Data() );
   else if ( !cmd.CompareTo ( "Split" ) ) SetSplitMode ( value.Data() );
   else if ( !cmd.CompareTo ( "SplitMaxInputFileNumber" ) ) SetSplitModeMaxNumOfFiles ( value.Atoi() );
   else if ( !cmd.CompareTo ( "SplitMaxInputFileSize" ) ) SetSplitModeMaxInputFileSize ( value.Atoi() );
   else if ( !cmd.CompareTo ( "SplitArguments" ) ) SetSplitArguments ( value.Data() );
   else if ( !cmd.CompareTo ( "Validationcommand" ) ) SetValidationCommand ( value.Data() );
   else if ( !cmd.CompareTo ( "MaxInitFailed" ) ) SetMaxInitFailed ( value.Atoi() );
   else if ( !cmd.CompareTo ( "InputSandbox" ) ) AddToInputSandbox ( value.Data() );
   else if ( !cmd.CompareTo ( "OutputSandbox" ) ) AddToOutputSandbox ( value.Data() );
   else if ( !cmd.CompareTo ( "InputData" ) ) AddToInputData ( value.Data() );
   else if ( !cmd.CompareTo ( "InputDataCollection" ) ) AddToInputDataCollection ( value.Data() );
   else if ( !cmd.CompareTo ( "Requirements" ) ) AddToRequirements ( value.Data() );
   else if ( !cmd.CompareTo ( "InputFile" ) ) AddToInputSandbox ( value.Data() );
   else if ( !cmd.CompareTo ( "OutputFile" ) ) AddToOutputSandbox ( value.Data() );
   else if ( !cmd.CompareTo ( "Packages" ) ) AddToPackages ( value.Data() );
   else if ( !cmd.CompareTo ( "OutputArchive" ) ) AddToOutputArchive ( value.Data() );
   else
      Error ( "SetValueByCmd()","Cmd Value not supported.." );
}

////////////////////////////////////////////////////////////////////////////////
/// fills the TAlienJDL from inputfile (should be AliEn JDL file)

void TAlienJDL::Parse(const char *filename)
{
   std::ifstream file;
   file.open ( filename );
   if ( !file.is_open() )  {
      Error("Parse", "error opening file %s", filename);
      return;
   }

   TString lineString;
   Char_t line[1024];
   while ( file.good() ) {
      file.getline ( line,1024 );
      lineString=line;
      lineString.ReplaceAll ( " ","" );
      if ( !lineString.IsNull() ) {
         if (lineString.Index('#') == 0) continue;
         TObjArray *strCmdOrValue = lineString.Tokenize ( "=" );
         TObjString*strObjCmd = ( TObjString* ) strCmdOrValue->At ( 0 );
         TObjString*strObjValue = ( TObjString* ) strCmdOrValue->At ( 1 );
         TString cmdString ( strObjCmd->GetString() );
         TString valueString ( strObjValue->GetString() );
         cmdString.ReplaceAll ( " ","" );
         valueString.ReplaceAll ( " ","" );
         valueString.ReplaceAll ( "\",\"","`" );

         TObjArray *strValues = valueString.Tokenize ( "`" );
         for ( Int_t i=0;i<strValues->GetEntries();i++ ) {
            TObjString *strObjValue2 = ( TObjString* ) strValues->At ( i );
            TString valueString2 ( strObjValue2->GetString() );
            valueString2.ReplaceAll ( "\"","" );
            valueString2.ReplaceAll ( "{","" );
            valueString2.ReplaceAll ( "}","" );
            valueString2.ReplaceAll ( ";","" );
            SetValueByCmd ( cmdString,valueString2 );
         }
      }
   }

   file.close();
}

////////////////////////////////////////////////////////////////////////////////
/// Not implemented

void TAlienJDL::Simulate()
{
}
