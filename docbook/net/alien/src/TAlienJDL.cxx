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

ClassImp(TAlienJDL)

//______________________________________________________________________________
void TAlienJDL::SetExecutable(const char *value, const char *description)
{
   // Sets the executable.

   if (value)
      SetValue("Executable", AddQuotes(value));
   if (description)
      SetDescription("Executable", description);
}

//______________________________________________________________________________
void TAlienJDL::SetArguments(const char *value, const char *description)
{
   // Sets the arguments.

   if (value)
      SetValue("Arguments", AddQuotes(value));
   if (description)
      SetDescription("Arguments", description);
}

//______________________________________________________________________________
void TAlienJDL::SetEMail(const char *value, const char *description)
{
   // Sets eMail address.

   if (value)
      SetValue("Email", AddQuotes(value));
   if (description)
      SetDescription("Email", description);
}

//______________________________________________________________________________
void TAlienJDL::SetOutputDirectory(const char *value, const char *description)
{
   // Sets OutputDirectory.

   if (value)
      SetValue ("OutputDir", AddQuotes(value));
   if (description)
      SetDescription("OutputDir", description);
}

//______________________________________________________________________________
void TAlienJDL::SetMergedOutputDirectory ( const char * value,const char* description)
{
   // Sets merged OutputDirectory.

   if (value)
      SetValue ("MergeOutputDir", AddQuotes(value));
   if (description)
      SetDescription("MergeOutputDir", description);
}

//______________________________________________________________________________
void TAlienJDL:: SetPrice(UInt_t price,const char* description)
{
   // Sets OutputDirectory.

   TString pricestring(Form("%d",price));
   SetValue("Price", AddQuotes(pricestring.Data()));

   if (description)
      SetDescription("Price", description);
}

//______________________________________________________________________________
void TAlienJDL:: SetTTL(UInt_t ttl, const char *description)
{
   // To inform AliEn master about estimated Time-To-Live of included nodes.

   TString ttlstring;
   ttlstring+= ttl;
   SetValue("TTL", ttlstring.Data());

   if (description)
      SetDescription("TTL", description);
}

//______________________________________________________________________________
void TAlienJDL::SetJobTag(const char* value,const char* description)
{
   // Sets Job Tag

   if (value)
      SetValue("JobTag", AddQuotes(value));
   if (description)
      SetDescription("JobTag", description);
}

//______________________________________________________________________________
void TAlienJDL::SetInputDataListFormat(const char* value,const char* description)
{
   // Sets InputDataListFormat - can be "xml-single" or "xml-multi"

   if (value)
      SetValue("InputDataListFormat", AddQuotes(value));
   if (description)
      SetDescription("InputDataListFormat", description);
}

//______________________________________________________________________________
void TAlienJDL::SetInputDataList(const char* value,const char* description)
{
   // Sets InputDataList name

   if (value)
      SetValue("InputDataList", AddQuotes(value));
   if (description)
      SetDescription("InputDataList", description);
}


//______________________________________________________________________________
void TAlienJDL::SetSplitMode(const char *value, UInt_t maxnumberofinputfiles,
                             UInt_t maxinputfilesize, const char *d1, const char *d2,
                             const char *d3)
{
   // Sets the split mode.

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

//______________________________________________________________________________
void TAlienJDL::SetSplitModeMaxNumOfFiles(UInt_t maxnumberofinputfiles, const char *description)
{
   // Sets the SplitMaxNumOfFiles.

   TString val;
   val += maxnumberofinputfiles;
   SetValue ( "SplitMaxInputFileNumber", AddQuotes ( val.Data() ) );

   if (description)
      SetDescription("SplitMaxInputFileNumber", description);
}

//______________________________________________________________________________
void TAlienJDL::SetSplitModeMaxInputFileSize(UInt_t maxinputfilesize, const char *description)
{
   // Sets the SplitMaxInputFileSize.

   TString val;
   val += maxinputfilesize;
   SetValue ( "SplitMaxInputFileSize", AddQuotes ( val.Data() ) );
   if (description)
      SetDescription("SplitMaxInputFileSize", description);
}

//______________________________________________________________________________
void TAlienJDL::SetSplitArguments(const char *splitarguments, const char *description)
{
   // Sets the split.

   if (splitarguments)
      SetValue("Split", AddQuotes(splitarguments));
   if (description)
      SetDescription("Split", description);
}

//______________________________________________________________________________
void TAlienJDL::SetValidationCommand(const char *value, const char *description)
{
   // Sets the validation command.

   SetValue("Validationcommand", AddQuotes(value));
   if (description)
      SetDescription("Validationcommand", description);
}

//______________________________________________________________________________
void TAlienJDL::SetMaxInitFailed(Int_t maxInitFailed, const char *description)
{
   // Sets the Maxium init failed
   TString str;
   str += maxInitFailed;
   SetValue("MaxInitFailed", AddQuotes(str.Data()));
   if (description)
      SetDescription("MaxInitFailed", description);
}

//______________________________________________________________________________
void TAlienJDL::SetOwnCommand(const char *command, const char *value, const char *description)
{
   // Sets the Own Command
   if ((command) && (value))
     SetValue(command, AddQuotes(value));
   if ((command) && (description))
     SetDescription(command, description);
}

//______________________________________________________________________________
void TAlienJDL::AddToRequirements(const char *value, const char *description)
{
   // Adds a requirement.

   if (value)
      AddToReqSet("Requirements", value);
   if (description)
      AddToSetDescription("Requirements", description);
}

//______________________________________________________________________________
void TAlienJDL::AddToInputSandbox(const char *value, const char *description)
{
   // Adds a file to the input sandbox.

   if (value)
      AddToSet("InputFile", value);
   if (description)
      AddToSetDescription("InputFile", description);
}

//______________________________________________________________________________
void TAlienJDL::AddToOutputSandbox(const char *value, const char *description)
{
   // Adds a file to the output sandbox.

   if (value)
      AddToSet("OutputFile", value);
   if (description)
      AddToSetDescription("OutputFile", description);
}

//______________________________________________________________________________
void TAlienJDL::AddToInputData(const char *value, const char *description)
{
   // Adds a file to the input data.

   if (value)
      AddToSet("InputData", value);
   if (description)
      AddToSetDescription("InputData", description);
}

//______________________________________________________________________________
void TAlienJDL::AddToInputDataCollection(const char *value, const char *description)
{
   // Adds a file to the input data collection.

   if (value)
      AddToSet("InputDataCollection", value);
   if (description)
      AddToSetDescription("InputDataCollection", description);
}

//______________________________________________________________________________
void TAlienJDL::AddToPackages(const char *name, const char *version,
                              const char *type, const char *description)
{
   // Adds a package name to the package section.

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

//______________________________________________________________________________
void TAlienJDL::AddToPackages(const char *name, const char *description)
{
   // Adds a package.

   AddToSet("Packages", name);
   if (description)
      AddToSetDescription("Packages", description);
}

//______________________________________________________________________________
void TAlienJDL::AddToOutputArchive(const char* value,const char* description)
{
   // Adds an output archive definition

   if (value)
      AddToSet("OutputArchive", value);
   if (description)
      AddToSetDescription("OutputArchive", description);
}

//______________________________________________________________________________
void TAlienJDL::AddToReqSet(const char *key, const char *value)
{
   // Adds a value to a key value which hosts a set of values.
   // E.g. InputSandbox: {"file1","file2"}

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

//______________________________________________________________________________
void TAlienJDL::AddToMerge(const char *filenameToMerge, const char *jdlToSubmit,
                           const char *mergedFile, const char *description )
{
   // Adds a package name to the package section.

   TString mergename ( filenameToMerge );
   mergename += ":";
   mergename += jdlToSubmit;
   mergename += ":";
   mergename += mergedFile;
   AddToSet ( "Merge", mergename.Data() );
   if (description)
      AddToSetDescription("Merge", description);
}

//______________________________________________________________________________
void TAlienJDL::AddToMerge(const char *merge, const char *description)
{
   // Adds a package name the the package section.

   AddToSet("Merge", merge);
   if (description)
      AddToSetDescription("Merge", description);
}

//______________________________________________________________________________
Bool_t TAlienJDL::SubmitTest()
{
   // Tests the submission of a simple job.

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

//______________________________________________________________________________
void TAlienJDL::SetValueByCmd(TString cmd, TString value)
{
   // Set the specified value to the specified command.

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

//______________________________________________________________________________
void TAlienJDL::Parse(const char *filename)
{
   // fills the TAlienJDL from inputfile (should be AliEn JDL file)

   ifstream file;
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

//______________________________________________________________________________
void TAlienJDL::Simulate()
{
   // Not implemented
}
