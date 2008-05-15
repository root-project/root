// @(#)root/alien:$Id$
// Author: Jan Fiete Grosse-Oetringhaus   28/9/2004
//         Lucia.Jancurova@cern.ch  2007
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
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
void TAlienJDL::SetExecutable(const char* value)
{
   // Sets the executable.

   if (value)
      SetValue("Executable", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL::SetArguments(const char* value)
{
   // Sets the arguments.

   if (value)
      SetValue("Arguments", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL::SetEMail(const char* value)
{
   // Sets eMail address.

   if (value)
      SetValue("EMail", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL::SetOutputDirectory(const char* value)
{
   // Sets OutputDirectory.

   if ( value )
      SetValue ("OutputDirectory", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL::SetMergedOutputDirectory ( const char * value )
{
   // Sets merged OutputDirectory.

   if ( value )
      SetValue ("MergedOutputDirectory", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL:: SetPrice(UInt_t price)
{
   // Sets OutputDirectory.

   TString pricestring(Form("%d",price));

   SetValue("Price", pricestring.Data());
}

//______________________________________________________________________________
void TAlienJDL:: SetTTL(UInt_t ttl)
{
   // To inform AliEn master about estimated Time-To-Live of included nodes.

   TString ttlstring;
   ttlstring+= ttl;

   SetValue("TTL", ttlstring.Data());
}

//______________________________________________________________________________
void TAlienJDL::SetJobTag(const char* value)
{
   // Sets Job Tag

   if (value)
      SetValue("JobTag", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL::SetInputDataListFormat(const char* value)
{
   // Sets InputDataListFormat - can be "xml-single" or "xml-multi"

   if (value)
      SetValue("InputDataListFormat", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL::SetInputDataList(const char* value)
{
   // Sets InputDataList name

   if (value)
      SetValue("InputDataList", AddQuotes(value));
}


//______________________________________________________________________________
void TAlienJDL::SetSplitMode(const char* value, UInt_t maxnumberofinputfiles, UInt_t maxinputfilesize)
{
   // Sets the split mode.

   if (value && !strcasecmp(value, "SE")) {

      SetValue("Split", AddQuotes(value));
      if (maxnumberofinputfiles) {
         TString val;
         val += maxnumberofinputfiles;
         SetValue("SplitMaxInputFileNumber", AddQuotes(val.Data()));
      }
      if (maxinputfilesize) {
         TString val;
         val += maxinputfilesize;
         SetValue("SplitMaxInputFileSize", AddQuotes(val.Data()));
      }
   } else {
      if (value)
         SetValue("Split",AddQuotes(value));
   }
}

//______________________________________________________________________________
void TAlienJDL::SetSplitModeMaxNumOfFiles ( UInt_t maxnumberofinputfiles )
{
   // Sets the SplitMaxNumOfFiles.
   TString val;
   val += maxnumberofinputfiles;
   SetValue ( "SplitMaxInputFileNumber", AddQuotes ( val.Data() ) );
}

//______________________________________________________________________________
void TAlienJDL::SetSplitModeMaxInputFileSize ( UInt_t maxinputfilesize )
{
   // Sets the SplitMaxInputFileSize.

   TString val;
   val += maxinputfilesize;
   SetValue ( "SplitMaxInputFileSize", AddQuotes ( val.Data() ) );

}

//______________________________________________________________________________
void TAlienJDL::SetSplitArguments(const char* splitarguments)
{
   // Sets the split.

   if (splitarguments)
      SetValue("SplitArguments", AddQuotes(splitarguments));
}

//______________________________________________________________________________
void TAlienJDL::SetValidationCommand(const char* value)
{
   // Sets the validation command.

   SetValue("ValidationCommand", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL::AddToRequirements(const char* value)
{
   // Adds a requirement.

   if (value)
      AddToReqSet("Requirements", value);
}

//______________________________________________________________________________
void TAlienJDL::AddToInputSandbox(const char* value)
{
   // Adds a file to the input sandbox.

   if (value)
      AddToSet("InputFile", value);
}

//______________________________________________________________________________
void TAlienJDL::AddToOutputSandbox(const char* value)
{
   // Adds a file to the output sandbox.

   if (value)
      AddToSet("OutputFile", value);
}

//______________________________________________________________________________
void TAlienJDL::AddToInputData(const char* value)
{
   // Adds a file to the input data.

   if (value)
      AddToSet("InputData", value);
}

//______________________________________________________________________________
void TAlienJDL::AddToInputDataCollection(const char* value)
{
   // Adds a file to the input data collection.

   if (value)
      AddToSet("InputDataCollection", value);
}

//______________________________________________________________________________
void TAlienJDL::AddToPackages(const char* name,const char* version, const char* type)
{
   // Adds a package name to the package section

   if (name) {
      TString packagename = type;
      packagename += "@";
      packagename += name;
      packagename += "::";
      packagename += version;

      AddToSet("Packages", packagename.Data());
   }
}

//______________________________________________________________________________
void TAlienJDL::AddToPackages ( const char * name )
{
   // Adds a package.

   AddToSet("Packages", name);
}

//______________________________________________________________________________
void TAlienJDL::AddToOutputArchive(const char* value)
{
   // Adds an output archive definition

   if (value)
      AddToSet("OutputArchive", value);
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
void TAlienJDL::AddToMerge(const char* filenameToMerge,const char* jdlToSubmit,
                           const char* mergedFile )
{
   // Adds a package name to the package section.

   TString mergename ( filenameToMerge );
   mergename += ":";
   mergename += jdlToSubmit;
   mergename += ":";
   mergename += mergedFile;
   AddToSet ( "Merge", mergename.Data() );
}

//______________________________________________________________________________
void TAlienJDL::AddToMerge(const char *merge)
{
   // Adds a package name the the package section.

   AddToSet("Merge", merge);
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
   else if ( !cmd.CompareTo ( "EMail" ) ) SetEMail ( value.Data() );
   else if ( !cmd.CompareTo ( "OutputDirectory" ) ) SetOutputDirectory ( value.Data() );
   else if ( !cmd.CompareTo ( "Merge" ) ) AddToMerge ( value.Data() );
   else if ( !cmd.CompareTo ( "MergedOutputDirectory" ) ) SetMergedOutputDirectory ( value.Data() );
   else if ( !cmd.CompareTo ( "Price" ) ) SetPrice ( value.Atoi() );
   else if ( !cmd.CompareTo ( "TTL" ) ) SetTTL ( value.Atoi() );
   else if ( !cmd.CompareTo ( "JobTag" ) ) SetJobTag ( value.Data() );
   else if ( !cmd.CompareTo ( "InputDataListFormat" ) ) SetInputDataListFormat ( value.Data() );
   else if ( !cmd.CompareTo ( "InputDataList" ) ) SetInputDataList ( value.Data() );
   else if ( !cmd.CompareTo ( "Split" ) ) SetSplitMode ( value.Data() );
   else if ( !cmd.CompareTo ( "SplitMaxInputFileNumber" ) ) SetSplitModeMaxNumOfFiles ( value.Atoi() );
   else if ( !cmd.CompareTo ( "SplitMaxInputFileSize" ) ) SetSplitModeMaxInputFileSize ( value.Atoi() );
   else if ( !cmd.CompareTo ( "SplitArguments" ) ) SetSplitArguments ( value.Data() );
   else if ( !cmd.CompareTo ( "ValidationCommand" ) ) SetValidationCommand ( value.Data() );
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
void TAlienJDL::Parse(const char * filename)
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
     if ( !lineString.IsNull() ) {
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
