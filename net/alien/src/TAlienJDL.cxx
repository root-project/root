// @(#)root/alien:$Id$
// Author: Jan Fiete Grosse-Oetringhaus   28/9/2004

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
   // Sets OutputDirectory

   if (value)
      SetValue("OutputDir", AddQuotes(value));
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
   //to inform AliEn master about estimated Time-To-Live of included nodes
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
