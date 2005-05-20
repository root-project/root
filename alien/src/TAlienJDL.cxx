// @(#)root/alien:$Name:  $:$Id: TAlienJDL.cxx,v 1.6 2004/11/04 14:56:00 dfeich Exp $
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

ClassImp(TAlienJDL)

//______________________________________________________________________________
void TAlienJDL::SetExecutable(const char* value)
{
   // Sets the executable.

   SetValue("Executable", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL::SetArguments(const char* value)
{
   // Sets the arguments.

   SetValue("Arguments", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL::SetEMail(const char* value)
{
   // Sets eMail address.

   SetValue("EMail", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL::SetSplitMode(const char* value)
{
   // Sets the split mode.

   SetValue("Split", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL::SetValidationCommand(const char* value)
{
   // Sets the validation command.

   SetValue("ValidationCommand", AddQuotes(value));
}

//______________________________________________________________________________
void TAlienJDL::SetRequirements(const char* value)
{
   // Sets the requirements.

   SetValue("Requirements", value);
}

//______________________________________________________________________________
void TAlienJDL::AddToInputSandbox(const char* value)
{
   // Adds a file to the input sandbox.

   AddToSet("InputFile", value);
}

//______________________________________________________________________________
void TAlienJDL::AddToOutputSandbox(const char* value)
{
   // Adds a file to the output sandbox.

   AddToSet("OutputFile", value);
}

//______________________________________________________________________________
void TAlienJDL::AddToInputData(const char* value)
{
   // Adds a file to the input data.

   AddToSet("InputData", value);
}

//______________________________________________________________________________
void TAlienJDL::AddToInputDataCollection(const char* value)
{
   // Adds a file to the input data collection.

   AddToSet("InputDataCollection", value);
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

   if (job == 0) {
      Error("SubmitTest", "submitting failed");
      return kFALSE;
   }

   return kTRUE;
}
