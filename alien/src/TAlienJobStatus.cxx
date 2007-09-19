// @(#)root/alien:$Id$
// Author: Jan Fiete Grosse-Oetringhaus   06/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienJobStatus                                                      //
//                                                                      //
// Alien implementation of TGridJobStatus                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGridJobStatus.h"
#include "TAlienJobStatus.h"
#include "TObjString.h"
#include "TBrowser.h"
#include "TNamed.h"
#include "TAlienDirectory.h"

ClassImp(TAlienJobStatus)

//______________________________________________________________________________
TAlienJobStatus::TAlienJobStatus(TMap *status)
{
   // Creates a TAlienJobStatus object.
   // If a status map is provided it is copied to the status information.

   TObjString* key;
   TObjString* val;

   if (status) {
      TMapIter next(status);
      while ( (key = (TObjString*)next())) {
         val = (TObjString*)status->GetValue(key->GetName());
         fStatus.Add(key->Clone(), val->Clone());
      }
   }
}

//______________________________________________________________________________
TAlienJobStatus::~TAlienJobStatus()
{
   // Cleanup.

   fStatus.DeleteAll();
}

//______________________________________________________________________________
void TAlienJobStatus::Browse(TBrowser* b)
{
   // Browser interface to ob status.

   if (b) {
      TIterator *iter = fStatus.MakeIterator();
      TObject *obj = 0;
      while ((obj = iter->Next()) != 0) {
         TObject* value = fStatus.GetValue(obj);

         TObjString* keyStr = dynamic_cast<TObjString*>(obj);
         TObjString* valueStr = dynamic_cast<TObjString*>(value);

         if (keyStr->GetString() == TString("jdl")) {
            TString valueParsed(valueStr->GetString());
            valueParsed.ReplaceAll("\n", 1);
            valueParsed.ReplaceAll("  ", 2);
            b->Add(new TPair(new TObjString("jdl"), new TObjString(valueParsed)));

            // list sandboxes
            const char* outputdir = GetJdlKey("OutputDir");

            TString sandbox;
            if (outputdir) {
               sandbox = outputdir;
            } else {
               sandbox = TString("/proc/") + TString(GetKey("user")) + TString("/") + TString(GetKey("queueId")) + TString("/job-output");
            }

            b->Add(new TAlienDirectory(sandbox.Data(),"job-output"));

         } else {
            if (keyStr && valueStr)
               b->Add(new TNamed(valueStr->GetString(), keyStr->GetString()));
         }
      }
      delete iter;
   }
}

//______________________________________________________________________________
const char *TAlienJobStatus::GetJdlKey(const char* key)
{
   // Return the JDL key.

   const char *jdl = GetKey("jdl");
   if (!jdl)
      return 0;
   const char* jdltagbegin = strstr(jdl,key);
   const char* jdltagquote = strchr(jdltagbegin,'"');
   const char* jdltagend   = strchr(jdltagbegin,';');

   if (!jdltagend) {
      return 0;
   }
   if (!jdltagquote) {
      return 0;
   }
   jdltagquote++;
   const char* jdltagquote2 = strchr(jdltagquote,'"');
   if (!jdltagquote2) {
      return 0;
   }
   fJdlTag = TString(jdltagquote);
   fJdlTag = fJdlTag(0,jdltagquote2-jdltagquote);

   return fJdlTag.Data();
}

//______________________________________________________________________________
const char *TAlienJobStatus::GetKey(const char* key)
{
   // Return a key.

   TObject* obj = fStatus.FindObject(key);
   TPair* pair = dynamic_cast<TPair*>(obj);
   if (pair) {
      TObjString* string = dynamic_cast<TObjString*> (pair->Value());
      return string->GetName();
   }
   return 0;
}

//______________________________________________________________________________
TGridJobStatus::EGridJobStatus TAlienJobStatus::GetStatus() const
{
   // Gets the status of the job reduced to the subset defined
   // in TGridJobStatus.

   TObject* obj = fStatus.FindObject("status");
   TPair* pair = dynamic_cast<TPair*>(obj);

   if (pair) {
      TObjString* string = dynamic_cast<TObjString*> (pair->Value());

      if (string) {
         const char* status = string->GetString().Data();

         if (strcmp(status, "INSERTING") == 0 ||
             strcmp(status, "WAITING") == 0 ||
             strcmp(status, "QUEUED") == 0 ||
             strcmp(status, "ASSIGNED") == 0)
            return kWAITING;
         else if (strcmp(status, "STARTED") == 0 ||
                  strcmp(status, "SAVING") == 0 ||
                  strcmp(status, "SPLITTING") == 0 ||
                  strcmp(status, "RUNNING") == 0 ||
                  strcmp(status, "SPLIT") == 0)
            return kRUNNING;
         else if (strcmp(status, "EXPIRED") == 0 ||
                  string->GetString().BeginsWith("ERROR_") == kTRUE ||
                  strcmp(status, "FAILED") == 0 ||
                  strcmp(status, "ZOMBIE") == 0)
            return kFAIL;
         else if (strcmp(status, "KILLED") == 0)
            return kABORTED;
         else if (strcmp(status, "DONE") == 0)
            return kDONE;
      }
   }
   return kUNKNOWN;
}

//______________________________________________________________________________
void TAlienJobStatus::Print(Option_t *) const
{
   // Prints the job information.

   PrintJob(kTRUE);
}

//______________________________________________________________________________
void TAlienJobStatus::PrintJob(Bool_t full) const
{
   // Prints this job.
   // If full is kFALSE only the status is printed, otherwise all information.

   TObject* obj = fStatus.FindObject("status");
   TPair* pair = dynamic_cast<TPair*>(obj);

   if (pair) {
      TObjString* string = dynamic_cast<TObjString*> (pair->Value());
      if (string) {
         printf("The status of the job is %s\n", string->GetString().Data());
      }
   }

   if (full != kTRUE)
      return;

   printf("==================================================\n");
   printf("Detail Information:\n");

   TIterator* iter = fStatus.MakeIterator();

   while ((obj = iter->Next()) != 0) {
      TObject* value = fStatus.GetValue(obj);

      TObjString* keyStr = dynamic_cast<TObjString*>(obj);
      TObjString* valueStr = dynamic_cast<TObjString*>(value);

      printf("%s => %s\n", (keyStr) ? keyStr->GetString().Data() : "", (valueStr) ? valueStr->GetString().Data() : "");
   }

   delete iter;
}
