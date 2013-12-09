// @(#)root/netx:$Id$
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TNetXNGFileStager                                                          //
//                                                                            //
// Authors: Justin Salmon, Lukasz Janyst                                      //
//          CERN, 2013                                                        //
//                                                                            //
// Enables access to XRootD staging capabilities using the new client.        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "TNetXNGFileStager.h"
#include "TNetXNGSystem.h"
#include "THashList.h"
#include "TFileInfo.h"
#include "TFileCollection.h"
#include <XrdCl/XrdClFileSystem.hh>

ClassImp( TNetXNGFileStager);

//______________________________________________________________________________
TNetXNGFileStager::TNetXNGFileStager(const char *url) :
      TFileStager("xrd")
{
   // Constructor
   //
   // param url: the URL of the entry-point server

   fSystem = new TNetXNGSystem(url);
}

//______________________________________________________________________________
TNetXNGFileStager::~TNetXNGFileStager()
{
   // Destructor

   delete fSystem;
}

//______________________________________________________________________________
Bool_t TNetXNGFileStager::IsStaged(const char *path)
{
   // Check if a file is staged
   //
   // param path: the URL of the file

   FileStat_t st;
   if (fSystem->GetPathInfo(path, st) != 0) {
      if (gDebug > 0)
         Info("IsStaged", "path %s cannot be stat'ed", path);
      return kFALSE;
   }

   if (R_ISOFF(st.fMode)) {
      if (gDebug > 0)
         Info("IsStaged", "path '%s' is offline", path);
      return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
Int_t TNetXNGFileStager::Locate(const char *path, TString &url)
{
   // Get actual endpoint URL
   //
   // param path:    the entry-point URL
   // param endpath: the actual endpoint URL
   // returns:       0 in the case of success and 1 if any error occurred

   return fSystem->Locate(path, url);
}

//______________________________________________________________________________
Int_t TNetXNGFileStager::LocateCollection(TFileCollection *fc,
                                          Bool_t addDummyUrl)
{
   // Bulk locate request for a collection of files
   //
   // param fc:          collection of files to be located
   // param addDummyUrl: append a dummy noop URL if the file is not staged or
   //                    redirector == endpoint
   // returns:           < 0 in case of errors, number of files processed
   //                    otherwise

   if (!fc) {
      Error("LocateCollection", "No input collection given");
      return -1;
   }

   int numFiles = 0;
   TFileInfo *info;
   TIter it(fc->GetList());
   TString startUrl, endUrl;

   while ((info = dynamic_cast<TFileInfo *>(it.Next())) != NULL) {
      startUrl = info->GetCurrentUrl()->GetUrl();

      // File not staged
      if (fSystem->Locate(startUrl.Data(), endUrl)) {
         info->ResetBit(TFileInfo::kStaged);

         if (addDummyUrl)
            info->AddUrl("noop://none", kTRUE);

         if (gDebug > 1)
            Info("LocateCollection", "Not found: %s", startUrl.Data());
      }

      // File staged
      else {
         info->SetBit(TFileInfo::kStaged);

         if (startUrl != endUrl) {
            info->AddUrl(endUrl.Data(), kTRUE);
         } else if (addDummyUrl) {
            // Returned URL identical to redirector URL
            info->AddUrl("noop://redir", kTRUE);
         }

         if (gDebug > 1)
            Info("LocateCollection", "Found: %s --> %s", startUrl.Data(),
                                      endUrl.Data());
      }
      numFiles++;
   }

   return numFiles;
}

//______________________________________________________________________________
Bool_t TNetXNGFileStager::Matches(const char *s)
{
   // Returns kTRUE if stager 's' is compatible with current stager. Avoids
   // multiple instantiations of the potentially the same TNetXNGFileStager.

   return ((s && (fName == s)) ? kTRUE : kFALSE);
}

//______________________________________________________________________________
Bool_t TNetXNGFileStager::Stage(const char *path, Option_t *opt)
{
   // Issue a stage request for a single file
   //
   // param path: the path of the file to stage
   // param opt:  defines 'option' and 'priority' for 'Prepare': the format is
   //             opt = "option=o priority=p"

   Int_t priority = ParseStagePriority(opt);
   return fSystem->Stage(path, priority);
}

//______________________________________________________________________________
Bool_t TNetXNGFileStager::Stage(TCollection *paths, Option_t *opt)
{
   // Issue stage requests for multiple files
   //
   // param pathlist: list of paths of files to stage
   // param opt:      defines 'option' and 'priority' for 'Prepare': the
   //                 format is opt = "option=o priority=p"

   Int_t priority = ParseStagePriority(opt);
   return fSystem->Stage(paths, priority);
}

//______________________________________________________________________________
UChar_t TNetXNGFileStager::ParseStagePriority(Option_t *opt)
{
   // Get a staging priority value from an option string

   UChar_t priority = 0;
   Ssiz_t from = 0;
   TString token;

   while (TString(opt).Tokenize(token, from, "[ ,|]")) {
      if (token.Contains("priority=")) {
         token.ReplaceAll("priority=", "");
         if (token.IsDigit()) {
            priority = token.Atoi();
         }
      }
   }

   return priority;
}
