// @(#)root/net:$Name:  $:$Id:$
// Author: A. Peters, G. Ganis   7/2/2007

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileStager                                                          //
//                                                                      //
// Abstract base class defining an interface to a stager.               //
//                                                                      //
// To open a connection to a stager use the static method               //
// Open("<stager>"), where <stager> contains a keyword allowing to load //
// the relevant plug-in, e.g.                                           //
//           TFileStager::Open("root://lxb6064.cern.ch")                //
// will load TXNetFileStager and initialize it for the redirector at    //
// lxb6046.cern.ch .                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TError.h"
#include "TList.h"
#include "TFileStager.h"
#include "TObjString.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TUrl.h"

//_____________________________________________________________________________
TList* TFileStager::GetStaged(TList *pathlist)
{
   // Retrieves the staging (online) status for a list of path names. Path names
   // are TUrl objects or TObjString. The returned list is the list of staged
   // files as TObjString (we use TObjString, because you can do a FindObject
   // on that list using the file name, which is not possible with TUrl objects.

   if (!pathlist) {
      Error("GetStaged", "list of pathnames was not specified!");
      return 0;
   }

   TList* stagedlist = new TList();
   stagedlist->SetOwner(kTRUE);

   TIter nxt(pathlist);
   TObject* obj = 0;
   while ((obj = nxt()))  {
      const char* pathname = 0;
      if (TString(obj->ClassName()) == "TUrl") {
         pathname = ((TUrl*)obj)->GetUrl();
      }
      if (TString(obj->ClassName()) == "TObjString") {
         pathname = ((TObjString*)obj)->GetName();
      }
      if (!pathname) {
         Warning("GetStaged", "object is of type %s : expecting TUrl or TObjString - ignoring",
                              obj->ClassName());
         continue;
      }

      if (IsStaged(pathname)) {
         stagedlist->Add(new TObjString(pathname));
      }
   }

   // List of online files
   return stagedlist;
}

//_____________________________________________________________________________
Bool_t TFileStager::Stage(TList *pathlist, Option_t *opt)
{
   // Issue a stage request for a list of files.
   // Return the '&' of all single Prepare commands.

   TIter nxt(pathlist);
   TObject* obj;
   Bool_t success = kFALSE;

   while ((obj = nxt()))  {
      const char* pathname = 0;
      if (TString(obj->ClassName()) == "TUrl") {
         pathname = ((TUrl*)obj)->GetUrl();
      }
      if (TString(obj->ClassName()) == "TObjString") {
         pathname = ((TObjString*)obj)->GetName();
      }

      if (!pathname) {
         Warning("Stage", "found object of type %s - expecting TUrl/TObjstring - ignoring",
                              obj->ClassName());
         continue;
      }

      // Issue to prepare
      success &= Stage(pathname, opt);
   }

   // return global flag
   return success;
}

//______________________________________________________________________________
TFileStager *TFileStager::Open(const char *stager)
{
   // Open a stager, after having loaded the relevant plug-in.
   // The format of 'stager' depends on the plug-in.

   TPluginHandler *h;
   TFileStager *s = 0;

   if (!stager) {
      ::Error("TFileStager::Open", "stager name missing: do nothing");
      return 0;
   }

   if ((h = gROOT->GetPluginManager()->FindHandler("TFileStager", stager))) {
      if (h->LoadPlugin() == -1)
         return 0;
      s = (TFileStager *) h->ExecPlugin(1, stager);
   }

   return s;
}
