// @(#)root/alien:$Name:$:$Id:$
// Author: Fons Rademakers   13/5/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlien                                                               //
//                                                                      //
// Class defining interface to AliEn GRID services.                     //
//                                                                      //
// To open a connection to a AliEn GRID use the static method           //
// TGrid::Connect("alien://<host>", ..., ...).                          //
//                                                                      //
// Related classes are TAlienResult and TAlienAttrResult.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TAlien.h"
#include "TAlienResult.h"
#include "TAlienAttrResult.h"
#include "TUrl.h"
#include "TBrowser.h"
#include <stdlib.h>


ClassImp(TAlien)

//______________________________________________________________________________
TAlien::TAlien(const char *grid, const char *uid, const char *pw)
{
   // Open a connection to the AliEn GRID. The grid argument should be
   // of the form "alien[://<host>][:<port>], e.g.: "alien" or
   // "alien://alice.cern.ch". The uid is the username and the pw the
   // password that should be used for the connection. In general the
   // uid and password will be taken from the AliEn client setup and
   // don't need to be specified.

   fAlien = 0;     // should be set as return code from AlienConnect()

   TUrl url(grid);

   if (!url.IsValid()) {
      Error("TAlien", "malformed grid argument %s", grid);
      MakeZombie();
      return;
   }

   TString host;

   if (!strncmp(url.GetProtocol(), "http", 4)) {
      if (strncmp(url.GetHost(), "alien", 5)) {
         Error("TAlien", "protocol in grid argument should be alien it is %s",
               url.GetHost());
         MakeZombie();
         return;
      }
   } else {
      if (strncmp(url.GetProtocol(), "alien", 5)) {
         Error("TAlien", "protocol in grid argument should be alien it is %s",
               url.GetProtocol());
         MakeZombie();
         return;
      }
      host = url.GetHost();
   }

   if (AlienConnect(uid, pw) == -1) {
      if (host.IsNull())
         Error("TAlien", "connection to AliEn failed");
      else
         Error("TAlien", "connection to AliEn at %s failed", host.Data());
      MakeZombie();
      return;
   }

   fPort = 0;
}

//______________________________________________________________________________
TAlien::~TAlien()
{
   // Clenaup AliEn object, closes connection to AliEn.

   Close();
}

//______________________________________________________________________________
void TAlien::Close(Option_t *)
{
   // Close connection to AliEn.

   if (AlienClose() == -1)
      Error("Close", "error closing connection to AliEn");

   fPort = -1;
}

//______________________________________________________________________________
TGridResult *TAlien::Query(const char *wildcard)
{
   // Query the AliEn file catalog to find the set of logical file name
   // matching the specified wildcard. For AliEn the wildcard must have
   // the form:
   // "lfn://alien.cern.ch/alice/bin/date"
   // "lfn:///alice/simulation/2001-04/V0.6*.root"
   // "lfn:///alice/simulation/2001-04*?tagIndex=5&value=\"s*\""
   // Returns 0 in case of error. Returned result must be deleted by user.

   AlienResult_t *ar = AlienGetFile(wildcard);

   if (!ar) return 0;

   return new TAlienResult(ar);
}

//______________________________________________________________________________
Int_t TAlien::AddFile(const char *lfn, const char *pfn)
{
   // Add physical filename to AliEn catalog with associated logical file name.
   // Returns -1 on error (e.g. when lfn or pfn already exists), 0 otherwise.
   // Example lfn: "lfn://[alien.cern.ch]/alice/cern.ch/user/r/rdm/aap.root"
   // Example pfn: "rfio:/castor/cern.ch/user/r/rdm/noot.root"

   if (AlienAddFile(lfn, pfn) == -1) {
      Error("AddFile", "error adding pfn %s with lfn %s", pfn, lfn);
      return -1;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TAlien::DeleteFile(const char *lfn)
{
   // Delete logical file from AliEn. Does not delete associated pfn's.
   // Returns -1 on error, 0 otherwise.

   if (AlienDeleteFile(lfn) == -1) {
      Error("DeleteFile", "error deleting lfn %s", lfn);
      return -1;
   }
   return 0;
}

//______________________________________________________________________________
char *TAlien::GetPhysicalFileName(const char *lfn)
{
   // Returns physical file name associated with logical file name.
   // Returns 0 in case of error. Returned value must be deleted
   // using delete[].

   char *pfn = (char *) AlienGetPhysicalFileName(lfn);

   if (!pfn) return 0;

   char *pfn2 = new char [strlen(pfn) + 1];
   strcpy(pfn2, pfn);

   free(pfn);

   return pfn2;
}

//______________________________________________________________________________
TGridResult *TAlien::GetPhysicalFileNames(const char *lfn)
{
   // Returns list of physical file names associated with logical file name.
   // Returns 0 in case of error. Returned result must be deleted by user.

   AlienResult_t *ar = AlienGetPhysicalFileNames(lfn);

   if (!ar) return 0;

   return new TAlienResult(ar);
}

//______________________________________________________________________________
Int_t TAlien::AddAttribute(const char *lfn, const char *attrname,
                           const char *attrval)
{
   // Add attribute attrname with value attrval to specified logical
   // file name. Returns -1 on error, 0 otherwise.

   // assume "standard" tag exists
   if (AlienAddAttribute(lfn, "standard", attrname, attrval) == -1) {
      Error("AddAttribute", "error adding attribute %s with value %s to lfn %s",
            attrname, attrval, lfn);
      return -1;
   }

   return 0;
}

//______________________________________________________________________________
Int_t TAlien::DeleteAttribute(const char *lfn, const char *attrname)
{
   // Delete specified attribute from logical file name. Returns -1 on error,
   // 0 otherwise.

   if (AlienDeleteAttribute(lfn, attrname) == -1) {
      Error("DeleteAttribute", "error deleting attribute %s from lfn %s",
            attrname, lfn);
      return -1;
   }

   return 0;
}

//______________________________________________________________________________
TGridResult *TAlien::GetAttributes(const char *lfn)
{
   // Return attributes associated with lfn. Returns 0 in case of error.
   // Returned result must be deleted by user.

   // assume "standard" tag
   AlienAttr_t *at = AlienGetAttributes(lfn, "standard");

   if (!at) return 0;

   return new TAlienAttrResult(at);
}

//______________________________________________________________________________
const char *TAlien::Pwd()
{
   // Returns current working directory in the AliEn file catalog.

   return AlienPwd();
}

//______________________________________________________________________________
Int_t TAlien::Cd(const char *dir)
{
   // Change directory in the AliEn file catalog. If dir is not specified,
   // it goes to the home directory. Returns -1 in case of failure,
   // 0 otherwise.

   if (AlienCd(dir) == -1) {
      Error("Cd", "error making %s the current working directory", dir);
      return -1;
   }

   return 0;
}

//______________________________________________________________________________
TGridResult *TAlien::Ls(const char *dir)
{
   // Returns contents of a directory in the AliEn file catalog.
   // Returns 0 in case of error. Returned result must be deleted by user.

   AlienResult_t *ar = AlienLs(dir);

   if (!ar) return 0;

   return new TAlienResult(ar);
}

//______________________________________________________________________________
void TAlien::Browse(TBrowser *b)
{
   // Browse AliEn file catalog in ROOT browser.

   if (!b) return;

   TGridResult *res = Ls(Pwd());

   const char *name;
   while (res && (name = res->Next())) {
      // make TGridFile object of TGridDirectory object, like for file browsing
      // and add to browser
   }
   delete res;
}

//______________________________________________________________________________
const char *TAlien::GetInfo()
{
   // Returns AliEn version string.

   return AlienGetInfo();
}
