// @(#)root/alien:$Name:  $:$Id: TAlien.cxx,v 1.6 2002/05/30 13:28:57 rdm Exp $
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
TAlien::TAlien(const char *grid, const char *uid, const char *pw,
               const char *options)
{
   // Open a connection to the AliEn GRID. The grid argument should be
   // of the form "alien[://<host>][:<port>], e.g.: "alien" or
   // "alien://alice.cern.ch". The uid is the username and the pw the
   // password that should be used for the connection. In general the
   // uid and password will be taken from the AliEn client setup and
   // don't need to be specified. Supported options are:
   // -domain=<domain name>
   // -debug=<debug level from 1 to 10>
   // Example: "-domain=cern.ch -debug=5"

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

   const char *user = "";
   if (uid)
      user = uid;

   const char *opt = "";
   if (options)
      opt = options;

   if (AlienConnect(user, pw, opt) == -1) {
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

   if (IsConnected())
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
   // matching the specified wildcard pattern. For AliEn the wildcard pattern
   // must have the form:
   //  "lfn://<host>/<path>?<tagname>:<tagcondition>
   // Examples:
   //  "lfn://alien.cern.ch/alice/bin/date"
   //  "lfn:///alice/simulation/2001-04/V0.6*.root"
   //  "lfn:///alice/simulation/2001-04*?MonteCarloRuns:HolesPHOSRICH=1"
   // Returns 0 in case of error. Returned result must be deleted by user.

   AlienResult_t *ar = AlienGetFile(wildcard);

   if (!ar) return 0;

   return new TAlienResult(ar);
}

//______________________________________________________________________________
Int_t TAlien::AddFile(const char *lfn, const char *pfn, Int_t size)
{
   // Add physical filename to AliEn catalog with associated logical file name.
   // Returns -1 on error (e.g. when lfn or pfn already exists), 0 otherwise.
   // Size, in bytes, is a hint to the file catalog. If size is -1 the system
   // will try to guess size from pfn.
   // Example lfn: "lfn://[alien.cern.ch]/alice/cern.ch/user/r/rdm/aap.root"
   // Example pfn: "rfio:/castor/cern.ch/user/r/rdm/noot.root"

   TString slfn = MakeLfn(lfn);

   if (AlienAddFile(slfn, pfn, size) == -1) {
      Error("AddFile", "error adding pfn %s with lfn %s (size %d)",
            pfn, slfn.Data(), size);
      return -1;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TAlien::DeleteFile(const char *lfn)
{
   // Delete logical file from AliEn. Does not delete associated pfn's.
   // Returns -1 on error, 0 otherwise.

   TString slfn = MakeLfn(lfn);

   if (AlienDeleteFile(slfn) == -1) {
      Error("DeleteFile", "error deleting lfn %s", slfn.Data());
      return -1;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TAlien::Mkdir(const char *dir, const char *options)
{
   // Create directory in AliEn. Returns -1 on error, 0 otherwise.
   // Example dir: "lfn://[alien.cern.ch]/alice/cern.ch/user/p/psaiz/directory"
   // Supported options:
   //  "p": make all directories
   //  "s": silent mode

   TString sdir = MakeLfn(dir);

   const char *opt = "";
   if (options)
      opt = options;

   if (AlienMkDir(sdir, opt) == -1) {
      Error("Mkdir", "error creating directory %s", sdir.Data());
      return -1;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TAlien::Rmdir(const char *dir, const char *options)
{
   // Remove directory from AliEn. Returns -1 on error, 0 otherwise.
   // Example dir: "lfn://[alien.cern.ch]/alice/cern.ch/user/p/psaiz/directory"
   // Supported options:
   //  "s": silent mode

   TString sdir = MakeLfn(dir);

   const char *opt = "";
   if (options)
      opt = options;

   if (AlienRmDir(sdir, opt) == -1) {
      Error("Rmdir", "error deleting directory %s", sdir.Data());
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

   TString slfn = MakeLfn(lfn);

   char *pfn = AlienGetPhysicalFileName(slfn);

   if (!pfn) {
      Error("GetPhysicalFileName", "no physical file name found for lfn %s",
            slfn.Data());
      return 0;
   }

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

   TString slfn = MakeLfn(lfn);

   AlienResult_t *ar = AlienGetPhysicalFileNames(slfn);

   if (!ar) {
      Error("GetPhysicalFileNames", "no physical file names found for lfn %s",
            slfn.Data());
      return 0;
   }

   return new TAlienResult(ar);
}

//______________________________________________________________________________
Int_t TAlien::GetPathInfo(const char *lfn, Long_t *size, Long_t *flags,
                          Long_t *modtime)
{
   // Get info about a lfn: size, flags, modification time.
   // Size    is the file size
   // Flags   is file type: 0 is regular file, bit 1 set directory
   // Modtime is modification time.
   // The function returns 0 in case of success and -1 if the file could
   // not be stat'ed.

#if 0
   TString slfn = MakeLfn(lfn);

   AlienStat_t buf;
   if (AlienStat(slfn, &buf) == -1) {
      return -1;
   }
   if (size)
      *size = (Long_t) buf.st_size;
   if (flags) {
      *flags = 0;
      if (buf.st_mode & AL_IFDIR)
         *flags |= 1;
   }
   if (modtime)
      *modtime = buf.st_mtime;

#else
   if (size)
      *size = 0;
   if (flags)
      *flags = 0;
   if (modtime)
      *modtime = 0;
#endif

   return 0;
}

//______________________________________________________________________________
Int_t TAlien::AddAttribute(const char *lfn, const char *attrname,
                           const char *attrval)
{
   // Add attribute attrname with value attrval to specified logical
   // file name. Returns -1 on error, 0 otherwise.

   TString slfn = MakeLfn(lfn);

   // assume "standard" tag exists
   if (AlienAddAttribute(slfn, "standard", attrname, attrval) == -1) {
      Error("AddAttribute", "error adding attribute %s with value %s to lfn %s",
            attrname, attrval, slfn.Data());
      return -1;
   }

   return 0;
}

//______________________________________________________________________________
Int_t TAlien::DeleteAttribute(const char *lfn, const char *attrname)
{
   // Delete specified attribute from logical file name. If attribute
   // is 0 delete all attributes. Returns -1 on error, 0 otherwise.

   TString slfn = MakeLfn(lfn);

   const char *attr = "";
   if (attrname)
      attr = attrname;

   if (AlienDeleteAttribute(slfn, "standard", attr) == -1) {
      if (strlen(attr) > 0)
         Error("DeleteAttribute", "error deleting attribute %s from lfn %s",
               attr, slfn.Data());
      else
         Error("DeleteAttribute", "error deleting all attributes from lfn %s",
               slfn.Data());
      return -1;
   }

   return 0;
}

//______________________________________________________________________________
TGridResult *TAlien::GetAttributes(const char *lfn)
{
   // Return attributes associated with lfn. Returns 0 in case of error.
   // Returned result must be deleted by user.

   TString slfn = MakeLfn(lfn);

   // assume "standard" tag
   AlienAttr_t *at = AlienGetAttributes(slfn, "standard");

   if (!at) {
      // Don't print error message
      return 0;
   }

   return new TAlienAttrResult(at);
}

//______________________________________________________________________________
const char *TAlien::Pwd() const
{
   // Returns current working directory in the AliEn file catalog.

   return AlienPwd();
}

//______________________________________________________________________________
Int_t TAlien::Cd(const char *dir) const
{
   // Change directory in the AliEn file catalog. If dir is not specified,
   // it goes to the home directory. Returns -1 in case of failure,
   // 0 otherwise.

   const char *d = "";
   if (dir)
      d = dir;

   TString sdir = MakeLfn(d);

   if (AlienCd(sdir) == -1) {
      Error("Cd", "error making %s the current working directory", sdir.Data());
      return -1;
   }

   return 0;
}

//______________________________________________________________________________
TGridResult *TAlien::Ls(const char *dir, const char *options) const
{
   // Returns contents of a directory in the AliEn file catalog.
   // Returns 0 in case of error. Returned result must be deleted by user.
   // Example dir: "lfn://[alien.cern.ch]/alice/cern.ch/user/p/psaiz"
   // will return only "psaiz"
   //              "lfn://[alien.cern.ch]/alice/cern.ch/user/p/psaiz/"
   // will return all the files in the directory.
   // Supported options:
   //  "l": long listing format
   //  "a": list all entries
   //  "d": list only directories

   const char *d = Pwd();
   if (dir && strlen(dir) > 0)
      d = dir;

   TString sdir = MakeLfn(d);

   const char *opt = "";
   if (options)
      opt = options;

   AlienResult_t *ar = AlienLs(sdir, opt);

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

//______________________________________________________________________________
TString TAlien::MakeLfn(const char *lfn) const
{
   // Make sure that an lfn has a valid format. If lfn:// is missing add it.
   // If it is not absolute add the Pwd() in front of it.

   TString s;

   if (!lfn || !strlen(lfn))
      return s;

   if (!strncmp(lfn, "lfn://", 6))
      return TString(lfn);

   if (!strncmp(lfn, "lfn:/", 5)) {
      s = "lfn://";
      if (lfn+5)
         s += lfn+5;
      return s;
   }

   if (lfn[0] == '/') {
      s = "lfn://";
      s += lfn;
      return s;
   }

   s = "lfn://";
   s += Pwd();
   s += lfn;

   return s;
}
