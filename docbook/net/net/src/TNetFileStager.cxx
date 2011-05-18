// @(#)root/net:$Id$
// Author: G. Ganis Feb 2011

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNetFileStager                                                       //
//                                                                      //
// TFileStager implementation for a 'rootd' backend.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TError.h"
#include "TObjString.h"
#include "TUrl.h"
#include "TNetFile.h"
#include "TNetFileStager.h"

//_____________________________________________________________________________
TNetFileStager::TNetFileStager(const char *url) : TFileStager("net")
{
   // Constructor. Init a TNetSystem instance to the remote rootd.

   fSystem = 0;
   if (url && strlen(url) > 0) {
      GetPrefix(url, fPrefix);

      fSystem = new TNetSystem(fPrefix);
   }
}

//_____________________________________________________________________________
TNetFileStager::~TNetFileStager()
{
   // Destructor

   SafeDelete(fSystem);
   fPrefix = "";
}

//_____________________________________________________________________________
Bool_t TNetFileStager::IsStaged(const char *path)
{
   // Check if the file defined by 'path' is ready to be used.

   if (!IsValid()) {
      GetPrefix(path, fPrefix);
      fSystem = new TNetSystem(path);
   }

   if (IsValid()) {
      TString p(path);
      if (!p.BeginsWith(fPrefix)) p.Insert(0, fPrefix);
      return (fSystem->AccessPathName(p, kReadPermission) ? kFALSE : kTRUE);
   }

   // Failure
   Warning("IsStaged","TNetSystem not initialized");
   return kFALSE;
}

//_____________________________________________________________________________
void TNetFileStager::GetPrefix(const char *url, TString &pfx)
{
   // Isolate prefix in url

   if (gDebug > 1)
      ::Info("TNetFileStager::GetPrefix", "enter: %s", url);

   TUrl u(url);
   pfx = TString::Format("%s://", u.GetProtocol());
   if (strlen(u.GetUser()) > 0)
      pfx += TString::Format("%s@", u.GetUser());
   pfx += u.GetHost();
   if (u.GetPort() != TUrl("root://host").GetPort())
      pfx += TString::Format(":%d", u.GetPort());
   pfx += "/";

   if (gDebug > 1)
      ::Info("TNetFileStager::GetPrefix", "found prefix: %s", pfx.Data());
}

//_____________________________________________________________________________
void TNetFileStager::Print(Option_t *) const
{
   // Print basic info about this stager

   Printf("+++ stager: %s  %s", GetName(), fPrefix.Data());
}

//______________________________________________________________________________
Int_t TNetFileStager::Locate(const char *path, TString &eurl)
{
   // Get actual end-point url for a path
   // Returns 0 in case of success and 1 if any error occured

   if (!IsValid()) {
      GetPrefix(path, fPrefix);
      fSystem = new TNetSystem(path);
   }

   if (IsValid()) {
      TString p(path);
      if (!p.BeginsWith(fPrefix)) p.Insert(0, fPrefix);
      if (!fSystem->AccessPathName(p, kReadPermission)) {
         eurl = p;
         return 0;
      }
   }

   // Unable to initialize TNetSystem or file does not exist
   return -1;
}

//______________________________________________________________________________
Bool_t TNetFileStager::Matches(const char *s)
{
   // Returns kTRUE if stager 's' is compatible with current stager.
   // Avoids multiple instantiations of the potentially the same TNetSystem.

   if (IsValid()) {
      TString pfx;
      GetPrefix(s, pfx);
      return ((fPrefix == pfx) ? kTRUE : kFALSE);
   }

   // Not valid
   return kFALSE;
}
