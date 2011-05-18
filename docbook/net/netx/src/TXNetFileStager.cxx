// @(#)root/netx:$Id$
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
// TXNetFileStager                                                      //
//                                                                      //
// Interface to the 'XRD' staging capabilities.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TError.h"
#include "TObjString.h"
#include "TUrl.h"
#include "TXNetFileStager.h"
#include "TXNetSystem.h"

//_____________________________________________________________________________
TXNetFileStager::TXNetFileStager(const char *url) : TFileStager("xrd")
{
   // Constructor. Init a TXNetSystem instance to the XRD system.

   fSystem = 0;
   if (url && strlen(url) > 0) {
      GetPrefix(url, fPrefix);

      fSystem = new TXNetSystem(fPrefix);
   }
}

//_____________________________________________________________________________
TXNetFileStager::~TXNetFileStager()
{
   // Destructor

   if (fSystem)
      delete fSystem;
   fSystem = 0;
   fPrefix = "";
}

//_____________________________________________________________________________
Bool_t TXNetFileStager::IsStaged(const char *path)
{
   // Check if the file defined by 'path' is ready to be used.

   if (!IsValid()) {
      GetPrefix(path, fPrefix);
      fSystem = new TXNetSystem(path);
   }

   if (IsValid()) {
      TString p(path);
      if (!p.BeginsWith("root:"))
         p.Insert(0, fPrefix);
      return (fSystem->IsOnline(p));
   }

   // Failure
   Warning("IsStaged","TXNetSystem not initialized");
   return kFALSE;
}

//_____________________________________________________________________________
Bool_t TXNetFileStager::Stage(TCollection *paths, Option_t *opt)
{
   // Issue a stage request for file defined by 'path'. The string 'opt'
   // defines 'option' and 'priority' for 'Prepare': the format is
   //    opt = "option=o priority=p".

   if (IsValid()) {
      UChar_t o = 8;
      UChar_t p = 0;
      // Parse options, if any
      if (opt && strlen(opt) > 0) {
         TString xo(opt), io;
         Ssiz_t from = 0;
         while (xo.Tokenize(io, from, "[ ,|]")) {
            if (io.Contains("option=")) {
               io.ReplaceAll("option=","");
               if (io.IsDigit()) {
                  Int_t i = io.Atoi();
                  if (i >= 0 && i <= 255)
                     o = (UChar_t) i;
               }
            }
            if (io.Contains("priority=")) {
               io.ReplaceAll("priority=","");
               if (io.IsDigit()) {
                  Int_t i = io.Atoi();
                  if (i >= 0 && i <= 255)
                     p = (UChar_t) i;
               }
            }
         }
      }
      // Run prepare
      return fSystem->Prepare(paths, o, p);
   }

   // Failure
   Warning("Stage","TXNetSystem not initialized");
   return kFALSE;

}

//_____________________________________________________________________________
Bool_t TXNetFileStager::Stage(const char *path, Option_t *opt)
{
   // Issue a stage request for file defined by 'path'. The string 'opt'
   // defines 'option' and 'priority' for 'Prepare': the format is
   //                opt = "option=o priority=p".

   if (!IsValid()) {
      GetPrefix(path, fPrefix);
      fSystem = new TXNetSystem(path);
   }

   if (IsValid()) {
      UChar_t o = 8;
      UChar_t p = 0;
      // Parse options
      TString xo(opt), io;
      Ssiz_t from = 0;
      while (xo.Tokenize(io, from, "[ ,|]")) {
         if (io.Contains("option=")) {
            io.ReplaceAll("option=","");
            if (io.IsDigit()) {
               Int_t i = io.Atoi();
               if (i >= 0 && i <= 255)
                  o = (UChar_t) i;
            }
         }
         if (io.Contains("priority=")) {
            io.ReplaceAll("priority=","");
            if (io.IsDigit()) {
               Int_t i = io.Atoi();
               if (i >= 0 && i <= 255)
                  p = (UChar_t) i;
            }
         }
      }
      // Make user the full path is used
      TString pp(path);
      if (!pp.BeginsWith("root:"))
         pp.Insert(0, fPrefix);
      return fSystem->Prepare(pp, o, p);
   }

   // Failure
   Warning("Stage","TXNetSystem not initialized");
   return kFALSE;
}

//_____________________________________________________________________________
void TXNetFileStager::GetPrefix(const char *url, TString &pfx)
{
   // Isolate prefix in url

   if (gDebug > 1)
      ::Info("TXNetFileStager::GetPrefix", "enter: %s", url);

   TUrl u(url);
   pfx = Form("%s://", u.GetProtocol());
   if (strlen(u.GetUser()) > 0)
      pfx += Form("%s@", u.GetUser());
   pfx += u.GetHost();
   if (u.GetPort() != TUrl("root://host").GetPort())
      pfx += Form(":%d", u.GetPort());
   pfx += "/";

   if (gDebug > 1)
      ::Info("TXNetFileStager::GetPrefix", "found prefix: %s", pfx.Data());
}

//_____________________________________________________________________________
void TXNetFileStager::Print(Option_t *) const
{
   // Print basic info about this stager

   Printf("+++ stager: %s  %s", GetName(), fPrefix.Data());
}

//______________________________________________________________________________
Int_t TXNetFileStager::Locate(const char *path, TString &eurl)
{
   // Get actual end-point url for a path
   // Returns 0 in case of success and 1 if any error occured

   if (!IsValid()) {
      GetPrefix(path, fPrefix);
      fSystem = new TXNetSystem(path);
   }

   if (IsValid())
      return fSystem->Locate(path, eurl);

   // Unable to initialize TXNetSystem
   return -1;
}

//______________________________________________________________________________
Bool_t TXNetFileStager::Matches(const char *s)
{
   // Returns kTRUE if stager 's' is compatible with current stager.
   // Avoids multiple instantiations of the potentially the same TXNetSystem.

   if (IsValid()) {
      TString pfx;
      GetPrefix(s, pfx);
      return ((fPrefix == pfx) ? kTRUE : kFALSE);
   }

   // Not valid
   return kFALSE;
}
