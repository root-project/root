// @(#)root/net:$Name:  $:$Id: TUrl.cxx,v 1.18 2004/07/19 09:43:58 rdm Exp $
// Author: Fons Rademakers   17/01/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TUrl                                                                 //
//                                                                      //
// This class represents a WWW compatible URL.                          //
// It provides member functions to return the different parts of        //
// an URL. The supported url format is:                                 //
//  [proto://][user[:passwd]@]host[:port]/file.ext[#anchor][?options]   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include "TUrl.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TEnv.h"

TObjArray *TUrl::fgSpecialProtocols;


ClassImp(TUrl)

//______________________________________________________________________________
TUrl::TUrl(const char *url, Bool_t defaultIsFile)
{
   // Parse url character string and split in its different subcomponents.
   // Use IsValid() to check if URL is legal.
   //
   // url: [proto://][user[:passwd]@]host[:port]/file.ext[#anchor][?options]
   //
   // Known protocols: http, root, proof, ftp, news and any special protocols
   // defined in the rootrc Url.Special key.
   // The default protocol is "http", unless defaultIsFile is true in which
   // case the url is assumed to be of type "file".
   // If a passwd contains a @ it must be escaped by a \\, e.g.
   // "pip@" becomes "pip\\@".
   //
   // Default ports: http=80, root=1094, proof=1093, ftp=20, news=119.
   // Port #1093 has been assigned by IANA (www.iana.org) to proofd.
   // Port #1094 has been assigned by IANA (www.iana.org) to rootd.

   if (!url || !strlen(url)) {
      fPort = -1;
      return;
   }

   // Set defaults
   fUrl      = "";
   fProtocol = "http";
   fUser     = "";
   fPasswd   = "";
   fHost     = "";
   fPort     = 80;
   fFile     = "/";
   fAnchor   = "";
   fOptions  = "";

   // Find protocol
   char *s, sav;

   // Handle special protocol cases: "file:", "rfio:", etc.
   for (int i = 0; i < GetSpecialProtocols()->GetEntries(); i++) {
      TObjString *os = (TObjString*) GetSpecialProtocols()->UncheckedAt(i);
      TString &s = os->String();
      int l = s.Length();
      if (!strncmp(url, s, l)) {
         fProtocol = s(0, l-1);
         if (!strncmp(url+5, "//", 2))
            fFile = url+l+2;
         else
            fFile = url+l;
         fPort = 0;
         // look for an anchor so we can get the desired member in case
         // of an archive file url
         if ((i = fFile.Last('#')) != kNPOS) {
            TString ff = fFile;
            fFile = ff(0, i);
            fAnchor = ff(i+1, ff.Length()-1);
         }
         return;
      }
   }

   char *u0, *u = StrDup(url);
   u0 = u;

   Bool_t isWin32File = kFALSE;
#ifdef R__WIN32
   isWin32File = defaultIsFile;
#endif

   char *t, *s2;
   if ((s = strstr(u, ":/")) && !isWin32File) {
      if (*(s+2) != '/') {
         Error("TUrl", "malformed, URL must contain \"://\"");
         fPort = -1;
         goto cleanup;
      }
      sav = *s;
      *s = 0;
      fProtocol = u;
      *s = sav;
      if (!fProtocol.CompareTo("http"))
         fPort = 80;
      else if (fProtocol.BeginsWith("proof"))  // can also be proofs or proofk
         fPort = 1093;
      else if (fProtocol.BeginsWith("root"))   // can also be roots or rootk
         fPort = 1094;
      else if (!fProtocol.CompareTo("ftp"))
         fPort = 20;
      else if (!fProtocol.CompareTo("news"))
         fPort = 119;
      else {
         // generic protocol (no default port)
         fPort = 0;
      }
      s += 3;
      if (!*s) {
         // error if we are at end of string
         fPort = -1;
         goto cleanup;
      }
   } else {
      if (defaultIsFile) {
         fProtocol = "file";
         fFile = u;
         fPort = 0;
         goto cleanup;
      }
      s = u;
   }

   // Find user and passwd
   u = s;
   t = s;
again:
   if ((s = strchr(t, '@'))) {
      if (*(s-1) == '\\') {
         t = s+1;
         goto again;
      }
      sav = *s;
      *s = 0;
      if ((s2 = strchr(u, ':'))) {
         *s2 = 0;
         fUser = u;
         *s2 = ':';
         s2++;
         if (*s2) {
            fPasswd = s2;
            fPasswd.ReplaceAll("\\@", "@");
         }
      } else
         fUser = u;
      *s = sav;
      s++;
   } else
      s = u;

   // Find host
   u = s;
   if ((s = strchr(u, ':')) || (s = strchr(u, '/'))) {
      if ((strchr (u, ':') > strchr(u, '/')) && (strchr (u, '/')))
	s = strchr(u, '/');
      sav = *s;
      *s = 0;
      fHost = u;
      *s = sav;
      if (sav == ':') {
         s++;
         // Get port #
         if (!*s) {
            fPort = -1;
            goto cleanup;
         }
         u = s;
         if ((s = strchr(u, '/'))) {
            sav = *s;
            *s = 0;
            fPort = atoi(u);
            *s = sav;
         } else {
            fPort = atoi(u);
            goto cleanup;
         }
      }
   } else {
      fHost = u;
      goto cleanup;
   }

   if (!*s) goto cleanup;

   // Find file
   u = s;
   if ((s = strchr(u, '#')) || (s = strchr(u, '?'))) {
      sav = *s;
      *s = 0;
      fFile = u;
      *s = sav;
      s++;
      if (sav == '#') {
         // Get anchor
         if (!*s) {
            fPort = -1;
            goto cleanup;
         }
         u = s;
         if ((s = strchr(u, '?'))) {
            sav = *s;
            *s = 0;
            fAnchor = u;
            *s = sav;
            s++;
         } else {
            fAnchor = u;
            goto cleanup;
         }
      }
      if (!*s) {
         // error if we are at end of string
         fPort = -1;
         goto cleanup;
      }
   } else {
      fFile = u;
      goto cleanup;
   }

   // Set option
   fOptions = s;

cleanup:
   delete [] u0;
}

//______________________________________________________________________________
TUrl::TUrl(const TUrl &url) : TObject(url)
{
   // TUrl copt ctor.

   fUrl      = url.fUrl;
   fProtocol = url.fProtocol;
   fUser     = url.fUser;
   fPasswd   = url.fPasswd;
   fHost     = url.fHost;
   fFile     = url.fFile;
   fAnchor   = url.fAnchor;
   fOptions  = url.fOptions;
   fPort     = url.fPort;
}

//______________________________________________________________________________
TUrl &TUrl::operator=(const TUrl &rhs)
{
   // TUrl assignment operator.

   if (this != &rhs) {
      TObject::operator=(rhs);
      fUrl      = rhs.fUrl;
      fProtocol = rhs.fProtocol;
      fUser     = rhs.fUser;
      fPasswd   = rhs.fPasswd;
      fHost     = rhs.fHost;
      fFile     = rhs.fFile;
      fAnchor   = rhs.fAnchor;
      fOptions  = rhs.fOptions;
      fPort     = rhs.fPort;
   }
   return *this;
}

//______________________________________________________________________________
const char *TUrl::GetUrl()
{
   // Return full URL.

   if (IsValid() && fUrl == "") {
      // Handle special protocol cases: file:, rfio:, etc.
      for (int i = 0; i < GetSpecialProtocols()->GetEntries(); i++) {
         TObjString *os = (TObjString*) GetSpecialProtocols()->UncheckedAt(i);
         TString &s = os->String();
         int l = s.Length();
         if (fProtocol == s(0, l-1)) {
            fUrl = fProtocol + ":" + fFile;
            return fUrl;
         }
      }

      Bool_t deflt = kFALSE;
      if ((!fProtocol.CompareTo("http")   && fPort == 80)   ||
          (!fProtocol.CompareTo("proof")  && fPort == 1093) ||
          (!fProtocol.CompareTo("proofs") && fPort == 1093) ||
          (!fProtocol.CompareTo("proofk") && fPort == 1093) ||
          (!fProtocol.CompareTo("root")   && fPort == 1094) ||
          (!fProtocol.CompareTo("roots")  && fPort == 1094) ||
          (!fProtocol.CompareTo("rootk")  && fPort == 1094) ||
          (!fProtocol.CompareTo("ftp")    && fPort == 20)   ||
          (!fProtocol.CompareTo("news")   && fPort == 119)  ||
           fPort == 0)
         deflt = kTRUE;

      fUrl = fProtocol + "://";
      if (fUser != "") {
         fUrl += fUser;
         if (fPasswd != "") {
            fUrl += ":";
            TString passwd = fPasswd;
            passwd.ReplaceAll("@", "\\@");
            fUrl += passwd;
         }
         fUrl += "@";
      }
      if (!deflt) {
         char p[10];
         sprintf(p, "%d", fPort);
         fUrl = fUrl + fHost + ":" + p + fFile;
      } else
         fUrl = fUrl + fHost + fFile;
      if (fAnchor != "") {
         fUrl += "#";
         fUrl += fAnchor;
      }
      if (fOptions != "") {
         fUrl += "?";
         fUrl += fOptions;
      }
   }

   return fUrl;
}

//______________________________________________________________________________
void TUrl::Print(Option_t *) const
{
   // Print URL on stdout.

   if (fPort == -1)
      Printf("Illegal URL");

   Printf("%s", ((TUrl*)this)->GetUrl());
}

//______________________________________________________________________________
TObjArray *TUrl::GetSpecialProtocols()
{
   // Read the list of special protocols from the rootrc files.
   // These protocols will be parsed in a protocol and a file part,
   // no host or other info will be determined. This is typically
   // used for legacy file descriptions like: rfio:host:/path/file.root.

   if (!fgSpecialProtocols)
      fgSpecialProtocols = new TObjArray;

   if (fgSpecialProtocols->GetEntries() > 0 || !gEnv)
      return fgSpecialProtocols;

   const char *protos = gEnv->GetValue("Url.Special", "file: rfio: hpss: castor: dcache: dcap:");

   if (protos) {
      Int_t cnt = 0;
      char *p = StrDup(protos);
      while (1) {
         TObjString *proto = new TObjString(strtok(!cnt ? p : 0, " "));
         if (proto->String().IsNull()) {
            delete proto;
            break;
         }
         fgSpecialProtocols->Add(proto);
         cnt++;
      }
      delete [] p;
   }
   return fgSpecialProtocols;
}
