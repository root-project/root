// @(#)root/net:$Name:  $:$Id: TUrl.cxx,v 1.19 2005/05/12 12:40:53 rdm Exp $
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
   fUrl       = "";
   fProtocol  = "http";
   fUser      = "";
   fPasswd    = "";
   fHost      = "";
   fPort      = 80;
   fFile      = "/";
   fAnchor    = "";
   fOptions   = "";

   // Find protocol
   char *s, sav;

   char *u, *u0 = StrDup(url);
tryfile:
   u = u0;

   // Handle special protocol cases: "file:", "rfio:", etc.
   for (int i = 0; i < GetSpecialProtocols()->GetEntries(); i++) {
      TObjString *os = (TObjString*) GetSpecialProtocols()->UncheckedAt(i);
      TString s1 = os->GetString();
      int l = s1.Length();
      Bool_t stripoff = kFALSE;
      if (s1.EndsWith("/-")) {
         stripoff = kTRUE;
         s1 = s1.Strip(TString::kTrailing, '-');
         l--;
      }
      if (!strncmp(u, s1, l)) {
         if (s1(0) == '/' && s1(l-1) == '/') {
            // case whith file namespace like: /alien/user/file.root
            fProtocol = s1(1, l-2);
            if (stripoff)
               l--;    // strip off namespace prefix from file name
            else
               l = 0;  // leave namespace prefix as part of file name
         } else {
            // case with protocol, like: rfio:machine:/data/file.root
            fProtocol = s1(0, l-1);
         }
         if (!strncmp(u+l, "//", 2))
            u += l+2;
         else
            u += l;
         fPort = 0;

         FindFile(u);

         delete [] u0;
         return;
      }
   }

   u = u0;

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
         char *newu = new char [strlen("file:") + strlen(u0) + 1];
         sprintf(newu, "file:%s", u0);
         delete [] u0;
         u0 = newu;
         goto tryfile;
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

   FindFile(u);

cleanup:
   delete [] u0;
}

//______________________________________________________________________________
void TUrl::FindFile(char *u)
{
   // Find file and optionally anchor and options.

   char *s, sav;

   if ((s = strchr(u, '#')) || (s = strchr(u, '?'))) {
      sav = *s;
      *s = 0;
      fFile = u;
      *s = sav;
      s++;
      if (sav == '#') {
         // Get anchor
         if (!*s) {
         // error if we are at end of string
            fPort = -1;
            return;
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
            return;
         }
      }
      if (!*s) {
         // error if we are at end of string
         fPort = -1;
         return;
      }
   } else {
      fFile = u;
      return;
   }

   // Set option
   fOptions = s;
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
            if (fAnchor != "") {
               fUrl += "#";
               fUrl += fAnchor;
            }
            if (fOptions != "") {
               fUrl += "?";
               fUrl += fOptions;
            }
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
const char *TUrl::GetFileAndOptions() const
{
   // Return the file and its options (the string specified behind the ?).
   // Convenience function useful when the option is used to pass
   // authetication/access information for the specified file.

   fFileAO = fFile;
   if (fAnchor != "") {
      fFileAO += "#";
      fFileAO += fAnchor;
   }
   if (fOptions != "") {
      fFileAO += "?";
      fFileAO += fOptions;
   }
   return fFileAO;
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

   static Bool_t usedEnv = kFALSE;

   if (!fgSpecialProtocols)
      fgSpecialProtocols = new TObjArray;

   if (!gEnv) {
      if (fgSpecialProtocols->GetEntries() == 0)
         fgSpecialProtocols->Add(new TObjString("file:"));
      return fgSpecialProtocols;
   }

   if (fgSpecialProtocols->GetEntries() > 0 && usedEnv)
      return fgSpecialProtocols;

   fgSpecialProtocols->Delete();

   const char *protos = gEnv->GetValue("Url.Special", "file: rfio: hpss: castor: dcache: dcap:");
   usedEnv = kTRUE;

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
