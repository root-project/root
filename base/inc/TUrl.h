// @(#)root/base:$Name:  $:$Id: TUrl.h,v 1.19 2007/02/12 14:56:11 rdm Exp $
// Author: Fons Rademakers   17/01/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TUrl
#define ROOT_TUrl


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

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


class TUrl : public TObject {

private:
   TString fUrl;            // full URL
   TString fProtocol;       // protocol: http, ftp, news, root, proof, ...
   TString fUser;           // user name
   TString fPasswd;         // password
   TString fHost;           // remote host
   TString fFile;           // remote object
   TString fAnchor;         // anchor in object (after #)
   TString fOptions;        // options/search (after ?)
   mutable TString fFileOA; //!file with option and anchor
   mutable TString fHostFQ; //!fully qualified host name
   Int_t   fPort;           // port through which to contact remote server

   static TObjArray  *fgSpecialProtocols;  // list of special protocols

   void FindFile(char *u, Bool_t stripDoubleSlash = kTRUE);

public:
   TUrl() : fUrl(), fProtocol(), fUser(), fPasswd(), fHost(), fFile(),
            fAnchor(), fOptions(), fFileOA(), fHostFQ(), fPort(-1) { }
   TUrl(const char *url, Bool_t defaultIsFile = kFALSE);
   TUrl(const TUrl &url);
   TUrl &operator=(const TUrl &rhs);
   virtual ~TUrl() { }

   const char *GetUrl(Bool_t withDeflt = kFALSE);
   const char *GetProtocol() const { return fProtocol; }
   const char *GetUser() const { return fUser; }
   const char *GetPasswd() const { return fPasswd; }
   const char *GetHost() const { return fHost; }
   const char *GetHostFQDN() const;
   const char *GetFile() const { return fFile; }
   const char *GetAnchor() const { return fAnchor; }
   const char *GetOptions() const { return fOptions; }
   const char *GetFileAndOptions() const;
   Int_t       GetPort() const { return fPort; }
   Bool_t      IsValid() const { return fPort == -1 ? kFALSE : kTRUE; }

   void        SetProtocol(const char *proto, Bool_t setDefaultPort = kFALSE);
   void        SetUser(const char *user) { fUser = user; fUrl = ""; }
   void        SetPasswd(const char *pw) { fPasswd = pw; fUrl = ""; }
   void        SetHost(const char *host) { fHost = host; fUrl = ""; }
   void        SetFile(const char *file) { fFile = file; fUrl = ""; fFileOA = "";}
   void        SetAnchor(const char *anchor) { fAnchor = anchor; fUrl = ""; fFileOA = ""; }
   void        SetOptions(const char *opt) { fOptions = opt; fUrl = ""; fFileOA = ""; }
   void        SetPort(Int_t port) { fPort = port; fUrl = ""; }
   void        SetUrl(const char *url, Bool_t defaultIsFile = kFALSE);

   Bool_t      IsSortable() const { return kTRUE; }
   Int_t       Compare(const TObject *obj) const;

   void        Print(Option_t *option="") const;

   static TObjArray *GetSpecialProtocols();

   ClassDef(TUrl,1)  //Represents an URL
};

#endif
