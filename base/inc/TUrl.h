// @(#)root/net:$Name:  $:$Id: TUrl.h,v 1.3 2002/10/25 00:19:59 rdm Exp $
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
   TString fUrl;         // full URL
   TString fProtocol;    // protocol: http, ftp, news, root, proof, rfio, hpss
   TString fUser;        // user name
   TString fPasswd;      // password
   TString fHost;        // remote host
   TString fFile;        // remote object
   TString fAnchor;      // anchor in object
   TString fOptions;     // options (after ?)
   Int_t   fPort;        // port through which to contact remote server

   static TObjArray  *fgSpecialProtocols;  // list of special protocols

   TUrl() { fPort = -1; }

   static TObjArray *GetSpecialProtocols();

public:
   TUrl(const char *url, Bool_t defaultIsFile = kFALSE);
   TUrl(const TUrl &url);
   TUrl &operator=(const TUrl &rhs);
   virtual ~TUrl() { }

   const char *GetUrl();
   const char *GetProtocol() const { return fProtocol; }
   const char *GetUser() const { return fUser; }
   const char *GetPasswd() const { return fPasswd; }
   const char *GetHost() const { return fHost; }
   const char *GetFile() const { return fFile; }
   const char *GetAnchor() const { return fAnchor; }
   const char *GetOptions() const { return fOptions; }
   Int_t       GetPort() const { return fPort; }
   Bool_t      IsValid() const { return fPort == -1 ? kFALSE : kTRUE; }
   void        Print(Option_t *option="") const;

   ClassDef(TUrl,1)  //Represents an URL
};

#endif
