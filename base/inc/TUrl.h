// @(#)root/net:$Name:  $:$Id: TUrl.h,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
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
// It provides member functions to returns the different parts of       //
// an URL.                                                              //
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
   TString fProtocol;    // protocol: http, ftp, news, root, rfio, hpss
   TString fHost;        // remote host
   TString fFile;        // remote object
   TString fAnchor;      // anchor in object
   TString fOptions;     // options (after ?)
   Int_t   fPort;        // port through which to contact remote server

   TUrl() { fPort = -1; }

public:
   TUrl(const char *url);
   TUrl(const TUrl &url);
   TUrl &operator=(const TUrl &rhs);
   virtual ~TUrl() { }

   const char *GetUrl();
   const char *GetProtocol() const { return fProtocol.Data(); }
   const char *GetHost() const { return fHost.Data(); }
   const char *GetFile() const { return fFile.Data(); }
   const char *GetAnchor() const { return fAnchor.Data(); }
   const char *GetOptions() const { return fOptions.Data(); }
   Int_t       GetPort() const { return fPort; }
   Bool_t      IsValid() const { return fPort == -1 ? kFALSE : kTRUE; }
   void        Print(Option_t *option="") const;

   ClassDef(TUrl,1)  //Represents an URL
};

#endif
