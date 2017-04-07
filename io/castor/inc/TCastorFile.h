// @(#)root/castor:$Id$
// Author: Fons Rademakers  17/09/2003 + Giulia Taurelli  29/06/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCastorFile
#define ROOT_TCastorFile

#include "TNetFile.h"


class TCastorFile : public TNetFile {

private:
   TString   fDiskServer;    ///< CASTOR remote disk server
   TString   fInternalPath;  ///< CASTOR internal path
   Bool_t    fIsCastor;      ///< true if internal path is valid
   Bool_t    fWrittenTo;     ///< true if data has been written to file

   TString   fAuthProto;     ///< Used to specific the auth protocol

   void FindServerAndPath();
   void ConnectServer(Int_t *stat, EMessageTypes *kind, Int_t netopt,
                      Int_t tcpwindowsize, Bool_t forceOpen,
                      Bool_t forceRead);
   Int_t SysClose(Int_t fd);

public:
   TCastorFile(const char *url, Option_t *option = "", const char *ftitle = "",
               Int_t compress = 1, Int_t netopt = 0);
   TCastorFile() : TNetFile() { fIsCastor = fWrittenTo = kFALSE; }
   virtual ~TCastorFile() { }

   Bool_t WriteBuffer(const char *buf, Int_t len);

   ClassDef(TCastorFile,1) //TFile reading/writing via rootd to a CASTOR server
};

#endif
