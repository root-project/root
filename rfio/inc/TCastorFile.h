// @(#)root/rfio:$Name:  $:$Id: TCastorFile.h,v 1.1 2003/09/21 21:38:30 rdm Exp $
// Author: Fons Rademakers   17/09/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCastorFile
#define ROOT_TCastorFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCastorFile                                                          //
//                                                                      //
// A TCastorFile is like a normal TNetFile except that it obtains the   //
// remote node (disk server) via the CASTOR API, once the disk server   //
// and the local file path are determined, the file will be accessed    //
// via the rootd daemon. File names have to be specified like:          //
//    castor:/castor/cern.ch/user/r/rdm/bla.root.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNetFile
#include "TNetFile.h"
#endif


class TCastorFile : public TNetFile {

private:
   TString   fDiskServer;    // CASTOR remote disk server
   TString   fInternalPath;  // CASTOR internal path
   Bool_t    fIsCastor;      // true if internal path is valid
   Bool_t    fWrittenTo;     // true if data has been written to file

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

   ClassDef(TCastorFile,1)  //A ROOT file that reads/writes via a rootd server to a CASTOR disk server
};

#endif
