// @(#)root/net:$Name:  $:$Id: TNetFile.h,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
// Author: Fons Rademakers   14/08/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TNetFile
#define ROOT_TNetFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNetFile                                                             //
//                                                                      //
// A TNetFile is like a normal TFile except that it reads and writes    //
// its data via a rootd server.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFile
#include "TFile.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif
#ifndef ROOT_MessageTypes
#include "MessageTypes.h"
#endif

class TSocket;


class TNetFile : public TFile {

private:
   TUrl      fUrl;        //URL of file
   TString   fUser;       //remote user name
   Seek_t    fOffset;     //seek offset
   TSocket  *fSocket;     //connection to rootd server

   TNetFile() : fUrl("dummy") { fSocket = 0; }
   void   Init(Bool_t create);
   void   Print(Option_t *option);
   void   PrintError(const char *where, Int_t err);
   Int_t  Recv(Int_t &status, EMessageTypes &kind);

public:
   TNetFile(const char *url, Option_t *option="", const char *ftitle="", Int_t compress=1);
   virtual ~TNetFile();

   void    Close(Option_t *option=""); // *MENU*
   void    Flush();
   Bool_t  IsOpen() const;
   Bool_t  ReadBuffer(char *buf, int len);
   Bool_t  WriteBuffer(const char *buf, int len);
   void    Seek(Seek_t offset, ERelativeTo pos = kBeg);

   ClassDef(TNetFile,1)  //A ROOT file that reads/writes via a rootd server
};

#endif
