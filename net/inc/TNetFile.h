// @(#)root/net:$Name:  $:$Id: TNetFile.h,v 1.3 2000/12/13 15:13:53 brun Exp $
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
   Int_t     fProtocol;   //rootd protocol level

   TNetFile() : fUrl("dummy") { fSocket = 0; }
   void   Init(Bool_t create);
   void   Print(Option_t *option) const;
   void   PrintError(const char *where, Int_t err) const;
   Int_t  Recv(Int_t &status, EMessageTypes &kind);
   Int_t  SysStat(Int_t fd, Long_t *id, Long_t *size, Long_t *flags, Long_t *modtime);

public:
   TNetFile(const char *url, Option_t *option="", const char *ftitle="", Int_t compress=1);
   virtual ~TNetFile();

   void    Close(Option_t *option=""); // *MENU*
   void    Flush();
   Bool_t  IsOpen() const;
   Bool_t  ReadBuffer(char *buf, Int_t len);
   Bool_t  WriteBuffer(const char *buf, Int_t len);
   void    Seek(Seek_t offset, ERelativeTo pos = kBeg);

   ClassDef(TNetFile,1)  //A ROOT file that reads/writes via a rootd server
};

#endif
