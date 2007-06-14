// @(#)root/net:$Name:  $:$Id: TWebFile.h,v 1.9 2006/05/29 07:59:27 rdm Exp $
// Author: Fons Rademakers   17/01/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWebFile
#define ROOT_TWebFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWebFile                                                             //
//                                                                      //
// A TWebFile is like a normal TFile except that it reads its data      //
// via a (slightly modified) apache web server. A TWebFile is a         //
// read-only file.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFile
#include "TFile.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif

class TSocket;


class TWebFile : public TFile {

private:
   mutable Long64_t  fSize;         // file size
   Bool_t            fHasModRoot;   // true if server has mod_root installed

   TWebFile() { }
   void   Init(Bool_t);
   Int_t  GetHead();
   Int_t  GetLine(TSocket *s, char *line, Int_t size);
   Int_t  GetFromWeb(char *buf, Int_t len, const TString &msg);
   Int_t  GetFromWeb10(char *buf, Int_t len, const TString &msg);
   Int_t  GetFromWebMulti10(char *buf, Int_t len, const TString &msg);
   Bool_t ReadBuffer10(char *buf, Int_t len);
   Bool_t ReadBuffers10(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf);

public:
   TWebFile(const char *url);
   TWebFile(TUrl url);
   virtual ~TWebFile() { }

   Long64_t GetSize() const;
   Bool_t   IsOpen() const;
   Int_t    ReOpen(Option_t *mode);
   Bool_t   ReadBuffer(char *buf, Int_t len);
   Bool_t   ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf);
   void     Seek(Long64_t offset, ERelativeTo pos = kBeg);

   ClassDef(TWebFile,1)  //A ROOT file that reads via a http server
};

#endif
