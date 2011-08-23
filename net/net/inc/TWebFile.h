// @(#)root/net:$Id$
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
// via a standard apache web server. A TWebFile is a read-only file.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFile
#include "TFile.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif
#ifndef ROOT_TSystem
#include "TSystem.h"
#endif

class TSocket;
class TWebSocket;


class TWebFile : public TFile {

friend class TWebSocket;
friend class TWebSystem;

private:
	TWebFile() : fSocket(0) { }

protected:
   mutable Long64_t  fSize;             // file size
   TSocket          *fSocket;           // socket for HTTP/1.1 (stays alive between calls)
   TUrl              fProxy;            // proxy URL
   Bool_t            fHasModRoot;       // true if server has mod_root installed
   Bool_t            fHTTP11;           // true if server support HTTP/1.1
   Bool_t            fNoProxy;          // don't use proxy
   TString           fMsgReadBuffer;    // cache ReadBuffer() msg
   TString           fMsgReadBuffer10;  // cache ReadBuffer10() msg
   TString           fMsgGetHead;       // cache GetHead() msg
   TString           fBasicUrl;         // basic url without authentication and options
   TUrl              fUrlOrg;           // save original url in case of temp redirection
   TString           fBasicUrlOrg;      // save original url in case of temp redirection

   static TUrl       fgProxy;           // globally set proxy URL

   virtual void        Init(Bool_t readHeadOnly);
   virtual void        CheckProxy();
   virtual TString     BasicAuthentication();
   virtual Int_t       GetHead();
   virtual Int_t       GetLine(TSocket *s, char *line, Int_t maxsize);
   virtual Int_t       GetHunk(TSocket *s, char *hunk, Int_t maxsize);
   virtual const char *HttpTerminator(const char *start, const char *peeked, Int_t peeklen);
   virtual Int_t       GetFromWeb(char *buf, Int_t len, const TString &msg);
   virtual Int_t       GetFromWeb10(char *buf, Int_t len, const TString &msg);
   virtual Bool_t      ReadBuffer10(char *buf, Int_t len);
   virtual Bool_t      ReadBuffers10(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf);
   virtual void        SetMsgReadBuffer10(const char *redirectLocation = 0, Bool_t tempRedirect = kFALSE);

public:
   TWebFile(const char *url, Option_t *opt="");
   TWebFile(TUrl url, Option_t *opt="");
   virtual ~TWebFile();

   virtual Long64_t    GetSize() const;
   virtual Bool_t      IsOpen() const;
   virtual Int_t       ReOpen(Option_t *mode);
   virtual Bool_t      ReadBuffer(char *buf, Int_t len);
   virtual Bool_t      ReadBuffer(char *buf, Long64_t pos, Int_t len);
   virtual Bool_t      ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf);
   virtual void        Seek(Long64_t offset, ERelativeTo pos = kBeg);

   static void        SetProxy(const char *url);
   static const char *GetProxy();

   ClassDef(TWebFile,2)  //A ROOT file that reads via a http server
};


class TWebSystem : public TSystem {

private:
   void *fDirp;    // directory handler

   void *GetDirPtr() const { return fDirp; }

public:
   TWebSystem();
   virtual ~TWebSystem() { }

   Int_t       MakeDirectory(const char *name);
   void       *OpenDirectory(const char *name);
   void        FreeDirectory(void *dirp);
   const char *GetDirEntry(void *dirp);
   Int_t       GetPathInfo(const char *path, FileStat_t &buf);
   Bool_t      AccessPathName(const char *path, EAccessMode mode);
   Int_t       Unlink(const char *path);

   ClassDef(TWebSystem,0)  // Directory handler for HTTP (TWebFiles)
};

#endif
