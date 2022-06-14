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

#include "TFile.h"
#include "TUrl.h"
#include "TSystem.h"

class TSocket;
class TWebSocket;


class TWebFile : public TFile {

friend class TWebSocket;
friend class TWebSystem;

private:
   TWebFile() : fSocket(nullptr) {}

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
   void             *fFullCache;        //! complete content of the file, some http server may return complete content
   Long64_t          fFullCacheSize;    //! size of the cached content

   static TUrl       fgProxy;           // globally set proxy URL
   static Long64_t   fgMaxFullCacheSize; // maximal size of full-cached content, 500 MB by default

           void        Init(Bool_t readHeadOnly) override;
   virtual void        CheckProxy();
   virtual TString     BasicAuthentication();
   virtual Int_t       GetHead();
   virtual Int_t       GetLine(TSocket *s, char *line, Int_t maxsize);
   virtual Int_t       GetHunk(TSocket *s, char *hunk, Int_t maxsize);
   virtual const char *HttpTerminator(const char *start, const char *peeked, Int_t peeklen);
   virtual Int_t       GetFromWeb(char *buf, Int_t len, const TString &msg);
   virtual Int_t       GetFromWeb10(char *buf, Int_t len, const TString &msg, Int_t nseg = 0, Long64_t *seg_pos = 0, Int_t *seg_len = 0);
   virtual Int_t       GetFromCache(char *buf, Int_t len, Int_t nseg, Long64_t *seg_pos, Int_t *seg_len);
   virtual Bool_t      ReadBuffer10(char *buf, Int_t len);
   virtual Bool_t      ReadBuffers10(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf);
   virtual void        SetMsgReadBuffer10(const char *redirectLocation = 0, Bool_t tempRedirect = kFALSE);
   virtual void        ProcessHttpHeader(const TString& headerLine);

public:
   TWebFile(const char *url, Option_t *opt="");
   TWebFile(TUrl url, Option_t *opt="");
   virtual ~TWebFile();

   Long64_t    GetSize() const override;
   Bool_t      IsOpen() const override;
   Int_t       ReOpen(Option_t *mode) override;
   Bool_t      ReadBuffer(char *buf, Int_t len) override;
   Bool_t      ReadBuffer(char *buf, Long64_t pos, Int_t len) override;
   Bool_t      ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf) override;
   void        Seek(Long64_t offset, ERelativeTo pos = kBeg) override;

   static void        SetProxy(const char *url);
   static const char *GetProxy();

   static Long64_t    GetMaxFullCacheSize();
   static void        SetMaxFullCacheSize(Long64_t sz);

   ClassDefOverride(TWebFile,2)  //A ROOT file that reads via a http server
};


class TWebSystem : public TSystem {

private:
   void *fDirp;    // directory handler

   void *GetDirPtr() const override { return fDirp; }

public:
   TWebSystem();
   virtual ~TWebSystem() {}

   Int_t       MakeDirectory(const char *name) override;
   void       *OpenDirectory(const char *name) override;
   void        FreeDirectory(void *dirp) override;
   const char *GetDirEntry(void *dirp) override;
   Int_t       GetPathInfo(const char *path, FileStat_t &buf) override;
   Bool_t      AccessPathName(const char *path, EAccessMode mode) override;
   Int_t       Unlink(const char *path) override;

   ClassDefOverride(TWebSystem,0)  // Directory handler for HTTP (TWebFiles)
};

#endif
