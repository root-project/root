// @(#)root/net:$Id$
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

#include "TFile.h"
#include "TUrl.h"
#include "TFTP.h"
#include "TSystem.h"
#include "MessageTypes.h"

class TSocket;


class TNetFile : public TFile {

protected:
   TUrl      fEndpointUrl; //URL of realfile (after possible redirection)
   TString   fUser;        //remote user name
   TSocket  *fSocket;      //connection to rootd server
   Int_t     fProtocol;    //rootd protocol level
   Int_t     fErrorCode;   //error code returned by rootd (matching gRootdErrStr)
   Int_t     fNetopt;      //initial network options (used for ReOpen())

   TNetFile(const TNetFile&);             // NetFile cannot be copied
   TNetFile& operator=(const TNetFile&);  // NetFile cannot be copied

   TNetFile(const char *url, const char *ftitle, Int_t comp, Bool_t);
   virtual void ConnectServer(Int_t *stat, EMessageTypes *kind, Int_t netopt,
                              Int_t tcpwindowsize, Bool_t forceOpen,
                              Bool_t forceRead);
   virtual void Create(const char *url, Option_t *option, Int_t netopt);
   virtual void Create(TSocket *s, Option_t *option, Int_t netopt);
   void         Init(Bool_t create) override;
   void         Print(Option_t *option) const override;
   void         PrintError(const char *where, Int_t err);
   Int_t        Recv(Int_t &status, EMessageTypes &kind);
   Int_t        SysOpen(const char *pathname, Int_t flags, UInt_t mode) override;
   Int_t        SysClose(Int_t fd) override;
   Int_t        SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime) override;

public:
   TNetFile(const char *url, Option_t *option = "", const char *ftitle = "",
            Int_t compress = ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault, Int_t netopt = 0);
   TNetFile() : fEndpointUrl(), fUser(), fSocket(nullptr), fProtocol(0), fErrorCode(0), fNetopt(0) { }
   virtual ~TNetFile();

   void    Close(Option_t *option="") override;  // *MENU*
   void    Flush() override;
   Int_t   GetErrorCode() const { return fErrorCode; }
   Bool_t  IsOpen() const override;
   Bool_t  Matches(const char *url) override;
   Int_t   ReOpen(Option_t *mode) override;
   Bool_t  ReadBuffer(char *buf, Int_t len) override;
   Bool_t  ReadBuffer(char *buf, Long64_t pos, Int_t len) override;
   Bool_t  ReadBuffers(char *buf,  Long64_t *pos, Int_t *len, Int_t nbuf) override;
   Bool_t  WriteBuffer(const char *buf, Int_t len) override;
   void    Seek(Long64_t offset, ERelativeTo pos = kBeg) override;

   const TUrl *GetEndpointUrl() const override { return &fEndpointUrl; }

   ClassDefOverride(TNetFile,1)  //A ROOT file that reads/writes via a rootd server
};


class TNetSystem : public TSystem {

private:
   Bool_t      fDir;         // true if a directory is open remotely
   void       *fDirp;        // directory handler
   TFTP       *fFTP;         // Connection to rootd
   TString     fHost;        // Remote host
   Bool_t      fFTPOwner;    // True if owner of the FTP instance
   TString     fUser;        // Remote user
   Int_t       fPort;        // Remote port

   TNetSystem(const TNetSystem&) = delete;
   TNetSystem& operator=(const TNetSystem&) = delete;

   void       *GetDirPtr() const override { return fDirp; }

protected:
   Bool_t      fIsLocal;     // TRUE if the path points to this host
   TString     fLocalPrefix; // if fIsLocal, prefix to be prepend locally

   void        Create(const char *url, TSocket *sock = nullptr);
   void        InitRemoteEntity(const char *url);

public:
   TNetSystem(Bool_t ftpowner = kTRUE);
   TNetSystem(const char *url, Bool_t ftpowner = kTRUE);
   virtual ~TNetSystem();

   Bool_t      ConsistentWith(const char *path, void *dirptr) override;
   Int_t       MakeDirectory(const char *name) override;
   void       *OpenDirectory(const char *name) override;
   void        FreeDirectory(void *dirp = nullptr) override;
   const char *GetDirEntry(void *dirp = nullptr) override;
   Int_t       GetPathInfo(const char *path, FileStat_t &buf) override;
   Bool_t      AccessPathName(const char *path, EAccessMode mode) override;
   int         Unlink(const char *path) override;

   ClassDefOverride(TNetSystem,0)  // Directory handler for NetSystem
};

#endif
