// @(#)root/net:$Id$
// Author: Fons Rademakers   13/02/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFTP
#define ROOT_TFTP

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFTP                                                                 //
//                                                                      //
// This class provides all infrastructure for a performant file         //
// transfer protocol. It works in conjuction with the rootd daemon      //
// and can use parallel sockets to improve performance over fat pipes.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TSystem.h"
#include "TString.h"
#include "MessageTypes.h"


class TSocket;


class TFTP : public TObject {

private:
   TString    fHost;        // FQDN of remote host
   TString    fUser;        // remote user
   Int_t      fPort;        // port to which to connect
   Int_t      fParallel;    // number of parallel sockets
   Int_t      fWindowSize;  // tcp window size used
   Int_t      fProtocol;    // rootd protocol level
   Int_t      fLastBlock;   // last block successfully transfered
   Int_t      fBlockSize;   // size of data buffer used to transfer
   Int_t      fMode;        // binary or ascii file transfer mode
   Long64_t   fRestartAt;   // restart transmission at specified offset
   TString    fCurrentFile; // file currently being get or put
   TSocket   *fSocket;      //! connection to rootd
   Long64_t   fBytesWrite;  // number of bytes sent
   Long64_t   fBytesRead;   // number of bytes received
   Bool_t     fDir;         // Indicates if a remote directory is open

   TFTP(): fHost(), fUser(), fPort(0), fParallel(0), fWindowSize(0),
      fProtocol(0), fLastBlock(0), fBlockSize(0), fMode(0),
      fRestartAt(0), fCurrentFile(), fSocket(0), fBytesWrite(0),
      fBytesRead(0), fDir(kFALSE) { }
   TFTP(const TFTP &);              // not implemented
   void   operator=(const TFTP &);  // idem
   void   Init(const char *url, Int_t parallel, Int_t wsize);
   void   PrintError(const char *where, Int_t err) const;
   Int_t  Recv(Int_t &status, EMessageTypes &kind) const;
   void   SetMode(Int_t mode) { fMode = mode; }

   static Long64_t fgBytesWrite;  //number of bytes sent by all TFTP objects
   static Long64_t fgBytesRead;   //number of bytes received by all TFTP objects

public:
   enum {
      kDfltBlockSize  = 0x80000,   // 512KB
      kDfltWindowSize = 65535,     // default tcp buffer size
      kBinary         = 0,         // binary data transfer (default)
      kAscii          = 1          // ascii data transfer
   };

   TFTP(const char *url, Int_t parallel = 1, Int_t wsize = kDfltWindowSize,
        TSocket *sock = nullptr);
   virtual ~TFTP();

   void     SetBlockSize(Int_t blockSize);
   Int_t    GetBlockSize() const { return fBlockSize; }
   void     SetRestartAt(Long64_t at) { fRestartAt = at; }
   Long64_t GetRestartAt() const { return fRestartAt; }
   Int_t    GetMode() const { return fMode; }

   Bool_t   IsOpen() const { return fSocket ? kTRUE : kFALSE; }
   void     Print(Option_t *opt = "") const override;

   Long64_t PutFile(const char *file, const char *remoteName = nullptr);
   Long64_t GetFile(const char *file, const char *localName = nullptr);

   Bool_t   AccessPathName(const char *path, EAccessMode mode = kFileExists,
                           Bool_t print = kFALSE);
   const char *GetDirEntry(Bool_t print = kFALSE);
   Int_t    GetPathInfo(const char *path, FileStat_t &buf, Bool_t print = kFALSE);
   Int_t    ChangeDirectory(const char *dir) const;
   Int_t    MakeDirectory(const char *dir, Bool_t print = kFALSE) const;
   Int_t    DeleteDirectory(const char *dir) const;
   Int_t    ListDirectory(Option_t *cmd = "") const;
   void     FreeDirectory(Bool_t print = kFALSE);
   Bool_t   OpenDirectory(const char *name, Bool_t print = kFALSE);
   Int_t    PrintDirectory() const;
   Int_t    RenameFile(const char *file1, const char *file2) const;
   Int_t    DeleteFile(const char *file) const;
   Int_t    ChangePermission(const char *file, Int_t mode) const;
   Int_t    Close();
   void     Binary() { SetMode(kBinary); }
   void     Ascii() { SetMode(kAscii); }
   TSocket *GetSocket() const { return fSocket; }

   // standard ftp equivalents...
   void put(const char *file, const char *remoteName = 0) { PutFile(file, remoteName); }
   void get(const char *file, const char *localName = 0) { GetFile(file, localName); }
   void cd(const char *dir) const { ChangeDirectory(dir); }
   void mkdir(const char *dir) const { MakeDirectory(dir); }
   void rmdir(const char *dir) const { DeleteDirectory(dir); }
   void ls(Option_t *cmd = "") const override { ListDirectory(cmd); }
   void pwd() const { PrintDirectory(); }
   void mv(const char *file1, const char *file2) const { RenameFile(file1, file2); }
   void rm(const char *file) const { DeleteFile(file); }
   void chmod(const char *file, Int_t mode) const { ChangePermission(file, mode); }
   void bye() { Close(); }
   void bin() { Binary(); }
   void ascii() { Ascii(); }

   ClassDefOverride(TFTP, 1)  // File Transfer Protocol class using rootd
};

#endif
