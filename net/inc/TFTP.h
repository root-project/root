// @(#)root/net:$Name:  $:$Id: TFTP.h,v 1.5 2001/02/26 02:49:06 rdm Exp $
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

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_MessageTypes
#include "MessageTypes.h"
#endif


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
   Seek_t     fRestartAt;   // restart transmission at specified offset
   TString    fCurrentFile; // file currently being get or put
   TSocket   *fSocket;      //! connection to rootd
   Double_t   fBytesWrite;  // number of bytes sent
   Double_t   fBytesRead;   // number of bytes received

   TFTP(const TFTP &);              // not implemented
   void   operator=(const TFTP &);  // idem
   void   Init(const char *url, Int_t parallel, Int_t wsize);
   void   PrintError(const char *where, Int_t err) const;
   Int_t  Recv(Int_t &status, EMessageTypes &kind) const;
   void   SetMode(Int_t mode) { fMode = mode; }

   static Double_t fgBytesWrite;  //number of bytes sent by all TFTP objects
   static Double_t fgBytesRead;   //number of bytes received by all TFTP objects

public:
   enum {
      kDfltBlockSize  = 0x80000,   // 512KB
      kDfltWindowSize = 65535,     // default tcp buffer size
      kBinary         = 0,         // binary data transfer (default)
      kAscii          = 1          // ascii data transfer
   };

   TFTP(const char *url, Int_t parallel = 1, Int_t wsize = kDfltWindowSize);
   virtual ~TFTP();

   void   SetBlockSize(Int_t blockSize);
   Int_t  GetBlockSize() const { return fBlockSize; }
   void   SetRestartAt(Seek_t at) { fRestartAt = at; }
   Seek_t GetRestartAt() const { return fRestartAt; }
   Int_t  GetMode() const { return fMode; }

   Bool_t IsOpen() const { return fSocket ? kTRUE : kFALSE; }
   void   Print(Option_t *opt = "") const;

   Seek_t PutFile(const char *file, const char *remoteName = 0);
   Seek_t GetFile(const char *file, const char *localName = 0);
   Int_t  ChangeDirectory(const char *dir) const;
   Int_t  MakeDirectory(const char *dir) const;
   Int_t  DeleteDirectory(const char *dir) const;
   Int_t  ListDirectory(Option_t *cmd = "") const;
   Int_t  PrintDirectory() const;
   Int_t  RenameFile(const char *file1, const char *file2) const;
   Int_t  DeleteFile(const char *file) const;
   Int_t  ChangePermission(const char *file, Int_t mode) const;
   Int_t  Close();
   void   Binary() { SetMode(kBinary); }
   void   Ascii() { SetMode(kAscii); }

   // standard ftp equivalents...
   void put(const char *file, const char *remoteName = 0) { PutFile(file, remoteName); }
   void get(const char *file, const char *localName = 0) { GetFile(file, localName); }
   void cd(const char *dir) const { ChangeDirectory(dir); }
   void mkdir(const char *dir) const { MakeDirectory(dir); }
   void rmdir(const char *dir) const { DeleteDirectory(dir); }
   void ls(Option_t *cmd = "") const { ListDirectory(cmd); }
   void pwd() const { PrintDirectory(); }
   void mv(const char *file1, const char *file2) const { RenameFile(file1, file2); }
   void rm(const char *file) const { DeleteFile(file); }
   void chmod(const char *file, Int_t mode) const { ChangePermission(file, mode); }
   void bye() { Close(); }
   void bin() { Binary(); }
   void ascii() { Ascii(); }

   ClassDef(TFTP, 1)  // File Transfer Protocol class using rootd
};

#endif
