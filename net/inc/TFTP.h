// @(#)root/net:$Name:$:$Id:$
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
   Seek_t     fFileSize;    // size of file being transfered
   Seek_t     fRestartAt;   // restart transmission at specified offset
   TString    fCurrentFile; // file currently being get or put
   TSocket   *fSocket;      //! connection to rootd
   Double_t   fBytesWrite;  // number of bytes written
   Double_t   fBytesRead;   // number of bytes read

   TFTP(const TFTP &);              // not implemented
   void   operator=(const TFTP &);  // idem
   void   Init(const char *url, Int_t parallel, Int_t wsize);
   void   PrintError(const char *where, Int_t err) const;
   Int_t  Recv(Int_t &status, EMessageTypes &kind);

   static Double_t fgBytesWrite;  //number of bytes written by all TFTP objects
   static Double_t fgBytesRead;   //number of bytes read by all TFTP objects

public:
   enum {
      kDfltBlockSize  = 0x80000,   // 512KB
      kDfltWindowSize = 65535      // default tcp buffer size
   };

   TFTP(const char *url, Int_t parallel = 1, Int_t wsize = kDfltWindowSize);
   virtual ~TFTP();

   void  SetBlockSize(Int_t blockSize);
   Int_t GetBlockSize() const { return fBlockSize; }

   Bool_t IsOpen() const { return fSocket ? kTRUE : kFALSE; }
   void   Print(Option_t *opt = "") const;

   Int_t PutFile(const char *file, const char *remoteName = 0);
   Int_t GetFile(const char *file, const char *localName = 0);
   Int_t ChangeDirectory(const char *dir);
   Int_t MakeDirectory(const char *dir);
   Int_t DeleteDirectory(const char *dir);
   Int_t ListDirectory();
   Int_t PrintDirectory();
   Int_t Rename(const char *file1, const char *file2);
   Int_t DeleteFile(const char *file);
   Int_t ChangeProtection(const char *file, Int_t mode);
   Int_t Close();

   // standard ftp equivalents...
   Int_t put(const char *file, const char *remoteName = 0) { return PutFile(file, remoteName); }
   Int_t get(const char *file, const char *localName = 0) { return GetFile(file, localName); }
   Int_t cd(const char *dir) { return ChangeDirectory(dir); }
   Int_t mkdir(const char *dir) { return MakeDirectory(dir); }
   Int_t rmdir(const char *dir) { return DeleteDirectory(dir); }
   Int_t ls() { return ListDirectory(); }
   Int_t pwd() { return PrintDirectory(); }
   Int_t rename(const char *file1, const char *file2) { return Rename(file1, file2); }
   Int_t rm(const char *file) { return DeleteFile(file); }
   Int_t chmod(const char *file, Int_t mode) { return ChangeProtection(file, mode); }
   Int_t bye() { return Close(); }

   ClassDef(TFTP, 1)  // File Transfer Protocol class using rootd
};

#endif
