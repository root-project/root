// @(#)root/net:$Name:$:$Id:$
// Author: Fons Rademakers   13/02/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFTP                                                                 //
//                                                                      //
// This class provides all infrastructure for a performant file         //
// transfer protocol. It works in conjuction with the rootd daemon      //
// and can use parallel sockets to improve performance over fat pipes.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#ifndef WIN32
#   include <unistd.h>
#else
#   define ssize_t int
#   include <io.h>
#   include <sys/types.h>
#endif

#include "TFTP.h"
#include "TPSocket.h"
#include "TNetFile.h"
#include "TAuthenticate.h"
#include "TUrl.h"
#include "TStopwatch.h"
#include "TSystem.h"

#if defined(R__UNIX)
#define HAVE_MMAP
#endif

#ifdef HAVE_MMAP
#   include <unistd.h>
#   include <sys/mman.h>
#endif


Double_t TFTP::fgBytesWrite = 0;
Double_t TFTP::fgBytesRead  = 0;


ClassImp(TFTP)

//______________________________________________________________________________
TFTP::TFTP(const char *url, Int_t par, Int_t wsize)
{
   // Open connection to host specified by the url using par parallel sockets.
   // The url has the form: [root[s]://]host[:port].
   // If port is not specified the default rootd port (1094) will be used.
   // Using wsize one can specify the tcp window size. Normally this is not
   // needed when using parallel sockets.

   fSocket = 0;

   TString s = url;
   if (s.Contains("://")) {
      if (!s.BeginsWith("root")) {
         Error("TFTP", "url must be of the for \"[root[s]://]host[:port]\"");
         MakeZombie();
         return;
      }
   } else
      s = "root://" + s;

   Init(s, par, wsize);
}

//______________________________________________________________________________
void TFTP::Init(const char *surl, Int_t par, Int_t wsize)
{
   // Set up the actual connection.

   TAuthenticate *auth;
   EMessageTypes kind;
   Int_t sec;

   TUrl url(surl);

again:
   if (par > 1) {
      fSocket = new TPSocket(url.GetHost(), url.GetPort(), par, wsize);
      if (!fSocket->IsValid()) {
         Warning("TFTP", "can't open %d parallel connections to rootd on host %s at port %d",
                 par, url.GetHost(), url.GetPort());
         delete fSocket;
         par = 1;
         goto again;
      }

      // NoDelay is internally set by TPSocket

   } else {
      fSocket = new TSocket(url.GetHost(), url.GetPort(), wsize);
      if (!fSocket->IsValid()) {
         Error("TFTP", "can't open connection to rootd on host %s at port %d",
               url.GetHost(), url.GetPort());
         goto zombie;
      }

      // Set some socket options
      fSocket->SetOption(kNoDelay, 1);

      // Tell rootd we want non parallel connection
      fSocket->Send((Int_t) 0, (Int_t) 0);
   }

   // Get rootd protocol level
   fSocket->Send(kROOTD_PROTOCOL);
   Recv(fProtocol, kind);

   // Authenticate to remote rootd server
   sec = !strcmp(url.GetProtocol(), "roots") ?
         TAuthenticate::kSRP : TAuthenticate::kNormal;
   auth = new TAuthenticate(fSocket, url.GetHost(), "rootd", sec);
   if (!auth->Authenticate()) {
      if (sec == TAuthenticate::kSRP)
         Error("TFTP", "secure authentication failed for host %s", url.GetHost());
      else
         Error("TFTP", "authentication failed for host %s", url.GetHost());
      delete auth;
      goto zombie;
   }
   fUser = auth->GetUser();
   delete auth;

   fHost       = url.GetHost();
   fPort       = url.GetPort();
   fParallel   = par;
   fWindowSize = wsize;
   fLastBlock  = 0;
   fFileSize   = 0;
   fRestartAt  = 0;
   fBlockSize  = kDfltBlockSize;
   fBytesWrite = 0;
   fBytesRead  = 0;

   return;

zombie:
   MakeZombie();
   SafeDelete(fSocket);
}

//______________________________________________________________________________
TFTP::~TFTP()
{
   // TFTP dtor. Send close message and close socket.

   Close();
   SafeDelete(fSocket);
}

//______________________________________________________________________________
void TFTP::Print(Option_t *) const
{
   // Print some info about the FTP connection.

   Printf("Remote host:          %s [%d]", fHost.Data(), fPort);
   Printf("Remote user:          %s", fUser.Data());
   if (fParallel > 1)
      Printf("Parallel sockets:     %d", fParallel);
   Printf("TCP window size:      %d", fWindowSize);
   Printf("Rootd protocol:       %d", fProtocol);
   Printf("Transfer block size:  %d", fBlockSize);
   Printf("Bytes written:        %g", fBytesWrite);
   Printf("Bytes read:           %g", fBytesRead);
}

//______________________________________________________________________________
void TFTP::PrintError(const char *where, Int_t err) const
{
   // Print error string depending on error code.

   Error(where, gRootdErrStr[err]);
}

//______________________________________________________________________________
Int_t TFTP::Recv(Int_t &status, EMessageTypes &kind)
{
   // Return status from rootd server and message kind. Returns -1 in
   // case of error otherwise 8 (sizeof 2 words, status and kind).

   kind   = kROOTD_ERR;
   status = 0;

   if (!fSocket) return -1;

   Int_t what;
   Int_t n = fSocket->Recv(status, what);
   kind = (EMessageTypes) what;
   return n;
}

//______________________________________________________________________________
void TFTP::SetBlockSize(Int_t blockSize)
{
   // Make sure the block size is a power of two, with a minimum of 32768.

   if (blockSize < 32768) {
      fBlockSize = 32768;
      return;
   }

   int i;
   for (i = 0; i < int(sizeof(blockSize)*8); i++)
      if ((blockSize >> i) == 1)
         break;

   fBlockSize = 1 << i;
}

//______________________________________________________________________________
Int_t TFTP::PutFile(const char *file, const char *remoteName)
{
   // Transfer binary file to remote host. Returns number of bytes
   // sent or < 0 in case of error. Error -1 connection is still
   // open, error -2 connection has been closed.

   if (!IsOpen()) return -1;

#ifndef WIN32
   Int_t fd = open(file, O_RDONLY);
#else
   Int_t fd = open(file, O_RDONLY | O_BINARY);
#endif
   if (fd < 0) {
      Error("FilePut", "can't open %s", file);
      return -1;
   }

   Long_t id, size, flags, modtime;
   gSystem->GetPathInfo(file, &id, &size, &flags, &modtime);
   if (flags > 1) {
      Error("FilePut", "%s not a regular file (%ld)", file, flags);
      close(fd);
      return -1;
   }

   if (!remoteName)
      remoteName = file;

   // fRestartAt can be set to restart transmission at specific file offset

   if (fSocket->Send(Form("%s %d %ld %ld", remoteName, fBlockSize, size,
                     (Long_t)fRestartAt), kROOTD_PUTFILE) < 0) {
      Error("FilePut", "error sending kROOTD_PUTFILE command");
      close(fd);
      return -1;
   }

   Int_t         stat;
   EMessageTypes kind;

   if (Recv(stat, kind) < 0 || kind == kROOTD_ERR) {
      PrintError("FilePut", stat);
      close(fd);
      return -1;
   }

   TStopwatch timer;
   timer.Start();

   Seek_t pos = fRestartAt & ~(fBlockSize-1);
   Int_t skip = fRestartAt - pos;

#ifndef HAVE_MMAP
   char *buf = new char[fBlockSize];
   lseek(fd, pos, SEEK_SET);
#endif

   while (pos < size) {
      Seek_t left = size - pos;
      if (left > fBlockSize)
         left = fBlockSize;
#ifdef HAVE_MMAP
      char *buf = (char*) mmap(0, left, PROT_READ, MAP_FILE | MAP_SHARED, fd, pos);
      if (buf == (char *) -1) {
         Error("FilePut", "mmap of file %s failed", file);
         close(fd);
         return -1;
      }
#else
      Int_t siz;
      while ((siz = read(fd, buf, left)) < 0 && TSystem::GetErrno() == EINTR)
         TSystem::ResetErrno();
      if (siz < 0 || siz != left) {
         Error("FilePut", "error reading from file %s", file);
         // SendUrgent message to rootd to stop tranfer
         delete [] buf;
         close(fd);
         return -1;
      }
#endif

      if (fSocket->SendRaw(buf+skip, left-skip) < 0) {
         Error("FilePut", "error sending buffer");
         // Send urgent message to rootd to stop transfer
#ifdef HAVE_MMAP
         munmap(buf, left);
#else
         delete [] buf;
#endif
         close(fd);
         return -2;
      }

      fBytesWrite  += left-skip;
      fgBytesWrite += left-skip;

      pos += left;
      skip = 0;

#ifdef HAVE_MMAP
      munmap(buf, left);
#endif
   }

#ifndef HAVE_MMAP
   delete [] buf;
#endif

   close(fd);

   // get acknowlegdement from server that file was stored correctly
   if (Recv(stat, kind) < 0 || kind == kROOTD_ERR) {
      PrintError("FilePut", stat);
      close(fd);
      return -1;
   }

   // provide timing numbers
   //timer.

   return size - fRestartAt;
}

//______________________________________________________________________________
Int_t TFTP::GetFile(const char *file, const char *localName)
{
   return 0;
}

//______________________________________________________________________________
Int_t TFTP::ChangeDirectory(const char *dir)
{
   return 0;
}

//______________________________________________________________________________
Int_t TFTP::MakeDirectory(const char *dir)
{
   return 0;
}

//______________________________________________________________________________
Int_t TFTP::DeleteDirectory(const char *dir)
{
   return 0;
}

//______________________________________________________________________________
Int_t TFTP::ListDirectory()
{
   return 0;
}

//______________________________________________________________________________
Int_t TFTP::PrintDirectory()
{
   return 0;
}

//______________________________________________________________________________
Int_t TFTP::Rename(const char *file1, const char *file2)
{
   return 0;
}

//______________________________________________________________________________
Int_t TFTP::DeleteFile(const char *file)
{
   return 0;
}

//______________________________________________________________________________
Int_t TFTP::ChangeProtection(const char *file, Int_t mode)
{
   return 0;
}

//______________________________________________________________________________
Int_t TFTP::Close()
{
   // Close ftp connection.

   if (!IsOpen()) return -1;

   fSocket->Send(kROOTD_CLOSE);

   return 0;
}
