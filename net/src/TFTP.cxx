// @(#)root/net:$Name:  $:$Id: TFTP.cxx,v 1.9 2001/03/05 15:27:26 rdm Exp $
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
#include "TError.h"

#if defined(R__UNIX)
#define HAVE_MMAP
#endif

#ifdef HAVE_MMAP
#   include <unistd.h>
#   include <sys/mman.h>
#ifndef MAP_FILE
#define MAP_FILE 0           /* compatability flag */
#endif
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
         Error("TFTP", "url must be of the form \"[root[s]://]host[:port]\"");
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
   fRestartAt  = 0;
   fBlockSize  = kDfltBlockSize;
   fMode       = kBinary;
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
   Printf("Transfer mode:        %s", fMode ? "ascii" : "binary");
   Printf("Bytes sent:           %g", fBytesWrite);
   Printf("Bytes received:       %g", fBytesRead);
}

//______________________________________________________________________________
void TFTP::PrintError(const char *where, Int_t err) const
{
   // Print error string depending on error code.

   Error(where, gRootdErrStr[err]);
}

//______________________________________________________________________________
Int_t TFTP::Recv(Int_t &status, EMessageTypes &kind) const
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
Seek_t TFTP::PutFile(const char *file, const char *remoteName)
{
   // Transfer file to remote host. Returns number of bytes
   // sent or < 0 in case of error. Error -1 connection is still
   // open, error -2 connection has been closed. In case of failure
   // fRestartAt is set to the number of bytes correclty transfered.
   // Calling PutFile() immediately afterwards will restart at fRestartAt.
   // If this is not desired call SetRestartAt(0) before calling PutFile().
   // If rootd reports that the file is locked, and you are sure this is not
   // the case (e.g. due to a crash), you can force unlock it by prepending
   // the remoteName with a '-'.

   if (!IsOpen() || !file || !*file) return -1;

#ifndef WIN32
   Int_t fd = open(file, O_RDONLY);
#else
   Int_t fd = open(file, O_RDONLY | O_BINARY);
#endif
   if (fd < 0) {
      Error("PutFile", "cannot open %s", file);
      return -1;
   }

   Long_t id, size, flags, modtime;
   if (gSystem->GetPathInfo(file, &id, &size, &flags, &modtime) == 0) {
      if (flags > 1) {
         Error("PutFile", "%s not a regular file (%ld)", file, flags);
         close(fd);
         return -1;
      }
   } else
      Warning("PutFile", "could not stat file, assuming it is a regular file");

   if (!remoteName)
      remoteName = file;

   Long_t restartat = fRestartAt;

   // check if restartat value makes sense
   if (restartat && (restartat >= size))
      restartat = 0;

   if (fSocket->Send(Form("%s %d %d %ld %ld", remoteName, fBlockSize, fMode,
                     size, restartat), kROOTD_PUTFILE) < 0) {
      Error("PutFile", "error sending kROOTD_PUTFILE command");
      close(fd);
      return -2;
   }

   Int_t         stat;
   EMessageTypes kind;

   if (Recv(stat, kind) < 0 || kind == kROOTD_ERR) {
      PrintError("PutFile", stat);
      close(fd);
      return -1;
   }

   Printf("<TFTP::PutFile>: sending file %s (%ld bytes, starting at %ld)",
          file, size, restartat);

   TStopwatch timer;
   timer.Start();

   Seek_t pos = restartat & ~(fBlockSize-1);
   Int_t skip = restartat - pos;

#ifndef HAVE_MMAP
   char *buf = new char[fBlockSize];
   lseek(fd, pos, SEEK_SET);
#endif

   while (pos < size) {
      Seek_t left = Seek_t(size - pos);
      if (left > fBlockSize)
         left = fBlockSize;
#ifdef HAVE_MMAP
      char *buf = (char*) mmap(0, left, PROT_READ, MAP_FILE | MAP_SHARED, fd, pos);
      if (buf == (char *) -1) {
         Error("PutFile", "mmap of file %s failed", file);
         close(fd);
         return -1;
      }
#else
      Int_t siz;
      while ((siz = read(fd, buf, left)) < 0 && TSystem::GetErrno() == EINTR)
         TSystem::ResetErrno();
      if (siz < 0 || siz != left) {
         Error("PutFile", "error reading from file %s", file);
         // Send urgent message to rootd to stop tranfer
         delete [] buf;
         close(fd);
         return -1;
      }
#endif

      if (fSocket->SendRaw(buf+skip, left-skip) < 0) {
         Error("PutFile", "error sending buffer");
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

      fRestartAt = pos;   // bytes correctly sent up till now

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

   fRestartAt = 0;

   // get acknowlegdement from server that file was stored correctly
   if (Recv(stat, kind) < 0 || kind == kROOTD_ERR) {
      PrintError("PutFile", stat);
      close(fd);
      return -1;
   }

   // provide timing numbers
   Double_t speed, t = timer.RealTime();
   if (t > 0)
      speed = (size - restartat) / t;
   else
      speed = 0.0;
   if (speed > 524288)
      Printf("<TFTP::PutFile>: %.3f seconds, %.2f Mbytes per second",
             t, speed / 1048576);
   else if (speed > 512)
      Printf("<TFTP::PutFile>: %.3f seconds, %.2f Kbytes per second",
             t, speed / 1024);
   else
      Printf("<TFTP::PutFile>: %.3f seconds, %.2f bytes per second",
             t, speed);

   return Seek_t(size - restartat);
}

//______________________________________________________________________________
Seek_t TFTP::GetFile(const char *file, const char *localName)
{
   // Transfer file from remote host. Returns number of bytes
   // received or < 0 in case of error. Error -1 connection is still
   // open, error -2 connection has been closed. In case of failure
   // fRestartAt is set to the number of bytes correclty transfered.
   // Calling GetFile() immediately afterwards will restart at fRestartAt.
   // If this is not desired call SetRestartAt(0) before calling GetFile().
   // If rootd reports that the file is locked, and you are sure this is not
   // the case (e.g. due to a crash), you can force unlock it by prepending
   // the file name with a '-'.

   if (!IsOpen() || !file || !*file) return -1;

   if (!localName) {
      if (file[0] == '-')
         localName = file+1;
      else
         localName = file;
   }

   Long_t restartat = fRestartAt;

   if (fSocket->Send(Form("%s %d %d %ld", file, fBlockSize, fMode,
                     restartat), kROOTD_GETFILE) < 0) {
      Error("GetFile", "error sending kROOTD_GETFILE command");
      return -2;
   }

   Int_t         stat;
   EMessageTypes kind;

   if (Recv(stat, kind) < 0 || kind == kROOTD_ERR) {
      PrintError("GetFile", stat);
      return -1;
   }

   // get size of remote file
   Long_t sizel;
   Int_t  what;
   char   mess[64];

   if (fSocket->Recv(mess, sizeof(mess), what) < 0) {
      Error("GetFile", "error receiving remote file size");
      return -2;
   }
   sscanf(mess, "%ld", &sizel);
   Seek_t size = (Seek_t) sizel;

   // check if restartat value makes sense
   if (restartat && (restartat >= size))
      restartat = 0;

   // open local file
   Int_t fd;
   if (!restartat) {
#ifndef WIN32
      fd = open(localName, O_CREAT | O_TRUNC | O_WRONLY, 0600);
#else
      if (fMode == kBinary)
         fd = open(localName, O_CREAT | O_TRUNC | O_WRONLY | O_BINARY,
                   S_IREAD | S_IWRITE);
      else
         fd = open(localName, O_CREAT | O_TRUNC | O_WRONLY,
                   S_IREAD | S_IWRITE);
#endif
   } else {
#ifndef WIN32
      fd = open(localName, O_WRONLY, 0600);
#else
      if (fMode == kBinary)
         fd = open(localName, O_WRONLY | O_BINARY, S_IREAD | S_IWRITE);
      else
         fd = open(localName, O_WRONLY, S_IREAD | S_IWRITE);
#endif
   }

   if (fd < 0) {
      Error("GetFile", "cannot open %s", localName);
      // send urgent message to rootd to stop tranfer
      return -1;
   }

   // check file system space
   if (strcmp(localName, "/dev/null")) {
      Long_t id, bsize, blocks, bfree;
      if (gSystem->GetFsInfo(localName, &id, &bsize, &blocks, &bfree) == 0) {
         Double_t space = (Double_t)bsize * (Double_t)bfree;
         if (space < size - restartat) {
            Error("GetFile", "not enough space to store file %s", localName);
            // send urgent message to rootd to stop tranfer
            return -1;
         }
      } else
         Warning("GetFile", "could not determine if there is enough free space to store file");
   }

   // seek to restartat position
   if (restartat) {
      if (lseek(fd, (off_t) restartat, SEEK_SET) < 0) {
         Error("GetFile", "cannot seek to position %ld in file %s",
               restartat, localName);
         // if cannot seek send urgent message to rootd to stop tranfer
         close(fd);
         return -1;
      }
   }

   Printf("<TFTP::GetFile>: getting file %s (%ld bytes, starting at %ld)",
          localName, sizel, restartat);

   TStopwatch timer;
   timer.Start();

   char *buf = new char[fBlockSize];
   char *buf2 = 0;
   if (fMode == kAscii)
      buf2 = new char[fBlockSize];

   Seek_t pos = restartat & ~(fBlockSize-1);
   Int_t skip = restartat - pos;

   while (pos < size) {
      Seek_t left = size - pos;
      if (left > fBlockSize)
         left = fBlockSize;

      Int_t n;
      while ((n = fSocket->RecvRaw(buf, Int_t(left-skip))) < 0 &&
             TSystem::GetErrno() == EINTR)
         TSystem::ResetErrno();

      if (n != Int_t(left-skip)) {
         Error("GetFile", "error receiving buffer of length %d, got %d",
               Int_t(left-skip), n);
         close(fd);
         delete [] buf; delete [] buf2;
         return -2;
      }

      // in case of ascii file, loop over buffer and remove \r's
      ssize_t siz;
      if (fMode == kAscii) {
         Int_t i = 0, j = 0;
         while (i < n) {
            if (buf[i] == '\r')
               i++;
            else
               buf2[j++] = buf[i++];
         }
         n = j;
         while ((siz = write(fd, buf2, n)) < 0 && TSystem::GetErrno() == EINTR)
            TSystem::ResetErrno();
      } else {
         while ((siz = write(fd, buf, n)) < 0 && TSystem::GetErrno() == EINTR)
            TSystem::ResetErrno();
      }

      if (siz < 0) {
         SysError("GetFile", "error writing file %s", localName);
         // send urgent message to rootd to stop tranfer
         close(fd);
         delete [] buf; delete [] buf2;
         return -1;
      }

      if (siz != n) {
         Error("GetFile", "error writing all requested bytes to file %s, wrote %d of %d",
               localName, siz, n);
         // send urgent message to rootd to stop tranfer
         close(fd);
         delete [] buf; delete [] buf2;
         return -1;
      }

      fBytesRead  += Int_t(left-skip);
      fgBytesRead += Int_t(left-skip);

      fRestartAt = pos;   // bytes correctly received up till now

      pos += left;
      skip = 0;
   }

   delete [] buf; delete [] buf2;

#ifndef WIN32
   fchmod(fd, 0644);
#endif

   close(fd);

   fRestartAt = 0;

   // provide timing numbers
   Double_t speed, t = timer.RealTime();
   if (t > 0)
      speed = (size - restartat) / t;
   else
      speed = 0.0;
   if (speed > 524288)
      Printf("<TFTP::GetFile>: %.3f seconds, %.2f Mbytes per second",
             t, speed / 1048576);
   else if (speed > 512)
      Printf("<TFTP::GetFile>: %.3f seconds, %.2f Kbytes per second",
             t, speed / 1024);
   else
      Printf("<TFTP::GetFile>: %.3f seconds, %.2f bytes per second",
             t, speed);

   return Seek_t(size - restartat);
}

//______________________________________________________________________________
Int_t TFTP::ChangeDirectory(const char *dir) const
{
   // Change the remote directory. If the remote directory contains a .message
   // file and it is < 1024 characters then the contents is echoed back.
   // Returns 0 in case of success and -1 in case of failure.

   if (!IsOpen()) return -1;

   if (!dir || !*dir) {
      Error("ChangeDirectory", "illegal directory name specified");
      return -1;
   }

   if (fSocket->Send(Form("%s", dir), kROOTD_CHDIR) < 0) {
      Error("ChangeDirectory", "error sending kROOTD_CHDIR command");
      return -1;
   }

   Int_t what;
   char  mess[1024];

   if (fSocket->Recv(mess, sizeof(mess), what) < 0) {
      Error("ChangeDirectory", "error receiving chdir confirmation");
      return -1;
   }
   if (what == kMESS_STRING) {
      Printf("%s\n", mess);

      if (fSocket->Recv(mess, sizeof(mess), what) < 0) {
         Error("ChangeDirectory", "error receiving chdir confirmation");
         return -1;
      }
   }

   Printf("<TFTP::ChangeDirectory>: %s", mess);

   return 0;
}

//______________________________________________________________________________
Int_t TFTP::MakeDirectory(const char *dir) const
{
   // Make a remote directory. Anonymous users may not create directories.
   // Returns 0 in case of success and -1 in case of failure.

   if (!IsOpen()) return -1;

   if (!dir || !*dir) {
      Error("MakeDirectory", "illegal directory name specified");
      return -1;
   }

   if (fSocket->Send(Form("%s", dir), kROOTD_MKDIR) < 0) {
      Error("MakeDirectory", "error sending kROOTD_MKDIR command");
      return -1;
   }

   Int_t what;
   char  mess[1024];;

   if (fSocket->Recv(mess, sizeof(mess), what) < 0) {
      Error("MakeDirectory", "error receiving mkdir confirmation");
      return -1;
   }

   Printf("<TFTP::MakeDirectory>: %s", mess);

   return 0;
}

//______________________________________________________________________________
Int_t TFTP::DeleteDirectory(const char *dir) const
{
   // Delete a remote directory. Anonymous users may not delete directories.
   // Returns 0 in case of success and -1 in case of failure.

   if (!IsOpen()) return -1;

   if (!dir || !*dir) {
      Error("DeleteDirectory", "illegal directory name specified");
      return -1;
   }

   if (fSocket->Send(Form("%s", dir), kROOTD_RMDIR) < 0) {
      Error("DeleteDirectory", "error sending kROOTD_RMDIR command");
      return -1;
   }

   Int_t what;
   char  mess[1024];;

   if (fSocket->Recv(mess, sizeof(mess), what) < 0) {
      Error("DeleteDirectory", "error receiving rmdir confirmation");
      return -1;
   }

   Printf("<TFTP::DeleteDirectory>: %s", mess);

   return 0;
}

//______________________________________________________________________________
Int_t TFTP::ListDirectory(Option_t *cmd) const
{
   // List remote directory. With cmd you specify the options and directory
   // to be listed to ls. Returns 0 in case of success and -1 in case of
   // failure.

   if (!IsOpen()) return -1;

   if (!cmd || !*cmd)
      cmd = "ls .";

   if (fSocket->Send(Form("%s", cmd), kROOTD_LSDIR) < 0) {
      Error("ListDirectory", "error sending kROOTD_LSDIR command");
      return -1;
   }

   Int_t what;
   char  mess[1024];;

   do {
      if (fSocket->Recv(mess, sizeof(mess), what) < 0) {
         Error("ListDirectory", "error receiving lsdir confirmation");
         return -1;
      }
      printf("%s", mess);
   } while (what == kMESS_STRING);

   return 0;
}

//______________________________________________________________________________
Int_t TFTP::PrintDirectory() const
{
   // Print path of remote working directory. Returns 0 in case of succes and
   // -1 in cse of failure.

   if (!IsOpen()) return -1;

   if (fSocket->Send("", kROOTD_PWD) < 0) {
      Error("DeleteDirectory", "error sending kROOTD_PWD command");
      return -1;
   }

   Int_t what;
   char  mess[1024];;

   if (fSocket->Recv(mess, sizeof(mess), what) < 0) {
      Error("PrintDirectory", "error receiving pwd confirmation");
      return -1;
   }

   Printf("<TFTP::PrintDirectory>: %s", mess);

   return 0;
}

//______________________________________________________________________________
Int_t TFTP::RenameFile(const char *file1, const char *file2) const
{
   // Rename a remote file. Anonymous users may not rename files.
   // Returns 0 in case of success and -1 in case of failure.

   if (!IsOpen()) return -1;

   if (!file1 || !file2 || !*file1 || !*file2) {
      Error("RenameFile", "illegal file names specified");
      return -1;
   }

   if (fSocket->Send(Form("%s %s", file1, file2), kROOTD_MV) < 0) {
      Error("RenameFile", "error sending kROOTD_MV command");
      return -1;
   }

   Int_t what;
   char  mess[1024];;

   if (fSocket->Recv(mess, sizeof(mess), what) < 0) {
      Error("RenameFile", "error receiving mv confirmation");
      return -1;
   }

   Printf("<TFTP::RenameFile>: %s", mess);

   return 0;
}

//______________________________________________________________________________
Int_t TFTP::DeleteFile(const char *file) const
{
   // Delete a remote file. Anonymous users may not delete files.
   // Returns 0 in case of success and -1 in case of failure.

   if (!IsOpen()) return -1;

   if (!file || !*file) {
      Error("DeleteFile", "illegal file name specified");
      return -1;
   }

   if (fSocket->Send(Form("%s", file), kROOTD_RM) < 0) {
      Error("DeleteFile", "error sending kROOTD_RM command");
      return -1;
   }

   Int_t what;
   char  mess[1024];;

   if (fSocket->Recv(mess, sizeof(mess), what) < 0) {
      Error("DeleteFile", "error receiving rm confirmation");
      return -1;
   }

   Printf("<TFTP::DeleteFile>: %s", mess);

   return 0;
}

//______________________________________________________________________________
Int_t TFTP::ChangePermission(const char *file, Int_t mode) const
{
   // Change permissions of a remote file. Anonymous users may not
   // chnage permissions. Returns 0 in case of success and -1 in case
   // of failure.

   if (!IsOpen()) return -1;

   if (!file || !*file) {
      Error("ChangePermission", "illegal file name specified");
      return -1;
   }

   if (fSocket->Send(Form("%s %d", file, mode), kROOTD_CHMOD) < 0) {
      Error("ChangePermission", "error sending kROOTD_CHMOD command");
      return -1;
   }

   Int_t what;
   char  mess[1024];;

   if (fSocket->Recv(mess, sizeof(mess), what) < 0) {
      Error("ChangePermission", "error receiving chmod confirmation");
      return -1;
   }

   Printf("<TFTP::ChangePermission>: %s", mess);

   return 0;
}

//______________________________________________________________________________
Int_t TFTP::Close()
{
   // Close ftp connection. Returns 0 in case of success and -1 in case of
   // failure.

   if (!IsOpen()) return -1;

   if (fSocket->Send(kROOTD_CLOSE) < 0) {
      Error("Close", "error sending kROOTD_CLOSE command");
      return -1;
   }

   return 0;
}
