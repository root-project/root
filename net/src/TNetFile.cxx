// @(#)root/net:$Name:  $:$Id: TNetFile.cxx,v 1.42 2003/12/10 11:03:40 rdm Exp $
// Author: Fons Rademakers   14/08/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNetFile                                                             //
//                                                                      //
// A TNetFile is like a normal TFile except that it reads and writes    //
// its data via a rootd server (for more on the rootd daemon see the    //
// source files root/rootd/src/*.cxx). TNetFile file names are in       //
// standard URL format with protocol "root" or "roots". The following   //
// are valid TNetFile URL's:                                            //
//                                                                      //
//    roots://hpsalo/files/aap.root                                     //
//    root://hpbrun.cern.ch/root/hsimple.root                           //
//    root://pcna49a:5151/~na49/data/run821.root                        //
//    root://pcna49d.cern.ch:5050//v1/data/run810.root                  //
//                                                                      //
// The only difference with the well known httpd URL's is that the root //
// of the remote file tree is the user's home directory. Therefore an   //
// absolute pathname requires a // after the host or port specifier     //
// (see last example). Further the expansion of the standard shell      //
// characters, like ~, $, .., are handled as expected.                  //
// TNetFile (actually TUrl) uses 1094 as default port for rootd.        //
//                                                                      //
// Connecting to a rootd requires the remote user id and password.      //
// TNetFile allows three ways for you to provide your login:            //
//   1) Setting it globally via the static functions:                   //
//         TAuthenticate::SetGlobalUser() and                           //
//         TAuthenticate::SetGlobalPasswd()                             //
//   2) Getting it from the ~/.netrc file (same file as used by ftp)    //
//   3) Command line prompt                                             //
// The different methods will be tried in the order given above.        //
// On machines with AFS rootd will authenticate using AFS (if it was    //
// compiled with AFS support).                                          //
//                                                                      //
// If the protocol is specified as "roots" a secure authetication       //
// method will be used. The secure method uses the SRP, Secure Remote   //
// Passwords, package. SRP uses a so called "asymmetric key exchange    //
// protocol" in which no passwords are ever send over the wire. This    //
// protocol is safe against all known security attacks. For more see:   //
// Begin_Html <a href=http://root.cern.ch/root/NetFile.html>NetFile</a> //
// End_Html                                                             //
// If the protocol is specified as "rootk" kerberos5 will be used for   //
// authentication.
//                                                                      //
// The rootd daemon lives in the directory $ROOTSYS/bin. It can be      //
// started either via inetd or by hand from the command line (no need   //
// to be super user). For more info about rootd see the web page:       //
// Begin_Html <a href=http://root.cern.ch/root/NetFile.html>NetFile</a> //
// End_Html                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <errno.h>

#include "TNetFile.h"
#include "TAuthenticate.h"
#include "TROOT.h"
#include "TPSocket.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TSysEvtHandler.h"
#include "TEnv.h"
#include "Bytes.h"
#include "NetErrors.h"

// Protocol changes:
// 6 -> 7: added support for ReOpen(), kROOTD_BYE and kROOTD_PROTOCOL2
// 7 -> 8: added support for update being a create (open stat = 2 and not 1)
// 8 -> 9: added new authentication features (see README.AUTH)
Int_t TNetFile::fgClientProtocol = 9;  // increase when client protocol changes


ClassImp(TNetFile)

//______________________________________________________________________________
TNetFile::TNetFile(const char *url, Option_t *option, const char *ftitle,
                   Int_t compress, Int_t netopt)
         : TFile(url, "NET", ftitle, compress), fUrl(url)
{
   // Create a TNetFile object. This is actually done inside Create(), so
   // for a description of the options and other arguments see Create().
   // Normally a TNetFile is created via TFile::Open().

   Create(url, option, netopt);
}
//______________________________________________________________________________
TNetFile::TNetFile(const char *url, const char *ftitle, Int_t compress, Bool_t)
         : TFile(url, "NET", ftitle, compress), fUrl(url)
{
   // Create a TNetFile object. To be used by derived classes, that need
   // to initialize the TFile base class but not open a connection at this
   // moment.

   fSocket = 0;
}

//______________________________________________________________________________
TNetFile::~TNetFile()
{
   // TNetFile dtor. Send close message and close socket.

   Close();
   SafeDelete(fSocket);
}

//______________________________________________________________________________
Int_t TNetFile::SysOpen(const char * /*file*/, Int_t /*flags*/, UInt_t /*mode*/)
{
   // Open a remote file. Requires fOption to be set correctly.

   if (!fSocket) return -1;

   fSocket->Send(Form("%s %s", fUrl.GetFile(), ToLower(fOption).Data()),
                 kROOTD_OPEN);

   EMessageTypes kind;
   int stat;
   Recv(stat, kind);

   if (kind == kROOTD_ERR) {
      PrintError("SysOpen", stat);
      return -1;
   }

   return -2;
}

//______________________________________________________________________________
Int_t TNetFile::SysClose(Int_t /*fd*/)
{
   // Close currently open file.

   if (fSocket)
      fSocket->Send(kROOTD_CLOSE);

   return 0;
}

//______________________________________________________________________________
Int_t TNetFile::SysStat(Int_t, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime)
{
   // Return file stat information. The interface and return value is
   // identical to TSystem::GetPathInfo().

   if (fProtocol < 3) return 1;

   if (!fSocket) return 1;

   fSocket->Send(kROOTD_FSTAT);

   char  msg[128];
   Int_t kind;
   fSocket->Recv(msg, 128, kind);

   sscanf(msg, "%ld %lld %ld %ld", id, size, flags, modtime);

   if (*id == -1)
      return 1;

   return 0;
}

//______________________________________________________________________________
void TNetFile::Close(Option_t *opt)
{
   // Close remote file.

   if (!fSocket) return;

   TFile::Close(opt);

   if (fProtocol > 6)
      fSocket->Send(kROOTD_BYE);
}

//______________________________________________________________________________
void TNetFile::Flush()
{
   // Flush file to disk.

   if (fSocket && fWritable)
      fSocket->Send(kROOTD_FLUSH);
}

//______________________________________________________________________________
void TNetFile::Init(Bool_t create)
{
   // Initialize a TNetFile object.

   Seek(0);

   TFile::Init(create);
   fD = -2;   // so TFile::IsOpen() will return true when in TFile::~TFile
}

//______________________________________________________________________________
Bool_t TNetFile::IsOpen() const
{
   // Retruns kTRUE if file is open, kFALSE otherwise.

   return fSocket == 0 ? kFALSE : kTRUE;
}

//______________________________________________________________________________
void TNetFile::Print(Option_t *) const
{
   // Print some info about the net file.

   const char *fname = fUrl.GetFile();
   Printf("URL:           %s", ((TUrl*)&fUrl)->GetUrl());
   Printf("Remote file:   %s", &fname[1]);
   Printf("Remote user:   %s", fUser.Data());
   Printf("Title:         %s", fTitle.Data());
   Printf("Option:        %s", fOption.Data());
   Printf("Bytes written: %g", fBytesWrite);
   Printf("Bytes read:    %g", fBytesRead);
}

//______________________________________________________________________________
void TNetFile::PrintError(const char *where, Int_t err)
{
   // Print error string depending on error code.

   fErrorCode = err;
   Error(where, gRootdErrStr[err]);
}

//______________________________________________________________________________
Int_t TNetFile::ReOpen(Option_t *mode)
{
   // Reopen a file with a different access mode, like from READ to
   // UPDATE or from NEW, CREATE, RECREATE, UPDATE to READ. Thus the
   // mode argument can be either "READ" or "UPDATE". The method returns
   // 0 in case the mode was successfully modified, 1 in case the mode
   // did not change (was already as requested or wrong input arguments)
   // and -1 in case of failure, in which case the file cannot be used
   // anymore.

   if (fProtocol < 7) {
      Error("ReOpen", "operation not supported by remote rootd (protocol = %d)",
            fProtocol);
      return 1;
   }

   return TFile::ReOpen(mode);
}

//______________________________________________________________________________
Bool_t TNetFile::ReadBuffer(char *buf, Int_t len)
{
   // Read specified byte range from remote file via rootd daemon.
   // Returns kTRUE in case of error.

   if (!fSocket) return kTRUE;

   Bool_t result = kFALSE;

   if (fCache) {
      Int_t st;
      Long64_t off = fOffset;
      if ((st = fCache->ReadBuffer(fOffset, buf, len)) < 0) {
         Error("ReadBuffer", "error reading from cache");
         return kTRUE;
      }
      if (st > 0) {
         // fOffset might have been changed via TCache::ReadBuffer(), reset it
         Seek(off + len);
         return result;
      }
   }

   if (gApplication && gApplication->GetSignalHandler())
      gApplication->GetSignalHandler()->Delay();

   if (fSocket->Send(Form("%ld %d", (Long_t)fOffset, len), kROOTD_GET) < 0) {
      Error("ReadBuffer", "error sending kROOTD_GET command");
      result = kTRUE;
      goto end;
   }

   Int_t         stat, n;
   EMessageTypes kind;

   fErrorCode = -1;
   if (Recv(stat, kind) < 0 || kind == kROOTD_ERR) {
      PrintError("ReadBuffer", stat);
      result = kTRUE;
      goto end;
   }

   while ((n = fSocket->RecvRaw(buf, len)) < 0 && TSystem::GetErrno() == EINTR)
      TSystem::ResetErrno();

   if (n != len) {
      Error("ReadBuffer", "error receiving buffer of length %d, got %d", len, n);
      result = kTRUE;
      goto end;
   }

   fOffset += len;

   fBytesRead  += len;
#ifdef WIN32
   SetFileBytesRead(GetFileBytesRead() + len);
#else
   fgBytesRead += len;
#endif

end:
   if (gApplication && gApplication->GetSignalHandler())
      gApplication->GetSignalHandler()->HandleDelayedSignal();

   return result;
}

//______________________________________________________________________________
Bool_t TNetFile::WriteBuffer(const char *buf, Int_t len)
{
   // Write specified byte range to remote file via rootd daemon.
   // Returns kTRUE in case of error.

   if (!fSocket || !fWritable) return kTRUE;

   Bool_t result = kFALSE;

   if (fCache) {
      Int_t st;
      Long64_t off = fOffset;
      if ((st = fCache->WriteBuffer(fOffset, buf, len)) < 0) {
         Error("WriteBuffer", "error writing to cache");
         return kTRUE;
      }
      if (st > 0) {
         // fOffset might have been changed via TCache::WriteBuffer(), reset it
         Seek(off + len);
         return result;
      }
   }

   gSystem->IgnoreInterrupt();

   if (fSocket->Send(Form("%ld %d", (Long_t)fOffset, len), kROOTD_PUT) < 0) {
      Error("WriteBuffer", "error sending kROOTD_PUT command");
      result = kTRUE;
      goto end;
   }
   if (fSocket->SendRaw(buf, len) < 0) {
      Error("WriteBuffer", "error sending buffer");
      result = kTRUE;
      goto end;
   }

   Int_t         stat;
   EMessageTypes kind;

   fErrorCode = -1;
   if (Recv(stat, kind) < 0 || kind == kROOTD_ERR) {
      PrintError("WriteBuffer", stat);
      result = kTRUE;
      goto end;
   }

   fOffset += len;

   fBytesWrite  += len;
#ifdef WIN32
   SetFileBytesWritten(GetFileBytesWritten() + len);
#else
   fgBytesWrite += len;
#endif

end:
   gSystem->IgnoreInterrupt(kFALSE);

   return result;
}

//______________________________________________________________________________
Int_t TNetFile::Recv(Int_t &status, EMessageTypes &kind)
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
void TNetFile::Seek(Long64_t offset, ERelativeTo pos)
{
   // Set position from where to start reading.

   switch (pos) {
   case kBeg:
      fOffset = offset;
      break;
   case kCur:
      fOffset += offset;
      break;
   case kEnd:
      fOffset = fEND + offset;  // is fEND really EOF or logical EOF?
      break;
   }
}

//______________________________________________________________________________
void TNetFile::ConnectServer(Int_t *stat, EMessageTypes *kind, Int_t netopt,
                             Int_t tcpwindowsize, Bool_t forceOpen,
                             Bool_t forceRead)
{
   // Connect to remote rootd server.

   TAuthenticate *auth;

   if (netopt < -1) {
      fSocket = new TPSocket(fUrl.GetHost(), fUrl.GetPort(), -netopt,
                             tcpwindowsize);
      if (!fSocket->IsValid()) {
         Error("ConnectServer", "can't open %d parallel connections to rootd on "
               "host %s at port %d", -netopt, fUrl.GetHost(), fUrl.GetPort());
         goto zombie;
      }

      // kNoDelay is internally set by TPSocket

   } else {
      fSocket = new TSocket(fUrl.GetHost(), fUrl.GetPort(), tcpwindowsize);
      if (!fSocket->IsValid()) {
         Error("ConnectServer", "can't open connection to rootd on host %s at port %d",
               fUrl.GetHost(), fUrl.GetPort());
         goto zombie;
      }

      // Set some socket options
      fSocket->SetOption(kNoDelay, 1);

      // Tell rootd we want non parallel connection
      fSocket->Send((Int_t) 0, (Int_t) 0);
   }

   // Get rootd protocol level
   EMessageTypes tmpkind;
   fSocket->Send(kROOTD_PROTOCOL);
   Recv(fProtocol, tmpkind);
   *kind = tmpkind;
   if (fProtocol > 6) {
      fSocket->Send(Form("%d", fgClientProtocol), kROOTD_PROTOCOL2);
      Recv(fProtocol, tmpkind);
      *kind = tmpkind;
   }

   // Check if rootd supports new options
   if (forceRead && fProtocol < 5) {
      Warning("ConnectServer", "rootd does not support \"+read\" option");
      forceRead = kFALSE;
   }

   // Authenticate remotely
   if (gDebug > 2) Info("ConnectServer", "user from Url: %s", fUrl.GetUser());
   auth = new TAuthenticate(fSocket, fUrl.GetHost(),
                Form("%s:%d",fUrl.GetProtocol(),fProtocol), fUrl.GetUser());

   // Attempt authentication
   if (!auth->Authenticate()) {
      Error("ConnectServer", "authentication failed for %s@%s",
            auth->GetUser(), fUrl.GetHost());
      delete auth;
      goto zombie;
   }
   fUser = auth->GetUser();
   delete auth;

   // Open the file
   if (forceOpen)
      fSocket->Send(Form("%s %s", fUrl.GetFile(), ToLower("f"+fOption).Data()), kROOTD_OPEN);
   else if (forceRead)
      fSocket->Send(Form("%s %s", fUrl.GetFile(), "+read"), kROOTD_OPEN);
   else
      fSocket->Send(Form("%s %s", fUrl.GetFile(), ToLower(fOption).Data()), kROOTD_OPEN);

   int  tmpstat;
   Recv(tmpstat, tmpkind);
   *stat = tmpstat;
   *kind = tmpkind;

   return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   SafeDelete(fSocket);
   gDirectory = gROOT;
}

//______________________________________________________________________________
void TNetFile::Create(const char *url, Option_t *option, Int_t netopt)
{
   // Create a NetFile object. A net file is the same as a TFile
   // except that it is being accessed via a rootd server. The url
   // argument must be of the form: root[s|k]://host.dom.ain/file.root.
   // When protocol is "roots" try using SRP authentication.
   // When protocol is "rootk" try using kerberos5 authentication.
   // If the file specified in the URL does not exist, is not accessable
   // or can not be created the kZombie bit will be set in the TNetFile
   // object. Use IsZombie() to see if the file is accessable.
   // If the remote daemon thinks the file is still connected, while you are
   // sure this is not the case you can force open the file by preceding the
   // option argument with an "-", e.g.: "-recreate". Do this only
   // in cases when you are very sure nobody else is using the file.
   // To bypass the writelock on a file, to allow the reading of a file
   // that is being written by another process, explicitely specify the
   // "+read" option ("read" being the default option).
   // The netopt argument can be used to specify the size of the tcp window in
   // bytes (for more info see: http://www.psc.edu/networking/perf_tune.html).
   // The default and minimum tcp window size is 65535 bytes.
   // If netopt < -1 then |netopt| is the number of parallel sockets that will
   // be used to connect to rootd. This option should be used on fat pipes
   // (i.e. high bandwidth, high latency links). The ideal number of parallel
   // sockets depends on the bandwidth*delay product. Generally 5-7 is a good
   // number.
   // For a description of the option and other arguments see the TFile ctor.
   // The preferred interface to this constructor is via TFile::Open().

   Int_t tcpwindowsize = 65535;

   fSocket    = 0;
   fOffset    = 0;
   fErrorCode = -1;

   fOption = option;

   Bool_t forceOpen = kFALSE;
   if (option[0] == '-') {
      fOption   = &option[1];
      forceOpen = kTRUE;
   }
   // accept 'f', like 'frecreate' still for backward compatibility
   if (option[0] == 'F' || option[0] == 'f') {
      fOption   = &option[1];
      forceOpen = kTRUE;
   }

   Bool_t forceRead = kFALSE;
   if (!strcasecmp(option, "+read")) {
      fOption   = &option[1];
      forceRead = kTRUE;
   }

   fOption.ToUpper();

   if (fOption == "NEW")
      fOption = "CREATE";

   Bool_t create   = (fOption == "CREATE") ? kTRUE : kFALSE;
   Bool_t recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   Bool_t update   = (fOption == "UPDATE") ? kTRUE : kFALSE;
   Bool_t read     = (fOption == "READ") ? kTRUE : kFALSE;
   if (!create && !recreate && !update && !read) {
      read    = kTRUE;
      fOption = "READ";
   }

   if (!fUrl.IsValid()) {
      Error("Create", "invalid URL specified: %s", url);
      goto zombie;
   }

   if (netopt > tcpwindowsize)
      tcpwindowsize = netopt;

   // Open connection to remote rootd server
   EMessageTypes kind;
   Int_t stat;
   ConnectServer(&stat, &kind, netopt, tcpwindowsize, forceOpen, forceRead);
   if (gDebug > 2) Info("Create", "got from host %d %d", stat, kind);

   if (kind == kROOTD_ERR) {
      PrintError("Create", stat);
      goto zombie;
   }

   if (recreate) {
      recreate = kFALSE;
      create   = kTRUE;
      fOption  = "CREATE";
   }

   if (update && stat > 1) {
      update = kFALSE;
      create = kTRUE;
      stat   = 1;
   }

   if (stat == 1)
      fWritable = kTRUE;
   else
      fWritable = kFALSE;

   Init(create);
   return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   SafeDelete(fSocket);
   gDirectory = gROOT;
}

//______________________________________________________________________________
Int_t TNetFile::GetClientProtocol()
{
   // Static method returning supported rootd client protocol.

   return fgClientProtocol;
}
