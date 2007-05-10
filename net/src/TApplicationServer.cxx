// @(#)root/proof:$Name:  $:$Id:$

// Author: G. Ganis Apr 2007

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TApplicationServer                                                   //
//                                                                      //
// TApplicationServer is the remote application run by the roots main   //
// program. The input is taken from the socket connection to the client.//
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef R__HAVE_CONFIG
#include "RConfigure.h"
#endif

#include "RConfig.h"
#include "Riostream.h"

#ifdef WIN32
   #include <io.h>
   typedef long off_t;
#endif
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#if (defined(__FreeBSD__) && (__FreeBSD__ < 4)) || \
    (defined(__APPLE__) && (!defined(MAC_OS_X_VERSION_10_3) || \
     (MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_3)))
#include <sys/file.h>
#define lockf(fd, op, sz)   flock((fd), (op))
#ifndef F_LOCK
#define F_LOCK             (LOCK_EX | LOCK_NB)
#endif
#ifndef F_ULOCK
#define F_ULOCK             LOCK_UN
#endif
#endif

#include "RRemoteProtocol.h"

#include "TApplicationServer.h"
#include "TEnv.h"
#include "TError.h"
#include "TException.h"
#include "TInterpreter.h"
#include "TMD5.h"
#include "TMessage.h"
#include "TROOT.h"
#include "TSocket.h"
#include "TSystem.h"
#include "TUrl.h"
#include "TObjString.h"
#include "compiledata.h"
#include "TClass.h"


//----- Interrupt signal handler -----------------------------------------------
//______________________________________________________________________________
class TASInterruptHandler : public TSignalHandler {
   TApplicationServer  *fServ;
public:
   TASInterruptHandler(TApplicationServer *s)
      : TSignalHandler(kSigUrgent, kFALSE) { fServ = s; }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TASInterruptHandler::Notify()
{
   // Handle this interrupt

   fServ->HandleUrgentData();
   if (TROOT::Initialized()) {
      Throw(GetSignal());
   }
   return kTRUE;
}

//----- SigPipe signal handler -------------------------------------------------
//______________________________________________________________________________
class TASSigPipeHandler : public TSignalHandler {
   TApplicationServer  *fServ;
public:
   TASSigPipeHandler(TApplicationServer *s) : TSignalHandler(kSigPipe, kFALSE)
      { fServ = s; }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TASSigPipeHandler::Notify()
{
   // Handle this signal

   fServ->HandleSigPipe();
   return kTRUE;
}

//----- Input handler for messages from client -----------------------
//______________________________________________________________________________
class TASInputHandler : public TFileHandler {
   TApplicationServer  *fServ;
public:
   TASInputHandler(TApplicationServer *s, Int_t fd) : TFileHandler(fd, 1)
      { fServ = s; }
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }
};

//______________________________________________________________________________
Bool_t TASInputHandler::Notify()
{
   // Handle this input

   fServ->HandleSocketInput();
   return kTRUE;
}

TString TASLogHandler::fgPfx = ""; // Default prefix to be prepended to messages
//______________________________________________________________________________
TASLogHandler::TASLogHandler(const char *cmd, TSocket *s, const char *pfx)
              : TFileHandler(-1, 1), fSocket(s), fPfx(pfx)
{
   // Execute 'cmd' in a pipe and handle output messages from the related file

   ResetBit(kFileIsPipe);
   fFile = 0;
   if (s && cmd) {
      fFile = gSystem->OpenPipe(cmd, "r");
      if (fFile) {
         SetFd(fileno(fFile));
         // Notify what already in the file
         Notify();
         // Used in the destructor
         SetBit(kFileIsPipe);
      } else {
         fSocket = 0;
         Error("TASLogHandler", "executing command in pipe");
      }
   } else {
      Error("TASLogHandler",
            "undefined command (%p) or socket (%p)", (int *)cmd, s);
   }
}
//______________________________________________________________________________
TASLogHandler::TASLogHandler(FILE *f, TSocket *s, const char *pfx)
              : TFileHandler(-1, 1), fSocket(s), fPfx(pfx)
{
   // Handle available message from the open file 'f'

   ResetBit(kFileIsPipe);
   fFile = 0;
   if (s && f) {
      fFile = f;
      SetFd(fileno(fFile));
      // Notify what already in the file
      Notify();
   } else {
      Error("TASLogHandler", "undefined file (%p) or socket (%p)", f, s);
   }
}
//______________________________________________________________________________
TASLogHandler::~TASLogHandler()
{
   // Handle available message in the open file

   if (TestBit(kFileIsPipe) && fFile)
      gSystem->ClosePipe(fFile);
   fFile = 0;
   fSocket = 0;
   ResetBit(kFileIsPipe);
}
//______________________________________________________________________________
Bool_t TASLogHandler::Notify()
{
   // Handle available message in the open file

   if (IsValid()) {
      TMessage m(kMESS_ANY);
      // Read buffer
      char line[4096];
      char *plf = 0;
      while (fgets(line, sizeof(line), fFile)) {
         if ((plf = strchr(line, '\n')))
            *plf = 0;
         // Send the message one level up
         m.Reset(kMESS_ANY);
         m << (Int_t)kRRT_Message;
         if (fPfx.Length() > 0) {
            // Prepend prefix specific to this instance
            m << TString(Form("%s: %s", fPfx.Data(), line));
         } else if (fgPfx.Length() > 0) {
            // Prepend default prefix
            m << TString(Form("%s: %s", fgPfx.Data(), line));
         } else {
            // Nothing to prepend
            m << TString(line);
         }
         fSocket->Send(m);
      }
   }
   return kTRUE;
}
//______________________________________________________________________________
void TASLogHandler::SetDefaultPrefix(const char *pfx)
{
   // Static method to set the default prefix

   fgPfx = pfx;
}

//______________________________________________________________________________
TASLogHandlerGuard::TASLogHandlerGuard(const char *cmd, TSocket *s,
                                       const char *pfx, Bool_t on)
{
   // Init a guard for executing a command in a pipe

   fExecHandler = 0;
   if (cmd && on) {
      fExecHandler = new TASLogHandler(cmd, s, pfx);
      if (fExecHandler->IsValid()) {
         gSystem->AddFileHandler(fExecHandler);
      } else {
         Error("TASLogHandlerGuard","invalid handler");
      }
   } else {
      if (on)
         Error("TASLogHandlerGuard","undefined command");
   }
}

//______________________________________________________________________________
TASLogHandlerGuard::TASLogHandlerGuard(FILE *f, TSocket *s,
                                       const char *pfx, Bool_t on)
{
   // Init a guard for executing a command in a pipe

   fExecHandler = 0;
   if (f && on) {
      fExecHandler = new TASLogHandler(f, s, pfx);
      if (fExecHandler->IsValid()) {
         gSystem->AddFileHandler(fExecHandler);
      } else {
         Error("TASLogHandlerGuard","invalid handler");
      }
   } else {
      if (on)
         Error("TASLogHandlerGuard","undefined file");
   }
}

//______________________________________________________________________________
TASLogHandlerGuard::~TASLogHandlerGuard()
{
   // Close a guard for executing a command in a pipe

   if (fExecHandler && fExecHandler->IsValid()) {
      gSystem->RemoveFileHandler(fExecHandler);
      SafeDelete(fExecHandler);
   }
}

ClassImp(TApplicationServer)

//______________________________________________________________________________
TApplicationServer::TApplicationServer(Int_t *argc, char **argv,
                                       FILE *flog, const char *logfile)
       : TApplication("server", argc, argv, 0, -1)
{
   // Main constructor. Create an application environment. The TApplicationServer
   // environment provides an eventloop via inheritance of TApplication.

   // Parse options
   GetOptions(argc, argv);

   // Abort on higher than kSysError's and set error handler
   gErrorAbortLevel = kSysError + 1;
   SetErrorHandler(ErrorHandler);

   fInterrupt       = kFALSE;
   fSocket          = 0;

   fLogFilePath     = logfile;
   fLogFile         = flog;
   fLogFileDes      = -1;
   if (!fLogFile || (fLogFileDes = fileno(fLogFile)) < 0)
      // For some reason we failed setting a redirection; we cannot continue
      Terminate(0);
   fRealTimeLog     = kFALSE;

   // Default prefix for notifications
   TASLogHandler::SetDefaultPrefix(Form("roots:%s", gSystem->HostName()));

   // Now we contact back the client: if we fail we set ourselves 
   // as invalid
   fIsValid = kFALSE;

   if (!(fSocket = new TSocket(GetHost(), GetPort()))) {
      Terminate(0);
      return;
   }
   Int_t sock = fSocket->GetDescriptor();

   if (Setup() != 0) {
      Error("TApplicationServer", "failed to setup - quitting");
      SendLogFile(-98);
      Terminate(0);
   }

   // Everybody expects iostream to be available, so load it...
   ProcessLine("#include <iostream>", kTRUE);
   ProcessLine("#include <_string>",kTRUE); // for std::string iostream.

   // Allow the usage of ClassDef and ClassImp in interpreted macros
   ProcessLine("#include <RtypesCint.h>", kTRUE);

   // Disallow the interpretation of Rtypes.h, TError.h and TGenericClassInfo.h
   ProcessLine("#define ROOT_Rtypes 0", kTRUE);
   ProcessLine("#define ROOT_TError 0", kTRUE);
   ProcessLine("#define ROOT_TGenericClassInfo 0", kTRUE);

   // Load user functions
   const char *logon;
   logon = gEnv->GetValue("Root.Load", (char *)0);
   if (logon) {
      char *mac = gSystem->Which(TROOT::GetMacroPath(), logon, kReadPermission);
      if (mac)
         ProcessLine(Form(".L %s", logon), kTRUE);
      delete [] mac;
   }

   // Execute logon macro
   logon = gEnv->GetValue("Root.Logon", (char *)0);
   if (logon && !NoLogOpt()) {
      char *mac = gSystem->Which(TROOT::GetMacroPath(), logon, kReadPermission);
      if (mac)
         ProcessFile(logon);
      delete [] mac;
   }

   // Save current interpreter context
   gInterpreter->SaveContext();
   gInterpreter->SaveGlobalsContext();

   // Install interrupt and message input handlers
   gSystem->AddSignalHandler(new TASInterruptHandler(this));
   gSystem->AddFileHandler(new TASInputHandler(this, sock));

   // We are done
   fIsValid = kTRUE;

   // Startup notification
   SendLogFile();
}

//______________________________________________________________________________
Int_t TApplicationServer::Setup()
{
   // Print the Remote Server logo on standard output.
   // Return 0 on success, -1 on failure

   char str[512];
   sprintf(str, "**** Remote session @ %s started ****", gSystem->HostName());
   if (fSocket->Send(str) != 1+static_cast<Int_t>(strlen(str))) {
      Error("Setup", "failed to send startup message");
      return -1;
   }

   // Send our protocol level to the client
   if (fSocket->Send(kRRemote_Protocol, kROOTD_PROTOCOL) != 2*sizeof(Int_t)) {
      Error("Setup", "failed to send local protocol");
      return -1;
   }

   // Send the host name and full path to log file
   TMessage msg(kMESS_ANY);
   msg << TString(gSystem->HostName()) << fLogFilePath;
   fSocket->Send(msg);

   // Set working directory
   fWorkDir = gSystem->WorkingDirectory();
   if (strlen(fUrl.GetFile()) > 0) {
      fWorkDir = fUrl.GetFile();
      char *workdir = gSystem->ExpandPathName(fWorkDir.Data());
      fWorkDir = workdir;
      delete [] workdir;
   }

   // Go to working dir
   if (gSystem->AccessPathName(fWorkDir)) {
      gSystem->mkdir(fWorkDir, kTRUE);
      if (!gSystem->ChangeDirectory(fWorkDir)) {
         SysError("Setup", "can not change to directory %s",
                  fWorkDir.Data());
      }
   } else {
      if (!gSystem->ChangeDirectory(fWorkDir)) {
         gSystem->Unlink(fWorkDir);
         gSystem->mkdir(fWorkDir, kTRUE);
         if (!gSystem->ChangeDirectory(fWorkDir)) {
            SysError("Setup", "can not change to directory %s",
                     fWorkDir.Data());
         }
      }
   }

#if 0 // G.Ganis May 11, 2007
   // This needs to be fixed: we disable for the time being
   // Socket options: incoming OOB should generate a SIGURG
   if (fSocket->SetOption(kProcessGroup, (-1)*gSystem->GetPid()) != 0)
      SysWarning("Setup", "failed to enable SIGURG generation on incoming OOB");
#endif

   // Send messages off immediately to reduce latency
   if (fSocket->SetOption(kNoDelay, 1) != 0) {}
      //SysWarning("Setup", "failed to set no-delay option on input socket");

   // Check every two hours if client is still alive
   if (fSocket->SetOption(kKeepAlive, 1) != 0) {}
      //SysWarning("Setup", "failed to set keepalive option on input socket");

   // Install SigPipe handler to handle kKeepAlive failure
   gSystem->AddSignalHandler(new TASSigPipeHandler(this));

   // Done
   return 0;
}

//______________________________________________________________________________
TApplicationServer::~TApplicationServer()
{
   // Cleanup. Not really necessary since after this dtor there is no
   // live anyway.

   SafeDelete(fSocket);
   close(fLogFileDes);
}

//______________________________________________________________________________
void TApplicationServer::GetOptions(Int_t *argc, char **argv)
{
   // Get and handle command line options. Fixed format:
   // "protocol  url"

   if (*argc < 4) {
      Fatal("GetOptions", "must be started with 4 arguments");
      gSystem->Exit(1);
   }

   // Protocol run by the client
   fProtocol = TString(argv[1]).Atoi();

   // Client URL
   fUrl.SetUrl(argv[2]);

   // Debug level
   gDebug = 0;
   TString argdbg(argv[3]);
   if (argdbg.BeginsWith("-d=")) {
      argdbg.ReplaceAll("-d=","");
      gDebug = argdbg.Atoi();
   }
}

//______________________________________________________________________________
void TApplicationServer::Run(Bool_t retrn)
{
   // Main server eventloop.

   // Setup the server
   if (fIsValid) {
      // Run the main event loop
      TApplication::Run(retrn);
   } else {
      Error("Run", "invalid instance: cannot Run()");
      gSystem->Exit(1);
   }
}

//______________________________________________________________________________
void TApplicationServer::HandleSocketInput()
{
   // Handle input coming from the client or from the master server.

   TMessage *mess;
   char      str[2048];
   Int_t     what;

   if (fSocket->Recv(mess) <= 0) {
      // Pending: do something more intelligent here
      // but at least get a message in the log file
      Error("HandleSocketInput", "retrieving message from input socket");
      Terminate(0);
      return;
   }

   what = mess->What();
   if (gDebug > 0)
      Info("HandleSocketInput", "got message of type %d", what);

   switch (what) {

      case kMESS_CINT:
         {  TASLogHandlerGuard hg(fLogFile, fSocket, "", fRealTimeLog);
            mess->ReadString(str, sizeof(str));
            if (gDebug > 1)
               Info("HandleSocketInput:kMESS_CINT", "processing: %s...", str);
            ProcessLine(str);
         }
         SendCanvases();
         SendLogFile();
         break;

      case kMESS_STRING:
         mess->ReadString(str, sizeof(str));
         break;

      case kMESS_OBJECT:
         mess->ReadObject(mess->GetClass());
         break;

      case kMESS_ANY:
         {
            Int_t type;
            (*mess) >> type;
            switch (type) {
               case kRRT_Reset:
                  mess->ReadString(str, sizeof(str));
                  Reset(str);
                  break;

               case kRRT_CheckFile:
                  // Handle file checking request
                  HandleCheckFile(mess);
                  break;

               case kRRT_File:
                  // A file follows
                  mess->ReadString(str, sizeof(str));
                  {  Long_t size;
                     Int_t  bin;
                     char   name[1024];
                     sscanf(str, "%s %d %ld", name, &bin, &size);
                     ReceiveFile(name, bin ? kTRUE : kFALSE, size);
                  }
                  break;

               case kRRT_Terminate:
                  // Terminate the session (will not return from here)
                  Int_t status;
                  (*mess) >> status;
                  Terminate(status);
                  break;

               default:
                  break;
            }
         }
         SendLogFile();
         break;
      default:
         Warning("HandleSocketInput","message type unknown (%d)", what);
         SendLogFile();
         break;
   }

   delete mess;
}

//______________________________________________________________________________
void TApplicationServer::HandleUrgentData()
{
   // Handle Out-Of-Band data sent by the master or client.

   char  oob_byte;
   Int_t n, nch, wasted = 0;

   const Int_t kBufSize = 1024;
   char waste[kBufSize];

   // Real-time notification of messages
   TASLogHandlerGuard hg(fLogFile, fSocket, "", fRealTimeLog);

   Info("HandleUrgentData", "handling oob...");

   // Receive the OOB byte
   while ((n = fSocket->RecvRaw(&oob_byte, 1, kOob)) < 0) {
      if (n == -2) {   // EWOULDBLOCK
         //
         // The OOB data has not yet arrived: flush the input stream
         //
         // In some systems (Solaris) regular recv() does not return upon
         // receipt of the oob byte, which makes the below call to recv()
         // block indefinitely if there are no other data in the queue.
         // FIONREAD ioctl can be used to check if there are actually any
         // data to be flushed.  If not, wait for a while for the oob byte
         // to arrive and try to read it again.
         //
         fSocket->GetOption(kBytesToRead, nch);
         if (nch == 0) {
            gSystem->Sleep(1000);
            continue;
         }

         if (nch > kBufSize) nch = kBufSize;
         n = fSocket->RecvRaw(waste, nch);
         if (n <= 0) {
            Error("HandleUrgentData", "error receiving waste");
            break;
         }
         wasted = 1;
      } else {
         Error("HandleUrgentData", "error receiving OOB (n = %d)",n);
         return;
      }
   }

   Info("HandleUrgentData", "got OOB byte: %d\n", oob_byte);

   switch (oob_byte) {

      case kRRI_Hard:
         Info("HandleUrgentData", "*** Hard Interrupt");

         // Flush input socket
         while (1) {
            Int_t atmark;

            fSocket->GetOption(kAtMark, atmark);

            if (atmark) {
               // Send the OOB byte back so that the client knows where
               // to stop flushing its input stream of obsolete messages
               n = fSocket->SendRaw(&oob_byte, 1, kOob);
               if (n <= 0)
                  Error("HandleUrgentData", "error sending OOB");
               break;
            }

            // find out number of bytes to read before atmark
            fSocket->GetOption(kBytesToRead, nch);
            if (nch == 0) {
               gSystem->Sleep(1000);
               continue;
            }

            if (nch > kBufSize) nch = kBufSize;
            n = fSocket->RecvRaw(waste, nch);
            if (n <= 0) {
               Error("HandleUrgentData", "error receiving waste (2)");
               break;
            }
         }

         SendLogFile();

         break;

      case kRRI_Soft:
         Info("HandleUrgentData", "Soft Interrupt");

         if (wasted) {
            Error("HandleUrgentData", "soft interrupt flushed stream");
            break;
         }

         Interrupt();

         SendLogFile();

         break;

      case kRRI_Shutdown:
         Info("HandleUrgentData", "Shutdown Interrupt");

         Terminate(0);

         break;

      default:
         Error("HandleUrgentData", "unexpected OOB byte");
         break;
   }
}

//______________________________________________________________________________
void TApplicationServer::HandleSigPipe()
{
   // Called when the client is not alive anymore (i.e. when kKeepAlive
   // has failed).

   // Real-time notification of messages
   TASLogHandlerGuard hg(fLogFile, fSocket, "", fRealTimeLog);

   Info("HandleSigPipe", "client died");
   Terminate(0);  // will not return from here....
}

//______________________________________________________________________________
void TApplicationServer::Reset(const char *dir)
{
   // Reset environment to be ready for execution of next command.

   // First go to new directory.
   gDirectory->cd(dir);

   // Clear interpreter environment.
   gROOT->Reset();

   // Make sure current directory is empty (don't delete anything when
   // we happen to be in the ROOT memory only directory!?)
   if (gDirectory != gROOT) {
      gDirectory->Delete();
   }
}

//______________________________________________________________________________
Int_t TApplicationServer::ReceiveFile(const char *file, Bool_t bin, Long64_t size)
{
   // Receive a file, either sent by a client or a master server.
   // If bin is true it is a binary file, other wise it is an ASCII
   // file and we need to check for Windows \r tokens. Returns -1 in
   // case of error, 0 otherwise.

   if (size <= 0) return 0;

   // open file, overwrite already existing file
   Int_t fd = open(file, O_CREAT | O_TRUNC | O_WRONLY, 0600);
   if (fd < 0) {
      SysError("ReceiveFile", "error opening file %s", file);
      return -1;
   }

   const Int_t kMAXBUF = 16384;  //32768  //16384  //65536;
   char buf[kMAXBUF], cpy[kMAXBUF];

   Int_t    left, r;
   Long64_t filesize = 0;

   while (filesize < size) {
      left = Int_t(size - filesize);
      if (left > kMAXBUF)
         left = kMAXBUF;
      r = fSocket->RecvRaw(&buf, left);
      if (r > 0) {
         char *p = buf;

         filesize += r;
         while (r) {
            Int_t w;

            if (!bin) {
               Int_t k = 0, i = 0, j = 0;
               char *q;
               while (i < r) {
                  if (p[i] == '\r') {
                     i++;
                     k++;
                  }
                  cpy[j++] = buf[i++];
               }
               q = cpy;
               r -= k;
               w = write(fd, q, r);
            } else {
               w = write(fd, p, r);
            }

            if (w < 0) {
               SysError("ReceiveFile", "error writing to file %s", file);
               close(fd);
               return -1;
            }
            r -= w;
            p += w;
         }
      } else if (r < 0) {
         Error("ReceiveFile", "error during receiving file %s", file);
         close(fd);
         return -1;
      }
   }

   close(fd);

   chmod(file, 0644);

   return 0;
}

//______________________________________________________________________________
void TApplicationServer::SendLogFile(Int_t status, Int_t start, Int_t end)
{
   // Send log file to master.
   // If start > -1 send only bytes in the range from start to end,
   // if end <= start send everything from start.

   // Determine the number of bytes left to be read from the log file.
   fflush(stdout);

   off_t ltot=0, lnow=0;
   Int_t left = -1;
   Bool_t adhoc = kFALSE;

   if (fLogFileDes > -1) {
      ltot = lseek(fileno(stdout),   (off_t) 0, SEEK_END);
      lnow = lseek(fLogFileDes, (off_t) 0, SEEK_CUR);

      if (start > -1) {
         lseek(fLogFileDes, (off_t) start, SEEK_SET);
         if (end <= start || end > ltot)
            end = ltot;
         left = (Int_t)(end - start);
         if (end < ltot)
            left++;
         adhoc = kTRUE;
      } else {
         left = (Int_t)(ltot - lnow);
      }
   }

   TMessage m(kMESS_ANY);

   if (left > 0) {

      m << (Int_t)kRRT_LogFile << left;
      fSocket->Send(m);

      const Int_t kMAXBUF = 32768;  //16384  //65536;
      char buf[kMAXBUF];
      Int_t wanted = (left > kMAXBUF) ? kMAXBUF : left;
      Int_t len;
      do {
         while ((len = read(fLogFileDes, buf, wanted)) < 0 &&
                TSystem::GetErrno() == EINTR)
            TSystem::ResetErrno();

         if (len < 0) {
            SysError("SendLogFile", "error reading log file");
            break;
         }

         if (end == ltot && len == wanted)
            buf[len-1] = '\n';

         if (fSocket->SendRaw(buf, len) < 0) {
            SysError("SendLogFile", "error sending log file");
            break;
         }

         // Update counters
         left -= len;
         wanted = (left > kMAXBUF) ? kMAXBUF : left;

      } while (len > 0 && left > 0);
   }

   // Restore initial position if partial send
   if (adhoc)
      lseek(fLogFileDes, lnow, SEEK_SET);

   m.Reset();
   m << (Int_t)kRRT_LogDone << status;

   fSocket->Send(m);
}

//______________________________________________________________________________
Int_t TApplicationServer::SendCanvases()
{
   // Send any created canvas to client

   Int_t nc = 0;

   TMessage mess(kMESS_OBJECT);
   TIter next(gROOT->GetListOfCanvases());
   TObject *o = 0;
   while ((o = next())) {
      mess.Reset(kMESS_OBJECT);
      mess.WriteObject(o);
      fSocket->Send(mess);
      nc++;
   }
   return nc;
}

//______________________________________________________________________________
void TApplicationServer::Terminate(Int_t status)
{
   // Terminate the proof server.

   // Close and remove the log file; remove the cleanup script
   if (fLogFile) {
      fclose(fLogFile);
      gSystem->Unlink(fLogFilePath);
      TString cleanup = fLogFilePath;
      cleanup.ReplaceAll(".log", ".cleanup");
      gSystem->Unlink(cleanup);
   }

   // Remove input handler to avoid spurious signals in socket
   // selection for closing activities executed upon exit()
   TIter next(gSystem->GetListOfFileHandlers());
   TObject *fh = 0;
   while ((fh = next())) {
      TASInputHandler *ih = dynamic_cast<TASInputHandler *>(fh);
      if (ih)
         gSystem->RemoveFileHandler(ih);
   }

   // Stop processing events
   gSystem->Exit(status);
}

//______________________________________________________________________________
void TApplicationServer::HandleCheckFile(TMessage *mess)
{
   // Handle file checking request.

   TString filenam;
   TMD5    md5;
   TMessage m(kMESS_ANY);

   // Parse message
   (*mess) >> filenam >> md5;

   // check file in working directory
   TMD5 *md5local = TMD5::FileChecksum(filenam);
   if (md5local && md5 == (*md5local)) {
      // We have an updated copy of the file
      m << (Int_t) kRRT_CheckFile << (Bool_t) kTRUE;
      fSocket->Send(m);
      if (gDebug > 0)
         Info("HandleCheckFile", "up-to-date version of %s available", filenam.Data());
   } else {
      m << (Int_t) kRRT_CheckFile << (Bool_t) kFALSE;
      fSocket->Send(m);
      if (gDebug > 0)
         Info("HandleCheckFile", "file %s needs to be uploaded", filenam.Data());
   }
   delete md5local;
}

//______________________________________________________________________________
void TApplicationServer::ErrorHandler(Int_t level, Bool_t abort, const char *location,
                              const char *msg)
{
   // The error handler function. It prints the message on stderr and
   // if abort is set it aborts the application.

   if (gErrorIgnoreLevel == kUnset) {
      gErrorIgnoreLevel = 0;
      if (gEnv) {
         TString level = gEnv->GetValue("Root.ErrorIgnoreLevel", "Print");
         if (!level.CompareTo("Print", TString::kIgnoreCase))
            gErrorIgnoreLevel = kPrint;
         else if (!level.CompareTo("Info", TString::kIgnoreCase))
            gErrorIgnoreLevel = kInfo;
         else if (!level.CompareTo("Warning", TString::kIgnoreCase))
            gErrorIgnoreLevel = kWarning;
         else if (!level.CompareTo("Error", TString::kIgnoreCase))
            gErrorIgnoreLevel = kError;
         else if (!level.CompareTo("Break", TString::kIgnoreCase))
            gErrorIgnoreLevel = kBreak;
         else if (!level.CompareTo("SysError", TString::kIgnoreCase))
            gErrorIgnoreLevel = kSysError;
         else if (!level.CompareTo("Fatal", TString::kIgnoreCase))
            gErrorIgnoreLevel = kFatal;
      }
   }

   if (level < gErrorIgnoreLevel)
      return;

   static TString syslogService;

   if (syslogService.IsNull()) {
      syslogService = "server";
      gSystem->Openlog(syslogService, kLogPid | kLogCons, kLogLocal5);
   }

   const char *type   = 0;
   ELogLevel loglevel = kLogInfo;

   if (level >= kPrint) {
      loglevel = kLogInfo;
      type = "Print";
   }
   if (level >= kInfo) {
      loglevel = kLogInfo;
      type = "Info";
   }
   if (level >= kWarning) {
      loglevel = kLogWarning;
      type = "Warning";
   }
   if (level >= kError) {
      loglevel = kLogErr;
      type = "Error";
   }
   if (level >= kBreak) {
      loglevel = kLogErr;
      type = "*** Break ***";
   }
   if (level >= kSysError) {
      loglevel = kLogErr;
      type = "SysError";
   }
   if (level >= kFatal) {
      loglevel = kLogErr;
      type = "Fatal";
   }

   TString node = "server";
   TString buf;

   if (!location || strlen(location) == 0 ||
       (level >= kPrint && level < kInfo) ||
       (level >= kBreak && level < kSysError)) {
      fprintf(stderr, "%s on %s: %s\n", type, node.Data(), msg);
      buf.Form("%s:%s:%s", node.Data(), type, msg);
   } else {
      fprintf(stderr, "%s in <%s> on %s: %s\n", type, location, node.Data(), msg);
      buf.Form("%s:%s:<%s>:%s", node.Data(), type, location, msg);
   }
   fflush(stderr);

   gSystem->Syslog(loglevel, buf);

   if (abort) {
      fprintf(stderr, "aborting\n");
      fflush(stderr);
      gSystem->StackTrace();
      gSystem->Abort();
   }
}
