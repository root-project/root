// @(#)root/proof:$Name:  $:$Id: TProofServ.cxx,v 1.25 2002/06/14 10:29:06 rdm Exp $
// Author: Fons Rademakers   16/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofServ                                                           //
//                                                                      //
// TProofServ is the PROOF server. It can act either as the master      //
// server or as a slave server, depending on its startup arguments. It  //
// receives and handles message coming from the client or from the      //
// master server.                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef WIN32
   #include <io.h>
   typedef long off_t;
#endif
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#if (defined(__FreeBSD__) && (__FreeBSD__ < 4)) || defined(__APPLE__)
#include <sys/file.h>
#define lockf(fd, op, sz)   flock((fd), (op))
#define F_LOCK             (LOCK_EX | LOCK_NB)
#define F_ULOCK             LOCK_UN
#endif

#include "TProofServ.h"
#include "TProofLimitsFinder.h"
#include "TProof.h"
#include "TROOT.h"
#include "TFile.h"
#include "TSysEvtHandler.h"
#include "TAuthenticate.h"
#include "TSystem.h"
#include "TInterpreter.h"
#include "TException.h"
#include "TSocket.h"
#include "TStopwatch.h"
#include "TMessage.h"
#include "TUrl.h"
#include "TEnv.h"
#include "TError.h"
#include "TTree.h"
#include "TProofPlayer.h"
#include "TDSetProxy.h"
#include "TTimeStamp.h"
#include "TProofDebug.h"

#ifndef R__WIN32
const char* const kCP = "/bin/cp -f";
const char* const kRM = "/bin/rm -rf";
const char* const kLS = "/bin/ls -l";
#else
const char* const kCP = "copy";
const char* const kRM = "delete";
const char* const kLS = "dir";
#endif


TProofServ *gProofServ;


//______________________________________________________________________________
static void ProofServErrorHandler(int level, Bool_t abort, const char *location,
                                  const char *msg)
{
   // The PROOF error handler function. It prints the message on stderr and
   // if abort is set it aborts the application.

   if (!gProofServ)
      return;

   if (level < gErrorIgnoreLevel)
      return;

   const char *type   = 0;
   ELogLevel loglevel = kLogInfo;

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
   if (level >= kSysError) {
      loglevel = kLogErr;
      type = "SysError";
   }
   if (level >= kFatal) {
      loglevel = kLogErr;
      type = "Fatal";
   }

   TString node = gProofServ->IsMaster() ? "master" : "slave ";
   if (node != "master") node += gProofServ->GetOrdinal();
   char *bp;

   if (!location || strlen(location) == 0) {
      fprintf(stderr, "%s on %s: %s\n", type, node.Data(), msg);
      bp = Form("%s:%s:%s:%s", gProofServ->GetUser(), node.Data(), type, msg);
   } else {
      fprintf(stderr, "%s in <%s> on %s: %s\n", type, location, node.Data(), msg);
      bp = Form("%s:%s:%s:<%s>:%s", gProofServ->GetUser(), node.Data(), type, location, msg);
   }
   fflush(stderr);
   gSystem->Syslog(loglevel, bp);

   if (abort) {
      static Bool_t recursive = kFALSE;

      if (!recursive) {
         recursive = kTRUE;
         gProofServ->GetSocket()->Send(kPROOF_FATAL);
         recursive = kFALSE;
      }

      fprintf(stderr, "aborting\n");
      fflush(stderr);
      gSystem->StackTrace();
      gSystem->Abort();
   }
}

//----- Interrupt signal handler -----------------------------------------------
//______________________________________________________________________________
class TProofServInterruptHandler : public TSignalHandler {
   TProofServ  *fServ;
public:
   TProofServInterruptHandler(TProofServ *s)
      : TSignalHandler(kSigUrgent, kFALSE) { fServ = s; }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TProofServInterruptHandler::Notify()
{
   fServ->HandleUrgentData();
   if (TROOT::Initialized()) {
      Throw(GetSignal());
   }
   return kTRUE;
}

//----- SigPipe signal handler -------------------------------------------------
//______________________________________________________________________________
class TProofServSigPipeHandler : public TSignalHandler {
   TProofServ  *fServ;
public:
   TProofServSigPipeHandler(TProofServ *s) : TSignalHandler(kSigPipe, kFALSE)
      { fServ = s; }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TProofServSigPipeHandler::Notify()
{
   fServ->HandleSigPipe();
   return kTRUE;
}

//----- Input handler for messages from parent or master -----------------------
//______________________________________________________________________________
class TProofServInputHandler : public TFileHandler {
   TProofServ  *fServ;
public:
   TProofServInputHandler(TProofServ *s, Int_t fd) : TFileHandler(fd, 1)
      { fServ = s; }
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }
};

//______________________________________________________________________________
Bool_t TProofServInputHandler::Notify()
{
   fServ->HandleSocketInput();
   return kTRUE;
}


ClassImp(TProofServ)

//______________________________________________________________________________
TProofServ::TProofServ(int *argc, char **argv)
       : TApplication("proofserv", argc, argv, 0, -1)
{
   // Create an application environment. The TProofServ environment provides
   // an eventloop via inheritance of TApplication.

   // debug hook
#ifdef R__DEBUG
   int debug = 1;
   while (debug)
      ;
#endif

   // make sure all registered dictionaries have been initialized
   gInterpreter->InitializeDictionaries();

   // abort on kSysError's or higher and set error handler
   gErrorAbortLevel = kSysError;
   SetErrorHandler(ProofServErrorHandler);

   fNcmd        = 0;
   fInterrupt   = kFALSE;
   fProtocol    = 0;
   fOrdinal     = -1;
   fGroupId     = -1;
   fGroupSize   = 0;
   fLogLevel    = 1;
   fRealTime    = 0.0;
   fCpuTime     = 0.0;
   fProof       = 0;
   fSocket      = new TSocket(0);

   GetOptions(argc, argv);

   // debug hooks
   if (IsMaster()) {
#ifdef R__MASTERDEBUG
      int debug = 1;
      while (debug)
         ;
#endif
   } else {
#ifdef R__SLAVEDEBUG
      int debug = 1;
      while (debug)
         ;
#endif
   }

   Setup();
   RedirectOutput();

   // Send message of the day to the client
   if (IsMaster()) {
      if (CatMotd() == -1) {
         SendLogFile(-99);
         Terminate(0);
      }
   } else {
      THLimitsFinder::SetLimitsFinder(new TProofLimitsFinder);
   }

   // Everybody expects iostream to be available, so load it...
#ifndef WIN32
   ProcessLine("#include <iostream>");
#endif

   // Load user functions
   const char *logon;
   logon = gEnv->GetValue("Proof.Load", (char*)0);
   if (logon && !gSystem->AccessPathName(logon, kReadPermission))
      ProcessLine(Form(".L %s",logon));

   // Execute logon macro
   logon = gEnv->GetValue("Proof.Logon", (char*)0);
   if (logon && !NoLogOpt() && !gSystem->AccessPathName(logon, kReadPermission))
      ProcessFile(logon);

   gInterpreter->SaveContext();
   gInterpreter->SaveGlobalsContext();

   // Install interrupt and message input handlers
   gSystem->AddSignalHandler(new TProofServInterruptHandler(this));
   gSystem->AddFileHandler(new TProofServInputHandler(this, 0));

   gProofServ = this;

   // if master, start slave servers
   if (IsMaster()) {
      TAuthenticate::SetGlobalUser(fUser);
      TAuthenticate::SetGlobalPasswd(fPasswd);
      TString master = "proof://__master__";
      TInetAddress a = gSystem->GetSockName(0);
      if (a.IsValid()) {
         master += ":";
         master += a.GetPort();
      }
      fProof = new TProof(master, fConfFile, fConfDir, fLogLevel);
      SendLogFile();
   }
}

//______________________________________________________________________________
TProofServ::~TProofServ()
{
   // Cleanup. Not really necessary since after this dtor there is no
   // live anyway.

   delete fSocket;
}

//______________________________________________________________________________
Int_t TProofServ::CatMotd()
{
   // Print message of the day (in fConfDir/proof/etc/motd). The motd
   // is not shown more than once a dat. If the file fConfDir/proof/etc/noproof
   // exists, show its contents and close the connection.

   TString motdname;
   TString lastname;
   FILE   *motd;
   Bool_t  show = kFALSE;

   motdname = fConfDir + "/proof/etc/noproof";
   if ((motd = fopen(motdname, "r"))) {
      int c;
      printf("\n");
      while ((c = getc(motd)) != EOF)
         putchar(c);
      fclose(motd);
      printf("\n");

      return -1;
   }

   // get last modification time of the file ~/proof/.prooflast
   lastname = TString(kPROOF_WorkDir) + "/.prooflast";
   char *last = gSystem->ExpandPathName(lastname.Data());
   Long_t id, size, flags, modtime, lasttime;
   if (gSystem->GetPathInfo(last, &id, &size, &flags, &lasttime) == 1)
      lasttime = 0;

   // show motd at least once per day
   if (time(0) - lasttime > (time_t)86400)
      show = kTRUE;

   motdname = fConfDir + "/proof/etc/motd";
   if (gSystem->GetPathInfo(motdname, &id, &size, &flags, &modtime) == 0) {
      if (modtime > lasttime || show) {
         if ((motd = fopen(motdname, "r"))) {
            int c;
            printf("\n");
            while ((c = getc(motd)) != EOF)
               putchar(c);
            fclose(motd);
            printf("\n");
         }
      }
   }

   int fd = creat(last, 0600);
   close(fd);
   delete [] last;

   return 0;
}

//______________________________________________________________________________
TObject *TProofServ::Get(const char *namecycle)
{
   // Get object with name "name;cycle" (e.g. "aap;2") from master or client.
   // This method is called by TDirectory::Get() in case the object can not
   // be found locally.

   fSocket->Send(namecycle, kPROOF_GETOBJECT);

   TMessage *mess;
   if (fSocket->Recv(mess) < 0)
      return 0;

   TObject *idcur = 0;
   if (mess->What() == kMESS_OBJECT)
      idcur = mess->ReadObject(mess->GetClass());
   delete mess;

   return idcur;
}

//______________________________________________________________________________
void TProofServ::GetLimits(Int_t dim, Int_t nentries, Int_t *nbins, Double_t *vmin, Double_t *vmax)
{
   // Get limits of histogram from master. This method is called by
   // TTree::TakeEstimate().

   TMessage mess(kPROOF_LIMITS);

   mess << dim << nentries << nbins[0] << vmin[0] << vmax[0];

   if (dim == 2)
      mess << nbins[1] << vmin[1] << vmax[1];

   if (dim == 3)
      mess << nbins[2] << vmin[2] << vmax[2];

   fSocket->Send(mess);

   TMessage *answ;
   if (fSocket->Recv(answ) != -1) {
      (*answ) >> nbins[0] >> vmin[0] >> vmax[0];

      if (dim == 2)
         (*answ) >> nbins[1] >> vmin[1] >> vmax[1];

      if (dim == 3)
         (*answ) >> nbins[2] >> vmin[2] >> vmax[2];

      delete answ;
   }
}

//______________________________________________________________________________
TDSetElement *TProofServ::GetNextPacket()
{
   // Get next range of entries to be processed on this server.

   fSocket->Send(kPROOF_GETPACKET);

   TMessage *mess;
   if (fSocket->Recv(mess) < 0)
      return 0;

   Bool_t         ok;
   TDSetElement  *e;
   TString        file;
   TString        dir;
   TString        obj;
   Long64_t       first;
   Long64_t       num;

   (*mess) >> ok;

   if (ok) {
      (*mess) >> file >> dir >> obj >> first >> num;
      e = new TDSetElement(0, file, obj, dir, first, num);
   } else {
      e = 0;
   }
   if (e != 0) {
      PDB(kLoop,2) Info("GetNextPacket", "'%s' '%s' '%s' %d %d", e->GetFileName(),
            e->GetDirectory(), e->GetObjName(),e->GetFirst(),e->GetNum());
   } else {
      PDB(kLoop,2) Info("GetNextPacket", "Done");
   }

   return e;
}

//______________________________________________________________________________
void TProofServ::GetOptions(int *argc, char **argv)
{
   // Get and handle command line options. Fixed format:
   // "proofserv"|"proofslave" <confdir>

   if (!strcmp(argv[1], "proofserv") || !strcmp(argv[1], "proofslave")) {
      fService = argv[1];
      fMasterServ = kTRUE;
      if (!strcmp(argv[1], "proofslave")) fMasterServ = kFALSE;
   }

   fConfDir = argv[2];
}

//______________________________________________________________________________
void TProofServ::HandleSocketInput()
{
   // Handle input coming from the client or from the master server.

   static TStopwatch timer;

   TMessage *mess;
   char      str[2048];
   Int_t     what;

   if (fSocket->Recv(mess) <= 0)
      Terminate(0);               // do something more intelligent here

   what = mess->What();

   timer.Start();
   fNcmd++;

   switch (what) {

      case kMESS_CINT:
         mess->ReadString(str, sizeof(str));
         if (IsMaster() && IsParallel()) {
            fProof->SendCommand(str);
         } else {
            PDB(kGlobal,1) Info("HandleSocketInput:kMESS_CINT", "processing: %s...", str);
            ProcessLine(str);
         }
         SendLogFile();
         break;

      case kMESS_STRING:
         mess->ReadString(str, sizeof(str));
         break;

      case kMESS_OBJECT:
         mess->ReadObject(mess->GetClass());
         break;

      case kPROOF_GROUPVIEW:
         mess->ReadString(str, sizeof(str));
         sscanf(str, "%d %d", &fGroupId, &fGroupSize);
         break;

      case kPROOF_LOGLEVEL:
         mess->ReadString(str, sizeof(str));
         sscanf(str, "%d", &fLogLevel);
         if (IsMaster())
            fProof->SetLogLevel(fLogLevel);
         break;

      case kPROOF_PING:
         if (IsMaster())
            fProof->Ping();
         // do nothing (ping is already acknowledged)
         break;

      case kPROOF_PRINT:
         Print();
         SendLogFile();
         break;

      case kPROOF_RESET:
         mess->ReadString(str, sizeof(str));
         Reset(str);
         break;

      case kPROOF_STATUS:
         SendStatus();
         break;

      case kPROOF_STOP:
         Terminate(0);
         break;

      case kPROOF_TREEDRAW:
         mess->ReadString(str, sizeof(str));
         {
            Int_t maxv, est;
            char name[64];
            sscanf(str, "%s %d %d", name, &maxv, &est);
            TTree *t = (TTree*)gDirectory->Get(name);
            if (t) {
               t->SetMaxVirtualSize(maxv);
               t->SetEstimate(est);
            }
         }
         break;

      case kPROOF_PROCESS:
         {
            TDSet *dset;
            TString filename;
            TList *input;
            Long64_t nentries, first;

            PDB(kGlobal,1) Info("HandleSocketInput:kPROOF_PROCESS", "enter");

            (*mess) >> dset >> filename >> input >> nentries >> first;

            TProofPlayer *p;

            if (IsMaster()) {
               p = new TProofPlayerRemote(fProof);
            } else {
               p = new TProofPlayerSlave(fSocket);
            }

            if (dset->IsA() == TDSetProxy::Class()) {
               ((TDSetProxy*)dset)->SetProofServ(this);
            }

            TIter next(input);
            for (TObject *obj; (obj = next()); ) {
               PDB(kGlobal,2) Info("HandleSocketInput:kPROOF_PROCESS", "Adding: %s", obj->GetName());
               p->AddInput(obj);
            }
            delete input;

            p->Process(dset, filename, nentries, first);

            // return output!

            PDB(kGlobal,2) Info("HandleSocketInput:kPROOF_PROCESS","Send Output");
            fSocket->SendObject(p->GetOutputList(), kPROOF_OUTPUTLIST);

            PDB(kGlobal,2) Info("HandleSocketInput:kPROOF_PROCESS","Send LogFile");
            SendLogFile();

            delete dset;
            delete p;

            PDB(kGlobal,1) Info("HandleSocketInput:kPROOF_PROCESS","Done");
         }
         break;

      case kPROOF_CHECKFILE:
         {
            TString filenam;
            TMD5    md5;
            (*mess) >> filenam >> md5;
            if (filenam.BeginsWith("-")) {
               // install package:
               // compare md5's, untar, store md5 in PROOF-INF, remove par file
               //...
               UnlockPackage();
               break;
            }
            if (filenam.BeginsWith("+")) {
               // check file in package directory
               filenam = filenam.Strip(TString::kLeading, '+');
               TString packf = fPackageDir + "/" + filenam;
               LockPackage();
               TMD5 *md5local = TMD5::FileChecksum(packf); // read md5 from PROOF-INF
               if (md5local && md5 == (*md5local)) {
                  // package already on server, unlock directory
                  UnlockPackage();
                  fSocket->Send(kPROOF_CHECKFILE);
                  if (fLogLevel > 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE", "package %s already on node", filenam.Data());
               } else {
                  fSocket->Send(kPROOF_FATAL);
                  if (fLogLevel > 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE", "package %s not yet on node", filenam.Data());
               }
               delete md5local;
            } else {
               // check file in cache directory
               TString cachef = fCacheDir + "/" + filenam;
               LockCache();
               TMD5 *md5local = TMD5::FileChecksum(cachef);
               if (md5local && md5 == (*md5local)) {
                  // copy file from cache to working directory
                  gSystem->Exec(Form("%s %s .", kCP, cachef.Data()));
                  fSocket->Send(kPROOF_CHECKFILE);
                  if (fLogLevel > 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE", "file %s already on node", filenam.Data());
               } else {
                  fSocket->Send(kPROOF_FATAL);
                  if (fLogLevel > 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE", "file %s not yet on node", filenam.Data());
               }
               delete md5local;
               UnlockCache();
            }
         }
         break;

      case kPROOF_SENDFILE:
         mess->ReadString(str, sizeof(str));
         {
            Long_t size;
            Int_t  bin;
            char  name[1024];
            sscanf(str, "%s %d %ld", name, &bin, &size);
            ReceiveFile(name, bin ? kTRUE : kFALSE, size);
            // copy file to cache
            if (size > 0) {
               LockCache();
               gSystem->Exec(Form("%s %s %s", kCP, name, fCacheDir.Data()));
               UnlockCache();
            }
            if (IsMaster())
               fProof->SendFile(name, bin);
         }
         break;

      case kPROOF_OPENFILE:
         {
            // open file on master, if successfull this will also send the
            // connect message to the slaves
            TString clsnam, filenam, option;
            (*mess) >> clsnam >> filenam >> option;
            TString cmd;
            cmd = "TFile::Open(\"" + filenam + "\", \"" + option + "\");";
            if (IsMaster()) {
               if (clsnam == "TNetFile") {
                  TUrl url(filenam);
                  Int_t sec = gEnv->GetValue("Rootd.Authentication",
                                             TAuthenticate::kClear);
                  if (!strcmp(url.GetProtocol(), "roots"))
                     sec = TAuthenticate::kSRP;
                  if (!strcmp(url.GetProtocol(), "rootk"))
                     sec = TAuthenticate::kKrb5;
                  TAuthenticate auth(0, url.GetHost(), "rootd", sec);
                  TString user, passwd;
                  if (auth.CheckNetrc(user, passwd))
                     ProcessLine(cmd);
                  else
                     Error("HandleSocketInput", "cannot execute \"%s\" since authentication is not possible",
                           cmd.Data());
               } else
                  ProcessLine(cmd);
            } else
               ProcessLine(cmd);
         }
         SendLogFile();
         break;

      case kPROOF_PARALLEL:
         if (IsMaster()) {
            Int_t nodes;
            (*mess) >> nodes;
            fProof->SetParallel(nodes);
            SendLogFile();
         }
         break;

      case kPROOF_CACHE:
         {
            // handle here all cache and package requests:
            // type: 1 = ShowCache, 2 = ClearCache, 3 = ShowPackages,
            // 4 = ClearPackages, 5 = ClearPackage, 6 = EnablePackage,
            // 7 = ShowEnabledPackages
            Int_t  type;
            Bool_t all;  //build;
            TString package;
            (*mess) >> type;
            switch (type) {
               case 1:
                  (*mess) >> all;
                  printf("*** File cache %s:%s ***\n", gSystem->HostName(),
                         fCacheDir.Data());
                  fflush(stdout);
                  gSystem->Exec(Form("%s %s", kLS, fCacheDir.Data()));
                  if (IsMaster() && all)
                     fProof->ShowCache(all);
                  break;
               case 2:
                  LockCache();
                  gSystem->Exec(Form("%s %s/*", kRM, fCacheDir.Data()));
                  UnlockCache();
                  if (IsMaster())
                     fProof->ClearCache();
                  break;
               case 3:
                  (*mess) >> all;
                  printf("*** Package cache %s:%s ***\n", gSystem->HostName(),
                         fPackageDir.Data());
                  fflush(stdout);
                  gSystem->Exec(Form("%s %s", kLS, fPackageDir.Data()));
                  if (IsMaster() && all)
                     fProof->ShowPackages(all);
                  break;
               case 4:
                  LockPackage();
                  gSystem->Exec(Form("%s %s/*", kRM, fPackageDir.Data()));
                  UnlockPackage();
                  if (IsMaster())
                     fProof->ClearPackages();
                  break;
               case 5:
                  (*mess) >> package;
                  LockPackage();
                  gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
                                package.Data()));
                  UnlockPackage();
                  if (IsMaster())
                     fProof->ClearPackage(package);
                  break;
               default:
                  Error("HandleSocketInput:kPROOF_CACHE", "unknown type %d", type);
                  break;
            }
            SendLogFile();
         }
         break;

      default:
         Error("HandleSocketInput", "unknown command %d", what);
         break;
   }

   fRealTime += (Float_t)timer.RealTime();
   fCpuTime  += (Float_t)timer.CpuTime();

   delete mess;
}

//______________________________________________________________________________
void TProofServ::HandleUrgentData()
{
   // Handle Out-Of-Band data sent by the master or client.

   char  oob_byte;
   int   n, nch, wasted = 0;

   const int kBufSize = 1024;
   char waste[kBufSize];

   if (fLogLevel > 5)
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
         Error("HandleUrgentData", "error receiving OOB");
         return;
      }
   }

   if (fLogLevel > 5)
      Info("HandleUrgentData", "got OOB byte: %d\n", oob_byte);

   switch (oob_byte) {

      case TProof::kHardInterrupt:
         Info("HandleUrgentData", "*** Hard Interrupt");

         // If master server, propagate interrupt to slaves
         if (IsMaster())
            fProof->Interrupt(TProof::kHardInterrupt);

         // Flush input socket
         while (1) {
            int atmark;

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

         break;

      case TProof::kSoftInterrupt:
         Info("HandleUrgentData", "Soft Interrupt");

         // If master server, propagate interrupt to slaves
         if (IsMaster())
            fProof->Interrupt(TProof::kSoftInterrupt);

         if (wasted) {
            Error("HandleUrgentData", "soft interrupt flushed stream");
            break;
         }

         Interrupt();

         break;

      case TProof::kShutdownInterrupt:
         Info("HandleUrgentData", "Shutdown Interrupt");

         // If master server, propagate interrupt to slaves
         if (IsMaster())
            fProof->Interrupt(TProof::kShutdownInterrupt);

         Terminate(0);  // will not return from here....

         break;

      default:
         Error("HandleUrgentData", "unexpected OOB byte");
         break;
   }

   SendLogFile();
}

//______________________________________________________________________________
void TProofServ::HandleSigPipe()
{
   // Called when the client is not alive anymore (i.e. when kKeepAlive
   // has failed).

   if (IsMaster()) {
      // Check if we are here because client is closed. Try to ping client,
      // if that works it we are here because some slave died
      if (fSocket->Send(kPROOF_PING | kMESS_ACK) < 0) {
         Info("HandleSigPipe", "keepAlive probe failed");
         // Tell slaves we are going to close since there is no client anymore
         fProof->Interrupt(TProof::kShutdownInterrupt);
         Terminate(0);
      }
   } else {
      Info("HandleSigPipe", "keepAlive probe failed");
      Terminate(0);  // will not return from here....
   }
}

//______________________________________________________________________________
Bool_t TProofServ::IsParallel() const
{
   // True if in parallel mode.

   if (IsMaster())
      return fProof->IsParallel();
   else
      return kFALSE;
}

//______________________________________________________________________________
Int_t TProofServ::LockDir(const TString &lock)
{
   // Lock a directory. Waits if lock is hold by an other process.
   // Returns 0 on success, -1 in case of error.

   Int_t *fid;
   if (lock == fCacheLock)
      fid = &fCacheLockId;
   else if (lock == fPackageLock)
      fid = &fPackageLockId;
   else {
      Error("LockDir", "unknown lock file specified %s", lock.Data());
      return -1;
   }

   const char *lfile = lock;

   if (gSystem->AccessPathName(lfile))
      *fid = open(lfile, O_CREAT|O_RDWR, 0644);
   else
      *fid = open(lfile, O_RDWR);

   if (*fid == -1) {
      SysError("LockDir", "cannot open lock file %s", lfile);
      return -1;
   }

   // lock the file
#ifndef R__WIN32
   if (lockf(*fid, F_LOCK, (off_t) 1) == -1) {
      SysError("LockDir", "error locking %s", lfile);
      close(*fid);
      *fid = -1;
      return -1;
   }
#endif

   if (fLogLevel > 2)
      Info("LockDir", "file %s locked", lfile);

   return 0;
}

//______________________________________________________________________________
Int_t TProofServ::UnlockDir(const TString &lock)
{
   // Unlock a directory. Returns 0 in case of success,
   // -1 in case of error.

   Int_t *fid;
   if (lock == fCacheLock)
      fid = &fCacheLockId;
   else if (lock == fPackageLock)
      fid = &fPackageLockId;
   else {
      Error("UnlockDir", "unknown lock file specified %s", lock.Data());
      return -1;
   }

   if (*fid == -1) return 0;

   // unlock the file
   lseek(*fid, 0, SEEK_SET);
#ifndef R__WIN32
   if (lockf(*fid, F_ULOCK, (off_t)1) == -1) {
      SysError("UnlockDir", "error unlocking %s", lock.Data());
      close(*fid);
      *fid = -1;
      return -1;
   }
#endif

   if (fLogLevel > 2)
      Info("UnlockDir", "file %s unlocked", lock.Data());

   close(*fid);
   *fid = -1;

   return 0;
}

//______________________________________________________________________________
void TProofServ::Print(Option_t *) const
{
   // Print status of slave server.

   if (IsMaster())
      fProof->Print();
   else
      Printf("This is slave %s", gSystem->HostName());
}

//______________________________________________________________________________
void TProofServ::RedirectOutput()
{
   // Redirect stdout to a log file. This log file will be flushed to the
   // client or master after each command.

   // Duplicate the initial socket (0), this will yield a socket with
   // a descriptor >0, which will free descriptor 0 for stdout.
   int isock;
   if ((isock = dup(fSocket->GetDescriptor())) < 0)
      SysError("RedirectOutput", "could not duplicate output socket");
   fSocket->SetDescriptor(isock);

   // Remove all previous log files and create new log files.
   char logfile[512];

   if (IsMaster()) {
      sprintf(logfile, "%s/proof_%d.log", fSessionDir.Data(), gSystem->GetPid());
   } else {
      sprintf(logfile, "%s/proofs%d_%d.log", fSessionDir.Data(), fOrdinal,
              gSystem->GetPid());
   }

   if ((freopen(logfile, "w", stdout)) == 0)
      SysError("RedirectOutput", "could not freopen stdout");

   if ((dup2(fileno(stdout), fileno(stderr))) < 0)
      SysError("RedirectOutput", "could not redirect stderr");

   if ((fLogFile = fopen(logfile, "r")) == 0)
      SysError("RedirectOutput", "could not open logfile");
}

//______________________________________________________________________________
void TProofServ::Reset(const char *dir)
{
   // Reset PROOF environment to be ready for execution of next command.

   // First go to new directory.
   gDirectory->cd(dir);

   // Clear interpreter environment.
   gROOT->Reset();

   // Make sure current directory is empty (don't delete anything when
   // we happen to be in the ROOT memory only directory!?)
   if (gDirectory != gROOT) {
      TObject *obj;
      TIter next(gDirectory->GetList());
      while ((obj = next()))
         if (!obj->InheritsFrom(TTree::Class())) {
            gDirectory->GetList()->Remove(obj);
            delete obj;
         }
   }
}

//______________________________________________________________________________
Int_t TProofServ::ReceiveFile(const char *file, Bool_t bin, Long_t size)
{
   // Receive a file, either sent by a client or a master server.
   // If bin is true it is a binary file, other wise it is an ASCII
   // file and we need to check for Windows \r tokens. Returns -1 in
   // case of error, 0 otherwise.

   if (size <= 0) return 0;

   // open file, overwrite already existing file
   int fd = open(file, O_CREAT | O_TRUNC | O_WRONLY, 0600);
   if (fd < 0) {
      SysError("ReceiveFile", "error opening file %s", file);
      return -1;
   }

   const Int_t kMAXBUF = 16384;  //32768  //16384  //65536;
   char buf[kMAXBUF], cpy[kMAXBUF];

   Int_t  left, r;
   Long_t filesize = 0;

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
void TProofServ::Run(Bool_t retrn)
{
   // Main server eventloop.

   TApplication::Run(retrn);
}

//______________________________________________________________________________
void TProofServ::SendLogFile(Int_t status)
{
   // Send log file to master.

   // Determine the number of bytes left to be read from the log file.
   fflush(stdout);

   off_t ltot, lnow;
   Int_t left;

   ltot = lseek(fileno(stdout),   (off_t) 0, SEEK_END);
   lnow = lseek(fileno(fLogFile), (off_t) 0, SEEK_CUR);
   left = Int_t(ltot - lnow);

   if (left > 0) {
      fSocket->Send(left, kPROOF_LOGFILE);

      const Int_t kMAXBUF = 32768;  //16384  //65536;
      char buf[kMAXBUF];
      Int_t len;
      do {
         while ((len = read(fileno(fLogFile), buf, kMAXBUF)) < 0 &&
                TSystem::GetErrno() == EINTR)
            TSystem::ResetErrno();

         if (len < 0) {
            SysError("SendLogFile", "error reading log file");
            break;
         }

         if (fSocket->SendRaw(buf, len) < 0) {
            SysError("SendLogFile", "error sending log file");
            break;
         }

      } while (len > 0);
   }

   TMessage mess(kPROOF_LOGDONE);
   if (IsMaster())
      mess << status << fProof->GetNumberOfActiveSlaves();
   else
      mess << status << (Int_t) 1;

   fSocket->Send(mess);
}

//______________________________________________________________________________
void TProofServ::SendStatus()
{
   // Send status of slave server to master or client.

   if (!IsMaster()) {
      TMessage mess(kPROOF_STATUS);
      TString workdir = gSystem->WorkingDirectory();  // expect TString on other side
      mess << TFile::GetFileBytesRead() << fRealTime << fCpuTime << workdir;
      fSocket->Send(mess);
   } else {
      fSocket->Send(fProof->GetNumberOfActiveSlaves(), kPROOF_STATUS);
   }
}

//______________________________________________________________________________
void TProofServ::Setup()
{
   // Print the ProofServ logo on standard output.

   char str[512];

   if (IsMaster()) {
      sprintf(str, "**** Welcome to the PROOF server @ %s ****", gSystem->HostName());
   } else {
      sprintf(str, "**** PROOF slave server @ %s started ****", gSystem->HostName());
   }
   fSocket->Send(str);

   TMessage *mess;
   fSocket->Recv(mess);

   if (IsMaster()) {

      (*mess) >> fUser >> fPasswd >> fConfFile >> fProtocol;

      for (int i = 0; i < fPasswd.Length(); i++) {
         char inv = ~fPasswd(i);
         fPasswd.Replace(i, 1, inv);
      }
   } else
      (*mess) >> fUser >> fProtocol >> fOrdinal;

   delete mess;

   // deny write access for group and world
   gSystem->Umask(022);

   if (IsMaster())
      gSystem->Openlog("proofserv", kLogPid | kLogCons, kLogLocal5);
   else
      gSystem->Openlog("proofslave", kLogPid | kLogCons, kLogLocal6);

   // Set $HOME and $PATH. The HOME directory was already set to the
   // user's home directory by proofd.
   gSystem->Setenv("HOME", gSystem->HomeDirectory());

#ifdef R__UNIX
   TString bindir;
# ifdef ROOTBINDIR
   bindir = ROOTBINDIR;
# else
   bindir = gSystem->Getenv("ROOTSYS");
   if (!bindir.IsNull()) bindir += "/bin";
# endif
   if (!bindir.IsNull()) bindir += ":";
   bindir += "/bin:/usr/bin:/usr/local/bin";
   gSystem->Setenv("PATH", bindir);
#endif

   // goto to the "~/proof" main PROOF working directory
   char *workdir = gSystem->ExpandPathName(kPROOF_WorkDir);

   if (gSystem->AccessPathName(workdir)) {
      gSystem->MakeDirectory(workdir);
      if (!gSystem->ChangeDirectory(workdir)) {
         SysError("Setup", "can not change to PROOF directory %s",
                  workdir);
      }
   } else {
      if (!gSystem->ChangeDirectory(workdir)) {
         gSystem->Unlink(workdir);
         gSystem->MakeDirectory(workdir);
         if (!gSystem->ChangeDirectory(workdir)) {
            SysError("Setup", "can not change to PROOF directory %s",
                     workdir);
         }
      }
   }

   // check and make sure "cache" directory exists
   fCacheDir = workdir;
   fCacheDir += TString("/") + kPROOF_CacheDir;
   if (gSystem->AccessPathName(fCacheDir))
      gSystem->MakeDirectory(fCacheDir);

   fCacheLock = fCacheDir + "/" + kPROOF_LockFile;

   // check and make sure "packages" directory exists
   fPackageDir = workdir;
   fPackageDir += TString("/") + kPROOF_PackDir;
   if (gSystem->AccessPathName(fPackageDir))
      gSystem->MakeDirectory(fPackageDir);

   fPackageLock = fPackageDir + "/" + kPROOF_LockFile;

   // create session directory and make it the working directory
   fSessionDir = workdir;
   if (IsMaster())
      fSessionDir += "/master-";
   else
      fSessionDir += "/slave-";
   fSessionDir += TTimeStamp().GetSec();
   fSessionDir += "-";
   fSessionDir += gSystem->GetPid();

   if (gSystem->AccessPathName(fSessionDir)) {
      gSystem->MakeDirectory(fSessionDir);
      if (!gSystem->ChangeDirectory(fSessionDir)) {
         SysError("Setup", "can not change to working directory %s",
                  fSessionDir.Data());
      }
   }

   delete [] workdir;

   // Incoming OOB should generate a SIGURG
   fSocket->SetOption(kProcessGroup, gSystem->GetPid());

   // Send packages off immediately to reduce latency
   fSocket->SetOption(kNoDelay, 1);

   // Check every two hours if client is still alive
   fSocket->SetOption(kKeepAlive, 1);

   // Install SigPipe handler to handle kKeepAlive failure
   gSystem->AddSignalHandler(new TProofServSigPipeHandler(this));
}

//______________________________________________________________________________
void TProofServ::Terminate(int status)
{
   // Terminate the proof server.

   // Cleanup session directory
   if (status == 0)
      gSystem->Exec(Form("%s %s", kRM, fSessionDir.Data()));

   gSystem->Exit(status);
}

//______________________________________________________________________________
Bool_t TProofServ::IsActive()
{
   // Static function that returns kTRUE in case we are a PROOF server.

   return gProofServ ? kTRUE : kFALSE;
}

//______________________________________________________________________________
TProofServ *TProofServ::This()
{
   // Static function returning pointer to global object gProofServ.
   // Mainly for use via CINT, where the gProofServ symbol might be
   // deleted from the symbol table.

   return gProofServ;
}

