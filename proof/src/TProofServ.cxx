// @(#)root/proof:$Name:  $:$Id: TProofServ.cxx,v 1.71 2004/05/18 11:32:49 rdm Exp $
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

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#ifdef WIN32
   #include <io.h>
   typedef long off_t;
#endif
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef __APPLE__
#include <AvailabilityMacros.h>
#endif
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

#include "TProofServ.h"

#include "TAuthenticate.h"
#include "TDSetProxy.h"
#include "TEnv.h"
#include "TError.h"
#include "TException.h"
#include "TFile.h"
#include "TInterpreter.h"
#include "TMessage.h"
#include "TPerfStats.h"
#include "TProofDebug.h"
#include "TProof.h"
#include "TProofLimitsFinder.h"
#include "TProofPlayer.h"
#include "TROOT.h"
#include "TSocket.h"
#include "TStopwatch.h"
#include "TSysEvtHandler.h"
#include "TSystem.h"
#include "TTimeStamp.h"
#include "TUrl.h"

#include "compiledata.h"

#ifndef R__WIN32
const char* const kCP     = "/bin/cp -f";
const char* const kRM     = "/bin/rm -rf";
const char* const kLS     = "/bin/ls -l";
const char* const kUNTAR  = "%s -c %s/%s | (cd %s; tar xf -)";
const char* const kGUNZIP = "gunzip";
#else
const char* const kCP     = "copy";
const char* const kRM     = "delete";
const char* const kLS     = "dir";
const char* const kUNTAR  = "...";
const char* const kGUNZIP = "gunzip";
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

   TString node = gProofServ->IsMaster() ? "master" : "slave ";
   if (node != "master") node += gProofServ->GetOrdinal();
   char *bp;

   if (!location || strlen(location) == 0 ||
       (level >= kBreak && level < kSysError)) {
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

   // get socket to be used (setup in proofd)
   Int_t sock = atoi(argv[8]);

   // make sure all registered dictionaries have been initialized
   // and that all types have been loaded
   gInterpreter->InitializeDictionaries();
   gInterpreter->UpdateListOfTypes();

   // abort on higher than kSysError's and set error handler
   gErrorAbortLevel = kSysError + 1;
   SetErrorHandler(ProofServErrorHandler);

   fNcmd            = 0;
   fInterrupt       = kFALSE;
   fProtocol        = 0;
   fOrdinal         = -1;
   fGroupId         = -1;
   fGroupSize       = 0;
   fLogLevel        = gEnv->GetValue("Proof.DebugLevel",0);
   fRealTime        = 0.0;
   fCpuTime         = 0.0;
   fProof           = 0;
   fSocket          = new TSocket(sock);
   fEnabledPackages = new TList;
   fEnabledPackages->SetOwner();

   gProofDebugLevel = gEnv->GetValue("Proof.DebugLevel",0);
   gProofDebugMask = (TProofDebug::EProofDebugMask) gEnv->GetValue("Proof.DebugMask",~0);

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
   ProcessLine("#include <iostream>", kTRUE);
   ProcessLine("#include <_string>",kTRUE); // for std::string iostream.

   // Allow the usage of ClassDef and ClassImp in interpreted macros
   ProcessLine("#include <RtypesCint.h>", kTRUE);

   // The following libs are also useful to have, make sure they are loaded...
   gROOT->LoadClass("TGeometry",   "Graf3d");
   gROOT->LoadClass("TTree",       "Tree");
   gROOT->LoadClass("TMatrix",     "Matrix");
   gROOT->LoadClass("TMinuit",     "Minuit");
   gROOT->LoadClass("TPostScript", "Postscript");
   gROOT->LoadClass("TCanvas",     "Gpad");

   // Load user functions
   const char *logon;
   logon = gEnv->GetValue("Proof.Load", (char*)0);
   if (logon) {
      char *mac = gSystem->Which(TROOT::GetMacroPath(), logon, kReadPermission);
      if (mac)
         ProcessLine(Form(".L %s", logon), kTRUE);
      delete [] mac;
   }

   // Execute logon macro
   logon = gEnv->GetValue("Proof.Logon", (char*)0);
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
   gSystem->AddSignalHandler(new TProofServInterruptHandler(this));
   gSystem->AddFileHandler(new TProofServInputHandler(this, sock));

   gProofServ = this;

   // if master, start slave servers
   if (IsMaster()) {
      TString master = "proof://__master__";
      TInetAddress a = gSystem->GetSockName(sock);
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

   delete fEnabledPackages;
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
   Long64_t size;
   Long_t id, flags, modtime, lasttime;
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
TDSetElement *TProofServ::GetNextPacket()
{
   // Get next range of entries to be processed on this server.

   Long64_t bytesRead = 0;

   if (gPerfStats != 0) bytesRead = gPerfStats->GetBytesRead();

   if (fCompute.Counter() > 0)
      fCompute.Stop();

   TMessage req(kPROOF_GETPACKET);
   req << fLatency.RealTime() << fCompute.RealTime() << fCompute.CpuTime() << bytesRead;

   fLatency.Start();
   Int_t rc = fSocket->Send(req);
   if (rc <= 0) {
      Error("GetNextPacket","Send() failed, returned %d", rc);
      return 0;
   }

   TMessage *mess;
   if ((rc = fSocket->Recv(mess)) <= 0) {
      fLatency.Stop();
      Error("GetNextPacket","Recv) failed, returned %d", rc);
      return 0;
   }

   fLatency.Stop();

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
      fCompute.Start();
      PDB(kLoop, 2) Info("GetNextPacket", "'%s' '%s' '%s' %lld %lld",
                         e->GetFileName(), e->GetDirectory(),
                         e->GetObjName(), e->GetFirst(),e->GetNum());
   } else {
      PDB(kLoop, 2) Info("GetNextPacket", "Done");
   }

   return e;
}

//______________________________________________________________________________
void TProofServ::GetOptions(int *argc, char **argv)
{
   // Get and handle command line options. Fixed format:
   // "proofserv"|"proofslave" <confdir>

   if (*argc <= 1) {
      fprintf(stderr, "proofserv: needs to be started from proofd with arguments\n");
      exit(1);
   }

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

   if (fProof) fProof->SetActive();

   switch (what) {

      case kMESS_CINT:
         mess->ReadString(str, sizeof(str));
         if (IsMaster() && IsParallel()) {
            fProof->SendCommand(str);
         } else {
            PDB(kGlobal, 1) Info("HandleSocketInput:kMESS_CINT", "processing: %s...", str);
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
         {
            UInt_t mask;
            mess->ReadString(str, sizeof(str));
            sscanf(str, "%d %u", &fLogLevel, &mask);
            gProofDebugLevel = fLogLevel;
            gProofDebugMask  = (TProofDebug::EProofDebugMask) mask;
            if (IsMaster())
               fProof->SetLogLevel(fLogLevel, mask);
         }
         break;

      case kPROOF_PING:
         if (IsMaster())
            fProof->Ping();
         // do nothing (ping is already acknowledged)
         break;

      case kPROOF_PRINT:
         mess->ReadString(str, sizeof(str));
         Print(str);
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

      case kPROOF_PROCESS:
         {
            TDSet *dset;
            TString filename, opt;
            TList *input;
            Long64_t nentries, first;

            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_PROCESS", "Enter");

            (*mess) >> dset >> filename >> input >> opt >> nentries >> first;

            if ( input == 0 ) {
               Error("HandleSocketInput:kPROOF_PROCESS", "input == 0");
            } else {
               PDB(kGlobal, 1) input->Print();
            }

            TProofPlayer *p;

            if (IsMaster()) {
               p = new TProofPlayerRemote(fProof);
               fProof->SetPlayer(p);
            } else {
               p = new TProofPlayerSlave(fSocket);
            }

            if (dset->IsA() == TDSetProxy::Class()) {
               ((TDSetProxy*)dset)->SetProofServ(this);
            }

            TIter next(input);
            for (TObject *obj; (obj = next()); ) {
               PDB(kGlobal, 2) Info("HandleSocketInput:kPROOF_PROCESS", "Adding: %s", obj->GetName());
               p->AddInput(obj);
            }

            p->Process(dset, filename, opt, nentries, first);

            // return output!

            PDB(kGlobal, 2) Info("HandleSocketInput:kPROOF_PROCESS","Send Output");
            fSocket->SendObject(p->GetOutputList(), kPROOF_OUTPUTLIST);

            PDB(kGlobal, 2) Info("HandleSocketInput:kPROOF_PROCESS","Send LogFile");

            SendLogFile();

            delete dset;

            if (fProof != 0) fProof->SetPlayer(0); // ensure player is no longer referenced
            delete p;
            delete input;

            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_PROCESS","Done");
         }
         break;

      case kPROOF_REPORTSIZE:
         {
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_REPORTSIZE", "Enter");
            Bool_t         isTree;
            TString        filename;
            TString        dir;
            TString        objname;
            Long64_t       entries;

            (*mess) >> isTree >> filename >> dir >> objname;

            PDB(kGlobal, 2) Info("HandleSocketInput:kPROOF_REPORTSIZE",
                                 "Report size of object %s (%s) in dir %s in file %s",
                                 objname.Data(), isTree ? "T" : "O",
                                 dir.Data(), filename.Data());

            entries = TDSet::GetEntries(isTree, filename, dir, objname);

            PDB(kGlobal, 2) Info("HandleSocketInput:kPROOF_REPORTSIZE",
                                 "Found %lld %s", entries, isTree ? "entries" : "objects");

            TMessage answ(kPROOF_REPORTSIZE);
            answ << entries;
            SendLogFile(); // in case of error messages
            fSocket->Send(answ);
            PDB(kGlobal, 1) Info("HandleSocketInput:kPROOF_REPORTSIZE", "Done");
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
               Int_t  st  = 0;
               Bool_t err = kFALSE;
               filenam = filenam.Strip(TString::kLeading, '-');
               TString packnam = filenam;
               packnam.Remove(packnam.Length() - 4);  // strip off ".par"
               // compare md5's to check if transmission was ok
               TMD5 *md5local = TMD5::FileChecksum(fPackageDir + "/" + filenam);
               if (md5local && md5 == (*md5local)) {
                  // remove any previous package directory with same name
                  st = gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
                                     packnam.Data()));
                  if (st)
                     Error("HandleInputSocket:kPROOF_CHECKFILE", "failure executing: %s %s/%s",
                           kRM, fPackageDir.Data(), packnam.Data());
                  // find gunzip...
                  char *gunzip = gSystem->Which(gSystem->Getenv("PATH"),kGUNZIP,
                                                kExecutePermission);
                  if (gunzip) {
                     // untar package
                     st = gSystem->Exec(Form(kUNTAR, gunzip, fPackageDir.Data(),
                                        filenam.Data(), fPackageDir.Data()));
                     if (st)
                        Error("HandleInputSocket:kPROOF_CHECKFILE", "failure executing: %s",
                              Form(kUNTAR, gunzip, fPackageDir.Data(),
                                   filenam.Data(), fPackageDir.Data()));
                     delete [] gunzip;
                  } else
                     Error("HandleInputSocket:kPROOF_CHECKFILE", "%s not found",
                           kGUNZIP);
                  // check that fPackageDir/packnam now exists
                  if (gSystem->AccessPathName(fPackageDir + "/" + packnam, kWritePermission)) {
                     // par file did not unpack itself in the expected directory, failure
                     fSocket->Send(kPROOF_FATAL);
                     err = kTRUE;
                     PDB(kPackage, 1)
                        Info("HandleSocketInput:kPROOF_CHECKFILE",
                             "package %s did not unpack into %s", filenam.Data(),
                             packnam.Data());
                  } else {
                     // store md5 in package/PROOF-INF/md5.txt
                     TString md5f = fPackageDir + "/" + packnam + "/PROOF-INF/md5.txt";
                     TMD5::WriteChecksum(md5f, md5local);
                     fSocket->Send(kPROOF_CHECKFILE);
                     PDB(kPackage, 1)
                        Info("HandleSocketInput:kPROOF_CHECKFILE",
                             "package %s installed on node", filenam.Data());
                  }
               } else {
                  fSocket->Send(kPROOF_FATAL);
                  err = kTRUE;
               }
               if (!IsMaster() || err) {
                  // delete par file when on slave or in case of error
                  gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
                                filenam.Data()));
               } else
                  // forward to slaves
                  fProof->UploadPackage(fPackageDir + "/" + filenam);
               delete md5local;
               UnlockPackage();
            } else if (filenam.BeginsWith("+")) {
               // check file in package directory
               filenam = filenam.Strip(TString::kLeading, '+');
               TString packnam = filenam;
               packnam.Remove(packnam.Length() - 4);  // strip off ".par"
               TString md5f = fPackageDir + "/" + packnam + "/PROOF-INF/md5.txt";
               LockPackage();
               TMD5 *md5local = TMD5::ReadChecksum(md5f);
               if (md5local && md5 == (*md5local)) {
                  // package already on server, unlock directory
                  UnlockPackage();
                  fSocket->Send(kPROOF_CHECKFILE);
                  PDB(kPackage, 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE",
                          "package %s already on node", filenam.Data());
                  if (IsMaster())
                     fProof->UploadPackage(fPackageDir + "/" + filenam);
               } else {
                  fSocket->Send(kPROOF_FATAL);
                  PDB(kPackage, 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE",
                          "package %s not yet on node", filenam.Data());
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
                  PDB(kPackage, 1)
                     Info("HandleSocketInput:kPROOF_CHECKFILE", "file %s already on node", filenam.Data());
               } else {
                  fSocket->Send(kPROOF_FATAL);
                  PDB(kPackage, 1)
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
            // 4 = ClearPackages, 5 = ClearPackage, 6 = BuildPackage,
            // 7 = LoadPackage, 8 = ShowEnabledPackages
            Int_t  status = 0;
            Int_t  type;
            Bool_t all;  //build;
            TString package, pdir, ocwd;
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
                  fEnabledPackages->Delete();
                  UnlockPackage();
                  if (IsMaster())
                     fProof->ClearPackages();
                  break;
               case 5:
                  (*mess) >> package;
                  LockPackage();
                  // remove package directory and par file
                  gSystem->Exec(Form("%s %s/%s", kRM, fPackageDir.Data(),
                                package.Data()));
                  gSystem->Exec(Form("%s %s/%s.par", kRM, fPackageDir.Data(),
                                package.Data()));
                  delete fEnabledPackages->Remove(fEnabledPackages->FindObject(package));
                  UnlockPackage();
                  if (IsMaster())
                     fProof->ClearPackage(package);
                  break;
               case 6:
                  (*mess) >> package;
                  LockPackage();
                  // check that package and PROOF-INF directory exists
                  pdir = fPackageDir + "/" + package;
                  if (gSystem->AccessPathName(pdir)) {
                     Error("HandleSocketInput:kPROOF_CACHE", "package %s does not exist",
                           package.Data());
                     status = -1;
                  } else if (gSystem->AccessPathName(pdir + "/PROOF-INF")) {
                     Error("HandleSocketInput:kPROOF_CACHE", "package %s does not have a PROOF-INF directory",
                           package.Data());
                     status = -1;
                  }

                  if (!status) {

                     PDB(kPackage, 1)
                        Info("HandleSocketInput:kPROOF_CACHE",
                             "package %s exists and has PROOF-INF directory", package.Data());

                     ocwd = gSystem->WorkingDirectory();
                     gSystem->ChangeDirectory(pdir);

                     // check for BUILD.sh and execute
                     if (!gSystem->AccessPathName(pdir + "/PROOF-INF/BUILD.sh")) {
                         if (gSystem->Exec("PROOF-INF/BUILD.sh"))
                            status = -1;
                     }

                     gSystem->ChangeDirectory(ocwd);

                  }
                  UnlockPackage();
                  // if built successful propagate to slaves
                  if (!status) {
                     if (IsMaster())
                        fProof->BuildPackage(package);

                     PDB(kPackage, 1)
                        Info("HandleSocketInput:kPROOF_CACHE",
                             "package %s successfully built", package.Data());
                  }
                  break;
               case 7:
                  (*mess) >> package;
                  // always follows on case 6 so no need to check for PROOF-INF
                  pdir = fPackageDir + "/" + package;

                  ocwd = gSystem->WorkingDirectory();
                  gSystem->ChangeDirectory(pdir);

                  // check for SETUP.C and execute
                  if (!gSystem->AccessPathName(pdir + "/PROOF-INF/SETUP.C")) {
                     gROOT->Macro("PROOF-INF/SETUP.C");
                  }

                  gSystem->ChangeDirectory(ocwd);

                  // create link to package in working directory
                  gSystem->Symlink(pdir, package);

                  // add package to list of include directories to be searched
                  // by ACliC
                  gSystem->AddIncludePath(TString("-I") + package);

                  // add package to list of include directories to be searched
                  // by CINT
                  gROOT->ProcessLine(TString(".include ") + package);

                  // if successful add to list and propagate to slaves
                  if (!status) {
                     fEnabledPackages->Add(new TObjString(package));
                     if (IsMaster())
                        fProof->LoadPackage(package);

                     PDB(kPackage, 1)
                         Info("HandleSocketInput:kPROOF_CACHE",
                              "package %s successfully loaded", package.Data());
                  }
                  break;
               case 8:
                  (*mess) >> all;
                  if (IsMaster())
                     printf("*** Enabled packages ***\n");
                  else
                     printf("*** Enabled packages on slave %d on %s\n", fOrdinal,
                            gSystem->HostName());
                  {
                     TIter next(fEnabledPackages);
                     while (TObjString *str = (TObjString*) next())
                        printf("%s\n", str->GetName());
                  }
                  if (IsMaster() && all)
                     fProof->ShowEnabledPackages(all);
                  break;
               default:
                  Error("HandleSocketInput:kPROOF_CACHE", "unknown type %d", type);
                  break;
            }
            SendLogFile(status);
         }
         break;

      default:
         Error("HandleSocketInput", "unknown command %d", what);
         break;
   }

   if (fProof) fProof->SetActive(kFALSE);


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

   PDB(kGlobal, 5)
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

   PDB(kGlobal, 5)
      Info("HandleUrgentData", "got OOB byte: %d\n", oob_byte);

   if (fProof) fProof->SetActive();

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

   if (fProof) fProof->SetActive(kFALSE);
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

         fProof->SetActive();
         fProof->Interrupt(TProof::kShutdownInterrupt);
         fProof->SetActive(kFALSE);
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
#if !defined(R__WIN32) && !defined(R__WINGCC)
   if (lockf(*fid, F_LOCK, (off_t) 1) == -1) {
      SysError("LockDir", "error locking %s", lfile);
      close(*fid);
      *fid = -1;
      return -1;
   }
#endif

   PDB(kPackage, 2)
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
#if !defined(R__WIN32) && !defined(R__WINGCC)
   if (lockf(*fid, F_ULOCK, (off_t)1) == -1) {
      SysError("UnlockDir", "error unlocking %s", lock.Data());
      close(*fid);
      *fid = -1;
      return -1;
   }
#endif

   PDB(kPackage, 2)
      Info("UnlockDir", "file %s unlocked", lock.Data());

   close(*fid);
   *fid = -1;

   return 0;
}

//______________________________________________________________________________
void TProofServ::Print(Option_t *option) const
{
   // Print status of slave server.

   if (IsMaster())
      fProof->Print(option);
   else
      Printf("This is slave %s", gSystem->HostName());
}

//______________________________________________________________________________
void TProofServ::RedirectOutput()
{
   // Redirect stdout to a log file. This log file will be flushed to the
   // client or master after each command.

   char logfile[512];

   if (IsMaster()) {
      sprintf(logfile, "%s/master.log", fSessionDir.Data());
   } else {
      sprintf(logfile, "%s/slave-%d.log", fSessionDir.Data(), fOrdinal);
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
      gDirectory->Delete();
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

   // exchange protocol level between client and master and between
   // master and slave
   Int_t what;
   fSocket->Recv(fProtocol, what);
   fSocket->Send(kPROOF_Protocol, kROOTD_PROTOCOL);

   // First receive, decode and store the public part of RSA key
   int retval, kind;
   fSocket->Recv(retval,kind);

   TApplication *lApp = gROOT->GetApplication();
   if (kind == kROOTD_RSAKEY) {

      if (retval > -1) {

         if (lApp && lApp->Argc() > 3 && strlen(lApp->Argv(3)) > 0 &&
             gROOT->IsProofServ()) {
            // We got a file name ... extract the tmp directory path
            TString KeyFile = lApp->Argv(3);
            KeyFile += "/rpk_";
            KeyFile += retval;

            FILE *fKey = 0;
            char PubKey[kMAXPATHLEN] = { 0 };
            if (!gSystem->AccessPathName(KeyFile.Data(), kReadPermission)) {
               fKey = fopen(KeyFile.Data(), "r");
               if (fKey) {
                  fgets(PubKey, sizeof(PubKey), fKey);
                  // Set RSA key
                  TAuthenticate::SetRSAPublic(PubKey);
                  fclose(fKey);
               }
            }
         }

         // Receive passwd
         fSocket->SecureRecv(fPasswd, 2);
      } else if (retval == -1) {

         // Receive inverted passwd
         TMessage *mess;
         fSocket->Recv(mess);
         (*mess) >> fPasswd;
         delete mess;

         for (int i = 0; i < fPasswd.Length(); i++) {
            char inv = ~fPasswd(i);
            fPasswd.Replace(i, 1, inv);
         }

      }
   }

   // Receive user and passwd information
   TMessage *mess;

   fSocket->Recv(mess);

   if (IsMaster())
      (*mess) >> fUser >> fPwHash >> fSRPPwd >> fConfFile;
   else
      (*mess) >> fUser >> fPwHash >> fSRPPwd >> fOrdinal;

   delete mess;

   // Set Globals for later use
   TAuthenticate::SetGlobalUser(fUser);
   TAuthenticate::SetGlobalPasswd(fPasswd);
   TAuthenticate::SetGlobalPwHash(fPwHash);
   TAuthenticate::SetGlobalSRPPwd(fSRPPwd);
   if (lApp && lApp->Argc() > 7 && strlen(lApp->Argv(7)) > 0) {
      Bool_t rha = (Bool_t)atoi(lApp->Argv(7));
      TAuthenticate::SetReadHomeAuthrc(rha);
   }

   // Read user or system authentication directives and
   // receive auth info transmitted from the client

   if (IsMaster())
      fSocket->RecvHostAuth("M", fConfFile);
   else
      fSocket->RecvHostAuth("S");

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
# ifdef COMPILER
   TString compiler = COMPILER;
   compiler.Remove(0, compiler.Index("is ") + 3);
   compiler = gSystem->DirName(compiler);
   if (!bindir.IsNull()) bindir += ":";
   bindir += compiler;
#endif
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

   fCacheLock = kPROOF_CacheLockFile;
   fCacheLock += fUser;

   // check and make sure "packages" directory exists
   fPackageDir = workdir;
   fPackageDir += TString("/") + kPROOF_PackDir;
   if (gSystem->AccessPathName(fPackageDir))
      gSystem->MakeDirectory(fPackageDir);

   fPackageLock = kPROOF_PackageLockFile;
   fPackageLock += fUser;

   // create session directory and make it the working directory
   TString host = gSystem->HostName();
   if (host.Index(".") != kNPOS)
      host.Remove(host.Index("."));
   fSessionDir = workdir;
   if (IsMaster())
      fSessionDir += "/master-";
   else {
      fSessionDir += "/slave-";
      fSessionDir += fOrdinal;
      fSessionDir += "-";
   }
   fSessionDir += host + "-";
   fSessionDir += TTimeStamp().GetSec();
   fSessionDir += "-";
   fSessionDir += gSystem->GetPid();

   if (gSystem->AccessPathName(fSessionDir)) {
      gSystem->MakeDirectory(fSessionDir);
      if (!gSystem->ChangeDirectory(fSessionDir)) {
         SysError("Setup", "can not change to working directory %s",
                  fSessionDir.Data());
      } else {
         gSystem->Setenv("PROOF_SANDBOX", fSessionDir);
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
void TProofServ::Terminate(Int_t status)
{
   // Terminate the proof server.

   // Cleanup session directory
   if (status == 0) {
      gSystem->ChangeDirectory("/"); // make sure we remain in a "connected" directory
      gSystem->MakeDirectory(fSessionDir+"/.delete");  // needed in case fSessionDir is on NFS ?!
      gSystem->Exec(Form("%s %s", kRM, fSessionDir.Data()));
   }

   // Remove input handler to avoid spurious signals in socket
   // selection for closing activities executed upon exit()
   TIter next(gSystem->GetListOfFileHandlers());
   TObject *fh = 0;
   while ((fh = next())) {
      TProofServInputHandler *ih = dynamic_cast<TProofServInputHandler *>(fh);
      if (ih)
         gSystem->RemoveFileHandler(ih);
   }

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
