// @(#)root/proof:$Name:  $:$Id: TProof.cxx,v 1.9 2000/12/19 14:34:31 rdm Exp $
// Author: Fons Rademakers   13/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProof                                                               //
//                                                                      //
// This class controls a Parallel ROOT Facility, PROOF, cluster.        //
// It fires the slave servers, it keeps track of how many slaves are    //
// running, it keeps track of the slaves running status, it broadcasts  //
// messages to all slaves, it collects results, etc.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <errno.h>
#ifdef WIN32
#   include <io.h>
#   include <sys/stat.h>
#   include <sys/types.h>
#else
#   include <unistd.h>
#endif

#include "TProof.h"
#include "TAuthenticate.h"
#include "TSortedList.h"
#include "TSlave.h"
#include "TSocket.h"
#include "TMonitor.h"
#include "TMessage.h"
#include "TSystem.h"
#include "TError.h"
#include "TUrl.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"


TProof *gProof = 0;


//----- Input handler for messages from TProofServ -----------------------------
//______________________________________________________________________________
class TProofInputHandler : public TFileHandler {
   TSocket *fSocket;
   TProof  *fProof;
public:
   TProofInputHandler(TProof *p, TSocket *s)
      : TFileHandler(s->GetDescriptor(), 1) { fProof = p; fSocket = s; }
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }
};

//______________________________________________________________________________
Bool_t TProofInputHandler::Notify()
{
   fProof->HandleAsyncInput(fSocket);
   return kTRUE;
}


ClassImp(TProof)

//______________________________________________________________________________
TProof::TProof(const char *masterurl, const char *conffile,
               const char *confdir, Int_t loglevel)
{
   // Create a PROOF environment. Starting PROOF involves either connecting
   // to a master server, which in turn will start a set of slave servers, or
   // directly starting as master server (if master = ""). Masterurl is of
   // the form: proof://host[:port] or proofs://host[:port]. Conffile is
   // the name of the config file describing the remote PROOF cluster
   // (this argument alows you to describe different cluster configurations).
   // The default proof.conf. Confdir is the directory where the config
   // file and other PROOF related files are (like motd and noproof files).
   // Loglevel is the og level (default = 1).

   // Can have only one PROOF session open at a time.
   if (gProof) {
      Warning("TProof", "closing currently open PROOF session");
      gProof->Close();
   }

   if (Init(masterurl, conffile, confdir, loglevel) == 0) {
      // on Init failure make sure IsValid() returns kFALSE
      SafeDelete(fActiveSlaves);
   }

   gProof = this;
}

//______________________________________________________________________________
TProof::~TProof()
{
   // Clean up PROOF environment.

   Close();

   SafeDelete(fSlaves);
   SafeDelete(fActiveSlaves);
   SafeDelete(fUniqueSlaves);
   SafeDelete(fBadSlaves);
   SafeDelete(fAllMonitor);
   SafeDelete(fActiveMonitor);

   gProof = 0;
}

//______________________________________________________________________________
Int_t TProof::Init(const char *masterurl, const char *conffile,
                   const char *confdir, Int_t loglevel)
{
   // Start the PROOF environment. Starting PROOF involves either connecting
   // to a master server, which in turn will start a set of slave servers, or
   // directly starting as master server (if master = ""). For a description
   // of the arguments see the TProof ctor.

   Assert(gSystem);

   TUrl *u;
   if (!masterurl || !*masterurl)
      u = new TUrl("proof://__master__");
   else if (strstr(masterurl, "://"))
      u = new TUrl(masterurl);
   else
      u = new TUrl(Form("proof://%s", masterurl));

   fMaster        = u->GetHost();
   fPort          = u->GetPort();
   fSecurity      = !strcmp(u->GetProtocol(), "proofs") ?
                    TAuthenticate::kSRP : TAuthenticate::kNormal;
   fConfDir       = confdir;
   fConfFile      = conffile;
   fWorkDir       = gSystem->WorkingDirectory();
   fLogLevel      = loglevel;
   fProtocol      = kPROOF_Protocol;
   fMasterServ    = fMaster == "__master__" ? kTRUE : kFALSE;
   fSendGroupView = kTRUE;
   fImage         = "";
   fStatus        = 0;
   fParallel      = 0;
   fTree          = 0;

   delete u;

   // sort slaves by descending performance index
   fSlaves        = new TSortedList(kSortDescending);
   fActiveSlaves  = new TList;
   fUniqueSlaves  = new TList;
   fBadSlaves     = new TList;
   fAllMonitor    = new TMonitor;
   fActiveMonitor = new TMonitor;

   // If this is a master server, find the config file and start slave
   // servers as specified in the config file
   if (IsMaster()) {

      // set in TProofServ
      fUser   = TAuthenticate::GetGlobalUser();
      fPasswd = TAuthenticate::GetGlobalPasswd();

      char fconf[256];
      sprintf(fconf, "%s/.%s", gSystem->Getenv("HOME"), conffile);
      if (gSystem->AccessPathName(fconf, kFileExists)) {
          sprintf(fconf, "%s/proof/etc/%s", confdir, conffile);
         if (gSystem->AccessPathName(fconf, kFileExists)) {
            Error("Init", "no PROOF config file found");
            return 0;
         }
      }
      if (gDebug > 1)
         Printf("Using PROOF config file: %s", fconf);

      FILE *pconf;
      if ((pconf = fopen(fconf, "r"))) {

         fConfFile = fconf;

         // read the config file
         char line[256];
         const char *host = gSystem->HostName();
         int  ord = 0;

         while (fgets(line, sizeof(line), pconf)) {
            char word[7][64];
            if (line[0] == '#') continue;   // skip comment lines
            int nword = sscanf(line, "%s %s %s %s %s %s %s", word[0], word[1],
                word[2], word[3], word[4], word[5], word[6]);

            // find node on which master runs
            if (nword >= 2 && !strcmp(word[0], "node") && !fImage.Length()) {
               TInetAddress a = gSystem->GetHostByName(word[1]);
               if (!strcmp(a.GetHostName(), host) ||
                   !strcmp(word[1], "localhost")) {
                  char *image = word[1];
                  if (nword > 2 && !strncmp(word[2], "image=", 6))
                     image = word[2]+6;
                  fImage = image;
               }
            }
            // find all slave servers
            if (nword >= 2 && !strcmp(word[0], "slave")) {
               int perfidx  = 100;
               int sport    = fPort;
               int security = TAuthenticate::kNormal;
               const char *image = word[1];
               for (int i = 2; i < nword; i++) {
                  if (!strncmp(word[i], "perf=", 5))
                     perfidx = atoi(word[i]+5);
                  if (!strncmp(word[i], "image=", 6))
                     image = word[i]+6;
                  if (!strncmp(word[i], "port=", 5))
                     sport = atoi(word[i]+5);
                  if (!strncmp(word[i], "srp", 3))
                     security = TAuthenticate::kSRP;
               }
               // create slave server
               TSlave *slave = new TSlave(word[1], sport, ord++, perfidx,
                                          image, security, this);
               fSlaves->Add(slave);
               if (slave->IsValid()) {
                  fAllMonitor->Add(slave->GetSocket());
                  slave->SetInputHandler(new TProofInputHandler(this,
                                         slave->GetSocket()));
               } else
                  fBadSlaves->Add(slave);
            }
         }
      }
      fclose(pconf);

      if (fImage.Length() == 0) {
         Error("Init", "no appropriate node line found in %s", fconf);
         return 0;
      }
   } else {
      // create master server
      TSlave *slave = new TSlave(fMaster, fPort, 0, 100, "master",
                                 fSecurity, this);
      if (slave->IsValid()) {
         fSlaves->Add(slave);
         fAllMonitor->Add(slave->GetSocket());
         Collect(slave);
         if (fStatus == -99) {
            Error("Init", "not allowed to connect to PROOF master server");
            return 0;
         }
         slave->SetInputHandler(new TProofInputHandler(this, slave->GetSocket()));
      } else {
         delete slave;
         Error("Init", "failed to connect to a PROOF master server");
         return 0;
      }
   }

   // De-activate monitor (will be activated in Collect)
   fAllMonitor->DeActivateAll();

   // By default go into parallel mode
   GoParallel(9999);

   // Send relevant initial state to slaves
   SendInitialState();

   if (IsValid())
      gROOT->GetListOfSockets()->Add(this);

   return fActiveSlaves->GetSize();
}

//______________________________________________________________________________
Int_t TProof::ConnectFile(const TFile *file)
{
   // Send message to all slaves to connect "file". This method is
   // called by the TFile ctor (no user method). Message is only send
   // if file was opened in READ mode.

   if (!IsValid() || !file) return 0;

   TString clsnam  = file->IsA()->GetName();
   TString filenam = file->GetName();
   TString option  = file->GetOption();

   // only propagate files opened in READ mode to PROOF servers
   if (option.CompareTo("READ", TString::kIgnoreCase))
      return 0;

   // A TFile can only be opened on all machines if the master and slaves
   // share the same file system image.
   if (clsnam == "TFile") {
      if (GetNumberOfUniqueSlaves() > 0)
         return 0;
      else {
         if (!gSystem->IsAbsoluteFileName(filenam)) {
            filenam = gSystem->WorkingDirectory();
            filenam += "/";
            filenam += file->GetName();
         }
      }
   }

   TMessage mess(kPROOF_OPENFILE);
   mess << clsnam << filenam << option;
   Broadcast(mess, kAll);
   return Collect(kAll);
}

//______________________________________________________________________________
void TProof::ConnectFiles()
{
   // Tell all servers to open all files currently opened by the client.

   if (!IsValid()) return;

   TIter  next(gROOT->GetListOfFiles());
   TFile *f;

   while ((f = (TFile *) next()))
      ConnectFile(f);
}

//______________________________________________________________________________
void TProof::Close(Option_t *)
{
   // Close all open slave servers.

   if (fSlaves) {
      //Broadcast(kPROOF_STOP, kAll);
      Interrupt(kShutdownInterrupt, kAll);

      fSlaves->Delete();
      fActiveSlaves->Clear();
      fUniqueSlaves->Clear();
      fBadSlaves->Clear();
   }
}

//______________________________________________________________________________
Int_t TProof::DisConnectFile(const TFile *file)
{
   // Send message to all slaves to disconnect "file". This method is
   // called by the TFile::Close() (no user method).

   if (!IsValid()) return 0;

   char str[512];
   sprintf(str, "{TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject(\"%s\"); if (f) f->Close();}", file->GetName());
   return SendCommand(str, kAll);
}

//______________________________________________________________________________
TSlave *TProof::FindSlave(TSocket *s) const
{
   // Find slave that has TSocket s. Returns 0 in case slave is not found.

   TSlave *sl;
   TIter   next(fSlaves);

   while ((sl = (TSlave *)next())) {
      if (sl->IsValid() && sl->GetSocket() == s)
         return sl;
   }
   return 0;
}

//______________________________________________________________________________
void TProof::FindUniqueSlaves()
{
   // Add to the fUniqueSlave list the active slaves that have a unique
   // (user) file system image. This information is used to transfer files
   // only once to nodes that share a file system (an image).

   fUniqueSlaves->Clear();

   TIter next(fActiveSlaves);

   TSlave *sl;
   while ((sl = (TSlave *)next())) {
      if (fImage == sl->fImage) continue;
      TIter next2(fUniqueSlaves);
      TSlave *sl2;
      Int_t   add = fUniqueSlaves->IsEmpty() ? 1 : 0;
      while ((sl2 = (TSlave *)next2())) {
         if (sl->fImage == sl2->fImage) continue;
         add++;
      }
      if (add)
         fUniqueSlaves->Add(sl);
   }
}

//______________________________________________________________________________
Int_t TProof::GetNumberOfSlaves() const
{
   // Return number of slaves as described in the config file.

   if (!fSlaves) return 0;
   return fSlaves->GetSize();
}

//______________________________________________________________________________
Int_t TProof::GetNumberOfActiveSlaves() const
{
   // Return number of active slaves, i.e. slaves that are valid and in
   // the current computing group.

   if (!fActiveSlaves) return 0;
   return fActiveSlaves->GetSize();
}

//______________________________________________________________________________
Int_t TProof::GetNumberOfUniqueSlaves() const
{
   // Return number of unique slaves, i.e. active slaves that have each a
   // unique different user files system.

   if (!fUniqueSlaves) return 0;
   return fUniqueSlaves->GetSize();
}

//______________________________________________________________________________
Int_t TProof::GetNumberOfBadSlaves() const
{
   // Return number of bad slaves. This are slaves that we in the config
   // file, but refused to startup or that died during the PROOF session.

   if (!fBadSlaves) return 0;
   return fBadSlaves->GetSize();
}

//______________________________________________________________________________
void TProof::AskStatus()
{
   // Ask the status of the slaves.

   if (!IsValid()) return;

   Broadcast(kPROOF_STATUS, kAll);
   Collect(kAll);
}

//______________________________________________________________________________
void TProof::Interrupt(EUrgent type, ESlaves list)
{
   // Send interrupt OOB byte to master or slave servers.

   if (!IsValid()) return;

   char oobc = (char) type;

   TList *slaves = 0;
   if (list == kAll)    slaves = fSlaves;
   if (list == kActive) slaves = fActiveSlaves;
   if (list == kUnique) slaves = fUniqueSlaves;

   if (slaves->GetSize() == 0) return;

   const int kBufSize = 1024;
   char waste[kBufSize];

   TSlave *sl;
   TIter   next(slaves);

   while ((sl = (TSlave *)next())) {
      if (sl->IsValid()) {
         TSocket *s = sl->GetSocket();

         // Send one byte out-of-band message to server
         if (s->SendRaw(&oobc, 1, kOob) <= 0) {
            Error("Interrupt", "error sending oobc to slave %d", sl->GetOrdinal());
            continue;
         }

         if (type == kHardInterrupt) {
            char  oob_byte;
            int   n, nch, nbytes = 0, nloop = 0;

            // Receive the OOB byte
            while ((n = s->RecvRaw(&oob_byte, 1, kOob)) < 0) {
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
                  s->GetOption(kBytesToRead, nch);
                  if (nch == 0) {
                     gSystem->Sleep(1000);
                     continue;
                  }

                  if (nch > kBufSize) nch = kBufSize;
                  n = s->RecvRaw(waste, nch);
                  if (n <= 0) {
                     Error("Interrupt", "error receiving waste from slave %d",
                           sl->GetOrdinal());
                     break;
                  }
                  nbytes += n;
               } else if (n == -3) {   // EINVAL
                  //
                  // The OOB data has not arrived yet
                  //
                  gSystem->Sleep(100);
                  if (++nloop > 100) {  // 10 seconds time-out
                     Error("Interrupt", "server %d does not respond", sl->GetOrdinal());
                     break;
                  }
               } else {
                  Error("Interrupt", "error receiving OOB from server %d",
                        sl->GetOrdinal());
                  break;
               }
            }

            //
            // Continue flushing the input socket stream until the OOB
            // mark is reached
            //
            while (1) {
               int atmark;

               s->GetOption(kAtMark, atmark);

               if (atmark)
                  break;

               // find out number of bytes to read before atmark
               s->GetOption(kBytesToRead, nch);
               if (nch == 0) {
                  gSystem->Sleep(1000);
                  continue;
               }

               if (nch > kBufSize) nch = kBufSize;
               n = s->RecvRaw(waste, nch);
               if (n <= 0) {
                  Error("Interrupt", "error receiving waste (2) from slave %d",
                        sl->GetOrdinal());
                  break;
               }
               nbytes += n;
            }
            if (nbytes > 0) {
               if (IsMaster())
                  Printf("*** Slave %d synchronized: %d bytes discarded",
                         sl->GetOrdinal(), nbytes);
               else
                  Printf("*** PROOF synchronized: %d bytes discarded", nbytes);
            }

            // Get log file from master or slave after a hard interrupt
            Collect(sl);

         } else if (type == kSoftInterrupt) {

            // Get log file from master or slave after a soft interrupt
            Collect(sl);

         } else if (type == kShutdownInterrupt) {

            ; // nothing expected to be returned

         } else {

            // Unexpected message, just receive log file
            Collect(sl);

         }
      }
   }
}

//______________________________________________________________________________
Bool_t TProof::IsParallel() const
{
   // Returns true if PROOF is in parallel mode.

   if (!IsValid()) return kFALSE;

   if (IsMaster())
      return GetNumberOfActiveSlaves() > 1 ? kTRUE : kFALSE;
   else
      return fParallel > 1 ? kTRUE : kFALSE;
}

//______________________________________________________________________________
Int_t TProof::Broadcast(const TMessage &mess, ESlaves list)
{
   // Broadcast a message to all slaves in the specified list (either
   // all slaves or only the active slaves). Returns the number of slaves
   // the message was successfully sent to.

   if (!IsValid()) return 0;

   TList *slaves = 0;
   if (list == kAll)    slaves = fSlaves;
   if (list == kActive) slaves = fActiveSlaves;
   if (list == kUnique) slaves = fUniqueSlaves;

   if (slaves->GetSize() == 0) return 0;

   int   nsent = 0;
   TIter next(slaves);

   TSlave *sl;
   while ((sl = (TSlave *)next())) {
      if (sl->IsValid()) {
         if (sl->GetSocket()->Send(mess) == -1)
            MarkBad(sl);
         else
            nsent++;
      }
   }

   return nsent;
}

//______________________________________________________________________________
Int_t TProof::Broadcast(const char *str, Int_t kind, ESlaves list)
{
   // Broadcast a character string buffer to all slaves in the specified
   // list (either all slaves or only the active slaves). Use kind to
   // set the TMessage what field. Returns the number of slaves the message
   // was sent to.

   TMessage mess(kind);
   if (str) mess.WriteString(str);
   return Broadcast(mess, list);
}

//______________________________________________________________________________
Int_t TProof::BroadcastObject(const TObject *obj, Int_t kind, ESlaves list)
{
   // Broadcast an object to all slaves in the specified list. Use kind to
   // set the TMEssage what field. Returns the number of slaves the message
   // was sent to.

   TMessage mess(kind);
   mess.WriteObject(obj);
   return Broadcast(mess, list);
}

//______________________________________________________________________________
Int_t TProof::BroadcastRaw(const void *buffer, Int_t length, ESlaves list)
{
   // Broadcast a raw buffer of specified length. Returns the number of slaves
   // the buffer was sent to.

   if (!IsValid()) return 0;

   TList *slaves = 0;
   if (list == kAll)    slaves = fSlaves;
   if (list == kActive) slaves = fActiveSlaves;
   if (list == kUnique) slaves = fUniqueSlaves;

   if (slaves->GetSize() == 0) return 0;

   int   nsent = 0;
   TIter next(slaves);

   TSlave *sl;
   while ((sl = (TSlave *)next())) {
      if (sl->IsValid()) {
         if (sl->GetSocket()->SendRaw(buffer, length) == -1)
            MarkBad(sl);
         else
            nsent++;
      }
   }

   return nsent;
}

//______________________________________________________________________________
Int_t TProof::Collect(const TSlave *sl)
{
   // Collect responses from slave sl. Returns the number of slaves that
   // responded (=1).

   if (!sl->IsValid()) return 0;

   TMonitor *mon = fAllMonitor;

   mon->DeActivateAll();
   mon->Activate(sl->GetSocket());

   return Collect(mon);
}

//______________________________________________________________________________
Int_t TProof::Collect(ESlaves list)
{
   // Collect responses from the slave servers. Returns the number of slaves
   // that responded.

   TMonitor *mon = 0;
   if (list == kAll)    mon = fAllMonitor;
   if (list == kActive) mon = fActiveMonitor;
   if (list == kUnique) {
      Error("Collect", "cannot be called with kUnique");
      return 0;
   }

   mon->ActivateAll();

   return Collect(mon);
}

//______________________________________________________________________________
Int_t TProof::Collect(TMonitor *mon)
{
   // Collect responses from the slave servers. Returns the number of messages
   // received. Can be 0 if there are no active slaves.

   fStatus = 0;
   if (!mon->GetActive()) return 0;

   DeActivateAsyncInput();

   int cnt = 0, loop = 1;

   fBytesRead = 0;
   fRealTime  = 0.0;
   fCpuTime   = 0.0;

   while (loop) {
      char      str[512];
      TMessage *mess;
      TSocket  *s;
      TSlave   *sl;
      TObject  *obj;
      Int_t     what;

      s = mon->Select();

      if (s->Recv(mess) < 0) {
         MarkBad(s);
         continue;
      }

      what = mess->What();

      switch (what) {

         case kMESS_OBJECT:
            obj = mess->ReadObject(mess->GetClass());
            if (obj->InheritsFrom(TH1::Class())) {
               TH1 *h = (TH1*)obj;
               h->SetDirectory(0);
               TH1 *horg = (TH1*)gDirectory->GetList()->FindObject(h->GetName());
               if (horg)
                  horg->Add(h);
               else
                  h->SetDirectory(gDirectory);
            }
            break;

         case kPROOF_LIMITS:
            if (fTree) Limits(s, *mess);
            break;

         case kPROOF_FATAL:
            MarkBad(s);
            if (!mon->GetActive()) loop = 0;
            break;

         case kPROOF_GETOBJECT:
            mess->ReadString(str, sizeof(str));
            obj = gDirectory->Get(str);
            if (obj)
               s->SendObject(obj);
            else
               s->Send(kMESS_NOTOK);
            break;

         case kPROOF_GETPACKET:
            if (fTree) {
               Int_t  nentries;
               Stat_t firstentry, processed;
               sl = FindSlave(s);
               fTree->GetPlayer()->GetNextPacket(sl, nentries, firstentry, processed);
               TMessage answ(kPROOF_GETPACKET);
               answ << nentries << firstentry << processed;
               s->Send(answ);
            }
            break;

         case kPROOF_LOGFILE:
            {
               Int_t size;
               (*mess) >> size;
               RecvLogFile(s, size);
            }
            break;

         case kPROOF_LOGDONE:
            (*mess) >> fStatus >> fParallel;
            mon->DeActivate(s);
            if (!mon->GetActive()) loop = 0;
            break;

         case kPROOF_STATUS:
            if (IsMaster()) {
               sl = FindSlave(s);
               (*mess) >> sl->fBytesRead >> sl->fRealTime >> sl->fCpuTime
                       >> sl->fWorkDir;
               fBytesRead += sl->fBytesRead;
               fRealTime  += sl->fRealTime;
               fCpuTime   += sl->fCpuTime;
            } else {
               (*mess) >> fParallel;
            }
            mon->DeActivate(s);
            if (!mon->GetActive()) loop = 0;
            break;

         default:
            Error("Collect", "unknown command received from slave (%d)", what);
            break;
      }

      cnt++;
      delete mess;
   }

   // make sure group view is up to date
   SendGroupView();

   ActivateAsyncInput();

   return cnt;
}

//______________________________________________________________________________
void TProof::ActivateAsyncInput()
{
   // Activate the a-sync input handler.

   TIter next(fSlaves);
   TSlave *sl;

   while ((sl = (TSlave*) next()))
      if (sl->GetInputHandler())
         sl->GetInputHandler()->Add();
}

//______________________________________________________________________________
void TProof::DeActivateAsyncInput()
{
   // De-actiate a-sync input handler.

   TIter next(fSlaves);
   TSlave *sl;

   while ((sl = (TSlave*) next()))
      if (sl->GetInputHandler())
         sl->GetInputHandler()->Remove();
}

//______________________________________________________________________________
void TProof::HandleAsyncInput(TSocket *sl)
{
   // Handle input coming from the master server (when this is a client)
   // or from a slave server (when this is a master server). This is mainly
   // for a-synchronous communication. Normally when PROOF issues a command
   // the (slave) server messages are directly handle by Collect().

   TMessage *mess;
   Int_t     what;

   if (sl->Recv(mess) < 0)
      return;                // do something more intelligent here

   what = mess->What();

   switch (what) {

      case kPROOF_PING:
         // do nothing (ping is already acknowledged)
         break;

      default:
         Error("HandleAsyncInput", "unknown command %d", what);
         break;
   }

   delete mess;
}

//______________________________________________________________________________
void TProof::Limits(TSocket *s, TMessage &mess)
{
   // Calculate histogram limits after TTree::fEstimate entries have
   // been processed.
   // This function is called via Collect() in response to a kPROOF_LIMITS
   // message send from a PROOF slave in TTree::TakeEstimate().

   static TObjArray arr;
   static Int_t     mxnbin[4], totevt;
   static Float_t   mxvmin[4], mxvmax[4];
   Int_t            dim, nentries, nbin[4];
   Float_t          vmin[4], vmax[4];

   mess >> dim >> nentries >> nbin[0] >> vmin[0] >> vmax[0];
   if (dim == 2)
      mess >> nbin[1] >> vmin[1] >> vmax[1];
   if (dim == 3)
      mess >> nbin[2] >> vmin[2] >> vmax[2];

   if (!fLimits) {
      arr.Clear();
      for (int i = 0; i < dim; i++) {
         mxnbin[i] = nbin[i];
         mxvmin[i] = vmin[i];
         mxvmax[i] = vmax[i];
      }
      totevt = nentries;
      arr.Add(s);
      fLimits++;
   } else if (totevt < fTree->GetEstimate()) {
      for (int i = 0; i < dim; i++) {
         mxnbin[i] = TMath::Max(mxnbin[i], nbin[i]);
         mxvmin[i] = TMath::Min(mxvmin[i], vmin[i]);
         mxvmax[i] = TMath::Max(mxvmax[i], vmax[i]);
      }
      totevt += nentries;
      arr.Add(s);
      fLimits++;
   }

   if (totevt >= fTree->GetEstimate() || fLimits == GetNumberOfActiveSlaves()) {
      TMessage msg(kPROOF_LIMITS);
      msg << mxnbin[0] << mxvmin[0] << mxvmax[0];
      if (dim == 2)
         msg << mxnbin[1] << mxvmin[1] << mxvmax[1];
      if (dim == 3)
         msg << mxnbin[2] << mxvmin[2] << mxvmax[2];

      if (arr.GetLast() != -1) {
         for (int i = 0; i < fLimits; i++)
            ((TSocket*)arr[i])->Send(msg);
         arr.Clear();
      } else
         s->Send(msg);
   }
}

//______________________________________________________________________________
void TProof::Loop(TTree *tree)
{
   // Handle message comming from the remote TTree method currently being
   // executed.

   fLimits = 0;
   fTree   = tree;

   Collect();

   fTree = 0;
}

//______________________________________________________________________________
void TProof::MarkBad(TSlave *sl)
{
   // Add a bad slave server to the bad slave list and remove it from
   // the active list and from the two monitor objects.

   fActiveSlaves->Remove(sl);
   FindUniqueSlaves();
   fBadSlaves->Add(sl);

   fAllMonitor->Remove(sl->GetSocket());
   fActiveMonitor->Remove(sl->GetSocket());

   sl->Close();

   fSendGroupView = kTRUE;
}

//______________________________________________________________________________
void TProof::MarkBad(TSocket *s)
{
   // Add slave with socket s to the bad slave list and remove if from
   // the active list and from the two monitor objects.

   TSlave *sl = FindSlave(s);
   MarkBad(sl);
}

//______________________________________________________________________________
Int_t TProof::Ping(ESlaves list)
{
   // Ping PROOF slaves. Returns the number of slaves that responded.

   TMessage mess(kPROOF_PING | kMESS_ACK);
   return Broadcast(mess, list);
}

//______________________________________________________________________________
void TProof::Print(Option_t *option) const
{
   // Print status of PROOF cluster.

   if (!IsMaster()) {
      Printf("Connected to:             %s (%s)", GetMaster(),
                                          IsValid() ? "valid" : "invalid");
      Printf("Port number:              %d", GetPort());
      Printf("User:                     %s", GetUser());
      Printf("Protocol version:         %d", GetProtocol());
      Printf("Log level:                %d", GetLogLevel());
      if (IsValid())
         ((TProof*)this)->SendPrint();

   } else {
      ((TProof*)this)->AskStatus();
      if (IsParallel())
         Printf("*** Master server (parallel mode, %d slaves):",
                GetNumberOfActiveSlaves());
      else
         Printf("*** Master server (sequential mode):");

      Printf("Master host name:         %s", gSystem->HostName());
      Printf("Port number:              %d", GetPort());
      Printf("User:                     %s", GetUser());
      Printf("Protocol version:         %d", GetProtocol());
      Printf("Image name:               %s", GetImage());
      Printf("Working directory:        %s", gSystem->WorkingDirectory());
      Printf("Config directory:         %s", GetConfDir());
      Printf("Config file:              %s", GetConfFile());
      Printf("Log level:                %d", GetLogLevel());
      Printf("Number of slaves:         %d", GetNumberOfSlaves());
      Printf("Number of active slaves:  %d", GetNumberOfActiveSlaves());
      Printf("Number of unique slaves:  %d", GetNumberOfUniqueSlaves());
      Printf("Number of bad slaves:     %d", GetNumberOfBadSlaves());
      Printf("Total MB's processed:     %.2f", float(GetBytesRead())/(1024*1024));
      Printf("Total real time used (s): %.3f", GetRealTime());
      Printf("Total CPU time used (s):  %.3f", GetCpuTime());
      if (GetNumberOfSlaves()) {
         Printf("List of slaves:");
         fSlaves->ForEach(TSlave,Print)(option);
      }
   }
}

//______________________________________________________________________________
void TProof::RecvLogFile(TSocket *s, Int_t size)
{
   // Receive the log file of the slave with socket s.

   const Int_t kMAXBUF = 16384;  //32768  //16384  //65536;
   char buf[kMAXBUF];

   Int_t  left, r;
   Long_t filesize = 0;

   while (filesize < size) {
      left = Int_t(size - filesize);
      if (left > kMAXBUF)
         left = kMAXBUF;
      r = s->RecvRaw(&buf, left);
      if (r > 0) {
         char *p = buf;

         filesize += r;
         while (r) {
            Int_t w;

            w = write(fileno(stdout), p, r);

            if (w < 0) {
               SysError("RecvLogFile", "error writing to stdout");
               break;
            }
            r -= w;
            p += w;
         }
      } else if (r < 0) {
         Error("RecvLogFile", "error during receiving log file");
         break;
      }
   }
}

//______________________________________________________________________________
Int_t TProof::SendGroupView()
{
   // Send to all active slaves servers the current slave group size
   // and their unique id. Returns number of active slaves.

   if (!IsValid() || !IsMaster()) return 0;
   if (!fSendGroupView) return 0;
   fSendGroupView = kFALSE;

   TIter   next(fActiveSlaves);
   TSlave *sl;

   int  bad = 0, cnt = 0, size = GetNumberOfActiveSlaves();
   char str[32];

   while ((sl = (TSlave *)next())) {
      sprintf(str, "%d %d", cnt, size);
      if (sl->GetSocket()->Send(str, kPROOF_GROUPVIEW) == -1) {
         MarkBad(sl);
         bad++;
      } else
         cnt++;
   }

   // Send the group view again in case there was a change in the
   // group size due to a bad slave

   if (bad) SendGroupView();

   return GetNumberOfActiveSlaves();
}

//______________________________________________________________________________
Int_t TProof::Exec(const char *cmd, ESlaves list)
{
   // Send command to be executed on the PROOF master and/or slaves.
   // Command can be any legal command line command. Commands like
   // ".x file.C" or ".L file.C" will cause the file file.C to be send
   // to the PROOF cluster. Returns -1 in case of error, >=0 in case of
   // succes.

   if (!IsValid()) return 0;

   TString s = cmd;
   s = s.Strip(TString::kBoth);

   if (!s.Length()) return 0;

   // check for macro file and make sure the file is available on all slaves
   if (s.BeginsWith(".L") || s.BeginsWith(".x") || s.BeginsWith(".X")) {
      TString file = s(2, s.Length());
      file = file.Strip(TString::kLeading);
      file = file.Strip(TString::kTrailing, '+');
      char *fn = gSystem->Which(TROOT::GetMacroPath(), file, kReadPermission);
      if (fn) {
         if (GetNumberOfUniqueSlaves() > 0) {
            if (SendFile(fn, kFALSE, kUnique) < 0) {
               Error("Exec", "file %s could not be transfered to PROOF", fn);
               delete [] fn;
               return -1;
            }
         } else {
            TString scmd = s(0,3) + fn;
            Int_t n = SendCommand(scmd);
            delete [] fn;
            return n;
         }
      } else {
         Error("Exec", "macro %s not found", file.Data());
         return -1;
      }
      delete [] fn;
   }

   return SendCommand(cmd);
}

//______________________________________________________________________________
Int_t TProof::SendCommand(const char *cmd, ESlaves list)
{
   // Send command to be executed on the PROOF master and/or slaves.
   // Command can be any legal command line command, however commands
   // like ".x file.C" or ".L file.C" will not cause the file.C to be
   // transfered to the PROOF cluster. In that case use TProof::Exec().
   // Returns the status send by the remote server as part of the
   // kPROOF_LOGDONE message. Typically this is the return code of the
   // command on the remote side.

   if (!IsValid()) return 0;

   Broadcast(cmd, kMESS_CINT, list);
   Collect(list);
   return fStatus;
}

//______________________________________________________________________________
Int_t TProof::SendCurrentState(ESlaves list)
{
   // Transfer the current state of the master to the active slave servers
   // just before starting the TTree loop. The current state includes: the
   // current working directory, TChain defintion, MaxVirtualSize, Selector,
   // etc.

   if (!IsValid()) return 0;

   // Go to the new directory, reset the interpreter environment and
   // tell slave to delete all objects from its new current directory.
   Broadcast(gDirectory->GetPath(), kPROOF_RESET, list);

   return GetNumberOfActiveSlaves();
}

//______________________________________________________________________________
Int_t TProof::SendInitialState()
{
   // Transfer the initial (i.e. current) state of the master to all
   // slave servers. The initial state includes: log level, currently open
   // files.

   if (!IsValid()) return 0;

   SetLogLevel(fLogLevel);
   if (IsMaster())
      ConnectFiles();

   return GetNumberOfActiveSlaves();
}

//______________________________________________________________________________
Int_t TProof::SendFile(const char *file, Bool_t bin, ESlaves list)
{
   // Send a file to master or slave servers. Returns number of slaves
   // the file was sent to, maybe 0 in case master and slaves have the same
   // file system image, -1 in case of error. If bin is true binary
   // file transfer is used, otherwise ASCII mode.

   TList *slaves = 0;
   if (list == kAll)    slaves = fSlaves;
   if (list == kActive) slaves = fActiveSlaves;
   if (list == kUnique) slaves = fUniqueSlaves;

   if (slaves->GetSize() == 0) return 0;

#ifndef R__WIN32
   Int_t fd = open(file, O_RDONLY);
#else
   Int_t fd = open(file, O_RDONLY | O_BINARY);
#endif
   if (fd < 0) {
      SysError("SendFile", "cannot open file %s", file);
      return -1;
   }

   Long_t id, size, flags, modtime;
   if (gSystem->GetPathInfo(file, &id, &size, &flags, &modtime) == 1) {
      Error("SendFile", "cannot get size of file %s", file);
      close(fd);
      return -1;
   }
   if (size == 0) {
      Error("SendFile", "empty file %s", file);
      close(fd);
      return -1;
   }

   const Int_t kMAXBUF = 32768;  //16384  //65536;
   char buf[kMAXBUF];

   sprintf(buf, "%s %d %ld", gSystem->BaseName(file), bin, size);
   if (!Broadcast(buf, kPROOF_SENDFILE, list)) {
      close(fd);
      return -1;
   }

   Int_t len, n;
   do {
      while ((len = read(fd, buf, kMAXBUF)) < 0 && TSystem::GetErrno() == EINTR)
         TSystem::ResetErrno();

      if (len < 0) {
         SysError("SendFile", "error reading from file %s", file);
         Interrupt(kSoftInterrupt, list);
         close(fd);
         return -1;
      }

      if (!(n = BroadcastRaw(buf, len, list))) {
         SysError("SendFile", "error broadcasting, no more active slaves");
         close(fd);
         return -1;
      }

   } while (len > 0);

   close(fd);

   return n;
}

//______________________________________________________________________________
Int_t TProof::SendObject(const TObject *obj, ESlaves list)
{
   // Send object to master or slave servers. Returns number slaves object
   // was sent to, 0 in case of error.

   if (!IsValid() || !obj) return 0;

   TMessage mess(kMESS_OBJECT);

   mess.WriteObject(obj);
   return Broadcast(mess, list);
}

//______________________________________________________________________________
Int_t TProof::SendPrint()
{
   // Send print command to master server.

   if (!IsValid()) return 0;

   Broadcast(kPROOF_PRINT, kActive);
   return Collect(kActive);
}

//______________________________________________________________________________
void TProof::SetLogLevel(Int_t level)
{
   // Set server logging level.

   char str[10];
   fLogLevel = level;
   sprintf(str, "%d", fLogLevel);
   Broadcast(str, kPROOF_LOGLEVEL, kAll);
}

//______________________________________________________________________________
Int_t TProof::SetParallel(Int_t nodes)
{
   // Tell RPOOF how many slaves to use in parallel. Returns the number of
   // parallel slaves.

   if (!IsValid()) return 0;

   if (IsMaster()) {
      GoParallel(nodes);
      return SendCurrentState();
   } else {
      TMessage mess(kPROOF_PARALLEL);
      mess << nodes;
      Broadcast(mess);
      Collect();
      return fParallel;
   }
}

//______________________________________________________________________________
Int_t TProof::GoParallel(Int_t nodes)
{
   // Go in parallel mode with at most "nodes" slaves. Since the fSlaves
   // list is sorted by slave performace the active list will contain first
   // the most performant nodes.

   if (nodes <= 0) nodes = 1;

   fActiveSlaves->Clear();
   fActiveMonitor->RemoveAll();

   TIter next(fSlaves);

   int cnt = 0;
   TSlave *sl;
   while (cnt < nodes && (sl = (TSlave *)next())) {
      if (sl->IsValid()) {
         fActiveSlaves->Add(sl);
         fActiveMonitor->Add(sl->GetSocket());
         cnt++;
      }
   }

   // Will be activated in Collect
   fActiveMonitor->DeActivateAll();

   // Get slave status (will set the slaves fWorkDir correctly)
   AskStatus();

   // Find active slaves with unique image
   FindUniqueSlaves();

   // Send new group-view to slaves
   SendGroupView();

   Int_t n = GetNumberOfActiveSlaves();
   if (IsMaster()) {
      if (n > 1)
         printf("PROOF set to parallel mode (%d slaves)\n", n);
      else
         printf("PROOF set to sequential mode\n");
   }

   return n;
}

//______________________________________________________________________________
Bool_t TProof::IsActive()
{
   // Static function that returns kTRUE in case a PROOF connection exists
   // with more than 1 active slave. When only one active slave we run in
   // sequential mode.

   return gProof ? kTRUE : kFALSE;
}

//______________________________________________________________________________
TProof *TProof::This()
{
   // Static function returning pointer to global object gProof.
   // Mainly for use via CINT, where the gProof symbol might be
   // deleted from the symbol table.

   return gProof;
}
