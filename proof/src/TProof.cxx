// @(#)root/proof:$Name:  $:$Id: TProof.cxx,v 1.1.1.1 2000/05/16 17:00:46 rdm Exp $
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

#include "TProof.h"
#include "TSortedList.h"
#include "TSlave.h"
#include "TSocket.h"
#include "TMonitor.h"
#include "TMessage.h"
#include "TSystem.h"
#include "TError.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"


TProof *gProof = 0;


ClassImp(TProof)

//______________________________________________________________________________
TProof::TProof(const char *cluster, const char *master, const char *version,
               Int_t port, Int_t loglevel, const char *confdir)
{
   // Create a PROOF environment. Starting PROOF involves reading a config
   // file describing the cluster and firing slave servers on all of the
   // available nodes.

   // Can have only one PROOF session open at a time.
   if (gProof)
      gProof->Close();

   if (Init(cluster, port, master, version, loglevel, confdir) == 0) {
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
   SafeDelete(fBadSlaves);
   SafeDelete(fAllMonitor);
   SafeDelete(fActiveMonitor);

   gProof = 0;
}

//______________________________________________________________________________
Int_t TProof::Init(const char *cluster, Int_t port, const char *master,
                   const char *vers, Int_t loglevel, const char *confdir)
{
   // Start the PROOF environment. Starting PROOF involves reading a config
   // file describing the cluster and starting slave servers on all of the
   // available nodes.

   Assert(gSystem);

   fCluster    = cluster;
   fPort       = port;
   fMaster     = master;
   fVersion    = vers;
   fConfDir    = confdir;
   fLogLevel   = loglevel;
   fProtocol   = kPROOF_Protocol;
   fMasterServ = fMaster == "" ? kTRUE : kFALSE;
   fTree       = 0;

   // sort slaves by descending performance index
   fSlaves        = new TSortedList(kSortDescending);
   fActiveSlaves  = new TList;
   fBadSlaves     = new TList;
   fAllMonitor    = new TMonitor;
   fActiveMonitor = new TMonitor;

   GetUserInfo();

   char fconf[256];
   sprintf(fconf, ".%s.conf", cluster);
   if (gSystem->AccessPathName(fconf, kFileExists)) {
      sprintf(fconf, "%s/.%s.conf", gSystem->Getenv("HOME"), cluster);
      if (gSystem->AccessPathName(fconf, kFileExists)) {
         sprintf(fconf, "%s/etc/%s.conf", confdir, cluster);
         if (gSystem->AccessPathName(fconf, kFileExists)) {
            Error("Init", "no PROOF config file found");
            return 0;
         }
      }
   }

   FILE *pconf;
   if ((pconf = fopen(fconf, "r"))) {

      fConfFile = fconf;

      // read the config file
      char line[256];
      int  ord = 0;

      while (fgets(line, sizeof(line), pconf)) {
         char word[4][64];
         if (line[0] == '#') continue;   // skip comment lines
         int nword = sscanf(line, " %s %s %s %s", word[0], word[1], word[2],
                            word[3]);

         if (IsMaster()) {
            // find all slave servers
            if (nword >= 2 && strcmp(word[0], "slave") == 0) {
               int perfidx = -1;
               if (nword == 3) perfidx = atoi(word[2]);
               TSlave *slave = new TSlave(word[1], ord++, perfidx, this);
               fSlaves->Add(slave);
               if (slave->IsValid())
                  fAllMonitor->Add(slave->GetSocket());
               else
                  fBadSlaves->Add(slave);
            }
         } else {
            // we are on client create only one master server remotely
         }
      }
      fclose(pconf);
   }

   // De-activate monitor (will be activated in Collect)
   fAllMonitor->DeActivateAll();

   if (IsMaster())
      SetParallel();   // by default use all valid slaves
   else
      SetParallel(1);  // created only one master server

   if (IsValid())
      gROOT->GetListOfSockets()->Add(this);

   return fActiveSlaves->GetSize();
}

//______________________________________________________________________________
Int_t TProof::ConnectFile(const TFile *file)
{
   // Send message to all slaves to connect "file". This method is
   // called by the TFile ctor (no user method).

   if (!IsValid()) return 0;

   return SendCommand(Form("new TFile(\"%s\", \"%s\");", file->GetName(),
                      file->GetOption()), kAll);
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
void TProof::GetUserInfo()
{
   // Get user info: user name and password. This info is needed to validate
   // the user on the PROOF cluster.

   fUser = gSystem->Getenv("USER");
   fPasswd = "aap";   // dummy for the time being
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
Int_t TProof::GetNumberOfBadSlaves() const
{
   // Return number of bad slaves. This are slaves that we in the config
   // file, but refused to startup or that died during the PROOF session.

   if (!fBadSlaves) return 0;
   return fBadSlaves->GetSize();
}

//______________________________________________________________________________
void TProof::GetStatus()
{
   // Get the status of the slaves.

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
Int_t TProof::Broadcast(const TMessage &mess, ESlaves list)
{
   // Broadcast a message to all slaves in the specified list (either
   // all slaves or only the active slaves). Returns the number of slaves
   // the message was successfully send to.

   if (!IsValid()) return 0;

   TList *slaves = 0;
   if (list == kAll)    slaves = fSlaves;
   if (list == kActive) slaves = fActiveSlaves;

   if (slaves->GetSize() == 0) return 0;

   int   nsend = 0;
   TIter next(slaves);

   TSlave *sl;
   while ((sl = (TSlave *)next())) {
      if (sl->IsValid()) {
         if (sl->GetSocket()->Send(mess) == -1)
            MarkBad(sl);
         else
            nsend++;
      }
   }

   return nsend;
}

//______________________________________________________________________________
Int_t TProof::Broadcast(const char *str, Int_t kind, ESlaves list)
{
   // Broadcast a character string buffer to all slaves in the specified
   // list (either all slaves or only the active slaves). Use kind to
   // set the TMessage what field. Returns the number of slaves the message
   // was send to.

   TMessage mess(kind);
   if (str) mess.WriteString(str);
   return Broadcast(mess, list);
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

   mon->ActivateAll();

   return Collect(mon);
}

//______________________________________________________________________________
Int_t TProof::Collect(TMonitor *mon)
{
   // Collect responses from the slave servers. Returns the number of slaves
   // that responded.

   if (!mon->GetActive()) return 0;

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
            RecvLogFile(s);              // no break
         case kPROOF_LOGDONE:
            mon->DeActivate(s);
            if (!mon->GetActive()) loop = 0;
            break;

         case kPROOF_STATUS:
            sl = FindSlave(s);
            mess->ReadString(str, sizeof(str));
            sscanf(str, "%lf %f %f", &sl->fBytesRead, &sl->fRealTime,
                   &sl->fCpuTime);
            fBytesRead += sl->fBytesRead;
            fRealTime  += sl->fRealTime;
            fCpuTime   += sl->fCpuTime;
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

   return cnt;
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
   fBadSlaves->Add(sl);

   fAllMonitor->Remove(sl->GetSocket());
   fActiveMonitor->Remove(sl->GetSocket());

   sl->Close();
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
void TProof::Print(Option_t *option)
{
   // Print status of PROOF cluster.

   Printf("Name of cluster:          %s  (%s)", GetClusterName(),
                                          IsValid() ? "valid" : "invalid");
   if (IsMaster()) {
      Printf("This is a:                master server");
      GetStatus();
   }
   Printf("Port number:              %d", GetPort());
   Printf("Server version:           %s", GetVersion());
   Printf("Protocol version:         %d", GetProtocol());
   Printf("Config file:              %s", GetConfFile());
   Printf("User:                     %s", GetUser());
   Printf("Log level:                %d", GetLogLevel());
   Printf("Number of slaves:         %d", GetNumberOfSlaves());
   Printf("Number of active slaves:  %d", GetNumberOfActiveSlaves());
   Printf("Number of bad slaves:     %d", GetNumberOfBadSlaves());
   Printf("Total MB's processed:     %.2f", float(GetBytesRead())/(1024*1024));
   Printf("Total real time used (s): %.3f", GetRealTime());
   Printf("Total CPU time used (s):  %.3f", GetCpuTime());
   if (GetNumberOfSlaves()) {
      Printf("List of slaves:");
      fSlaves->ForEach(TSlave,Print)(option);
   }
}

//______________________________________________________________________________
void TProof::RecvLogFile(TSocket *s)
{
   // Receive the log file of the slave with socket s.

   while (1) {
      char str[256];
      int  what;

      s->Recv(str, sizeof(str), what);
      if (what == kPROOF_LOGDONE) break;

      printf("%s", str);
   }
}

//______________________________________________________________________________
Int_t TProof::SendGroupView()
{
   // Send to all active slaves servers the current slave group size
   // and their unique id. Returns number of active slaves.

   if (!IsValid()) return 0;

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
Int_t TProof::SendCommand(const char *cmd, ESlaves list)
{
   // Send command to be executed on the PROOF master and/or slaves.
   // Command can be any legal command line command.

   if (!IsValid()) return 0;

   Broadcast(cmd, kMESS_CINT, list);
   return Collect(list);
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
   ConnectFiles();

   return GetNumberOfActiveSlaves();
}

//______________________________________________________________________________
Int_t TProof::SendObject(const TObject *obj, ESlaves list)
{
   // Send object to master or slave servers. Returns number slaves message
   // was sent too, 0 in case of error.

   if (!IsValid() || !obj) return 0;

   TMessage mess(kMESS_OBJECT);

   mess.WriteObject(obj);
   return Broadcast(mess, list);
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
   // Set the number of nodes that should work in parallel. Since the fSlaves
   // list is sorted by slave performace the active list will contain first
   // the most performant nodes.

   if (nodes <= 0) nodes = 1;

   fActiveSlaves->Clear();

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

   SendInitialState();
   SendGroupView();

   return GetNumberOfActiveSlaves();
}

//______________________________________________________________________________
Bool_t TProof::IsActive()
{
   // Static function that returns kTRUE in case a PROOF connection exists
   // with more than 1 active slave. When only one active slave we run in
   // sequential mode.

   return (gProof && gProof->GetNumberOfActiveSlaves() > 1) ? kTRUE : kFALSE;
}

//______________________________________________________________________________
TProof *TProof::This()
{
   // Static function returning pointer to global object gProof.
   // Mainly for use via CINT, where the gProof symbol might be
   // deleted from the symbol table.

   return gProof;
}
