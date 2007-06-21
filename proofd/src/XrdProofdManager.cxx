// @(#)root/proofd:$Name:  $:$Id:$
// Author: G. Ganis June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdManager                                                     //
//                                                                      //
// Author: G. Ganis, CERN, 2007                                         //
//                                                                      //
// Class mapping manager fonctionality.                                 //
// On masters it keeps info about the available worker nodes and allows //
// communication with them.                                             //
// On workers it handles the communication with the master.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdClient/XrdClientMessage.hh"
#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XProofProtocol.h"
#include "XrdProofConn.h"
#include "XrdProofdManager.h"
#include "XrdProofdPlatform.h"
#include "XrdProofdResponse.h"
#include "XrdProofWorker.h"

#define XPD_DEF_PORT 1093

// Tracing utilities
#include "XrdProofdTrace.h"
static const char *gTraceID = " ";
extern XrdOucTrace *XrdProofdTrace;
#define TRACEID gTraceID

//__________________________________________________________________________
XrdProofdManager::XrdProofdManager()
{
   // Constructor

   fSrvType  = kXPD_AnyServer;
   fEffectiveUser = "";
   fHost = "";
   fPort = XPD_DEF_PORT;
   fImage = "";        // image name for these servers
   fWorkDir = "";
   fDataSetDir = "";
   fPROOFcfg.fName = "";
   fPROOFcfg.fMtime = 0;
   fWorkers.clear();
   fNumLocalWrks = -1;
   fEDest = 0;
}

//__________________________________________________________________________
XrdProofdManager::~XrdProofdManager()
{
   // Destructor

   // Cleanup the worker list and create default info
   std::list<XrdProofWorker *>::iterator w = fWorkers.begin();
   while (w != fWorkers.end()) {
      delete *w;
      w = fWorkers.erase(w);
   }
}

//__________________________________________________________________________
int XrdProofdManager::Config(const char *fn, XrdOucError *e)
{
   // Config or re-config this instance using the information in file 'fn'
   // return 0 on success, -1 on error

   XrdOucString mp;

   // Error handler
   fEDest = (e) ? e : 0;

   if (fCfgFile.fName.length() <= 0 && !fn || strlen(fn) <= 0) {
      // Done
      if (fEDest)
         fEDest->Say(0, "ProofdManager: Config: no config file!");
      return -1;
   }

   // We need an error handler
   if (!fEDest) {
      TRACE(XERR, "Config: error handler undefined!");
      return -1;
   }

   // Did the file changed ?
   if (fCfgFile.fName != fn) {
      fCfgFile.fName = fn;
      XrdProofdAux::Expand(fCfgFile.fName);
      fCfgFile.fMtime = 0;
   }

   // Get the modification time
   struct stat st;
   if (stat(fCfgFile.fName.c_str(), &st) != 0)
      return -1;
   TRACE(DBG, "Config: enter: time of last modification: " << st.st_mtime);

   // File should be loaded only once
   if (st.st_mtime <= fCfgFile.fMtime)
      return 0;

   // Save the modification time
   fCfgFile.fMtime = st.st_mtime;

   // This part must be modified in atomic way
   XrdOucMutexHelper mhp(fMutex);

   // Effective user
   XrdProofUI ui;
   if (XrdProofdAux::GetUserInfo(geteuid(), ui) == 0) {
      fEffectiveUser += ui.fUser;
   } else {
      mp = "ProofdManager: Config: could not resolve effective user (getpwuid, errno: ";
      mp += errno;
      mp += ")";
      fEDest->Say(0, mp.c_str());
      return -1;
   }

   // Local FQDN
   char *host = XrdNetDNS::getHostName();
   fHost = host ? host : "";
   SafeFree(host);

   XrdOucStream cfg(fEDest, getenv("XRDINSTANCE"));

   // Open and attach the config file
   int cfgFD;
   const char *cfn = fCfgFile.fName.c_str();
   if ((cfgFD = open(cfn, O_RDONLY, 0)) < 0) {
      fEDest->Say(0, "ProofdManager: Config: unable to open : ", cfn);
      return -1;
   }
   cfg.Attach(cfgFD);

   // Process items
   char *var = 0, *val = 0;
   while ((var = cfg.GetMyFirstWord())) {

      if (!(strncmp("xpd.", var, 4)) && var[4]) {
         var += 4;
         // Get the value
         val = cfg.GetToken();
         if (val && val[0]) {

            TRACE(DBG, "Config: var: "<<var<<" - val: "<<  val);

            if (!strcmp("resource",var)) {
               // Specifies the resource broker
               if (!strcmp("static",val)) {
                  // We just take the path of the config file here; the
                  // rest is used by the static scheduler
                  fResourceType = kRTStatic;
                  while ((val = cfg.GetToken()) && val[0]) {
                     XrdOucString s(val);
                     if (s.beginswith("ucfg:")) {
                     } else if (s.beginswith("wmx:")) {
                     } else if (s.beginswith("selopt:")) {
                     } else {
                        // Config file
                        fPROOFcfg.fName = val;
                        XrdProofdAux::Expand(fPROOFcfg.fName);
                        // Make sure it exists and can be read
                        if (access(fPROOFcfg.fName.c_str(), R_OK)) {
                           fEDest->Say(0, "Config: configuration file cannot be read: ",
                                          fPROOFcfg.fName.c_str());
                           fPROOFcfg.fName = "";
                           fPROOFcfg.fMtime = 0;
                        }
                     }
                  }
               }

            } else {

               // Save 'val' first
               XrdOucString tval = val;
               // Number of matching chars: the parameter will be updated only
               // if condition is absent or equivalent/better matching
               if (XrdProofdAux::CheckIf(&cfg, fHost.c_str()) != 0) {
                  // Now check
                  if (!strcmp("localwrks",var)) {
                     // Number of workers for local sessions
                     fNumLocalWrks = strtol(tval.c_str(), 0, 10);
                  } else if (!strcmp("image",var)) {
                     // Image name of this server
                     fImage = tval;
                  } else if (!strcmp("workdir",var)) {
                     // Workdir for this server
                     fWorkDir = tval;
                  } else if (!strcmp("datasetdir",var)) {
                     // Dataset dir for this master server (ignored on workers)
                     fDataSetDir = tval;
                  } else if (!strcmp("role",var)) {
                     // Role this server
                     if (tval == "master")
                        fSrvType = kXPD_TopMaster;
                     else if (tval == "submaster")
                        fSrvType = kXPD_MasterServer;
                     else if (tval == "worker")
                        fSrvType = kXPD_WorkerServer;
                  }
               }
            }
         }
      } else if (!(strncmp("xrd.protocol", var, strlen("xrd.protocol")))) {
         // Get the value
         XrdOucString proto = cfg.GetToken();
         if (proto.length() > 0 && proto.beginswith("xproofd:")) {
            proto.replace("xproofd:","");
            fPort = strtol(proto.c_str(), 0, 10);
            fPort = (fPort < 0) ? XPD_DEF_PORT : fPort;
         }
      }
   }

   // Image
   if (fImage.length() <= 0)
      // Use the local host name
      fImage = fHost;

   // Work directory, if specified
   if (fWorkDir.length() > 0) {
      // Make sure it exists
      if (XrdProofdAux::AssertDir(fWorkDir.c_str(), ui) != 0) {
         fEDest->Say(0, "ProofdManager: Config: unable to assert working dir: ",
                        fWorkDir.c_str());
         return -1;
      }
      fEDest->Say(0, "ProofdManager: Config: working directories under: ",
                     fWorkDir.c_str());
   }

   // Dataset directory, if specified
   if (fDataSetDir.length() > 0) {
      // Make sure it exists
      if (XrdProofdAux::AssertDir(fDataSetDir.c_str(), ui) != 0) {
         fEDest->Say(0, "ProofdManager: Config: unable to assert dataset dir: ",
                        fDataSetDir.c_str());
         return -1;
      }
      fEDest->Say(0, "ProofdManager: Config: dataset directories under: ",
                     fDataSetDir.c_str());
   }

   if (fSrvType != kXPD_WorkerServer || fSrvType == kXPD_AnyServer) {

      if (fResourceType == kRTStatic) {
         // Initialize the list of workers if a static config has been required
         // Default file path, if none specified
         if (fPROOFcfg.fName.length() <= 0) {
            fNumLocalWrks = XrdProofdAux::GetNumCPUs();
            CreateDefaultPROOFcfg();
         }
         fEDest->Say(0, "ProofdManager: Config: PROOF config file: ",
                         ((fPROOFcfg.fName.length() > 0) ? fPROOFcfg.fName.c_str()
                                                         : "none"));
         // Load file content in memory
         if (ReadPROOFcfg() != 0) {
            fEDest->Say(0, "ProofdManager: Config: unable to find valid information"
                           "in PROOF config file ", fPROOFcfg.fName.c_str());
            fPROOFcfg.fMtime = 0;
            return 0;
         }
      }
   }

   // Done
   return 0;
}

//__________________________________________________________________________
int XrdProofdManager::Broadcast(int type, const char *msg, XrdProofdResponse *r)
{
   // Broadcast request to known potential sub-nodes.
   // Return 0 on success, -1 on error
   int rc = 0;

   TRACE(ACT, "Broadcast: enter: type: "<<type);

   // We try only once
   int maxtry_save = -1;
   int timewait_save = -1;
   XrdProofConn::GetRetryParam(maxtry_save, timewait_save);
   XrdProofConn::SetRetryParam(1, 1);

   // Loop over worker nodes
   std::list<XrdProofWorker *>::iterator iw = fWorkers.begin();
   XrdProofWorker *w = 0;
   XrdClientMessage *xrsp = 0;
   while (iw != fWorkers.end()) {
      if ((w = *iw) && w->fType != 'M') {
         // Do not send it to ourselves
         bool us = (((w->fHost.find("localhost") != STR_NPOS ||
                     fHost.find(w->fHost.c_str()) != STR_NPOS)) &&
                    (w->fPort == -1 || w->fPort == fPort)) ? 1 : 0;
         if (!us) {
            // Create 'url'
            XrdOucString u = fEffectiveUser;
            u += '@';
            u += w->fHost;
            if (w->fPort != -1) {
               u += ':';
               u += w->fPort;
            }
            // Type of server
            int srvtype = (w->fType != 'W') ? (kXR_int32) kXPD_MasterServer
                                            : (kXR_int32) kXPD_WorkerServer;
            TRACE(HDBG,"Broadcast: sending request to "<<u);
            // Send request
            if (!(xrsp = Send(u.c_str(), type, msg, srvtype, r))) {
               TRACE(XERR,"Broadcast: problems sending request to "<<u);
            }
            // Cleanup answer
            SafeDelete(xrsp);
         }
      }
      // Next worker
      iw++;
   }

   // Restore original retry parameters
   XrdProofConn::SetRetryParam(maxtry_save, timewait_save);

   // Done
   return rc;
}

//__________________________________________________________________________
XrdClientMessage *XrdProofdManager::Send(const char *url, int type,
                                         const char *msg, int srvtype,
                                         XrdProofdResponse *r)
{
   // Broadcast request to known potential sub-nodes.
   // Return 0 on success, -1 on error
   XrdClientMessage *xrsp = 0;

   TRACE(ACT, "Send: enter: type: "<<type);

   if (!url || strlen(url) <= 0)
      return xrsp;

   // Open the connection
   XrdOucString buf = "session-cleanup-from-";
   buf += fHost;
   buf += "|ord:000";
   char m = 'A'; // log as admin
   XrdProofConn *conn = new XrdProofConn(url, m, -1, -1, 0, buf.c_str());

   bool ok = 1;
   if (conn && conn->IsValid()) {
      // Prepare request
      XPClientRequest reqhdr;
      const void *buf = 0;
      void **vout = 0;
      memset(&reqhdr, 0, sizeof(reqhdr));
      conn->SetSID(reqhdr.header.streamid);
      reqhdr.header.requestid = kXP_admin;
      reqhdr.proof.int1 = type;
      switch (type) {
         case kROOTVersion:
            reqhdr.header.dlen = (msg) ? strlen(msg) : 0;
            buf = (msg) ? (const void *)msg : buf;
            break;
         case kCleanupSessions:
            reqhdr.proof.int2 = (kXR_int32) srvtype;
            reqhdr.proof.sid = -1;
            reqhdr.header.dlen = (msg) ? strlen(msg) : 0;
            buf = (msg) ? (const void *)msg : buf;
            break;
         default:
            ok = 0;
            TRACE(XERR,"Send: invalid request type "<<type);
            break;
      }

      // Send over
      if (ok)
         xrsp = conn->SendReq(&reqhdr, buf, vout, "XrdProofdManager::Send");

      // Print error msg, if any
      if (r && !xrsp && conn->GetLastErr()) {
         XrdOucString cmsg = url;
         cmsg += ": ";
         cmsg += conn->GetLastErr();
         r->Send(kXR_attn, kXPD_srvmsg, (char *) cmsg.c_str(), cmsg.length());
      }

      // Close physically the connection
      conn->Close("S");

      // Delete it
      SafeDelete(conn);

   } else {
      TRACE(XERR,"Send: could not open connection to "<<url);
      if (r) {
         XrdOucString cmsg = "failure attempting connection to ";
         cmsg += url;
         r->Send(kXR_attn, kXPD_srvmsg, (char *) cmsg.c_str(), cmsg.length());
      }
   }

   // Done
   return xrsp;
}

//__________________________________________________________________________
void XrdProofdManager::CreateDefaultPROOFcfg()
{
   // Fill-in fWorkers for a localhost based on the number of
   // workers fNumLocalWrks.

   TRACE(ACT, "CreateDefaultPROOFcfg: enter");

   // Create a default master line
   XrdOucString mm("master ",128);
   mm += fImage; mm += " image="; mm += fImage;
   fWorkers.push_back(new XrdProofWorker(mm.c_str()));
   TRACE(DBG, "CreateDefaultPROOFcfg: added line: " << mm);

   // Create 'localhost' lines for each worker
   int nwrk = fNumLocalWrks;
   while (nwrk--) {
      mm = "worker localhost port=";
      mm += fPort;
      fWorkers.push_back(new XrdProofWorker(mm.c_str()));
      TRACE(DBG, "CreateDefaultPROOFcfg: added line: " << mm);
   }

   TRACE(ACT, "CreateDefaultPROOFcfg: done ("<<fWorkers.size()-1<<")");

   // We are done
   return;
}

//__________________________________________________________________________
std::list<XrdProofWorker *> *XrdProofdManager::GetActiveWorkers()
{
   // Return the list of workers after having made sure that the info is
   // up-to-date

   XrdOucMutexHelper mhp(fMutex);

   if (fResourceType == kRTStatic) {
      // Check if there were any changes in the config file
      if (ReadPROOFcfg() != 0) {
         TRACE(XERR, "GetActiveWorkers: unable to read the configuration file");
         return (std::list<XrdProofWorker *> *)0;
      }
   }

   return &fWorkers;
}

//__________________________________________________________________________
int XrdProofdManager::ReadPROOFcfg()
{
   // Read PROOF config file and load the information in memory in 
   // fWorkers.
   // NB: 'master' information here is ignored, because it is passed
   //     via the 'xpd.workdir' and 'xpd.image' config directives

   TRACE(ACT, "ReadPROOFcfg: enter: saved time of last modification: " <<
              fPROOFcfg.fMtime);

   // Check inputs
   if (fPROOFcfg.fName.length() <= 0)
      return -1;

   // Get the modification time
   struct stat st;
   if (stat(fPROOFcfg.fName.c_str(), &st) != 0)
      return -1;
   TRACE(DBG, "ReadPROOFcfg: enter: time of last modification: " << st.st_mtime);

   // File should be loaded only once
   if (st.st_mtime <= fPROOFcfg.fMtime)
      return 0;

   // Cleanup the worker list
   std::list<XrdProofWorker *>::iterator w = fWorkers.begin();
   while (w != fWorkers.end()) {
      delete *w;
      w = fWorkers.erase(w);
   }

   // Save the modification time
   fPROOFcfg.fMtime = st.st_mtime;

   // Open the defined path.
   FILE *fin = 0;
   if (!(fin = fopen(fPROOFcfg.fName.c_str(), "r")))
      return -1;

   // Create a default master line
   XrdOucString mm("master ",128);
   mm += fImage; mm += " image="; mm += fImage;
   fWorkers.push_back(new XrdProofWorker(mm.c_str()));

   // Read now the directives
   int nw = 1;
   char lin[2048];
   while (fgets(lin,sizeof(lin),fin)) {
      // Skip empty lines
      int p = 0;
      while (lin[p++] == ' ') { ; } p--;
      if (lin[p] == '\0' || lin[p] == '\n')
         continue;

      // Skip comments
      if (lin[0] == '#')
         continue;

      // Remove trailing '\n';
      if (lin[strlen(lin)-1] == '\n')
         lin[strlen(lin)-1] = '\0';

      TRACE(DBG, "ReadPROOFcfg: found line: " << lin);

      const char *pfx[2] = { "master", "node" };
      if (!strncmp(lin, pfx[0], strlen(pfx[0])) ||
          !strncmp(lin, pfx[1], strlen(pfx[1]))) {
         // Init a master instance
         XrdProofWorker *pw = new XrdProofWorker(lin);
         if (pw->fHost == "localhost" ||
             pw->Matches(fHost.c_str())) {
            // Replace the default line (the first with what found in the file)
            XrdProofWorker *fw = fWorkers.front();
            fw->Reset(lin);
            // If the image was not specified use the default
            if (fw->fImage == "" ||
                fw->fHost.beginswith(fw->fImage))
               fw->fImage = fImage;
         }
         SafeDelete(pw);
     } else {
         // Build the worker object
         fWorkers.push_back(new XrdProofWorker(lin));
         nw++;
      }
   }

   // Close files
   fclose(fin);

   // We are done
   return ((nw == 0) ? -1 : 0);
}
