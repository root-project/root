// This macro attaches to a PROOF session, possibly at the indicated URL.
// In the case non existing PROOF session is found and no URL is given, the macro
// tries to start a local PROOF session.

#include "TEnv.h"
#include "TProof.h"
#include "TString.h"
#include "TSystem.h"

const char *refloc = "proof://localhost";

TProof *getProof(const char *url, Int_t nwrks, const char *wrkdir)
{
   TProof *p = 0;
   TProof *pold = gProof;

   // If an URL has specified get a session there
   if (url && strlen(url) > 0 && strcmp(url, refloc)) {
      p = TProof::Open(url);
      if (p && p->IsValid()) {
         // Done
         return p;
      } else {
         Printf("getProof: could not get/start a valid session at %s - try local", url);
      }
   }
   p = 0;

   // Is there something valid running elsewhere?
   if (pold && pold->IsValid()) {
      if (url && strlen(url) > 0)
         Printf("getProof: attaching to existing valid session at %s ", pold->GetMaster());
      return pold;
   }

   // Local url (use a special port to try to not disturb running daemons)
   Int_t lportx = 11094;
   Int_t lportp = 11093;
   url = (url && strlen(url) > 0) ? url : refloc;
   TUrl u(url);
   u.SetProtocol("proof");
   u.SetPort(lportp);
   TString lurl = u.GetUrl();

   // Prepare to start the daemon
   TString xpdcf("xpd.cf.");
   TString xpdlog, cmd;
   Int_t rc = 0;

   // Is there something listening already ? 
   gEnv->SetValue("XProof.FirstConnectMaxCnt",1);
   Printf("getProof: checking for an existing daemon ...");
   TProofMgr *mgr = TProof::Mgr(lurl);
   if (mgr && mgr->IsValid()) {

      Printf("getProof: daemon found: stop it first ...");

      // Stop it
      cmd = "killall -s=9 xrootd";
      if ((rc = gSystem->Exec(cmd)) != 0)
         Printf("getProof: problems stopping xrootd (%d)", rc);

      // Remove the conf files
      cmd = Form("rm -f %s/%s*", gSystem->TempDirectory(), xpdcf.Data());
      gSystem->Exec(cmd);
      // Remove the log files
      cmd.ReplaceAll(".cf.", ".log.");
      gSystem->Exec(cmd);
   }

   // Try to start something locally; make sure that evrythign is there
   char *xrootd = gSystem->Which(gSystem->Getenv("PATH"), "xrootd", kExecutePermission);
   if (!xrootd) {
      Printf("getProof: xrootd not found: please check the environment!");
      return p;
   }

   // Try to start something locally; create the xrootd config file
   FILE *fcf = gSystem->TempFileName(xpdcf, gSystem->TempDirectory());
   if (!fcf) {
      Printf("getProof: could not create config file for XPD");
      return p;
   }
   fprintf(fcf,"### Load the XrdXrootd protocol on port %d\n", lportx);
   fprintf(fcf,"xrd.port %d\n", lportx);
   fprintf(fcf,"### Load the XrdProofd protocol on port %d\n", lportp);
   fprintf(fcf,"xrd.protocol xproofd:%d libXrdProofd.so\n", lportp);
   if (nwrks > 0) {
      fprintf(fcf,"### Force number of local workers\n");
      fprintf(fcf,"xpd.localwrks %d\n", nwrks);
   }
   if (wrkdir && strlen(wrkdir) > 0) {
      fprintf(fcf,"### Root path for working dir\n");
      fprintf(fcf,"xpd.workdir %s\n", wrkdir);
   }
   fclose(fcf);
   Printf("getProof: xrootd config file at %s", xpdcf.Data());

   // Start xrootd in the background
   xpdlog = xpdcf;
   xpdlog.ReplaceAll(".cf.", ".log.");
   Printf("getProof: xrootd log file at %s", xpdlog.Data());
   cmd = Form("%s -c %s -b -l %s -n xpd-temp", xrootd, xpdcf.Data(), xpdlog.Data());
   if ((rc = gSystem->Exec(cmd)) != 0) {
      Printf("getProof: problems starting xrootd (%d)", rc);
      return p;
   }
   delete[] xrootd;

   // Wait a bit
   Printf("getProof: waiting for xrootd to start ...");
   gSystem->Sleep(2000);

   Printf("getProof: start the PROOF session ...");

   // Start the session now
   p = TProof::Open(Form("localhost:%d", lportp));
   if (!p || !(p->IsValid())) {
      Printf("getProof: starting local session failed");
      if (p) delete p;
      p = 0;
      return p;
   }

   // Return the session
   return p;
}
