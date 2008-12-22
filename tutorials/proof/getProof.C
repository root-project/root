//
// This macro attaches to a PROOF session, possibly at the indicated URL.
// If no existing PROOF session is found and no URL is given, the macro
// tries to start a local PROOF session.

#include "Getline.h"
#include "TEnv.h"
#include "TProof.h"
#include "TString.h"
#include "TSystem.h"

Int_t getXrootdPid(Int_t port);

// By default we start a cluster on the local machine
const char *refloc = "proof://localhost:11093";

TProof *getProof(const char *url = "proof://localhost:11093", Int_t nwrks = -1, const char *dir = 0,
                 const char *opt = "ask", Bool_t dyn = kFALSE)
{
   // Arguments:
   //     'url'      URL of the master where to start/attach the PROOF session;
   //                this is also the place where to force creation of a new session,
   //                if needed (use option 'N', e.g. "proof://mymaster:myport/?N")
   //
   // The following arguments apply to xrootd responding at 'refloc' only:
   //     'nwrks'    Number of workers to be started. []
   //     'dir'      Directory to be used for the files and working areas [].
   //     'opt'      Defines what to do if an existing xrootd uses the same ports; possible
   //                options are: "ask", ask the user; "force", kill the xrootd and start
   //                a new one; if any other string is specified the existing xrootd will be
   //                used ["ask"].
   //                NB: for a change in 'nwrks' to be effective you need to specify opt = "force"
   //     'dyn'      This flag can be used to switch on dynamic, per-job worker setup scheduling
   //                [kFALSE].
   //

   TProof *p = 0;

   // If an URL has specified get a session there
   TUrl uu(url), uref(refloc);
   Bool_t ext = (strcmp(uu.GetHost(), uref.GetHost()) ||
                 (uu.GetPort() != uref.GetPort())) ? kTRUE : kFALSE;
   if (ext && url && strlen(url) > 0) {
      if (!strcmp(url, "lite") && nwrks > 0)
         uu.SetOptions(Form("workers=%d", nwrks));
      p = TProof::Open(uu.GetUrl());
      if (p && p->IsValid()) {
         // Check consistency
         if (ext && nwrks > 0) {
            Printf("getProof: WARNING: started/attached a session on external cluster (%s):"
                   " 'nwrks=%d' ignored", url, nwrks);
         }
         if (ext && dir && strlen(dir) > 0) {
            Printf("getProof: WARNING: started/attached a session on external cluster (%s):"
                   " 'dir=\"%s\"' ignored", url, dir);
         }
         if (ext && !strcmp(opt,"force")) {
            Printf("getProof: WARNING: started/attached a session on external cluster (%s):"
                   " 'opt=\"force\"' ignored", url);
         }
         if (ext && dyn) {
            Printf("getProof: WARNING: started/attached a session on external cluster (%s):"
                   " 'dyn=kTRUE' ignored", url);
         }
         // Done
         return p;
      } else {
         if (ext)
            Printf("getProof: could not get/start a valid session at %s - try local", url);
      }
      if (p) delete p;
      p = 0;
   }

   // Temp dir for tutorial daemons
   TString tutdir = dir;
   if (tutdir.IsNull() || gSystem->AccessPathName(dir, kWritePermission)) {
      Printf("getProof: tutorial dir missing or not writable - try temp ");
      tutdir = gSystem->TempDirectory();
      TString us;
      UserGroup_t *ug = gSystem->GetUserInfo(gSystem->GetUid());
      if (!ug) {
         printf("getProof: could not get user info");
         return p;
      }
      us.Form("/%s", ug->fUser.Data());
      if (!tutdir.EndsWith(us.Data())) tutdir += us;
      gSystem->mkdir(tutdir.Data(), kTRUE);
      if (gSystem->AccessPathName(tutdir, kWritePermission)) {
         Printf("getProof: unable to get a writable tutorial directory (tried: %s)"
                " - cannot continue", tutdir.Data());
         return p;
      }
      Printf("getProof: tutorial dir: %s", tutdir.Data());
   }

#ifdef WIN32
   // No support for local PROOF on Win32 (yet; the optimized local Proof will work there too)
   Printf("getProof: local PROOF not yet supported on Windows, sorry!");
   return p;
#else

   // Local url (use a special port to try to not disturb running daemons)
   TUrl u(refloc);
   u.SetProtocol("proof");
   Int_t lportp = u.GetPort();
   Int_t lportx = lportp + 1;
   TString lurl = u.GetUrl();

   // Prepare to start the daemon
   TString workarea = Form("%s/proof", tutdir.Data());
   TString xpdcf(Form("%s/xpd.cf",tutdir.Data()));
   TString xpdlog(Form("%s/xpd.log",tutdir.Data()));
   TString xpdlogprt(Form("%s/xpd-tutorial/xpd.log",tutdir.Data()));
   TString xpdpid(Form("%s/xpd.pid",tutdir.Data()));
   TString proofsessions(Form("%s/sessions",tutdir.Data()));
   TString cmd;
   Int_t rc = 0;

   // Is there something listening already ?
   Int_t pid = -1;
   Bool_t restart = kTRUE;
   gEnv->SetValue("XProof.FirstConnectMaxCnt",1);
   Printf("getProof: checking for an existing daemon ...");
   TProofMgr *mgr = TProof::Mgr(lurl);
   if (mgr && mgr->IsValid()) {

      restart = kFALSE;

      pid = getXrootdPid(lportx);
      Printf("getProof: daemon found listening on dedicated ports {%d,%d} (pid: %d)",
              lportx, lportp, pid);
      if (!strcmp(opt,"ask")) {
         char *answer = Getline("getProof: would you like to restart it (N,Y)? [N] ");
         if (answer && (answer[0] == 'Y' || answer[0] == 'y'))
            restart = kTRUE;
      }
      if (!strcmp(opt,"force"))
         // Always restart
         restart = kTRUE;

      // Cleanup, if required
      if (restart) {

         Printf("getProof: cleaning existing instance ...");

         // Disconnect the manager
         delete mgr;

         // Cleanimg up existing daemon
         cmd = Form("kill -9 %d", pid);
         if ((rc = gSystem->Exec(cmd)) != 0)
            Printf("getProof: problems stopping xrootd process %p (%d)", pid, rc);

         // Remove the tutorial dir
         cmd = Form("rm -fr %s/*", tutdir.Data());
         gSystem->Exec(cmd);
      }
   }

   if (restart) {
      // Try to start something locally; make sure that everything is there
      char *xrootd = gSystem->Which(gSystem->Getenv("PATH"), "xrootd", kExecutePermission);
      if (!xrootd) {
         Printf("getProof: xrootd not found: please check the environment!");
         return p;
      }

      // Try to start something locally; create the xrootd config file
      FILE *fcf = fopen(xpdcf.Data(), "w");
      if (!fcf) {
         Printf("getProof: could not create config file for XPD (%s)", xpdcf.Data());
         return p;
      }
      fprintf(fcf,"### Use admin path at %s/admin to avoid interferences with other users\n", tutdir.Data());
      fprintf(fcf,"xrd.adminpath %s/admin\n", tutdir.Data());
      fprintf(fcf,"### Load the XrdProofd protocol on port %d\n", lportp);
      fprintf(fcf,"xrd.protocol xproofd:%d libXrdProofd.so\n", lportp);
      if (nwrks > 0) {
         fprintf(fcf,"### Force number of local workers\n");
         fprintf(fcf,"xpd.localwrks %d\n", nwrks);
      }
      fprintf(fcf,"### Root path for working dir\n");
      fprintf(fcf,"xpd.workdir %s\n", workarea.Data());
      fprintf(fcf,"### Allow different users to connect\n");
      fprintf(fcf,"xpd.multiuser 1\n");
      fprintf(fcf,"### Limit the number of query results kept in the master sandbox\n");
      fprintf(fcf,"xpd.putrc ProofServ.UserQuotas: maxquerykept=10\n");
      if (dyn) {
         fprintf(fcf,"### Use dynamic, per-job scheduling\n");
         fprintf(fcf,"xpd.putrc Proof.DynamicStartup 1\n");
      }
      fclose(fcf);
      Printf("getProof: xrootd config file at %s", xpdcf.Data());

      // Start xrootd in the background
      Printf("getProof: xrootd log file at %s", xpdlogprt.Data());
      cmd = Form("%s -c %s -b -l %s -n xpd-tutorial -p %d",
               xrootd, xpdcf.Data(), xpdlog.Data(), lportx);
      Printf("(NB: any error line from XrdClientSock::RecvRaw and XrdClientMessage::ReadRaw should be ignored)");
      if ((rc = gSystem->Exec(cmd)) != 0) {
         Printf("getProof: problems starting xrootd (%d)", rc);
         return p;
      }
      delete[] xrootd;

      // Wait a bit
      Printf("getProof: waiting for xrootd to start ...");
      gSystem->Sleep(2000);

      pid = getXrootdPid(lportx);
      Printf("getProof: xrootd pid: %d", pid);

      // Save it in the PID file
      FILE *fpid = fopen(xpdpid.Data(), "w");
      if (!fpid) {
         Printf("getProof: could not create pid file for XPD");
      } else {
         fprintf(fpid,"%d\n", pid);
         fclose(fpid);
      }
   }
   Printf("getProof: start / attach the PROOF session ...");

   // Start / attach the session now
   p = TProof::Open(lurl);
   if (!p || !(p->IsValid())) {
      Printf("getProof: starting local session failed");
      if (p) delete p;
      p = 0;
      return p;
   }

   // Return the session
   return p;
#endif
}

Int_t getXrootdPid(Int_t port)
{
   // Get the pid of the started xrootd process
   Int_t pid = -1;
#if defined(__sun)
   const char *com = "-eo pid,comm";
#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)
   const char *com = "ax -w -w";
#else
   const char *com = "-w -w -eo pid,command";
#endif
   TString cmd = Form("ps %s | grep xrootd | grep \"\\-p %d\" | grep xpd-tutorial", com, port);
   FILE *fp = gSystem->OpenPipe(cmd.Data(), "r");
   if (fp) {
      char line[2048], rest[2048];
      while (fgets(line, sizeof(line), fp)) {
         sscanf(line,"%d %s", &pid, rest);
         break;
      }
      gSystem->ClosePipe(fp);
   }
   // Done
   return pid;
}
