/// \file
/// \ingroup tutorial_proof
///
/// Attaches to a PROOF session, possibly at the indicated URL.
/// If no existing PROOF session is found and no URL is given,
/// try to start a local PROOF session.
///
/// Arguments:
///     'url'      URL of the master where to start/attach the PROOF session;
///                this is also the place where to force creation of a new session,
///                if needed (use option 'N', e.g. "proof://mymaster:myport/?N")
///
/// The following arguments apply to xrootd responding at 'refloc' only:
///     'nwrks'    Number of workers to be started. []
///     'dir'      Directory to be used for the files and working areas []. When starting a new
///                instance of the daemon this directory is cleaned with 'rm -fr'. If 'dir'
///                is null, the default is used: `/tmp/<user>/.getproof`
///     'opt'      Defines what to do if an existing xrootd uses the same ports; possible
///                options are: "ask", ask the user; "force", kill the xrootd and start
///                a new one; if any other string is specified the existing xrootd will be
///                used ["ask"].
///                NB: for a change in 'nwrks' to be effective you need to specify opt = "force"
///     'dyn'      This flag can be used to switch on dynamic, per-job worker setup scheduling
///                [kFALSE].
///     'tutords'  This flag can be used to force a dataset dir under the tutorial dir [kFALSE]
///
/// It is possible to trigger the automatic valgrind setup by defining the env GETPROOF_VALGRIND.
/// E.g. to run the master in valgrind do
///
///     $ export GETPROOF_VALGRIND="valgrind=master"
///
/// (or
///     $ export GETPROOF_VALGRIND="valgrind=master valgrind_opts:--leak-check=full"
///
/// to set some options) before running getProof. Note that 'getProof' is also called by 'stressProof',
/// so this holds for 'stressProof' runs too.
///
///
/// \macro_code
///
/// \author Gerardo Ganis

#include "Bytes.h"
#include "Getline.h"
#include "TEnv.h"
#include "TProof.h"
#include "TSocket.h"
#include "TString.h"
#include "TSystem.h"

// Auxilliary functions
int getDebugEnum(const char *what);
Int_t getXrootdPid(Int_t port, const char *subdir = "xpdtut");
Int_t checkXrootdAt(Int_t port, const char *host = "localhost");
Int_t checkXproofdAt(Int_t port, const char *host = "localhost");
Int_t startXrootdAt(Int_t port, const char *exportdirs = 0, Bool_t force = kFALSE);
Int_t killXrootdAt(Int_t port, const char *id = 0);

// Auxilliary structures for Xrootd/Xproofd pinging ...
// The client request
typedef struct {
   int first;
   int second;
   int third;
   int fourth;
   int fifth;
} clnt_HS_t;
// The body received after the first handshake's header
typedef struct {
   int msglen;
   int protover;
   int msgval;
} srv_HS_t;

// By default we start a cluster on the local machine
const char *refloc = "proof://localhost:40000";

TProof *getProof(const char *url = "proof://localhost:40000", Int_t nwrks = -1, const char *dir = 0,
                 const char *opt = "ask", Bool_t dyn = kFALSE, Bool_t tutords = kFALSE)
{


   TProof *p = 0;

   // Valgrind options, if any
   TString vopt, vopts;
#ifndef WIN32
   if (gSystem->Getenv("GETPROOF_VALGRIND")) {
      TString s(gSystem->Getenv("GETPROOF_VALGRIND")), t;
      Int_t from = 0;
      while (s.Tokenize(t, from , " ")) {
         if (t.BeginsWith("valgrind_opts:"))
            vopts = t;
         else
            vopt = t;
      }
      if (vopts.IsNull()) vopts = "valgrind_opts:--leak-check=full --track-origins=yes";
      TProof::AddEnvVar("PROOF_WRAPPERCMD", vopts.Data());
      Printf("getProof: valgrind run: '%s' (opts: '%s')", vopt.Data(), vopts.Data());
   }
#endif

   // If an URL has specified get a session there
   TUrl uu(url), uref(refloc);
   Bool_t ext = (strcmp(uu.GetHost(), uref.GetHost()) ||
                 (uu.GetPort() != uref.GetPort())) ? kTRUE : kFALSE;
   Bool_t lite = kFALSE;
   if (ext && url) {
      if (!strcmp(url, "lite://") || !url[0]) {
         if (!url[0]) uu.SetUrl("lite://");
         if (dir && strlen(dir) > 0) gEnv->SetValue("Proof.Sandbox", dir);
         TString swrk("<default> workers");
         if (nwrks > 0) {
            uu.SetOptions(Form("workers=%d", nwrks));
            swrk.Form("%d workers", nwrks);
         }
         lite = kTRUE;
         gEnv->SetValue("Proof.MaxOldSessions", 1);
         Printf("getProof: trying to open a PROOF-Lite session with %s", swrk.Data());
      } else {
         Printf("getProof: trying to open a session on the external cluster at '%s'", url);
      }
      p = TProof::Open(uu.GetUrl(), vopt);
      if (p && p->IsValid()) {
         // Check consistency
         if (ext && !lite && nwrks > 0) {
            Printf("getProof: WARNING: started/attached a session on external cluster (%s):"
                   " 'nwrks=%d' ignored", url, nwrks);
         }
         if (ext && !lite && dir && strlen(dir) > 0) {
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
      } else {
         Printf("getProof: could not get/start a valid session at %s", url);
         if (p) delete p;
         p = 0;
      }
      // Done
      return p;
   }

#ifdef WIN32
   // No support for local PROOF on Win32 (yet; the optimized local Proof will work there too)
   Printf("getProof: local PROOF not yet supported on Windows, sorry!");
   return p;
#else

   // Temp dir for tutorial daemons
   TString tutdir = dir;
   if (!tutdir.IsNull()) {
      if (gSystem->AccessPathName(tutdir)) {
         // Directory does not exist: try to make it
         gSystem->mkdir(tutdir.Data(), kTRUE);
         if (gSystem->AccessPathName(tutdir, kWritePermission)) {
            if (gSystem->AccessPathName(tutdir)) {
               Printf("getProof: unable to create the working area at the requested path: '%s'"
                      " - cannot continue", tutdir.Data());
            } else {
               Printf("getProof: working area at the requested path '%s'"
                      " created but it is not writable - cannot continue", tutdir.Data());
            }
            return p;
         }
      } else {
         // Check if it is writable ...
         if (gSystem->AccessPathName(dir, kWritePermission)) {
            // ... fail if not
            Printf("getProof: working area at the requested path '%s'"
                      " exists but is not writable - cannot continue", tutdir.Data());
            return p;
         }
      }
   } else {
      // Notify
      Printf("getProof: working area not specified temp ");
      // Force "/tmp/<user>" whenever possible to avoid length problems on MacOsX
      tutdir="/tmp";
      if (gSystem->AccessPathName(tutdir, kWritePermission)) tutdir = gSystem->TempDirectory();
      TString us;
      UserGroup_t *ug = gSystem->GetUserInfo(gSystem->GetUid());
      if (!ug) {
         Printf("getProof: could not get user info");
         return p;
      }
      us.Form("/%s", ug->fUser.Data());
      if (!tutdir.EndsWith(us.Data())) tutdir += us;
      // Add our own subdir
      tutdir += "/.getproof";
      if (gSystem->AccessPathName(tutdir)) {
         gSystem->mkdir(tutdir.Data(), kTRUE);
         if (gSystem->AccessPathName(tutdir, kWritePermission)) {
            Printf("getProof: unable to get a writable working area (tried: %s)"
                  " - cannot continue", tutdir.Data());
            return p;
         }
      }
   }
   Printf("getProof: working area (tutorial dir): %s", tutdir.Data());

   // Dataset dir
   TString datasetdir;
   if (tutords) {
      datasetdir = Form("%s/dataset", tutdir.Data());
      if (gSystem->AccessPathName(datasetdir, kWritePermission)) {
         gSystem->mkdir(datasetdir, kTRUE);
         if (gSystem->AccessPathName(datasetdir, kWritePermission)) {
            Printf("getProof: unable to get a writable dataset directory (tried: %s)"
                   " - cannot continue", datasetdir.Data());
            return p;
         }
         Printf("getProof: dataset dir: %s", datasetdir.Data());
      }
   }

   // Local url (use a special port to try to not disturb running daemons)
   TUrl u(refloc);
   u.SetProtocol("proof");
   if (!strcmp(uu.GetHost(), uref.GetHost()) && (uu.GetPort() != uref.GetPort()))
      u.SetPort(uu.GetPort());
   Int_t lportp = u.GetPort();
   Int_t lportx = lportp + 1;
   TString lurl = u.GetUrl();

   // Prepare to start the daemon
   TString workarea = Form("%s/proof", tutdir.Data());
   TString xpdcf(Form("%s/xpd.cf",tutdir.Data()));
   TString xpdlog(Form("%s/xpd.log",tutdir.Data()));
   TString xpdlogprt(Form("%s/xpdtut/xpd.log",tutdir.Data()));
   TString xpdpid(Form("%s/xpd.pid",tutdir.Data()));
   TString proofsessions(Form("%s/sessions",tutdir.Data()));
   TString cmd;
   Int_t rc = 0;

   // Is there something listening already ?
   Int_t pid = -1;
   Bool_t restart = kTRUE;
   if ((rc = checkXproofdAt(lportp)) == 1) {
      Printf("getProof: something else the a XProofd service is running on"
             " port %d - cannot continue", lportp);
      return p;

   } else if (rc == 0) {

      restart = kFALSE;

      pid = getXrootdPid(lportx);
      Printf("getProof: daemon found listening on dedicated ports {%d,%d} (pid: %d)",
              lportx, lportp, pid);
      if (isatty(0) == 0 || isatty(1) == 0) {
         // Cannot ask: always restart
         restart = kTRUE;
      } else {
         if (!strcmp(opt,"ask")) {
            char *answer = (char *) Getline("getProof: would you like to restart it (N,Y)? [N] ");
            if (answer && (answer[0] == 'Y' || answer[0] == 'y'))
               restart = kTRUE;
         }
      }
      if (!strcmp(opt,"force"))
         // Always restart
         restart = kTRUE;

      // Cleanup, if required
      if (restart) {
         Printf("getProof: cleaning existing instance ...");
         // Cleaning up existing daemon
         cmd = Form("kill -9 %d", pid);
         if ((rc = gSystem->Exec(cmd)) != 0)
            Printf("getProof: problems stopping xrootd process %d (%d)", pid, rc);
         // Wait for all previous connections being cleaned
         Printf("getProof: wait 5 secs so that previous connections are cleaned ...");
         gSystem->Sleep(5000);
      }
   }

   if (restart) {

      // Try to start something locally; make sure that everything is there
      char *xpd = gSystem->Which(gSystem->Getenv("PATH"), "xproofd", kExecutePermission);
      if (!xpd) {
         Printf("getProof: xproofd not found: please check the environment!");
         return p;
      }

      // Cleanup the working area
      cmd = Form("rm -fr %s/xpdtut %s %s %s %s", tutdir.Data(), workarea.Data(),
                                                 xpdcf.Data(), xpdpid.Data(), proofsessions.Data());
      gSystem->Exec(cmd);

      // Try to start something locally; create the xproofd config file
      FILE *fcf = fopen(xpdcf.Data(), "w");
      if (!fcf) {
         Printf("getProof: could not create config file for XPD (%s)", xpdcf.Data());
         return p;
      }
      fprintf(fcf,"### Use admin path at %s/admin to avoid interferences with other users\n", tutdir.Data());
      fprintf(fcf,"xrd.adminpath %s/admin\n", tutdir.Data());
#if defined(R__MACOSX)
      fprintf(fcf,"### Use dedicated socket path under /tmp to avoid length problems\n");
      fprintf(fcf,"xpd.sockpathdir /tmp/xpd-sock\n");
#endif
      fprintf(fcf,"### Load the XrdProofd protocol on port %d\n", lportp);
      fprintf(fcf,"xrd.protocol xproofd libXrdProofd.so\n");
      fprintf(fcf,"xpd.port %d\n", lportp);
      if (nwrks > 0) {
         fprintf(fcf,"### Force number of local workers\n");
         fprintf(fcf,"xpd.localwrks %d\n", nwrks);
      }
      fprintf(fcf,"### Root path for working dir\n");
      fprintf(fcf,"xpd.workdir %s\n", workarea.Data());
      fprintf(fcf,"### Allow different users to connect\n");
      fprintf(fcf,"xpd.multiuser 1\n");
      fprintf(fcf,"### Limit the number of query results kept in the master sandbox\n");
      fprintf(fcf,"xpd.putrc ProofServ.UserQuotas: maxquerykept=2\n");
      fprintf(fcf,"### Limit the number of sessions kept in the sandbox\n");
      fprintf(fcf,"xpd.putrc Proof.MaxOldSessions: 1\n");
      if (tutords) {
         fprintf(fcf,"### Use dataset directory under the tutorial dir\n");
         fprintf(fcf,"xpd.datasetsrc file url:%s opt:-Cq:Av:As:\n", datasetdir.Data());
      }
      if (dyn) {
         fprintf(fcf,"### Use dynamic, per-job scheduling\n");
         fprintf(fcf,"xpd.putrc Proof.DynamicStartup 1\n");
      }
      fprintf(fcf,"### For internal file serving use the xrootd protocol on the same port\n");
      fprintf(fcf,"xpd.xrootd libXrdXrootd-4.so\n");
      fprintf(fcf,"### Set the local data server for the temporary output files accordingly\n");
      fprintf(fcf,"xpd.putenv LOCALDATASERVER=root://%s:%d\n", gSystem->HostName(), lportp);
      fclose(fcf);
      Printf("getProof: xproofd config file at %s", xpdcf.Data());

      // Start xrootd in the background
      Printf("getProof: xproofd log file at %s", xpdlogprt.Data());
      cmd = Form("%s -c %s -b -l %s -n xpdtut -p %d",
               xpd, xpdcf.Data(), xpdlog.Data(), lportp);
      Printf("(NB: any error line from XrdClientSock::RecvRaw and XrdClientMessage::ReadRaw should be ignored)");
      if ((rc = gSystem->Exec(cmd)) != 0) {
         Printf("getProof: problems starting xproofd (%d)", rc);
         return p;
      }
      delete[] xpd;

      // Wait a bit
      Printf("getProof: waiting for xproofd to start ...");
      gSystem->Sleep(2000);

      pid = getXrootdPid(lportp);
      Printf("getProof: xproofd pid: %d", pid);

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
   p = TProof::Open(lurl, vopt.Data());
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

Int_t getXrootdPid(Int_t port, const char *subdir)
{
#ifdef WIN32
   // No support for Xrootd/Proof on Win32 (yet; the optimized local Proof will work there too)
   Printf("getXrootdPid: Xrootd/Proof not supported on Windows, sorry!");
   return -1;
#else
   // Get the pid of the started xrootd process
   Int_t pid = -1;
#if defined(__sun)
   const char *com = "-eo pid,comm";
#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)
   const char *com = "ax -w -w";
#else
   const char *com = "-w -w -eo pid,command";
#endif
   TString cmd;
   if (subdir && strlen(subdir) > 0) {
      cmd.Form("ps %s | grep xrootd | grep \"\\-p %d\" | grep %s", com, port, subdir);
   } else {
      cmd.Form("ps %s | grep xrootd | grep \"\\-p %d\"", com, port);
   }
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
#endif
}

Int_t checkXrootdAt(Int_t port, const char *host)
{
   // Check if a XrdXrootd service is running on 'port' at 'host'
   // Return
   //        0 if OK
   //       -1 if nothing is listening on the port (connection cannot be open)
   //        1 if something is listening but not XROOTD

   // Open the connection
   TSocket s(host, port);
   if (!(s.IsValid())) {
      if (gDebug > 0)
         Printf("checkXrootdAt: could not open connection to %s:%d", host, port);
      return -1;
   }
   // Send the first bytes
   clnt_HS_t initHS;
   memset(&initHS, 0, sizeof(initHS));
   initHS.fourth = host2net((int)4);
   initHS.fifth  = host2net((int)2012);
   int len = sizeof(initHS);
   s.SendRaw(&initHS, len);
   // Read first server response
   int type;
   len = sizeof(type);
   int readCount = s.RecvRaw(&type, len); // 4(2+2) bytes
   if (readCount != len) {
      if (gDebug > 0)
         Printf("checkXrootdAt: 1st: wrong number of bytes read: %d (expected: %d)",
                                readCount, len);
      return 1;
   }
   // to host byte order
   type = net2host(type);
   // Check if the server is the eXtended proofd
   if (type == 0) {
      srv_HS_t xbody;
      len = sizeof(xbody);
      readCount = s.RecvRaw(&xbody, len); // 12(4+4+4) bytes
      if (readCount != len) {
         if (gDebug > 0)
            Printf("checkXrootdAt: 2nd: wrong number of bytes read: %d (expected: %d)",
                                   readCount, len);
         return 1;
      }

   } else if (type == 8) {
      // Standard proofd
      if (gDebug > 0)
         Printf("checkXrootdAt: server is ROOTD");
      return 1;
   } else {
      // We don't know the server type
      if (gDebug > 0)
         Printf("checkXrootdAt: unknown server type: %d", type);
      return 1;
   }
   // Done
   return 0;
}

Int_t checkXproofdAt(Int_t port, const char *host)
{
   // Check if a XrdProofd service is running on 'port' at 'host'
   // Return
   //        0 if OK
   //       -1 if nothing is listening on the port (connection cannot be open)
   //        1 if something is listening but not XPROOFD

   // Open the connection
   TSocket s(host, port);
   if (!(s.IsValid())) {
      if (gDebug > 0)
         Printf("checkXproofdAt: could not open connection to %s:%d", host, port);
      return -1;
   }
   // Send the first bytes
   clnt_HS_t initHS;
   memset(&initHS, 0, sizeof(initHS));
   initHS.third  = (int)host2net((int)1);
   int len = sizeof(initHS);
   s.SendRaw(&initHS, len);
   // These 8 bytes are need by 'proofd' and discarded by XPD
   int dum[2];
   dum[0] = (int)host2net((int)4);
   dum[1] = (int)host2net((int)2012);
   s.SendRaw(&dum[0], sizeof(dum));
   // Read first server response
   int type;
   len = sizeof(type);
   int readCount = s.RecvRaw(&type, len); // 4(2+2) bytes
   if (readCount != len) {
      if (gDebug > 0)
         Printf("checkXproofdAt: 1st: wrong number of bytes read: %d (expected: %d)",
                                 readCount, len);
      return 1;
   }
   // to host byte order
   type = net2host(type);
   // Check if the server is the eXtended proofd
   if (type == 0) {
      srv_HS_t xbody;
      len = sizeof(xbody);
      readCount = s.RecvRaw(&xbody, len); // 12(4+4+4) bytes
      if (readCount != len) {
         if (gDebug > 0)
            Printf("checkXproofdAt: 2nd: wrong number of bytes read: %d (expected: %d)",
                                    readCount, len);
         return 1;
      }
      xbody.protover = net2host(xbody.protover);
      xbody.msgval = net2host(xbody.msglen);
      xbody.msglen = net2host(xbody.msgval);

   } else if (type == 8) {
      // Standard proofd
      if (gDebug > 0)
         Printf("checkXproofdAt: server is PROOFD");
      return 1;
   } else {
      // We don't know the server type
      if (gDebug > 0)
         Printf("checkXproofdAt: unknown server type: %d", type);
      return 1;
   }
   // Done
   return 0;
}

Int_t startXrootdAt(Int_t port, const char *exportdirs, Bool_t force)
{
   // Start a basic xrootd service on 'port' exporting the dirs in 'exportdirs'
   // (blank separated list)

#ifdef WIN32
   // No support for Xrootd on Win32 (yet; the optimized local Proof will work there too)
   Printf("startXrootdAt: Xrootd not supported on Windows, sorry!");
   return -1;
#else
   Bool_t restart = kTRUE;

   // Already there?
   Int_t rc = 0;
   if ((rc = checkXrootdAt(port)) == 1) {

      Printf("startXrootdAt: some other service running on port %d - cannot proceed ", port);
      return -1;

   } else if (rc == 0) {

      restart = kFALSE;

      if (force) {
         // Always restart
         restart = kTRUE;
      } else {
         Printf("startXrootdAt: xrootd service already available on port %d: ", port);
         char *answer = (char *) Getline("startXrootdAt: would you like to restart it (N,Y)? [N] ");
         if (answer && (answer[0] == 'Y' || answer[0] == 'y')) {
            restart = kTRUE;
         }
      }

      // Cleanup, if required
      if (restart) {
         Printf("startXrootdAt: cleaning existing instance ...");

         // Get the Pid
         Int_t pid = getXrootdPid(port, "xrd-basic");

         // Cleanimg up existing daemon
         TString cmd = Form("kill -9 %d", pid);
         if ((rc = gSystem->Exec(cmd)) != 0)
            Printf("startXrootdAt: problems stopping xrootd process %d (%d)", pid, rc);
      }
   }

   if (restart) {
      if (gSystem->AccessPathName("/tmp/xrd-basic")) {
         gSystem->mkdir("/tmp/xrd-basic");
         if (gSystem->AccessPathName("/tmp/xrd-basic")) {
            Printf("startXrootdAt: could not assert dir for log file");
            return -1;
         }
      }
      TString cmd;
      cmd.Form("xrootd -d -p %d -b -l /tmp/xrd-basic/xrootd.log", port);
      if (exportdirs && strlen(exportdirs) > 0) {
         TString dirs(exportdirs), d;
         Int_t from = 0;
         while (dirs.Tokenize(d, from, " ")) {
            if (!d.IsNull()) {
               cmd += " ";
               cmd += d;
            }
         }
      }
      Printf("cmd: %s", cmd.Data());
      if ((rc = gSystem->Exec(cmd)) != 0) {
         Printf("startXrootdAt: problems executing starting command (%d)", rc);
         return -1;
      }
      // Wait a bit
      Printf("startXrootdAt: waiting for xrootd to start ...");
      gSystem->Sleep(2000);
      // Check the result
      if ((rc = checkXrootdAt(port)) != 0) {
         Printf("startXrootdAt: xrootd service not available at %d (rc = %d) - startup failed",
                                port, rc);
         return -1;
      }
      Printf("startXrootdAt: basic xrootd started!");
   }

   // Done
   return 0;
#endif
}

Int_t killXrootdAt(Int_t port, const char *id)
{
   // Kill running xrootd service on 'port'

#ifdef WIN32
   // No support for Xrootd on Win32 (yet; the optimized local Proof will work there too)
   Printf("killXrootdAt: Xrootd not supported on Windows, sorry!");
   return -1;
#else

   Int_t pid = -1, rc= 0;
   if ((pid = getXrootdPid(port, id)) > 0) {

      // Cleanimg up existing daemon
      TString cmd = Form("kill -9 %d", pid);
      if ((rc = gSystem->Exec(cmd)) != 0)
         Printf("killXrootdAt: problems stopping xrootd process %d (%d)", pid, rc);
   }

   // Done
   return rc;
#endif
}

int getDebugEnum(const char *what)
{
   // Check if 'what' matches one of the TProofDebug enum and return the corresponding
   // integer. Relies on a perfect synchronization with the content of TProofDebug.h .

   TString sws(what), sw;
   int rcmask = 0;
   int from = 0;
   while (sws.Tokenize(sw, from , "|")) {
      if (sw.BeginsWith("k")) sw.Remove(0,1);

      if (sw == "None") {
         rcmask |= TProofDebug::kNone;
      } else if (sw == "Packetizer") {
         rcmask |= TProofDebug::kPacketizer;
      } else if (sw == "Loop") {
         rcmask |= TProofDebug::kLoop;
      } else if (sw == "Selector") {
         rcmask |= TProofDebug::kSelector;
      } else if (sw == "Output") {
         rcmask |= TProofDebug::kOutput;
      } else if (sw == "Input") {
         rcmask |= TProofDebug::kInput;
      } else if (sw == "Global") {
         rcmask |= TProofDebug::kGlobal;
      } else if (sw == "Package") {
         rcmask |= TProofDebug::kPackage;
      } else if (sw == "Feedback") {
         rcmask |= TProofDebug::kFeedback;
      } else if (sw == "Condor") {
         rcmask |= TProofDebug::kCondor;
      } else if (sw == "Draw") {
         rcmask |= TProofDebug::kDraw;
      } else if (sw == "Asyn") {
         rcmask |= TProofDebug::kAsyn;
      } else if (sw == "Cache") {
         rcmask |= TProofDebug::kCache;
      } else if (sw == "Collect") {
         rcmask |= TProofDebug::kCollect;
      } else if (sw == "Dataset") {
         rcmask |= TProofDebug::kDataset;
      } else if (sw == "Submerger") {
         rcmask |= TProofDebug::kSubmerger;
      } else if (sw == "Monitoring") {
         rcmask |= TProofDebug::kMonitoring;
      } else if (sw == "All") {
         rcmask |= TProofDebug::kAll;
      } else if (!sw.IsNull()) {
         Printf("WARNING: requested debug enum name '%s' does not exist: assuming 'All'", sw.Data());
         rcmask |= TProofDebug::kAll;
      }
   }
   // Done
   return rcmask;
}
