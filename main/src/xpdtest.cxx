// @(#)root/main:$Id$
// Author: G. Ganis 2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// xpdtest                                                              //
//                                                                      //
// Program used to test existence and status of a proof daemon          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <errno.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <signal.h>

#include "Riostream.h"
#include "TSystem.h"
#include "TProof.h"
#include "TString.h"

//______________________________________________________________________________
// Prototypes
int xpd_ping(const char *host, int port);
int getsocket(struct hostent *h, int);
void printhelp();
int recvn(int sock, void *buffer, int length);
int sendn(int sock, const void *buffer, int length);
int proof_open(const char *master, long to = 10);
int parse_args(int argc, char **argv,
               TString &url, TString &sboxdir, time_t &span, int &test,
               TString &logfile, bool &keep, bool &verbose, long &to,
               TString &pidfile);
void printhelp();
int set_timer(bool on = 1, long to = 10);

//______________________________________________________________________________
// The client request structure
typedef struct {
   int first;
   int second;
   int third;
   int fourth;
   int fifth;
} clnt_HS_t;

//______________________________________________________________________________
// The body received after the first handshake's header
typedef struct {
   int msglen;
   int protover;
   int msgval;
} srv_HS_t;

////////////////////////////////////////////////////////////////////////////////
///  PROOF daemon test program.
///  Syntax
///          xpdtest  `<url>` `<sandbox_dir>` `<time_span>`
///
///          `<url>`            URL to test; default 'localhost:1093'
///          `<sandbox_dir>`    directory with users sandboxes; used to find out
///                           users to check connection; default '/tmp/proofbox'
///          `<time_span>`      check only users whose latest activity was within
///                           'time_span' minutes; use -1 to check all users;
///                           default -1.
///
///  Exits 0 on success, 1 on error

int main(int argc, char **argv)
{
   TString url, sboxdir, logfile, pidfile;
   time_t span = -1;
   int test = 0;
   bool keep = 0, verbose = 0;
   long timeout = -1;
   int rc = parse_args(argc, argv, url, sboxdir, span, test, logfile, keep,
                       verbose, timeout, pidfile);
   if (rc < 0) {
      fprintf(stderr, "xpdtest: parse_args failure\n");
      gSystem->Exit(1);
   } else if (rc > 0) {
      gSystem->Exit(0);
   }
   gDebug = (verbose) ? 1 : 0;

   rc = 0;

   // Set up log file if required
   RedirectHandle_t redirH;
   if (!logfile.IsNull()) {
      gSystem->RedirectOutput(logfile, "w", &redirH);
   }

   // Extract process ID, if a file has been passed
   if (!pidfile.IsNull()) {
      std::fstream infile(pidfile.Data(), std::ios::in);
      if (infile.is_open()) {
         TString line;
         line.ReadToDelim(infile);
         line.Remove(TString::kTrailing, '\n');
         if (line.IsDigit()) {
            pid_t pid = (pid_t) line.Atoi();
            if (kill(pid, 0) != 0) {
               fprintf(stderr, "xpdtest: process '%d' does not exist\n", (int) pid);
               rc = 1;
            }
         } else {
            fprintf(stderr, "xpdtest: pId file does not contain pid in expected form (first line: %s)\n", line.Data());
            rc = 1;
         }
      } else {
         fprintf(stderr, "xpdtest: pId file '%s' could not be opened\n", pidfile.Data());
         rc = 1;
      }
   }

   // Check sandbox dir
   FileStat_t fst;
   if (rc == 0 && test > 1) {
      if (gSystem->GetPathInfo(sboxdir, fst) != 0) {
         fprintf(stderr, "xpdtest: stat failure for '%s'\n", sboxdir.Data());
         rc = 1;
      }
      if (rc == 0 && !R_ISDIR(fst.fMode)) {
         fprintf(stderr, "xpdtest: '%s' is not a directory\n", sboxdir.Data());
         rc = 1;
      }
   }

   // Setup URL
   TUrl u;
   TString defusr;
   if (rc == 0) {
      u.SetUrl(url.Data());
      defusr = u.GetUser();
      if (defusr.IsNull()) {
         UserGroup_t *pw = gSystem->GetUserInfo();
         if (pw) {
            defusr = pw->fUser;
            delete pw;
         }
      }
      if (!url.BeginsWith(u.GetProtocol())) {
         if (u.GetPort() == 80 && !strcmp(u.GetProtocol(), "http")) u.SetPort(1093);
         u.SetProtocol("proof");
      }
   }

   // Do ping
   if (rc == 0) {
      set_timer(1, timeout);
      if ((rc = xpd_ping(u.GetHost(), u.GetPort())) != 0)
         fprintf(stderr, "xpdtest: failure pinging '%s'\n", url.Data());
      set_timer(0);
   }

   if (test > 0) {
      // Do TProof::Open in "masteronly" mode
      if (rc == 0) {
         if (!logfile.IsNull()) gSystem->RedirectOutput(0, 0, &redirH);
         rc = proof_open(u.GetUrl(), timeout);
         if (!logfile.IsNull()) gSystem->RedirectOutput(logfile, "a", &redirH);
         if (rc != 0)
            fprintf(stderr, "xpdtest: TProof::Open failure for default user '%s'\n", u.GetUrl());
      }

      if (test > 1) {
         if (rc == 0) {

            // Scan sandbox dir
            time_t now = time(0);
            void *dirp = gSystem->OpenDirectory(sboxdir.Data());
            const char *ent = 0;
            TString dent;
            while ((ent = gSystem->GetDirEntry(dirp))) {
               if (!strcmp(ent, "..") || !strcmp(ent, ".")) continue;
               if (defusr == ent) continue;
               FileStat_t st;
               if (span > 0) {
                  dent.Form("%s/%s", sboxdir.Data(), ent);
                  if (gSystem->GetPathInfo(dent, st) != 0) {
                     fprintf(stderr, "xpdtest: stat failure for '%s'\n", dent.Data());
                     rc = 1;
                     break;
                  }
               }
               if (span < 0 || st.fMtime > (now - span)) {
                  u.SetUser(ent);
                  fprintf(stderr, "proof_open: url: '%s'\n", u.GetUrl());
                  if (!logfile.IsNull()) gSystem->RedirectOutput(0, 0, &redirH);
                  rc = proof_open(u.GetUrl(), timeout);
                  if (!logfile.IsNull()) gSystem->RedirectOutput(logfile, "a", &redirH);
                  if (rc != 0) {
                     fprintf(stderr, "xpdtest: failure scanning sandbox dir '%s'\n", sboxdir.Data());
                     break;
                  }
               }
            }
            gSystem->FreeDirectory(dirp);
         }
      }
   }
   if (!logfile.IsNull()) {
      gSystem->RedirectOutput(0, 0, &redirH);
      if (rc == 0 && !keep) gSystem->Unlink(logfile);
   }
   gSystem->Exit(rc);
}

// Auxilliary functions

////////////////////////////////////////////////////////////////////////////////
/// Extract control info from arguments

int parse_args(int argc, char **argv,
               TString &url, TString &sboxdir, time_t &span, int &test,
               TString &logfile, bool &keep, bool &verbose, long &to, TString &pidfile)
{
   url = "localhost:1093";
   sboxdir = "/tmp/proofbox";
   span = -1;
   test = 0;
   logfile= "";
   keep = 0;

   // Check environment settings first
   if (getenv("XPDTEST_URL")) {
      url = getenv("XPDTEST_URL");
   }
   if (getenv("XPDTEST_SBOXDIR")) {
      sboxdir = getenv("XPDTEST_SBOXDIR");
   }
   if (getenv("XPDTEST_TIMESPAN")) {
      errno = 0;
      long xspan = strtol(getenv("XPDTEST_TIMESPAN"), 0, 10);
      if (errno == 0) span = (time_t) xspan;
   }
   if (getenv("XPDTEST_TEST")) {
      errno = 0;
      long xtest = strtol(getenv("XPDTEST_TEST"), 0, 10);
      if (errno == 0 && xtest >= 0 && xtest <= 2) sscanf(getenv("XPDTEST_TEST"), "%d", &test);
   }
   if (getenv("XPDTEST_LOGFILE")) {
      logfile = getenv("XPDTEST_LOGFILE");
   }
   if (getenv("XPDTEST_KEEP")) {
      keep = 1;
   }
   if (getenv("XPDTEST_VERBOSE")) {
      verbose = 1;
   }
   if (getenv("XPDTEST_TIMEOUT")) {
      errno = 0;
      long xto = strtol(getenv("XPDTEST_TIMEOUT"), 0, 10);
      if (errno == 0 && xto > 0) sscanf(getenv("XPDTEST_TIMEOUT"), "%ld", &to);
   }
   if (getenv("XPDTEST_PIDFILE")) {
      pidfile = getenv("XPDTEST_PIDFILE");
   }

   // -u url -d sboxdir -s span -t test
   for (int i = 1; i < argc; i++) {
      if (argv[i]) {
         if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printhelp();
            return 1;
         } else if (!strcmp(argv[i], "-u")) {
            if (argv[++i]) {
               url = argv[i];
            } else {
               fprintf(stderr, "parse_args: '-u' requires the specification of the URL!\n");
               printhelp();
               return -1;
            }
         } else if (!strcmp(argv[i], "-d")) {
            if (argv[++i]) {
               sboxdir = argv[i];
            } else {
               fprintf(stderr, "parse_args: '-d' requires the specification of the sandbox directory!\n");
               printhelp();
               return -1;
            }
         } else if (!strcmp(argv[i], "-l")) {
            if (argv[++i]) {
               logfile = argv[i];
            } else {
               fprintf(stderr, "parse_args: '-l' requires the specification of the logfile!\n");
               printhelp();
               return -1;
            }
         } else if (!strcmp(argv[i], "-k")) {
            keep = 1;
         } else if (!strcmp(argv[i], "-s")) {
            if (argv[++i]) {
               errno = 0;
               long xspan = strtol(argv[i], 0, 10);
               if (errno == 0 && xspan > 0) {
                  span = (time_t) xspan;
               } else {
                  fprintf(stderr, "parse_args: time span must be a positive integer! (argv: %s, xspan: %ld)\n",
                                  argv[i], xspan);
                  printhelp();
                  return -1;
               }
            } else {
               fprintf(stderr, "parse_args: '-s' requires the specification of the time span!\n");
               printhelp();
               return -1;
            }
         } else if (!strcmp(argv[i], "-t")) {
            if (argv[++i]) {
               errno = 0;
               long xtest = strtol(argv[i], 0, 10);
               if (errno == 0 && xtest >= 0 && xtest <= 2) {
                  sscanf(argv[i], "%d", &test);
               } else {
                  fprintf(stderr, "parse_args: test must be an integer in [0,2]! (argv: %s, xtest: %ld)\n",
                                  argv[i], xtest);
                  printhelp();
                  return -1;
               }
            } else {
               fprintf(stderr, "parse_args: '-t' requires the specification of the test!\n");
               printhelp();
               return -1;
            }
         } else if (!strcmp(argv[i], "-T")) {
            if (argv[++i]) {
               errno = 0;
               long xto = strtol(argv[i], 0, 10);
               if (errno == 0 && xto > 0) {
                  to = (time_t) xto;
               } else {
                  fprintf(stderr, "parse_args: timeout must be a positive integer! (argv: %s, xto: %ld)\n",
                                  argv[i], xto);
                  printhelp();
                  return -1;
               }
            } else {
               fprintf(stderr, "parse_args: '-T' requires the specification of the timeout (in secs)!\n");
               printhelp();
               return -1;
            }
         } else if (!strcmp(argv[i], "-p")) {
            if (argv[++i]) {
               pidfile = argv[i];
            } else {
               fprintf(stderr, "parse_args: '-p' requires the specification of the file with the process ID!\n");
               printhelp();
               return -1;
            }
         } else if (!strcmp(argv[i], "-v")) {
            verbose = 1;
         }
      }
   }
   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Help function

void printhelp()
{
   fprintf(stderr, "\n");
   fprintf(stderr, "   xpdtest: test xproofd service on (remote) host\n");
   fprintf(stderr, "\n");
   fprintf(stderr, "   Syntax:\n");
   fprintf(stderr, "            xpdtest [-u url] [-t test] [-d sbdir] [-s span] [-T timeout] [-l logfile]\n");
   fprintf(stderr, "                    [-p pidfile] [-k] [-v] [-h|--help]\n");
   fprintf(stderr, "\n");
   fprintf(stderr, "   url:     URL where the xproofd under test responds [localhost:1093]\n");
   fprintf(stderr, "   test:    type of test [0]\n");
   fprintf(stderr, "                 0   ping\n");
   fprintf(stderr, "                 1   '0' + check connection for default user\n");
   fprintf(stderr, "                 2   '1' + check connection for all recent users\n");
   fprintf(stderr, "   sbdir:   sandbox directory used to find out the users of the facility\n");
   fprintf(stderr, "            when test is 2 ['/tmp/proofbox']\n");
   fprintf(stderr, "   span:    define the time interval to define 'recent' users when test\n");
   fprintf(stderr, "            is 2: only users who connected within this interval are checked;\n");
   fprintf(stderr, "            use -1 for infinite [-1]\n");
   fprintf(stderr, "   timeout: max time waited for a successful session start in seconds [10 secs]\n");
   fprintf(stderr, "   logfile: log file if not screen; deleted if the test fails unless '-k' is\n");
   fprintf(stderr, "            specified (see below) [terminal]\n");
   fprintf(stderr, "   pidfile: file with the daemon process id; if specified, an initial test of the\n");
   fprintf(stderr, "            process existence is done (using kill(pid, 0))\n");
   fprintf(stderr, "   -k:      keep log file at path given via '-l' in all cases [do not keep]\n");
   fprintf(stderr, "   -v:      set gDebug = 1 for ROOT\n");
   fprintf(stderr, "   -h, --help: print this screen\n");
   fprintf(stderr, "\n");
   fprintf(stderr, "   Return:  0   the required test succeeded\n");
   fprintf(stderr, "            1   the test failed\n");
   fprintf(stderr, "\n");
}

////////////////////////////////////////////////////////////////////////////////
/// Test proof open.
/// Return 0 on success, 1 on failure

int proof_open(const char *master, long to)
{
   int rc = 0;
   RedirectHandle_t rh;
   TString popenfile = TString::Format("%s/xpdtest_popen_file", gSystem->TempDirectory());
   gSystem->RedirectOutput(popenfile, "w", &rh);
   set_timer(1, to);
   TProof *p = TProof::Open(master, "masteronly");
   set_timer(0);
   gSystem->RedirectOutput(0, 0, &rh);
   if (!p || (p && !p->IsValid())) {
      TMacro m;
      m.ReadFile(popenfile);
      if (!m.GetLineWith("Server not allowed to be top master")) rc = 1;
   }
   if (p) delete p;
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// If we are called it means that we failed

static void handle_sigalarm(int)
{
   gSystem->Exit(1);
}

////////////////////////////////////////////////////////////////////////////////
/// Do nothing

static void ignore_signal(int)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Start an asynchronous firing after 'to' seconds
/// Return
///        0 on success
///        1 otherwise

int set_timer(bool on, long to)
{
   struct itimerval itv;
   bool sett = (on && to > 0) ? 1 : 0;
   itv.it_value.tv_sec     = (sett) ? time_t(to) : time_t(0);
   itv.it_value.tv_usec    = 0;
   itv.it_interval.tv_sec  = 0;
   itv.it_interval.tv_usec = 0;
   // Enable/disable handling of the related signal
   if (sett) {
      signal(SIGALRM, handle_sigalarm);
   } else {
      signal(SIGALRM, ignore_signal);
   }
   errno = 0;
   return setitimer(ITIMER_REAL, &itv, 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Non-blocking check for a PROOF service at 'url'
/// Return
///        0 if a XProofd daemon is listening at 'url'
///        1 otherwise

int xpd_ping(const char *host, int port)
{
   // Check arguments
   if (!host || (host && strlen(host) <= 0)) {
      fprintf(stderr,"xpd_ping: host must be given!\n");
      return 1;
   }

   struct hostent *h = gethostbyname(host);
   if (!h) {
      fprintf(stderr,"xpd_ping: unknown host '%s'\n", host);
      return 1;
   }

   // Get socket and listen to it
   int sd = getsocket(h,port);
   if (sd == -1) {
      fprintf(stderr,"xpd_ping: problems creating socket ... exit\n");
      return 1;
   }

   // Send the first bytes
   clnt_HS_t initHS;
   memset(&initHS, 0, sizeof(initHS));
   int len = sizeof(initHS);
   initHS.third  = (int)htonl((int)1);
   if (sendn(sd, &initHS, len) != len) {
      fprintf(stderr,"xpd_ping: problems sending first set of handshake bytes\n");
      close(sd);
      return 1;
   }

   // These 8 bytes are need by 'rootd/proofd' and discarded by XRD/XPD
   int dum[2];
   dum[0] = (int)htonl((int)4);
   dum[1] = (int)htonl((int)2012);
   if (sendn(sd, &dum[0], sizeof(dum)) != sizeof(dum)) {
      fprintf(stderr,"xpd_ping: problems sending second set of handshake bytes\n");
      close(sd);
      return 1;
   }

   // Read first server response
   int type;
   len = sizeof(type);
   int nr = 0;
   if ((nr = recvn(sd, &type, len)) != len) { // 4 bytes
      fprintf(stderr, "xpd_ping: 1st: wrong number of bytes read: %d (expected: %d)\n",
                      nr, len);
      close(sd);
      return 1;
   }

   // To host byte order
   type = ntohl(type);
   // Check if the server is the eXtended proofd
   if (type == 0) {
      srv_HS_t xbody;
      len = sizeof(xbody);
      if ((nr = recvn(sd, &xbody, len)) != len) { // 12(4+4+4) bytes
         fprintf(stderr, "xpd_ping: 2nd: wrong number of bytes read: %d (expected: %d)\n",
                         nr, len);
         close(sd);
         return 1;
      }
      xbody.protover = ntohl(xbody.protover);
      xbody.msgval = ntohl(xbody.msglen);
      xbody.msglen = ntohl(xbody.msgval);

   } else if (type == 8) {
      // Standard proofd
      fprintf(stderr, "xpd_ping: server is PROOFD\n");
      close(sd);
      return 1;
   } else {
      // We don't know the server type
      fprintf(stderr, "xpd_ping: unknown server type: %d\n", type);
      close(sd);
      return 1;
   }
   // Cleanup
   close(sd);
   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////

int getsocket(struct hostent *h, int port)
{
   int sd, rc;
   struct sockaddr_in localAddr = sockaddr_in();
   struct sockaddr_in servAddr = sockaddr_in();

   servAddr.sin_family = h->h_addrtype;
   memcpy((char *) &servAddr.sin_addr.s_addr, h->h_addr_list[0], h->h_length);
   servAddr.sin_port = htons(port);

   /* create socket */
   sd = socket(AF_INET, SOCK_STREAM, 0);
   if(sd < 0) {
      perror("cannot open socket ");
      return -1;
   }

   /* bind any port number */
   localAddr.sin_family = AF_INET;
   localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
   localAddr.sin_port = htons(0);

   rc = bind(sd, (struct sockaddr *) &localAddr, sizeof(localAddr));
   if(rc < 0) {
      perror("error ");
      close(sd);
      return -1;
   }

   /* connect to server */
   rc = connect(sd, (struct sockaddr *) &servAddr, sizeof(servAddr));
   if(rc < 0) {
      perror("cannot connect ");
      close(sd);
      return -1;
   }

   return sd;

}

////////////////////////////////////////////////////////////////////////////////
/// Send exactly length bytes from buffer.

int sendn(int sock, const void *buffer, int length)
{
   if (sock < 0) return -1;

   int n, nsent = 0;
   const char *buf = (const char *)buffer;

   for (n = 0; n < length; n += nsent) {
      if ((nsent = send(sock, buf+n, length-n, 0)) <= 0) {
         perror("problems sending ");
         return nsent;
      }
   }

   return n;
}

////////////////////////////////////////////////////////////////////////////////
/// Receive exactly length bytes into buffer. Returns number of bytes
/// received. Returns -1 in case of error.

int recvn(int sock, void *buffer, int length)
{
   if (sock < 0) return -1;

   int n, nrecv = 0;
   char *buf = (char *)buffer;

   for (n = 0; n < length; n += nrecv) {
      while ((nrecv = recv(sock, buf+n, length-n, 0)) == -1 && errno == EINTR)
         errno = 0;   // probably a SIGCLD that was caught
      if (nrecv < 0) {
         perror("problems receiving ");
         return nrecv;
      } else if (nrecv == 0)
         break;         // EOF
   }

   return n;
}
