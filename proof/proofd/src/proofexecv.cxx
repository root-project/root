// @(#)root/main:$Id$
// Author: Gerardo Ganis Mar 2011

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// proofexecv                                                           //
//                                                                      //
// Program executed via system starting proofserv instances.            //
// It also performs other actions requiring a separate process, e.g.    //
// XrdProofAdmin file system requests.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <syslog.h>
#include <errno.h>
#include <pwd.h>
#include <ios>
#include <fstream>
#include <list>
#include <string>
#include <string.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <grp.h>
#include <dirent.h>

#include "Varargs.h"
#include "rpdconn.h"
#include "rpdpriv.h"

#ifdef R__GLOBALSTL
namespace std { using ::string; }
#endif

static int gType = 0;
static int gDebug = 0;
static FILE *gLogger = 0;

#define kMAXPATHLEN 4096

int assertdir(const std::string &path, uid_t u, gid_t g, unsigned int mode);
int changeown(const std::string &path, uid_t u, gid_t g);
int exportsock(rpdunix *conn);
int loginuser(const std::string &home, const std::string &user, uid_t u, gid_t g);
int mvfile(const std::string &from, const std::string &to, uid_t u, gid_t g, unsigned int mode);
int completercfile(const std::string &rcfile, const std::string &sessdir,
                   const std::string &stag, const std::string &adminpath);
int setownerships(int euid, const std::string &us, const std::string &gr,
                  const std::string &creds, const std::string &dsrcs,
                  const std::string &ddir, const std::string &ddiro,
                  const std::string &ord, const std::string &stag);
int setproofservenv(const std::string &envfile,
                    const std::string &logfile, const std::string &rcfile);
int redirectoutput(const std::string &logfile);

void start_ps(int argc, char **argv);
void start_rootd(int argc, char **argv);

//______________________________________________________________________________
void Info(const char *va_(fmt), ...)
{
   // Write info message to syslog.

   char    buf[kMAXPATHLEN];
   va_list ap;

   va_start(ap,va_(fmt));
   vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);

   if (gLogger)
      fprintf(gLogger, "proofexecv: %s\n", buf);
   else
      fprintf(stderr, "proofexecv: %s\n", buf);
}

//______________________________________________________________________________
int main(int argc, char **argv)
{
   // Program executed via system starting proofserv instances.
   // It also performs other actions requiring a separate process, e.g.
   // XrdProofAdmin file system requests.

   // Default logger
   gLogger = stderr;
   if (argc < 3) {
      Info("argc=%d: at least 2 additional argument (the process type and debug level) are required - exit",
           argc);
      exit(1);
   }
   if ((gType = atoi(argv[1])) < 0) {
      Info("ERROR: invalid process type %d (must be > 0) - exit", gType);
      exit(1);
   }
   gDebug = atoi(argv[2]);

   if (gType <= 3) {
      // Start a proofserv process
      start_ps(argc, argv);
      exit(1);
   } else if (gType == 20) {
      // Start a rootd to serve a file
      start_rootd(argc, argv);
      exit(1);
   } else {
      Info("ERROR: process type %d not yet implemented", gType);
      exit(1);
   }

   // Done
   exit(0);
}

//______________________________________________________________________________
void start_rootd(int argc, char **argv)
{
   // Process a request to start a rootd server

   if (argc < 6) {
      Info("argc=%d: at least 5 additional arguments required - exit", argc);
      return;
   }

   // Parse arguments:
   //     1     process type (2=top-master, 1=sub-master 0=worker, 3=test, 10=admin, 20=rootd)
   //     2     debug level
   //     3     path to unix socket to the parent to receive the open descriptor
   //     4     path to rootd executable
   //   >=5     arguments to rootd

   // Call back the parent, so that it can move to other processes
   std::string sockpath = argv[3];
   rpdunix *uconn = new rpdunix(sockpath.c_str());
   if (!uconn || (uconn && !uconn->isvalid(0))) {
      Info("ERROR: failure calling back parent on '%s'", sockpath.c_str());
      return;
   }

   int rcc = 0;
   // Receive the open descriptor to be used in rootd
   int fd = -1;
   if ((rcc = uconn->recvdesc(fd)) != 0) {
      Info("ERROR: failure receiving open descriptor from parent (errno: %d)", -rcc);
      delete uconn;
      return;
   }
   // Close the connection to the parent
   delete uconn;

   // Force stdin/out to point to the socket FD (this will also bypass the
   // close on exec setting for the socket)
   if (dup2(fd, STDIN_FILENO) != 0)
      Info("WARNING: failure duplicating STDIN (errno: %d)", errno);
   if (dup2(fd, STDOUT_FILENO) != 0)
      Info("WARNING: failure duplicating STDOUT (errno: %d)", errno);

   // Prepare execv
   int na = argc - 4;
   char **argvv = new char *[na + 1];

   // Fill arguments
   argvv[0] = argv[4];
   int ia = 5, ka = 1;
   while (ia < argc) {
      argvv[ka] = argv[ia];
      ka++; ia++;
   }
   argvv[na] = 0;

   // Run the program
   execv(argv[4], argvv);

   // We should not be here!!!
   Info("ERROR: returned from execv: bad, bad sign !!!");
   delete [] argvv;
   return;
}

//______________________________________________________________________________
void start_ps(int argc, char **argv)
{
   // Process a request to start a proofserv process

   if (argc < 6) {
      Info("argc=%d: at least 5 additional arguments required - exit", argc);
      return;
   }

   // Parse arguments:
   //     1     process type (2=top-master, 1=sub-master 0=worker, 3=test, 10=admin, 20=rootd)
   //     2     debug level
   //     3     user name
   //     4     root path for relevant directories and files (to be completed with PID)
   //     5     path to unix socket to be used to call back the parent
   //     6     log files for errors (prior to final log redirection)

#if 0
   int dbg = 1;
   while (dbg) {}
#endif

   // Open error logfile
   std::string errlog(argv[6]);
   if (!(gLogger = fopen(errlog.c_str(), "a"))) {
      Info("FATAL: could not open '%s' for error logging - errno: %d",
           errlog.c_str(), (int) errno);
      return;
   }

   // Pid string
   char spid[20];
   snprintf(spid, 20, "%d", (int)getpid());

   // Identity of session's owner
   std::string user = argv[3];
   struct passwd *pw = getpwnam(user.c_str());
   if (!pw) {
      Info("ERROR: could noy get identity info for '%s' - errno: %d", user.c_str(), (int) errno);
      return;
   }
   uid_t uid = pw->pw_uid;
   uid_t gid = pw->pw_gid;

   std::string::size_type loc = 0;

   // All relevant files an directories derived from argv[4], inclusing base-path for temporary
   // env- and rc-files
   std::string sessdir(argv[4]), logfile(argv[4]), tenvfile, trcfile;
   if (gType == 2) {
      // Top master
      if ((loc = sessdir.rfind('/')) != std::string::npos) sessdir.erase(loc, std::string::npos);
      tenvfile = sessdir;
   } else {
      // Sub-masters, workers (the session dir is already fully defined ...)
      tenvfile = sessdir;
      if ((loc = sessdir.rfind('/')) != std::string::npos) sessdir.erase(loc, std::string::npos);
   }
   if ((loc = tenvfile.rfind("<pid>")) != std::string::npos) tenvfile.erase(loc, std::string::npos);
   trcfile = tenvfile;
   tenvfile += ".env";
   trcfile += ".rootrc";

   // Complete the session dir path and assert it
   if ((loc = sessdir.find("<pid>")) != std::string::npos) sessdir.replace(loc, 5, spid);
   if (assertdir(sessdir, uid, gid, 0755) != 0) {
      Info("ERROR: could not assert dir '%s'", sessdir.c_str());
      return;
   }
   Info("session dir: %s", sessdir.c_str());

   // The session files now
   while ((loc = logfile.find("<pid>")) != std::string::npos) { logfile.replace(loc, 5, spid); }
   std::string stag(logfile), envfile(logfile), userdir(logfile), rcfile(logfile);
   logfile += ".log";
   envfile += ".env";
   rcfile += ".rootrc";

   // Assert working directory
   if (assertdir(userdir, uid, gid, 0755) != 0) {
      Info("ERROR: could not assert dir '%s'", userdir.c_str());
      return;
   }

   // The session tag
   if ((loc = stag.rfind('/')) != std::string::npos) stag.erase(0, loc);
   if ((loc = stag.find('-')) != std::string::npos) loc = stag.find('-', loc+1);
   if (loc != std::string::npos) stag.erase(0, loc+1);
   Info("session tag: %s", stag.c_str());

   // Call back the parent, so that it can move to other processes
   std::string sockpath = argv[5];
   rpdunix *uconn = new rpdunix(sockpath.c_str());
   if (!uconn || (uconn && !uconn->isvalid(0))) {
      Info("ERROR: failure calling back parent on '%s'", sockpath.c_str());
      if (uconn) delete uconn;
      return;
   }

   // Send the pid
   int rcc = 0;
   if ((rcc = uconn->send((int) getpid())) != 0) {
      Info("ERROR: failure sending pid to parent (errno: %d)", -rcc);
      delete uconn;
      return;
   }

   // Receive the adminpath and the executable path
   rpdmsg msg;
   if ((rcc = uconn->recv(msg)) != 0) {
      Info("ERROR: failure receiving admin path and executable from parent (errno: %d)", -rcc);
      delete uconn;
      return;
   }
   int ppid;
   std::string srvadmin, adminpath, pspath;
   msg >> srvadmin >> adminpath >> pspath >> ppid;
   Info("srv admin path: %s", srvadmin.c_str());
   Info("partial admin path: %s", adminpath.c_str());
   Info("executable: %s", pspath.c_str());
   Info("parent pid: %d", ppid);

   // Receive information about dataset and data dir(s)
   msg.reset();
   if ((rcc = uconn->recv(msg)) != 0) {
      Info("ERROR: failure receiving information about dataset and data dir(s) from parent (errno: %d)", -rcc);
      delete uconn;
      return;
   }
   int euid;
   std::string group, creds, ord, datadir, ddiropts, datasetsrcs;
   msg >> euid >> group >> creds >> ord >> datadir >> ddiropts >> datasetsrcs;
   Info("euid at startup: %d", euid);
   Info("group, ord: %s, %s", group.c_str(), ord.c_str());
   Info("datadir: %s", datadir.c_str());
   Info("datasetsrcs: %s", datasetsrcs.c_str());

   // Set user ownerships
   if (setownerships(euid, user, group, creds, datasetsrcs, datadir, ddiropts,
                     ord, stag) != 0) {
      Info("ERROR: problems setting relevant user ownerships");
      delete uconn;
      return;
   }

   // Move the environment configuration file in the session directory
   if (mvfile(tenvfile, envfile, uid, gid, 0644) != 0) {
      Info("ERROR: problems renaming '%s' to '%s' (errno: %d)",
           tenvfile.c_str(), envfile.c_str(), errno);
      delete uconn;
      return;
   }
   // Move the rootrc file in the session directory
   if (mvfile(trcfile, rcfile, uid, gid, 0644) != 0) {
      Info("ERROR: problems renaming '%s' to '%s' (errno: %d)",
           trcfile.c_str(), rcfile.c_str(), errno);
      delete uconn;
      return;
   }

   // Add missing information to the rc file
   if (completercfile(rcfile, userdir, stag, adminpath) != 0) {
      Info("ERROR: problems completing '%s'", rcfile.c_str());
      delete uconn;
      return;
   }
   // Set the environment following the content of the env file
   if (setproofservenv(envfile, logfile, rcfile) != 0) {
      Info("ERROR: problems setting environment from '%s'", envfile.c_str());
      delete uconn;
      return;
   }

   // Export the file descriptor
   if (exportsock(uconn) != 0) {
      Info("ERROR: problems exporting file descriptor");
      delete uconn;
      return;
   }
   delete uconn;

   // Login now
   if (loginuser(userdir, user, uid, gid) != 0) {
      Info("ERROR: problems login user '%s' in", user.c_str());
      return;
   }

#if 1
   // Redirect the logs now
   if (redirectoutput(logfile) != 0) {
      Info("ERROR: problems redirecting logs to '%s'", logfile.c_str());
      return;
   }
#endif

   // Prepare for execv
   char *argvv[6] = {0};

   char *sxpd = 0;
   if (adminpath.length() > 0) {
      // We add our admin path to be able to identify processes coming from us
      int len = srvadmin.length() + strlen("xpdpath:") + 1;
      sxpd = new char[len];
      snprintf(sxpd, len, "xpdpath:%s", adminpath.c_str());
   } else {
      // We add our PID to be able to identify processes coming from us
      sxpd = new char[10];
      snprintf(sxpd, 10, "%d", ppid);
   }

   // Log level
   char slog[10] = {0};
   snprintf(slog, 10, "%d", gDebug);

   // Fill arguments
   argvv[0] = (char *) pspath.c_str();
   argvv[1] = (char *)((gType == 0) ? "proofslave" : "proofserv");
   argvv[2] = (char *)"xpd";
   argvv[3] = (char *)sxpd;
   argvv[4] = (char *)slog;
   argvv[5] = 0;

   // Unblock SIGUSR1 and SIGUSR2
   sigset_t myset;
   sigemptyset(&myset);
   sigaddset(&myset, SIGUSR1);
   sigaddset(&myset, SIGUSR2);
   pthread_sigmask(SIG_UNBLOCK, &myset, 0);

   Info("%d: uid: %d, euid: %d", (int)getpid(), getuid(), geteuid());
   Info("argvv: '%s' '%s' '%s' '%s' '%s'", argvv[0],  argvv[1], argvv[2], argvv[3],  argvv[4]);

   // Run the program
   execv(pspath.c_str(), argvv);

   // We should not be here!!!
   Info("ERROR: returned from execv: bad, bad sign !!!");
   return;
}

//_______________________________________________________________________________
int loginuser(const std::string &home, const std::string &user, uid_t uid, gid_t gid)
{
   // Login the user in its space

   if (chdir(home.c_str()) != 0) {
      Info("loginuser: ERROR: can't change directory to %s, euid: %d, uid: %d; errno: %d",
           home.c_str(), geteuid(), getuid(), errno);
      return -1;
   }

   // set HOME env
   size_t len = home.length() + 8;
   char *h = new char[len];
   snprintf(h, len, "HOME=%s", home.c_str());
   putenv(h);
   if (gDebug > 0) Info("loginuser: set '%s'", h);

   // set USER env
   char *u = new char[len];
   snprintf(u, len, "USER=%s", user.c_str());
   putenv(u);
   if (gDebug > 0) Info("loginuser: set '%s'", u);

   // Set access control list from /etc/initgroup
   // (super-user privileges required)
   if (geteuid() != uid) {
      rpdprivguard pguard((uid_t)0, (gid_t)0);
      if (rpdbadpguard(pguard, uid)) {
         Info("loginuser: ERROR: could not get required privileges");
         return -1;
      }
      initgroups(user.c_str(), gid);
   }

   // acquire permanently target user privileges
   if (gDebug > 0)
      Info("loginuser: acquiring target user identity (%d,%d)", uid, gid);
   if (rpdpriv::changeperm(uid, gid) != 0) {
      Info("loginuser: ERROR: can't acquire '%s' identity", user.c_str());
      return -1;
   }

   // Done
   return 0;
}

//_____________________________________________________________________________
int assertdir(const std::string &path, uid_t u, gid_t g, unsigned int mode)
{
   // Make sure that 'path' exists, it is owned by the entity
   // described by {u,g} and its mode is 'mode'.
   // Return 0 in case of success, -1 in case of error

   if (path.length() <= 0) return -1;

   rpdprivguard pguard((uid_t)0, (gid_t)0);
   if (rpdbadpguard(pguard, u)) {
      Info("assertdir: ERROR: could not get privileges (errno: %d)", errno);
      return -1;
   }

   // Make the directory: ignore failure if already existing ...
   if (mkdir(path.c_str(), mode) != 0 && (errno != EEXIST)) {
      Info("assertdir: ERROR: unable to create path: %s (errno: %d)", path.c_str(), errno);
      return -1;
   }
   // Set ownership of the path to the client
   if (chown(path.c_str(), u, g) == -1) {
      Info("assertdir: ERROR: unable to set ownership on path: %s (errno: %d)", path.c_str(), errno);
      return -1;
   }

   // We are done
   return 0;
}

//_____________________________________________________________________________
int mvfile(const std::string &from, const std::string &to, uid_t u, gid_t g, unsigned int mode)
{
   // Move file form 'from' to 'to', making sure that it is owned by the entity
   // described by {u,g} and its mode is 'mode' (at the final destination).
   // Return 0 in case of success, -1 in case of error

   if (from.length() <= 0 || to.length() <= 0) return -1;

   rpdprivguard pguard((uid_t)0, (gid_t)0);
   if (rpdbadpguard(pguard, u)) {
      Info("mvfile: ERROR: could not get privileges (errno: %d)", errno);
      return -1;
   }

   // Rename the file
   if (rename(from.c_str(), to.c_str()) != 0) {
      Info("mvfile: ERROR: unable to rename '%s' to '%s' (errno: %d)", from.c_str(), to.c_str(), errno);
      return -1;
   }

   // Set ownership of the path to the client
   if (chmod(to.c_str(), mode) == -1) {
      Info("mvfile: ERROR: unable to set mode %o on path: %s (errno: %d)", mode, to.c_str(), errno);
      return -1;
   }

   // Make sure the ownership is right
   if (chown(to.c_str(), u, g) == -1) {
      Info("mvfile: ERROR: unable to set ownership on path: %s (errno: %d)", to.c_str(), errno);
      return -1;
   }

   // We are done
   return 0;
}

//_____________________________________________________________________________
int completercfile(const std::string &rcfile, const std::string &sessdir,
                   const std::string &stag, const std::string &adminpath)
{
   // Finalize the rc file with the missing pieces

   FILE *frc = fopen(rcfile.c_str(), "a");
   if (!frc) {
      Info("completercfile: ERROR: unable to open rc file: '%s' (errno: %d)", rcfile.c_str(), errno);
      return -1;
   }

   fprintf(frc, "# The session working dir\n");
   fprintf(frc, "ProofServ.SessionDir: %s\n", sessdir.c_str());

   fprintf(frc, "# Session tag\n");
   fprintf(frc, "ProofServ.SessionTag: %s\n", stag.c_str());

   fprintf(frc, "# Admin path\n");
   fprintf(frc, "ProofServ.AdminPath: %s%d.status\n", adminpath.c_str(), (int)getpid());

   fclose(frc);

   // Done
   return 0;
}

//_____________________________________________________________________________
int setproofservenv(const std::string &envfile,
                    const std::string &logfile, const std::string &rcfile)
{
   // Initialize the environment following the content of 'envfile'

   if (envfile.length() <= 0) return -1;

   int len = 0;
   char *h = 0;
   // The logfile path
   len = logfile.length() + strlen("ROOTPROOFLOGFILE") + 4;
   h = new char[len + 1];
   snprintf(h, len + 1, "ROOTPROOFLOGFILE=%s", logfile.c_str());
   putenv(h);
   if (gDebug > 0)
      Info("setproofservenv: set '%s'", h);
   // The rcfile path
   len = rcfile.length() + strlen("ROOTRCFILE") + 4;
   h = new char[len + 1];
   snprintf(h, len + 1, "ROOTRCFILE=%s", rcfile.c_str());
   putenv(h);
   if (gDebug > 0)
      Info("setproofservenv: set '%s'", h);

   std::fstream fin(envfile.c_str(), std::ios::in);
   if (!fin.good()) {
      Info("setproofservenv: ERROR: unable to open env file: %s (errno: %d)", envfile.c_str(), errno);
      return -1;
   }

   std::string line;
   while (!fin.eof()) {
      std::getline(fin, line);
      if (line[line.length()-1] == '\n') line.erase(line.length()-1);
      if (line.length() > 0) {
         h = new char[line.length() + 1];
         snprintf(h, line.length()+1, "%s", line.c_str());
         putenv(h);
         if (gDebug > 0)
            Info("setproofservenv: set '%s'", h);
      }
   }
   // Close the stream
   fin.close();
   // Done
   return 0;
}

//_____________________________________________________________________________
int exportsock(rpdunix *conn)
{
   // Export the descriptor of 'conn' so that it can used in the execv application.
   // Make sure it duplicates to a reasonable value first.
   // Return 0 on success, -1 on error

   // Check the input connection
   if (!conn || (conn && !conn->isvalid(0))) {
      Info("exportsock: ERROR: connection is %s", (conn ? "invalid" : "undefined"));
      return -1;
   }

   // Get the descriptor
   int d = conn->exportfd();

   // Make sure it is outside the standard I/O range
   if (d == 0 || d == 1 || d == 2) {
      int fd, natt = 1000;
      while (natt > 0 && (fd = dup(d)) <= 2) { natt--; }
      if (natt <= 0 && fd <= 2) {
         Info("exportsock: ERROR: no free filedescriptor!");
         return -1;
      }
      close(d);
      d = fd;
      close(2);
      close(1);
      close(0);
   }

   // Export the descriptor in the env ROOTOPENSOCK
   char *rootopensock = new char[33];
   snprintf(rootopensock, 33, "ROOTOPENSOCK=%d", d);
   putenv(rootopensock);

   // Done
   return 0;
}

//______________________________________________________________________________
int redirectoutput(const std::string &logfile)
{
   // Redirect stdout to 'logfile'
   // On success return 0. Return -1 on failure.

   if (gDebug > 0)
      Info("redirectoutput: enter: %s", logfile.c_str());

   if (logfile.length() <= 0) {
      Info("redirectoutput: ERROR:  logfile path undefined");
      return -1;
   }

   if (gDebug > 0)
      Info("redirectoutput: reopen %s", logfile.c_str());
   FILE *flog = freopen(logfile.c_str(), "a", stdout);
   if (!flog) {
      Info("redirectoutput: ERROR:  could not freopen stdout (errno: %d)", errno);
      return -1;
   }

   if (gDebug > 0)
      Info("redirectoutput: dup2 ...");
   if ((dup2(fileno(stdout), fileno(stderr))) < 0) {
      Info("redirectoutput: ERROR:  could not redirect stderr (errno: %d)", errno);
      return -1;
   }

   // Close the error logger
   if (gLogger != stderr) fclose(gLogger);
   gLogger = 0;

   // Export the descriptor in the env ROOTPROOFDONOTREDIR
   int len = strlen("ROOTPROOFDONOTREDIR=2");
   char *notredir = new char[len + 1];
   snprintf(notredir, len+1, "ROOTPROOFDONOTREDIR=2");
   putenv(notredir);

   if (gDebug > 0)
      Info("redirectoutput: done!");
   // We are done
   return 0;
}

//___________________________________________________________________________
int setownerships(int euid, const std::string &us, const std::string &gr,
                  const std::string &creds, const std::string &dsrcs,
                  const std::string &ddir, const std::string &ddiro,
                  const std::string &ord, const std::string &stag)
{
   // Set user ownerships on some critical files or directories.
   // Return 0 on success, -1 if enything goes wrong.

   // Get identities
   struct passwd *pwad, *pwus;
   if (!(pwad = getpwuid(euid))) {
      Info("setownerships: ERROR: problems getting 'struct passwd' for"
                                " uid: %d (errno: %d)", euid, (int)errno);
      return -1;
   }
   if (!(pwus = getpwnam(us.c_str()))) {
      Info("setownerships: ERROR: problems getting 'struct passwd' for"
                                " user: '%s' (errno: %d)", us.c_str(), (int)errno);
      return -1;
   }

   // If applicable, make sure that the private dataset dir for this user exists
   // and has the right permissions
   if (dsrcs.length() > 0) {
      std::string dsrc(dsrcs);
      std::string::size_type loc = dsrcs.find(',', 0);
      do {
         if (loc != std::string::npos) dsrc.erase(loc, std::string::npos);
         if (dsrc.length() > 0) {
            std::string d(dsrc);
            // Analyse now
            d += "/"; d += gr;
            if (assertdir(d, pwad->pw_uid, pwad->pw_gid, 0777) == 0) {
               d += "/"; d += us;
               if (assertdir(d, pwus->pw_uid, pwus->pw_gid, 0755) != 0) {
                  Info("setownerships: ERROR: problems asserting '%s' in mode 0755"
                                     " (errno: %d)", d.c_str(), (int)errno);
               }
            } else {
               Info("setownerships: ERROR: problems asserting '%s' in mode 0777"
                                  " (errno: %d)", d.c_str(), (int)errno);
            }
         }
         dsrc.assign(dsrcs, loc + 1, dsrcs.length() - loc);
         loc++;
      } while ((loc = dsrcs.find(',', loc)) != std::string::npos);
   }

   // If applicable, make sure that the private data dir for this user exists
   // and has the right permissions
   if (ddir.length() > 0 && ord.length() > 0 && stag.length() > 0) {
      std::string dgr(ddir);
      dgr += "/"; dgr += gr;
      if (assertdir(dgr, pwad->pw_uid, pwad->pw_gid, 0777) == 0) {
         int drc = -1;
         unsigned int mode = 0755;
         if (ddiro.find('g') != std::string::npos) mode = 0775;
         if (ddiro.find('a') != std::string::npos ||
             ddiro.find('o') != std::string::npos) mode = 0777;
         std::string dus(dgr);
         dus += "/"; dus += us;
         if (assertdir(dus, pwus->pw_uid, pwus->pw_gid, mode) == 0) {
            dus += "/"; dus += ord;
            if (assertdir(dus, pwus->pw_uid, pwus->pw_gid, mode) == 0) {
               dus += "/"; dus += stag;
               if (assertdir(dus, pwus->pw_uid, pwus->pw_gid, mode) == 0) drc = 0;
            }
         }
         if (drc == -1)
            Info("setownerships: ERROR: problems asserting '%s' in mode %o"
                               " (errno: %d)", dus.c_str(), mode, (int)errno);
      } else {
         Info("setownerships: ERROR: problems asserting '%s' in mode 0777"
                              " (errno: %d)", dgr.c_str(), (int)errno);
      }
   }

   // The credential directory
   if (creds.length() > 0) {
      if (changeown(creds, pwus->pw_uid, pwus->pw_gid) != 0) {
         Info("setownerships: ERROR: problems changing owenership of '%s'", creds.c_str());
         return -1;
      }
   }

   // Done
   return 0;
}

//_____________________________________________________________________________
int changeown(const std::string &path, uid_t u, gid_t g)
{
   // Change the ownership of 'path' to the entity described by {u,g}.
   // If 'path' is a directory, go through the paths inside it recursively.
   // Return 0 in case of success, -1 in case of error

   if (path.length() <= 0) return -1;

   // If is a directory apply this on it
   DIR *dir = opendir(path.c_str());
   if (dir) {
      // Loop over the dir
      std::string proot(path);
      if (!(proot.rfind('/') !=  proot.length() - 1)) proot += "/";

      struct dirent *ent = 0;
      while ((ent = readdir(dir))) {
         if (ent->d_name[0] == '.' || !strcmp(ent->d_name, "..")) continue;
         std::string fn(proot);
         fn += ent->d_name;

         // Apply recursively
         if (changeown(fn.c_str(), u, g) != 0) {
            Info("changeown: ERROR: problems changing recursively ownership of '%s'",
                  fn.c_str());
            closedir(dir);
            return -1;
         }

      }
      // Close the directory
      closedir(dir);
   } else {
      // If it was a directory and opening failed, we fail
      if (errno != 0 && (errno != ENOTDIR)) {
         Info("changeown: ERROR: problems opening '%s' (errno: %d)",
              path.c_str(), (int)errno);
         return -1;
      }
      // Else it may be a file ... get the privileges, if needed
      rpdprivguard pguard((uid_t)0, (gid_t)0);
      if (rpdbadpguard(pguard, u)) {
         Info("changeown: ERROR: could not get privileges (errno: %d)", errno);
         return -1;
      }
      // Set ownership of the path to the client
      if (chown(path.c_str(), u, g) == -1) {
         Info("changeown: ERROR: cannot set user ownership on path '%s' (errno: %d)",
               path.c_str(), errno);
         return -1;
      }
   }

   // We are done
   return 0;
}
