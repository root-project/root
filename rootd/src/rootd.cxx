// @(#)root/rootd:$Name:  $:$Id: rootd.cxx,v 1.75 2003/12/11 11:23:07 rdm Exp $
// Author: Fons Rademakers   11/08/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*
 * Parts of this file are copied from the MIT krb5 distribution and
 * are subject to the following license:
 *
 * Copyright 1990,1991 by the Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * Export of this software from the United States of America may
 *   require a specific license from the United States Government.
 *   It is the responsibility of any person or organization contemplating
 *   export to obtain such a license before exporting.
 *
 * WITHIN THAT CONSTRAINT, permission to use, copy, modify, and
 * distribute this software and its documentation for any purpose and
 * without fee is hereby granted, provided that the above copyright
 * notice appear in all copies and that both that copyright notice and
 * this permission notice appear in supporting documentation, and that
 * the name of M.I.T. not be used in advertising or publicity pertaining
 * to distribution of the software without specific, written prior
 * permission.  Furthermore if you modify this software you must label
 * your software as modified software and not distribute it in such a
 * fashion that it might be confused with the original M.I.T. software.
 * M.I.T. makes no representations about the suitability of
 * this software for any purpose.  It is provided "as is" without express
 * or implied warranty.
 */

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Rootd                                                                //
//                                                                      //
// Root remote file server daemon.                                      //
// This small server is started either by inetd when a client requests  //
// a connection to a rootd server or by hand (i.e. from the command     //
// line). The rootd server works with the ROOT TNetFile class. It       //
// allows remote access to ROOT database files in either read or        //
// write mode. By default TNetFile uses port 1094 (allocated by IANA,   //
// www.iana.org, to rootd). To run rootd via inetd add the              //
// following line to /etc/services:                                     //
//                                                                      //
// rootd     1094/tcp                                                   //
//                                                                      //
// and to /etc/inetd.conf:                                              //
//                                                                      //
// rootd stream tcp nowait root /usr/local/root/bin/rootd rootd -i      //
//                                                                      //
// Force inetd to reread its conf file with "kill -HUP <pid inetd>".    //
//                                                                      //
// If xinetd is used instead, a file named 'rootd' should be created    //
// under /etc/xinetd.d with content:                                    //
//                                                                      //
// # default: off                                                       //
// # description: The root daemon                                       //
// #                                                                    //
// service rootd                                                        //
// {                                                                    //
//      disable         = no                                            //
//      flags           = REUSE                                         //
//      socket_type     = stream                                        //
//      wait            = no                                            //
//      user            = root                                          //
//      server          = /usr/local/root/bin/rootd                     //
//      server_args     = -i -d 0                                       //
// }                                                                    //
//                                                                      //
// and xinetd restarted (/sbin/service xinetd restart).                 //
//                                                                      //
// You can also start rootd by hand running directly under your private //
// account (no root system priviliges needed). For example to start     //
// rootd listening on port 5151 just type:                              //
//                                                                      //
// rootd -p 5151                                                        //
//                                                                      //
// Notice: no & is needed. Rootd will go in background by itself.       //
// In this case, the port number and process id will be printed, e.g.   //
//                                                                      //
// ROOTD_PORT=5151                                                      //
// ROOTD_PID=14433                                                      //
//                                                                      //
// Rootd arguments:                                                     //
//   -i                says we were started by inetd                    //
//   -p port#          specifies a different port to listen on.         //
//                     Use port1-port2 to find first available port in  //
//                     range. Use 0-N for range relative to service     //
//                     port.                                            //
//   -b tcpwindowsize  specifies the tcp window size in bytes (e.g. see //
//                     http://www.psc.edu/networking/perf_tune.html)    //
//                     Default is 65535. Only change default for pipes  //
//                     with a high bandwidth*delay product.             //
//   -d level          level of debug info written to syslog            //
//                     0 = no debug (default)                           //
//                     1 = minimum                                      //
//                     2 = medium                                       //
//                     3 = maximum                                      //
//   -r                files can only be opened in read-only mode       //
//   -f                do not run as daemon, run in the foreground      //
//   -P file           use this password file, instead of .srootdpass   //
//   -R bitmask        bit mask specifies which methods will allow      //
//                     authentication to be re-used                     //
//   -S keytabfile     use this keytab file, instead of the default     //
//                     (option only supported when compiled with        //
//                     Kerberos5 support)                               //
//   -T <tmpdir>       specifies the directory path to be used to place //
//                     temporary files; default is /usr/tmp.            //
//                     Useful if not running as root.                   //
//   -G gridmapfile    defines the gridmap file to be used for globus   //
//                     authentication if different from globus default  //
//                     (/etc/grid-security/gridmap); (re)defines the    //
//                     GRIDMAP environment variable.                    //
//   -C hostcertfile   defines a file where to find information for the //
//                     local host Globus information (see GLOBUS.README //
//                     for details)                                     //
//   -s <sshd_port>    specifies the port number for the sshd daemon    //
//                     (default is 22)                                  //
//   rootsys_dir       directory containing the ROOT etc and bin        //
//                     directories. Superseeds ROOTSYS or built-in      //
//                     (as specified to ./configure).                   //
//                                                                      //
// Rootd can also be configured for anonymous usage (like anonymous     //
// ftp). To setup rootd to accept anonymous logins do the following     //
// (while being logged in as root):                                     //
//                                                                      //
// - Add the following line to /etc/passwd:                             //
//                                                                      //
//   rootd:*:71:72:Anonymous rootd:/var/spool/rootd:/bin/false          //
//                                                                      //
//   where you may modify the uid, gid (71, 72) and the home directory  //
//   to suite your system.                                              //
//                                                                      //
// - Add the following line to /etc/group:                              //
//                                                                      //
//   rootd:*:72:rootd                                                   //
//                                                                      //
//   where the gid must match the gid in /etc/passwd.                   //
//                                                                      //
// - Create the directories:                                            //
//                                                                      //
//   mkdir /var/spool/rootd                                             //
//   mkdir /var/spool/rootd/tmp                                         //
//   chmod 777 /var/spool/rootd/tmp                                     //
//                                                                      //
//   Where /var/spool/rootd must match the rootd home directory as      //
//   specified in the rootd /etc/passwd entry.                          //
//                                                                      //
// - To make writeable directories for anonymous do, for example:       //
//                                                                      //
//   mkdir /var/spool/rootd/pub                                         //
//   chown rootd:rootd /var/spool/rootd/pub                             //
//                                                                      //
// That's all.                                                          //
//                                                                      //
// Several remarks:                                                     //
//  - you can login to an anonymous server either with the names        //
//    "anonymous" or "rootd".                                           //
//  - the passwd should be of type user@host.do.main. Only the @ is     //
//    enforced for the time being.                                      //
//  - in anonymous mode the top of the file tree is set to the rootd    //
//    home directory, therefore only files below the home directory     //
//    can be accessed.                                                  //
//  - anonymous mode only works when the server is started via inetd.   //
//                                                                      //
//  When your system uses shadow passwords you have to compile rootd    //
//  with -DR__SHADOWPW. Since shadow passwords can only be accessed     //
//  while being superuser (root) this works only when the server is     //
//  started via inetd. Another solution is to create a file             //
//  ~/.rootdpass containing an encrypted password. If this file exists  //
//  its password is used for authentication. This method overrides      //
//  all other authentication methods. To create an encrypted password   //
//  do something like:                                                  //
//     perl -e '$pw = crypt("<secretpasswd>","salt"); print "$pw\n"'    //
//  and store this string in ~/.rootdpass.                              //
//                                                                      //
//  To use AFS for authentication compile rootd with the -DR__AFS flag. //
//  In that case you also need to link with the AFS libraries. See      //
//  the Makefiles for more details.                                     //
//                                                                      //
//  To use Secure Remote Passwords (SRP) for authentication compile     //
//  rootd with the -DR__SRP flag. In that case you also need to link    //
//  with the SRP and gmp libraries. See the Makefile for more details.  //
//  SRP is described at: http://srp.stanford.edu/.                      //
//                                                                      //
//  See README.AUTH for more details on the authentication features.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// Protocol changes (see gProtocol):
// 2 -> 3: added handling of kROOTD_FSTAT message.
// 3 -> 4: added support for TFTP (i.e. kROOTD_PUTFILE, kROOTD_GETFILE, etc.)
// 4 -> 5: added support for "+read" to allow readers when file is opened for writing
// 5 -> 6: added support for kerberos5 authentication
// 6 -> 7: added support for kROOTD_BYE and kROOTD_PROTOCOL2
// 7 -> 8: added support for Globus, SSH and Rfio authentication and negotiation
// 8 -> 9: change in Kerberos authentication protocol

#include "config.h"

#include <ctype.h>
#include <fcntl.h>
#include <pwd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <errno.h>
#include <netdb.h>

#if defined(__CYGWIN__) && defined(__GNUC__)
#   define cygwingcc
#endif
#if defined(__alpha) && !defined(linux)
#   ifdef _XOPEN_SOURCE
#      if _XOPEN_SOURCE+0 > 0
#         define R__TRUE64
#      endif
#   endif
#include <sys/mount.h>
#ifndef R__TRUE64
extern "C" int fstatfs(int file_descriptor, struct statfs *buffer);
#endif
#elif defined(__APPLE__)
#include <sys/mount.h>
extern "C" int fstatfs(int file_descriptor, struct statfs *buffer);
#elif defined(linux) || defined(__hpux) || defined(cygwingcc)
#include <sys/vfs.h>
#elif defined(__FreeBSD__)
#include <sys/param.h>
#include <sys/mount.h>
#else
#include <sys/statfs.h>
#endif

#if defined(linux) || defined(__hpux) || defined(_AIX) || defined(__alpha) || \
    defined(__sun) || defined(__sgi) || defined(__FreeBSD__) || \
    defined(__APPLE__) || defined(cygwingcc)
#define HAVE_MMAP
#endif

#ifdef HAVE_MMAP
#   include <sys/mman.h>
#ifndef MAP_FILE
#define MAP_FILE 0           /* compatability flag */
#endif
#endif

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

#if defined(cygwingcc)
#define F_LOCK F_WRLCK
#define F_ULOCK F_UNLCK
static int fcntl_lockf(int fd, int op, off_t off)
{
   flock fl;
   fl.l_whence = SEEK_SET;
   fl.l_start  = off;
   fl.l_len    = 0;       // whole file
   fl.l_pid    = getpid();
   fl.l_type   = op;
   return fcntl(fd, F_SETLK, &fl);
}
#define lockf fcntl_lockf
#endif

#if defined(linux) || defined(__sun) || defined(__sgi) || \
    defined(_AIX) || defined(__FreeBSD__) || defined(__APPLE__) || \
    defined(__MACH__) || defined(cygwingcc)
#include <grp.h>
#include <sys/types.h>
#include <signal.h>
#endif

#if defined(__alpha) && !defined(linux) && !defined(__FreeBSD__)
extern "C" int initgroups(const char *name, int basegid);
#endif

#if defined(__sgi) && !defined(__GNUG__) && (SGI_REL<62)
extern "C" {
   int seteuid(int euid);
   int setegid(int egid);
}
#endif

#if defined(_AIX)
extern "C" {
   //int initgroups(const char *name, int basegid);
   int seteuid(uid_t euid);
   int setegid(gid_t egid);
}
#endif

#include "rootdp.h"

#ifdef R__KRB5
#include "Krb5Auth.h"
namespace ROOT {
   extern krb5_keytab  gKeytab;      // to allow specifying on the command line
   extern krb5_context gKcontext;
}
#endif

//--- Globals ------------------------------------------------------------------

enum { kBinary, kAscii };

int     gAuthListSent            = 0;
double  gBytesRead               = 0;
double  gBytesWritten            = 0;
char    gConfDir[kMAXPATHLEN]    = { 0 };    // Needed to localize root stuff if not running as root
int     gDebug                   = 0;
int     gDownloaded              = 0;
int     gFd                      = -1;
int     gForegroundFlag          = 0;
int     gFtp                     = 0;
int     gInetdFlag               = 0;
char    gOption[32]              = { 0 };
int     gPort1                   = 0;
int     gPort2                   = 0;
int     gProtocol                = 9;       // increase when protocol changes
int     gRootLog                 = 0;
char    gRpdAuthTab[kMAXPATHLEN] = { 0 };   // keeps track of authentication info
char    gRootdTab[kMAXPATHLEN]   = { 0 };   // keeps track of open files
int     gUploaded                = 0;
int     gWritable                = 0;
int     gReadOnly                = 0;

using namespace ROOT;

//--- Machine specific routines ------------------------------------------------

#if !defined(__hpux) && !defined(linux) && !defined(__FreeBSD__) || \
    defined(cygwingcc)
static int setresgid(gid_t r, gid_t e, gid_t)
{
   if (setgid(r) == -1)
      return -1;
   return setegid(e);
}

static int setresuid(uid_t r, uid_t e, uid_t)
{
   if (setuid(r) == -1)
      return -1;
   return seteuid(e);
}
#else
#if defined(linux) && !defined(HAS_SETRESUID)
extern "C" {
   int setresgid(gid_t r, gid_t e, gid_t s);
   int setresuid(uid_t r, uid_t e, uid_t s);
}
#endif
#endif


//--- Error handlers -----------------------------------------------------------

//______________________________________________________________________________
void Err(int level,const char *msg)
{
   Perror((char *)msg);
   if (level > -1) NetSendError((ERootdErrors)level);
}
//______________________________________________________________________________
void ErrFatal(int level,const char *msg)
{
   Perror((char *)msg);
   if (level > -1) NetSendError((ERootdErrors)level);
   RootdClose();
   exit(1);
}
//______________________________________________________________________________
void ErrSys(int level,const char *msg)
{
   Perror((char *)msg);
   ErrFatal(level,msg);
}

//--- Rootd routines -----------------------------------------------------------

const char *shellMeta   = "~*[]{}?$";
const char *shellStuff  = "(){}<>\"'";
const char  shellEscape = '\\';

//______________________________________________________________________________
static int EscChar(const char *src, char *dst, int dstlen, const char *specchars, char escchar)
{
   // Escape specchars in src with escchar and copy to dst.

   const char *p;
   char *q, *end = dst+dstlen-1;

   for (p = src, q = dst; *p && q < end; ) {
      if (strchr(specchars, *p)) {
         *q++ = escchar;
         if (q < end)
            *q++ = *p++;
      } else
         *q++ = *p++;
   }
   *q = '\0';

   if (*p != 0)
      return -1;
   return q-dst;
}

//______________________________________________________________________________
void SigPipe(int)
{
   // After SO_KEEPALIVE times out we probably get a SIGPIPE.

   ErrorInfo("SigPipe: rootd.cxx: got a SIGPIPE");

   // Treminate properly
   RootdAuthCleanup(0, 0);
   RootdClose();
   exit(1);
}

//______________________________________________________________________________
void RootdAuthCleanup(const char *sstr, int opt)
{
   // Terminate correctly by cleaning up the auth table (and shared memories
   // in case of Globus) and closing the file.
   // Called upon receipt of a kROOTD_CLOSE and on SIGPIPE.

   int rpid = 0;
   if (sstr) sscanf(sstr, "%d", &rpid);

   // Turn back to superuser for cleaning, if the case
   if (gRootLog == 0) {
     if (setresgid(0, 0, 0) == -1)
        if (gDebug > 0)
           ErrorInfo("RootdAuthCleanup: can't setgid to superuser");
     if (setresuid(0, 0, 0) == -1)
        if (gDebug > 0)
           ErrorInfo("RootdAuthCleanup: can't setuid to superuser");
   }
   if (opt == 0) {
      RpdCleanupAuthTab("all", 0);            // Cleanup everything (SIGPIPE)
      ErrorInfo("RootdAuthCleanup: cleanup ('all',0) done");
   } else if (opt == 1) {
      RpdCleanupAuthTab(gOpenHost, rpid);    // Cleanup only specific host (kROOTD_CLOSE)
      ErrorInfo("RootdAuthCleanup: cleanup ('%s',%d) done", gOpenHost, rpid);
   }
}

//______________________________________________________________________________
static const char *HomeDirectory(const char *name)
{
   // Returns the user's home directory.

   static char path[kMAXPATHLEN], mydir[kMAXPATHLEN];
   struct passwd *pw;

   if (name) {
      pw = getpwnam(name);
      if (pw) {
         strncpy(path, pw->pw_dir, kMAXPATHLEN);
         return path;
      }
   } else {
      if (mydir[0])
         return mydir;
      pw = getpwuid(getuid());
      if (pw) {
         strncpy(mydir, pw->pw_dir, kMAXPATHLEN);
         return mydir;
      }
   }
   return 0;
}

//______________________________________________________________________________
char *RootdExpandPathName(const char *name)
{
   // Expand a pathname getting rid of special shell characters like ~.$, etc.
   // Returned string must be freed by caller.

   const char *patbuf = name;

   // skip leading blanks
   while (*patbuf == ' ')
      patbuf++;

   // any shell meta characters?
   for (const char *p = patbuf; *p; p++)
      if (strchr(shellMeta, *p))
         goto needshell;

   return strdup(name);

needshell:
   // escape shell quote characters
   char escPatbuf[kMAXPATHLEN];
   EscChar(patbuf, escPatbuf, sizeof(escPatbuf), shellStuff, shellEscape);

   char cmd[kMAXPATHLEN];
#ifdef __hpux
   strcpy(cmd, "/bin/echo ");
#else
   strcpy(cmd, "echo ");
#endif

   // emulate csh -> popen executes sh
   if (escPatbuf[0] == '~') {
      const char *hd;
      if (escPatbuf[1] != '\0' && escPatbuf[1] != '/') {
         // extract user name
         char uname[70], *p, *q;
         for (p = &escPatbuf[1], q = uname; *p && *p !='/';)
            *q++ = *p++;
         *q = '\0';
         hd = HomeDirectory(uname);
         if (hd == 0)
            strcat(cmd, escPatbuf);
         else {
            strcat(cmd, hd);
            strcat(cmd, p);
         }

    } else {
         hd = HomeDirectory(0);
         if (hd == 0) {
            Error(ErrSys, kErrFatal, "RootdExpandPathName: no home directory");
            return 0;
         }
         strcat(cmd, hd);
         strcat(cmd, &escPatbuf[1]);
      }
   } else
      strcat(cmd, escPatbuf);

   FILE *pf;
   if ((pf = ::popen(&cmd[0], "r")) == 0) {
      Error(ErrSys, kErrFatal, "RootdExpandPathName: error in popen(%s)", cmd);
      return 0;
   }

   // read first argument
   char expPatbuf[kMAXPATHLEN];
   int  ch, i, cnt = 0;
again:
   for (i = 0, ch = fgetc(pf); ch != EOF && ch != ' ' && ch != '\n'; i++, ch = fgetc(pf)) {
      expPatbuf[i] = ch;
      cnt++;
   }
   // this will be true if forked process was not yet ready to be read
   if (cnt == 0 && ch == EOF) goto again;
   expPatbuf[cnt] = '\0';

   // skip rest of pipe
   while (ch != EOF) {
      ch = fgetc(pf);
      if (ch == ' ' || ch == '\t') {
         ::pclose(pf);
         Error(ErrFatal, kErrFatal, "RootdExpandPathName: expression ambigous");
         return 0;
      }
   }

   ::pclose(pf);

   return strdup(expPatbuf);
}

//______________________________________________________________________________
int RootdCheckTab(int mode)
{
   // Checks gRootdTab file to see if file can be opened. If mode = 1 then
   // check if file can safely be opened in write mode, i.e. see if file
   // is not already opened in either read or write mode. If mode = 0 then
   // check if file can safely be opened in read mode, i.e. see if file
   // is not already opened in write mode. If mode = -1 check write mode
   // like 1 but do not update rootdtab file. Returns 1 if file can be
   // opened safely, otherwise 0.
   //
   // The format of the file is:
   // filename inode mode username pid
   // where inode is the unique file ref number, mode is either "read"
   // or "write", username the user who has the file open and pid is the
   // pid of the rootd having the file open.

   // Open rootdtab file. Try first /usr/tmp and then /tmp.
   // The lockf() call can fail if the directory is NFS mounted
   // and the lockd daemon is not running.

   const char *sfile = gRootdTab;
   int fid, create = 0;

   int noupdate = 0;
   if (mode < 0) {
      mode = 1;
      noupdate = 1;
   }

again:
   if (access(sfile, F_OK) == -1) {
      fid = open(sfile, O_CREAT|O_RDWR, 0644);
      if (fid != -1) fchmod(fid, 0666);    // override umask setting
      create = 1;
   } else
      fid = open(sfile, O_RDWR);

   if (fid == -1) {
      if (sfile[1] == 'u') {
         sfile = gRootdTab+4;
         goto again;
      }
      Error(ErrSys, kErrFatal, "RootdCheckTab: error opening %s", sfile);
   }

   // lock the file
   if (lockf(fid, F_LOCK, (off_t)1) == -1) {
      if (sfile[1] == 'u' && create) {
         close(fid);
         remove(sfile);
         sfile = gRootdTab+4;
         goto again;
      }
      Error(ErrSys, kErrFatal, "RootdCheckTab: error locking %s", sfile);
   }
   if (gDebug > 2)
      ErrorInfo("RootdCheckTab: file %s locked", sfile);

   struct stat sbuf;
   fstat(fid, &sbuf);
   size_t siz = sbuf.st_size;

   ino_t inode;
   if (stat(gFile, &sbuf) == -1)
      inode = 0;
   else
      inode = sbuf.st_ino;

   char msg[kMAXPATHLEN];
   const char *smode = (mode == 1) ? "write" : "read";
   int result = 1;

   if (siz > 0) {
      int changed = 0;
      char *fbuf = new char[siz+1];
      char *flast = fbuf + siz;

      while (read(fid, fbuf, siz) < 0 && GetErrno() == EINTR)
         ResetErrno();
      fbuf[siz] = 0;

      char *n, *s = fbuf;
      while ((n = strchr(s, '\n')) && siz > 0) {
         n++;
         char user[64], gmode[32];
         int  pid;
         unsigned long ino;
         sscanf(s, "%s %lu %s %s %d", msg, &ino, gmode, user, &pid);
         if (kill(pid, 0) == -1 && GetErrno() == ESRCH) {
            ErrorInfo("RootdCheckTab: remove stale lock (%s %u %s %s %d)\n", msg, ino, gmode, user, pid);
            if (n >= flast) {
               siz = int(s - fbuf);
               changed = 1;
               break;
            } else {
               int l = int(flast - n) + 1;
               memmove(s, n, l);
               siz -= int(n - s);
               n = s;
            }
            flast = fbuf + siz;
            changed = 1;
         } else if (ino == inode) {
            if (mode == 1)
               result = 0;
            else if (!strcmp(gmode, "write"))
               result = 0;
         }
         s = n;
      }
      if (changed) {
         ftruncate(fid, 0);
         lseek(fid, 0, SEEK_SET);
         if (siz > 0) {
            while (write(fid, fbuf, siz) < 0 && GetErrno() == EINTR)
               ResetErrno();
         }
      }
      delete [] fbuf;
   }

   if (result && !noupdate) {
      unsigned long ino = inode;
      sprintf(msg, "%s %lu %s %s %d\n", gFile, ino, smode, gUser, (int) getpid());
      write(fid, msg, strlen(msg));
   }

   // unlock the file
   lseek(fid, 0, SEEK_SET);
   if (lockf(fid, F_ULOCK, (off_t)1) == -1)
      Error(ErrSys, kErrFatal, "RootdCheckTab: error unlocking %s", sfile);
   if (gDebug > 2)
      ErrorInfo("RootdCheckTab: file %s unlocked", sfile);

   close(fid);

   return result;
}

//______________________________________________________________________________
void RootdCloseTab(int force = 0)
{
   // Removes from the gRootdTab file the reference to gFile for the
   // current rootd. If force = 1, then remove all references for gFile
   // from the gRootdTab file. This might be necessary in case something
   // funny happened and the original reference was not correctly removed.
   // Stale locks are detected by checking each pid and then removed.

   const char *sfile = gRootdTab;
   int fid;

again:
   if (access(sfile, F_OK) == -1) {
      if (sfile[1] == 'u') {
         sfile = gRootdTab+4;
         goto again;
      }
      ErrorInfo("RootdCloseTab: file %s does not exist", sfile);
      return;
   }

   fid = open(sfile, O_RDWR);

   if (fid == -1) {
      ErrorInfo("RootdCloseTab: error opening %s", sfile);
      return;
   }

   // lock the file
   if (lockf(fid, F_LOCK, (off_t)1) == -1) {
      ErrorInfo("RootdCloseTab: error locking %s", sfile);
      return;
   }
   if (gDebug > 2)
      ErrorInfo("RootdCloseTab: file %s locked", sfile);

   struct stat sbuf;
   fstat(fid, &sbuf);
   size_t siz = sbuf.st_size;

   stat(gFile, &sbuf);
   ino_t inode = sbuf.st_ino;

   if (siz > 0) {
      int changed = 0;
      int mypid   = getpid();
      char *fbuf  = new char[siz+1];
      char *flast = fbuf + siz;

      while (read(fid, fbuf, siz) < 0 && GetErrno() == EINTR)
         ResetErrno();
      fbuf[siz] = 0;

      char *n, *s = fbuf;
      while ((n = strchr(s, '\n')) && siz > 0) {
         n++;
         char msg[kMAXPATHLEN], user[64], gmode[32];
         int  pid, stale = 0;
         unsigned int ino;
         sscanf(s, "%s %u %s %s %d", msg, &ino, gmode, user, &pid);
         if (kill(pid, 0) == -1 && GetErrno() == ESRCH) {
            stale = 1;
            ErrorInfo("Remove Stale Lock (%s %u %s %s %d)\n", msg, ino, gmode, user, pid);
         }
         if (stale || (!force && mypid == pid) ||
             (force && inode == ino && !strcmp(gUser, user))) {
            if (n >= flast) {
               siz = int(s - fbuf);
               changed = 1;
               break;
            } else {
               int l = int(flast - n) + 1;
               memmove(s, n, l);
               siz -= int(n - s);
               n = s;
            }
            flast = fbuf + siz;
            changed = 1;
         }
         s = n;
      }
      if (changed) {
         ftruncate(fid, 0);
         lseek(fid, 0, SEEK_SET);
         if (siz > 0) {
            while (write(fid, fbuf, siz) < 0 && GetErrno() == EINTR)
               ResetErrno();
         }
      }
      delete [] fbuf;
   }

   // unlock the file
   lseek(fid, 0, SEEK_SET);
   if (lockf(fid, F_ULOCK, (off_t)1) == -1) {
      ErrorInfo("RootdCloseTab: error unlocking %s", sfile);
      return;
   }
   if (gDebug > 2)
      ErrorInfo("RootdCloseTab: file %s unlocked", sfile);

   close(fid);
}

//______________________________________________________________________________
int RootdIsOpen()
{
   if (gFd == -1) return 0;
   return 1;
}

//______________________________________________________________________________
void RootdCloseFtp()
{
   if (gDebug > 0)
      ErrorInfo("RootdCloseFtp: %d files uploaded, %d files downloaded, rd=%g, wr=%g, rx=%g, tx=%g",
                gUploaded, gDownloaded, gBytesRead, gBytesWritten, gBytesRecv, gBytesSent);
   else
      ErrorInfo("Rootd: %d files uploaded, %d files downloaded, rd=%g, wr=%g, rx=%g, tx=%g",
                gUploaded, gDownloaded, gBytesRead, gBytesWritten, gBytesRecv, gBytesSent);
}

//______________________________________________________________________________
void RootdClose()
{
   if (gFtp) {
      RootdCloseFtp();
      return;
   }

   if (RootdIsOpen()) {
      close(gFd);
      gFd = -1;
   }

   RootdCloseTab();

   if (gDebug > 0)
      ErrorInfo("RootdClose: file %s closed, rd=%g, wr=%g, rx=%g, tx=%g",
                gFile, gBytesRead, gBytesWritten, gBytesRecv, gBytesSent);
   else
      ErrorInfo("Rootd: file %s closed, rd=%g, wr=%g, rx=%g, tx=%g", gFile,
                gBytesRead, gBytesWritten, gBytesRecv, gBytesSent);
}

//______________________________________________________________________________
void RootdFlush()
{
   if (RootdIsOpen() && gWritable) {
#ifndef WIN32
      if (fsync(gFd) < 0)
         Error(ErrSys, kErrFatal, "RootdFlush: error flushing file %s", gFile);
#endif
   }

   if (gDebug > 0)
      ErrorInfo("RootdFlush: file %s flushed", gFile);
}

//______________________________________________________________________________
void RootdStat()
{

}

//______________________________________________________________________________
void RootdFstat()
{
   // Return file stat information in same format as TSystem::GetPathInfo().

   char     msg[128];
   long     id, flags, modtime;
   Long64_t size;

#if defined(R__SEEK64)
   struct stat64 statbuf;
   if (RootdIsOpen() && fstat64(gFd, &statbuf) >= 0) {
#elif defined(WIN32)
   struct _stati64 statbuf;
   if (RootdIsOpen() && _fstati64(gFd, &statbuf) >= 0) {
#else
   struct stat statbuf;
   if (RootdIsOpen() && fstat(gFd, &statbuf) >= 0) {
#endif
#if defined(__KCC) && defined(linux)
      id = (statbuf.st_dev.__val[0] << 24) + statbuf.st_ino;
#else
      id = (statbuf.st_dev << 24) + statbuf.st_ino;
#endif
      size = statbuf.st_size;
      modtime = statbuf.st_mtime;
      flags = 0;
      if (statbuf.st_mode & ((S_IEXEC)|(S_IEXEC>>3)|(S_IEXEC>>6)))
         flags |= 1;
      if ((statbuf.st_mode & S_IFMT) == S_IFDIR)
         flags |= 2;
      if ((statbuf.st_mode & S_IFMT) != S_IFREG &&
          (statbuf.st_mode & S_IFMT) != S_IFDIR)
         flags |= 4;
      sprintf(msg, "%ld %lld %ld %ld", id, size, flags, modtime);
   } else
      sprintf(msg, "-1 -1 -1 -1");
   NetSend(msg, kROOTD_FSTAT);
}

//______________________________________________________________________________
void RootdProtocol()
{
   // Return rootd protocol.

   // all old client protocols, before intro of kROOTD_PROTOCOL2
   gClientProtocol = 6;

   NetSend(gProtocol, kROOTD_PROTOCOL);

   if (gDebug > 0)
      ErrorInfo("RootdProtocol: gClientProtocol = %d", gClientProtocol);
}

//______________________________________________________________________________
void RootdProtocol2(const char *proto)
{
   // Receives client protocol and returns rootd protocol.

   sscanf(proto, "%d", &gClientProtocol);

   NetSend(gProtocol, kROOTD_PROTOCOL);

   if (gDebug > 0)
      ErrorInfo("RootdProtocol: gClientProtocol = %d", gClientProtocol);
}

//______________________________________________________________________________
void RootdLogin()
{
   // Authentication was successful, set user environment.

   struct passwd *pw = getpwnam(gUser);
   if (gDebug > 2)
      ErrorInfo("RootdLogin: login dir: %s (uid: %d)", pw->pw_dir, getuid());

   if (chdir(pw->pw_dir) == -1) {
      ErrorInfo("RootdLogin: can't change directory to %s",pw->pw_dir);
      return;
   }

   if (gDebug > 2)
      ErrorInfo("RootdLogin: gid: %d, uid: %d", pw->pw_gid, pw->pw_uid);

   if (getuid() == 0) {

      if (gAnon && chroot(pw->pw_dir) == -1) {
         ErrorInfo("RootdLogin: can't chroot to %s", pw->pw_dir);
         return;
      }

      // set access control list from /etc/initgroup
      initgroups(gUser, pw->pw_gid);

      // set gid
      if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1) {
         ErrorInfo("RootdLogin: can't setgid for user %s", gUser);
         return;
      }
      // set uid
      if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1) {
         ErrorInfo("RootdLogin: can't setuid for user %s", gUser);
         return;
      }
   }

   umask(022);

   // Notify authentication to client ...
   NetSend(gAuth, kROOTD_AUTH);

   // Send also new offset if it changed ...
   if (gAuth == 2) NetSend(gOffSet, kROOTD_AUTH);

   if (gDebug > 0) {
      if (gAnon)
         ErrorInfo("RootdLogin: user %s/%s authenticated (OffSet: %d)", gUser, gPasswd, gOffSet);
      else
         ErrorInfo("RootdLogin: user %s authenticated (OffSet: %d)", gUser, gOffSet);
   }
}

//______________________________________________________________________________
void RootdPass(const char *pass)
{
   // Check user's password.

   // Reset global variable
   gAuth = 0;

   // Evaluate credentials ...
   RpdPass(pass);

   // Login, if ok ...
   if (gAuth == 1) RootdLogin();
}

//______________________________________________________________________________
void RootdUser(const char *sstr)
{
   // Check user's UID.

   // Reset global variable
   gAuth = 0;

   // Evaluate credentials ...
   RpdUser(sstr);

   // Login, if ok ...
   if (gAuth == 1) RootdLogin();
}

//______________________________________________________________________________
void RootdRfioAuth(const char *sstr)
{
   // Authenticate via UID/GID (Rfio).

   // Reset global variable
   gAuth = 0;

   // Evaluate credentials ...
   RpdRfioAuth(sstr);

   // ... and login
   if (gAuth == 1) RootdLogin();
}

//______________________________________________________________________________
void RootdKrb5Auth(const char *sstr)
{
   // Authenticate via Kerberos.

   // Reset global variable
   gAuth = 0;

   // Evaluate credentials ...
   RpdKrb5Auth(sstr);

   // Login, if ok ...
   if (gAuth == 1) RootdLogin();
}

//______________________________________________________________________________
void RootdSRPUser(const char *user)
{
   // Use Secure Remote Password protocol.
   // Check user id in $HOME/.srootdpass file.

   // Reset global variable
   gAuth = 0;

   // Evaluate credentials ...
   RpdSRPUser(user);

   // Login, if ok ...
   if (gAuth == 1) RootdLogin();
}

//______________________________________________________________________________
void RootdGlobusAuth(const char *sstr)
{
   // Authenticate via Globus.

   // Reset global variable
   gAuth = 0;

   // Evaluate credentials ...
   RpdGlobusAuth(sstr);

   // Login, if ok ...
   if (gAuth == 1) RootdLogin();
}

//______________________________________________________________________________
void RootdSshAuth(const char *sstr)
{
   // Authenticate via SSH.

   // Reset global variable
   gAuth = 0;

   // Evaluate credentials ...
   RpdSshAuth(sstr);

   // Login, if ok ...
   if (gAuth == 1) RootdLogin();
}

//______________________________________________________________________________
static int SysOpen(const char *pathname, int flags, unsigned int mode)
{
   // System independent open().

#if defined(R__WINGCC)
   // ALWAYS use binary mode - even cygwin text should be in unix format
   // although this is posix default it has to be set explicitly
   return ::open(pathname, flags | O_BINARY, mode);
#elif defined(R__SEEK64)
   return ::open64(pathname, flags, mode);
#else
   return ::open(pathname, flags, mode);
#endif
}

//______________________________________________________________________________
void RootdOpen(const char *msg)
{
   // Open file in mode depending on specified option. If file is already
   // opened by another rootd in write mode, do not open the file.

   char file[kMAXPATHLEN], option[32];

   gBytesRead = gBytesWritten = gBytesRecv = gBytesSent = 0;

   sscanf(msg, "%s %s", file, option);

   if (file[0] == '/')
      strcpy(gFile, &file[1]);
   else
      strcpy(gFile, file);

   gFile[strlen(file)] = '\0';

   strcpy(gOption, option);

   int forceOpen = 0;
   if (option[0] == 'f') {
      forceOpen = 1;
      strcpy(gOption, &option[1]);
   }

   int forceRead = 0;
   if (!strcmp(option, "+read")) {
      forceRead = 1;
      strcpy(gOption, &option[1]);
   }

   int create = 0;
   if (!strcmp(gOption, "new") || !strcmp(gOption, "create"))
      create = 1;
   int recreate = strcmp(gOption, "recreate") ? 0 : 1;
   int update   = strcmp(gOption, "update")   ? 0 : 1;
   int read     = strcmp(gOption, "read")     ? 0 : 1;
   if (!create && !recreate && !update && !read) {
      read = 1;
      strcpy(gOption, "read");
   }

   if (!read && gReadOnly)
      Error(ErrFatal, kErrNoAccess, "RootdOpen: file %s can only be opened in \"READ\" mode", gFile);

   if (!gAnon) {
      char *fname;
      if ((fname = RootdExpandPathName(gFile))) {
         strcpy(gFile, fname);
         free(fname);
      } else
         Error(ErrFatal, kErrBadFile, "RootdOpen: bad file name %s", gFile);
   }

   if (forceOpen)
      RootdCloseTab(1);

   int trunc = 0;
   if (recreate) {
      if (!RootdCheckTab(-1))
         Error(ErrFatal, kErrFileWriteOpen, "RootdOpen: file %s already opened in read or write mode", gFile);
      if (!access(gFile, F_OK))
         trunc = O_TRUNC;
      else {
         recreate = 0;
         create   = 1;
         strcpy(gOption, "create");
      }
   }

   if (create && !access(gFile, F_OK))
      Error(ErrFatal, kErrFileExists, "RootdOpen: file %s already exists", gFile);

   if (update) {
      if (access(gFile, F_OK)) {
         update = 0;
         create = 1;
         strcpy(gOption, "create");
      }
      if (update && access(gFile, W_OK))
         Error(ErrFatal, kErrNoAccess, "RootdOpen: no write permission for file %s", gFile);
   }

   if (read) {
      if (access(gFile, F_OK))
         Error(ErrFatal, kErrNoFile, "RootdOpen: file %s does not exist (errno: 0x%x)", gFile, errno);
      if (access(gFile, R_OK))
         Error(ErrFatal, kErrNoAccess, "RootdOpen: no read permission for file %s (errno: 0x%x)", gFile, errno);
   }

   if (create || recreate || update) {
      if (create || recreate) {
         // make sure file exists so RootdCheckTab works correctly
#ifndef WIN32
         gFd = SysOpen(gFile, O_RDWR | O_CREAT | trunc, 0644);
#else
         gFd = SysOpen(gFile, O_RDWR | O_CREAT | O_BINARY | trunc, S_IREAD | S_IWRITE);
#endif
         close(gFd);
         gFd = -1;
      }
#ifndef WIN32
      gFd = SysOpen(gFile, O_RDWR, 0644);
#else
      gFd = SysOpen(gFile, O_RDWR | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (gFd == -1)
         Error(ErrSys, kErrFileOpen, "RootdOpen: error opening file %s in write mode", gFile);

      if (!RootdCheckTab(1)) {
         close(gFd);
         Error(ErrFatal, kErrFileWriteOpen, "RootdOpen: file %s already opened in read or write mode", gFile);
      }

      gWritable = 1;

   } else {
#ifndef WIN32
      gFd = SysOpen(gFile, O_RDONLY, 0644);
#else
      gFd = SysOpen(gFile, O_RDONLY | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (gFd == -1)
         Error(ErrSys, kErrFileOpen, "RootdOpen: error opening file %s in read mode", gFile);

      if (!RootdCheckTab(0)) {
         if (!forceRead) {
            close(gFd);
            Error(ErrFatal, kErrFileReadOpen, "RootdOpen: file %s already opened in write mode", gFile);
         }
      }

      gWritable = 0;

   }

   NetSend(gWritable, kROOTD_OPEN);

   struct stat sbuf;
   fstat(gFd, &sbuf);
   unsigned long dev = (unsigned long) sbuf.st_dev;
   unsigned long ino = (unsigned long) sbuf.st_ino;

   if (gDebug > 0)
      ErrorInfo("RootdOpen: file %s opened in mode %s", gFile, gOption);
   else {
      if (gAnon)
         ErrorInfo("RootdOpen: file %s (dev=%lu,inode=%lu,%s) opened by %s/%s",
                   gFile, dev, ino, gOption, gUser, gPasswd);
      else
         ErrorInfo("RootdOpen: file %s (dev=%lu,inode=%lu,%s) opened by %s",
                   gFile, dev, ino, gOption, gUser);
   }
}

//______________________________________________________________________________
void RootdPut(const char *msg)
{
   // Receive a buffer and write it at the specified offset in the currently
   // open file.

   Long64_t offset;
   int      len;

   sscanf(msg, "%lld %d", &offset, &len);

   char *buf = new char[len];
   NetRecvRaw(buf, len);

   if (!RootdIsOpen() || !gWritable)
      Error(ErrFatal, kErrNoAccess, "RootdPut: file %s not opened in write mode", gFile);

#if defined (R__SEEK64)
   if (lseek64(gFd, offset, SEEK_SET) < 0)
#elif defined(WIN32)
   if (_lseeki64(gFd, offset, SEEK_SET) < 0)
#else
   if (lseek(gFd, offset, SEEK_SET) < 0)
#endif
      Error(ErrSys, kErrFilePut, "RootdPut: cannot seek to position %lld in file %s", offset, gFile);

   ssize_t siz;
   while ((siz = write(gFd, buf, len)) < 0 && GetErrno() == EINTR)
      ResetErrno();

   if (siz < 0)
      Error(ErrSys, kErrFilePut, "RootdPut: error writing to file %s", gFile);

   if (siz != len)
      Error(ErrFatal, kErrFilePut, "RootdPut: error writing all requested bytes to file %s, wrote %d of %d",
            gFile, siz, len);

   NetSend(0, kROOTD_PUT);

   delete [] buf;

   gBytesWritten += len;

   if (gDebug > 0)
      ErrorInfo("RootdPut: written %d bytes starting at %lld to file %s",
                len, offset, gFile);
}

//______________________________________________________________________________
void RootdGet(const char *msg)
{
   // Get a buffer from the specified offset from the currently open file
   // and send it to the client.

   Long64_t offset;
   int      len;

   sscanf(msg, "%lld %d", &offset, &len);

   char *buf = new char[len];

   if (!RootdIsOpen())
      Error(ErrFatal, kErrNoAccess, "RootdGet: file %s not open", gFile);

#if defined (R__SEEK64)
   if (lseek64(gFd, offset, SEEK_SET) < 0)
#elif defined(WIN32)
   if (_lseeki64(gFd, offset, SEEK_SET) < 0)
#else
   if (lseek(gFd, offset, SEEK_SET) < 0)
#endif
      Error(ErrSys, kErrFileGet, "RootdGet: cannot seek to position %lld in file %s", offset, gFile);

   ssize_t siz;
   while ((siz = read(gFd, buf, len)) < 0 && GetErrno() == EINTR)
      ResetErrno();

   if (siz < 0)
      Error(ErrSys, kErrFileGet, "RootdGet: error reading from file %s", gFile);

   if (siz != len)
      Error(ErrFatal, kErrFileGet, "RootdGet: error reading all requested bytes from file %s, got %d of %d",
                 gFile, siz, len);

   NetSend(0, kROOTD_GET);

   NetSendRaw(buf, len);

   delete [] buf;

   gBytesRead += len;

   if (gDebug > 0)
      ErrorInfo("RootdGet: read %d bytes starting at %lld from file %s",
                len, offset, gFile);
}

//______________________________________________________________________________
void RootdPutFile(const char *msg)
{
   // Receive a file from the remote client (upload).

   char     file[kMAXPATHLEN];
   Long64_t size, restartat;
   int      blocksize, mode, forceopen = 0;

   gFtp = 1;   // rootd is used for ftp instead of file serving

   sscanf(msg, "%s %d %d %lld %lld", file, &blocksize, &mode, &size, &restartat);

   if (file[0] == '-') {
      forceopen = 1;
      strcpy(gFile, file+1);
   } else
      strcpy(gFile, file);

   // anon user may not overwrite existing files...
   struct stat st;
   if (!stat(gFile, &st)) {
      if (gAnon) {
         Error(Err, kErrFileExists, "RootdPutFile: anonymous users may not overwrite existing file %s", gFile);
         return;
      }
   } else if (GetErrno() != ENOENT) {
      Error(Err, kErrFatal, "RootdPutFile: can't check for file presence");
      return;
   }

   // remove lock from file
   if (restartat || forceopen)
      RootdCloseTab(1);

   // open local file
   int fd;
   if (!restartat) {

      // make sure file exists so RootdCheckTab works correctly
#ifndef WIN32
      fd = SysOpen(gFile, O_RDWR | O_CREAT, 0600);
#else
      fd = SysOpen(gFile, O_RDWR | O_CREAT | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fd < 0) {
         Error(Err, kErrFileOpen, "RootdPutFile: cannot open file %s", gFile);
         return;
      }

      close(fd);

      // check if file is not in use by somebody and prevent from somebody
      // using it before upload is completed
      if (!RootdCheckTab(1)) {
         Error(Err, kErrFileWriteOpen, "RootdPutFile: file %s already opened in read or write mode", gFile);
         return;
      }

#ifndef WIN32
      fd = SysOpen(gFile, O_CREAT | O_TRUNC | O_WRONLY, 0600);
#else
      if (mode == kBinary)
         fd = SysOpen(gFile, O_CREAT | O_TRUNC | O_WRONLY | O_BINARY,
                      S_IREAD | S_IWRITE);
      else
         fd = SysOpen(gFile, O_CREAT | O_TRUNC | O_WRONLY,
                      S_IREAD | S_IWRITE);
#endif
   } else {
#ifndef WIN32
      fd = SysOpen(gFile, O_WRONLY, 0600);
#else
      if (mode == kBinary)
         fd = SysOpen(gFile, O_WRONLY | O_BINARY, S_IREAD | S_IWRITE);
      else
         fd = SysOpen(gFile, O_WRONLY, S_IREAD | S_IWRITE);
#endif
      if (fd < 0) {
         Error(Err, kErrFileOpen, "RootdPutFile: cannot open file %s", gFile);
         return;
      }
      if (!RootdCheckTab(1)) {
         close(fd);
         Error(Err, kErrFileWriteOpen, "RootdPutFile: file %s already opened in read or write mode", gFile);
         return;
      }
   }

   // check file system space
   if (strcmp(gFile, "/dev/null")) {
      struct statfs statfsbuf;
#if defined(__sgi) || (defined(__sun) && !defined(linux))
      if (fstatfs(fd, &statfsbuf, sizeof(struct statfs), 0) == 0) {
         Long64_t space = (Long64_t)statfsbuf.f_bsize * (Long64_t))statfsbuf.f_bfree;
#else
      if (fstatfs(fd, &statfsbuf) == 0) {
         Long64_t space = (Long64_t)statfsbuf.f_bsize * (Long64_t)statfsbuf.f_bavail;
#endif
         if (space < size - restartat) {
            Error(Err, kErrNoSpace, "RootdPutFile: not enough space to store file %s", gFile);
            close(fd);
            return;
         }
      }
   }

   // seek to restartat position
   if (restartat) {
#if defined(R__SEEK64)
      if (lseek64(fd, restartat, SEEK_SET) < 0) {
#elif defined(WIN32)
      if (_lseeki64(fd, restartat, SEEK_SET) < 0) {
#else
      if (lseek(fd, restartat, SEEK_SET) < 0) {
#endif
         Error(Err, kErrRestartSeek, "RootdPutFile: cannot seek to position %lld in file %s",
               restartat, gFile);
         close(fd);
         return;
      }
   }

   // setup ok
   NetSend(0, kROOTD_PUTFILE);

   struct timeval started, ended;
   gettimeofday(&started, 0);

   char *buf = new char[blocksize];
   char *buf2 = 0;
   if (mode == 1)
      buf2 = new char[blocksize];

   Long64_t pos = restartat & ~(blocksize-1);
   int skip = restartat - pos;

   while (pos < size) {
      Long64_t left = Long64_t(size - pos);
      if (left > blocksize)
         left = blocksize;

      NetRecvRaw(buf, int(left-skip));

      int n = int(left-skip);

      // in case of ascii file, loop here over buffer and remove \r's
      ssize_t siz;
      if (mode == kAscii) {
         int i = 0, j = 0;
         while (i < n) {
            if (buf[i] == '\r')
               i++;
            else
               buf2[j++] = buf[i++];
         }
         n = j;
         while ((siz = write(fd, buf2, n)) < 0 && GetErrno() == EINTR)
            ResetErrno();
      } else {
         while ((siz = write(fd, buf, n)) < 0 && GetErrno() == EINTR)
            ResetErrno();
      }

      if (siz < 0)
         Error(ErrSys, kErrFilePut, "RootdPutFile: error writing to file %s", gFile);

      if (siz != n)
         Error(ErrFatal, kErrFilePut, "RootdPutFile: error writing all requested bytes to file %s, wrote %d of %d",
               gFile, siz, int(left-skip));

      gBytesWritten += n;

      pos += left;
      skip = 0;
   }

   gettimeofday(&ended, 0);

   // file stored ok
   NetSend(0, kROOTD_PUTFILE);

   delete [] buf; delete [] buf2;

   fchmod(fd, 0644);

   close(fd);

   RootdCloseTab();

   gUploaded++;

   double speed, t;
   t = (ended.tv_sec + ended.tv_usec / 1000000.0) -
       (started.tv_sec + started.tv_usec / 1000000.0);
   if (t > 0)
      speed = double(size - restartat) / t;
   else
      speed = 0.0;
   if (speed > 524288)
      ErrorInfo("RootdPutFile: uploaded file %s (%lld bytes, %.3f seconds, "
                "%.2f Mbytes/s)", gFile, size, t, speed / 1048576);
   else if (speed > 512)
      ErrorInfo("RootdPutFile: uploaded file %s (%lld bytes, %.3f seconds, "
                "%.2f Kbytes/s)", gFile, size, t, speed / 1024);
   else
      ErrorInfo("RootdPutFile: uploaded file %s (%lld bytes, %.3f seconds, "
                "%.2f bytes/s)", gFile, size, t, speed);
}

//______________________________________________________________________________
void RootdGetFile(const char *msg)
{
   // Send a file to a remote client (download).

   char     file[kMAXPATHLEN];
   Long64_t restartat;
   int      blocksize, mode, forceopen = 0;

   gFtp = 1;   // rootd is used for ftp instead of file serving

   sscanf(msg, "%s %d %d %lld", file, &blocksize, &mode, &restartat);

   if (file[0] == '-') {
      forceopen = 1;
      strcpy(gFile, file+1);
   } else
      strcpy(gFile, file);

   // remove lock from file
   if (forceopen)
      RootdCloseTab(1);

   // open file for reading
#if defined(WIN32) || defined(R__WINGCC)
   int fd = SysOpen(gFile, O_RDONLY | O_BINARY, S_IREAD | S_IWRITE);
#else
   int fd = SysOpen(gFile, O_RDONLY, 0600);
#endif
   if (fd < 0) {
      Error(Err, kErrFileOpen, "RootdGetFile: cannot open file %s", gFile);
      return;
   }

   // check if file is not in use by somebody and prevent from somebody
   // using it before download is completed
   if (!RootdCheckTab(0)) {
      close(fd);
      Error(Err, kErrFileOpen, "RootdGetFile: file %s is already open in write mode", gFile);
      return;
   }

#if defined(R__SEEK64)
   struct stat64 st;
   if (fstat64(fd, &st)) {
#elif defined(WIN32)
   struct _stati64 st;
   if (_fstati64(fd, &st)) {
#else
   struct stat st;
   if (fstat(fd, &st)) {
#endif
      Error(Err, kErrFatal, "RootdGetFile: cannot get size of file %s", gFile);
      close(fd);
      return;
   }
   Long64_t size = st.st_size;

   if (!S_ISREG(st.st_mode)) {
      Error(Err, kErrBadFile, "RoodGetFile: not a regular file %s", gFile);
      close(fd);
      return;
   }

   // check if restartat value makes sense
   if (restartat && (restartat >= size))
      restartat = 0;

   // setup ok
   NetSend(0, kROOTD_GETFILE);

   char mess[128];
   sprintf(mess, "%lld", size);
   NetSend(mess, kROOTD_GETFILE);

   struct timeval started, ended;
   gettimeofday(&started, 0);

   Long64_t pos  = restartat & ~(blocksize-1);
   int  skip = int(restartat - pos);

#ifndef HAVE_MMAP
   char *buf = new char[blocksize];
#if defined(R__SEEK64)
   lseek64(fd, pos, SEEK_SET);
#elif defined(WIN32)
   _lseeki64(fd, pos, SEEK_SET);
#else
   lseek(fd, pos, SEEK_SET);
#endif
#endif

   while (pos < size) {
      Long64_t left = size - pos;
      if (left > blocksize)
         left = blocksize;
#ifdef HAVE_MMAP
#if defined(R__SEEK64)
      char *buf = (char*) mmap64(0, left, PROT_READ, MAP_FILE | MAP_SHARED, fd, pos);
#else
      char *buf = (char*) mmap(0, left, PROT_READ, MAP_FILE | MAP_SHARED, fd, pos);
#endif
      if (buf == (char *) -1)
         Error(ErrFatal, kErrFileGet, "RootdGetFile: mmap of file %s failed", gFile);
#else
      int siz;
      while ((siz = read(fd, buf, (int)left)) < 0 && GetErrno() == EINTR)
         ResetErrno();
      if (siz < 0 || siz != left)
         Error(ErrFatal, kErrFileGet, "RootdGetFile: error reading from file %s", gFile);
#endif

      NetSendRaw(buf+skip, int(left-skip));

      gBytesRead += left-skip;

      pos += left;
      skip = 0;

#ifdef HAVE_MMAP
      munmap(buf, left);
#endif
   }

   gettimeofday(&ended, 0);

#ifndef HAVE_MMAP
   delete [] buf;
#endif

   close(fd);

   RootdCloseTab();

   gDownloaded++;

   double speed, t;
   t = (ended.tv_sec + ended.tv_usec / 1000000.0) -
       (started.tv_sec + started.tv_usec / 1000000.0);
   if (t > 0)
      speed = double(size - restartat) / t;
   else
      speed = 0.0;
   if (speed > 524288)
      ErrorInfo("RootdGetFile: downloaded file %s (%lld bytes, %.3f seconds, "
                "%.2f Mbytes/s)", gFile, size, t, speed / 1048576);
   else if (speed > 512)
      ErrorInfo("RootdGetFile: downloaded file %s (%lld bytes, %.3f seconds, "
                "%.2f Kbytes/s)", gFile, size, t, speed / 1024);
   else
      ErrorInfo("RootdGetFile: downloaded file %s (%lld bytes, %.3f seconds, "
                "%.2f bytes/s)", gFile, size, t, speed);
}

//______________________________________________________________________________
void RootdChdir(const char *dir)
{
   // Change directory.

   char buffer[kMAXPATHLEN + 256];

   if (dir && *dir == '~') {
      struct passwd *pw;
      int i = 0;
      const char *p = dir;

      p++;
      while (*p && *p != '/')
         buffer[i++] = *p++;
      buffer[i] = 0;

      if ((pw = getpwnam(i ? buffer : gUser)))
         sprintf(buffer, "%s%s", pw->pw_dir, p);
      else
         *buffer = 0;
   } else
      *buffer = 0;

   if (chdir(*buffer ? buffer : (dir && *dir ? dir : "/")) == -1) {
      sprintf(buffer, "cannot change directory to %s", dir);
      Perror(buffer);
      NetSend(buffer, kROOTD_CHDIR);
      return;
   } else {
      FILE *msg;

      if ((msg = fopen(".message", "r"))) {
         int len = fread(buffer, 1, kMAXPATHLEN, msg);
         fclose(msg);
         if (len > 0 && len < 1024) {
            buffer[len] = 0;
            NetSend(buffer, kMESS_STRING);
         }
      }

      if (!getcwd(buffer, kMAXPATHLEN)) {
         if (*dir == '/')
            sprintf(buffer, "%s", dir);
      }
      NetSend(buffer, kROOTD_CHDIR);
   }
}

//______________________________________________________________________________
void RootdMkdir(const char *dir)
{
   // Make directory.

   char buffer[kMAXPATHLEN];

   if (gAnon) {
      sprintf(buffer, "anonymous users may not create directories");
      ErrorInfo("RootdMkdir: %s", buffer);
   } else if (mkdir(dir, 0755) < 0) {
      sprintf(buffer, "cannot create directory %s", dir);
      Perror(buffer);
      ErrorInfo("RootdMkdir: %s", buffer);
   } else
      sprintf(buffer, "created directory %s", dir);

   NetSend(buffer, kROOTD_MKDIR);
}

//______________________________________________________________________________
void RootdRmdir(const char *dir)
{
   // Delete directory.

   char buffer[kMAXPATHLEN];

   if (gAnon) {
      sprintf(buffer, "anonymous users may not delete directories");
      ErrorInfo("RootdRmdir: %s", buffer);
   } else if (rmdir(dir) < 0) {
      sprintf(buffer, "cannot delete directory %s", dir);
      Perror(buffer);
      ErrorInfo("RootdRmdir: %s", buffer);
   } else
      sprintf(buffer, "deleted directory %s", dir);

   NetSend(buffer, kROOTD_RMDIR);
}

//______________________________________________________________________________
void RootdLsdir(const char *cmd)
{
   // List directory.

   char buffer[kMAXPATHLEN];

   // make sure all commands start with ls (should use snprintf)
   if (gAnon) {
      if (strlen(cmd) < 2 || strncmp(cmd, "ls", 2))
         sprintf(buffer, "ls %s", cmd);
      else
         sprintf(buffer, "%s", cmd);
   } else {
      if (strlen(cmd) < 2 || strncmp(cmd, "ls", 2))
         sprintf(buffer, "ls %s 2>/dev/null", cmd);
      else
         sprintf(buffer, "%s 2>/dev/null", cmd);
   }

   FILE *pf;
   if ((pf = popen(buffer, "r")) == 0) {
      sprintf(buffer, "error in popen");
      Perror(buffer);
      NetSend(buffer, kROOTD_LSDIR);
      ErrorInfo("RootdLsdir: %s", buffer);
      return;
   }

   // read output of ls
   int  ch, i = 0, cnt = 0;
//again:
   for (ch = fgetc(pf); ch != EOF; ch = fgetc(pf)) {
      buffer[i++] = ch;
      cnt++;
      if (i == kMAXPATHLEN-1) {
         buffer[i] = 0;
         NetSend(buffer, kMESS_STRING);
         i = 0;
      }
   }
   // this will be true if forked process was not yet ready to be read
//   if (cnt == 0 && ch == EOF) goto again;

   pclose(pf);

   buffer[i] = 0;
   NetSend(buffer, kROOTD_LSDIR);
}

//______________________________________________________________________________
void RootdPwd()
{
   // Print path of working directory.

   char buffer[kMAXPATHLEN];

   if (!getcwd(buffer, kMAXPATHLEN)) {
      sprintf(buffer, "current directory not readable");
      Perror(buffer);
      ErrorInfo("RootdPwd: %s", buffer);
   }

   NetSend(buffer, kROOTD_PWD);
}

//______________________________________________________________________________
void RootdMv(const char *msg)
{
   // Rename a file.

   char file1[kMAXPATHLEN], file2[kMAXPATHLEN], buffer[kMAXPATHLEN];
   sscanf(msg, "%s %s", file1, file2);

   if (gAnon) {
      sprintf(buffer, "anonymous users may not rename files");
      ErrorInfo("RootdMv: %s", buffer);
   } else if (rename(file1, file2) < 0) {
      sprintf(buffer, "cannot rename file %s to %s", file1, file2);
      Perror(buffer);
      ErrorInfo("RootdMv: %s", buffer);
   } else
      sprintf(buffer, "renamed file %s to %s", file1, file2);

   NetSend(buffer, kROOTD_MV);
}

//______________________________________________________________________________
void RootdRm(const char *file)
{
   // Delete a file.

   char buffer[kMAXPATHLEN];

   if (gAnon) {
      sprintf(buffer, "anonymous users may not delete files");
      ErrorInfo("RootdRm: %s", buffer);
   } else if (unlink(file) < 0) {
      sprintf(buffer, "cannot unlink file %s", file);
      Perror(buffer);
      ErrorInfo("RootdRm: %s", buffer);
   } else
      sprintf(buffer, "removed file %s", file);

   NetSend(buffer, kROOTD_RM);
}

//______________________________________________________________________________
void RootdChmod(const char *msg)
{
   // Delete a file.

   char file[kMAXPATHLEN], buffer[kMAXPATHLEN];
   int  mode;

   sscanf(msg, "%s %d", file, &mode);

   if (gAnon) {
      sprintf(buffer, "anonymous users may not change file permissions");
      ErrorInfo("RootdChmod: %s", buffer);
   } else if (chmod(file, mode) < 0) {
      sprintf(buffer, "cannot chmod file %s to 0%o", file, mode);
      Perror(buffer);
      ErrorInfo("RootdChmod: %s", buffer);
   } else
      sprintf(buffer, "changed permission of file %s to 0%o", file, mode);

   NetSend(buffer, kROOTD_CHMOD);
}

//______________________________________________________________________________
void RootdParallel()
{
   // Handle initialization message from remote host. If size > 0 then
   // so many parallel sockets will be opened to the remote host.

   int buf[3];
   if (NetRecvRaw(buf, sizeof(buf)) < 0)
      Error(ErrFatal, kErrFatal, "RootdParallel: error receiving message");

   int size = ntohl(buf[1]);
   int port = ntohl(buf[2]);

   if (gDebug > 0)
      ErrorInfo("RootdParallel: port = %d, size = %d", port, size);

   if (size > 0)
      NetParOpen(port, size);
}

//______________________________________________________________________________
bool RootdReUseAuth(const char *sstr, int kind)
{
   // Check the requiring subject has already authenticated during this session
   // and its 'ticket' is still valid
   // Not implemented for SRP and Krb5 (yet)

   if (RpdReUseAuth(sstr, kind)) {

      // Already authenticated ... we can login now
      RootdLogin();
      return 1;

   }
   return 0;
}

//______________________________________________________________________________
static void RootdTerm(int)
{
   // Termination upon receipt of a SIGTERM.

   ErrorInfo("RootdTerm: rootd.cxx: got a SIGTERM");
   // Terminate properly
   RootdAuthCleanup(0,0);
   // Trim Auth Table
   RpdUpdateAuthTab(0,0,0);
}

//______________________________________________________________________________
void RootdLoop()
{
   // Handle all rootd commands. Returns after file close command.

   const int     kMaxBuf = 1024;
   char          recvbuf[kMaxBuf];
   EMessageTypes kind;
   int           authmeth;

   // Set debug level in RPDUtil ...
   RpdSetDebugFlag(gDebug);

   // CleanUp authentication table, if needed or required ...
   RpdCheckSession();

   // Init Random machinery ...
   RpdInitRand();

   // Get Host name
   const char *OpenHost = NetRemoteHost();
   strcpy(gOpenHost, OpenHost);

   while (1) {

      if (NetRecv(recvbuf, kMaxBuf, kind) < 0)
         Error(ErrFatal, kErrFatal, "RootdLoop: error receiving message");

      if (gDebug > 2 && kind != kROOTD_PASS)
         ErrorInfo("RootdLoop: kind:%d -- buf:'%s' (len:%d) -- auth:%d",
                   kind, recvbuf, strlen(recvbuf), gAuth);

      // For gClientProtocol >= 9:
      // if authentication required, check if we accept the method proposed;
      // if not send back the list of accepted methods, if any ...
      if ((authmeth = RpdGetAuthMethod(kind)) != -1) {

         if (gClientProtocol == 0)
            gClientProtocol = RpdGuessClientProt(recvbuf, kind);

         if (gClientProtocol > 8) {

            // Check if accepted ...
            if (RpdCheckAuthAllow(authmeth, gOpenHost)) {
               if (gNumAllow > 0) {
                  if (gAuthListSent == 0) {
                     if (gDebug > 0)
                        ErrorInfo("RootdLoop: %s method not accepted from host: %s",
                                  kAuthMeth[authmeth], gOpenHost);
                     NetSend(kErrNotAllowed, kROOTD_ERR);
                     RpdSendAuthList();
                     gAuthListSent = 1;
                     goto next;
                  } else {
                     Error(ErrFatal, kErrNotAllowed, "RootdLoop: method not in the list sent to client");
                  }
               } else
                  Error(ErrFatal, kErrConnectionRefused, "RootdLoop: connection refused from host %s", gOpenHost);
            }

            // Then check if a previous authentication exists and is valid
            // ReUse does not apply for RFIO
            if (kind != kROOTD_RFIO && RootdReUseAuth(recvbuf,kind)) continue;
         }
      }

      if (kind != kROOTD_PASS     && kind != kROOTD_CLEANUP   &&
          kind != kROOTD_PROTOCOL && kind != kROOTD_PROTOCOL2 &&
          authmeth == -1 && gAuth == 0)
         Error(ErrFatal, kErrNoUser, "RootdLoop: not authenticated");

      switch (kind) {
         case kROOTD_USER:
            RootdUser(recvbuf);
            break;
         case kROOTD_SRPUSER:
            RootdSRPUser(recvbuf);
            break;
         case kROOTD_PASS:
            RootdPass(recvbuf);
            break;
         case kROOTD_KRB5:
            RootdKrb5Auth(recvbuf);
            break;
         case kROOTD_GLOBUS:
            RootdGlobusAuth(recvbuf);
            break;
         case kROOTD_SSH:
            RootdSshAuth(recvbuf);
            break;
         case kROOTD_RFIO:
            RootdRfioAuth(recvbuf);
            break;
         case kROOTD_CLEANUP:
            RootdAuthCleanup(recvbuf,1);
            return;
         case kROOTD_OPEN:
            RootdOpen(recvbuf);
            break;
         case kROOTD_PUT:
            RootdPut(recvbuf);
            break;
         case kROOTD_GET:
            RootdGet(recvbuf);
            break;
         case kROOTD_FLUSH:
            RootdFlush();
            break;
         case kROOTD_CLOSE:
            RootdClose();
            if (gClientProtocol < 7)
               return;
            break;
         case kROOTD_FSTAT:
            RootdFstat();
            break;
         case kROOTD_STAT:
            RootdStat();
            break;
         case kROOTD_PROTOCOL:
            RootdProtocol();
            break;
         case kROOTD_PROTOCOL2:
            RootdProtocol2(recvbuf);
            break;
         case kROOTD_PUTFILE:
            RootdPutFile(recvbuf);
            break;
         case kROOTD_GETFILE:
            RootdGetFile(recvbuf);
            break;
         case kROOTD_CHDIR:
            RootdChdir(recvbuf);
            break;
         case kROOTD_MKDIR:
            RootdMkdir(recvbuf);
            break;
         case kROOTD_RMDIR:
            RootdRmdir(recvbuf);
            break;
         case kROOTD_LSDIR:
            RootdLsdir(recvbuf);
            break;
         case kROOTD_PWD:
            RootdPwd();
            break;
         case kROOTD_MV:
            RootdMv(recvbuf);
            break;
         case kROOTD_RM:
            RootdRm(recvbuf);
            break;
         case kROOTD_CHMOD:
            RootdChmod(recvbuf);
            break;
         case kROOTD_BYE:
            return;
         default:
            Error(ErrFatal, kErrBadOp, "RootdLoop: received bad opcode %d", kind);
      }

      if (gClientProtocol > 8) {

         // If authentication failure prepare or continue negotiation
         // Don't do this if this was a SSH notification failure
         // because in such a case it was already done in the
         // appropriate daemon child
         int doneg = (authmeth != -1 || kind == kROOTD_PASS) &&
                     (gRemPid > 0 || kind != kROOTD_SSH);
         if (gDebug > 2 && doneg)
            ErrorInfo("RootdLoop: %s: kind:%d -- meth:%d -- gAuth:%d -- gNumLeft:%d",
                      "Authentication",kind, authmeth, gAuth, gNumLeft);
         // If authentication failure, check if other methods could be tried ...
         if (gAuth == 0 && doneg) {
            if (gNumLeft > 0) {
               if (gAuthListSent == 0) {
                  RpdSendAuthList();
                  gAuthListSent = 1;
               } else
                  NetSend(-1, kROOTD_NEGOTIA);
            } else
               Error(ErrFatal, kErrFatal, "RootdLoop: authentication failed");
         }
      }
next:
      continue;
   }
}

//______________________________________________________________________________
int main(int argc, char **argv)
{
   char *s;
   int   tcpwindowsize = 65535;

   // Error handlers
   gErrSys   = ErrSys;
   gErrFatal = ErrFatal;
   gErr      = Err;

   // function for dealing with SIGPIPE signals (used in NetSetOptions(),
   // rpdutils/net.cxx)
   gSigPipeHook = SigPipe;

   ErrorInit(argv[0]);

   // To terminate correctly ... maybe not needed
   signal(SIGTERM, RootdTerm);

#ifdef R__KRB5
   const char *kt_fname;

   int retval = krb5_init_context(&gKcontext);
   if (retval)
      Error(ErrFatal, kErrFatal, "%s while initializing krb5",
            error_message(retval));
#endif
#ifdef R__GLBS
   char    GridMap[kMAXPATHLEN]         = { 0 };
#endif

   // Define service
   strcpy(gService, "rootd");

   // Try determining gExecDir and gConfDir
   char *exec;
   exec = RootdExpandPathName(argv[0]);
   if (exec && exec[0] == '/') {
      char *pstr = strrchr(exec, '/');
      if (pstr) {
         int plen = (int)(pstr-exec);
         strncpy(gExecDir, exec, plen);
         gExecDir[plen] = 0;
         pstr--;
         pstr = strrchr(pstr, '/');
         if (pstr) {
            plen = (int)(pstr-exec);
            strncpy(gConfDir, exec, plen);
            gConfDir[plen] = 0;
         }
      }
      free(exec);
   }

   while (--argc > 0 && (*++argv)[0] == '-')
      for (s = argv[0]+1; *s != 0; s++)
         switch (*s) {
            case 'i':
               gInetdFlag = 1;
               break;

            case 'r':
               gReadOnly = 1;
               break;

            case 'p':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-p requires a port number as argument\n");
                  Error(ErrFatal, kErrFatal, "-p requires a port number as argument");
               }
               char *p;
               gPort1 = strtol(*++argv, &p, 10);
               if (*p == '-')
                  gPort2 = strtol(++p, &p, 10);
               else if (*p == '\0')
                  gPort2 = gPort1;
               if (*p != '\0' || gPort2 < gPort1 || gPort2 < 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "invalid port number or range: %s\n", *argv);
                  Error(ErrFatal, kErrFatal, "invalid port number or range: %s", *argv);
               }
               break;

            case 'd':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-d requires a debug level as argument\n");
                  Error(ErrFatal, kErrFatal, "-d requires a debug level as argument");
               }
               gDebug = atoi(*++argv);
               break;

            case 'b':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-b requires a buffersize in bytes as argument\n");
                  Error(ErrFatal, kErrFatal, "-b requires a buffersize in bytes as argument");
               }
               tcpwindowsize = atoi(*++argv);
               break;

            case 'f':
               gForegroundFlag = 1;
               break;

            case 'P':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-P requires a file name for SRP password file\n");
                  Error(ErrFatal, kErrFatal, "-P requires a file name for SRP password file");
               }
               gAltSRP = 1;
               sprintf(gAltSRPPass, "%s", *++argv);
               break;

            case 'R':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-R requires a hex bit mask as argument\n");
                  Error(ErrFatal, kErrFatal, "-R requires a hex but mask as argument");
               }
               gReUseAllow = strtol(*++argv, (char **)0, 16);
               break;

            case 'T':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-T requires a dir path for temporary files [/usr/tmp]\n");
                  Error(ErrFatal, kErrFatal, "-T requires a dir path for temporary files [/usr/tmp]");
               }
               sprintf(gTmpDir, "%s", *++argv);
               break;

            case 's':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-s requires as argument a port number for the sshd daemon\n");
                  Error(ErrFatal, kErrFatal, "-s requires as argument a port number for the sshd daemon");
               }
               gSshdPort = atoi(*++argv);
               break;

#ifdef R__KRB5
            case 'S':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-S requires a path to your keytab\n");
                  Error(ErrFatal, kErrFatal, "-S requires a path to your keytab\n");
               }
               kt_fname = *++argv;
               if ((retval = krb5_kt_resolve(gKcontext, kt_fname, &gKeytab)))
                  Error(ErrFatal, kErrFatal, "%s while resolving keytab file %s",
                        error_message(retval), kt_fname);
               break;
#endif

#ifdef R__GLBS
            case 'G':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-G requires a file name for the gridmap file\n");
                  Error(ErrFatal, kErrFatal, "-G requires a file name for the gridmap file");
               }
               sprintf(GridMap, "%s", *++argv);
               if (setenv("GRIDMAP",GridMap,1) ) {
                  Error(ErrFatal, kErrFatal, "while setting the GRIDMAP environment variable");
               }
               break;

            case 'C':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-C requires a file name for the host certificates file location\n");
                  Error(ErrFatal, kErrFatal, "-C requires a file name for the host certificates file location");
               }
               sprintf(gHostCertConf, "%s", *++argv);
               break;
#endif

            default:
               if (!gInetdFlag)
                  fprintf(stderr, "unknown command line option: %c\n", *s);
               Error(ErrFatal, kErrFatal, "unknown command line option: %c", *s);
         }

   // dir for temporary files
   if (strlen(gTmpDir) == 0) {
      strcpy(gTmpDir, "/usr/tmp");
      if (access(gTmpDir, W_OK) == -1) {
         strcpy(gTmpDir, "/tmp");
      }
   }

   // root tab file
   sprintf(gRootdTab, "%s/rootdtab", gTmpDir);

   // authentication tab file
   sprintf(gRpdAuthTab, "%s/rpdauthtab", gTmpDir);

   // Set auth tab file in rpdutils...
   RpdSetAuthTabFile(gRpdAuthTab);

   // Log to stderr if not started as daemon ...
   if (gForegroundFlag) RpdSetRootLogFlag(1);

   if (argc > 0) {
      strncpy(gConfDir, *argv, kMAXPATHLEN-1);
      gConfDir[kMAXPATHLEN-1] = 0;
      sprintf(gExecDir, "%s/bin", gConfDir);
      sprintf(gSystemDaemonRc, "%s/etc/system%s", gConfDir, kDaemonRc);
   } else {
      // try to guess the config directory...
#ifndef ROOTPREFIX
      if (strlen(gConfDir) == 0) {
         if (getenv("ROOTSYS")) {
            strcpy(gConfDir, getenv("ROOTSYS"));
            sprintf(gExecDir, "%s/bin", gConfDir);
            sprintf(gSystemDaemonRc, "%s/etc/system%s", gConfDir, kDaemonRc);
            if (gDebug > 0)
               ErrorInfo("main: no config directory specified using ROOTSYS (%s)", gConfDir);
         } else {
            if (!gInetdFlag)
               fprintf(stderr, "rootd: no config directory specified\n");
            Error(ErrFatal, kErrFatal, "main: no config directory specified");
         }
      }
#else
      strcpy(gConfDir, ROOTPREFIX);
#endif
#ifdef ROOTBINDIR
      strcpy(gExecDir, ROOTBINDIR);
#endif
#ifdef ROOTETCDIR
      sprintf(gSystemDaemonRc, "%s/system%s", ROOTETCDIR, kDaemonRc);
#endif
   }

   if (!gInetdFlag) {

      // Start rootd up as a daemon process (in the background).
      // Also initialize the network connection - create the socket
      // and bind our well-know address to it.

      gPort = gPort1;
      int fdkeep = NetInit(gService, gPort, gPort2, tcpwindowsize);
      if (!gForegroundFlag) DaemonStart(1, fdkeep, kROOTD);
   }

   if (gDebug > 0)
      ErrorInfo("main: pid = %d, ppid = %d, gInetdFlag = %d, gProtocol = %d",
                getpid(), getppid(), gInetdFlag, gProtocol);

   // Concurrent server loop.
   // The child created by NetOpen() handles the client's request.
   // The parent waits for another request. In the inetd case,
   // the parent from NetOpen() never returns.

   while (1) {

      if (NetOpen(gInetdFlag, kROOTD) == 0) {
         RootdParallel();  // see if we should use parallel sockets
         RootdLoop();      // child processes client's requests
         NetClose();       // till we are done
         exit(0);
      }

      // parent waits for another client to connect

   }

#ifdef R__KRB5
   // never called... needed?
   krb5_free_context(gKcontext);
#endif
}

