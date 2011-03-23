// @(#)root/rootd:$Id$
// Author: Fons Rademakers   11/08/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
//   -b tcpwindowsize  specifies the tcp window size in bytes (e.g. see //
//                     http://www.psc.edu/networking/perf_tune.html)    //
//                     Default is 65535. Only change default for pipes  //
//                     with a high bandwidth*delay product.             //
//   -C hostcertfile   defines a file where to find information for the //
//                     local host Globus information (see GLOBUS.README //
//                     for details)                                     //
//   -d level          level of debug info written to syslog            //
//                     0 = no debug (default)                           //
//                     1 = minimum                                      //
//                     2 = medium                                       //
//                     3 = maximum                                      //
//   -D rootdaemonrc   read access rules from file <rootdaemonrc>.      //
//                     By default <root_etc_dir>/system.rootdaemonrc is //
//                     used for access rules; for privately started     //
//                     daemons $HOME/.rootdaemonrc (if present) takes   //
//                     highest priority.                                //
//   -E                obsolete; up to v4.00.08 this option was used to //
//                     force exclusivity of the authentication tokens;  //
//                     with the new approach for authentication tab     //
//                     files this option is dummy.                      //
//   -f                do not run as daemon, run in the foreground      //
//   -F filename       Specify that rootd is in CASTOR mode and should  //
//                     serve this file.                                 //
//   -G gridmapfile    defines the gridmap file to be used for globus   //
//                     authentication if different from globus default  //
//                     (/etc/grid-security/gridmap); (re)defines the    //
//                     GRIDMAP environment variable.                    //
//   -h                print usage message                              //
//   -H reqid          In CASTOR mode, specify the ID of the request    //
//                     that should be accepted                          //
//   -i                says we were started by inetd                    //
//   -noauth           do not require client authentication             //
//   -nologin          do not login the client to its $HOME as it may   //
//                     not exist                                        //
//   -p port#          specifies a different port to listen on.         //
//                     Use port1-port2 to find first available port in  //
//                     range. Use 0-N for range relative to service     //
//                     port.                                            //
//   -P file           use this password file, instead of .srootdpass   //
//   -r                files can only be opened in read-only mode       //
//   -R bitmask        bit mask specifies which methods will allow      //
//                     authentication to be re-used                     //
//   -s <sshd_port>    specifies the port number for the sshd daemon    //
//                     (default is 22)                                  //
//   -S keytabfile     use this keytab file, instead of the default     //
//                     (option only supported when compiled with        //
//                     Kerberos5 support)                               //
//   -T <tmpdir>       specifies the directory path to be used to place //
//                     temporary files; default is /usr/tmp.            //
//                     Useful if not running as root.                   //
//   -w                do not check /etc/hosts.equiv, $HOME/.rhosts     //
//                     for UsrPwd authentications; by default these     //
//                     files are checked first by calling ruserok(...); //
//                     if this option is specified a password is always //
//                     required.
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
// 9 -> 10: Receives client protocol with kROOTD_PROTOCOL + change cleaning protocol
// 10 -> 11: modified SSH protocol + support for server 'no authentication' mode
// 11 -> 12: added support for stat functionality (access,opendir,...) (cfr.TNetSystem)
//           and support for OpenSSL keys for encryption
// 12 -> 13: changed return message of RootdFstat()
// 13 -> 14: support for TNetFile setup via TXNetFile
// 14 -> 15: support for SSH authentication via SSH tunnel
// 15 -> 16: cope with the bug fix in TUrl::GetFile
// 16 -> 17: Addition of "Gets" (multiple buffers in a single request)
// 17 -> 18: fix problems with '//' in admin paths; partial logging in castor mode

#include "RConfigure.h"
#include "RConfig.h"

#include <ctype.h>
#include <fcntl.h>
#include <pwd.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <errno.h>
#include <netdb.h>
#include "snprintf.h"

#include <sys/types.h>
#include <dirent.h>

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
#elif defined(__FreeBSD__) || defined(__OpenBSD__)
#include <sys/param.h>
#include <sys/mount.h>
#else
#include <sys/statfs.h>
#endif

#if defined(linux) || defined(__hpux) || defined(_AIX) || defined(__alpha) || \
    defined(__sun) || defined(__sgi) || defined(__FreeBSD__) || \
    defined(__APPLE__) || defined(cygwingcc) || defined(__OpenBSD__)
#define HAVE_MMAP
#endif

#ifdef HAVE_MMAP
#   include <sys/mman.h>
#ifndef MAP_FILE
#define MAP_FILE 0           /* compatability flag */
#endif
#endif

#if (defined(__FreeBSD__) && (__FreeBSD__ < 4)) || defined(__OpenBSD__) || \
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
    defined(__MACH__) || defined(cygwingcc) || defined(__OpenBSD__)
#include <grp.h>
#include <sys/types.h>
#include <signal.h>
#define ROOT_SIGNAL_INCLUDED
#endif

#if defined(__alpha) && !defined(linux) && !defined(__FreeBSD__) && \
    !defined(__OpenBSD__)
extern "C" int initgroups(const char *name, int basegid);
#ifndef ROOT_SIGNAL_INCLUDED
#include <signal.h>
#endif
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

extern "C" {
#include "rsadef.h"
#include "rsalib.h"
}

// Debug flag
int gDebug  = 0;

//--- Local Globals -----------------------------------------------------------

enum EFileMode{ kBinary, kAscii };

static std::string gRootdTab;     // keeps track of open files
static std::string gRpdAuthTab;   // keeps track of authentication info
static EService gService         = kROOTD;
static int gProtocol             = 18;      // increase when protocol changes
static int gClientProtocol       = -1;      // Determined by RpdInitSession
static int gAnon                 = 0;       // anonymous user flag
static double gBytesRead         = 0;
static double gBytesWritten      = 0;
static DIR *gRDDirectory         = 0;
static int gDownloaded           = 0;
static int gFd                   = -1;
static int gFtp                  = 0;
static int gInetdFlag            = 0;
static char gOption[32]          = { 0 };
static char gFile[kMAXPATHLEN]   = { 0 };
static int gUploaded             = 0;
static int gWritable             = 0;
static int gReadOnly             = 0;
static std::string gUser;
static std::string gPasswd;

// CASTOR specific
static int gCastorFlag           = 0;
static std::string gCastorFile;
static std::string gCastorReqId;

using namespace ROOT;

//--- Error handlers -----------------------------------------------------------

//______________________________________________________________________________
void Err(int level,const char *msg, int size)
{
   Perror((char *)msg,size);
   if (level > -1) NetSendError((ERootdErrors)level);
}
//______________________________________________________________________________
void ErrFatal(int level,const char *msg, int size)
{
   Perror((char *)msg,size);
   if (level > -1) NetSendError((ERootdErrors)level);
   RootdClose();
   exit(1);
}
//______________________________________________________________________________
void ErrSys(int level,const char *msg, int size)
{
   Perror((char *)msg,size);
   ErrFatal(level,msg,size);
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

   // Terminate properly
   RpdAuthCleanup(0, 0);
   RootdClose();
   exit(1);
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
         strncpy(path, pw->pw_dir, kMAXPATHLEN-1);
         path[sizeof(path)-1] = '\0';
         return path;
      }
   } else {
      if (mydir[0])
         return mydir;
      pw = getpwuid(getuid());
      if (pw) {
         strncpy(mydir, pw->pw_dir, kMAXPATHLEN-1);
         mydir[sizeof(mydir)-1] = '\0';
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
   strlcpy(cmd, "/bin/echo ", sizeof(cmd));
#else
   strlcpy(cmd, "echo ", sizeof(cmd));
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
   // filename device inode mode username pid
   // where device is the unique file system id, inode is the unique file
   // ref number, mode is either "read" or "write", username the user
   // who has the file open and pid is the pid of the rootd having the
   // file open.

   // Open rootdtab file. Try first /usr/tmp and then /tmp.
   // The lockf() call can fail if the directory is NFS mounted
   // and the lockd daemon is not running.

   const char *sfile = gRootdTab.c_str();
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
         sfile = gRootdTab.c_str()+4;
         goto again;
      }
      Error(ErrSys, kErrFatal, "RootdCheckTab: error opening %s", sfile);
   }

   // lock the file
   if (lockf(fid, F_LOCK, (off_t)1) == -1) {
      if (sfile[1] == 'u' && create) {
         close(fid);
         remove(sfile);
         sfile = gRootdTab.c_str()+4;
         goto again;
      }
      Error(ErrSys, kErrFatal, "RootdCheckTab: error locking %s", sfile);
   }
   if (gDebug > 2)
      ErrorInfo("RootdCheckTab: file %s locked", sfile);

   struct stat sbuf;
   fstat(fid, &sbuf);
   size_t siz = sbuf.st_size;

   dev_t device;
   ino_t inode;
   if (stat(gFile, &sbuf) == -1) {
      device = 0;
      inode  = 0;
   } else {
      device = sbuf.st_dev;
      inode  = sbuf.st_ino;
   }

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
         unsigned long dev, ino;
         sscanf(s, "%s %lu %lu %s %s %d", msg, &dev, &ino, gmode, user, &pid);
         if (kill(pid, 0) == -1 && GetErrno() == ESRCH) {
            ErrorInfo("RootdCheckTab: remove stale lock (%s %lu %lu %s %s %d)\n",
                msg, dev, ino, gmode, user, pid);
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
         } else if ((dev_t)dev == device && (ino_t)ino == inode) {
            if (mode == 1)
               result = 0;
            else if (!strcmp(gmode, "write"))
               result = 0;
         }
         s = n;
      }
      if (changed) {
         if (ftruncate(fid, 0) == -1)
            ErrorInfo("RootdCheckTab: ftruncate failed");
         lseek(fid, 0, SEEK_SET);
         if (siz > 0) {
            while (write(fid, fbuf, siz) < 0 && GetErrno() == EINTR)
               ResetErrno();
         }
      }
      delete [] fbuf;
   }

   if (result && !noupdate) {
      unsigned long dev = device;
      unsigned long ino = inode;
      char *tmsg = msg;
      int lmsg = strlen(gFile) + gUser.length() + strlen(smode) + 40;
      if (lmsg > kMAXPATHLEN)
         tmsg = new char[lmsg];
      sprintf(tmsg, "%s %lu %lu %s %s %d\n",
                   gFile, dev, ino, smode, gUser.c_str(), (int) getpid());
      if (write(fid, tmsg, strlen(tmsg)) == -1)
         Error(ErrSys, kErrFatal, "RootdCheckTab: error writing %s", sfile);
      if (tmsg != msg)
         delete[] tmsg;
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

   const char *sfile = gRootdTab.c_str();
   int fid;

again:
   if (access(sfile, F_OK) == -1) {
      if (sfile[1] == 'u') {
         sfile = gRootdTab.c_str()+4;
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
   dev_t device = sbuf.st_dev;
   ino_t inode  = sbuf.st_ino;

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
         unsigned long dev, ino;
         sscanf(s, "%s %lu %lu %s %s %d", msg, &dev, &ino, gmode, user, &pid);
         if (kill(pid, 0) == -1 && GetErrno() == ESRCH) {
            stale = 1;
            ErrorInfo("Remove Stale Lock (%s %lu %lu %s %s %d)\n",
                       msg, dev, ino, gmode, user, pid);
         }
         if (stale || (!force && mypid == pid) ||
            (force && device == (dev_t)dev && inode == (ino_t)ino &&
             !strcmp(gUser.c_str(), user))) {
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
         if (ftruncate(fid, 0) == -1)
            ErrorInfo("RootdCheckTab: ftruncate failed");
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
      ErrorInfo("RootdCloseFtp: %d files uploaded, %d files downloaded,"
                " rd=%g, wr=%g, rx=%g, tx=%g",
                gUploaded, gDownloaded, gBytesRead, gBytesWritten,
                NetGetBytesRecv(), NetGetBytesSent());
   else
      ErrorInfo("Rootd: %d files uploaded, %d files downloaded, rd=%g,"
                " wr=%g, rx=%g, tx=%g",
                gUploaded, gDownloaded, gBytesRead, gBytesWritten,
                NetGetBytesRecv(), NetGetBytesSent());
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
                gFile, gBytesRead, gBytesWritten,
                NetGetBytesRecv(), NetGetBytesSent());
   else
      ErrorInfo("Rootd: file %s closed, rd=%g, wr=%g, rx=%g, tx=%g", gFile,
                gBytesRead, gBytesWritten,
                NetGetBytesRecv(), NetGetBytesSent());
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
void RootdFstat(const char *buf)
{
   // Return file stat information in same format as TSystem::GetPathInfo().

   char     msg[256];
   int      islink = 0;

#if defined(R__SEEK64)
   struct stat64 statbuf;
#elif defined(WIN32)
   struct _stati64 statbuf;
#else
   struct stat statbuf;
#endif

   int rc = -1;
   if (!buf || !strlen(buf)) {

      if (RootdIsOpen()) {
#if defined(R__SEEK64)
         rc = fstat64(gFd, &statbuf);
#elif defined(WIN32)
         rc = _fstati64(gFd, &statbuf);
#else
         rc = fstat(gFd, &statbuf);
#endif
      }
   } else {

      char *epath = (char *)buf;
      if (buf[0] == '/' && buf[1] == '/')
         epath++;
#if defined(R__SEEK64)
      rc = lstat64(epath, &statbuf);
#elif defined(WIN32)
      rc = _stati64(epath, &statbuf);
#else
      rc = lstat(epath, &statbuf);
#endif
      if (rc >= 0) {
         islink = S_ISLNK(statbuf.st_mode);
         if (islink) {
#if defined(R__SEEK64)
            rc = stat64(epath, &statbuf);
#elif defined(WIN32)
            rc = _stati64(epath, &statbuf);
#else
            rc = stat(epath, &statbuf);
#endif
         }
      }
   }

   // New format for recent clients
   if (gClientProtocol > 11) {
      if (rc >= 0)
         sprintf(msg, "%ld %ld %d %d %d %lld %ld %d", (long)statbuf.st_dev,
                 (long)statbuf.st_ino, statbuf.st_mode, (int)(statbuf.st_uid),
                 (int)(statbuf.st_gid), (Long64_t)statbuf.st_size, statbuf.st_mtime,
                 islink);
      else
         sprintf(msg, "-1 -1 -1 -1 -1 -1 -1 -1");
   } else {
      // Old client: use previous incomplete format
      if (rc >= 0) {
         long id = (statbuf.st_dev << 24) + statbuf.st_ino;
         Long64_t size = statbuf.st_size;
         long modtime = statbuf.st_mtime;
         long flags = 0;
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
   }

   NetSend(msg, kROOTD_FSTAT);
}

//______________________________________________________________________________
void RootdParallel()
{
   // Handle initialization message from remote host. If size > 1 then
   // so many parallel sockets will be opened to the remote host.

   int buf[3];
   if (NetRecvRaw(buf, sizeof(buf)) < 0)
      Error(ErrFatal, kErrFatal, "RootdParallel: error receiving message");

   int size = ntohl(buf[1]);
   int port = ntohl(buf[2]);

   if (gDebug > 0)
      ErrorInfo("RootdParallel: port = %d, size = %d", port, size);

   if (size > 1)
      NetParOpen(port, size);
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

   gBytesRead = gBytesWritten = 0;
   NetResetByteCount();

   sscanf(msg, "%s %s", file, option);

   if (gCastorFlag) {

      // Checking the CASTOR Request ID
      if (gCastorReqId.length() > 0) {
         if (strstr(file, gCastorReqId.c_str()) == 0) {
            Error(ErrFatal, kErrNoAccess,
                  "RootdOpen: Bad CASTOR Request ID: %s rather than %s",
                  file, gCastorReqId.c_str());
         }
      }

      ErrorInfo("RootdOpen: CASTOR Flag on, file: %s", gCastorFile.c_str());
      strncpy(gFile, gCastorFile.c_str(), kMAXPATHLEN-1);
      gFile[kMAXPATHLEN-1] = '\0';

   } else {

      if (gClientProtocol > 14) {
         strlcpy(gFile, file, sizeof(gFile));
      } else {
         // Old clients send an additional slash at the beginning
         if (file[0] == '/')
            strlcpy(gFile, &file[1], sizeof(gFile));
         else
            strlcpy(gFile, file, sizeof(gFile));
      }

      gFile[strlen(file)] = '\0';
   }

   strlcpy(gOption, option, sizeof(gOption));

   int forceOpen = 0;
   if (option[0] == 'f') {
      forceOpen = 1;
      strlcpy(gOption, &option[1], sizeof(gOption));
   }

   int forceRead = 0;
   if (!strcmp(option, "+read")) {
      forceRead = 1;
      strlcpy(gOption, &option[1], sizeof(gOption));
   }

   int create = 0;
   if (!strcmp(gOption, "new") || !strcmp(gOption, "create"))
      create = 1;
   int recreate = strcmp(gOption, "recreate") ? 0 : 1;
   int update   = strcmp(gOption, "update")   ? 0 : 1;
   int read     = strcmp(gOption, "read")     ? 0 : 1;
   if (!create && !recreate && !update && !read) {
      read = 1;
      strlcpy(gOption, "read", sizeof(gOption));
   }

   if (!read && gReadOnly)
      Error(ErrFatal, kErrNoAccess,
            "RootdOpen: file %s can only be opened in \"READ\" mode", gFile);

   if (!gAnon) {
      char *fname;
      if ((fname = RootdExpandPathName(gFile))) {
         strlcpy(gFile, fname, sizeof(gFile));
         free(fname);
      } else
         Error(ErrFatal, kErrBadFile, "RootdOpen: bad file name %s", gFile);
   }

   if (forceOpen)
      RootdCloseTab(1);

   int trunc = 0;
   if (recreate) {
      if (!RootdCheckTab(-1))
         Error(ErrFatal, kErrFileWriteOpen,
               "RootdOpen: file %s already opened in read or write mode", gFile);
      if (!access(gFile, F_OK))
         trunc = O_TRUNC;
      else {
         recreate = 0;
         create   = 1;
         strlcpy(gOption, "create", sizeof(gOption));
      }
   }

   if (create && !access(gFile, F_OK))
      Error(ErrFatal, kErrFileExists, "RootdOpen: file %s already exists", gFile);

   int wasupdt = 0;
   if (update) {
      if (access(gFile, F_OK)) {
         update = 0;
         create = 1;
         wasupdt = 1;
         strlcpy(gOption, "create", sizeof(gOption));
      }
      if (update && access(gFile, W_OK))
         Error(ErrFatal, kErrNoAccess,
               "RootdOpen: no write permission for file %s", gFile);
   }

   if (read) {
      if (access(gFile, F_OK))
         Error(ErrFatal, kErrNoFile,
               "RootdOpen: file %s does not exist (errno: 0x%x)", gFile, errno);
      if (access(gFile, R_OK))
         Error(ErrFatal, kErrNoAccess,
               "RootdOpen: no read permission for file %s (errno: 0x%x)", gFile, errno);
   }

   if (create || recreate || update) {
      if (create || recreate) {
         // make sure file exists so RootdCheckTab works correctly
#ifndef WIN32
         gFd = SysOpen(gFile, O_RDWR | O_CREAT | trunc, 0644);
#else
         gFd = SysOpen(gFile, O_RDWR | O_CREAT | O_BINARY | trunc, S_IREAD | S_IWRITE);
#endif
         if (gFd != -1)
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

      gWritable = wasupdt ? 2 : 1;

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
   unsigned long dev = sbuf.st_dev;
   unsigned long ino = sbuf.st_ino;

   if (gDebug > 0)
      ErrorInfo("RootdOpen: file %s opened in mode %s", gFile, gOption);
   else {
      if (gAnon)
         ErrorInfo("RootdOpen: file %s (dev=%lu,inode=%lu,%s) opened by %s/%s",
                   gFile, dev, ino, gOption, gUser.c_str(), gPasswd.c_str());
      else
         ErrorInfo("RootdOpen: file %s (dev=%lu,inode=%lu,%s) opened by %s",
                   gFile, dev, ino, gOption, gUser.c_str());
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
      Error(ErrSys, kErrFileGet, "RootdGet: cannot seek to position %lld in"
            " file %s", offset, gFile);

   ssize_t siz;
   while ((siz = read(gFd, buf, len)) < 0 && GetErrno() == EINTR)
      ResetErrno();

   if (siz < 0)
      Error(ErrSys, kErrFileGet, "RootdGet: error reading from file %s", gFile);

   if (siz != len)
      Error(ErrFatal, kErrFileGet, "RootdGet: error reading all requested bytes"
            " from file %s, got %d of %d",gFile, siz, len);

   NetSend(0, kROOTD_GET);

   NetSendRaw(buf, len);

   delete [] buf;

   gBytesRead += len;

   if (gDebug > 0)
      ErrorInfo("RootdGet: read %d bytes starting at %lld from file %s",
                len, offset, gFile);
}

//______________________________________________________________________________
void RootdGets(const char *msg)
{
   // Gets multiple buffers from the specified list of offsets and lengths from
   // the currently open file and send it to the client in a single buffer.
   // (BUt rem it gets the buffer with the info in the same way it would get
   // new data)

   if (!RootdIsOpen())
      Error(ErrFatal, kErrNoAccess, "RootdGets: file %s not open", gFile);

   Int_t nbuf;      // Number of buffers
   Int_t len;       // len of the data buffer with the list of buffers
   Int_t npar;      // compatibility issues
   Int_t size;      // size of the readv block (all the small reads)
   Int_t maxTransz; // blocksize for the transfer

   npar = sscanf(msg, "%d %d %d", &nbuf, &len, &maxTransz);

   Long64_t *offsets = new Long64_t[nbuf];  // list to be filled
   Int_t    *lens    = new Int_t[nbuf];     // list to be filled
   char     *buf_in  = new char[len+1];     // buff coming from the server

   NetRecvRaw(buf_in, len);
   buf_in[len] = '\0';

   char *ptr = buf_in;
   size = 0;
   for(Int_t i = 0 ; i < nbuf ; i++) {
      sscanf(ptr, "%llu-%d/", &offsets[i], &lens[i]);
      ptr = strchr(ptr, '/') + 1;
      size += lens[i];
   }

   // If the blocksize is not specified the try to send
   // just a big block
   if( npar == 2  )
      maxTransz = size;

   // We are Ready to begin the transference
   NetSend(0, kROOTD_GETS);

   char *buf_out  = new char[maxTransz];
   char *buf_send = new char[maxTransz];
   Int_t actual_pos = 0; // position for the whole size
   Int_t buf_pos    = 0; // position in the buffer
   ssize_t siz = 0;

   for (Int_t i = 0; i < nbuf; i++) {
      Long64_t left = size - actual_pos;
      if (left > maxTransz)
         left = maxTransz;

      Int_t pos = 0; // Position for the disk read
      while ( pos < lens[i] ) {
#if defined (R__SEEK64)
         if (lseek64(gFd, offsets[i] + pos, SEEK_SET) < 0)
#elif defined(WIN32)
         if (_lseeki64(gFd, offsets[i] + pos, SEEK_SET) < 0)
#else
         if (lseek(gFd, offsets[i] + pos, SEEK_SET) < 0)
#endif
         Error(ErrSys, kErrFileGet, "RootdGets: cannot seek to position %lld in"
            " file %s", offsets[i], gFile);

         Int_t readsz = lens[i] - pos;
         if( readsz > ( left - buf_pos) )
            readsz = left - buf_pos;

         while ((siz = read(gFd, buf_out + buf_pos, readsz)) < 0 && GetErrno() == EINTR)
            ResetErrno();

         if (siz != readsz)
            goto end;

         pos += readsz;
         buf_pos += readsz;
         if ( buf_pos == left ) {
            if (gDebug > 0 )
               ErrorInfo("RootdGets: Sending %d bytes", left);

            // Swap buffers
            char *buf_tmp = buf_out;
            buf_out = buf_send;
            buf_send = buf_tmp;

            NetSendRaw(buf_send, left);
            actual_pos += left;
            buf_pos = 0;

            if ( left > (size - actual_pos) )
               left = size - actual_pos;
         }
      }
   }

end:
   if (siz < 0)
      Error(ErrSys, kErrFileGet, "RootdGets: error reading from file %s", gFile);

   if (actual_pos != size)
      Error(ErrFatal, kErrFileGet, "RootdGets: error reading all requested bytes"
            " from file %s, got %d of %d",gFile, actual_pos, size);

   delete [] buf_in;
   delete [] buf_out;
   delete [] buf_send;
   delete [] lens;
   delete [] offsets;

   gBytesRead += actual_pos;

   if (gDebug > 0)
      ErrorInfo("RootdGets: read %d bytes from file %s",
                actual_pos, gFile);
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
      strlcpy(gFile, file+1, sizeof(gFile));
   } else
      strlcpy(gFile, file, sizeof(gFile));

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
         Long64_t space = (Long64_t)statfsbuf.f_bsize * (Long64_t)statfsbuf.f_bfree;
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
      strlcpy(gFile, file+1, sizeof(gFile));
   } else
      strlcpy(gFile, file, sizeof(gFile));

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
   SPrintf(mess, 128, "%lld", size);
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

   const int kMAXBUFLEN = kMAXPATHLEN + 256;
   char buffer[kMAXBUFLEN];

   if (dir && *dir == '~') {
      struct passwd *pw;
      int i = 0;
      const char *p = dir;

      p++;
      while (*p && *p != '/')
         buffer[i++] = *p++;
      buffer[i] = 0;

      if ((pw = getpwnam(i ? buffer : gUser.c_str())))
         SPrintf(buffer, kMAXBUFLEN, "%s%s", pw->pw_dir, p);
      else
         *buffer = 0;
   } else
      *buffer = 0;

   if (chdir(*buffer ? buffer : (dir && *dir ? dir : "/")) == -1) {
      SPrintf(buffer,kMAXBUFLEN,"cannot change directory to %s",dir);
      Perror(buffer,kMAXBUFLEN);
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
         if (dir && *dir == '/')
            SPrintf(buffer, kMAXBUFLEN, "%s", dir);
      }
      NetSend(buffer, kROOTD_CHDIR);
   }
}

//______________________________________________________________________________
void RootdAccess(const char *buf)
{
   // Test access permission on path

   char buffer[kMAXPATHLEN];
   char path[kMAXPATHLEN];
   int mode;

   int nw = 0;
   if (buf)
      nw = sscanf(buf,"%s %d",path,&mode);

   if (nw >= 2) {

      char *epath = &path[0];
      if (path[0] == '/' && path[1] == '/')
         epath = &path[1];

      if (access(epath, mode) == -1) {
         SPrintf(buffer,kMAXPATHLEN,"cannot stat %s",epath);
         Perror(buffer);
         ErrorInfo("RootdAccess: %s", buffer);
      } else
         SPrintf(buffer,kMAXPATHLEN,"OK");

   } else {
      SPrintf(buffer,kMAXPATHLEN,"bad input format %s",buf);
      ErrorInfo("RootdAccess: %s", buffer);
   }

   NetSend(buffer, kROOTD_ACCESS);
}


//______________________________________________________________________________
void RootdFreeDir()
{
   // Free open directory.

   char buffer[kMAXPATHLEN];

   if (!gRDDirectory) {
      SPrintf(buffer,kMAXPATHLEN,"no directory open");
      ErrorInfo("RootdFreeDir: %s", buffer);
   } else if (closedir(gRDDirectory) == -1) {
      SPrintf(buffer,kMAXPATHLEN,"cannot free open directory");
      Perror(buffer);
      ErrorInfo("RootdFreeDir: %s", buffer);
   } else
      SPrintf(buffer,kMAXPATHLEN,"open directory freed");

   NetSend(buffer, kROOTD_FREEDIR);
}

//______________________________________________________________________________
void RootdGetDirEntry()
{
   // Get directory entry.

   char buffer[kMAXPATHLEN];
   struct dirent *dp = 0;

   if (!gRDDirectory) {
      SPrintf(buffer,kMAXPATHLEN,"no directory open");
      ErrorInfo("RootdGetDirEntry: %s", buffer);
   } else if ((dp = readdir(gRDDirectory)) == 0) {
      if (GetErrno() == EBADF) {
         SPrintf(buffer,kMAXPATHLEN,"cannot read open directory");
         Perror(buffer);
         ErrorInfo("RootdGetDirEntry: %s", buffer);
      } else
         SPrintf(buffer,kMAXPATHLEN,"no more entries");
   } else {
      SPrintf(buffer,kMAXPATHLEN,"OK:%s",dp->d_name);
   }

   NetSend(buffer, kROOTD_DIRENTRY);
}

//______________________________________________________________________________
void RootdOpenDir(const char *dir)
{
   // Open directory.

   char buffer[kMAXPATHLEN];

   char *edir = (char *)dir;
   if (dir[0] == '/' && dir[1] == '/')
      edir++;

   if ((gRDDirectory = opendir(edir)) == 0) {
      SPrintf(buffer,kMAXPATHLEN,"cannot open directory %s",edir);
      Perror(buffer);
      ErrorInfo("RootdOpenDir: %s", buffer);
   } else
      SPrintf(buffer,kMAXPATHLEN,"OK: directory %s open",edir);

   NetSend(buffer, kROOTD_OPENDIR);
}

//______________________________________________________________________________
void RootdMkdir(const char *fdir)
{
   // Make directory.

   char buffer[kMAXPATHLEN];

   char *dir = (char *)fdir;
   if (fdir[0] == '/' && fdir[1] == '/')
      dir++;

   if (gAnon) {
      SPrintf(buffer,kMAXPATHLEN,
              "anonymous users may not create directories");
      ErrorInfo("RootdMkdir: %s", buffer);
   } else if (mkdir(dir, 0755) < 0) {
      SPrintf(buffer,kMAXPATHLEN,"cannot create directory %s",dir);
      Perror(buffer);
      ErrorInfo("RootdMkdir: %s", buffer);
   } else
      SPrintf(buffer,kMAXPATHLEN,"OK: created directory %s",dir);

   NetSend(buffer, kROOTD_MKDIR);
}

//______________________________________________________________________________
void RootdRmdir(const char *fdir)
{
   // Delete directory.

   char buffer[kMAXPATHLEN];

   char *dir = (char *)fdir;
   if (fdir[0] == '/' && fdir[1] == '/')
      dir++;

   if (gAnon) {
      SPrintf(buffer,kMAXPATHLEN,
              "anonymous users may not delete directories");
      ErrorInfo("RootdRmdir: %s", buffer);
   } else if (rmdir(dir) < 0) {
      SPrintf(buffer, kMAXPATHLEN, "cannot delete directory %s", dir);
      Perror(buffer);
      ErrorInfo("RootdRmdir: %s", buffer);
   } else
      SPrintf(buffer, kMAXPATHLEN, "deleted directory %s", dir);

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
         SPrintf(buffer, kMAXPATHLEN, "ls %s", cmd);
      else
         SPrintf(buffer, kMAXPATHLEN, "%s", cmd);
   } else {
      if (strlen(cmd) < 2 || strncmp(cmd, "ls", 2))
         SPrintf(buffer, kMAXPATHLEN, "ls %s 2>/dev/null", cmd);
      else
         SPrintf(buffer, kMAXPATHLEN, "%s 2>/dev/null", cmd);
   }

   FILE *pf;
   if ((pf = popen(buffer, "r")) == 0) {
      SPrintf(buffer,kMAXPATHLEN, "error in popen");
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
      SPrintf(buffer, kMAXPATHLEN, "current directory not readable");
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
      SPrintf(buffer, kMAXPATHLEN, "anonymous users may not rename files");
      ErrorInfo("RootdMv: %s", buffer);
   } else if (rename(file1, file2) < 0) {
      SPrintf(buffer, kMAXPATHLEN, "cannot rename file %s to %s",
              file1, file2);
      Perror(buffer);
      ErrorInfo("RootdMv: %s", buffer);
   } else
      SPrintf(buffer, kMAXPATHLEN, "renamed file %s to %s",
              file1, file2);

   NetSend(buffer, kROOTD_MV);
}

//______________________________________________________________________________
void RootdRm(const char *file)
{
   // Delete a file.

   char buffer[kMAXPATHLEN];

   if (gAnon) {
      SPrintf(buffer, kMAXPATHLEN, "anonymous users may not delete files");
      ErrorInfo("RootdRm: %s", buffer);
   } else if (unlink(file) < 0) {
      SPrintf(buffer, kMAXPATHLEN, "cannot unlink file %s", file);
      Perror(buffer);
      ErrorInfo("RootdRm: %s", buffer);
   } else
      SPrintf(buffer, kMAXPATHLEN, "removed file %s", file);

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
      SPrintf(buffer, kMAXPATHLEN,
              "anonymous users may not change file permissions");
      ErrorInfo("RootdChmod: %s", buffer);
   } else if (chmod(file, mode) < 0) {
      SPrintf(buffer, kMAXPATHLEN, "cannot chmod file %s to 0%o", file, mode);
      Perror(buffer);
      ErrorInfo("RootdChmod: %s", buffer);
   } else
      SPrintf(buffer, kMAXPATHLEN, "changed permission of file %s to 0%o",
              file, mode);

   NetSend(buffer, kROOTD_CHMOD);
}

//______________________________________________________________________________
static void RootdTerm(int)
{
   // Termination upon receipt of a SIGTERM or SIGINT.

   ErrorInfo("RootdTerm: rootd.cxx: got a SIGTERM/SIGINT");
   // Terminate properly
   RpdAuthCleanup(0,0);
   // Close network connection
   NetClose();
   // exit
   exit(0);
}

//______________________________________________________________________________
void RootdLoop()
{
   // Handle all rootd commands. Returns after file close command.

   char recvbuf[kMAXRECVBUF];
   EMessageTypes kind;

//#define R__ROOTDDBG
#ifdef R__ROOTDDBG
   int debug = 1;
   while (debug)
      ;
#endif

   // Check if we will go for parallel sockets
   // (in early days was done before entering main loop)
   if (gClientProtocol > 9)
      RootdParallel();

   // Main loop
   while (1) {

      if (NetRecv(recvbuf, kMAXRECVBUF, kind) < 0)
         Error(ErrFatal, kErrFatal, "RootdLoop: error receiving message");

      if (gDebug > 2 && kind != kROOTD_PASS)
         ErrorInfo("RootdLoop: kind:%d -- buf:'%s' (len:%d)",
                   kind, recvbuf, strlen(recvbuf));

      switch (kind) {
         case kROOTD_OPEN:
            RootdOpen(recvbuf);
            break;
         case kROOTD_PUT:
            RootdPut(recvbuf);
            break;
         case kROOTD_GET:
            RootdGet(recvbuf);
            break;
         case kROOTD_GETS:
            RootdGets(recvbuf);
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
            RootdFstat(recvbuf);
            break;
         case kROOTD_STAT:
            RootdStat();
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
         case kROOTD_OPENDIR:
            RootdOpenDir(recvbuf);
            break;
         case kROOTD_FREEDIR:
            RootdFreeDir();
            break;
         case kROOTD_DIRENTRY:
            RootdGetDirEntry();
            break;
         case kROOTD_ACCESS:
            RootdAccess(recvbuf);
            break;
         case kROOTD_BYE:
            return;
         default:
            Error(ErrFatal, kErrBadOp, "RootdLoop: received bad opcode %d", kind);
      }
      continue;
   }
}

//______________________________________________________________________________
void Usage(const char* name, int rc)
{
   fprintf(stderr, "\nUsage: %s [options] [rootsys-dir]\n", name);
   fprintf(stderr, "\nOptions:\n");
   fprintf(stderr, "\t-b tcpwindowsize  Specify the tcp window size in bytes\n");
#ifdef R__GLBS
   fprintf(stderr, "\t-C hostcertfile   Specify the location of the Globus host certificate\n");
#endif
   fprintf(stderr, "\t-d level          set debug level [0..3]\n");
   fprintf(stderr, "\t-D rootdaemonrc   Use alternate rootdaemonrc file\n");
   fprintf(stderr, "\t                  (see documentation)\n");
   fprintf(stderr, "\t-E                Ignored for backward compatibility\n");
   fprintf(stderr, "\t-f                Run in foreground\n");
#ifdef R__GLBS
   fprintf(stderr, "\t-G gridmapfile    Specify the location of th Globus gridmap\n");
#endif
   fprintf(stderr, "\t-i                Running from inetd\n");
   fprintf(stderr, "\t-noauth           Do not require client authentication\n");
   fprintf(stderr, "\t-p port#          Specify a different port to listen on\n");
   fprintf(stderr, "\t-P pwfile         Use pwfile instead of .srootdpass\n");
   fprintf(stderr, "\t-r                Files can only be opened in read-only mode\n");
   fprintf(stderr, "\t-R bitmask        Bitmask specifies which methods allow authentication re-use\n");
   fprintf(stderr, "\t-s sshd_port#     Specify the port for the sshd daemon\n");
#ifdef R__KRB5
   fprintf(stderr, "\t-S keytabfile     Use an alternate keytab file\n");
#endif
   fprintf(stderr, "\t-T <tmpdir>       Use an alternate temp dir\n");
   fprintf(stderr, "\t-w                Do not check /etc/hosts.equiv and $HOME/.rhosts\n");

   exit(rc);
}

//______________________________________________________________________________
int main(int argc, char **argv)
{
   char *s;
   int checkhostsequiv= 1;
   int requireauth    = 1;
   int tcpwindowsize  = 65535;
   int sshdport       = 22;
   int port1          = 0;
   int port2          = 0;
   int reuseallow     = 0x1F;
   int login          = 2; // form rootd we fully login users, by default
   int foregroundflag = 0;
   std::string tmpdir = "";
   std::string confdir = "";
   std::string rootbindir = "";
   std::string altSRPpass = "";
   std::string daemonrc = "";
   std::string rootetcdir = "";
#ifdef R__GLBS
   std::string gridmap = "";
   std::string hostcertconf = "";
#endif
   char *progname = argv[0];

   // Init error handlers
   RpdSetErrorHandler(Err, ErrSys, ErrFatal);

   // function for dealing with SIGPIPE signals
   // (used by NetSetOptions() in rpdutils/net.cxx)
   NetSetSigPipeHook(SigPipe);

   // Init syslog
   ErrorInit(argv[0]);

   // Output to syslog ...
   RpdSetSysLogFlag(1);

   // ... unless we are running in the foreground and we are
   // attached to terminal; make also sure that "-i" and "-f"
   // are not simultaneously specified
   int i = 1;
   for (i = 1; i < argc; i++) {
      if (!strncmp(argv[i],"-f",2))
         foregroundflag = 1;
      if (!strncmp(argv[i],"-i",2))
         gInetdFlag = 1;
   }
   if (foregroundflag) {
      if (isatty(0) && isatty(1)) {
         RpdSetSysLogFlag(0);
         ErrorInfo("main: running in foreground mode:"
                   " sending output to stderr");
      }
      if (gInetdFlag)
         Error(ErrFatal,-1,"-i and -f options are incompatible");
   }

   // To terminate correctly ... maybe not needed
   signal(SIGTERM, RootdTerm);
   signal(SIGINT, RootdTerm);

   char *tmp = RootdExpandPathName(argv[0]);
   if (tmp) {
      int p = strlen(tmp)-1;
      while ((p+1) && tmp[p] != '/')
         p--;
      if (p+1) {
         tmp[p] = '\0';
         rootbindir = std::string(tmp);
         while ((p+1) && tmp[p] != '/')
            p--;
         if (p+1) {
            tmp[p] = '\0';
            confdir = std::string(tmp);
         }
      }
      free(tmp);
   }

   while (--argc > 0 && (*++argv)[0] == '-')
      for (s = argv[0]+1; *s != 0; s++)
         switch (*s) {

            case 'b':
               if (--argc <= 0) {
                  Error(ErrFatal,kErrFatal,"-b requires a buffersize in bytes"
                                    " as argument");
               }
               tcpwindowsize = atoi(*++argv);
               break;
#ifdef R__GLBS
            case 'C':
               if (--argc <= 0) {
                  Error(ErrFatal, kErrFatal,"-C requires a file name for"
                                    " the host certificates file location");
               }
               hostcertconf = std::string(*++argv);
               break;
#endif
            case 'd':
               if (--argc <= 0) {
                  Error(ErrFatal,kErrFatal,"-d requires a debug level as"
                                    " argument");
               }
               gDebug = atoi(*++argv);
               break;

            case 'D':
               if (--argc <= 0) {
                  Error(ErrFatal, kErrFatal,"-D requires a file path name"
                                    "  for the file defining access rules");
               }
               daemonrc = std::string(*++argv);
               break;

            case 'E':
               Error(ErrFatal, kErrFatal,"Option '-E' is now dummy "
                          "- ignored (see proofd/src/proofd.cxx for"
                          " additional details)");
               break;

            case 'f':
               if (gInetdFlag) {
                  Error(ErrFatal,-1,"-i and -f options are incompatible");
               }
               foregroundflag = 1;
               break;

            case 'F':
               gCastorFlag = 1;
               gInetdFlag  = 1;
               reuseallow = 0x0; // No auth reuse for castor
               login = 1; // No full logins for castor (user $HOMEs may not exist on servers)
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr,"-F requires a file path name for the"
                             " CASTOR disk file to be accessed\n");
                  Error(ErrFatal, kErrFatal,"-F requires a file path name"
                        " for the CASTOR disk file to be accessed");
               }
               gCastorFile = std::string(*++argv);
               break;

#ifdef R__GLBS
            case 'G':
               if (--argc <= 0) {
                  Error(ErrFatal,kErrFatal,"-G requires a file name for"
                                    " the gridmap file");
               }
               gridmap = std::string(*++argv);
               break;
#endif
            case 'h':
               Usage(progname, 0);
               break;

            case 'H':
               if (--argc <= 0) {
                  if (!gInetdFlag && !gCastorFlag)
                     fprintf(stderr,"-H requires the CASTOR request ID");
                  Error(ErrFatal, kErrFatal,"-H requires the CASTOR request ID");
               }
               gCastorReqId = std::string(*++argv);
               break;

            case 'i':
               if (foregroundflag) {
                  Error(ErrFatal,-1,"-i and -f options are incompatible");
               }
               gInetdFlag = 1;
               break;

            case 'n':
               if (!strncmp(argv[0]+1,"noauth",6)) {
                  requireauth = 0;
                  s += 5;
               } else if (!strncmp(argv[0]+1,"nologin",7)) {
                  login = 0;
                  s += 6;
               }
               break;

            case 'p':
               if (--argc <= 0) {
                  Error(ErrFatal,kErrFatal,"-p requires a port number as"
                                    " argument");
               }
               char *p;
               port1 = strtol(*++argv, &p, 10);
               if (*p == '-') {
                  p++;
                  port2 = strtol(p, &p, 10);
               } else if (*p == '\0')
                  port2 = port1;
               if (*p != '\0' || port2 < port1 || port2 < 0) {
                  Error(ErrFatal,kErrFatal,"invalid port number or range: %s",
                                     *argv);
               }
               break;

            case 'P':
               if (--argc <= 0) {
                  Error(ErrFatal,kErrFatal,"-P requires a file name for SRP"
                                    " password file");
               }
               altSRPpass = std::string(*++argv);
               break;

            case 'r':
               gReadOnly = 1;
               break;

            case 'R':
               if (--argc <= 0) {
                  Error(ErrFatal,kErrFatal,"-R requires a hex but mask as"
                                    " argument");
               }
               reuseallow = strtol(*++argv, (char **)0, 16);
               break;

            case 's':
               if (--argc <= 0) {
                  Error(ErrFatal,kErrFatal,"-s requires as argument a port"
                                    " number for the sshd daemon");
               }
               sshdport = atoi(*++argv);
               break;
#ifdef R__KRB5
            case 'S':
               if (--argc <= 0) {
                  Error(ErrFatal,kErrFatal,"-S requires a path to your"
                                    " keytab\n");
               }
               RpdSetKeytabFile((const char *)(*++argv));
               break;
#endif
            case 'T':
               if (--argc <= 0) {
                  Error(ErrFatal, kErrFatal,"-T requires a dir path for"
                                    " temporary files [/usr/tmp]");
               }
               tmpdir = std::string(*++argv);
               break;

            case 'w':
               checkhostsequiv = 0;
               break;

            default:
               if (!foregroundflag) fprintf(stderr, "\nUnknown command line option: %c\n", *s);
               Error(0, -1, "unknown command line option: %c", *s);
               Usage(progname, 1);
         }

   // dir for temporary files
   if (!tmpdir.length())
      tmpdir = std::string("/usr/tmp");
   if (access(tmpdir.c_str(), W_OK) == -1)
      tmpdir = std::string("/tmp");

   // root tab file
   gRootdTab = std::string(tmpdir).append("/rootdtab");

   if (argc > 0) {
      confdir = std::string(*argv);
   } else {
      // try to guess the config directory...
#ifndef ROOTPREFIX
      if (!confdir.length()) {
         if (getenv("ROOTSYS")) {
            confdir = getenv("ROOTSYS");
            if (gDebug > 0)
               ErrorInfo("main: no config directory specified using"
                         " ROOTSYS (%s)", confdir.c_str());
         } else {
            if (gDebug > 0)
               ErrorInfo("main: no config directory specified");
         }
      }
#else
      confdir = ROOTPREFIX;
#endif
   }
#ifdef ROOTBINDIR
   rootbindir= ROOTBINDIR;
#endif
#ifdef ROOTETCDIR
   rootetcdir= ROOTETCDIR;
#endif

   // Define rootbindir if not done already
   if (!rootbindir.length())
      rootbindir = std::string(confdir).append("/bin");
   // Make it available to all the session via env
   if (rootbindir.length()) {
      char *tmp1 = new char[15 + rootbindir.length()];
      sprintf(tmp1, "ROOTBINDIR=%s", rootbindir.c_str());
      putenv(tmp1);
   }

   // Define rootetcdir if not done already
   if (!rootetcdir.length())
      rootetcdir = std::string(confdir).append("/etc");
   // Make it available to all the session via env
   if (rootetcdir.length()) {
      char *tmp1 = new char[15 + rootetcdir.length()];
      sprintf(tmp1, "ROOTETCDIR=%s", rootetcdir.c_str());
      putenv(tmp1);
   }

   // If specified, set the special daemonrc file to be used
   if (daemonrc.length()) {
      char *tmp1 = new char[15+daemonrc.length()];
      sprintf(tmp1, "ROOTDAEMONRC=%s", daemonrc.c_str());
      putenv(tmp1);
   }
#ifdef R__GLBS
   // If specified, set the special gridmap file to be used
   if (gridmap.length()) {
      char *tmp1 = new char[15+gridmap.length()];
      sprintf(tmp1, "GRIDMAP=%s", gridmap.c_str());
      putenv(tmp1);
   }
   // If specified, set the special hostcert.conf file to be used
   if (hostcertconf.length()) {
      char *tmp1 = new char[15+hostcertconf.length()];
      sprintf(tmp1, "ROOTHOSTCERT=%s", hostcertconf.c_str());
      putenv(tmp1);
   }
#endif

   // Parent ID
   int rootdparentid = -1;      // Parent process ID
   if (!gInetdFlag)
      rootdparentid = getpid(); // Identifies this family
   else
      rootdparentid = getppid(); // Identifies this family

   // default job options
   unsigned int options = kDMN_RQAUTH | kDMN_HOSTEQ | kDMN_SYSLOG;
   // modify them if required
   if (!requireauth)
      options &= ~kDMN_RQAUTH;
   if (!checkhostsequiv)
      options &= ~kDMN_HOSTEQ;
   if (foregroundflag)
      options &= ~kDMN_SYSLOG;
   RpdInit(gService, rootdparentid, gProtocol, options,
           reuseallow, sshdport,
           tmpdir.c_str(),altSRPpass.c_str(),login);

   // Generate Local RSA keys for the session
   if (RpdGenRSAKeys(0)) {
      Error(Err, -1, "rootd: unable to generate local RSA keys");
   }

   if (!gInetdFlag) {

      // Start rootd up as a daemon process (in the background).
      // Also initialize the network connection - create the socket
      // and bind our well-know address to it.

      int fdkeep = NetInit(gService, port1, port2, tcpwindowsize);
      if (!foregroundflag)
         DaemonStart(1, fdkeep, gService);
   }

   if (gDebug > 0)
      ErrorInfo("main: pid = %d, ppid = %d, gInetdFlag = %d, gProtocol = %d",
                getpid(), getppid(), gInetdFlag, gProtocol);

   // Concurrent server loop.
   // The child created by NetOpen() handles the client's request.
   // The parent waits for another request. In the inetd case,
   // the parent from NetOpen() never returns.

   while (1) {

      if (NetOpen(gInetdFlag, gService) == 0) {

         // Init Session (get protocol, run authentication, login, ...)
         int rci = RpdInitSession(gService, gUser,
                                  gClientProtocol, gAnon, gPasswd);
         if (rci == -1)
            Error(ErrFatal, -1, "rootd: failure initializing session");
         else if (rci == -2)
            // Special session (eg. cleanup): just exit
            exit(0);

         ErrorInfo("main: rootdparentid = %d (%d)", rootdparentid, getppid());

         // RootdParallel is called after authentication in RootdLogin
         RootdLoop();      // child processes client's requests
         NetClose();       // till we are done
         exit(0);
      }

      // parent waits for another client to connect
      // (except in CASTOR mode)
      if (gCastorFlag) break;

   }

}
