// @(#)root/rootd:$Name:  $:$Id: rootd.cxx,v 1.7 2000/09/13 10:38:15 rdm Exp $
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
// rootd stream tcp nowait root /user/rdm/root/bin/rootd rootd -i       //
//                                                                      //
// Force inetd to reread its conf file with "kill -HUP <pid inetd>".    //
// You can also start rootd by hand running directly under your private //
// account (no root system priviliges needed). For example to start     //
// rootd listening on port 5151 just type:                              //
//                                                                      //
// rootd -p 5151                                                        //
//                                                                      //
// Notice: no & is needed. Rootd will go in background by itself.       //
//                                                                      //
// Rootd arguments:                                                     //
//   -i                says we were started by inetd                    //
//   -p port#          specifies a different port to listen on          //
//   -d level          level of debug info written to syslog            //
//                     0 = no debug (default)                           //
//                     1 = minimum                                      //
//                     2 = medium                                       //
//                     3 = maximum                                      //
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
//  When you system uses shadow passwords you have to compile rootd     //
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
//  SRP is described at: http://jafar.stanford.edu/srp/index.html.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <ctype.h>
#include <fcntl.h>
#include <pwd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

#if defined(__linux)
#   include <features.h>
#   if __GNU_LIBRARY__ == 6
#      ifndef R__GLIBC
#         define R__GLIBC
#      endif
#   endif
#endif

#if defined(__FreeBSD__)
#include <sys/file.h>
#define lockf(fd, op, sz)   flock((fd), (op))
#define	F_LOCK             (LOCK_EX | LOCK_NB)
#define	F_ULOCK             LOCK_UN
#endif

#if defined(__linux) || defined(__linux__) || defined(__sun) || defined(__sgi) || \
    defined(_AIX) || defined(__FreeBSD__)
#include <grp.h>
#include <sys/types.h>
#endif

#if defined(__sun) || defined(R__GLIBC)
#include <crypt.h>
#endif

#if defined(__osf__) || defined(__sgi)
extern "C" char *crypt(const char *, const char *);
#endif

#if defined(__alpha) && !defined(__linux) && !defined(__FreeBSD__)
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
   int initgroups(const char *name, int basegid);
   int seteuid(uid_t euid);
   int setegid(gid_t egid);
}
#endif

#if defined(__sun)
#ifndef R__SHADOWPW
#define R__SHADOWPW
#endif
#endif

#ifdef R__SHADOWPW
#include <shadow.h>
#endif

#ifdef R__AFS
//#include <afs/kautils.h>
#define KA_USERAUTH_VERSION 1
#define KA_USERAUTH_DOSETPAG 0x10000
#define NOPAG  0xffffffff
extern "C" int ka_UserAuthenticateGeneral(int,char*,char*,char*,char*,int,int,int,char**);
#endif

#ifdef R__SRP
extern "C" {
#include <t_pwd.h>
#include <t_server.h>
}
#endif

#include "rootdp.h"


//--- Globals ------------------------------------------------------------------

const char kRootdService[] = "rootd";
const char kRootdTab[]     = "/usr/tmp/rootdtab";
const char kRootdPass[]    = ".rootdpass";
const char kSRootdPass[]   = ".srootdpass";
const int  kMAXPATHLEN     = 1024;

int     gInetdFlag         = 0;
int     gPort              = 0;
int     gDebug             = 0;
int     gSockFd            = -1;
int     gAuth              = 0;
int     gAnon              = 0;
int     gFd                = -1;
int     gWritable          = 0;
int     gProtocol          = 2;       // increase when protocol changes
double  gBytesRead         = 0;
double  gBytesWritten      = 0;
char    gUser[64]          = { 0 };
char    gPasswd[64]        = { 0 };   // only used for anonymous access
char    gOption[32]        = { 0 };
char    gFile[kMAXPATHLEN] = { 0 };

//--- Machine specific routines ------------------------------------------------

#if !defined(__hpux)
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
#endif


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
            ErrorSys(kErrFatal, "RootdExpandPathName: no home directory");
            return 0;
         }
         strcat(cmd, hd);
         strcat(cmd, &escPatbuf[1]);
      }
   } else
      strcat(cmd, escPatbuf);

   FILE *pf;
   if ((pf = ::popen(&cmd[0], "r")) == 0) {
      ErrorSys(kErrFatal, "RootdExpandPathName: error in popen(%s)", cmd);
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

   // skip rest of pipe
   while (ch != EOF) {
      ch = fgetc(pf);
      if (ch == ' ' || ch == '\t') {
         ::pclose(pf);
         ErrorFatal(kErrFatal, "RootdExpandPathName: expression ambigous");
         return 0;
      }
   }

   ::pclose(pf);

   return strdup(expPatbuf);
}

//______________________________________________________________________________
int RootdCheckTab(int mode)
{
   // Checks kRootdTab file to see if file can be opened. If mode=1 then
   // check if file can safely be opened in write mode, i.e. see if file
   // is not already opened in either read or write mode. If mode=0 then
   // check if file can safely be opened in read mode, i.e. see if file
   // is not already opened in write mode. Returns 1 if file can be
   // opened safely, otherwise 0. If mode is -1 check write mode like 1
   // but do not update rootdtab file.
   //
   // The format of the file is:
   // filename inode mode username pid
   // where inode is the unique file ref number, mode is either "read"
   // or "write", username the user who has the file open and pid is the
   // pid of the rootd having the file open.

   // Open rootdtab file. Try first /usr/tmp and then /tmp.
   // The lockf() call can fail if the directory is NFS mounted
   // and the lockd daemon is not running.

   const char *sfile = kRootdTab;
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
         sfile = kRootdTab+4;
         goto again;
      }
      ErrorSys(kErrFatal, "RootdCheckTab: error opening %s", sfile);
   }

   // lock the file
   if (lockf(fid, F_LOCK, (off_t)1) == -1) {
      if (sfile[1] == 'u' && create) {
         close(fid);
         remove(sfile);
         sfile = kRootdTab+4;
         goto again;
      }
      ErrorSys(kErrFatal, "RootdCheckTab: error locking %s", sfile);
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
      char *fbuf = new char[siz+1];

      while (read(fid, fbuf, siz) < 0 && GetErrno() == EINTR)
         ResetErrno();
      fbuf[siz] = 0;

      char *n, *s = fbuf;
      while ((n = strchr(s, '\n'))) {
         char user[64], gmode[32];
         int  pid;
         unsigned int ino;
         sscanf(s, "%s %u %s %s %d", msg, &ino, gmode, user, &pid);
         if (ino == inode) {
            if (mode == 1)
               result = 0;
            else if (!strcmp(gmode, "write"))
               result = 0;
            break;
         }
         s = n + 1;
      }
      delete [] fbuf;
   }

   if (result && !noupdate) {
      sprintf(msg, "%s %lu %s %s %d\n", gFile, inode, smode, gUser, getpid());
      write(fid, msg, strlen(msg));
   }

   // unlock the file
   lseek(fid, 0, SEEK_SET);
   if (lockf(fid, F_ULOCK, (off_t)1) == -1)
      ErrorSys(kErrFatal, "RootdCheckTab: error unlocking %s", sfile);
   if (gDebug > 2)
      ErrorInfo("RootdCheckTab: file %s unlocked", sfile);

   close(fid);

   return result;
}

//______________________________________________________________________________
void RootdCloseTab(int force=0)
{
   // Removes from the kRootdTab file the reference to gFile for the
   // current rootd. If force = 1, then remove all references for gFile
   // from the kRootdTab file. This might be necessary in case something
   // funny happened and the original reference was not correctly removed.

   const char *sfile = kRootdTab;
   int fid;

again:
   if (access(sfile, F_OK) == -1) {
      if (sfile[1] == 'u') {
         sfile = kRootdTab+4;
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
         int  pid;
         unsigned int ino;
         sscanf(s, "%s %u %s %s %d", msg, &ino, gmode, user, &pid);
         if ((!force && mypid == pid) ||
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
            if (!force) break;
         }
         s = n;
      }
      if (changed) {
         ftruncate(fid, 0);
         if (siz > 0) {
            lseek(fid, 0, SEEK_SET);
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
void RootdClose()
{
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
         ErrorSys(kErrFatal, "RootdFlush: error flushing file %s", gFile);
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
void RootdProtocol()
{
   // Return rootd protocol version id.

   NetSend(gProtocol, kROOTD_PROTOCOL);
}

//______________________________________________________________________________
void RootdUser(const char *user)
{
   // Check user id. If user id is not equal to rootd's effective uid, user
   // will not be allowed access, unless effective uid = 0 (i.e. root).

   if (!*user)
      ErrorFatal(kErrBadUser, "RootdUser: bad user name");

   ERootdErrors err = kErrNoUser;

   if (!strcmp(user, "anonymous") || !strcmp(user, "rootd")) {
      user  = "rootd";
      err   = kErrNoAnon;
      gAnon = 1;
   }

   struct passwd *pw;
   if ((pw = getpwnam(user)) == 0)
      ErrorFatal(err, "RootdUser: user %s unknown", user);

   // If server is not started as root and remote user is not same as the
   // one who started rootd then authetication is not ok.
   uid_t uid = getuid();
   if (uid && uid != pw->pw_uid)
      ErrorFatal(kErrBadUser, "RootdUser: remote user not same as effective user of rootd");

   strcpy(gUser, user);

   NetSend(gAuth, kROOTD_AUTH);
}

//______________________________________________________________________________
void RootdSRPUser(const char *user)
{
   // Use Secure Remote Password protocol.
   // Check user id in $HOME/.srootdpass file.

   if (!*user)
      ErrorFatal(kErrBadUser, "RootdUser: bad user name");

#ifdef R__SRP

   char srootdpass[kMAXPATHLEN], srootdconf[kMAXPATHLEN];

   struct passwd *pw = getpwnam(user);
   if (!pw)
      ErrorFatal(kErrNoUser, "RootdSRPUser: user %s unknown", user);

   // If server is not started as root and remote user is not same as the
   // one who started rootd then authetication is not ok.
   uid_t uid = getuid();
   if (uid && uid != pw->pw_uid)
      ErrorFatal(kErrBadUser, "RootdSRPUser: remote user not same as effective user of rootd");

   NetSend(gAuth, kROOTD_AUTH);

   strcpy(gUser, user);

   sprintf(srootdpass, "%s/%s", pw->pw_dir, kSRootdPass);
   sprintf(srootdconf, "%s/%s.conf", pw->pw_dir, kSRootdPass);

   FILE *fp1 = fopen(srootdpass, "r");
   if (!fp1) {
      NetSend(2, kROOTD_AUTH);
      ErrorInfo("RootdSRPUser: error opening %s", srootdpass);
      return;
   }
   FILE *fp2 = fopen(srootdconf, "r");
   if (!fp2) {
      NetSend(2, kROOTD_AUTH);
      ErrorInfo("RootdSRPUser: error opening %s", srootdconf);
      if (fp1) fclose(fp1);
      return;
   }

   struct t_pw *tpw = t_openpw(fp1);
   if (!tpw) {
      NetSend(2, kROOTD_AUTH);
      ErrorInfo("RootdSRPUser: unable to open password file %s", srootdpass);
      fclose(fp1);
      fclose(fp2);
      return;
   }

   struct t_conf *tcnf = t_openconf(fp2);
   if (!tcnf) {
      NetSend(2, kROOTD_AUTH);
      ErrorInfo("RootdSRPUser: unable to open configuration file %s", srootdconf);
      t_closepw(tpw);
      fclose(fp1);
      fclose(fp2);
      return;
   }

   struct t_server *ts = t_serveropenfromfiles(gUser, tpw, tcnf);
   if (!ts)
      ErrorFatal(kErrNoUser, "user %s not found SRP password file", gUser);

   if (tcnf) t_closeconf(tcnf);
   if (tpw)  t_closepw(tpw);
   if (fp2)  fclose(fp2);
   if (fp1)  fclose(fp1);

   char hexbuf[MAXHEXPARAMLEN];

   // send n to client
   NetSend(t_tob64(hexbuf, (char*)ts->n.data, ts->n.len), kROOTD_SRPN);
   // send g to client
   NetSend(t_tob64(hexbuf, (char*)ts->g.data, ts->g.len), kROOTD_SRPG);
   // send salt to client
   NetSend(t_tob64(hexbuf, (char*)ts->s.data, ts->s.len), kROOTD_SRPSALT);

   struct t_num *B = t_servergenexp(ts);

   // receive A from client
   EMessageTypes kind;
   if (NetRecv(hexbuf, MAXHEXPARAMLEN, kind) < 0)
      ErrorFatal(kErrFatal, "RootdSRPUser: error receiving A from client");
   if (kind != kROOTD_SRPA)
      ErrorFatal(kErrFatal, "RootdSRPUser: expected kROOTD_SRPA message");

   unsigned char buf[MAXPARAMLEN];
   struct t_num A;
   A.data = buf;
   A.len  = t_fromb64((char*)A.data, hexbuf);

   // send B to client
   NetSend(t_tob64(hexbuf, (char*)B->data, B->len), kROOTD_SRPB);

   t_servergetkey(ts, &A);

   // receive response from client
   if (NetRecv(hexbuf, MAXHEXPARAMLEN, kind) < 0)
      ErrorFatal(kErrFatal, "RootdSRPUser: error receiving response from client");
   if (kind != kROOTD_SRPRESPONSE)
      ErrorFatal(kErrFatal, "RootdSRPUser: expected kROOTD_SRPRESPONSE message");

   unsigned char cbuf[20];
   t_fromhex((char*)cbuf, hexbuf);

   if (!t_serververify(ts, cbuf)) {
      // authentication successful

      gAuth = 1;

      if (chdir(pw->pw_dir) == -1)
         ErrorFatal(kErrFatal, "RootdSRPUser: can't change directory to %s", pw->pw_dir);

      if (getuid() == 0) {

         // set access control list from /etc/initgroup
         initgroups(gUser, pw->pw_gid);

         if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1)
            ErrorFatal(kErrFatal, "RootdSRPUser: can't setgid for user %s", gUser);

         if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1)
            ErrorFatal(kErrFatal, "RootdSRPUser: can't setuid for user %s", gUser);

      }

      umask(022);

      NetSend(gAuth, kROOTD_AUTH);

      if (gDebug > 0)
         ErrorInfo("RootdSRPUser: user %s authenticated", gUser);

   } else
      ErrorFatal(kErrBadPasswd, "RootdSRPUser: authentication failed for user %s", gUser);

   t_serverclose(ts);

#else
   NetSend(2, kROOTD_AUTH);
#endif
}

//______________________________________________________________________________
int RootdCheckSpecialPass(const char *passwd)
{
   // Check user's password against password in $HOME/.rootdpass. If matches
   // skip other authentication mechanism. Returns 1 in case of success
   // authentication, 0 otherwise.

   char rootdpass[kMAXPATHLEN];

   struct passwd *pw = getpwnam(gUser);

   sprintf(rootdpass, "%s/%s", pw->pw_dir, kRootdPass);

   int fid = open(rootdpass, O_RDONLY);
   if (fid == -1)
      return 0;

   int n;
   if ((n = read(fid, rootdpass, sizeof(rootdpass)-1)) <= 0) {
      close(fid);
      return 0;
   }
   close(fid);

   rootdpass[n] = 0;
   char *s = strchr(rootdpass, '\n');
   if (s) *s = 0;

   char *pass_crypt = crypt(passwd, rootdpass);
   n = strlen(rootdpass);

   if (strncmp(pass_crypt, rootdpass, n+1) != 0)
      return 0;

   if (gDebug > 0)
      ErrorInfo("RootdCheckSpecialPass: user %s authenticated via ~/.rootdpass", gUser);

   return 1;
}

//______________________________________________________________________________
void RootdPass(const char *pass)
{
   // Check user's password, if ok, change to user's id and to user's directory.

   char   passwd[64];
   char  *passw;
   char  *pass_crypt;
   struct passwd *pw;
#ifdef R__SHADOWPW
   struct spwd *spw;
#endif
#ifdef R__AFS
   char  *reason;
   int    afs_auth = 0;
#endif

   if (!*gUser)
      ErrorFatal(kErrFatal, "RootdPass: user needs to be specified first");

   int i;
   int n = strlen(pass);

   if (!n)
      ErrorFatal(kErrBadPasswd, "RootdPass: null passwd not allowed");

   if (n > (int)sizeof(passwd))
      ErrorFatal(kErrBadPasswd, "RootdPass: passwd too long");

   for (i = 0; i < n; i++)
      passwd[i] = ~pass[i];
   passwd[i] = '\0';

   pw = getpwnam(gUser);

   if (gAnon) {
      strcpy(gPasswd, passwd);
      goto skipauth;
   }

   if (RootdCheckSpecialPass(passwd))
      goto skipauth;

#ifdef R__AFS
   afs_auth = !ka_UserAuthenticateGeneral(
        KA_USERAUTH_VERSION + KA_USERAUTH_DOSETPAG,
        gUser,             //user name
        (char *) 0,        //instance
        (char *) 0,        //realm
        passwd,            //password
        0,                 //default lifetime
        0, 0,              //two spares
        &reason);          //error string

   if (!afs_auth) {
      ErrorInfo("RootdPass: AFS login failed for user %s: %s", gUser, reason);
      // try conventional login...
#endif

#ifdef R__SHADOWPW
   // System V Rel 4 style shadow passwords
   if ((spw = getspnam(gUser)) == 0) {
      ErrorInfo("RootdPass: Shadow passwd not available for user %s", gUser);
      passw = pw->pw_passwd;
   } else
      passw = spw->sp_pwdp;
#else
   passw = pw->pw_passwd;
#endif
   pass_crypt = crypt(passwd, passw);
   n = strlen(passw);

   if (strncmp(pass_crypt, passw, n+1) != 0)
      ErrorFatal(kErrBadPasswd, "RootdPass: invalid password for user %s", gUser);

#ifdef R__AFS
   }  // afs_auth
#endif

skipauth:
   gAuth = 1;

   if (chdir(pw->pw_dir) == -1)
      ErrorFatal(kErrFatal, "RootdPass: can't change directory to %s", pw->pw_dir);

   if (getuid() == 0) {

      if (gAnon && chroot(pw->pw_dir) == -1)
         ErrorFatal(kErrFatal, "RootdPass: can't chroot to %s", pw->pw_dir);

      // set access control list from /etc/initgroup
      initgroups(gUser, pw->pw_gid);

      if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1)
         ErrorFatal(kErrFatal, "RootdPass: can't setgid for user %s", gUser);

      if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1)
         ErrorFatal(kErrFatal, "RootdPass: can't setuid for user %s", gUser);

   }

   umask(022);

   NetSend(gAuth, kROOTD_AUTH);

   if (gDebug > 0) {
      if (gAnon)
         ErrorInfo("RootdPass: user %s/%s authenticated", gUser, gPasswd);
      else
         ErrorInfo("RootdPass: user %s authenticated", gUser);
   }
}

//______________________________________________________________________________
void RootdOpen(const char *msg)
{
   // Open file in mode depending on specified option. If file is already
   // opened by another rootd in write mode, do not open the file.

   char file[kMAXPATHLEN], option[32];

   sscanf(msg, "%s %s", file, option);

   if (file[0] == '/')
      strcpy(gFile, &file[1]);
   else
      strcpy(gFile, file);

   int  forceOpen = 0;
   if (option[0] == 'f') {
      forceOpen = 1;
      strcpy(gOption, &option[1]);
   } else
      strcpy(gOption, option);

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

   if (!gAnon) {
      char *fname;
      if ((fname = RootdExpandPathName(gFile))) {
         strcpy(gFile, fname);
         free(fname);
      } else
         ErrorFatal(kErrBadFile, "RootdOpen: bad file name %s", gFile);
   }

   if (forceOpen)
      RootdCloseTab(1);

   if (recreate) {
      if (!RootdCheckTab(-1))
         ErrorFatal(kErrFileWriteOpen, "RootdOpen: file %s already opened in read or write mode", gFile);
      if (!access(gFile, F_OK))
         unlink(gFile);
      recreate = 0;
      create   = 1;
      strcpy(gOption, "create");
   }

   if (create && !access(gFile, F_OK))
      ErrorFatal(kErrFileExists, "RootdOpen: file %s already exists", gFile);

   if (update) {
      if (access(gFile, F_OK)) {
         update = 0;
         create = 1;
      }
      if (update && access(gFile, W_OK))
         ErrorFatal(kErrNoAccess, "RootdOpen: no write permission for file %s", gFile);
   }

   if (read) {
      if (access(gFile, F_OK))
         ErrorFatal(kErrNoFile, "RootdOpen: file %s does not exist", gFile);
      if (access(gFile, R_OK))
         ErrorFatal(kErrNoAccess, "RootdOpen: no read permission for file %s", gFile);
   }

   if (create || update) {
      if (create) {
         // make sure file exists so RootdCheckTab works correctly
#ifndef WIN32
         gFd = open(gFile, O_RDWR | O_CREAT, 0644);
#else
         gFd = open(gFile, O_RDWR | O_CREAT | O_BINARY, S_IREAD | S_IWRITE);
#endif
         close(gFd);
         gFd = -1;
      }
      if (!RootdCheckTab(1))
         ErrorFatal(kErrFileWriteOpen, "RootdOpen: file %s already opened in read or write mode", gFile);
#ifndef WIN32
      gFd = open(gFile, O_RDWR, 0644);
#else
      gFd = open(gFile, O_RDWR | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (gFd == -1)
         ErrorSys(kErrFileOpen, "RootdOpen: error opening file %s in write mode", gFile);

      gWritable = 1;

   } else {
      if (!RootdCheckTab(0))
         ErrorFatal(kErrFileReadOpen, "RootdOpen: file %s already opened in write mode", gFile);
#ifndef WIN32
      gFd = open(gFile, O_RDONLY);
#else
      gFd = open(gFile, O_RDONLY | O_BINARY);
#endif
      if (gFd == -1)
         ErrorSys(kErrFileOpen, "RootdOpen: error opening file %s in read mode", gFile);

      gWritable = 0;

   }

   NetSend(gWritable, kROOTD_OPEN);

   if (gDebug > 0)
      ErrorInfo("RootdOpen: file %s opened in mode %s", gFile, gOption);
   else {
      if (gAnon)
         ErrorInfo("Rootd: file %s (%s) opened by %s/%s", gFile, gOption,
                   gUser, gPasswd);
      else
         ErrorInfo("Rootd: file %s (%s) opened by %s", gFile, gOption, gUser);
   }
}

//______________________________________________________________________________
void RootdPut(const char *msg)
{
   int offset, len;

   sscanf(msg, "%d %d", &offset, &len);

   char *buf = new char[len];
   NetRecvRaw(buf, len);

   if (!RootdIsOpen() || !gWritable)
      ErrorFatal(kErrNoAccess, "RootdPut: file %s not opened in write mode", gFile);

   if (lseek(gFd, offset, SEEK_SET) < 0)
      ErrorSys(kErrFilePut, "RootdPut: cannot seek to position %d in file %s", offset, gFile);

   ssize_t siz;
   while ((siz = write(gFd, buf, len)) < 0 && GetErrno() == EINTR)
      ResetErrno();

   if (siz < 0)
      ErrorSys(kErrFilePut, "RootdPut: error writing to file %s", gFile);

   if (siz != len)
      ErrorFatal(kErrFilePut, "RootdPut: error writing all requested bytes to file %s, wrote %d of %d",
                 gFile, siz, len);

   NetSend(0, kROOTD_PUT);

   delete [] buf;

   gBytesWritten += len;

   if (gDebug > 0)
      ErrorInfo("RootdPut: written %d bytes starting at %d to file %s",
                len, offset, gFile);
}

//______________________________________________________________________________
void RootdGet(const char *msg)
{
   int offset, len;

   sscanf(msg, "%d %d", &offset, &len);

   char *buf = new char[len];

   if (!RootdIsOpen())
      ErrorFatal(kErrNoAccess, "RootdGet: file %s not open", gFile);

   if (lseek(gFd, offset, SEEK_SET) < 0)
      ErrorSys(kErrFileGet, "RootdGet: cannot seek to position %d in file %s", offset, gFile);

   ssize_t siz;
   while ((siz = read(gFd, buf, len)) < 0 && GetErrno() == EINTR)
      ResetErrno();

   if (siz < 0)
      ErrorSys(kErrFileGet, "RootdGet: error reading from file %s", gFile);

   if (siz != len)
      ErrorFatal(kErrFileGet, "RootdGet: error reading all requested bytes from file %s, got %d of %d",
                 gFile, siz, len);

   NetSend(0, kROOTD_PUT);

   NetSendRaw(buf, len);

   delete [] buf;

   gBytesRead += len;

   if (gDebug > 0)
      ErrorInfo("RootdGet: read %d bytes starting at %d from file %s",
                len, offset, gFile);
}

//______________________________________________________________________________
void RootdLoop()
{
   // Handle all rootd commands. Returns after file close command.

   const int kMaxBuf = 1024;
   char recvbuf[kMaxBuf];
   EMessageTypes kind;

   while (1) {
      if (NetRecv(recvbuf, kMaxBuf, kind) < 0)
         ErrorFatal(kErrFatal, "RootdLoop: error receiving message");

      if (kind != kROOTD_USER    && kind != kROOTD_PASS &&
          kind != kROOTD_SRPUSER && kind != kROOTD_PROTOCOL && gAuth == 0)
         ErrorFatal(kErrNoUser, "RootdLoop: not authenticated");

      if (gDebug > 2 && kind != kROOTD_PASS)
         ErrorInfo("RootdLoop: %d -- %s", kind, recvbuf);

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
            return;
         case kROOTD_STAT:
            RootdStat();
            break;
         case kROOTD_PROTOCOL:
            RootdProtocol();
            break;
         default:
            ErrorFatal(kErrBadOp, "RootdLoop: received bad opcode %d", kind);
      }
   }
}

//______________________________________________________________________________
int main(int argc, char **argv)
{
   int    childpid;
   char  *s;

   ErrorInit(argv[0]);

   while (--argc > 0 && (*++argv)[0] == '-')
      for (s = argv[0]+1; *s != 0; s++)
         switch (*s) {
            case 'i':
               gInetdFlag = 1;
               break;

            case 'p':
               if (--argc <= 0)
                  ErrorFatal(kErrFatal, "-p requires a port number as argument");
               gPort = atoi(*++argv);
               break;

            case 'd':
               if (--argc <= 0)
                  gDebug = 0;
               else
                  gDebug = atoi(*++argv);
               break;

            default:
               ErrorFatal(kErrFatal, "unknown command line option: %s", *s);
         }

   if (!gInetdFlag) {

      // Start rootd up as a daemon process (in the background).
      // Also initialize the network connection - create the socket
      // and bind our well-know address to it.

      DaemonStart(1);

      NetInit(kRootdService, gPort);
   }

   if (gDebug > 0)
      ErrorInfo("main: pid = %d, gInetdFlag = %d, gProtocol = %d",
                getpid(), gInetdFlag, gProtocol);

   // Concurrent server loop.
   // The child created by NetOpen() handles the client's request.
   // The parent waits for another request. In the inetd case,
   // the parent from NetOpen() never returns.

   while (1) {
      if ((childpid = NetOpen(gInetdFlag)) == 0) {
         NetSetOptions();  // set optimal socket options
         RootdLoop();      // child processes client's requests
         NetClose();       // then we are done
         exit(0);
      }

      // parent waits for another client to connect

   }

   // not reached
   return 0;
}
