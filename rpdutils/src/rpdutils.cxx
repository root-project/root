// @(#)root/rpdutils:$Name:  $:$Id: rpdutils.cxx,v 1.36 2004/04/20 21:32:02 brun Exp $
// Author: Gerardo Ganis    7/4/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// rpdutils                                                             //
//                                                                      //
// Set of utilities for rootd/proofd daemon authentication.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "config.h"

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
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <netdb.h>
#include <math.h>
#include "snprintf.h"

#if defined(__CYGWIN__) && defined(__GNUC__)
#   define cygwingcc
#endif

#if defined(linux) || defined(__sun) || defined(__sgi) || \
    defined(_AIX) || defined(__FreeBSD__) || defined(__APPLE__) || \
    defined(__MACH__) || defined(cygwingcc)
#include <grp.h>
#include <sys/types.h>
#include <signal.h>
#endif

#if defined(linux) || defined(__sun) || defined(__sgi) || \
    defined(_AIX) || defined(__FreeBSD__) || defined(__APPLE__) || \
    defined(__MACH__) || defined(cygwingcc)
#include <grp.h>
#include <sys/types.h>
#include <signal.h>
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
#elif defined(linux) || defined(__hpux)
#include <sys/vfs.h>
#elif defined(__FreeBSD__)
#include <sys/param.h>
#include <sys/mount.h>
#else
#include <sys/statfs.h>
#endif

#if defined(linux)
#   include <features.h>
#   if __GNU_LIBRARY__ == 6
#      ifndef R__GLIBC
#         define R__GLIBC
#      endif
#   endif
#endif
#if defined(cygwingcc) || (defined(__MACH__) && !defined(__APPLE__))
#   define R__GLIBC
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

#if defined(__sun) || defined(R__GLIBC)
#include <crypt.h>
#endif

#if defined(__osf__) || defined(__sgi) || defined(R__MACOSX)
extern "C" char *crypt(const char *, const char *);
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
extern "C" int ka_UserAuthenticateGeneral(int, char *, char *, char *,
                                          char *, int, int, int, char **);
#endif

#ifdef R__SRP
extern "C" {
   #include <t_pwd.h>
   #include <t_server.h>
}
#endif

#ifdef R__KRB5
#include "Krb5Auth.h"
#include <string>
extern krb5_deltat krb5_clockskew;
#endif

#include "rpdp.h"
extern "C" {
   #include "rsadef.h"
   #include "rsalib.h"
}

//--- Machine specific routines ------------------------------------------------

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
   int seteuid(uid_t euid);
   int setegid(gid_t egid);
}
#endif

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

extern int gDebug;

namespace ROOT {

//
// rpdutils module globals
ErrorHandler_t gErrSys   = 0;
ErrorHandler_t gErrFatal = 0;
ErrorHandler_t gErr      = 0;
int gRootLog = 0;
std::string gServName[3] = { "sockd", "rootd", "proofd" };

//
// Local global consts
static const int kAUTH_CLR_MSK = 0x1;     // Masks for authentication methods
static const int kAUTH_SRP_MSK = 0x2;
static const int kAUTH_KRB_MSK = 0x4;
static const int kAUTH_GLB_MSK = 0x8;
static const int kAUTH_SSH_MSK = 0x10;
static const int kMAXTABSIZE = 1000000000;
static const std::string kAuthMeth[kMAXSEC] = { "UsrPwd", "SRP", "Krb5",
                                                "Globus", "SSH", "UidGid" };
static const std::string kAuthTab    = "/rpdauthtab";   // auth table
static const std::string kDaemonRc   = ".rootdaemonrc"; // daemon access rules
static const std::string kRootdPass  = ".rootdpass";    // special rootd passwd
static const std::string kSRootdPass = "/.srootdpass";  // SRP passwd

//
// RW dir for temporary files (needed by gRpdAuthTab: do not move)
static std::string gTmpDir = "/tmp";

//
// Local global vars
static char gBufOld[kMAXRECVBUF] = {0}; // msg sync for old client (<=3.05/07)
static int gClientOld = 0;              // msg sync for old client (<=3.05/07)
static int gClientProtocol = -1;
static int gCryptRequired = -1;
static int gAllowMeth[kMAXSEC];
static std::string gAltSRPPass;
static int gAnon = 0;
static int gAuth = 0;
static int gAuthListSent = 0;
static int gHaveMeth[kMAXSEC];
static int gInclusiveToken = 0;
static EMessageTypes gKindOld;          // msg sync for old client (<=3.05/07)
static int gMethInit = 0;
static int gNumAllow = -1;
static int gNumLeft = -1;
static int gOffSet = -1;
static std::string gOpenHost = "????";
static int gParentId = -1;
static char gPasswd[64] = { 0 };
static char gPubKey[kMAXPATHLEN] = { 0 };
static int gRandInit = 0;
static int gRemPid = -1;
static int gReUseAllow = 0x1F;  // define methods for which tokens can be asked
static int gReUseRequired = -1;
static std::string gRpdAuthTab = std::string(gTmpDir).append(kAuthTab);
static rsa_NUMBER gRSA_d;
static rsa_NUMBER gRSA_n;
static int gRSAInit = 0;
static int gRSAKey = 0;
static rsa_KEY gRSAPriKey;
static rsa_KEY_export gRSAPubExport;
static rsa_KEY gRSAPubKey;
static int gSaltRequired = -1;
static int gSec = -1;
static int gServerProtocol = -1;
static EService gService = kROOTD;
static int gSshdPort = 22;
static int gTriedMeth[kMAXSEC];
static char gUser[64] = { 0 };
static char *gUserAllow[kMAXSEC] = { 0 };          // User access control
static unsigned int gUserAlwLen[kMAXSEC] = { 0 };
static unsigned int gUserIgnLen[kMAXSEC] = { 0 };
static char *gUserIgnore[kMAXSEC] = { 0 };

//
// Kerberos stuff
#ifdef R__KRB5
static krb5_context gKcontext;
static krb5_keytab gKeytab = 0;        // default Keytab file can be changed
static std::string gKeytabFile = "";   // via RpdSetKeytabFile
#endif

//
// Globus stuff
#ifdef R__GLBS
static int gShmIdCred = -1;
static gss_ctx_id_t GlbContextHandle = GSS_C_NO_CONTEXT;
#endif

//______________________________________________________________________________
static int rpdstrncasecmp(const char *str1, const char *str2, int n)
{
   // Case insensitive string compare of n characters.

   while (n > 0) {
      int c1 = *str1;
      int c2 = *str2;

      if (isupper(c1))
         c1 = tolower(c1);

      if (isupper(c2))
         c2 = tolower(c2);

      if (c1 != c2)
         return c1 - c2;

      str1++;
      str2++;
      n--;
   }
   return 0;
}

//______________________________________________________________________________
static int rpdstrcasecmp(const char *str1, const char *str2)
{
   // Case insensitive string compare.

   return rpdstrncasecmp(str1, str2, strlen(str2) + 1);
}

//______________________________________________________________________________
void RpdSetRootLogFlag(int RootLog)
{
   // Change the value of the static gRootLog to RootLog.
   // Recognized values:
   //                       0      log to syslog (for root started daemons)
   //                       1      log to stderr (for user started daemons)

   gRootLog = RootLog;
   if (gDebug > 2)
      ErrorInfo("RpdSetRootLogFlag: gRootLog set to %d", gRootLog);
}

#ifdef R__KRB5
//______________________________________________________________________________
void RpdSetKeytabFile(const char *keytabfile)
{
   // Change the value of the static gKeytab to keytab.
   gKeytabFile = std::string(keytabfile);
   if (gDebug > 2)
      ErrorInfo("RpdSetKeytabFile: using keytab file %s", gKeytabFile.c_str());
}

//______________________________________________________________________________
void RpdFreeKrb5Vars(krb5_context context, krb5_principal principal,
                     krb5_ticket *ticket, krb5_auth_context auth_context,
                     krb5_creds **creds)
{
   // Free allocated quantities for Krb stuff

   if (context) {
      // free creds
      if (creds)
         krb5_free_tgt_creds(context,creds);

      // free auth_context
      if (auth_context)
         krb5_auth_con_free(context, auth_context);

      // free ticket
      if (ticket)
         krb5_free_ticket(context,ticket);

      // free principal
      if (principal)
         krb5_free_principal(context, principal);

      // free context
      krb5_free_context(context);
   }
}

#endif

//______________________________________________________________________________
int RpdGetAuthMethod(int kind)
{
   int Meth = -1;

   if (kind == kROOTD_USER)
      Meth = 0;
   if (kind == kROOTD_SRPUSER)
      Meth = 1;
   if (kind == kROOTD_KRB5)
      Meth = 2;
   if (kind == kROOTD_GLOBUS)
      Meth = 3;
   if (kind == kROOTD_SSH)
      Meth = 4;
   if (kind == kROOTD_RFIO)
      Meth = 5;

   return Meth;
}

//______________________________________________________________________________
int RpdUpdateAuthTab(int opt, const char *line, char **token)
{
   // Update tab file.
   // If opt = -1 : delete file (backup saved in <file>.bak);
   // If opt =  0 : eliminate all inactive entries;
   // if opt =  1 : append 'line'.
   // Returns offset for 'line' (opt = 1) or -1 if any error occurs
   // and token.

   int retval = -1;
   int itab = 0;
   char fbuf[kMAXPATHLEN];

   if (gDebug > 2)
      ErrorInfo("RpdUpdateAuthTab: analyzing: opt: %d, line: %s", opt,
                line);

   if (opt == -1) {
      if (!access(gRpdAuthTab.c_str(), F_OK)) {
         // Save the content ...
         std::string bak = std::string(gRpdAuthTab).append(".bak");
         FILE *fbak = fopen(bak.c_str(), "w");
         FILE *ftab = fopen(gRpdAuthTab.c_str(), "r");
         char buf[kMAXPATHLEN];
         while (fgets(buf, sizeof(buf), ftab)) {
            fprintf(fbak, "%s", buf);
         }
         fclose(fbak);
         fclose(ftab);
         // ... before deleting the original ...
         unlink(gRpdAuthTab.c_str());
      }
      return 0;
   } else if (opt == 0) {
      // Open file for update
      itab = open(gRpdAuthTab.c_str(), O_RDWR | O_CREAT, 0666);
      if (itab == -1) {
         ErrorInfo("RpdUpdateAuthTab: opt=%d: error opening %s"
                   " (errno: %d)", opt, gRpdAuthTab.c_str(), GetErrno());
         return retval;
      }
      // override umask setting
      fchmod(itab, 0666);
      // lock tab file
      if (lockf(itab, F_LOCK, (off_t) 1) == -1) {
         ErrorInfo("RpdUpdateAuthTab: opt=%d: error locking %s"
                   " (errno: %d)", opt, gRpdAuthTab.c_str(), GetErrno());
         close(itab);
         return retval;
      }
      // File is open: get FILE descriptor
      FILE *ftab = fdopen(itab, "w+");
      // and set indicator to beginning
      if (lseek(itab, 0, SEEK_SET) == -1) {
         ErrorInfo("RpdUpdateAuthTab: opt=%d: lseek error (errno: %d)",
              opt, GetErrno());
         if (lockf(itab, F_ULOCK, (off_t) 1) == -1) {
            ErrorInfo("RpdUpdateAuthTab: error unlocking %s",
                      gRpdAuthTab.c_str());
         }
         fclose(ftab);
         return retval;
      }

      // Now scan over entries
      int pr = 0, pw = 0;
      int lsec, act;
      char line[kMAXPATHLEN], dumm[kMAXPATHLEN];
      bool fwr = 0;

      while (fgets(line, sizeof(line), ftab)) {
         pr = lseek(itab, 0, SEEK_CUR);
         sscanf(line, "%d %d %s", &lsec, &act, dumm);

         if (act > 0) {
            if (fwr) {
               lseek(itab, pw, SEEK_SET);
               SPrintf(fbuf, kMAXPATHLEN, "%s\n", line);
               while (write(itab, fbuf, strlen(fbuf)) < 0
                      && GetErrno() == EINTR)
                  ResetErrno();
               pw = lseek(itab, 0, SEEK_CUR);
               lseek(itab, pr, SEEK_SET);
            } else
               pw = lseek(itab, 0, SEEK_CUR);
         } else {
            fwr = 1;
         }
      }

      // Truncate file to new length
      if (ftruncate(itab, pw) == -1)
         ErrorInfo("RpdUpdateAuthTab: opt=%d: ftruncate error (errno: %d)",
              opt, GetErrno());

      retval = 0;

   } else if (opt == 1) {
      // open file for append
      if (gDebug > 2)
         ErrorInfo("RpdUpdateAuthTab: opening file %s",
                   gRpdAuthTab.c_str());

      if (access(gRpdAuthTab.c_str(), F_OK)) {
         itab = open(gRpdAuthTab.c_str(), O_RDWR | O_CREAT, 0666);
         if (itab == -1) {
            ErrorInfo("RpdUpdateAuthTab: opt=%d: error opening %s"
                     "(errno: %d)", opt, gRpdAuthTab.c_str(), GetErrno());
            return retval;
         }
         // override umask setting
         fchmod(itab, 0666);
      } else {
         itab = open(gRpdAuthTab.c_str(), O_RDWR);
      }
      if (itab == -1) {
         ErrorInfo("RpdUpdateAuthTab: opt=%d: error opening"
                   " or creating %s (errno: %d)",
                   opt, gRpdAuthTab.c_str(), GetErrno());
         return retval;
      }
      // lock tab file
      if (gDebug > 2)
         ErrorInfo("RpdUpdateAuthTab: locking file %s", gRpdAuthTab.c_str());
      if (lockf(itab, F_LOCK, (off_t) 1) == -1) {
         ErrorInfo("RpdUpdateAuthTab: opt=%d: error locking %s"
                   " (errno: %d)", opt, gRpdAuthTab.c_str(), GetErrno());
         close(itab);
         return retval;
      }
      // saves offset
      retval = lseek(itab, 0, SEEK_END);
      if (gDebug > 2)
         ErrorInfo("RpdUpdateAuthTab: offset is %d", retval);

      // Generate token
      *token = RpdGetRandString(3, 8);   // 8 crypt-like chras
      char *CryptToken = crypt(*token, *token);
      SPrintf(fbuf, kMAXPATHLEN, "%s %s\n", line, CryptToken);
      if (gDebug > 2)
         ErrorInfo("RpdUpdateAuthTab: token: '%s'", CryptToken);

      // adds line
      while (write(itab, fbuf, strlen(fbuf)) < 0 && GetErrno() == EINTR)
         ResetErrno();

   } else {
      ErrorInfo("RpdUpdateAuthTab: unrecognized option (opt= %d)", opt);
      return retval;
   }

   // unlock the file
   lseek(itab, 0, SEEK_SET);
   if (lockf(itab, F_ULOCK, (off_t) 1) == -1) {
      ErrorInfo("RpdUpdateAuthTab: error unlocking %s",
                gRpdAuthTab.c_str());
   }
   // closing file ...
   close(itab);

   return retval;
}

//______________________________________________________________________________
int RpdCleanupAuthTab(const char *Host, int RemId, int OffSet)
{
   // In tab file, cleanup (set inactive) entry at offset
   // 'OffSet' from remote PiD 'RemId' at 'Host'.
   // If Host="all" or RemId=0 discard all entries.
   // Return number of entries not cleaned properly ...

   int retval = 0;

   if (gDebug > 2)
      ErrorInfo("RpdCleanupAuthTab: Host: '%s', RemId:%d, OffSet: %d",
                Host, RemId, OffSet);

   // Open file for update
   int itab = -1;
   if (access(gRpdAuthTab.c_str(),F_OK) == 0) {

      itab = open(gRpdAuthTab.c_str(), O_RDWR);
      if (itab == -1) {
         ErrorInfo("RpdCleanupAuthTab: error opening %s (errno: %d)",
                  gRpdAuthTab.c_str(), GetErrno());
         //     return retval;
         return -1;
      }
   } else {
      if (gDebug > 0)
         ErrorInfo("RpdCleanupAuthTab: file %s does not exist",
                   gRpdAuthTab.c_str());
      return -3;
   }

   // lock tab file
   if (lockf(itab, F_LOCK, (off_t) 1) == -1) {
      ErrorInfo("RpdCleanupAuthTab: error locking %s (errno: %d)",
                gRpdAuthTab.c_str(), GetErrno());
      close(itab);
      //     return retval;
      return -2;
   }
   // File is open: get FILE descriptor
   FILE *ftab = fdopen(itab, "r+");

   // Now access entry or scan over entries
   int pr = 0, pw = 0;
   int nw, lsec, act, parid, remid, pkey;
   char line[kMAXPATHLEN], line1[kMAXPATHLEN], host[kMAXPATHLEN];
   char dumm[kMAXPATHLEN], user[kMAXPATHLEN];
#ifdef R__GLBS
   char subj[kMAXPATHLEN];
#endif

   // Set indicator
   int all = (int)(!strcmp(Host, "all") || RemId == 0);
   if (all || OffSet < 0)
      pr = lseek(itab, 0, SEEK_SET);
   else
      pr = lseek(itab, OffSet, SEEK_SET);
   pw = pr;
   while (fgets(line, sizeof(line), ftab)) {
      pr += strlen(line);
      if (gDebug > 2)
         ErrorInfo("RpdCleanupAuthTab: pr:%d pw:%d (line:%s) (pId:%d)",
                    pr, pw, line, gParentId);

      nw = sscanf(line, "%d %d %d %d %d %s %s %s", &lsec, &act, &pkey,
                  &parid, &remid, host, user, dumm);

      if (nw > 5) {
         if (all || OffSet > -1 ||
            (!strcmp(Host, host) && (RemId == remid))) {

            // Delete Public Key file
            char strpw[20];
            snprintf(strpw,20,"%d",pw);
            std::string PubKeyFile;
            PubKeyFile = gTmpDir + "/rpk_" + strpw;

            if (gDebug > 0) {
               struct stat st;
               if (stat(PubKeyFile.c_str(), &st) == 0) {
                  ErrorInfo("RpdCleanupAuthTab: file uid:%d gid:%d",
                             st.st_uid,st.st_gid);
               }
               ErrorInfo("RpdCleanupAuthTab: proc uid:%d gid:%d",
               getuid(),getgid());
            }

            if (unlink(PubKeyFile.c_str()) == -1) {
               if (gDebug > 0 && GetErrno() != ENOENT) {
                  ErrorInfo("RpdCleanupAuthTab: problems unlinking pub"
                            " key file '%s' (errno: %d)",
                            PubKeyFile.c_str(),GetErrno());
               }
            }

            // Deactivate active entries (either inclusive or exclusive:
            // remote client has gone ...)
            if (act > 0) {
               if (lsec == 3) {
#ifdef R__GLBS
                  int shmid;
                  nw = sscanf(line, "%d %d %d %d %d %s %s %d %s %s", &lsec,
                              &act, &pkey, &parid, &remid, host, user,
                              &shmid, subj, dumm);
                  struct shmid_ds shm_ds;
                  if (shmctl(shmid, IPC_RMID, &shm_ds) == -1) {
                     ErrorInfo("RpdCleanupAuthTab: unable to mark shared"
                               " memory segment %d", shmid);
                     ErrorInfo("RpdCleanupAuthTab: for desctruction"
                               " (errno: %d)", GetErrno());
                     retval++;
                  }
                  SPrintf(line1, kMAXPATHLEN,
                          "%d 0 %d %d %d %s %s %d %s %s\n",
                          lsec, pkey, parid, remid, host,
                          user, shmid, subj, dumm);
#else
                  ErrorInfo("RpdCleanupAuthTab: compiled without Globus"
                            " support: you shouldn't have got here!");
                  SPrintf(line1, kMAXPATHLEN,
                          "%d %d %d %d %d %s %s %s - WARNING: bad line\n",
                          lsec, 0, pkey, parid, remid, host, user, dumm);
#endif
               } else {
                  SPrintf(line1, kMAXPATHLEN, "%d 0 %d %d %d %s %s %s\n",
                          lsec, pkey, parid, remid, host, user, dumm);
               }
               lseek(itab, pw, SEEK_SET);
               while (write(itab, line1, strlen(line1)) < 0
                      && GetErrno() == EINTR)
                  ResetErrno();
               if (all || OffSet < 0)
                  lseek(itab, pr, SEEK_SET);
               else
                  lseek(itab,  0, SEEK_END);
            }
         }
      }
      pw = pr;
   }

   // unlock the file
   lseek(itab, 0, SEEK_SET);
   if (lockf(itab, F_ULOCK, (off_t) 1) == -1) {
      ErrorInfo("RpdCleanupAuthTab: error unlocking %s", gRpdAuthTab.c_str());
   }
   // closing file ...
   fclose(ftab);

   return retval;
}

//______________________________________________________________________________
int RpdCheckAuthTab(int Sec, const char *User, const char *Host, int RemId,
                    int *OffSet)
{
   // Check authentication entry in tab file.

   int retval = 0;
   if (gDebug > 2)
      ErrorInfo("RpdCheckAuthTab: analyzing: %d %s %s %d %d", Sec, User,
                Host, RemId, *OffSet);

   // Check OffSet first
   char *tkn = 0, *user =0;
   int shmid;
   bool GoodOfs = RpdCheckOffSet(Sec,User,Host,RemId,
                                 OffSet,&tkn,&shmid,&user);

   if (gDebug > 2)
      ErrorInfo("RpdCheckAuthTab: GoodOfs: %d", GoodOfs);

   // Notify the result of the check
   if (GoodOfs) {
      // We will receive the user token next
      NetSend(1, kROOTD_AUTH);
   } else {
      // No authentication available for re-use
      NetSend(0, kROOTD_AUTH);
      // Cleanup and return: we need a new one ...
      if (tkn) delete[] tkn;
      if (user) delete[] user;
      // ... no need to continue receiving the old token
      return retval;
   }

   // Now Receive Token
   int ofs = *OffSet;
   char *token = 0;
   if (gRSAKey > 0) {
      // Get Public Key
      char strofs[20];
      snprintf(strofs,20,"%d",ofs);
      std::string PubKeyFile;
      PubKeyFile = gTmpDir + "/rpk_" + strofs;
      if (gDebug > 2)
         ErrorInfo("RpdCheckAuthTab: RSAKey ofs file: %d %d '%s' ",
                   gRSAKey, ofs, PubKeyFile.c_str());
      if (RpdGetRSAKeys(PubKeyFile.c_str(), 1) > 0) {
         if (RpdSecureRecv(&token) == -1) {
            ErrorInfo
                ("RpdCheckAuthTab: problems secure-receiving token %s",
                 "- may result in authentication failure ");
         }
      }
   } else {
      EMessageTypes kind;
      int Tlen = 9;
      token = new char[Tlen];
      NetRecv(token, Tlen, kind);
      if (kind != kMESS_STRING)
         ErrorInfo
             ("RpdCheckAuthTab: got msg kind: %d instead of %d (kMESS_STRING)",
              kind, kMESS_STRING);
      // Invert Token
      for (int i = 0; i < (int) strlen(token); i++) {
         token[i] = ~token[i];
      }
   }

   if (gDebug > 2)
      ErrorInfo
          ("RpdCheckAuthTab: received from client: token: '%s' ",
           token);

   // Now check Token validity
   if (GoodOfs && token && RpdCheckToken(token, tkn)) {

      if (Sec == 3) {
#ifdef R__GLBS
         // kGlobus:
         if (GlbsToolCheckContext(shmid)) {
            retval = 1;
            strcpy(gUser, user);
         } else {
            // set entry inactive
            RpdCleanupAuthTab(Host,RemId,*OffSet);
         }
#else
         ErrorInfo
                ("RpdCheckAuthTab: compiled without Globus support:%s",
                 " you shouldn't have got here!");
#endif
      } else {
            retval = 1;
      }

      // Comunicate new offset to remote client
      if (retval) *OffSet = ofs;
   }

   if (tkn) delete[] tkn;
   if (user) delete[] user;

   return retval;
}

//______________________________________________________________________________
int RpdCheckOffSet(int Sec, const char *User, const char *Host, int RemId,
                   int *OffSet, char **Token, int *ShmId, char **GlbsUser)
{
   // Check offset received from client entry in tab file.

   int retval = 0;
   bool GoodOfs = 0;
   int ofs = *OffSet >= 0 ? *OffSet : 0;

   if (gDebug > 2)
      ErrorInfo("RpdCheckOffSet: analyzing: %d %s %s %d %d", Sec, User,
                Host, RemId, *OffSet);

   // First check if file exists and can be read
   if (access(gRpdAuthTab.c_str(), R_OK)) {
      ErrorInfo("RpcCheckOffSet: can't read file %s (errno: %d)",
                gRpdAuthTab.c_str(), GetErrno());
      return retval;
   }
   // Open file
   int itab = open(gRpdAuthTab.c_str(), O_RDWR);
   if (itab == -1) {
      ErrorInfo("RpcCheckOffSet: error opening %s (errno: %d)",
                gRpdAuthTab.c_str(), GetErrno());
      return retval;
   }
   // lock tab file
   if (lockf(itab, F_LOCK, (off_t) 1) == -1) {
      ErrorInfo("RpcCheckOffSet: error locking %s (errno: %d)",
                gRpdAuthTab.c_str(), GetErrno());
      close(itab);
      return retval;
   }
   // File is open: set position at wanted location
   FILE *ftab = fdopen(itab, "r+");
   fseek(ftab, ofs, SEEK_SET);

   // Now read the entry
   char line[kMAXPATHLEN];
   fgets(line, sizeof(line), ftab);

   // and parse its content according to auth method
   int lsec, act, parid, remid, shmid;
   char host[kMAXPATHLEN], user[kMAXPATHLEN], subj[kMAXPATHLEN],
       dumm[kMAXPATHLEN], tkn[20];
   int nw =
       sscanf(line, "%d %d %d %d %d %s %s %s %s", &lsec, &act, &gRSAKey,
              &parid, &remid, host, user, tkn, dumm);
   if (gDebug > 2)
      ErrorInfo("RpcCheckOffSet: found line: %s", line);

   if (nw > 5 && act > 0 &&
      (gInclusiveToken || (act == 2 && gParentId == parid))) {
      if ((lsec == Sec)) {
         if (lsec == 3) {
            sscanf(line, "%d %d %d %d %d %s %s %d %s %s %s", &lsec, &act,
                   &gRSAKey, &parid, &remid, host, user, &shmid, subj, tkn,
                   dumm);
            if ((remid == RemId)
                && !strcmp(host, Host) && !strcmp(subj, User))
               GoodOfs = 1;
         } else {
            if ((remid == RemId) &&
                !strcmp(host, Host) && !strcmp(user, User))
               GoodOfs = 1;
         }
      }
   }
   if (!GoodOfs) {
      // Tab may have been cleaned in the meantime ... try a scan
      fseek(ftab, 0, SEEK_SET);
      ofs = ftell(ftab);
      while (fgets(line, sizeof(line), ftab)) {
         nw = sscanf(line, "%d %d %d %d %d %s %s %s %s", &lsec, &act,
                     &gRSAKey, &parid, &remid, host, user, tkn, dumm);
         if (gDebug > 2)
            ErrorInfo("RpcCheckOffSet: found line: %s", line);

         if (nw > 5 && act > 0 &&
            (gInclusiveToken || (act == 2 && gParentId == parid))) {
            if (lsec == Sec) {
               if (lsec == 3) {
                  sscanf(line, "%d %d %d %d %d %s %s %d %s %s %s", &lsec,
                         &act, &gRSAKey, &parid, &remid, host, user,
                         &shmid, subj, tkn, dumm);
                  if ((remid == RemId)
                      && !strcmp(host, Host) && !strcmp(subj, User)) {
                     GoodOfs = 1;
                     goto found;
                  }
               } else {
                  if ((remid == RemId) &&
                      !strcmp(host, Host) && !strcmp(user, User)) {
                     GoodOfs = 1;
                     goto found;
                  }
               }
            }
         }
      }
   }

 found:
   // unlock the file
   lseek(itab, 0, SEEK_SET);
   if (lockf(itab, F_ULOCK, (off_t) 1) == -1) {
      ErrorInfo("RpcCheckOffSet: error unlocking %s",
                gRpdAuthTab.c_str());
   }
   // closing file ...
   close(itab);

   if (gDebug > 2)
      ErrorInfo("RpcCheckOffSet: GoodOfs: %d (active: %d)",
                GoodOfs, act);

   // Rename the key file, if needed
   if (*OffSet > 0 && *OffSet != ofs) {
      std::string OldName = std::string(gTmpDir).append("/rpk_");
      OldName.append(ItoA(*OffSet));
      if (!access(OldName.c_str(), W_OK)) {
         GoodOfs = 0;
         ErrorInfo("RpdCleanupAuthTab: NO access to key file %s"
                   " (errno: %d) (uid: %d)",
                   OldName.c_str(),GetErrno(),getuid());
         if (!getuid()) {
            struct stat st;
            if (stat(OldName.c_str(), &st) == -1) {
               ErrorInfo("RpdCleanupAuthTab: unable to stat key file %s"
                         " (errno: %d)",OldName.c_str(),GetErrno());
            } else {
               if (st.st_uid != getuid() || st.st_gid != getgid())
                  ErrorInfo("RpdCleanupAuthTab: NOT superuser"
                            " and NOT owner of key file");
            }
         }
      } else {
         // Ok, we should have full rights on the key file
         std::string NewName = std::string(gTmpDir).append("/rpk_");
         NewName.append(ItoA(ofs));
         if (rename(OldName.c_str(), NewName.c_str()) == -1) {
            // Error: set entry inactive
            if (gDebug > 0)
               ErrorInfo("RpcCheckOffSet: Error renaming key file"
                         " %s to %s (errno: %d)",
                         OldName.c_str(),NewName.c_str(),GetErrno());
            RpdCleanupAuthTab(Host,RemId,*OffSet);
         }
      }
   }

   // Comunicate new offset to remote client
   if (GoodOfs) {
      *OffSet = ofs;
      // return token if requested
      if (Token) {
         *Token = new char[strlen(tkn)+1];
         strcpy(*Token,tkn);
      }
      if (Sec == 3) {
         if (GlbsUser) {
            *GlbsUser = new char[strlen(user)+1];
            strcpy(*GlbsUser,user);
         }
         if (ShmId)
            *ShmId = shmid;
      }
   }

   return GoodOfs;
}

//______________________________________________________________________________
bool RpdCheckToken(char *token, char *tknref)
{
   // Check token validity.

   // Get rid of '\n'
   char *s = strchr(token, '\n');
   if (s)
      *s = 0;
   s = strchr(tknref, '\n');
   if (s)
      *s = 0;

   char *tkn_crypt = crypt(token, tknref);

   if (gDebug > 2)
      ErrorInfo("RpdCheckToken: ref:'%s' crypt:'%s'", tknref, tkn_crypt);

   if (!strncmp(tkn_crypt, tknref, 13))
      return 1;
   else
      return 0;
}

//______________________________________________________________________________
bool RpdReUseAuth(const char *sstr, int kind)
{
   // Check the requiring subject has already authenticated during this session
   // and its 'ticket' is still valid.
   // Not implemented for SRP and Krb5 (yet).

   int Ulen, OffSet, Opt;
   gOffSet = -1;
   gAuth = 0;

   if (gDebug > 2)
      ErrorInfo("RpdReUseAuth: analyzing: %s, %d", sstr, kind);

   char *User = new char[strlen(sstr)];
   char *Token = 0;

   // kClear
   if (kind == kROOTD_USER) {
      if (!(gReUseAllow & kAUTH_CLR_MSK))
         return 0;              // re-authentication required by administrator
      gSec = 0;
      // Decode subject string
      sscanf(sstr, "%d %d %d %d %s", &gRemPid, &OffSet, &Opt, &Ulen, User);
      User[Ulen] = '\0';
      if ((gReUseRequired = (Opt & kAUTH_REUSE_MSK))) {
         gOffSet = OffSet;
         if (gRemPid > 0 && gOffSet > -1) {
            gAuth =
                RpdCheckAuthTab(gSec, User, gOpenHost.c_str(), gRemPid, &gOffSet);
         }
         if ((gAuth == 1) && (OffSet != gOffSet))
            gAuth = 2;
         // Fill gUser and free allocated memory
         strcpy(gUser, User);
      }
   }
   // kSRP
   if (kind == kROOTD_SRPUSER) {
      if (!(gReUseAllow & kAUTH_SRP_MSK))
         return 0;              // re-authentication required by administrator
      gSec = 1;
      // Decode subject string
      sscanf(sstr, "%d %d %d %d %s", &gRemPid, &OffSet, &Opt, &Ulen, User);
      User[Ulen] = '\0';
      if ((gReUseRequired = (Opt & kAUTH_REUSE_MSK))) {
         gOffSet = OffSet;
         if (gRemPid > 0 && gOffSet > -1) {
            gAuth =
                RpdCheckAuthTab(gSec, User, gOpenHost.c_str(), gRemPid, &gOffSet);
         }
         if ((gAuth == 1) && (OffSet != gOffSet))
            gAuth = 2;
         // Fill gUser and free allocated memory
         strcpy(gUser, User);
      }
   }
   // kKrb5
   if (kind == kROOTD_KRB5) {
      if (!(gReUseAllow & kAUTH_KRB_MSK))
         return 0;              // re-authentication required by administrator
      gSec = 2;
      // Decode subject string
      sscanf(sstr, "%d %d %d %d %s", &gRemPid, &OffSet, &Opt, &Ulen, User);
      User[Ulen] = '\0';
      if ((gReUseRequired = (Opt & kAUTH_REUSE_MSK))) {
         gOffSet = OffSet;
         if (gRemPid > 0 && gOffSet > -1) {
            gAuth =
                RpdCheckAuthTab(gSec, User, gOpenHost.c_str(), gRemPid, &gOffSet);
         }
         if ((gAuth == 1) && (OffSet != gOffSet))
            gAuth = 2;
         // Fill gUser and free allocated memory
         strcpy(gUser, User);
      }
   }
   // kGlobus
   if (kind == kROOTD_GLOBUS) {
      if (!(gReUseAllow & kAUTH_GLB_MSK))
         return 0;              //  re-authentication required by administrator
      gSec = 3;
      // Decode subject string
      int Slen;
      sscanf(sstr, "%d %d %d %d %s", &gRemPid, &OffSet, &Opt, &Slen, User);
      User[Slen] = '\0';
      if ((gReUseRequired = (Opt & kAUTH_REUSE_MSK))) {
         gOffSet = OffSet;
         if (gRemPid > 0 && gOffSet > -1) {
            gAuth =
                RpdCheckAuthTab(gSec, User, gOpenHost.c_str(), gRemPid, &gOffSet);
         }
         if ((gAuth == 1) && (OffSet != gOffSet))
            gAuth = 2;
      }
   }
   // kSSH
   if (kind == kROOTD_SSH) {
      if (!(gReUseAllow & kAUTH_SSH_MSK))
         return 0;              //  re-authentication required by administrator
      gSec = 4;
      // Decode subject string
      char *Pipe = new char[strlen(sstr)];
      sscanf(sstr, "%d %d %d %s %d %s", &gRemPid, &OffSet, &Opt, Pipe,
             &Ulen, User);
      User[Ulen] = '\0';
      if ((gReUseRequired = (Opt & kAUTH_REUSE_MSK))) {
         gOffSet = OffSet;
         if (gRemPid > 0 && gOffSet > -1) {
            gAuth =
                RpdCheckAuthTab(gSec, User, gOpenHost.c_str(), gRemPid, &gOffSet);
         }
         if ((gAuth == 1) && (OffSet != gOffSet))
            gAuth = 2;
         // Fill gUser and free allocated memory
         strcpy(gUser, User);
      }
      if (Pipe) delete[] Pipe;
   }

   if (User) delete[] User;
   if (Token) delete[] Token;

   // Return value
   if (gAuth >= 1) {
      return 1;
   } else {
      return 0;
   }
}

//______________________________________________________________________________
int RpdCheckAuthAllow(int Sec, const char *Host)
{
   // Check if required auth method is allowed for 'Host'.
   // If 'yes', returns 0, if 'no', returns 1, the number of allowed
   // methods in NumAllow, and the codes of the allowed methods (in order
   // of preference) in AllowMeth. Memory for AllowMeth must be allocated
   // outside. Directives read from (in decreasing order of priority):
   // $ROOTDAEMONRC, $HOME/.rootdaemonrc (privately startd daemons only)
   // or $ROOTETCDIR/system.rootdaemonrc.

   int retval = 1, found = 0;

   std::string theDaemonRc;

   // Check if a non-standard file has been requested
   if (getenv("ROOTDAEMONRC"))
      theDaemonRc = getenv("ROOTDAEMONRC");

   if (theDaemonRc.length() || access(theDaemonRc.c_str(), R_OK)) {
      if (getuid()) {
         // Check if user has a private daemon access file ...
         struct passwd *pw = getpwuid(getuid());
         if (pw != 0) {
            theDaemonRc = std::string(pw->pw_dir).append("/");
            theDaemonRc.append(kDaemonRc);
         }
         if (pw == 0 || access(theDaemonRc.c_str(), R_OK)) {
            if (getenv("ROOTETCDIR")) {
               theDaemonRc = std::string(getenv("ROOTETCDIR")).append("/system");
               theDaemonRc.append(kDaemonRc);
            } else
               theDaemonRc = std::string("/etc/root/system").append(kDaemonRc);
         }
      } else {
         // If running as super-user, check system file only
         if (getenv("ROOTETCDIR")) {
            theDaemonRc = std::string(getenv("ROOTETCDIR")).append("/system");
            theDaemonRc.append(kDaemonRc);
         } else
            theDaemonRc = std::string("/etc/root/system").append(kDaemonRc);
      }
   }
   if (gDebug > 2)
      ErrorInfo("RpdCheckAuthAllow: Checking file: %s for meth:%d"
                " host:%s (gNumAllow: %d)",
                theDaemonRc.c_str(), Sec, Host, gNumAllow);

   // Check if info already loaded (not first call ...)
   if (gMethInit == 1) {

      // Look for the method in the allowed list and flag this method
      // as tried, if found ...
      int newtry = 0, i;
      for (i = 0; i < gNumAllow; i++) {
         if (gTriedMeth[i] == 0 && gAllowMeth[i] == Sec) {
            newtry = 1;
            gTriedMeth[i] = 1;
            gNumLeft--;
         }
      }
      if (newtry == 0) {
         ErrorInfo
             ("RpdCheckAuthAllow: new auth method proposed by %s",
              " client not in the list or already attempted");
         return retval;
      }
      retval = 0;

   } else {
      // This is the first call ... check for host specific directives
      gMethInit = 1;

      // First check if file exists and can be read
      if (access(theDaemonRc.c_str(), R_OK)) {
         ErrorInfo("RpdCheckAuthAllow: can't read file %s (errno: %d)",
                   theDaemonRc.c_str(), GetErrno());
         return retval;
      }
      // Open file
      FILE *ftab = fopen(theDaemonRc.c_str(), "r");
      if (ftab == 0) {
         ErrorInfo("RpdCheckAuthAllow: error opening %s (errno: %d)",
                   theDaemonRc.c_str(), GetErrno());
         return retval;
      }
      // Now read the entry
      char line[kMAXPATHLEN], host[kMAXPATHLEN], rest[kMAXPATHLEN],
          cmth[kMAXPATHLEN];
      int nmet = 0, mth[6] = { 0 };

      int cont = 0, jm = -1;
      while (fgets(line, sizeof(line), ftab)) {
         int rc = 0, i;
         if (line[0] == '#')
            continue;           // skip comment lines
         if (line[strlen(line) - 1] == '\n')
            line[strlen(line) - 1] = '\0';   // get rid of '\n', if any ...
         // Analyze the line now ...
         int nw = 0;
         char *pstr = line;
         // Check if a continuation line
         if (cont == 1) {
            cont = 0;
            nw = sscanf(pstr, "%s", rest);
            if (nw == 0)
               continue;        // empty line
         } else {
            jm = -1;
            // Get 'host' first ...
            nw = sscanf(pstr, "%s %s", host, rest);
            if (nw < 2)
               continue;        // no method defined for this host
            pstr = line + strlen(host) + 1;

            // Check if a service is specified
            char *pcol = strstr(host, ":");
            if (pcol) {
               if (!strstr(pcol+1, gServName[gService].c_str()))
                  continue;
               else
                  host[(int)(pcol-host)] = '\0';
            }
            if (strlen(host) == 0)
               strcpy(host, "default");

            if (gDebug > 2)
               ErrorInfo("RpdCheckAuthAllow: found host: %s ", host);

            if (strcmp(host, "default")) {
               // now check validity of 'host' format
               if (!RpdCheckHost(Host,host)) {
                  goto next;
               }
            } else {
               // This is a default entry: ignore it if a host-specific entry was already
               // found, analyse it otherwise ...
               if (found == 1)
                  goto next;
            }

            if (rc != 0)
               continue;        // bad or unmatched name

            // Reset mth[kMAXSEC]
            nmet = 0;
            for (i = 0; i < kMAXSEC; i++) {
               mth[i] = -1;
            }

         }

         // We are at the end and there will be a continuation line ...
         if (rest[0] == '\\') {
            cont = 1;
            continue;
         }

         while (pstr != 0) {
            int tmet = -1;
            char *pd = 0, *pd2 = 0;
            cmth[0] = '\0';
            rest[0] = '\0';
            nw = sscanf(pstr, "%s %s", cmth, rest);
            if (!strcmp(cmth, "none")) {
               nmet = 0;
               goto nexti;
            }
            pd = strchr(cmth, ':');
            // Parse the method
            char tmp[20];
            if (pd != 0) {
               int mlen = pd - cmth;
               strncpy(tmp, cmth, mlen);
               tmp[mlen] = '\0';
            } else {
               strcpy(tmp, cmth);
            }

            if (strlen(tmp) > 1) {

               for (tmet = 0; tmet < kMAXSEC; tmet++) {
                  if (!rpdstrcasecmp(kAuthMeth[tmet].c_str(), tmp))
                     break;
               }
               if (tmet < kMAXSEC) {
                  ErrorInfo("RpdCheckAuthAllow: tmet %d", tmet);
               } else {
                  ErrorInfo("RpdCheckAuthAllow: unknown methods %s - ignore", tmp);
                  goto nexti;
               }

            } else {
               tmet = atoi(tmp);
            }
            jm = -1;
            if (gDebug > 2)
               ErrorInfo("RpdCheckAuthAllow: found method %d (have?:%d)",
                         tmet, gHaveMeth[tmet]);
            if (tmet >= 0 && tmet <= kMAXSEC) {
               if (gHaveMeth[tmet] == 1) {
                  int i;
                  for (i = 0; i < nmet; i++) {
                     if (mth[i] == tmet) {
                        jm = i;
                     }
                  }
               } else
                  goto nexti;
            } else
               goto nexti;
            if (jm == -1) {
               // New method ...
               mth[nmet] = tmet;
               jm = nmet;
               nmet++;
            }
            // Now parse users list, if any ...
            while (pd != 0 && (int) (pd[1]) != 32) {
               pd2 = strchr(pd + 1, ':');
               if (pd[1] == '-') {
                  pd += 2;
                  // Ignore
                  if (gUserIgnore[mth[jm]] == 0) {
                     gUserIgnLen[mth[jm]] = kMAXPATHLEN;
                     gUserIgnore[mth[jm]] = new char[gUserIgnLen[mth[jm]]];
                     gUserIgnore[mth[jm]][0] = '\0';
                  }
                  if (strlen(gUserIgnore[mth[jm]]) >
                      (gUserIgnLen[mth[jm]] - 10)) {
                     char *UItmp = strdup(gUserIgnore[mth[jm]]);
                     free(gUserIgnore[mth[jm]]);
                     gUserIgnLen[mth[jm]] += kMAXPATHLEN;
                     gUserIgnore[mth[jm]] = new char[gUserIgnLen[mth[jm]]];
                     strcpy(gUserIgnore[mth[jm]], UItmp);
                     free(UItmp);
                  }
                  char usr[256];
                  if (pd2 != 0) {
                     int ulen = pd2 - pd;
                     strncpy(usr, pd, ulen);
                     usr[ulen] = '\0';
                  } else {
                     strcpy(usr, pd);
                  }
                  struct passwd *pw = getpwnam(usr);
                  if (pw != 0)
                     sprintf(gUserIgnore[mth[jm]], "%s %d",
                             gUserIgnore[mth[jm]], (int)pw->pw_uid);
               } else {
                  pd += 1;
                  if (pd[1] == '+')
                     pd += 1;
                  // Keep
                  if (gUserAllow[mth[jm]] == 0) {
                     gUserAlwLen[mth[jm]] = kMAXPATHLEN;
                     gUserAllow[mth[jm]] = new char[gUserAlwLen[mth[jm]]];
                     gUserAllow[mth[jm]][0] = '\0';
                  }
                  if (strlen(gUserAllow[mth[jm]]) >
                      (gUserAlwLen[mth[jm]] - 10)) {
                     char *UItmp = strdup(gUserAllow[mth[jm]]);
                     free(gUserAllow[mth[jm]]);
                     gUserAlwLen[mth[jm]] += kMAXPATHLEN;
                     gUserAllow[mth[jm]] = new char[gUserAlwLen[mth[jm]]];
                     strcpy(gUserAllow[mth[jm]], UItmp);
                     free(UItmp);
                  }
                  char usr[256];
                  if (pd2 != 0) {
                     int ulen = pd2 - pd;
                     strncpy(usr, pd, ulen);
                     usr[ulen] = '\0';
                  } else {
                     strcpy(usr, pd);
                  }
                  struct passwd *pw = getpwnam(usr);
                  if (pw != 0)
                     sprintf(gUserAllow[mth[jm]], "%s %d",
                             gUserAllow[mth[jm]], (int)pw->pw_uid);
               }
               pd = pd2;
            }
            // Get next item
          nexti:
            if (nw > 1 && (int) rest[0] != 92) {
               pstr = strstr(pstr, rest);
            } else {
               if ((int) rest[0] == 92)
                  cont = 1;
               pstr = 0;
            }
         }
         if (gDebug > 2) {
            ErrorInfo("RpdCheckAuthAllow: for host %s found %d methods",
                      host, nmet);
            ErrorInfo("RpdCheckAuthAllow: %d %d %d %d %d %d", mth[0],
                      mth[1], mth[2], mth[3], mth[4], mth[5]);
         }
         // Found new entry matching: superseed previous result
         found = 1;
         retval = 1;
         gNumAllow = gNumLeft = nmet;
         for (i = 0; i < kMAXSEC; i++) {
            gAllowMeth[i] = -1;
            gTriedMeth[i] = 0;
            if (i < gNumAllow) {
               gAllowMeth[i] = mth[i];
               if (Sec == mth[i]) {
                  retval = 0;
                  gNumLeft--;
                  gTriedMeth[i] = 1;
               }
            }
         }
       next:
         continue;
      }

      // closing file ...
      fclose(ftab);

      // Use defaults if nothing found
      if (!found) {
         if (gDebug > 2)
            ErrorInfo
            ("RpdCheckAuthAllow: no specific or 'default' entry found: %s",
             "using system defaults");
         int i;
         for (i = 0; i < gNumAllow; i++) {
            if (Sec == gAllowMeth[i]) {
               retval = 0;
               gNumLeft--;
               gTriedMeth[i] = 1;
            }
         }

      }

   }
   if (gDebug > 2) {
      ErrorInfo
          ("RpdCheckAuthAllow: returning: %d (gNumAllow: %d, gNumLeft:%d)",
           retval, gNumAllow, gNumLeft);
      int i, jm;
      for (i = 0; i < kMAXSEC; i++) {
         jm = gAllowMeth[i];
         if (gUserAlwLen[jm] > 0)
            ErrorInfo("RpdCheckAuthAllow: users allowed for method %d: %s",
                      jm, gUserAllow[jm]);
      }
      for (i = 0; i < kMAXSEC; i++) {
         jm = gAllowMeth[i];
         if (gUserIgnLen[jm] > 0)
            ErrorInfo("RpdCheckAuthAllow: users ignored for method %d: %s",
                      jm, gUserIgnore[jm]);
      }
   }

   return retval;
}

//______________________________________________________________________________
int RpdCheckHost(const char *Host, const char *host)
{
   // Checks if 'host' is compatible with 'Host' taking into account
   // wild cards in the host name
   // Returns 1 if successful, 0 otherwise ...

   int rc = 1;

   // Strings must be both defined
   if (!Host || !host)
      return 0;

   // If host is a just wild card accept it
   if (!strcmp(host,"*"))
      return 1;

   // Try now to understand whether it is an address or a name ...
   int name = 0, i = 0;
   for (i = 0; i < (int) strlen(host); i++) {
      if ((host[i] < 48 || host[i] > 57) &&
           host[i] != '*' && host[i] != '.') {
         name = 1;
         break;
      }
   }

   // If ref host is an IP, get IP of Host
   char *H;
   if (!name) {
      H = RpdGetIP(Host);
      if (gDebug > 2)
         ErrorInfo("RpdCheckHost: Checking Host IP: %s", H);
   } else {
      H = new char[strlen(Host)+1];
      strcpy(H,Host);
      if (gDebug > 2)
         ErrorInfo("RpdCheckHost: Checking Host name: %s", H);
   }

   // Check if starts with wild
   // Starting with '.' defines a full (sub)domain
   int sos = 0;
   if (host[0] == '*' || host[0] == '.')
      sos = 1;

   // Check if ends with wild
   // Ending with '.' defines a name
   int eos = 0, le = strlen(host);
   if (host[le-1] == '*' || host[le-1] == '.')
      eos = 1;

   int first= 1;
   int ends= 0;
   int starts= 0;
   char *h = new char[strlen(host)+1];
   strcpy(h,host);
   char *tk = strtok(h,"*");
   while (tk) {

      char *ps = strstr(H,tk);
      if (!ps) {
         rc = 0;
         break;
      }
      if (!sos && first && ps == H)
         starts = 1;
      first = 0;

      if (ps == H + strlen(H) - strlen(tk))
         ends = 1;

      tk = strtok(0,"*");

   }
   if (h) delete[] h;
   if (H) delete[] H;

   if ((!sos || !eos) && !starts && !ends)
      rc = 0;

   return rc;
}

//______________________________________________________________________________
char *RpdGetIP(const char *host)
{
   // Get IP address of 'host' as a string. String must be deleted by
   // the user.

   struct hostent *h;
   unsigned long ip;
   unsigned char ip_fld[4];

   // Check server name
   if ((h = gethostbyname(host)) == 0) {
      ErrorInfo("RpdGetIP: unknown host %s", host);
      return 0;
   }
   // Decode ...
   ip = ntohl(*(unsigned long *) h->h_addr_list[0]);
   ip_fld[0] = (unsigned char) ((0xFF000000 & ip) >> 24);
   ip_fld[1] = (unsigned char) ((0x00FF0000 & ip) >> 16);
   ip_fld[2] = (unsigned char) ((0x0000FF00 & ip) >> 8);
   ip_fld[3] = (unsigned char) ((0x000000FF & ip));

   // Prepare output
   char *output = new char[20];
   SPrintf(output, 20, "%d.%d.%d.%d",
           ip_fld[0], ip_fld[1], ip_fld[2], ip_fld[3]);

   // return
   return output;
}

//______________________________________________________________________________
void RpdSendAuthList()
{
   // Send list of authentication methods not yet tried.

   if (gDebug > 2)
      ErrorInfo("RpdSendAuthList: analyzing (gNumLeft: %d)", gNumLeft);

   // Send Number of methods left
   NetSend(gNumLeft, kROOTD_NEGOTIA);

   if (gNumLeft > 0) {
      int i = 0;
      const int ldum = kMAXSEC*4;
      char sdum[ldum];
      for (i = 0; i < gNumAllow; i++) {
         if (gDebug > 2)
            ErrorInfo("RpdSendAuthList: gTriedMeth[%d]: %d", i,
                      gTriedMeth[i]);
         if (gTriedMeth[i] == 0) {
            SPrintf(sdum, ldum, "%s %d", sdum, gAllowMeth[i]);
         }
      }
      NetSend(sdum, ldum, kMESS_STRING);
      if (gDebug > 2)
         ErrorInfo("RpdSendAuthList: sent list: %s", sdum);
   }
}


//______________________________________________________________________________
void RpdSshAuth(const char *sstr)
{
   // Authenitcation via ssh.

   gAuth = 0;

   if (gDebug > 2)
      ErrorInfo("RpdSshAuth: contacted by host: %s for user %s",
                gOpenHost.c_str(),sstr);

   // Decode subject string
   char *User = new char[strlen(sstr)];
   char PipeId[10];
   int Ulen, ofs, opt;
   char dumm[20];
   sscanf(sstr, "%d %d %d %s %d %s %s", &gRemPid, &ofs, &opt, PipeId, &Ulen,
          User, dumm);
   User[Ulen] = '\0';
   gReUseRequired = (opt & kAUTH_REUSE_MSK);

   // Check if we have been called to notify failure ...
   if (gRemPid < 0) {

      if (gDebug > 2)
         ErrorInfo
             ("RpdSshAuth: this is a failure notification (%s,%s,%d,%s)",
              User, gOpenHost.c_str(), gRemPid, PipeId);

      struct passwd *pw = getpwnam(User);
      if (pw) {
         std::string PipeFile =
            std::string(pw->pw_dir) + std::string("/RootSshPipe.") + PipeId;
         if (access(PipeFile.c_str(),R_OK) && GetErrno() == ENOENT)
            PipeFile= gTmpDir + std::string("/RootSshPipe.") + PipeId;

         if (!access(PipeFile.c_str(),R_OK)) {

            FILE *fpipe = fopen(PipeFile.c_str(), "r");
            char Pipe[kMAXPATHLEN];
            if (fpipe) {
               while (fgets(Pipe, sizeof(Pipe), fpipe)) {
                  if (Pipe[strlen(Pipe)-1] == '\n')
                  Pipe[strlen(Pipe)-1] = 0;
               }
               fclose(fpipe);
               // Remove the temporary file
               unlink(PipeFile.c_str());

               if (SshToolNotifyFailure(Pipe))
                  ErrorInfo("RpdSshAuth: failure notification perhaps"
                         " unsuccessful ... ");
            } else
               ErrorInfo("RpdSshAuth: cannot open file with pipe info"
                         " (errno= %d)",GetErrno());
         } else
            ErrorInfo("RpdSshAuth: unable to localize pipe file"
                      " (errno: %d)", GetErrno());
      } else
         ErrorInfo("RpdSshAuth: unable to get user info for '%s'"
                   " (errno: %d)",User,GetErrno());

      if (User) delete[] User;

      gClientProtocol = atoi(dumm);

      // Improved diagnostics, check if there is something listening on
      // port gSshdPort
      char buf[20] = {0};
#if defined(linux)
      if (!RpdCheckSshd(0)) {
         // Nothing found by netstat ... try opening a socket
         if (!RpdCheckSshd(1)) {
            if (gDebug > 2)
               ErrorInfo("RpdSshAuth: sshd not found - return");
            if (gClientProtocol > 9) {
               SPrintf(buf, 20, "%d",gSshdPort);
               NetSend(strlen(buf), kROOTD_SSH);
               NetSend(buf, strlen(buf), kMESS_STRING);
            }
            return;
         }
      }
#else
      if (!RpdCheckSshd(1)) {
         if (gDebug > 2)
            ErrorInfo("RpdSshAuth: sshd not found - return");
         if (gClientProtocol > 9) {
            SPrintf(buf, 20,"%d",gSshdPort);
            NetSend(strlen(buf), kROOTD_SSH);
            NetSend(buf, strlen(buf), kMESS_STRING);
         }
         return;
      }
#endif
      if (gClientProtocol > 9) {
         SPrintf(buf,20,"OK");
         NetSend(strlen(buf), kROOTD_SSH);
         NetSend(buf, strlen(buf), kMESS_STRING);
         ErrorInfo("RpdSshAuth: failure notified");
      } else {
         Error(gErrFatal,kErrAuthNotOK,"RpdSshAuth: failure notified");
      }
      return;
   }
   // Check user existence and get its environment
   struct passwd *pw = getpwnam(User);
   if (!pw) {
      ErrorInfo("RpdSshAuth: entry for user % not found in /etc/passwd",
                User);
      NetSend(-2, kROOTD_SSH);
      if (User) delete[] User;
      return;
   }
   // Method cannot be attempted for anonymous users ... (ie data servers )...
   if (!strcmp(pw->pw_shell, "/bin/false")) {
      ErrorInfo("RpdSshAuth: no SSH for anonymous user '%s' ", User);
      NetSend(-2, kROOTD_SSH);
      if (User) delete[] User;
      return;
   }

   // Now we create an internal (UNIX) socket to listen to the
   // result of sshd from ssh2rpd.
   // Path will be /tmp/rootdSSH_<random_string>
   int UnixFd;
   char *UniquePipe = new char[22];
   if ((UnixFd =
        SshToolAllocateSocket(pw->pw_uid, pw->pw_gid, &UniquePipe)) < 0) {
      ErrorInfo
          ("RpdSshAuth: can't allocate UNIX socket for authentication");
      NetSend(0, kROOTD_SSH);
      if (User) delete[] User;
      if (UniquePipe) delete[] UniquePipe;
      return;
   }

   // Open a file to put the pipe to be read by ssh2rpd
   int itmp = 0;
   char *PipeFile = new char[strlen(pw->pw_dir) + 25];
   sprintf(PipeFile, "%s/RootSshPipe.XXXXXX", pw->pw_dir);
   int ipipe = mkstemp(PipeFile);
   if (ipipe == -1) {
      delete[] PipeFile;
      char *PipeFile = new char[gTmpDir.length() + 25];
      sprintf(PipeFile, "%s/RootSshPipe.XXXXXX", gTmpDir.c_str());
      ipipe = mkstemp(PipeFile);
      itmp = 1;
   }
   FILE *fpipe = 0;
   if (ipipe == -1 || !(fpipe = fdopen(ipipe,"w")) ) {
      ErrorInfo("RpdSshAuth: failure creating pipe file %s (errno: %d)",
                 PipeFile,GetErrno());
      // Could not open the file: notify failure and close
      // properly everything
      if (SshToolNotifyFailure(UniquePipe))
         ErrorInfo("RpdSshAuth: failure notification perhaps"
                   " unsuccessful ... ");
      NetSend(kErrNoPipeInfo, kROOTD_ERR);
      if (User) delete[] User;
      if (UniquePipe) delete[] UniquePipe;
      if (PipeFile) delete[] PipeFile;
      return;
   } else {
      // File open: fill it
      fprintf(fpipe,"%s\n",UniquePipe);
      fclose(fpipe);
      // Set strict protections
      chmod(PipeFile, 0600);
      // Set ownership of the pipe file to the user
      if (getuid() == 0)
         if (chown(PipeFile,pw->pw_uid,pw->pw_gid) == -1)
            ErrorInfo("RpdSshAuth: cannot change ownership of %s (errno: %d)",
                      PipeFile,GetErrno());
   }

   // Get ID
   strcpy(PipeId,(char *)strstr(PipeFile,"SshPipe.")+strlen("SshPipe."));

   // Communicate command to be executed via ssh ...
   std::string rootbindir;
   if (getenv("ROOTBINDIR"))
      rootbindir = getenv("ROOTBINDIR");
   char dbgstr[4] = {0};
   snprintf(dbgstr,3,"%d ",gDebug);
   std::string CmdInfo = std::string(rootbindir).append("/ssh2rpd ");
   CmdInfo.append(dbgstr);
   CmdInfo.append(" ");
   CmdInfo.append(PipeId);

   // Add Tmp dir, if used
   if (itmp) {
      CmdInfo.append(" ");
      CmdInfo.append(gTmpDir);
   }

   // Add non-standard port, if so
   if (gSshdPort != 22) {
      char sshp[10];
      snprintf(sshp,10," p:%d",gSshdPort);
      CmdInfo.append(sshp);
   }

   if (gDebug > 2)
      ErrorInfo("RpdSshAuth: sending CmdInfo (%d) %s", CmdInfo.length(),
                CmdInfo.c_str());
   NetSend(CmdInfo.length(), kROOTD_SSH);
   NetSend(CmdInfo.c_str(), CmdInfo.length(), kROOTD_SSH);

   // Wait for verdict from sshd (via ssh2rpd ...)
   // Additional check on the username ...
   gAuth = SshToolGetAuth(UnixFd, User);

   // Close socket
   SshToolDiscardSocket(UniquePipe, UnixFd);

   // If failure, notify and return ...
   if (gAuth <= 0) {
      if (gAuth == -1)
         NetSend(kErrWrongUser, kROOTD_ERR);  // Send message length first
      else
         NetSend(kErrAuthNotOK, kROOTD_ERR);  // Send message length first
      if (User) delete[] User;
      if (UniquePipe) delete[] UniquePipe;
      if (PipeFile) delete[] PipeFile;
      // Set to Auth failed
      gAuth = 0;
      return;
   }
   // notify the client
   if (gDebug > 0 && gAuth == 1)
      ErrorInfo("RpdSshAuth: user %s authenticated by sshd", User);
   gSec = 4;

   // Save username ...
   strcpy(gUser, User);

   char line[kMAXPATHLEN];
   if ((gReUseAllow & kAUTH_SSH_MSK) && gReUseRequired) {

      // Ask for the RSA key
      NetSend(1, kROOTD_RSAKEY);

      // Receive the key securely
      if (RpdRecvClientRSAKey()) {
         ErrorInfo("RpdSshAuth: could not import a valid key"
                   " - switch off reuse for this session");
         gReUseRequired = 0;
      }

      // Set an entry in the auth tab file for later (re)use, if required ...
      int OffSet = -1;
      char *token = 0;
      if (gReUseRequired) {
         int Act = gInclusiveToken ? 1 : 2;
         SPrintf(line, kMAXPATHLEN, "4 %d %d %d %d %s %s",
                 Act, gRSAKey, gParentId, gRemPid, gOpenHost.c_str(), gUser);
         OffSet = RpdUpdateAuthTab(1, line, &token);
      }
      // Comunicate login user name to client
      SPrintf(line, kMAXPATHLEN, "%s %d", gUser, OffSet);
      NetSend(strlen(line), kROOTD_SSH);   // Send message length first
      NetSend(line, kMESS_STRING);

      if (gReUseRequired) {
         // Send over the token
         if (RpdSecureSend(token) == -1) {
            ErrorInfo
                ("RpdSshAuth: problems secure-sending token"
                 " - may result in corrupted token");
         }
         if (token) delete[] token;

         // Save RSA public key into file for later use by other rootd/proofd
         RpdSavePubKey(gPubKey, OffSet, gUser);
      }
      gOffSet = OffSet;
   } else {
      // Comunicate login user name to client
      SPrintf(line, kMAXPATHLEN, "%s -1", gUser);
      NetSend(strlen(line), kROOTD_SSH);   // Send message length first
      NetSend(line, kMESS_STRING);
   }

   // Release allocated memory
   if (User)       delete[] User;
   if (UniquePipe) delete[] UniquePipe;
   if (PipeFile)   delete[] PipeFile;

   return;
}

//______________________________________________________________________________
void RpdKrb5Auth(const char *sstr)
{
   // Authenticate via Kerberos.

   gAuth = 0;

#ifdef R__KRB5
   NetSend(1, kROOTD_KRB5);
   // TAuthenticate will respond to our encouragement by sending krb5
   // authentication through the socket

   int retval;

   if (gDebug > 2)
      ErrorInfo("RpdKrb5Auth: analyzing ... %s", sstr);

   if (gClientProtocol > 8) {
      int Ulen, ofs, opt;
      char dumm[256];
      // Decode subject string
      sscanf(sstr, "%d %d %d %d %s", &gRemPid, &ofs, &opt, &Ulen, dumm);
      gReUseRequired = (opt & kAUTH_REUSE_MSK);
   }

   // Init context
   retval = krb5_init_context(&gKcontext);
   if (retval) {
      ErrorInfo("RpdKrb5Auth: %s while initializing krb5",
            error_message(retval));
      return;
   }

   // Use special Keytab file, if specified
   if (gKeytabFile.length()) {
      if ((retval = krb5_kt_resolve(gKcontext, gKeytabFile.c_str(), &gKeytab)))
         ErrorInfo("RpdKrb5Auth: %s while resolving keytab file %s",
                        error_message(retval),gKeytabFile.c_str());
   }

   // get service principal
   const char *service = "host";

   if (gDebug > 2)
      ErrorInfo("RpdKrb5Auth: using service: %s ",service);

   krb5_principal server;
   if ((retval = krb5_sname_to_principal(gKcontext, 0, service,
                                         KRB5_NT_SRV_HST, &server))) {
      ErrorInfo("RpdKrb5Auth: while generating service name (%s): %d %s",
                service, retval, error_message(retval));
      RpdFreeKrb5Vars(gKcontext, 0, 0, 0, 0);
      return;
   }

   // listen for authentication from the client
   krb5_auth_context auth_context = 0;
   krb5_ticket *ticket;
   char proto_version[100] = "krootd_v_1";
   int sock = NetGetSockFd();

   if (gDebug > 2)
      ErrorInfo("RpdKrb5Auth: recvauth ... ");

   if ((retval = krb5_recvauth(gKcontext, &auth_context,
                               (krb5_pointer) &sock, proto_version, server,
                               0, gKeytab,   // default gKeytab is 0
                               &ticket))) {
      ErrorInfo("RpdKrb5Auth: recvauth failed--%s", error_message(retval));
      RpdFreeKrb5Vars(gKcontext, server, 0, 0, 0);
      return;
   }

   // get client name
   char *cname;
   if ((retval =
        krb5_unparse_name(gKcontext, ticket->enc_part2->client, &cname))) {
      ErrorInfo("RpdKrb5Auth: unparse failed: %s", error_message(retval));
      RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, 0);
      return;
   }
   if (gDebug > 2)
         ErrorInfo("RpdKrb5Auth: name in ticket is: %s",cname);

   using std::string;
   std::string user = std::string(cname);
   free(cname);
   std::string reply = std::string("authenticated as ").append(user);

   // set user name
#if 0
   user = user.erase(user.find("@"));   // cut off realm
   string::size_type pos = user.find("/");   // see if there is an instance
   if (pos != string::npos)
      user = user.erase(pos);   // drop the instance
   strncpy(gUser, user.c_str(), 64);
#else
   // avoid using 'erase' (it is buggy with some compilers)
   snprintf(gUser,64,"%s",user.c_str());
   char *pc = 0;
   // cut off realm
   if ((pc = (char *)strstr(gUser,"@")))
      *pc = '\0';
   // drop instances, if any
   if ((pc = (char *)strstr(gUser,"/")))
      *pc = '\0';
#endif

   std::string targetUser = std::string(gUser);

   if (gClientProtocol >= 9) {

       // Receive target name

      if (gDebug > 2)
         ErrorInfo("RpdKrb5Auth: receiving target user ... ");

       EMessageTypes kind;
       char buffer[66];
       NetRecv(buffer, 65, kind);

       if (kind != kROOTD_KRB5) {
          ErrorInfo("RpdKrb5Auth: protocol error, received message"
                    " of type %d instead of %d\n",kind,kROOTD_KRB5);
       }
       buffer[65] = 0;
       targetUser = std::string(buffer);

      if (gDebug > 2)
         ErrorInfo("RpdKrb5Auth: received target user %s ",buffer);
   }

   // const char* targetUser = gUser; // "cafuser";
   if (krb5_kuserok(gKcontext, ticket->enc_part2->client,
                                     targetUser.c_str())) {
      if (gDebug > 2)
      ErrorInfo("RpdKrb5Auth: change user from %s to %s successful",
                gUser,targetUser.c_str());
      snprintf(gUser,64,"%s",targetUser.c_str());
      reply =  std::string("authenticated as ").append(gUser);
   } else {
      ErrorInfo("RpdKrb5Auth: could not change user from %s to %s",
                gUser,targetUser.c_str());
      ErrorInfo("RpdKrb5Auth: continuing with user: %s",gUser);
   }

   if (gClientProtocol >= 9) {

      char *data = 0;
      int size = 0;
      if (gDebug > 2)
         ErrorInfo("RpdKrb5Auth: receiving forward cred ... ");

      {
         EMessageTypes kind;
         char BufLen[20];
         NetRecv(BufLen, 20, kind);

         if (kind != kROOTD_KRB5) {
            ErrorInfo("RpdKrb5Auth: protocol error, received"
                      " message of type %d instead of %d\n",
                      kind, kROOTD_KRB5);
         }

         size = atoi(BufLen);
         if (gDebug > 3)
            ErrorInfo("RpdKrb5Auth: got len '%s' %d ", BufLen, size);

         data = new char[size+1];

         // Receive and decode encoded public key
         int Nrec = NetRecvRaw(data, size);

         if (gDebug > 3)
            ErrorInfo("RpdKrb5Auth: received %d ", Nrec);
      }

      krb5_data forwardCreds;
      forwardCreds.data = data;
      forwardCreds.length = size;

      if (gDebug > 2)
         ErrorInfo("RpdKrb5Auth: received forward cred ... %d %d %d",
                   data[0], data[1], data[2]);

      int net = sock;
      retval = krb5_auth_con_genaddrs(gKcontext, auth_context, net,
                   KRB5_AUTH_CONTEXT_GENERATE_REMOTE_FULL_ADDR);
      if (retval) {
         ErrorInfo("RpdKrb5Auth: failed auth_con_genaddrs is: %s\n",
                   error_message(retval));
      }

      krb5_creds **creds = 0;
      if ((retval = krb5_rd_cred(gKcontext, auth_context,
                                 &forwardCreds, &creds, 0))) {
         ErrorInfo("RpdKrb5Auth: rd_cred failed--%s", error_message(retval));
         RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, 0);
         return;
      }
      if (data) delete[] data;

      struct passwd *pw = getpwnam(gUser);
      if (pw) {
         Int_t fromUid = getuid();

         if (setresuid(pw->pw_uid, pw->pw_uid, fromUid) == -1) {
            ErrorInfo("RpdKrb5Auth: can't setuid for user %s", gUser);
            NetSend(kErrNotAllowed, kROOTD_ERR);
            RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, creds);
            return;
         }

         if (gDebug>5)
            ErrorInfo("RpdKrb5Auth: saving ticket to cache ...");

         krb5_context context;
         // Init context
         retval = krb5_init_context(&context);
         if (retval) {
            ErrorInfo("RpdKrb5Auth: %s while initializing second krb5",
                error_message(retval));
            NetSend(kErrNotAllowed, kROOTD_ERR);
            RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, creds);
            return;
         }

         krb5_ccache cache = 0;
         if ((retval = krb5_cc_default(context, &cache))) {
            ErrorInfo("RpdKrb5Auth: cc_default failed--%s",
                      error_message(retval));
            NetSend(kErrNotAllowed, kROOTD_ERR);
            krb5_free_context(context);
            RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, creds);
            return;
         }

         if (gDebug>5)
            ErrorInfo("RpdKrb5Auth: working (1) on ticket to cache (%s) ... ",
                      krb5_cc_get_name(context,cache));

         // this is not working (why?)
         // this would mean that if a second user comes in, it will tremple
         // the existing one :(
         //       if ((retval = krb5_cc_gen_new(context,&cache))) {
         //          ErrorInfo("RpdKrb5Auth: cc_gen_new failed--%s",
         //                    error_message(retval));
         //          return;
         //       }

         const char *cacheName = krb5_cc_get_name(context,cache);

         if (gDebug>5)
            ErrorInfo("RpdKrb5Auth: working (2) on ticket"
                      " to cache (%s) ... ",cacheName);

         if ((retval = krb5_cc_initialize(context,cache,
                                          ticket->enc_part2->client))) {
            ErrorInfo("RpdKrb5Auth: cc_initialize failed--%s",
                      error_message(retval));
            RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, creds);
            krb5_free_context(context);
            NetSend(kErrNotAllowed, kROOTD_ERR);
            return;
         }

         if ((retval = krb5_cc_store_cred(context,cache, *creds))) {
            ErrorInfo("RpdKrb5Auth: cc_store_cred failed--%s",
                       error_message(retval));
            NetSend(kErrNotAllowed, kROOTD_ERR);
            krb5_free_context(context);
            RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, creds);
            return;
         }

         if ((retval = krb5_cc_close(context,cache))) {
            ErrorInfo("RpdKrb5Auth: cc_close failed--%s",
                       error_message(retval));
            NetSend(kErrNotAllowed, kROOTD_ERR);
            krb5_free_context(context);
            RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, creds);
            return;
         }

         // free context
         krb5_free_context(context);

         //       if ( chown( cacheName, pw->pw_uid, pw->pw_gid) != 0 ) {
         //          ErrorInfo("RpdKrb5Auth: could not change the owner"
         //                    " ship of the cache file %s",cacheName);
         //       }

         if (gDebug>5)
            ErrorInfo("RpdKrb5Auth: done ticket to cache (%s) ... ",
                      cacheName);

         if (setresuid(fromUid,fromUid, fromUid) == -1) {
            ErrorInfo("RpdKrb5Auth: can't setuid back to original uid");
            NetSend(kErrNotAllowed, kROOTD_ERR);
            return;
         }
      }

      // free creds
      krb5_free_tgt_creds(gKcontext,creds);

   }

   NetSend(reply.c_str(), kMESS_STRING);

   // free allocated stuff
   RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, (krb5_creds **)0);

   // Authentication was successfull
   gAuth = 1;
   gSec = 2;

   if (gClientProtocol > 8) {

      char line[kMAXPATHLEN];
      if ((gReUseAllow & kAUTH_KRB_MSK) && gReUseRequired) {

         // Ask for the RSA key
         NetSend(1, kROOTD_RSAKEY);

         // Receive the key securely
         if (RpdRecvClientRSAKey()) {
            ErrorInfo("RpdKrb5Auth: could not import a valid key"
                      " - switch off reuse for this session");
            gReUseRequired = 0;
         }

         // Set an entry in the auth tab file for later (re)use,
         // if required ...
         int OffSet = -1;
         char *token = 0;
         if (gReUseRequired) {
            int Act = gInclusiveToken ? 1 : 2;
            SPrintf(line, kMAXPATHLEN, "2 %d %d %d %d %s %s",
                    Act, gRSAKey, gParentId, gRemPid, gOpenHost.c_str(), gUser);
            OffSet = RpdUpdateAuthTab(1, line, &token);
            if (gDebug > 2)
               ErrorInfo("RpdKrb5Auth: line:%s OffSet:%d", line, OffSet);
         }
         // Comunicate login user name to client
         SPrintf(line, kMAXPATHLEN, "%s %d", gUser, OffSet);
         NetSend(strlen(line), kROOTD_KRB5);   // Send message length first
         NetSend(line, kMESS_STRING);

         // Send Token
         if (gReUseRequired) {
            if (RpdSecureSend(token) == -1) {
               ErrorInfo("RpdKrb5Auth: problems secure-sending token"
                         " - may result in corrupted token");
            }
            if (token) delete[] token;

            // Save RSA public key into file for later use by
            // other daemon server
            RpdSavePubKey(gPubKey, OffSet, gUser);
         }
         gOffSet = OffSet;

      } else {

         // Comunicate login user name to client
         SPrintf(line, kMAXPATHLEN, "%s -1", gUser);
         NetSend(strlen(line), kROOTD_KRB5);   // Send message length first
         NetSend(line, kMESS_STRING);

      }
   } else {
      NetSend(user.c_str(), kMESS_STRING);
   }

   if (gDebug > 0)
      ErrorInfo("RpdKrb5Auth: user %s authenticated", gUser);

#else

   // no krb5 support
   if (sstr) { }   // remove compiler warning

   NetSend(0, kROOTD_KRB5);

#endif
}

//______________________________________________________________________________
void RpdSRPUser(const char *sstr)
{
   // Use Secure Remote Password protocol.
   // Check user id in $HOME/.srootdpass file.

   gAuth = 0;

   if (!*sstr) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: bad user name");
      return;
   }

#ifdef R__SRP

   // Decode subject string
   char *user = new char[strlen(sstr) + 1];
   if (gClientProtocol > 8) {
      int Ulen, ofs, opt;
      char dumm[20];
      sscanf(sstr, "%d %d %d %d %s %s", &gRemPid, &ofs, &opt, &Ulen, user,
             dumm);
      user[Ulen] = '\0';
      gReUseRequired = (opt & kAUTH_REUSE_MSK);
   } else {
      strcpy(user, sstr);
   }

   struct passwd *pw = getpwnam(user);
   if (!pw) {
      NetSend(kErrNoUser, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: user %s unknown", user);
      return;
   }
   // Method cannot be attempted for anonymous users ... (ie data servers )...
   if (!strcmp(pw->pw_shell, "/bin/false")) {
      NetSend(kErrNotAllowed, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: no SRP for anonymous user '%s' ", user);
      return;
   }
   // If server is not started as root and user is not same as the
   // one who started rootd then authetication is not ok.
   uid_t uid = getuid();
   if (uid && uid != pw->pw_uid) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: user not same as effective user of rootd");
      return;
   }

   NetSend(gAuth, kROOTD_AUTH);

   strcpy(gUser, user);

   if (user) delete[] user;

   std::string srootdpass, srootdconf;
   if (gAltSRPPass.length()) {
      srootdpass = gAltSRPPass;
   } else {
      srootdpass = std::string(pw->pw_dir).append(kSRootdPass);
   }
   srootdconf = srootdpass + std::string(".conf");

   FILE *fp1 = fopen(srootdpass.c_str(), "r");
   if (!fp1) {
      NetSend(kErrFileOpen, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: error opening %s", srootdpass.c_str());
      return;
   }
   FILE *fp2 = fopen(srootdconf.c_str(), "r");
   if (!fp2) {
      NetSend(kErrFileOpen, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: error opening %s", srootdconf.c_str());
      if (fp1)
         fclose(fp1);
      return;
   }

   struct t_pw *tpw = t_openpw(fp1);
   if (!tpw) {
      NetSend(kErrFileOpen, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: unable to open password file %s",
                srootdpass.c_str());
      fclose(fp1);
      fclose(fp2);
      return;
   }

   struct t_conf *tcnf = t_openconf(fp2);
   if (!tcnf) {
      NetSend(kErrFileOpen, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: unable to open configuration file %s",
                srootdconf.c_str());
      t_closepw(tpw);
      fclose(fp1);
      fclose(fp2);
      return;
   }
#if R__SRP_1_1
   struct t_server *ts = t_serveropen(gUser, tpw, tcnf);
#else
   struct t_server *ts = t_serveropenfromfiles(gUser, tpw, tcnf);
#endif
   if (!ts) {
      NetSend(kErrNoUser, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: user %s not found SRP password file", gUser);
      return;
   }

   if (tcnf)
      t_closeconf(tcnf);
   if (tpw)
      t_closepw(tpw);
   if (fp2)
      fclose(fp2);
   if (fp1)
      fclose(fp1);

   char hexbuf[MAXHEXPARAMLEN];

   // send n to client
   NetSend(t_tob64(hexbuf, (char *) ts->n.data, ts->n.len), kROOTD_SRPN);
   // send g to client
   NetSend(t_tob64(hexbuf, (char *) ts->g.data, ts->g.len), kROOTD_SRPG);
   // send salt to client
   NetSend(t_tob64(hexbuf, (char *) ts->s.data, ts->s.len),
           kROOTD_SRPSALT);

   struct t_num *B = t_servergenexp(ts);

   // receive A from client
   EMessageTypes kind;
   if (NetRecv(hexbuf, MAXHEXPARAMLEN, kind) < 0) {
      NetSend(kErrFatal, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: error receiving A from client");
      return;
   }
   if (kind != kROOTD_SRPA) {
      NetSend(kErrFatal, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: expected kROOTD_SRPA message");
      return;
   }

   unsigned char buf[MAXPARAMLEN];
   struct t_num A;
   A.data = buf;
   A.len = t_fromb64((char *) A.data, hexbuf);

   // send B to client
   NetSend(t_tob64(hexbuf, (char *) B->data, B->len), kROOTD_SRPB);

   t_servergetkey(ts, &A);

   // receive response from client
   if (NetRecv(hexbuf, MAXHEXPARAMLEN, kind) < 0) {
      NetSend(kErrFatal, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: error receiving response from client");
      return;
   }
   if (kind != kROOTD_SRPRESPONSE) {
      NetSend(kErrFatal, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: expected kROOTD_SRPRESPONSE message");
      return;
   }

   unsigned char cbuf[20];
   t_fromhex((char *) cbuf, hexbuf);

   if (!t_serververify(ts, cbuf)) {

      // authentication successful
      if (gDebug > 0)
         ErrorInfo("RpdSRPUser: user %s authenticated", gUser);
      gAuth = 1;
      gSec = 1;

      if (gClientProtocol > 8) {

         char line[kMAXPATHLEN];
         if ((gReUseAllow & kAUTH_SRP_MSK) && gReUseRequired) {

            // Ask for the RSA key
            NetSend(1, kROOTD_RSAKEY);

            // Receive the key securely
            if (RpdRecvClientRSAKey()) {
               ErrorInfo
                   ("RpdSRPAuth: could not import a valid key"
                    " - switch off reuse for this session");
               gReUseRequired = 0;
            }

            // Set an entry in the auth tab file for later (re)use, if required ...
            int OffSet = -1;
            char *token = 0;
            if (gReUseRequired) {
               int Act = gInclusiveToken ? 1 : 2;
               SPrintf(line, kMAXPATHLEN, "1 %d %d %d %d %s %s",
                       Act, gRSAKey, gParentId, gRemPid,
                       gOpenHost.c_str(), gUser);
               OffSet = RpdUpdateAuthTab(1, line, &token);
            }
            // Comunicate login user name to client
            SPrintf(line, kMAXPATHLEN, "%s %d", gUser, OffSet);
            NetSend(strlen(line), kROOTD_SRPUSER);   // Send message length first
            NetSend(line, kMESS_STRING);

            if (gReUseRequired) {
               // Send Token
               if (RpdSecureSend(token) == -1) {
                  ErrorInfo("RpdSRPUser: problems secure-sending token"
                            " - may result in corrupted token");
               }
               if (token) delete[] token;

               // Save RSA public key into file for later use by other rootd/proofd
               RpdSavePubKey(gPubKey, OffSet, gUser);
            }
            gOffSet = OffSet;

         } else {
            // Comunicate login user name to client
            SPrintf(line, kMAXPATHLEN, "%s -1", gUser);
            NetSend(strlen(line), kROOTD_SRPUSER);   // Send message length first
            NetSend(line, kMESS_STRING);
         }

      }

   } else {
      if (gClientProtocol > 8) {
         NetSend(kErrBadPasswd, kROOTD_ERR);
         ErrorInfo("RpdSRPUser: authentication failed for user %s", gUser);
         return;
      }
   }

   t_serverclose(ts);

#else
   NetSend(0, kROOTD_SRPUSER);
#endif
}

//______________________________________________________________________________
int RpdCheckSpecialPass(const char *passwd)
{
   // Check user's password against password in $HOME/.rootdpass. If matches
   // skip other authentication mechanism. Returns 1 in case of success
   // authentication, 0 otherwise.

   char rootdpass[kMAXPATHLEN];

   struct passwd *pw = getpwnam(gUser);

   SPrintf(rootdpass, kMAXPATHLEN, "%s/%s", pw->pw_dir, kRootdPass.c_str());

   int fid = open(rootdpass, O_RDONLY);
   if (fid == -1)
      return 0;

   int n;
   if ((n = read(fid, rootdpass, sizeof(rootdpass) - 1)) <= 0) {
      close(fid);
      return 0;
   }
   close(fid);

   rootdpass[n] = 0;
   char *s = strchr(rootdpass, '\n');
   if (s)
      *s = 0;

   if (gClientProtocol > 8) {
      n = strlen(rootdpass);
      if (strncmp(passwd, rootdpass, n + 1) != 0)
         return 0;
   } else {
      char *pass_crypt = crypt(passwd, rootdpass);
      n = strlen(rootdpass);
      if (strncmp(pass_crypt, rootdpass, n+1) != 0)
         return 0;
   }

   if (gDebug > 0)
      ErrorInfo
          ("RpdCheckSpecialPass: user %s authenticated via ~/.rootdpass",
           gUser);

   return 1;
}

//______________________________________________________________________________
void RpdPass(const char *pass)
{
   // Check user's password.

   char passwd[64];
   char *passw;
   char *pass_crypt;
   struct passwd *pw;
#ifdef R__SHADOWPW
   struct spwd *spw;
#endif
#ifdef R__AFS
   char *reason;
   int afs_auth = 0;
#endif

   if (gDebug > 2)
      ErrorInfo("RpdPass: Enter");

   gAuth = 0;
   if (!*gUser) {
      NetSend(kErrFatal, kROOTD_ERR);
      ErrorInfo("RpdPass: user needs to be specified first");
      return;
   }

   int n = strlen(pass);
   // Passwd length should be in the correct range ...
   if (!n) {
      NetSend(kErrBadPasswd, kROOTD_ERR);
      ErrorInfo("RpdPass: null passwd not allowed");
      return;
   }
   if (n > (int) sizeof(passwd)) {
      NetSend(kErrBadPasswd, kROOTD_ERR);
      ErrorInfo("RpdPass: passwd too long");
      return;
   }
   // Inversion is done in RpdUser, if needed
   strcpy(passwd, pass);

   // Special treatment for anonimous ...
   if (gAnon) {
      strcpy(gPasswd, passwd);
      goto authok;
   }
   // ... and SpecialPass ...
   if (RpdCheckSpecialPass(passwd)) {
      goto authok;
   }
   // Get local passwd info for gUser
   pw = getpwnam(gUser);

#ifdef R__AFS
   afs_auth = !ka_UserAuthenticateGeneral(KA_USERAUTH_VERSION + KA_USERAUTH_DOSETPAG,
                                          gUser,        //user name
                                          (char *) 0,   //instance
                                          (char *) 0,   //realm
                                          passwd,       //password
                                          0,            //default lifetime
                                          0, 0,         //two spares
                                          &reason);     //error string

   if (!afs_auth) {
      if (gDebug > 0)
         ErrorInfo("RpdPass: AFS login failed for user %s: %s",
                    gUser, reason);
      // try conventional login...
#endif

#ifdef R__SHADOWPW
      // System V Rel 4 style shadow passwords
      if ((spw = getspnam(gUser)) == 0) {
         ErrorInfo("RpdPass: Shadow passwd not available for user %s",
                   gUser);
         passw = pw->pw_passwd;
      } else
         passw = spw->sp_pwdp;
#else
      passw = pw->pw_passwd;
#endif
      if (gClientProtocol <= 8 || !gSaltRequired) {
         char Salt[20] = {0};
         int Slen = 2;
         if (!strncmp(passw, "$1$", 3)) {
            // Shadow passwd
            char *pd = strstr(passw + 4, "$");
            Slen = (int) (pd - passw);
            strncpy(Salt, passw, Slen);
         } else
            strncpy(Salt, passw, Slen);
         Salt[Slen] = 0;
         pass_crypt = crypt(passwd, Salt);   // Comment this
      } else {
         pass_crypt = passwd;
      }
      n = strlen(passw);

      if (strncmp(pass_crypt, passw, n + 1) != 0) {
         NetSend(kErrBadPasswd, kROOTD_ERR);
         ErrorInfo("RpdPass: invalid password for user %s", gUser);
         return;
      }
      if (gDebug > 2)
         ErrorInfo("RpdPass: valid password for user %s", gUser);
#ifdef R__AFS
   } else                            // afs_auth
      if (gDebug > 2)
         ErrorInfo("RpdPass: AFS login successful for user %s", gUser);
#endif

 authok:
   gAuth = 1;
   gSec = 0;

   if (gClientProtocol > 8) {
      // Set an entry in the auth tab file for later (re)use, if required ...
      int OffSet = -1;
      char *token = 0;
      char line[kMAXPATHLEN];
      if ((gReUseAllow & kAUTH_CLR_MSK) && gReUseRequired) {

         int Act = gInclusiveToken ? 1 : 2;
         SPrintf(line, kMAXPATHLEN, "0 %d %d %d %d %s %s",
                 Act, gRSAKey, gParentId, gRemPid,
                 gOpenHost.c_str(), gUser);
         OffSet = RpdUpdateAuthTab(1, line, &token);
         if (gDebug > 2)
            ErrorInfo("RpdPass: got offset %d", OffSet);

         // Comunicate login user name to client
         SPrintf(line, kMAXPATHLEN, "%s %d", gUser, OffSet);
         if (gDebug > 2)
            ErrorInfo("RpdPass: sending back line %s", line);
         NetSend(strlen(line), kROOTD_PASS);   // Send message length first
         NetSend(line, kMESS_STRING);

         if (gDebug > 2)
            ErrorInfo("RpdPass: sending token %s (Crypt: %d)", token,
                      gCryptRequired);
         if (gCryptRequired) {
            // Send over the token
            if (RpdSecureSend(token) == -1) {
               ErrorInfo("RpdPass: problems secure-sending token"
                         " - may result in corrupted token");
            }
         } else {
            // Send token inverted
            for (int i = 0; i < (int) strlen(token); i++) {
               token[i] = ~token[i];
            }
            NetSend(token, kMESS_STRING);
         }
         if (token) delete[] token;
         gOffSet = OffSet;

      } else {
         // Comunicate login user name to client
         SPrintf(line, kMAXPATHLEN, "%s -1", gUser);
         if (gDebug > 2)
            ErrorInfo("RpdPass: sending back line %s", line);
         NetSend(strlen(line), kROOTD_PASS);   // Send message length first
         NetSend(line, kMESS_STRING);
      }

      if (gCryptRequired) {
         // Save RSA public key into file for later use by other rootd/proofd
         RpdSavePubKey(gPubKey, OffSet, gUser);
      }
   }
}

//______________________________________________________________________________
void RpdGlobusAuth(const char *sstr)
{
   // Authenticate via Globus.

   gAuth = 0;

#ifndef R__GLBS

   if (sstr) { }  // use sstr
   NetSend(0, kROOTD_GLOBUS);
   return;

#else

   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;
   OM_uint32 GssRetFlags = 0;
   gss_ctx_id_t GlbContextHandle = GSS_C_NO_CONTEXT;
   gss_cred_id_t GlbCredHandle = GSS_C_NO_CREDENTIAL;
   gss_cred_id_t GlbDelCredHandle = GSS_C_NO_CREDENTIAL;
   int GlbTokenStatus = 0;
   char *GlbClientName;
   FILE *FILE_SockFd;
   char *gridmap_default = "/etc/grid-security/grid-mapfile";
   EMessageTypes kind;
   int lSubj, OffSet = -1;
   char *user = 0;
   int ulen = 0;

   if (gDebug > 2)
      ErrorInfo("RpdGlobusAuth: contacted by host: %s", gOpenHost.c_str());

   // Tell the remote client that we may accept Globus credentials ...
   NetSend(1, kROOTD_GLOBUS);

   // Decode subject string
   char *Subj = new char[strlen(sstr) + 1];
   int opt;
   char dumm[20];
   sscanf(sstr, "%d %d %d %d %s %s", &gRemPid, &OffSet, &opt, &lSubj, Subj,
          dumm);
   Subj[lSubj] = '\0';
   gReUseRequired = (opt & kAUTH_REUSE_MSK);
   if (gDebug > 2)
      ErrorInfo("RpdGlobusAuth: gRemPid: %d, Subj: %s (%d %d)", gRemPid,
                Subj, lSubj, strlen(Subj));
   if (Subj) delete[] Subj;   // GlbClientName will be determined from the security context ...

   // Now wait for client to communicate the issuer name of the certificate ...
   char *answer = new char[20];
   NetRecv(answer, (int) sizeof(answer), kind);
   if (kind != kMESS_STRING) {
      Error(gErr, kErrAuthNotOK,
            "RpdGlobusAuth: client_issuer_name:received unexpected"
            " type of message (%d)",kind);
      if (answer) delete[] answer;
      return;
   }
   int client_issuer_name_len = atoi(answer);
   if (answer) delete[] answer;
   char *client_issuer_name = new char[client_issuer_name_len + 1];
   NetRecv(client_issuer_name, client_issuer_name_len, kind);
   if (kind != kMESS_STRING) {
      Error(gErr, kErrAuthNotOK,
            "RpdGlobusAuth: client_issuer_name:received unexpected"
            " type of message (%d)",kind);
      if (client_issuer_name) delete[] client_issuer_name;
      return;
   }
   if (gDebug > 2)
      ErrorInfo("RpdGlobusAuth: client issuer name is: %s",
                client_issuer_name);

   // Inquire Globus credentials:
   // This is looking to file X509_USER_CERT for valid a X509 cert (default
   // /etc/grid-security/hostcert.pem) and to dir X509_CERT_DIR for trusted CAs
   // (default /etc/grid-security/certificates).
   if ((MajStat =
        globus_gss_assist_acquire_cred(&MinStat, GSS_C_ACCEPT,
                                       &GlbCredHandle)) !=
       GSS_S_COMPLETE) {
      GlbsToolError("RpdGlobusAuth: gss_assist_acquire_cred", MajStat,
                    MinStat, 0);
      if (getuid() > 0)
         ErrorInfo("RpdGlobusAuth: non-root: make sure you have"
                   " initialized (manually) your proxies");
      return;
   }

   // Now we open the certificates and we check if we are able to
   // autheticate the client. In the affirmative case we send our
   // subject name to the client ...
   // NB: we try first the user proxies; if it does not work we
   // try using the local host certificates; but only if we have
   // the rigth privileges
   char *subject_name;
   int CertRc = 0;
   CertRc = GlbsToolCheckProxy(client_issuer_name, &subject_name);
   if (CertRc && getuid() == 0)
     CertRc = GlbsToolCheckCert(client_issuer_name, &subject_name);

   if (CertRc) {
      ErrorInfo("RpdGlobusAuth: %s (%s)",
                "host does not seem to have certificate for the requested CA",
                 client_issuer_name);
      NetSend(0, kROOTD_GLOBUS);   // Notify that we did not find it
      if (client_issuer_name) delete[] client_issuer_name;
      return;
   } else {
      int sjlen = strlen(subject_name) + 1;
      //      subject_name[sjlen] = '\0';

      int bsnd = NetSend(sjlen, kROOTD_GLOBUS);
      if (gDebug > 2)
         ErrorInfo("RpdGlobusAuth: sent: %d (due >=%d))", bsnd,
                   2 * sizeof(sjlen));

      bsnd = NetSend(subject_name, sjlen, kMESS_STRING);
      if (gDebug > 2)
         ErrorInfo("RpdGlobusAuth: sent: %d (due >=%d))", bsnd, sjlen);

      free(subject_name);
   }
   // not needed anymore ...
   if (client_issuer_name) delete[] client_issuer_name;

   // We need to associate a FILE* stream with the socket
   // It will automatically closed when the socket will be closed ...
   FILE_SockFd = fdopen(NetGetSockFd(), "w+");

   // Now we are ready to start negotiating with the Client
   if ((MajStat =
        globus_gss_assist_accept_sec_context(&MinStat, &GlbContextHandle,
                                             GlbCredHandle, &GlbClientName,
                                             &GssRetFlags, 0,
                                             &GlbTokenStatus,
                                             &GlbDelCredHandle,
                                             globus_gss_assist_token_get_fd,
                                             (void *) FILE_SockFd,
                                             globus_gss_assist_token_send_fd,
                                             (void *) FILE_SockFd)) !=
       GSS_S_COMPLETE) {
      GlbsToolError("RpdGlobusAuth: gss_assist_accept_sec_context",
                    MajStat, MinStat, GlbTokenStatus);
      return;
   } else {
      gAuth = 1;
      gSec = 3;
      if (gDebug > 0)
         ErrorInfo("RpdGlobusAuth: user: %s \n authenticated",
                   GlbClientName);
   }

   // Check if there might be the need of credentials ...
   if (gService == kPROOFD) {
      // Check that we got delegation to autheticate the slaves
      if (GssRetFlags | GSS_C_DELEG_FLAG) {
         if (gDebug > 2)
            ErrorInfo("RpdGlobusAuth: Pointer to del cred is 0x%x",
                      (int) GlbDelCredHandle);
      } else {
         Error(gErr, kErrAuthNotOK,
               "RpdGlobusAuth: did not get delegated credentials (RetFlags: 0x%x)",
               GssRetFlags);
         return;
      }
      // Now we have to export these delegated credentials to a shared memory segment
      // for later use in 'proofserv' ...
      //   credential= (gss_buffer_t)malloc(sizeof(gss_buffer_desc));
      gss_buffer_t credential = new gss_buffer_desc;
      if ((MajStat =
           gss_export_cred(&MinStat, GlbDelCredHandle, 0, 0,
                           credential)) != GSS_S_COMPLETE) {
         GlbsToolError("RpdGlobusAuth: gss_export_cred", MajStat, MinStat,
                       0);
         return;
      } else if (gDebug > 2)
         ErrorInfo("RpdGlobusAuth: credentials prepared for export");

      // Now store it in shm for later use in proofserv ...
      int rc;
      if ((rc = GlbsToolStoreToShm(credential, &gShmIdCred))) {
         ErrorInfo
             ("RpdGlobusAuth: credentials not correctly stored in shm (rc: %d)",
              rc);
      }
      if (gDebug > 2)
         ErrorInfo
             ("RpdGlobusAuth: credentials stored in shared memory segment %d",
              gShmIdCred);

      delete credential;
   } else {
      if (gDebug > 2)
          ErrorInfo("RpdGlobusAuth: no need for delegated credentials (%s)",
                     gServName[gService].c_str());
   }

   // For Now we set the gUser to the certificate owner using the gridmap file ...
   // Should be understood if this is really necessary ...
   if (getenv("GRIDMAP") == 0) {
      // The installation did not specify a special location for the gridmap file
      // We assume the usual default ...
      setenv("GRIDMAP", gridmap_default, 1);
      if (gDebug > 2)
         ErrorInfo("RpdGlobusAuth: gridmap: using default file (%s)",
                   gridmap_default);
   } else if (gDebug > 2)
      ErrorInfo("RpdGlobusAuth: gridmap: using file %s",
                getenv("GRIDMAP"));

   // Get local login name for the subject ...
   char AnonUser[10] = "rootd";
   if (globus_gss_assist_gridmap(GlbClientName, &user)) {
      if (gDebug > 2)
         ErrorInfo
             ("RpdGlobusAuth: unable to get local username from gridmap: using: %s",
              AnonUser);
      user = strdup(AnonUser);
      if (gDebug > 2)
         ErrorInfo("RpdGlobusAuth: user ", user);
   }
   if (!strcmp(user, "anonymous"))
      user = strdup(AnonUser);
   if (!strcmp(user, AnonUser))
      gAnon = 1;

   // Fill gUser and free allocated memory
   ulen = strlen(user);
   strncpy(gUser, user, ulen + 1);

   char line[kMAXPATHLEN];
   if ((gReUseAllow & kAUTH_GLB_MSK) && gReUseRequired) {

      // Ask for the RSA key
      NetSend(1, kROOTD_RSAKEY);

      // Receive the key securely
      if (RpdRecvClientRSAKey()) {
         ErrorInfo
             ("RpdGlobusAuth: could not import a valid key"
              " - switch off reuse for this session");
         gReUseRequired = 0;
      }

      // Store security context and related info for later use ...
      OffSet = -1;
      char *token = 0;
      if (gReUseRequired) {
         int ShmId = GlbsToolStoreContext(GlbContextHandle, user);
         if (ShmId > 0) {
            int Act = gInclusiveToken ? 1 : 2;
            SPrintf(line, kMAXPATHLEN, "3 %d %d %d %d %s %s %d %s",
                    Act, gRSAKey, gParentId, gRemPid, gOpenHost.c_str(),
                    user, ShmId, GlbClientName);
            OffSet = RpdUpdateAuthTab(1, line, &token);
         } else if (gDebug > 0)
            ErrorInfo
                ("RpdGlobusAuth: unable to export context to shm for later use");
      }
      // Comunicate login user name to client (and token)
      SPrintf(line, kMAXPATHLEN, "%s %d", gUser, OffSet);
      NetSend(strlen(line), kROOTD_GLOBUS);   // Send message length first
      NetSend(line, kMESS_STRING);

      if (gReUseRequired) {
         // Send Token
         if (RpdSecureSend(token) == -1) {
            ErrorInfo("RpdGlobusAuth: problems secure-sending token"
                      " - may result in corrupted token");
         }
         if (token) delete[] token;

         // Save RSA public key into file for later use by other rootd/proofd
         RpdSavePubKey(gPubKey, OffSet, gUser);
      }
      gOffSet = OffSet;
   } else {
      // Comunicate login user name to client (and token)
      SPrintf(line, kMAXPATHLEN, "%s %d", gUser, OffSet);
      NetSend(strlen(line), kROOTD_GLOBUS);   // Send message length first
      NetSend(line, kMESS_STRING);
   }

   // and free allocated memory
   free(user);
   free(GlbClientName);

   if (gDebug > 0)
      ErrorInfo("RpdGlobusAuth: logging as %s ", gUser);

#endif
}

//______________________________________________________________________________
void RpdRfioAuth(const char *sstr)
{
   // Check if user and group id specified in the request exist in the
   // passwd file. If they do then grant access. Very insecure: to be used
   // with care.

   gAuth = 0;

   if (gDebug > 2)
      ErrorInfo("RpdRfioAuth: analyzing ... %s", sstr);

   if (!*sstr) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo("RpdRfioAuth: subject string is empty");
      return;
   }
   // Decode subject string
   unsigned int uid, gid;
   sscanf(sstr, "%u %u", &uid, &gid);

   // Now inquire passwd ...
   struct passwd *pw;
   if ((pw = getpwuid((uid_t) uid)) == 0) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo("RpdRfioAuth: uid %u not found", uid);
      return;
   }
   // Check if authorized
   char cuid[20];
   SPrintf(cuid, 20, "%u", uid);
   if (gUserIgnLen[5] > 0 && strstr(gUserIgnore[5], cuid) != 0) {
      NetSend(kErrNotAllowed, kROOTD_ERR);
      ErrorInfo
          ("RpdRfioAuth: user (%u,%s) not authorized to use (uid:gid) method",
           uid, pw->pw_name);
      return;
   }
   if (gUserAlwLen[5] > 0 && strstr(gUserAllow[5], cuid) == 0) {
      NetSend(kErrNotAllowed, kROOTD_ERR);
      ErrorInfo
          ("RpdRfioAuth: user (%u,%s) not authorized to use (uid:gid) method",
           uid, pw->pw_name);
      return;
   }

   // Now check group id ...
   if (gid != (unsigned int) pw->pw_gid) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo
          ("RpdRfioAuth: group id does not match (remote:%u,local:%u)",
           gid, (unsigned int) pw->pw_gid);
      return;
   }
   // Set username ....
   strcpy(gUser, pw->pw_name);


   // Notify, if required ...
   if (gDebug > 0)
      ErrorInfo("RpdRfioAuth: user %s authenticated (uid:%u, gid:%u)",
                gUser, uid, gid);

   // Set Auth flag
   gAuth = 1;
   gSec = 5;
}

//______________________________________________________________________________
void RpdAuthCleanup(const char *sstr, int opt)
{
   // Terminate correctly by cleaning up the auth table (and shared
   // memories in case of Globus) and closing the file.
   // Called upon receipt of a kROOTD_CLEANUP, kROOTD_CLOSE and on SIGPIPE.

   int rpid = 0, sec = -1, offs = -1, nw = 0;
   char usr[64] = {0};
   if (sstr)
      nw = sscanf(sstr, "%d %d %d %s", &rpid, &sec, &offs, usr);

   // Turn back to superuser for cleaning, if the case
   if (getuid() == 0) {
     if (setresgid(0, 0, 0) == -1)
        if (gDebug > 0)
           ErrorInfo("RpdAuthCleanup: can't setgid to superuser");
     if (setresuid(0, 0, 0) == -1)
        if (gDebug > 0)
           ErrorInfo("RpdAuthCleanup: can't setuid to superuser");
   }
   if (opt == 0) {
      RpdCleanupAuthTab("all", 0, -1);            // Cleanup everything (SIGPIPE)
      ErrorInfo("RpdAuthCleanup: cleanup ('all',0) done");
   } else if (opt == 1) {
      if (nw == 1) {
         // host specific cleanup (kROOTD_CLOSE)
         RpdCleanupAuthTab(gOpenHost.c_str(), rpid, -1);
         ErrorInfo("RpdAuthCleanup: cleanup ('%s',%d) done",
                   gOpenHost.c_str(), rpid);
      } else if (nw == 4) {
         // (host,usr,method) specific cleanup (kROOTD_CLOSE)
         if (RpdCheckOffSet(sec,usr,gOpenHost.c_str(),rpid,&offs,0,0,0)) {
            RpdCleanupAuthTab(gOpenHost.c_str(), rpid, offs);
            ErrorInfo("RpdAuthCleanup: cleanup (%s,%d,%d,%d,%s) done",
                      gOpenHost.c_str(), rpid, sec, offs, usr);
         } else {
            ErrorInfo("RpdAuthCleanup: cleanup not done: %s",
                      "wrong offset or already cleaned up");
         }
      }
   }
}

//______________________________________________________________________________
void RpdInitAuth()
{

   // Check auth tab file size
   struct stat st;
   if (stat(gRpdAuthTab.c_str(), &st) == 0) {

      // Cleanup auth tab file if too big
      if (st.st_size > kMAXTABSIZE)
         RpdUpdateAuthTab(0, (const char *)0, 0);

      // New file if still too big
      if (stat(gRpdAuthTab.c_str(), &st) == 0) {
         if (st.st_size > kMAXTABSIZE)
            RpdUpdateAuthTab(-1,(const char *)0, 0);
      }
   }

   // Reset
   int i;
   gNumAllow = gNumLeft = 0;
   for (i = 0; i < kMAXSEC; i++) {
      gAllowMeth[i] = -1;
      gHaveMeth[i] = 1;
   }

   // List of default authentication methods
   RpdDefaultAuthAllow();
}

//______________________________________________________________________________
void RpdDefaultAuthAllow()
{
   // Check configuration options and running daemons to build a default list
   // of secure methods.

   if (gDebug > 2)
      ErrorInfo("RpdDefaultAuthAllow: Enter");

   // UsrPwdClear
   gAllowMeth[gNumAllow] = 0;
   gNumAllow++;
   gNumLeft++;

   // SSH
   gAllowMeth[gNumAllow] = 4;
   gNumAllow++;
   gNumLeft++;

   // SRP
#ifdef R__SRP
   gAllowMeth[gNumAllow] = 1;
   gNumAllow++;
   gNumLeft++;
#else
   // Don't have this method
   gHaveMeth[1] = 0;
#endif

   // Kerberos
#ifdef R__KRB5
   gAllowMeth[gNumAllow] = 2;
   gNumAllow++;
   gNumLeft++;
#else
   // Don't have this method
   gHaveMeth[2] = 0;
#endif

   // Globus
#ifdef R__GLBS
   gAllowMeth[gNumAllow] = 3;
   gNumAllow++;
   gNumLeft++;
#else
   // Don't have this method
   gHaveMeth[3] = 0;
#endif

   if (gDebug > 2) {
      int i;
      char temp[200];
      temp[0] = '\0';
      if (gNumAllow == 0)
         strcpy(temp, "none");
      for (i = 0; i < gNumAllow; i++) {
         SPrintf(temp, 200, "%s %d", temp, gAllowMeth[i]);
      }
      ErrorInfo
          ("RpdDefaultAuthAllow: default list of secure methods available: %s",
           temp);
   }
}

//______________________________________________________________________________
int RpdCheckDaemon(const char *daemon)
{
   // Check the running of process 'daemon'.
   // Info got from 'ps ax'.

   char cmd[kMAXPATHLEN] = { 0 };
   int ch, i = 0, cnt = 0;

   if (gDebug > 2)
      ErrorInfo("RpdCheckDaemon: Enter ... %s", daemon);

   // Return if empty
   if (daemon == 0 || strlen(daemon) == 0)
      return cnt;

   // Build command
   SPrintf(cmd, kMAXPATHLEN, "ps ax | grep %s 2>/dev/null", daemon);

   // Run it ...
   FILE *fp = popen(cmd, "r");
   if (fp != 0) {
      for (ch = fgetc(fp); ch != EOF; ch = fgetc(fp)) {
         if (ch != 10) {
            cmd[i++] = ch;
         } else {
            cmd[i] = '\0';
            if (strstr(cmd, "grep") == 0 && strstr(cmd, "rootd") == 0
                && strstr(cmd, "proofd") == 0) {
               cnt++;
            }
            i = 0;
         }
      }
      if (i > 0) {
         cmd[i] = '\0';
         cnt++;
      }
      pclose(fp);
      if (gDebug > 2)
         ErrorInfo("RpdCheckDaemon: found %d instances of daemon %s",
                   cnt, daemon);

   } else {
      ErrorInfo("RpdCheckDaemon: problems executing cmd ...");
   }
   return cnt;
}

//______________________________________________________________________________
int RpdCheckSshd(int opt)
{
   // Tries to connect to sshd daemon on its standard port (22)
   // Used if RpdCheckDaemon returns a negative result

   if (gDebug > 2)
      ErrorInfo("RpdCheckSshd: Enter ... ");

   int rc = 0;
   if (opt == 0) {

      //
      // Check netstat output ...
      //

      // build check string
      char cs[20];
      SPrintf(cs, 20, ":%d",gSshdPort);

      // Run 'netstat' to check listening processes
      char cmd[kMAXPATHLEN] = { 0 };
      SPrintf(cmd, kMAXPATHLEN,
           "netstat -apn 2>/dev/null | grep LISTEN | grep -v LISTENING");
      FILE *fp= popen(cmd,"r");
      if (fp != 0) {
         while (fgets(cmd, sizeof(cmd), fp) != 0) {
            if (gDebug > 3)
               ErrorInfo("RpdCheckSshd: read: %s",cmd);
            if (strstr(cmd,cs)) {
               rc = 1;
               break;
            }
         }
         pclose(fp);
      } else {
         ErrorInfo("RpdCheckSshd: Problems executing 'netstat' ...");
      }

      if (gDebug > 2 && rc) {
         ErrorInfo("RpdCheckSshd: %s: %s %d", "diagnostics report",
                   "something is listening on port", gSshdPort);
      }

      if (!rc) {
         ErrorInfo("RpdCheckSshd: nothing seem to listening on port %d",
                   gSshdPort);
      }

   } else if (opt == 1) {

      //
      // Open a socket on port gSshdPort ...
      //

      // First get local host address
      struct hostent *h = gethostbyname("localhost");
      if (h == 0) {
         // Make further attempt with HOSTNAME
         if (getenv("HOSTNAME") == 0) {
            ErrorInfo("RpdCheckSshd: unable to resolve local host name");
            return 0;
         } else {
            h = gethostbyname(getenv("HOSTNAME"));
            if (h == 0) {
               ErrorInfo
                   ("RpdCheckSshd: local host name is unknown to gethostbyname: '%s'",
                    getenv("HOSTNAME"));
               return 0;
            }
         }
      }

      // Fill relevant sockaddr_in structure
      struct sockaddr_in servAddr;
      servAddr.sin_family = h->h_addrtype;
      memcpy((char *) &servAddr.sin_addr.s_addr, h->h_addr_list[0],
             h->h_length);
      servAddr.sin_port = htons(gSshdPort);

      // create AF_INET socket
      int sd = socket(AF_INET, SOCK_STREAM, 0);
      if (sd < 0) {
         ErrorInfo("RpdCheckSshd: cannot open new AF_INET socket (errno:%d) ",
                   errno);
         return 0;
      }

      /* bind any port number */
      struct sockaddr_in localAddr;
      localAddr.sin_family = AF_INET;
      localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
      localAddr.sin_port = htons(0);
      int rc = bind(sd, (struct sockaddr *) &localAddr, sizeof(localAddr));
      if (rc < 0) {
         ErrorInfo("RpdCheckSshd: cannot bind to local port %u", gSshdPort);
         return 0;
      }
      // connect to server
      rc = connect(sd, (struct sockaddr *) &servAddr, sizeof(servAddr));
      if (rc < 0) {
         ErrorInfo("RpdCheckSshd: cannot connect to local port %u",
                   gSshdPort);
         return 0;
      }
      // Sshd successfully contacted
      if (gDebug > 2)
         ErrorInfo("RpdCheckSshd: success!");
      rc = 1;
   }

   return rc;
}

//______________________________________________________________________________
void RpdUser(const char *sstr)
{
   // Check user id. If user id is not equal to rootd's effective uid, user
   // will not be allowed access, unless effective uid = 0 (i.e. root).
   const int kMaxBuf = 256;
   char recvbuf[kMaxBuf];
   char rootdpass[kMAXPATHLEN];
   char specpass[64] = {0};
   EMessageTypes kind;
   struct passwd *pw;
   if (gDebug > 2)
      ErrorInfo("RpdUser: Enter ... %s", sstr);

   gAuth = 0;

   // Nothing can be done if empty message
   if (!*sstr) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo("RpdUser: received empty string");
      return;
   }
   // Parse input message
   char *user = new char[strlen(sstr) + 1];
   if (gClientProtocol > 8) {
      int ulen, ofs, opt;
      // Decode subject string
      sscanf(sstr, "%d %d %d %d %s", &gRemPid, &ofs, &opt, &ulen, user);
      user[ulen] = '\0';
      gReUseRequired = (opt & kAUTH_REUSE_MSK);
      gCryptRequired = (opt & kAUTH_CRYPT_MSK);
      gSaltRequired  = (opt & kAUTH_SSALT_MSK);
      gOffSet = ofs;
   } else {
      strcpy(user, sstr);
   }
   if (gDebug > 2)
      ErrorInfo("RpdUser: gReUseRequired: %d gCryptRequired: %d",
                gReUseRequired, gCryptRequired);

   ERootdErrors err = kErrNoUser;
   if (gService == kROOTD) {
      // Default anonymous account ...
      if (!strcmp(user, "anonymous")) {
         user[0] = '\0';
         strcpy(user, "rootd");
      }
   }

   if ((pw = getpwnam(user)) == 0) {
      NetSend(err, kROOTD_ERR);
      ErrorInfo("RpdUser: user %s unknown", user);
      return;
   }
   // Check if of type anonymous ...
   if (!strcmp(pw->pw_shell, "/bin/false")) {
      err = kErrNoAnon;
      gAnon = 1;
      gReUseRequired = 0;
   }
   // If server is not started as root and user is not same as the
   // one who started rootd then authetication is not ok.
   uid_t uid = getuid();
   if (uid && uid != pw->pw_uid) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo("RpdUser: user not same as effective user of rootd");
      return;
   }
   // Check if authorized
   // If not anonymous, try to get passwd
   // (if our system uses shadow passwds and we are not superuser
   // we cannot authenticate users ...)
   //   char *passw = 0;
   char *passw = specpass;
   if (gAnon == 0) {

      // Try if special password is given via .rootdpass
      SPrintf(rootdpass, kMAXPATHLEN, "%s/%s", pw->pw_dir, kRootdPass.c_str());

      int fid = open(rootdpass, O_RDONLY);
      if (fid != -1) {
         if (read(fid, specpass, sizeof(specpass) - 1) > 0) {
            passw = specpass;
         }
         close(fid);
      }

      if (strlen(passw) == 0 || !strcmp(passw, "x")) {
#ifdef R__AFS
         gSaltRequired = 0;
#else

#ifdef R__SHADOWPW
         struct spwd *spw = 0;
         // System V Rel 4 style shadow passwords
         if ((spw = getspnam(user)) == 0) {
            if (gDebug > 0) {
               ErrorInfo("RpdUser: Shadow passwd not accessible for user %s",user);
               ErrorInfo("RpdUser: trying normal or special root passwd");
            }
            passw = pw->pw_passwd;
         } else
            passw = spw->sp_pwdp;
#else
         passw = pw->pw_passwd;
#endif
         // Check if successful
         if (strlen(passw) == 0 || !strcmp(passw, "x")) {
            NetSend(kErrNotAllowed, kROOTD_ERR);
            ErrorInfo("RpdUser: passwd hash not available for user %s", user);
            ErrorInfo
                ("RpdUser: user %s cannot be authenticated with this method",
                 user);
            return;
         }
#endif
      }
   }
   // Check if the administrator allows authentication
   char cuid[20];
   SPrintf(cuid, 20, "%d", (int)pw->pw_uid);
   if (gUserIgnLen[0] > 0 && strstr(gUserIgnore[0], cuid) != 0) {
      NetSend(kErrNotAllowed, kROOTD_ERR);
      ErrorInfo
          ("RpdUser: user (%d,%s) not authorized to use UsrPwd method",
           uid, pw->pw_name);
      return;
   }
   if (gUserAlwLen[0] > 0 && strstr(gUserAllow[0], cuid) == 0) {
      NetSend(kErrNotAllowed, kROOTD_ERR);
      ErrorInfo
          ("RpdUser: user (%d,%s) not authorized to use UsrPwd method",
           uid, pw->pw_name);
      return;
   }
   // Ok: Save username and go to next steps
   strcpy(gUser, user);
   if (user) delete[] user;

   // Salt vars
   char Salt[20] = { 0 };
   int Slen = 0;

   if (gClientProtocol > 8) {

      // Prepare status flag to send back
      if (gAnon == 1) {
         // Anonymous user: we will receive a text pass in the form
         // user@remote.host.dom
         NetSend(-1, kROOTD_AUTH);

      } else {

         if (gCryptRequired) {
            // Named user: first we receive a session public key
            // Ask for the RSA key
            NetSend(1, kROOTD_RSAKEY);

            // Receive the key securely
            if (RpdRecvClientRSAKey()) {
               ErrorInfo("RpdUser: could not import a valid key -"
                         " switch off reuse for this session");
               gReUseRequired = 0;
            }

            if (gSaltRequired) {
               if (!strncmp(passw, "$1$", 3)) {
                  // Shadow passwd
                  char *pd = strstr(passw + 4, "$");
                  Slen = (int) (pd - passw);
                  strncpy(Salt, passw, Slen);
                  Salt[Slen] = 0;
               } else {
                  Slen = 2;
                  strncpy(Salt, passw, Slen);
                  Salt[Slen] = 0;
               }
               if (gDebug > 2)
                  ErrorInfo("RpdUser: salt: '%s' ",Salt);

               // Send it over encrypted
               if (RpdSecureSend(Salt) == -1) {
                  ErrorInfo("RpdUser: problems secure-sending salt -"
                            " may result in corrupted salt");
               }
            } else {
               NetSend(0, kMESS_ANY);
            }
         } else {
            // We continue the aythentication process in clear
            NetSend(0, kROOTD_AUTH);
         }
      }

   } else {
      // If we are talking to a old client protocol
      NetSend(0, kROOTD_AUTH);
   }

   // Get the password hash or anonymous string
   if (NetRecv(recvbuf, kMaxBuf, kind) < 0) {
      NetSend(kErrFatal, kROOTD_ERR);
      ErrorInfo("RpdUser: error receiving message");
      return;
   }
   if (kind != kROOTD_PASS) {
      NetSend(kErrFatal, kROOTD_ERR);
      ErrorInfo("RpdUser: received wrong message type: %d (expecting: %d)",
                kind, (int) kROOTD_PASS);
      return;
   }
   if (!strncmp(recvbuf,"-1",2)) {
      if (gDebug > 0)
         ErrorInfo("RpdUser: client did not send a password - return");
      return;
   }
   // Get passwd
   char *passwd = 0;
   if (gAnon == 0 && gClientProtocol > 8 && gCryptRequired) {

      // Receive encrypted pass or its hash
      if (RpdSecureRecv(&passwd) == -1) {
         ErrorInfo
             ("RpdUser: problems secure-receiving pass hash - %s",
              "may result in authentication failure");
      }
      // If we required an hash check that we got it
      // (the client sends the passwd if the crypt version is different)
      if (gSaltRequired) {
         if (strncmp(passwd,Salt,strlen(Salt)))
            gSaltRequired = 0;
      }

   } else {

      // Receive clear or anonymous pass
      passwd = new char[strlen(recvbuf) + 1];

      // Re-invert pass
      int i, n = strlen(recvbuf);
      for (i = 0; i < n; i++)
         passwd[i] = ~recvbuf[i];
      passwd[i] = '\0';

      if (gDebug > 2 && gAnon)
         ErrorInfo("RpdUser: received anonymous pass: '%s'", passwd);
   }

   // Check the passwd and login if ok ...
   RpdPass(passwd);

   if (passwd) delete[] passwd;

}

//______________________________________________________________________________
int RpdGuessClientProt(const char *buf, EMessageTypes kind)
{
   // Try a guess of the client protocol from what she/he sent over
   // the net ...

   if (gDebug > 2)
      ErrorInfo("RpdGuessClientProt: Enter: buf: '%s', kind: %d", buf,
                (int) kind);

   // Assume same version as us.
   int proto = 9;

   // Clear authentication
   if (kind == kROOTD_USER) {
      char usr[64], rest[256];
      int ns = sscanf(buf, "%s %s", usr, rest);
      if (ns == 1)
         proto = 8;
   }
   // SRP authentication
   if (kind == kROOTD_SRPUSER) {
      char usr[64], rest[256];
      int ns = sscanf(buf, "%s %s", usr, rest);
      if (ns == 1)
         proto = 8;
   }
   // Kerberos authentication
   if (kind == kROOTD_KRB5) {
      if (strlen(buf) == 0)
         proto = 8;
   }

   if (gDebug > 2)
      ErrorInfo("RpdGuessClientProt: guess for gClientProtocol is %d",
                proto);

   // Return the guess
   return proto;
}

//______________________________________________________________________________
char *RpdGetRandString(int Opt, int Len)
{
   // Allocates and Fills a NULL terminated buffer of length Len+1 with
   // Len random characters.
   // Return pointer to the buffer (to be deleted by the caller)
   // Opt = 0      any non dangerous char
   //       1      letters and numbers  (upper and lower case)
   //       2      hex characters       (upper and lower case)
   //       3      crypt like           [a-zA-Z0-9./]

   int iimx[4][4] = { { 0x0, 0xffffff08, 0xafffffff, 0x2ffffffe }, // Opt = 0
                      { 0x0, 0x3ff0000,  0x7fffffe,  0x7fffffe },  // Opt = 1
                      { 0x0, 0x3ff0000,  0x7e,       0x7e },       // Opt = 2
                      { 0x0, 0x3ffc000,  0x7fffffe,  0x7fffffe }   // Opt = 3
   };

   const char *cOpt[4] = { "Any", "LetNum", "Hex", "Crypt" };

   //  Default option 0
   if (Opt < 0 || Opt > 3) {
      Opt = 0;
      if (gDebug > 2)
         ErrorInfo("RpdGetRandString: Unknown option: %d : assume 0", Opt);
   }
   if (gDebug > 2)
      ErrorInfo("RpdGetRandString: Enter ... Len: %d %s", Len, cOpt[Opt]);

   // Allocate buffer
   char *Buf = new char[Len + 1];

   // Init Random machinery ...
   if (!gRandInit)
      RpdInitRand();

   // randomize
   int k = 0;
   int i, j, l, m, frnd;
   while (k < Len) {
      frnd = rand();
      for (m = 7; m < 32; m += 7) {
         i = 0x7F & (frnd >> m);
         j = i / 32;
         l = i - j * 32;
         if ((iimx[Opt][j] & (1 << l))) {
            Buf[k] = i;
            k++;
         }
         if (k == Len)
            break;
      }
   }

   // NULL terminated
   Buf[Len] = 0;
   if (gDebug > 2)
      ErrorInfo("RpdGetRandString: got '%s' ", Buf);

   return Buf;
}

//______________________________________________________________________________
int RpdGetRSAKeys(const char *PubKey, int Opt)
{
   // Get public key from file PubKey (Opt == 1) or string PubKey (Opt == 0).

   char Str[kMAXPATHLEN] = { 0 };
   int KeyType = 0;

   if (gDebug > 2)
      ErrorInfo("RpdGetRSAKeys: enter: string len: %d, opt %d ", strlen(PubKey), Opt);

   if (!PubKey)
      return KeyType;

   FILE *fKey = 0;
   // Parse input type
   KeyType = 1;
   if (Opt == 1) {
      // Input is a File name: should get the string first
      if (access(PubKey, R_OK)) {
         ErrorInfo("RpdGetRSAKeys: Key File cannot be read - return ");
         return 0;
      }
      fKey = fopen(PubKey, "r");
      if (!fKey) {
         ErrorInfo("RpdGetRSAKeys: cannot open key file %s ", PubKey);
         return 0;
      }
      fgets(Str, sizeof(Str), fKey);
   }

   if (Opt == 0) {
      strcpy(Str, PubKey);
   }
   if (strlen(Str) > 0) {
      // The format is #<hex_n>#<hex_d>#
      char *pd1 = strstr(Str, "#");
      char *pd2 = strstr(pd1 + 1, "#");
      char *pd3 = strstr(pd2 + 1, "#");
      if (pd1 && pd2 && pd3) {
         // Get <hex_n> ...
         int l1 = (int) (pd2 - pd1 - 1);
         char *RSA_n_exp = new char[l1 + 1];
         strncpy(RSA_n_exp, pd1 + 1, l1);
         RSA_n_exp[l1] = 0;
         if (gDebug > 2)
            ErrorInfo("RpdGetRSAKeys: got %d bytes for RSA_n_exp", strlen(RSA_n_exp));
         // Now <hex_d>
         int l2 = (int) (pd3 - pd2 - 1);
         char *RSA_d_exp = new char[l2 + 1];
         strncpy(RSA_d_exp, pd2 + 1, l2);
         RSA_d_exp[l2] = 0;
         if (gDebug > 2)
            ErrorInfo("RpdGetRSAKeys: got %d bytes for RSA_d_exp", strlen(RSA_d_exp));

         rsa_num_sget(&gRSA_n, RSA_n_exp);
         rsa_num_sget(&gRSA_d, RSA_d_exp);

         if (RSA_n_exp) delete[] RSA_n_exp;
         if (RSA_d_exp) delete[] RSA_d_exp;

      } else
         return 0;
   }

   if (fKey)
      fclose(fKey);

   return KeyType;

}

//______________________________________________________________________________
void RpdSavePubKey(const char *PubKey, int OffSet, char *user)
{
   // Save RSA public key into file for later use by other rootd/proofd.

   if (gRSAKey == 0 || OffSet < 0)
      return;

   char strofs[20];
   snprintf(strofs,20,"%d",OffSet);
   std::string PubKeyFile = gTmpDir + "/rpk_" + std::string(strofs);
   FILE *fKey = fopen(PubKeyFile.c_str(), "w");
   if (fKey) {
      if (gRSAKey == 1) {
         fprintf(fKey, "%s", PubKey);
      }
   } else {
      ErrorInfo
          ("RpdSavePubKey: cannot save public key: set entry inactive");
      RpdCleanupAuthTab(gOpenHost.c_str(), gRemPid, OffSet);
   }

   if (fKey) {
      fclose(fKey);
      //      chmod(PubKeyFile, 0666);
      chmod(PubKeyFile.c_str(), 0600);

      if (getuid() == 0) {
         // Set ownership of the pub key to the user

         struct passwd *pw = getpwnam(user);

         if (chown(PubKeyFile.c_str(),pw->pw_uid,pw->pw_gid) == -1) {
            ErrorInfo
                ("RpdSavePubKey: cannot change ownership of %s (errno: %d)",
                  PubKeyFile.c_str(),GetErrno());
         }
      }
   }
}

//______________________________________________________________________________
int RpdSecureSend(char *Str)
{
   // Encode null terminated Str using the session private key indcated by Key
   // and sends it over the network.
   // Returns number of bytes sent.or -1 in case of error.

   char BufTmp[kMAXSECBUF];
   char BufLen[20];

   int sLen = strlen(Str) + 1;

   int Ttmp = 0;
   int Nsen = -1;

   if (gRSAKey == 1) {
      strncpy(BufTmp, Str, sLen);
      BufTmp[sLen] = 0;
      Ttmp = rsa_encode(BufTmp, sLen, gRSA_n, gRSA_d);
      SPrintf(BufLen, 20, "%d", Ttmp);
      NetSend(BufLen, kROOTD_ENCRYPT);
      Nsen = NetSendRaw(BufTmp, Ttmp);
      if (gDebug > 4)
         ErrorInfo
             ("RpdSecureSend: Local: sent %d bytes (expected: %d)",
              Nsen, Ttmp);
   } else {
      ErrorInfo("RpdSecureSend: Unknown key option (%d) - return",
                gRSAKey);
   }

   return Nsen;

}

//______________________________________________________________________________
int RpdSecureRecv(char **Str)
{
   // Receive buffer and decode it in Str using key indicated by Key type.
   // Return number of received bytes or -1 in case of error.

   char BufTmp[kMAXSECBUF];
   char BufLen[20];

   int Nrec = -1;
   // We must get a pointer ...
   if (!Str)
      return Nrec;

   if (gDebug > 2)
      ErrorInfo("RpdSecureRecv: enter ... (key is %d)", gRSAKey);

   EMessageTypes kind;
   NetRecv(BufLen, 20, kind);
   int Len = atoi(BufLen);
   if (gDebug > 4)
      ErrorInfo("RpdSecureRecv: got len '%s' %d ", BufLen, Len);
   if (!strncmp(BufLen, "-1", 2))
      return Nrec;

   // Now proceed
   if (gRSAKey == 1) {
      Nrec = NetRecvRaw(BufTmp, Len);
      rsa_decode(BufTmp, Len, gRSA_n, gRSA_d);
      if (gDebug > 2)
         ErrorInfo("RpdSecureRecv: Local: decoded string is %d bytes long", strlen(BufTmp));
   } else {
      ErrorInfo("RpdSecureRecv: Unknown key option (%d) - return",
                gRSAKey);
   }

   *Str = new char[strlen(BufTmp) + 1];
   strcpy(*Str, BufTmp);

   return Nrec;

}

//______________________________________________________________________________
int RpdGenRSAKeys(int setrndinit)
{
   // Generate a valid pair of private/public RSA keys to protect for
   // authentication password and token exchange
   // Returns 1 if a good key pair is not found after kMAXRSATRIES attempts
   // Returns 0 if a good key pair is found
   // If setrndinit = 1, no futher init of the random engine

   if (gDebug > 2)
      ErrorInfo("RpdGenRSAKeys: enter");

   // Init Random machinery ...
   if (!gRandInit)
      RpdInitRand();
   gRandInit = setrndinit;

   // Sometimes some bunch is not decrypted correctly
   // That's why we make retries to make sure that encryption/decryption
   // works as expected
   bool NotOk = 1;
   rsa_NUMBER p1, p2, rsa_n, rsa_e, rsa_d;
   int l_n = 0, l_e = 0, l_d = 0;
#if R__RSADEB
   char buf[rsa_STRLEN];
#endif
   char buf_n[rsa_STRLEN], buf_e[rsa_STRLEN], buf_d[rsa_STRLEN];

   int NAttempts = 0;
   int thePrimeLen = 20;
   int thePrimeExp = 45;   // Prime probability = 1-0.5^thePrimeExp
   while (NotOk && NAttempts < kMAXRSATRIES) {

      NAttempts++;
      if (gDebug > 2 && NAttempts > 1) {
            ErrorInfo("RpdGenRSAKeys: retry no. %d",NAttempts);
         srand(rand());
      }

      // Valid pair of primes
      p1 = rsa_genprim(thePrimeLen, thePrimeExp);
      p2 = rsa_genprim(thePrimeLen+1, thePrimeExp);

      // Retry if equal
      int NPrimes = 0;
      while (rsa_cmp(&p1, &p2) == 0 && NPrimes < kMAXRSATRIES) {
         NPrimes++;
         if (gDebug > 2)
            ErrorInfo("RpdGenRSAKeys: equal primes: regenerate (%d times)",NPrimes);
         srand(rand());
         p1 = rsa_genprim(thePrimeLen, thePrimeExp);
         p2 = rsa_genprim(thePrimeLen+1, thePrimeExp);
      }

#if R__RSADEB
      if (gDebug > 2) {
         rsa_num_sput(&p1, buf, rsa_STRLEN);
         ErrorInfo("RpdGenRSAKeys: local: p1: '%s' ", buf);
         rsa_num_sput(&p2, buf, rsa_STRLEN);
         ErrorInfo("RpdGenRSAKeys: local: p2: '%s' ", buf);
      }
#endif

      // Generate keys
      if (rsa_genrsa(p1, p2, &rsa_n, &rsa_e, &rsa_d)) {
         ErrorInfo("RpdGenRSAKeys: genrsa: unable to generate keys (%d)",NAttempts);
         continue;
      }

      // Determine their lengths
      rsa_num_sput(&rsa_n, buf_n, rsa_STRLEN);
      l_n = strlen(buf_n);
      rsa_num_sput(&rsa_e, buf_e, rsa_STRLEN);
      l_e = strlen(buf_e);
      rsa_num_sput(&rsa_d, buf_d, rsa_STRLEN);
      l_d = strlen(buf_d);

#if R__RSADEB
      if (gDebug > 2) {
         ErrorInfo("RpdGenRSAKeys: local: n: '%s' length: %d", buf_n, l_n);
         ErrorInfo("RpdGenRSAKeys: local: e: '%s' length: %d", buf_e, l_e);
         ErrorInfo("RpdGenRSAKeys: local: d: '%s' length: %d", buf_d, l_d);
      }
#endif
      if (rsa_cmp(&rsa_n, &rsa_e) <= 0)
         continue;
      if (rsa_cmp(&rsa_n, &rsa_d) <= 0)
         continue;

      // Now we try the keys
      char Test[2 * rsa_STRLEN] = "ThisIsTheStringTest01203456-+/";
      Int_t lTes = 31;
      char *Tdum = RpdGetRandString(0, lTes - 1);
      strncpy(Test, Tdum, lTes);
      delete[]Tdum;
      char buf[2 * rsa_STRLEN];
      if (gDebug > 3)
         ErrorInfo("RpdGenRSAKeys: local: test string: '%s' ", Test);

      // Private/Public
      strncpy(buf, Test, lTes);
      buf[lTes] = 0;

      // Try encryption with private key
      int lout = rsa_encode(buf, lTes, rsa_n, rsa_e);
      if (gDebug > 3)
         ErrorInfo("GenRSAKeys: local: length of crypted string: %d bytes", lout);

      // Try decryption with public key
      rsa_decode(buf, lout, rsa_n, rsa_d);
      buf[lTes] = 0;
      if (gDebug > 3)
         ErrorInfo("RpdGenRSAKeys: local: after private/public : '%s' ", buf);

      if (strncmp(Test, buf, lTes))
         continue;

      // Public/Private
      strncpy(buf, Test, lTes);
      buf[lTes] = 0;

      // Try encryption with public key
      lout = rsa_encode(buf, lTes, rsa_n, rsa_d);
      if (gDebug > 3)
         ErrorInfo("RpdGenRSAKeys: local: length of crypted string: %d bytes ",
              lout);

      // Try decryption with private key
      rsa_decode(buf, lout, rsa_n, rsa_e);
      buf[lTes] = 0;
      if (gDebug > 3)
         ErrorInfo("RpdGenRSAKeys: local: after public/private : '%s' ", buf);

      if (strncmp(Test, buf, lTes))
         continue;

      NotOk = 0;
   }

   if (NotOk) {
      ErrorInfo("RpdGenRSAKeys: unable to generate good RSA key pair - return");
      return 1;
   }

   // Save Private key
   rsa_assign(&gRSAPriKey.n, &rsa_n);
   rsa_assign(&gRSAPriKey.e, &rsa_e);

   // Save Public key
   rsa_assign(&gRSAPubKey.n, &rsa_n);
   rsa_assign(&gRSAPubKey.e, &rsa_d);

#if R__RSADEB
   if (gDebug > 2) {
      // Determine their lengths
      ErrorInfo("RpdGenRSAKeys: local: generated keys are:");
      ErrorInfo("RpdGenRSAKeys: local: n: '%s' length: %d", buf_n, l_n);
      ErrorInfo("RpdGenRSAKeys: local: e: '%s' length: %d", buf_e, l_e);
      ErrorInfo("RpdGenRSAKeys: local: d: '%s' length: %d", buf_d, l_d);
   }
#endif
   // Export form
   gRSAPubExport.len = l_n + l_d + 4;
   if (gRSAPubExport.keys)
      delete[] gRSAPubExport.keys;
   gRSAPubExport.keys = new char[gRSAPubExport.len];

   gRSAPubExport.keys[0] = '#';
   memcpy(gRSAPubExport.keys + 1, buf_n, l_n);
   gRSAPubExport.keys[l_n + 1] = '#';
   memcpy(gRSAPubExport.keys + l_n + 2, buf_d, l_d);
   gRSAPubExport.keys[l_n + l_d + 2] = '#';
   gRSAPubExport.keys[l_n + l_d + 3] = 0;
#if R__RSADEB
   if (gDebug > 2)
      ErrorInfo("RpdGenRSAKeys: local: export pub: '%s'", gRSAPubExport.keys);
#else
   if (gDebug > 2)
      ErrorInfo("RpdGenRSAKeys: local: export pub length: %d bytes", gRSAPubExport.len);
#endif

   gRSAInit = 1;
   return 0;
}

//______________________________________________________________________________
int RpdRecvClientRSAKey()
{
   // Generates local public/private RSA key pair
   // Send request for Client Public Key and Local public key
   // Receive encoded Client Key
   // Decode Client public key
   // NB: key is not saved to file here

   if (gRSAInit == 0) {
      // Generate Local RSA keys for the session
      if (RpdGenRSAKeys(1)) {
         ErrorInfo("RpdRecvClientRSAKey: unable to generate local keys");
         return 1;
      }
   }

   // Send server public key
   NetSend(gRSAPubExport.keys, gRSAPubExport.len, kROOTD_RSAKEY);

   // Receive length of message with encode client public key
   EMessageTypes kind;
   char BufLen[20];
   NetRecv(BufLen, 20, kind);
   int Len = atoi(BufLen);
   if (gDebug > 3)
      ErrorInfo("RpdRecvClientRSAKey: got len '%s' %d ", BufLen, Len);

   // Receive and decode encoded public key
   NetRecvRaw(gPubKey, Len);
   rsa_decode(gPubKey, Len, gRSAPriKey.n, gRSAPriKey.e);
   if (gDebug > 2)
      ErrorInfo("RpdRecvClientRSAKey: Local: decoded string is %d bytes long ",
         strlen(gPubKey));

   // Import Key and Determine key type
   gRSAKey = RpdGetRSAKeys(gPubKey, 0);
   if (gRSAKey == 0) {
      ErrorInfo("RpdRecvClientRSAKey: could not import a valid key");
      return 2;
   }

   return 0;
}

//______________________________________________________________________________
void RpdInitRand()
{
   // Init random machine.

   const char *randdev = "/dev/urandom";

   int fd;
   unsigned int seed;
   if ((fd = open(randdev, O_RDONLY)) != -1) {
      if (gDebug > 2)
         ErrorInfo("RpdInitRand: taking seed from %s", randdev);
      read(fd, &seed, sizeof(seed));
      close(fd);
   } else {
      if (gDebug > 2)
         ErrorInfo("RpdInitRand: %s not available: using time()", randdev);
      seed = time(0);   //better use times() + win32 equivalent
   }
   srand(seed);
}

//______________________________________________________________________________
void RpdAuthenticate()
{
   // Handle user authentication.
   char buf[kMAXRECVBUF];
   int Meth;
   EMessageTypes kind;

// #define R__DEBUG
#ifdef R__DEBUG
   int debug = 1;
   while (debug)
      ;
#endif

   // Reset gAuth (if we have been called this means that we need
   // to check at least that a valid authentication exists ...)
   gAuth = 0;

   while (!gAuth) {

      // Receive next
      if (!gClientOld) {
         if (NetRecv(buf, kMAXRECVBUF, kind) < 0)
            Error(gErrFatal, -1, "RpdAuthenticate: error receiving message");
      } else {
         strcpy(buf,gBufOld);
         kind = gKindOld;
         gBufOld[0] = '\0';
         gClientOld = 0;
      }

      // Decode the method ...
      Meth = RpdGetAuthMethod(kind);

      if (gDebug > 2) {
         if (kind != kROOTD_PASS) {
            ErrorInfo("RpdAuthenticate got: %d -- %s", kind, buf);
         } else {
            ErrorInfo("RpdAuthenticate got: %d ", kind);
         }
      }

      // Guess the client procotol if not received via Rootd/ProofdProtocol
      if (gClientProtocol == 0)
         gClientProtocol = RpdGuessClientProt(buf, kind);

      // If the client supports it check if we accept the method proposed;
      // if not send back the list of accepted methods, if any ...
      if (Meth != -1 && gClientProtocol > 8) {

         // Check if accepted ...
         if (RpdCheckAuthAllow(Meth, gOpenHost.c_str())) {
            if (gNumAllow>0) {
               if (gAuthListSent == 0) {
                  if (gDebug > 0)
                     ErrorInfo("Authenticate: %s method not"
                               " accepted from host: %s",
                                kAuthMeth[Meth].c_str(), gOpenHost.c_str());
                  NetSend(kErrNotAllowed, kROOTD_ERR);
                  RpdSendAuthList();
                  gAuthListSent = 1;
                  goto next;
               } else {
                  Error(gErrFatal,kErrNotAllowed,"Authenticate: method not"
                       " in the list sent to the client");
               }
            } else
               Error(gErrFatal,kErrConnectionRefused,"Authenticate:"
                       " connection refused from host %s", gOpenHost.c_str());
         }

         // Then check if a previous authentication exists and is valid
         // ReUse does not apply for RFIO
         if (kind != kROOTD_RFIO && RpdReUseAuth(buf, kind))
            goto next;
      }

      // Reset global variable
      gAuth = 0;

      switch (kind) {
         case kROOTD_USER:
            RpdUser(buf);
            break;
         case kROOTD_SRPUSER:
            RpdSRPUser(buf);
            break;
         case kROOTD_PASS:
            RpdPass(buf);
            break;
         case kROOTD_KRB5:
            RpdKrb5Auth(buf);
            break;
         case kROOTD_GLOBUS:
            RpdGlobusAuth(buf);
            break;
         case kROOTD_SSH:
            RpdSshAuth(buf);
            break;
         case kROOTD_RFIO:
            RpdRfioAuth(buf);
            break;
         case kROOTD_CLEANUP:
            RpdAuthCleanup(buf,1);
            ErrorInfo("RpdAuthenticate: authentication stuff cleaned - exit");
         case kROOTD_BYE:
            if (gRSAPubExport.keys) delete[] gRSAPubExport.keys;
            NetClose();
            exit(0);
            break;
         default:
            Error(gErrFatal,-1,"RpdAuthenticate: received bad opcode %d", kind);
      }

      if (gClientProtocol > 8) {

         // If failure prepare or continue negotiation
         // Don't do this if this was a SSH notification failure
         // because in such a case it was already done in the
         // appropriate daemon child
         int doneg = (Meth != -1 || kind == kROOTD_PASS) &&
                     (gRemPid > 0 || kind != kROOTD_SSH);
         if (gDebug > 2 && doneg)
            ErrorInfo("RpdAuthenticate: kind:%d Meth:%d gAuth:%d gNumLeft:%d",
                      kind, Meth, gAuth, gNumLeft);

         // If authentication failure, check if other methods could be tried ...
         if (gAuth == 0 && doneg) {
            if (gNumLeft > 0) {
               if (gAuthListSent == 0) {
                  RpdSendAuthList();
                  gAuthListSent = 1;
               } else
                  NetSend(-1, kROOTD_NEGOTIA);
            } else {
               NetSend(0, kROOTD_NEGOTIA);
               Error(gErrFatal, -1, "RpdAuthenticate: authentication failed");
            }
         }
      }
next:
      continue;
   }
}


//______________________________________________________________________________
void RpdProtocol(int ServType)
{
   // Receives client protocol and returns daemon protocol.

#ifdef R__DEBUG
   int debug = 1;
   while (debug)
      ;
#endif

   if (gDebug > 2)
      ErrorInfo("RpdProtocol: Enter: server type = %d", ServType);

   int readbuf = 1;
   EMessageTypes kind;
   char proto[kMAXRECVBUF];
   // For backward compatibility, for rootd we need to understand
   // whether we are talking to a OLD client: protocol information is
   // available only later on ...
   if (ServType == 1) {
      int lbuf[3];
      if (NetRecvRaw(lbuf, sizeof(lbuf)) < 0)
         Error(gErrFatal, kErrFatal, "RpdAuthenticate: error receiving message");

      // if kind is kROOTD_PROTOCOL then it is a recent one
      kind = (EMessageTypes) ntohl(lbuf[1]);
      if (kind == kROOTD_PROTOCOL || kind == kROOTD_CLEANUP ||
          kind == kROOTD_SSH) {
         // Decode the third int received
         memcpy(proto,((char *)lbuf)+8,4);
         // Receive the rest
         int len = ntohl(lbuf[0]) - 2*sizeof(int);
         if (len) {
            char *tmpbuf = new char[len];
            NetRecvRaw(tmpbuf, len);
            memcpy(proto+4,tmpbuf,len);
            delete[] tmpbuf;
         }
         proto[len+4] = '\0';
         readbuf = 0;
      } else {
         // Need to open parallel sockets first
         int size = ntohl(lbuf[1]);
         int port = ntohl(lbuf[2]);
         if (gDebug > 0)
            ErrorInfo("RpdProtocol: port = %d, size = %d", port, size);
         if (size > 1)
            NetParOpen(port, size);
      }
   }

   int Done = 0;
   gClientOld = 0;
   while (!Done) {

      // Receive next
      if (readbuf) {
         if (NetRecv(proto, kMAXRECVBUF, kind) < 0)
               Error(gErrFatal, -1, "RpdProtocol: error receiving message");
      }
      readbuf = 1;

      switch(kind) {

         case kROOTD_CLEANUP:
            RpdAuthCleanup(proto,1);
            ErrorInfo("RpdProtocol: authentication stuff cleaned - exit");
         case kROOTD_BYE:
            if (gRSAPubExport.keys) delete[] gRSAPubExport.keys;
            NetClose();
            exit(0);
            break;
         case kROOTD_PROTOCOL:

            if (strlen(proto) > 0) {
               sscanf(proto, "%d", &gClientProtocol);
            } else {
               if (ServType == kROOTD) {
                  // This is an old (TNetFile,TFTP) client:
                  // send our protocol first ...
                  if (NetSend(gServerProtocol, kROOTD_PROTOCOL) < 0)
                     Error(gErrFatal, -1,
                           "RpdProtocol: error sending kROOTD_PROTOCOL");
                  // ... and receive protocol via kROOTD_PROTOCOL2
                  if (NetRecv(proto, kMAXRECVBUF, kind) < 0)
                     Error(gErrFatal, -1,
                           "RpdProtocol: error receiving message");
                  if (kind != kROOTD_PROTOCOL2) {
                     strcpy(gBufOld, proto);
                     gKindOld = kind;
                     gClientOld = 1;
                     gClientProtocol = 0;
                  } else
                     sscanf(proto, "%d", &gClientProtocol);
               } else
                  gClientProtocol = 0;
            }
            if (!gClientOld) {
               // Notify
               if (gDebug > 0) {
                  ErrorInfo("RpdProtocol: gClientProtocol = %d",
                            gClientProtocol);
                  ErrorInfo("RpdProtocol: Sending gServerProtocol = %d",
                            gServerProtocol);
               }
               // send our protocol
               if (NetSend(gServerProtocol, kROOTD_PROTOCOL) < 0)
                  Error(gErrFatal, -1,
                           "RpdProtocol: error sending kROOTD_PROTOCOL");
            }
            Done = 1;
            break;
         case kROOTD_SSH:
            // Failure notification ...
            RpdSshAuth(proto);
            Error(gErrFatal,kErrAuthNotOK,"RpdProtocol: SSH failure notified");
            break;
         default:
            Error(gErrFatal,-1,"RpdProtocol: received bad option (%d)",kind);
      } // Switch

   } // Done
}

//______________________________________________________________________________
void RpdLogin(int ServType)
{
   // Authentication was successful, set user environment.

   if (gDebug > 2)
      ErrorInfo("RpdLogin: enter ... Server: %d ... gUser: %s",
                ServType, gUser);

   // Login only if in rootd/proofd environment
   if (ServType != kROOTD && ServType != kPROOFD)
      return;

   struct passwd *pw = getpwnam(gUser);

   if (!pw) {
      Error(gErrFatal, -1,
        "RpdLogin: user %s does not exist locally\n", gUser);
      return;
   } else if (chdir(pw->pw_dir) == -1) {
      Error(gErrFatal, -1,
        "RpdLogin: can't change directory to %s", pw->pw_dir);
      return;
   }

   if (getuid() == 0) {

#ifdef R__GLBS
      if (ServType == 2) {
         // We need to change the ownership of the shared memory segments used
         // for credential export to allow proofserv to destroy them
         struct shmid_ds shm_ds;
         if (gShmIdCred > 0) {
            if (shmctl(gShmIdCred, IPC_STAT, &shm_ds) == -1) {
               Error(gErrFatal, -1,
                     "RpdLogin: can't get info about shared memory segment %d",
                     gShmIdCred);
               return;
            }
            shm_ds.shm_perm.uid = pw->pw_uid;
            shm_ds.shm_perm.gid = pw->pw_gid;
            if (shmctl(gShmIdCred, IPC_SET, &shm_ds) == -1) {
                Error(gErrFatal, -1,
                    "RpdLogin: can't change ownership of shared memory segment %d",
                    gShmIdCred);
                return;
            }
         }
      }
#endif
      // set access control list from /etc/initgroup
      initgroups(gUser, pw->pw_gid);

      // set uid and gid
      if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1) {
         Error(gErrFatal, -1, "RpdLogin: can't setgid for user %s", gUser);
         return;
      }
      if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1) {
         Error(gErrFatal, -1, "RpdLogin: can't setuid for user %s", gUser);
         return;
      }
   }

   if (ServType == 2) {
      // set HOME env
      char *home = new char[8+strlen(pw->pw_dir)];
      sprintf(home, "HOME=%s", pw->pw_dir);
      putenv(home);
   }

   umask(022);

   // Notify authentication to client ...
   NetSend(gAuth, kROOTD_AUTH);
   // Send also new offset if it changed ...
   if (gAuth == 2) NetSend(gOffSet, kROOTD_AUTH);

   if (gDebug > 0)
      ErrorInfo("RpdLogin: user %s authenticated", gUser);
}

//______________________________________________________________________________
int RpdInitSession(int ServType, std::string &User,
                   int &Cproto, int &Anon, std::string &Passwd)
{
   // Perform the action needed to commence the new session:
   //   - set debug flag
   //   - check authentication table
   //   - Inquire protocol
   //   - authenticate the client
   //   - login the client
   // Returns 1 for a PROOF master server, 0 otherwise
   // Returns logged-in User, the remote client procotol Cproto, the
   // client kind of user Anon and, if anonymous user, the client Passwd.
   // Called just after opening the connection

   if (gDebug > 2)
      ErrorInfo("RpdInitSession: %s", gServName[ServType].c_str());

   int retval = 0;

   // CleanUp authentication table, if needed or required ...
   RpdInitAuth();

   // Get Host name
   NetGetRemoteHost(gOpenHost);

   if (ServType == kPROOFD) {

      // find out if we are supposed to be a master or a slave server
      char  msg[80];
      if (NetRecv(msg, sizeof(msg)) < 0)
         Error(gErrFatal,-1,
               "RpdInitSession: Cannot receive master/slave status");

      retval = !strcmp(msg, "master") ? 1 : 0;

      if (gDebug > 0)
         ErrorInfo("RpdInitSession: PROOF master/slave = %s", msg);
   }

   // Get protocol first (does not return in case of failure)
   RpdProtocol(ServType);

   // user authentication (does not return in case of failure)
   RpdAuthenticate();

   // Login the user (if in rootd/proofd environment)
   if (ServType == kROOTD || ServType == kPROOFD)
      RpdLogin(ServType);

   User = std::string(gUser);
   Cproto = gClientProtocol;
   Anon = gAnon;
   if (gAnon)
      Passwd = std::string(gPasswd);

   return retval;
}

//______________________________________________________________________________
int RpdInitSession(int ServType, std::string &User, int &Rid)
{
   // Perform the action needed to commence the new session:
   //   - set debug flag
   //   - check authentication table
   //   - Inquire protocol
   //   - authenticate the client
   //   - login the client
   // Returns 1 for a PROOF master server, 0 otherwise
   // Returns logged-in User and remote process id in Rid
   // Called just after opening the connection

   int Dum1 = 0, Dum2 = 0;
   std::string Dum3;
   Rid = gRemPid;
   return RpdInitSession(ServType,User,Dum1,Dum2,Dum3);

}

//______________________________________________________________________________
void RpdSetUid(int uid)
{
   // Change current user id to uid (and gid).

   if (gDebug > 2)
      ErrorInfo("RpdSetUid: enter ...uid: %d", uid);

   struct passwd *pw = getpwuid(uid);

   if (!pw) {
      Error(gErrFatal, -1,
        "RpdSetUid: uid %d does not exist locally", uid);
      return;
   } else if (chdir(pw->pw_dir) == -1) {
      Error(gErrFatal, -1,
        "RpdSetUid: can't change directory to %s", pw->pw_dir);
      return;
   }

   if (getuid() == 0) {

      // set access control list from /etc/initgroup
      initgroups(pw->pw_name, pw->pw_gid);

      // set uid and gid
      if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1) {
         Error(gErrFatal, -1, "RpdSetUid: can't setgid for uid %d", uid);
         return;
      }
      if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1) {
         Error(gErrFatal, -1, "RpdSetUid: can't setuid for uid %d", uid);
         return;
      }
   }

   if (gDebug > 0)
      ErrorInfo("RpdSetUid: uid set (%d,%s)", uid, pw->pw_name);
}

//_____________________________________________________________________________
void RpdInit(EService serv, int pid, int sproto, int inctok, int rumsk,
             int rlog, int sshp, const char *tmpd, const char *asrpp)
{
   // Change defaults job control options.

   gService        = serv;
   gParentId       = pid;
   gServerProtocol = sproto;
   gInclusiveToken = inctok;
   gReUseAllow     = rumsk;
   gRootLog        = rlog;
   gSshdPort       = sshp;
   if (tmpd && strlen(tmpd)) {
      gTmpDir      = tmpd;
      gRpdAuthTab  = gTmpDir + kAuthTab;
   }
   if (asrpp && strlen(asrpp))
      gAltSRPPass  = asrpp;

   if (gDebug > 0) {
      ErrorInfo("RpdSetOptions: gService= %s, gRootLog= %d, gSshdPort= %d",
                 gServName[gService].c_str(), gRootLog, gSshdPort);
      ErrorInfo("RpdSetOptions: gParentId= %d, gInclusiveToken= %d",
                 gParentId, gInclusiveToken);
      ErrorInfo("RpdSetOptions: gReUseAllow= 0x%x", gReUseAllow);
      ErrorInfo("RpdSetOptions: gServerProtocol= %d", gServerProtocol);
      if (tmpd)
         ErrorInfo("RpdSetOptions: gTmpDir= %s", gTmpDir.c_str());
      if (asrpp)
         ErrorInfo("RpdSetOptions: gAltSRPPass= %s", gAltSRPPass.c_str());
   }
}


//______________________________________________________________________________
int SPrintf(char *buf, size_t size, const char *va_(fmt), ...)
{
   // Acts like snprintf with some printou in case of error if required
   // Returns number of  characters printed (excluding the trailing `\0').
   // Returns 0 is buf or size are not defined or inconsistent.
   // Returns -1 if the buffer is truncated.

   // Check buf
   if (!buf) {
      if (gDebug > 0)
         ErrorInfo("SPrintf: buffer not allocated: do nothing");
      return 0;
   }

   // Check size
   if (size < 1) {
      if (gDebug > 0)
         ErrorInfo("SPrintf: cannot determine buffer size (%d): do nothing",size);
      return 0;
   }

   // Now fill buf
   va_list ap;
   va_start(ap,va_(fmt));
   int np = vsnprintf(buf, size, fmt, ap);
   va_end(ap);

   if (np == -1 && gDebug > 0)
      ErrorInfo("SPrintf: buffer truncated (%s)",buf);

   return np;
}


//______________________________________________________________________________
char *ItoA(int i)
{
   // Return pointer to a static string containing the string
   // version of integer 'i', up to a max of kMAXCHR (=30)
   // characters; returns "-1" if more chars are needed.
   const int kMAXCHR = 30;
   static char str[kMAXCHR];

   // This is the number of characters we need
   int nchr = (int)log10(double(i)) + 1;
   if (nchr > kMAXCHR)
      strcpy(str,"-1");
   else
      snprintf(str,30,"%d",i);

   return str;
}

//______________________________________________________________________________
void RpdSetErrorHandler(ErrorHandler_t err, ErrorHandler_t sys, ErrorHandler_t fatal)
{
   // Set global pointers to error handler functions

   gErr      = err;
   gErrSys   = sys;
   gErrFatal = fatal;
}

#ifdef R__GLBS
//______________________________________________________________________________
int RpdGetShmIdCred()
{
   // Returns shared memory id

   return gShmIdCred;
}
#endif

} // namespace ROOT

