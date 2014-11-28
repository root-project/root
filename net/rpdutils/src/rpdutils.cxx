// @(#)root/rpdutils:$Id$
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
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <netdb.h>
#include <math.h>
#include "snprintf.h"

#if defined(__CYGWIN__) && defined(__GNUC__)
#   define cygwingcc
#endif

#ifdef __linux__
#define linux
#endif

#if defined(linux) || defined(__sun) || defined(__sgi) || \
    defined(_AIX) || defined(__FreeBSD__) || defined(__OpenBSD__) || \
    defined(__APPLE__) || defined(__MACH__) || defined(cygwingcc)
#include <grp.h>
#include <sys/types.h>
#include <signal.h>
#endif

#ifdef _AIX
extern "C" int ruserok(char *, int, char *, char *);
#endif

#if defined(__APPLE__)
#include <sys/mount.h>
extern "C" int fstatfs(int file_descriptor, struct statfs *buffer);
#elif defined(linux) || defined(__hpux)
#include <sys/vfs.h>
#elif defined(__FreeBSD__) || defined(__OpenBSD__)
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

#if defined(cygwingcc) && !defined(F_LOCK) && !defined(F_ULOCK)
#define F_LOCK F_WRLCK
#define F_ULOCK F_UNLCK
int ruserok(const char *, int, const char *, const char *) {
   return 0;
}
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

#ifdef R__WIN32
#define R__NOCRYPT
#endif

#ifdef R__NOCRYPT
static std::string gRndmSalt = std::string("ABCDEFGH");
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
#include "AFSAuth.h"
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

#ifdef R__SSL
// SSL specific headers for RSA keys
#include <openssl/bio.h>
#include <openssl/blowfish.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/rsa.h>
#include <openssl/ssl.h>
#endif

#include "rpdp.h"
#include "rsadef.h"
#include "rsalib.h"
//
// To improve error logging for UsrPwd on the client side
static ERootdErrors gUsrPwdErr[4][4] = {
   {kErrNoPasswd, kErrNoPassHEquNoFiles, kErrNoPassHEquBadFiles, kErrNoPassHEquFailed},
   {kErrBadPasswd, kErrBadPassHEquNoFiles, kErrBadPassHEquBadFiles, kErrBadPassHEquFailed},
   {kErrBadRtag, kErrBadRtagHEquNoFiles, kErrBadRtagHEquBadFiles, kErrBadRtagHEquFailed},
   {kErrBadPwdFile, kErrBadPwdFileHEquNoFiles, kErrBadPwdFileHEquBadFiles,
    kErrBadPwdFileHEquFailed}};

//--- Machine specific routines ------------------------------------------------

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

#if !defined(__hpux) && !defined(linux) && !defined(__FreeBSD__) && \
    !defined(__OpenBSD__) || defined(cygwingcc)
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
#if defined(linux) && !defined(R__HAS_SETRESUID)
extern "C" {
   int setresgid(gid_t r, gid_t e, gid_t s);
   int setresuid(uid_t r, uid_t e, uid_t s);
}
#endif
#endif

#if defined(__sun)
#if defined(R__SUNGCC3)
extern "C" int gethostname(char *, unsigned int);
#endif
#endif

extern int gDebug;

namespace ROOT {

//
// rpdutils module globals
ErrorHandler_t gErrSys   = 0;
ErrorHandler_t gErrFatal = 0;
ErrorHandler_t gErr      = 0;
bool gSysLog = 0;
std::string gServName[3] = { "sockd", "rootd", "proofd" };

//
// Local global consts
static const int gAUTH_CLR_MSK = 0x1;     // Masks for authentication methods
static const int gAUTH_SRP_MSK = 0x2;
static const int gAUTH_KRB_MSK = 0x4;
static const int gAUTH_GLB_MSK = 0x8;
static const int gAUTH_SSH_MSK = 0x10;
static const int gMAXTABSIZE = 50000000;

static const std::string gAuthMeth[kMAXSEC] = { "UsrPwd", "SRP", "Krb5",
                                                "Globus", "SSH", "UidGid" };
static const std::string gAuthTab    = "/rpdauthtab";   // auth table
static const std::string gDaemonRc   = ".rootdaemonrc"; // daemon access rules
static const std::string gRootdPass  = ".rootdpass";    // special rootd passwd
static const std::string gSRootdPass = "/.srootdpass";  // SRP passwd
static const std::string gKeyRoot    = "/rpk.";         // Root for key files

//
// RW dir for temporary files (needed by gRpdAuthTab: do not move)
static std::string gTmpDir = "/tmp";

//
// Local global vars
static int gAuthProtocol = -1;  // Protocol used fro a successful authentication
static char gBufOld[kMAXRECVBUF] = {0}; // msg sync for old client (<=3.05/07)
static bool gCheckHostsEquiv = 1;
static int gClientOld = 0;              // msg sync for old client (<=3.05/07)
static int gClientProtocol = -1;
static int gCryptRequired = -1;
static std::string gCryptToken;
static int gAllowMeth[kMAXSEC];
static std::string gAltSRPPass;
static int gAnon = 0;
static int gExistingAuth = 0;
static int gAuthListSent = 0;
static int gHaveMeth[kMAXSEC];
static EMessageTypes gKindOld;          // msg sync for old client (<=3.05/07)
static int gMethInit = 0;
static int gNumAllow = -1;
static int gNumLeft = -1;
static int gOffSet = -1;
static std::string gOpenHost = "????";
static int gParentId = -1;
static char gPasswd[kMAXUSERLEN] = { 0 };
static char gPubKey[kMAXPATHLEN] = { 0 };
static int gPubKeyLen = 0;
static int gRandInit = 0;
static int gRemPid = -1;
static bool gRequireAuth = 1;
static int gReUseAllow = 0x1F;  // define methods for which tokens can be asked
static int gReUseRequired = -1;
static int gDoLogin = 0;  // perform login
static std::string gRpdAuthTab = std::string(gTmpDir).append(gAuthTab);
static std::string gRpdKeyRoot = std::string(gTmpDir).append(gKeyRoot);
static rsa_NUMBER gRSA_d;
static rsa_NUMBER gRSA_n;
static int gRSAInit = 0;
static int gRSAKey = 0;
static rsa_KEY gRSAPriKey;
static rsa_KEY_export gRSAPubExport[2] = {{0,0},{0,0}};
static rsa_KEY gRSAPubKey;
#ifdef R__SSL
static BF_KEY gBFKey;            // Session symmetric key
static RSA *gRSASSLKey = 0;      // Local RSA SSL key
#endif
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
static gss_cred_id_t gGlbCredHandle = GSS_C_NO_CREDENTIAL;
static bool gHaveGlobus = 1;
static std::string gGlobusSubjName;
#endif

//______________________________________________________________________________
static int rpd_rand()
{
   // rand() implementation using /udev/random or /dev/random, if available

#ifndef WIN32
   int frnd = open("/dev/urandom", O_RDONLY);
   if (frnd < 0) frnd = open("/dev/random", O_RDONLY);
   int r;
   if (frnd >= 0) {
      ssize_t rs = read(frnd, (void *) &r, sizeof(int));
      close(frnd);
      if (r < 0) r = -r;
      if (rs == sizeof(int)) return r;
   }
   ErrorInfo("+++ERROR+++ : rpd_rand: neither /dev/urandom nor /dev/random are available or readable!");
   struct timeval tv;
   if (gettimeofday(&tv,0) == 0) {
      int t1, t2;
      memcpy((void *)&t1, (void *)&tv.tv_sec, sizeof(int));
      memcpy((void *)&t2, (void *)&tv.tv_usec, sizeof(int));
      r = t1 + t2;
      if (r < 0) r = -r;
      return r;
   }
   return -1;
#else
   // No special random device available: use rand()
   return rand();
#endif
}

//______________________________________________________________________________
static int reads(int fd, char *buf, int len)
{
   //  reads in at most one less than len characters from open
   //  descriptor fd and stores them into the buffer pointed to by buf.
   //  Reading stops after an EOF or a newline. If a newline is
   //  read, it  is stored into the buffer.
   //  A '\0' is stored after the last character in the buffer.
   //  The number of characters read is returned (newline included).
   //  Returns < 0 in case of error.

   int k = 0;
   int nread = -1;
   int nr = read(fd,buf,1);
   while (nr > 0 && buf[k] != '\n' && k < (len-1)) {
      k++;
      nr = read(fd,buf+k,1);
   }
   if (k >= len-1) {
      buf[k] = 0;
      nread = len-1;
   } else if (buf[k] == '\n'){
      if (k <= len-2) {
         buf[k+1] = 0;
         nread = k+1;
      } else {
         buf[k] = 0;
         nread = k;
      }
   } else if (nr == 0) {
      if (k > 0) {
         buf[k-1] = 0;
         nread = k-1;
      } else {
         buf[0] = 0;
         nread = 0;
      }
   } else if (nr < 0) {
      if (k > 0) {
         buf[k] = 0;
         nread = -(k-1);
      } else {
         buf[0] = 0;
         nread = -1;
      }
   }
   // Fix the lengths
   if (nread >= 0) buf[nread] = 0;

   return nread;
}

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
static volatile void *rpdmemset(volatile void *dst, int c, int len)
{
   // To avoid problems due to compiler optmization
   // Taken from Viega&Messier, "Secure Programming Cookbook", O'Really, #13.2
   // (see discussion there)
   volatile char *buf;

   for (buf = (volatile char *)dst; len; (buf[--len] = c)) { }
   return dst;
}

#ifdef R__NOCRYPT
//______________________________________________________________________________
char *rpdcrypt(const char *pw, const char *sa)
{
   // This applies simple nor encryption with sa to the first 64 bytes
   // pw. Returns the hex of the result (max length 128).
   // This is foreseen for systms where crypt is not available
   // (on windows ...), to provide some protection of tokens.

   static char buf[129];
   char tbuf[64];
   int np = (strlen(pw) < 64) ? strlen(pw) : 64;
   int ns = strlen(sa);
   char c;

   int i = 0;
   for (i=0; i<np; i++) {
      // We use in turn all the salt bits; and re-use
      // if they are not enough
      int ks = i%ns;
      tbuf[i] = pw[i]^sa[ks];
      // Convert the result in two hexs: the first ...
      int j = 0xF & tbuf[i];
      if (j < 10)
         c = 48 + j;
      else
         c = 55 + j;
      int k = 2*i;
      buf[k] = c;
      // .. the second
      j = (0xF0 & tbuf[i]) >> 4;
      if (j < 10)
         c = 48 + j;
      else
         c = 55 + j;
      k = 2*i + 1;
      buf[k] = c;
   }
   // Null termination
   buf[np*2] = 0;

   return buf;
}
#endif

//______________________________________________________________________________
void RpdSetSysLogFlag(int syslog)
{
   // Change the value of the static gSysLog to syslog.
   // Recognized values:
   //                       0      log to syslog (for root started daemons)
   //                       1      log to stderr (for user started daemons)

   gSysLog = syslog;
   if (gDebug > 2)
      ErrorInfo("RpdSetSysLogFlag: gSysLog set to %d", gSysLog);
}

//______________________________________________________________________________
void RpdSetMethInitFlag(int methinit)
{
   // Change the value of the static gMethInit to methinit.
   // Recognized values:
   //                       0      reset
   //                       1      initialized already

   gMethInit = methinit;
   if (gDebug > 2)
      ErrorInfo("RpdSetMethInitFlag: gMethInit set to %d", gMethInit);
}
//______________________________________________________________________________
const char *RpdGetKeyRoot()
{
   // Return pointer to the root string for key files
   // Used by proofd.
   return (const char *)gRpdKeyRoot.c_str();
}

//______________________________________________________________________________
int RpdGetClientProtocol()
{
   // Return protocol version run by the client.
   // Used by proofd.
   return gClientProtocol;
}

//______________________________________________________________________________
int RpdGetAuthProtocol()
{
   // Return authentication protocol used for the handshake.
   // Used by proofd.
   return gAuthProtocol;
}

//______________________________________________________________________________
int RpdGetOffSet()
{
   // Return offset in the authtab file.
   // Used by proofd.
   return gOffSet;
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
   int method = -1;

   if (kind == kROOTD_USER)
      method = 0;
   if (kind == kROOTD_SRPUSER)
      method = 1;
   if (kind == kROOTD_KRB5)
      method = 2;
   if (kind == kROOTD_GLOBUS)
      method = 3;
   if (kind == kROOTD_SSH)
      method = 4;
   if (kind == kROOTD_RFIO)
      method = 5;

   return method;
}

//______________________________________________________________________________
int RpdDeleteKeyFile(int ofs)
{
   // Delete Public Key file
   // Returns: 0 if ok
   //          1 if error unlinking (check errno);
   int retval = 0;

   std::string pukfile = gRpdKeyRoot;
   pukfile.append(ItoA(ofs));

   // Some debug info
   if (gDebug > 2) {
      ErrorInfo("RpdDeleteKeyFile: proc uid:%d gid:%d",
      getuid(),getgid());
   }

   // Unlink
   if (unlink(pukfile.c_str()) == -1) {
      if (gDebug > 0 && GetErrno() != ENOENT) {
         ErrorInfo("RpdDeleteKeyFile: problems unlinking pub"
                   " key file '%s' (errno: %d)",
                   pukfile.c_str(),GetErrno());
      }
      retval = 1;
   }
   return retval;
}

//______________________________________________________________________________
int RpdUpdateAuthTab(int opt, const char *line, char **token, int ilck)
{
   // Update tab file.
   // If ilck <= 0 open and lock the file; if ilck > 0, use file
   // descriptor ilck, which should correspond to an open and locked file.
   // If opt = -1 : delete file (backup saved in <file>.bak);
   // If opt =  0 : eliminate all inactive entries
   //               (if line="size" act only if size > gMAXTABSIZE)
   // if opt =  1 : append 'line'.
   // Returns -1 in case of error.
   // Returns offset for 'line' and token for opt = 1.
   // Returns new file size for opt = 0.

   int retval = -1;
   int itab = 0;
   char fbuf[kMAXPATHLEN];

   if (gDebug > 2)
      ErrorInfo("RpdUpdateAuthTab: analyzing: opt: %d, line: %s, ilck: %d",
                opt, line, ilck);

   if (ilck <= 0) {

      // Open file for reading/ writing
      itab = open(gRpdAuthTab.c_str(), O_RDWR);
      if (itab == -1) {
         if (opt == 1 && GetErrno() == ENOENT) {
            // Try creating the file
            itab = open(gRpdAuthTab.c_str(), O_RDWR | O_CREAT, 0600);
            if (itab == -1) {
               ErrorInfo("RpdUpdateAuthTab: opt=%d: error opening %s"
                         "(errno: %d)",
                         opt, gRpdAuthTab.c_str(), GetErrno());
               return retval;
            }
         } else {
            ErrorInfo("RpdUpdateAuthTab: opt=%d: error opening %s"
                      " (errno: %d)",
                       opt, gRpdAuthTab.c_str(), GetErrno());
            return retval;
         }
      }

      // lock tab file
      if (lockf(itab, F_LOCK, (off_t) 1) == -1) {
         ErrorInfo("RpdUpdateAuthTab: opt=%d: error locking %s"
                   " (errno: %d)", opt, gRpdAuthTab.c_str(), GetErrno());
         close(itab);
         return retval;
      }
      if (gDebug > 0)
         ErrorInfo("RpdUpdateAuthTab: opt= %d - file LOCKED", opt);
   } else {
      itab = ilck;
   }

   // File size
   int fsize = 0;
   if ((fsize = lseek(itab, 0, SEEK_END)) == -1) {
      ErrorInfo("RpdUpdateAuthTab: opt=%d: lseek error (errno: %d)",
           opt, GetErrno());
      goto goingout;
   }

   // Set indicator to beginning
   if (lseek(itab, 0, SEEK_SET) == -1) {
      ErrorInfo("RpdUpdateAuthTab: opt=%d: lseek error (errno: %d)",
           opt, GetErrno());
      goto goingout;
   }

   if (opt == -1) {

      //
      // Save file in .bak and delete its content

      // Open backup file
      std::string bak = std::string(gRpdAuthTab).append(".bak");
      int ibak = open(bak.c_str(), O_RDWR | O_CREAT, 0600);
      if (ibak == -1) {
         ErrorInfo("RpdUpdateAuthTab: opt=%d: error opening/creating %s"
                   " (errno: %d)", opt, bak.c_str(), GetErrno());
         goto goingout;
      }

      // Truncate file to new length
      if (ftruncate(ibak, 0) == -1)
         ErrorInfo("RpdUpdateAuthTab: opt=%d: ftruncate error (%s)"
                   " (errno: %d)", opt, bak.c_str(), GetErrno());

      // Copy the content
      char buf[kMAXPATHLEN];
      int ofs = 0, nr = 0;
      while ((nr = reads(itab, buf, sizeof(buf)))) {
         int slen = strlen(buf);

         // Make sure there is a '\n' before writing
         if (buf[slen-1] != '\n') {
            if (slen >= kMAXPATHLEN -1)
               buf[slen-1] = '\n';
            else {
               buf[slen] = '\n';
               buf[slen+1] = '\0';
            }
         }
         if (slen) {
            while (write(ibak, buf, slen) < 0 && GetErrno() == EINTR)
               ResetErrno();
         }

         // Delete Public Key file
         RpdDeleteKeyFile(ofs);
         // Next OffSet
         ofs += slen;
      }
      close(ibak);

      // Truncate file to new length
      if (ftruncate(itab, 0) == -1)
         ErrorInfo("RpdUpdateAuthTab: opt=%d: ftruncate error (%s)"
                   " (errno: %d)", opt, gRpdAuthTab.c_str(), GetErrno());
      retval = 0;

   } else if (opt == 0) {

      //
      // Cleanup the file (remove inactive entries)

      // Now scan over entries
      int pr = 0, pw = 0;
      int lsec, act = 0, oldofs = 0, bytesread = 0;
      char ln[kMAXPATHLEN], dumm[kMAXPATHLEN];
      bool fwr = 0;

      while ((bytesread = reads(itab, ln, sizeof(ln)))) {

         bool ok = 1;
         // Current position
         if ((pr = lseek(itab,0,SEEK_CUR)) < 0) {
            ErrorInfo("RpdUpdateAuthTab: opt=%d: problems lseeking file %s"
                      " (errno: %d)", opt, gRpdAuthTab.c_str(), errno);
            fwr = 1;
            ok = 0;
         }

         // Check file corruption: length and number of items
         int slen = bytesread;
         if (ok && slen < 1) {
            ErrorInfo("RpdUpdateAuthTab: opt=%d: file %s seems corrupted"
                      " (slen: %d)", opt, gRpdAuthTab.c_str(), slen);
            fwr = 1;
            ok = 0;
         }
         if (ok) {
            // Check file corruption: number of items
            int ns = sscanf(ln, "%d %d %4095s", &lsec, &act, dumm);
            if (ns < 3 ) {
               ErrorInfo("RpdUpdateAuthTab: opt=%d: file %s seems corrupted"
                         " (ns: %d)", opt, gRpdAuthTab.c_str(), ns);
               fwr = 1;
               ok = 0;
            }
         }

         if (ok && act > 0) {
            if (fwr) {
               // We have to update the key file name
               int nr = 0;
               if ((nr = RpdRenameKeyFile(oldofs,pw)) == 0) {
                  // Write the entry at new position
                  lseek(itab, pw, SEEK_SET);

                  if (ln[slen-1] != '\n') {
                     if (slen >= kMAXPATHLEN -1)
                        ln[slen-1] = '\n';
                     else {
                        ln[slen] = '\n';
                        ln[slen+1] = '\0';
                     }
                  }
                  while (write(itab, ln, strlen(ln)) < 0
                         && GetErrno() == EINTR)
                     ResetErrno();
                  pw += strlen(ln);
               } else
                  RpdDeleteKeyFile(oldofs);
               lseek(itab, pr, SEEK_SET);
            } else
               pw += strlen(ln);
         } else {
            fwr = 1;
         }
         // Set old offset
         oldofs = pr;
      }

      // Truncate file to new length
      if (ftruncate(itab, pw) == -1)
         ErrorInfo("RpdUpdateAuthTab: opt=%d: ftruncate error (errno: %d)",
              opt, GetErrno());

      // Return new file size
      retval = pw;

   } else if (opt == 1) {

      //
      // Add 'line' at the end
      // (check size and cleanup/truncate if needed)

      // Check size ...
      if ((int)(fsize+strlen(line)) > gMAXTABSIZE) {

         // If it is going to be too big, cleanup or truncate first
         fsize = RpdUpdateAuthTab(0,(const char *)0,0,itab);

         // If still too big: delete everything
         if ((int)(fsize+strlen(line)) > gMAXTABSIZE)
            fsize = RpdUpdateAuthTab(-1,(const char *)0,0,itab);
      }
      // We are going to write at the end
      retval = lseek(itab, 0, SEEK_END);

      // Save first RSA public key into file for later use by the
      // same or other rootd/proofd; we will update the tab file
      // only if this operation is successful
      int ntry = 10;
      int rs = 0;
      while ((rs = RpdSavePubKey(gPubKey, retval, gUser)) == 2 && ntry--) {
         // We are here if a file with the same name exists already
         // and can not be deleted: we shift the offset with a
         // dummy entry
         char ltmp[256];
         SPrintf(ltmp, 256,
                 "0 0 %d %d %s error: pubkey file in use: shift offset\n",
                 gRSAKey, gRemPid, gOpenHost.c_str());

         // adds line
         while (write(itab, ltmp, strlen(ltmp)) < 0 && GetErrno() == EINTR)
            ResetErrno();

         // Set to the new end
         retval = lseek(itab, 0, SEEK_END);
      }

      if (rs > 0) {
         // Something wrong
         retval = -1;
         if (gDebug > 0)
            ErrorInfo("RpdUpdateAuthTab: pub key could not be saved (%d)",rs);
      } else {
         // Generate token
         *token = RpdGetRandString(3, 8);   // 8 crypt-like chars
#ifndef R__NOCRYPT
         char *cryptToken = crypt(*token, *token);
#else
         char *cryptToken = rpdcrypt(*token,gRndmSalt.c_str());
#endif
         SPrintf(fbuf, kMAXPATHLEN, "%s %s\n", line, cryptToken);
         if (gDebug > 2)
            ErrorInfo("RpdUpdateAuthTab: token: '%s'", cryptToken);
         // Save it for later use in kSOCKD servers
         gCryptToken = std::string(cryptToken);

         // adds line
         while (write(itab, fbuf, strlen(fbuf)) < 0 && GetErrno() == EINTR)
            ResetErrno();
      }

   } else {

      //
      // Unknown option
      ErrorInfo("RpdUpdateAuthTab: unrecognized option (opt= %d)", opt);
   }

   goingout:
   if (itab != ilck) {
      // unlock the file
      lseek(itab, 0, SEEK_SET);
      if (lockf(itab, F_ULOCK, (off_t) 1) == -1) {
         ErrorInfo("RpdUpdateAuthTab: error unlocking %s",
                   gRpdAuthTab.c_str());
      }

      // closing file ...
      close(itab);
   }

   return retval;
}

//______________________________________________________________________________
int RpdCleanupAuthTab(const char *crypttoken)
{
   // De-activates entry related to token with crypt crypttoken.
   // Returns: 0 if successful
   //         -4 if entry not found or inactive
   //         -1 problems opening auth tab file
   //         -2 problems locking auth tab file
   //         -3 auth tab file does not exists

   int retval = -4;

   if (gDebug > 2)
      ErrorInfo("RpdCleanupAuthTab: Crypt-token: '%s'",crypttoken);

   // Open file for update
   int itab = -1;
   if ((itab = open(gRpdAuthTab.c_str(), O_RDWR)) == -1) {
      if (GetErrno() == ENOENT) {
         if (gDebug > 0)
            ErrorInfo("RpdCleanupAuthTab: file %s does not exist",
                       gRpdAuthTab.c_str());
         return -3;
      } else {
         ErrorInfo("RpdCleanupAuthTab: error opening %s (errno: %d)",
                  gRpdAuthTab.c_str(), GetErrno());
         return -1;
      }
   }

   // lock tab file
   if (lockf(itab, F_LOCK, (off_t) 1) == -1) {
      ErrorInfo("RpdCleanupAuthTab: error locking %s (errno: %d)",
                gRpdAuthTab.c_str(), GetErrno());
      close(itab);
      return -2;
   }
   if (gDebug > 0)
      ErrorInfo("RpdCleanupAuthTab: file LOCKED (ctkn: '%s')",crypttoken);


   // Now access entry or scan over entries
   int pr = 0, pw = 0;
   int nw, lsec, act, remid, pkey;
   char line[kMAXPATHLEN];

   // Set indicators
   if ((pr = lseek(itab, 0, SEEK_SET)) < 0) {
      ErrorInfo("RpdCleanupAuthTab: error lseeking %s (errno: %d)",
                gRpdAuthTab.c_str(), GetErrno());
      close(itab);
      return -2;
   }
   pw = pr;
   while (reads(itab,line, sizeof(line))) {

      pr += strlen(line);
      if (gDebug > 2)
         ErrorInfo("RpdCleanupAuthTab: pr:%d pw:%d (line:%s) (pId:%d)",
                    pr, pw, line, gParentId);

      char dum1[kMAXPATHLEN] = {0}, host[kMAXUSERLEN] = {0}, user[kMAXUSERLEN] = {0},
           ctkn[30] = {0}, dum2[30] = {0};
      nw = sscanf(line, "%d %d %d %d %127s %127s %29s %4095s %29s",
                        &lsec, &act, &pkey, &remid, host, user, ctkn, dum1, dum2);

      int deactivate = 0;

      if (act > 0) {

         if (lsec == 3 && nw == 9) {
            if (!strncmp(dum2,crypttoken,strlen(crypttoken)))
               deactivate = 1;
         } else if (nw == 7) {
            if (!strncmp(ctkn,crypttoken,strlen(crypttoken)))
               deactivate = 1;
         }

         // Deactivate active entries: remote client has gone ...
         if (deactivate) {

            retval = 0;

            // Delete Public Key file
            RpdDeleteKeyFile(pw);

#ifdef R__GLBS
            if (lsec == 3) {
               int shmid = atoi(ctkn);
               struct shmid_ds shm_ds;
               if (shmctl(shmid, IPC_RMID, &shm_ds) == -1) {
                  if (GetErrno() != EIDRM) {
                     ErrorInfo("RpdCleanupAuthTab: unable to mark shared"
                               " memory segment %d (buf:%s)", shmid, ctkn);
                     ErrorInfo("RpdCleanupAuthTab: for destruction"
                               " (errno: %d)", GetErrno());
                     retval++;
                  }
               }
            }
#endif
            // Locate 'act' ... skeep initial spaces, if any
            int slen = (int)strlen(line);
            int ka = 0;
            while (ka < slen && line[ka] == 32)
               ka++;
            // skeep method
            while (ka < slen && line[ka] != 32)
               ka++;
            // skeep spaces before 'act'
            while (ka < slen && line[ka] == 32)
               ka++;
            // This is 'act'
            line[ka] = '0';
            // Make sure there is a '\n' before writing
            int sl = strlen(line);
            if (line[sl-1] != '\n') {
               if (sl >= kMAXPATHLEN -1)
                  line[sl-1] = '\n';
               else {
                  line[sl] = '\n';
                  line[sl+1] = '\0';
               }
            }
            // Write it now
            lseek(itab, pw, SEEK_SET);
            while (write(itab, line, strlen(line)) < 0
                   && GetErrno() == EINTR)
               ResetErrno();
            // We are done
            lseek(itab,  0, SEEK_END);
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
   if ((itab = open(gRpdAuthTab.c_str(), O_RDWR)) == -1) {
      if (GetErrno() == ENOENT) {
         if (gDebug > 0)
            ErrorInfo("RpdCleanupAuthTab: file %s does not exist",
                       gRpdAuthTab.c_str());
         return -3;
      } else {
         ErrorInfo("RpdCleanupAuthTab: error opening %s (errno: %d)",
                  gRpdAuthTab.c_str(), GetErrno());
         return -1;
      }
   }

   // lock tab file
   if (lockf(itab, F_LOCK, (off_t) 1) == -1) {
      ErrorInfo("RpdCleanupAuthTab: error locking %s (errno: %d)",
                gRpdAuthTab.c_str(), GetErrno());
      close(itab);
      //     return retval;
      return -2;
   }
   if (gDebug > 0)
      ErrorInfo("RpdCleanupAuthTab: file LOCKED"
                " (Host: '%s', RemId:%d, OffSet: %d)",
                  Host, RemId, OffSet);

   // Now access entry or scan over entries
   int pr = 0, pw = 0;
   int nw, lsec, act, remid, pkey;
   char line[kMAXPATHLEN];

   // Clean all flag
   int all = (!strcmp(Host, "all") || RemId == 0);

   // Set indicator
   if (all || OffSet < 0)
      pr = lseek(itab, 0, SEEK_SET);
   else
      pr = lseek(itab, OffSet, SEEK_SET);
   if (pr < 0) {
      ErrorInfo("RpdCleanupAuthTab: error lseeking %s (errno: %d)",
                gRpdAuthTab.c_str(), GetErrno());
      close(itab);
      //     return retval;
      return -2;
   }
   pw = pr;
   while (reads(itab,line, sizeof(line))) {

      pr += strlen(line);
      if (gDebug > 2)
         ErrorInfo("RpdCleanupAuthTab: pr:%d pw:%d (line:%s) (pId:%d)",
                    pr, pw, line, gParentId);

      char dumm[kMAXPATHLEN], host[kMAXUSERLEN], user[kMAXUSERLEN], shmbuf[30];
      nw = sscanf(line, "%d %d %d %d %127s %127s %29s %4095s",
                        &lsec, &act, &pkey, &remid, host, user, shmbuf, dumm);

      if (nw > 5) {
         if (all || OffSet > -1 ||
            (strstr(line,Host) && (RemId == remid))) {

            // Delete Public Key file
            RpdDeleteKeyFile(pw);

#ifdef R__GLBS
            if (lsec == 3 && act > 0) {
               int shmid = atoi(shmbuf);
               struct shmid_ds shm_ds;
               if (shmctl(shmid, IPC_RMID, &shm_ds) == -1) {
                  if (GetErrno() != EIDRM) {
                     ErrorInfo("RpdCleanupAuthTab: unable to mark shared"
                               " memory segment %d (buf:%s)", shmid, shmbuf);
                     ErrorInfo("RpdCleanupAuthTab: for destruction"
                               " (errno: %d)", GetErrno());
                     retval++;
                  }
               }
            }
#endif
            // Deactivate active entries: remote client has gone ...
            if (act > 0) {

               // Locate 'act' ... skeep initial spaces, if any
               int slen = (int)strlen(line);
               int ka = 0;
               while (ka < slen && line[ka] == 32)
                  ka++;
               // skeep method
               while (ka < slen && line[ka] != 32)
                  ka++;
               // skeep spaces before 'act'
               while (ka < slen && line[ka] == 32)
                  ka++;
               // This is 'act'
               line[ka] = '0';
               // Make sure there is a '\n' before writing
               int sl = strlen(line);
               if (line[sl-1] != '\n') {
                  if (sl >= kMAXPATHLEN -1)
                     line[sl-1] = '\n';
                  else {
                     line[sl] = '\n';
                     line[sl+1] = '\0';
                  }
               }
               // Write it now
               lseek(itab, pw, SEEK_SET);
               while (write(itab, line, strlen(line)) < 0
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
   close(itab);

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
   bool goodOfs = RpdCheckOffSet(Sec,User,Host,RemId,
                                 OffSet,&tkn,&shmid,&user);
   if (gDebug > 2)
      ErrorInfo("RpdCheckAuthTab: goodOfs: %d", goodOfs);

   // Notify the result of the check
   int tag = 0;
   if (gClientProtocol >= 10) {
      if (goodOfs) {
         if (gClientProtocol > 11) {
            // Generate tag
            RpdInitRand();
            while ((tag = rpd_rand()) == 1) ; // .ne.1 for backward comptibility

            // We will receive the user token next
            NetSend(tag, kROOTD_AUTH);
         } else
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
   }

   // Now Receive Token
   int ofs = *OffSet;
   char *token = 0;
   if (gRSAKey > 0) {
      if (RpdSecureRecv(&token) == -1) {
         ErrorInfo("RpdCheckAuthTab: problems secure-"
                   "receiving token %s",
                   "- may result in authentication failure ");
      }

   } else {
      EMessageTypes kind;
      int lenToken = 9;
      token = new char[lenToken];
      NetRecv(token, lenToken, kind);
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

   // Check tag, if there
   if (token && strlen(token) > 8) {
      // Create hex from tag
      char tagref[9] = {0};
      SPrintf(tagref,9,"%08x",tag);
      if (strncmp(token+8,tagref,8)) {
         ErrorInfo("RpdCheckAuthTab: token tag does not match - failure");
         goodOfs = 0;
      } else
         // Drop tag
         token[8] = 0;
   }

   // Now check Token validity
   if (goodOfs && token && RpdCheckToken(token, tkn)) {

      if (Sec == 3) {
#ifdef R__GLBS
         // kGlobus:
         if (GlbsToolCheckContext(shmid)) {
            retval = 1;
            strlcpy(gUser, user, sizeof(gUser));
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
   if (token) delete[] token;
   if (user) delete[] user;

   return retval;
}

//______________________________________________________________________________
int RpdCheckOffSet(int Sec, const char *User, const char *Host, int RemId,
                   int *OffSet, char **Token, int *ShmId, char **GlbsUser)
{
   // Check offset received from client entry in tab file.

   int retval = 0;
   bool goodOfs = 0;
   int ofs = *OffSet >= 0 ? *OffSet : 0;

   if (gDebug > 2)
      ErrorInfo("RpdCheckOffSet: analyzing: %d %s %s %d %d", Sec, User,
                Host, RemId, *OffSet);

   // Open file
   int itab = open(gRpdAuthTab.c_str(), O_RDWR);
   if (itab == -1) {
      if (GetErrno() == ENOENT)
         ErrorInfo("RpcCheckOffSet: file %s does not exist",
                   gRpdAuthTab.c_str());
      else
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
   if (gDebug > 0)
      ErrorInfo("RpdCheckOffSet: file LOCKED");

   // File is open: set position at wanted location
   if (lseek(itab, ofs, SEEK_SET) < 0) {
      ErrorInfo("RpcCheckOffSet: error lseeking %s (errno: %d)",
                gRpdAuthTab.c_str(), GetErrno());
      close(itab);
      return retval;
   }

   // Now read the entry
   char line[kMAXPATHLEN];
   if (reads(itab,line, sizeof(line)) < 0) {
      ErrorInfo("RpcCheckOffSet: error reading %d bytes from %s (errno: %d)",
                sizeof(line), gRpdAuthTab.c_str(), GetErrno());
      close(itab);
      return retval;
   }

   // and parse its content according to auth method
   int lsec, act, remid, shmid = -1;
   char host[kMAXPATHLEN], usr[kMAXPATHLEN], subj[kMAXPATHLEN],
       dumm[kMAXPATHLEN], tkn[20];
   int nw =
       sscanf(line, "%d %d %d %d %4095s %4095s %19s %4095s",
                    &lsec, &act, &gRSAKey, &remid, host, usr, tkn, dumm);
   if (gDebug > 2)
      ErrorInfo("RpdCheckOffSet: found line: %s", line);

   if (nw > 5 && act > 0) {
      if (lsec == Sec) {
         if (lsec == 3) {
            sscanf(line, "%d %d %d %d %4095s %4095s %d %4095s %19s %4095s",
                         &lsec, &act, &gRSAKey, &remid, host, usr, &shmid, subj, tkn, dumm);
            if ((remid == RemId)
                && !strcmp(host, Host) && !strcmp(subj, User))
               goodOfs = 1;
         } else {
            if ((remid == RemId) &&
                !strcmp(host, Host) && !strcmp(usr, User))
               goodOfs = 1;
         }
      }
   }
   if (!goodOfs) {
      // Tab may have been cleaned in the meantime ... try a scan
      lseek(itab, 0, SEEK_SET);
      ofs = 0;
      while (reads(itab, line, sizeof(line))) {

         nw = sscanf(line, "%d %d %d %d %4095s %4095s %19s %4095s",
                     &lsec, &act, &gRSAKey, &remid, host, usr, tkn, dumm);
         if (gDebug > 2)
            ErrorInfo("RpdCheckOffSet: found line: %s", line);

         if (nw > 5 && act > 0) {
            if (lsec == Sec) {
               if (lsec == 3) {
                  sscanf(line, "%d %d %d %d %4095s %4095s %d %4095s %19s %4095s",
                         &lsec, &act, &gRSAKey, &remid, host, usr, &shmid, subj, tkn, dumm);
                  if ((remid == RemId)
                      && !strcmp(host, Host) && !strcmp(subj, User)) {
                     goodOfs = 1;
                     goto found;
                  }
               } else {
                  if ((remid == RemId) &&
                      !strcmp(host, Host) && !strcmp(usr, User)) {
                     goodOfs = 1;
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

   // Read public key
   std::string pukfile = gRpdKeyRoot;
   pukfile.append(ItoA(*OffSet));
   if (gDebug > 2)
      ErrorInfo("RpdCheckOffSet: RSAKey ofs file: %d %d '%s' ",
                gRSAKey, ofs, pukfile.c_str());

   struct passwd *pw = getpwnam(usr);
   if (pw) {
      uid_t fromUid = getuid();
      uid_t fromEUid = geteuid();
      // The check must be done with 'usr' UIDs to prevent
      // unauthorized users from forcing the server to read
      // manipulated key files.
      if (fromUid == 0)
         if (setresuid(pw->pw_uid, pw->pw_uid, fromEUid) == -1)
            // Since we could not set the user IDs, we will
            // not trust the client
            goodOfs = 0;

      // Get the key now
      if (goodOfs)
         if (RpdGetRSAKeys(pukfile.c_str(), 1) < 1)
            goodOfs = 0;

      // Reset original IDs
      if (getuid() != fromUid)
         if (setresuid(fromUid, fromEUid, pw->pw_uid) == -1)
            goodOfs = 0;

   } else {
      // Since we could not set the user IDs, we will
      // not trust the client
      goodOfs = 0;
      if (gDebug > 0)
         ErrorInfo("RpdCheckOffSet: error in getpwname(%s) (errno: %d)",
                   usr,GetErrno());
   }

   if (gDebug > 2)
      ErrorInfo("RpdCheckOffSet: goodOfs: %d (active: %d)",
                goodOfs, act);

   // Comunicate new offset to remote client
   if (goodOfs) {

      // Rename the key file, if needed
      if (*OffSet > 0 && *OffSet != ofs) {
         if (RpdRenameKeyFile(*OffSet,ofs) > 0) {
            goodOfs = 0;
            // Error: set entry inactive
            RpdCleanupAuthTab(Host,RemId,ofs);
         }
      }

      *OffSet = ofs;
      // return token if requested
      if (Token) {
         const size_t tokenSize = strlen(tkn)+1;
         *Token = new char[tokenSize];
         strlcpy(*Token,tkn,tokenSize);
      }
      if (Sec == 3) {
         if (GlbsUser) {
            const size_t glbsUserSize = strlen(usr)+1;
            *GlbsUser = new char[glbsUserSize];
            strlcpy(*GlbsUser,usr,glbsUserSize);
         }
         if (ShmId)
            *ShmId = shmid;
      }
   }

   return goodOfs;
}

//______________________________________________________________________________
int RpdRenameKeyFile(int oldofs, int newofs)
{
   // Rename public file with new offset
   // Returns: 0 if OK
   //          1 if problems renaming
   int retval = 0;

   // Old name
   std::string oldname = gRpdKeyRoot;
   oldname.append(ItoA(oldofs));
   // New name
   std::string newname = gRpdKeyRoot;
   newname.append(ItoA(newofs));

   if (rename(oldname.c_str(), newname.c_str()) == -1) {
      if (gDebug > 0)
         ErrorInfo("RpdRenameKeyFile: error renaming key file"
                   " %s to %s (errno: %d)",
                   oldname.c_str(),newname.c_str(),GetErrno());
      retval = 2;
   }

   return retval;
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

#ifndef R__NOCRYPT
   char *tkn_crypt = crypt(token, tknref);
   int tlen = 13;
#else
   char *tkn_crypt = rpdcrypt(token,gRndmSalt.c_str());
   int tlen = 16;
#endif

   if (gDebug > 2)
      ErrorInfo("RpdCheckToken: ref:'%s' crypt:'%s'", tknref, tkn_crypt);

   if (!strncmp(tkn_crypt, tknref, tlen))
      return 1;
   else
      return 0;
}

//______________________________________________________________________________
int RpdReUseAuth(const char *sstr, int kind)
{
   // Check the requiring subject has already authenticated during this session
   // and its 'ticket' is still valid.
   // Not implemented for SRP and Krb5 (yet).

   int lenU, offset, opt;
   gOffSet = -1;
   gExistingAuth = 0;
   int auth= 0;

   if (gDebug > 2)
      ErrorInfo("RpdReUseAuth: analyzing: %s, %d", sstr, kind);

   char user[64];

   // kClear
   if (kind == kROOTD_USER) {
      if (!(gReUseAllow & gAUTH_CLR_MSK)) {
         return 0;              // re-authentication required by administrator
      }
      gSec = 0;
      // Decode subject string
      sscanf(sstr, "%d %d %d %d %63s", &gRemPid, &offset, &opt, &lenU, user);
      user[lenU] = '\0';
      if ((gReUseRequired = (opt & kAUTH_REUSE_MSK))) {
         gOffSet = offset;
         if (gRemPid > 0 && gOffSet > -1) {
            auth =
                RpdCheckAuthTab(gSec, user, gOpenHost.c_str(), gRemPid, &gOffSet);
         }
         if ((auth == 1) && (offset != gOffSet))
            auth = 2;
         // Fill gUser and free allocated memory
         strlcpy(gUser, user, sizeof(gUser));
      }
   }
   // kSRP
   if (kind == kROOTD_SRPUSER) {
      if (!(gReUseAllow & gAUTH_SRP_MSK)) {
         return 0;              // re-authentication required by administrator
      }
      gSec = 1;
      // Decode subject string
      sscanf(sstr, "%d %d %d %d %63s", &gRemPid, &offset, &opt, &lenU, user);
      user[lenU] = '\0';
      if ((gReUseRequired = (opt & kAUTH_REUSE_MSK))) {
         gOffSet = offset;
         if (gRemPid > 0 && gOffSet > -1) {
            auth =
                RpdCheckAuthTab(gSec, user, gOpenHost.c_str(), gRemPid, &gOffSet);
         }
         if ((auth == 1) && (offset != gOffSet))
            auth = 2;
         // Fill gUser and free allocated memory
         strlcpy(gUser, user, sizeof(gUser));
      }
   }
   // kKrb5
   if (kind == kROOTD_KRB5) {
      if (!(gReUseAllow & gAUTH_KRB_MSK)) {
         return 0;              // re-authentication required by administrator
      }
      gSec = 2;
      // Decode subject string
      sscanf(sstr, "%d %d %d %d %63s", &gRemPid, &offset, &opt, &lenU, user);
      user[lenU] = '\0';
      if ((gReUseRequired = (opt & kAUTH_REUSE_MSK))) {
         gOffSet = offset;
         if (gRemPid > 0 && gOffSet > -1) {
            auth =
                RpdCheckAuthTab(gSec, user, gOpenHost.c_str(), gRemPid, &gOffSet);
         }
         if ((auth == 1) && (offset != gOffSet))
            auth = 2;
         // Fill gUser and free allocated memory
         strlcpy(gUser, user, sizeof(gUser));
      }
   }
   // kGlobus
   if (kind == kROOTD_GLOBUS) {
      if (!(gReUseAllow & gAUTH_GLB_MSK)) {
         return 0;              //  re-authentication required by administrator
      }
      gSec = 3;
      // Decode subject string
      int lenS;
      sscanf(sstr, "%d %d %d %d %63s", &gRemPid, &offset, &opt, &lenS, user);
      user[lenS] = '\0';
      if ((gReUseRequired = (opt & kAUTH_REUSE_MSK))) {
         gOffSet = offset;
         if (gRemPid > 0 && gOffSet > -1) {
            auth =
                RpdCheckAuthTab(gSec, user, gOpenHost.c_str(), gRemPid, &gOffSet);
         }
         if ((auth == 1) && (offset != gOffSet))
            auth = 2;
      }
   }
   // kSSH
   if (kind == kROOTD_SSH) {
      if (!(gReUseAllow & gAUTH_SSH_MSK)) {
         return 0;              //  re-authentication required by administrator
      }
      gSec = 4;
      // Decode subject string
      char pipe[kMAXPATHLEN];
      sscanf(sstr, "%d %d %d %4095s %d %63s", &gRemPid, &offset, &opt, pipe, &lenU, user);
      user[lenU] = '\0';
      if ((gReUseRequired = (opt & kAUTH_REUSE_MSK))) {
         gOffSet = offset;
         if (gRemPid > 0 && gOffSet > -1) {
            auth =
                RpdCheckAuthTab(gSec, user, gOpenHost.c_str(), gRemPid, &gOffSet);
         }
         if ((auth == 1) && (offset != gOffSet))
            auth = 2;
         // Fill gUser and free allocated memory
         strlcpy(gUser, user, sizeof(gUser));
      }
   }

   // Flag if existing token has been re-used
   if (auth > 0)
      gExistingAuth = 1;

   // Return value
   return auth;
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

#ifdef R__GBLS
   if (Sec == 3 && !gHaveGlobus) {
      ErrorInfo("RpdCheckAuthAllow: meth: 3:"
                " server does not have globus/GSI credentials");
      return 1;
   }
#endif

   std::string theDaemonRc;

   // Check if a non-standard file has been requested
   if (getenv("ROOTDAEMONRC"))
      theDaemonRc = getenv("ROOTDAEMONRC");

   if (theDaemonRc.length() <= 0) {
      if (getuid()) {
         // Check if user has a private daemon access file ...
         struct passwd *pw = getpwuid(getuid());
         if (pw != 0) {
            theDaemonRc = std::string(pw->pw_dir).append("/");
            theDaemonRc.append(gDaemonRc);
         } else {
            if (getenv("ROOTETCDIR")) {
               theDaemonRc = std::string(getenv("ROOTETCDIR")).append("/system");
               theDaemonRc.append(gDaemonRc);
            } else
               theDaemonRc = std::string("/etc/root/system").append(gDaemonRc);
         }
      } else {
         // If running as super-user, check system file only
         if (getenv("ROOTETCDIR")) {
            theDaemonRc = std::string(getenv("ROOTETCDIR")).append("/system");
            theDaemonRc.append(gDaemonRc);
         } else
            theDaemonRc = std::string("/etc/root/system").append(gDaemonRc);
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

      // Open file
      FILE *ftab = fopen(theDaemonRc.c_str(), "r");
      if (ftab == 0) {
         if (GetErrno() == ENOENT)
            ErrorInfo("RpdCheckAuthAllow: file %s does not exist",
                      theDaemonRc.c_str());
         else
            ErrorInfo("RpdCheckAuthAllow: error opening %s (errno: %d)",
                      theDaemonRc.c_str(), GetErrno());
      }
      // Now read the entry
      char line[kMAXPATHLEN], host[kMAXPATHLEN], rest[kMAXPATHLEN],
          cmth[kMAXPATHLEN];
      int nmet = 0, mth[6] = { 0 };

      int cont = 0, jm = -1;
      while (ftab && fgets(line, sizeof(line), ftab)) {
         int i;
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
            strlcpy(rest, pstr, kMAXPATHLEN);
         } else {
            jm = -1;
            // Get 'host' first ...
            nw = sscanf(pstr, "%4095s %4095s", host, rest);
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
            if (!host[0])
               strlcpy(host, "default", kMAXPATHLEN);

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
            nw = sscanf(pstr, "%4095s %4095s", cmth, rest);
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
               strlcpy(tmp, cmth, sizeof(tmp));
            }

            if (strlen(tmp) > 1) {

               for (tmet = 0; tmet < kMAXSEC; tmet++) {
                  if (!rpdstrcasecmp(gAuthMeth[tmet].c_str(), tmp))
                     break;
               }
               if (tmet < kMAXSEC) {
                  if (gDebug > 2)
                     ErrorInfo("RpdCheckAuthAllow: tmet %d", tmet);
               } else {
                  if (gDebug > 1)
                     ErrorInfo("RpdCheckAuthAllow: unknown methods"
                               " %s - ignore", tmp);
                  goto nexti;
               }

            } else {
               tmet = atoi(tmp);
            }
            jm = -1;
            if (gDebug > 2)
               ErrorInfo("RpdCheckAuthAllow: found method %d (have?:%d)",
                         tmet, (tmet >= 0 && tmet < kMAXSEC) ? gHaveMeth[tmet] : 0);
            if (tmet >= 0 && tmet < kMAXSEC) {
               if (gHaveMeth[tmet] == 1) {
                  int ii;
                  for (ii = 0; ii < nmet; ii++) {
                     if (mth[ii] == tmet) {
                        jm = ii;
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
                     char *tmpUI = strdup(gUserIgnore[mth[jm]]);
                     free(gUserIgnore[mth[jm]]);
                     gUserIgnLen[mth[jm]] += kMAXPATHLEN;
                     gUserIgnore[mth[jm]] = new char[gUserIgnLen[mth[jm]]];
                     strlcpy(gUserIgnore[mth[jm]], tmpUI, sizeof(gUserIgnLen[mth[jm]]));
                     free(tmpUI);
                  }
                  char usr[256];
                  if (pd2 != 0) {
                     int ulen = pd2 - pd;
                     strncpy(usr, pd, ulen);
                     usr[ulen] = '\0';
                  } else {
                     strlcpy(usr, pd, sizeof(usr));
                  }
                  struct passwd *pw = getpwnam(usr);
                  if (pw != 0)
                     SPrintf(gUserIgnore[mth[jm]], gUserIgnLen[mth[jm]], "%s %d",
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
                     char *tmpUI = strdup(gUserAllow[mth[jm]]);
                     free(gUserAllow[mth[jm]]);
                     gUserAlwLen[mth[jm]] += kMAXPATHLEN;
                     gUserAllow[mth[jm]] = new char[gUserAlwLen[mth[jm]]];
                     strlcpy(gUserAllow[mth[jm]], tmpUI, sizeof(gUserAlwLen[mth[jm]]));
                     free(tmpUI);
                  }
                  char usr[256];
                  if (pd2 != 0) {
                     int ulen = pd2 - pd;
                     strncpy(usr, pd, ulen);
                     usr[ulen] = '\0';
                  } else {
                     strlcpy(usr, pd, sizeof(usr));
                  }
                  struct passwd *pw = getpwnam(usr);
                  if (pw != 0)
                     SPrintf(gUserAllow[mth[jm]], gUserIgnLen[mth[jm]], "%s %d",
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
      if (ftab)
         fclose(ftab);

      // Host specific directives have been checked for ...
      gMethInit = 1;

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
   char *hh;
   if (!name) {
      hh = RpdGetIP(Host);
      if (gDebug > 2)
         ErrorInfo("RpdCheckHost: Checking Host IP: %s", hh);
   } else {
      const size_t hhSize = strlen(Host)+1;
      hh = new char[hhSize];
      strlcpy(hh,Host,hhSize);
      if (gDebug > 2)
         ErrorInfo("RpdCheckHost: Checking Host name: %s", hh);
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
   const size_t hSize = strlen(host)+1;
   char *h = new char[hSize];
   strlcpy(h,host,hSize);
   char *tk = strtok(h,"*");
   while (tk) {

      char *ps = strstr(hh,tk);
      if (!ps) {
         rc = 0;
         break;
      }
      if (!sos && first && ps == hh)
         starts = 1;
      first = 0;

      if (ps == hh + strlen(hh) - strlen(tk))
         ends = 1;

      tk = strtok(0,"*");

   }
   delete[] h;
   delete[] hh;

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
      std::string alist;
      char cm[5];
      for (i = 0; i < gNumAllow; i++) {
         if (gDebug > 2)
            ErrorInfo("RpdSendAuthList: gTriedMeth[%d]: %d", i,
                      gTriedMeth[i]);
         if (gTriedMeth[i] == 0) {
            SPrintf(cm, 5, " %d",gAllowMeth[i]);
            alist.append(cm);
         }
      }
      NetSend(alist.c_str(), alist.length() + 1, kMESS_STRING);
      if (gDebug > 2)
         ErrorInfo("RpdSendAuthList: sent list: %s", alist.c_str());
   }
}

//______________________________________________________________________________
int RpdSshAuth(const char *sstr)
{
   // Authenitcation via ssh.

   int auth = 0;

   if (gDebug > 2)
      ErrorInfo("RpdSshAuth: contacted by host: %s for user %s",
                gOpenHost.c_str(),sstr);

   // Decode subject string
   char user[kMAXUSERLEN];
   char pipeId[10] = {0};
   int lenU, ofs, opt;
   char rproto[20] = {0};
   sscanf(sstr, "%d %d %d %9s %d %127s %19s", &gRemPid, &ofs, &opt, pipeId, &lenU, user, rproto);

   user[lenU] = '\0';
   gReUseRequired = (opt & kAUTH_REUSE_MSK);
#ifdef R__SSL
   if (gRSASSLKey) {
      // Determine type of RSA key required
      gRSAKey = (opt & kAUTH_RSATY_MSK) ? 2 : 1;
   } else
      gRSAKey = 1;
#else
   gRSAKey = 1;
#endif

   // Check if we have been called to notify failure ...
   if (gRemPid < 0) {

      if (gDebug > 2)
         ErrorInfo
             ("RpdSshAuth: this is a failure notification (%s,%s,%d,%s)",
              user, gOpenHost.c_str(), gRemPid, pipeId);

      struct passwd *pw = getpwnam(user);
      if (pw) {
         std::string pipeFile =
            std::string(pw->pw_dir) + std::string("/RootSshPipe.") + pipeId;
         FILE *fpipe = fopen(pipeFile.c_str(), "r");
         if (!fpipe) {
            pipeFile= gTmpDir + std::string("/RootSshPipe.") + pipeId;
            fpipe = fopen(pipeFile.c_str(), "r");
         }
         char pipe[kMAXPATHLEN];
         if (fpipe) {
            while (fgets(pipe, sizeof(pipe), fpipe)) {
               if (pipe[strlen(pipe)-1] == '\n')
               pipe[strlen(pipe)-1] = 0;
            }
            fclose(fpipe);
            // Remove the temporary file
            unlink(pipeFile.c_str());

            if (SshToolNotifyFailure(pipe))
               ErrorInfo("RpdSshAuth: failure notification may have"
                         " failed ");
         } else {
            if (GetErrno() == ENOENT)
               ErrorInfo("RpdSshAuth: pipe file %s does not exists",
                          pipeFile.c_str());
            else
               ErrorInfo("RpdSshAuth: cannot open pipe file %s"
                      " (errno= %d)",pipeFile.c_str(),GetErrno());
         }

      } else
         ErrorInfo("RpdSshAuth: unable to get user info for '%s'"
                   " (errno: %d)",user,GetErrno());

      gClientProtocol = atoi(rproto);

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
            return auth;
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
         return auth;
      }
#endif
      if (gClientProtocol > 9) {
         SPrintf(buf,20,"OK");
         NetSend(strlen(buf), kROOTD_SSH);
         NetSend(buf, strlen(buf), kMESS_STRING);
         ErrorInfo("RpdSshAuth: failure notified");
      } else {
         NetSend(kErrAuthNotOK,kROOTD_ERR);
         ErrorInfo("RpdSshAuth: failure notified");
      }
      return auth;
   }

   // Protocol of our ssh implementation
   int sshproto = atoi(rproto);

   // Check user existence and get its environment
   struct passwd *pw = getpwnam(user);
   if (!pw) {
      ErrorInfo("RpdSshAuth: entry for user % not found in /etc/passwd",
                user);
      NetSend(-2, kROOTD_SSH);
      return auth;
   }
   // Method cannot be attempted for anonymous users ... (ie data servers )...
   if (!strcmp(pw->pw_shell, "/bin/false")) {
      ErrorInfo("RpdSshAuth: no SSH for anonymous user '%s' ", user);
      NetSend(-2, kROOTD_SSH);
      return auth;
   }

   // Some useful variables
   char *pipeFile = 0;
   char *authFile = 0;
   char *uniquePipe = 0;
   std::string cmdInfo = "";
   int unixFd = -1;
   struct stat st0, st1;

   if (sshproto == 0) {

      // Now we create an internal (UNIX) socket to listen to the
      // result of sshd from ssh2rpd.
      // Path will be /tmp/rootdSSH_<random_string>
      if ((unixFd =
           SshToolAllocateSocket(pw->pw_uid, pw->pw_gid, &uniquePipe)) < 0) {
         ErrorInfo
             ("RpdSshAuth: can't allocate UNIX socket for authentication");
         NetSend(0, kROOTD_SSH);
         delete[] uniquePipe;
         return auth;
      }

      // Open a file to put the pipe to be read by ssh2rpd
      int itmp = 0;
      pipeFile = new char[strlen(pw->pw_dir) + 25];
      SPrintf(pipeFile, strlen(pw->pw_dir) + 25, "%s/RootSshPipe.XXXXXX", pw->pw_dir);
      mode_t oldumask = umask(0700);
      int ipipe = mkstemp(pipeFile);
      if (ipipe == -1) {
         delete[] pipeFile;
         pipeFile = new char[gTmpDir.length() + 25];
         SPrintf(pipeFile, gTmpDir.length() + 25, "%s/RootSshPipe.XXXXXX", gTmpDir.c_str());
         ipipe = mkstemp(pipeFile);
         itmp = 1;
      }
      umask(oldumask);
      FILE *fpipe = 0;
      if (ipipe == -1 || !(fpipe = fdopen(ipipe,"w")) ) {
         ErrorInfo("RpdSshAuth: failure creating pipe file %s (errno: %d)",
                    pipeFile,GetErrno());
         // Could not open the file: notify failure and close
         // properly everything
         if (SshToolNotifyFailure(uniquePipe))
            ErrorInfo("RpdSshAuth: failure notification perhaps"
                      " unsuccessful ... ");
         NetSend(kErrNoPipeInfo, kROOTD_ERR);
         delete[] uniquePipe;
         delete[] pipeFile;
         close(unixFd);
         return auth;
      } else {
         // File open: fill it
         fprintf(fpipe,"%s\n",uniquePipe);
         fclose(fpipe);
         // Set strict protections
         chmod(pipeFile, 0600);
         // Set ownership of the pipe file to the user
         if (getuid() == 0)
            if (chown(pipeFile,pw->pw_uid,pw->pw_gid) == -1)
               ErrorInfo("RpdSshAuth: cannot change ownership of %s (errno: %d)",
                         pipeFile,GetErrno());
      }

      // Get ID
      char *pId = (char *)strstr(pipeFile,"SshPipe.")+strlen("SshPipe.");
      strlcpy(pipeId, pId, sizeof(pipeId));

      // Communicate command to be executed via ssh ...
      std::string rootbindir;
      if (getenv("ROOTBINDIR"))
         rootbindir = getenv("ROOTBINDIR");
      char dbgstr[4] = {0};
      snprintf(dbgstr,3,"%d ",gDebug);
      cmdInfo = std::string(rootbindir).append("/ssh2rpd ");
      cmdInfo.append(dbgstr);
      cmdInfo.append(" ");
      cmdInfo.append(pipeId);

      // Add Tmp dir, if used
      if (itmp) {
         cmdInfo.append(" ");
         cmdInfo.append(gTmpDir);
      }

   } else {
      // New protocol using scp
      // Allocate a file to be overwritten by the client
      authFile = new char[strlen(pw->pw_dir) + 25];
      SPrintf(authFile, strlen(pw->pw_dir) + 25, "%s/RootSshAuth.XXXXXX", pw->pw_dir);
      mode_t oldumask = umask(0700);
      int iauth = mkstemp(authFile);
      if (iauth == -1) {
         if (gDebug > 2)
            ErrorInfo("RpdSshAuth: failure creating Auth file %s (errno: %d)",
                      authFile,GetErrno());
         delete[] authFile;
         authFile = new char[gTmpDir.length() + 25];
         SPrintf(authFile, gTmpDir.length() + 25, "%s/RootSshAuth.XXXXXX", gTmpDir.c_str());
         if ((iauth = mkstemp(authFile)) == -1) {
            ErrorInfo("RpdSshAuth: failure creating Auth file %s (errno: %d)",
                      authFile,GetErrno());
            NetSend(kErrFileOpen, kROOTD_ERR);
            delete[] authFile;
            umask(oldumask);
            return auth;
         }
      }
      umask(oldumask);

      // Store stat result to check changes
      if (fstat(iauth, &st0) == -1)
         ErrorInfo("RpdSshAuth: cannot stat %s",authFile);

      // Make sure the permissions are the rigth ones
      if (fchmod(iauth,0600)) {
         if (gDebug > 0) {
            ErrorInfo("RpdSshAuth: chmod: could not change"
                      " '%s' permission (errno= %d)",authFile, errno);
            ErrorInfo("RpdSshAuth: path (uid,gid) are: %d %d",
                      st0.st_uid, st0.st_gid);
            NetSend(kErrNoChangePermission, kROOTD_ERR);
            delete[] authFile;
            return auth;
         }
      }

      if ((unsigned int)st0.st_uid != pw->pw_uid ||
          (unsigned int)st0.st_gid != pw->pw_gid) {
         if (fchown(iauth, pw->pw_uid, pw->pw_gid)) {
            if (gDebug > 0) {
               ErrorInfo("RpdSshAuth: chown: could not change file"
                         " '%s' ownership (errno= %d)",authFile, errno);
               ErrorInfo("RpdSshAuth: path (uid,gid) are: %d %d",
                         st0.st_uid, st0.st_gid);
               ErrorInfo("RpdSshAuth: may follow authentication problems");
            }
         }
      }

      // Reset reference time
      if (fstat(iauth, &st0) == -1)
         ErrorInfo("RpdSshAuth: cannot stat %s",authFile);

      // Send Back the name of the file
      if (gClientProtocol > 13) {
         // Add the full coordinates, so that it works in all cases,
         // included SSH tunnelling ...
         char hostname[64];
         gethostname(hostname, sizeof(hostname));
         char *cmd = new char[strlen(authFile) + strlen(user) + strlen(hostname) + 5];
         SPrintf(cmd, strlen(authFile) + strlen(user) + strlen(hostname) + 5,
                      " %s@%s:%s ", user, hostname, authFile);
         cmdInfo.append(cmd);
         delete[] cmd;
      } else {
         cmdInfo = std::string(authFile);
      }
   }

   // Add non-standard port, if so
   if (gSshdPort != 22) {
      char sshp[10];
      snprintf(sshp,10," p:%d",gSshdPort);
      cmdInfo.append(sshp);
   }

   // Add key type, if SSL
   if (gRSAKey == 2) {
      char key[10];
      snprintf(key,10," k:%d",gRSAKey);
      cmdInfo.append(key);
   }

   if (gDebug > 2)
      ErrorInfo("RpdSshAuth: sending cmdInfo (%d) %s", cmdInfo.length(),
                cmdInfo.c_str());
   NetSend(cmdInfo.length(), kROOTD_SSH);
   NetSend(cmdInfo.c_str(), cmdInfo.length(), kROOTD_SSH);

   if (sshproto == 0) {

      // Old protocol:
      // Wait for verdict from sshd (via ssh2rpd ...)
      // Additional check on the username ...
      auth = SshToolGetAuth(unixFd, user);

      // Close socket
      SshToolDiscardSocket(uniquePipe, unixFd);

   } else {

      auth = 0;

      // New Protocol:
      // Get result from client and check it locally
      EMessageTypes kind;
      char res[5];
      NetRecv(res, 5, kind);
      if (kind != kROOTD_SSH) {
         ErrorInfo("RpdSshAuth: expecting message kind: %d"
                      " - received: %d", (int)kROOTD_SSH, kind);
         if (!strncmp(res,"1",1))
            NetSend(kErrBadOp, kROOTD_ERR);
         if (unlink(authFile) == -1)
            if (GetErrno() != ENOENT)
               ErrorInfo("RpdSshAuth: cannot unlink file %s (errno: %d)",
                         authFile,GetErrno());
         delete[] authFile;
         // Set to Auth failed
         auth = 0;
         return auth;
      }
      if (!strncmp(res,"0",1)) {
         // Failure
         if (unlink(authFile) == -1)
            if (GetErrno() != ENOENT)
               ErrorInfo("RpdSshAuth: cannot unlink file %s (errno: %d)",
                         authFile,GetErrno());
         delete[] authFile;
         // Set to Auth failed
         auth = 0;
         return auth;
      }
      if (!strncmp(res,"1",1)) {
         // Client pretends success: lets check it locally
         FILE *floc = fopen(authFile,"r");
         if (!floc) {
            ErrorInfo("RpdSshAuth: cannot open auth file:"
                      " %s (errno: %d)", authFile, GetErrno());
            NetSend(kErrFileOpen, kROOTD_ERR);
            if (unlink(authFile) == -1)
               if (GetErrno() != ENOENT)
                  ErrorInfo("RpdSshAuth: cannot unlink file %s (errno: %d)",
                            authFile,GetErrno());
            delete[] authFile;
            // Set to Auth failed
            auth = 0;
            return auth;
         }
         // Stat again file to check for modification
         if (fstat(fileno(floc), &st1) == -1) {
            ErrorInfo("RpdSshAuth: cannot fstat descriptor %d", fileno(floc));
            fclose(floc);
            delete[] authFile;
            // Set to Auth failed
            auth = 0;
            return auth;
         }

         char line[kMAXPATHLEN];
         while (fgets(line, sizeof(line), floc) != 0) {
            // Get rid of '\n'
            if (line[strlen(line) - 1] == '\n')
               line[strlen(line) - 1] = '\0';
            if (gDebug > 2)
               ErrorInfo("RpdSshAuth: read line ... '%s'", line);
            if (!strncmp(line,"k:",2)) {
               // The file contains some meaningful info ...
               auth = 1;
               // Get the key, if there
               char key[4], val[10];
               int nw = sscanf(line,"%3s %9s",key,val);
               if (nw >= 2 && strncmp(val,"-1",2)) {
                  gPubKeyLen = fread((void *)gPubKey,1,sizeof(gPubKey),floc);
                  // Import Key and Determine key type
                  gRSAKey = RpdGetRSAKeys(gPubKey, 0);
                  if (gRSAKey == 0) {
                     ErrorInfo("RpdSshAuth: could not import a valid key");
                     gReUseRequired = 0;
                  }
               }
            }
         }
         fclose(floc);

         // If the file is still empty or scrappy return
         if (auth == 0) {
            // Send error only if the client really got in
            // otherwise it already quit, and sending error
            // would screw up negotiation
            if (gDebug > 2)
               ErrorInfo("RpdSshAuth: %d %d",st1.st_ctime,st0.st_ctime);
            if (st1.st_ctime != st0.st_ctime)
               // the file has been overwritten: the client got in
               // but something went wrong
               NetSend(kErrAuthNotOK, kROOTD_ERR);
            if (unlink(authFile) == -1)
               if (GetErrno() != ENOENT)
                  ErrorInfo("RpdSshAuth: cannot unlink file %s (errno: %d)",
                            authFile, GetErrno());
            delete[] authFile;
            return auth;
         }
      } else {
         ErrorInfo("RpdSshAuth: got unknown reply: %s", res);
         NetSend(kErrBadMess, kROOTD_ERR);
         if (unlink(authFile) == -1)
            if (GetErrno() != ENOENT)
               ErrorInfo("RpdSshAuth: cannot unlink file %s (errno: %d)",
                         authFile,GetErrno());
         delete[] authFile;
         // Set to Auth failed
         auth = 0;
         return auth;
      }
      // Remove the file
      if (unlink(authFile) == -1)
         if (GetErrno() != ENOENT)
            ErrorInfo("RpdSshAuth: cannot unlink file %s (errno: %d)",
                      authFile,GetErrno());
   }

   // If failure, notify and return ...
   if (auth <= 0) {
      if (auth == -1)
         NetSend(kErrWrongUser, kROOTD_ERR);  // Send message length first
      else
         NetSend(kErrAuthNotOK, kROOTD_ERR);  // Send message length first
      delete[] uniquePipe;
      delete[] pipeFile;
      // Set to Auth failed
      auth = 0;
      return auth;
   }
   // notify the client
   if (gDebug > 0 && auth == 1)
      ErrorInfo("RpdSshAuth: user %s authenticated by sshd", user);
   gSec = 4;

   // Save username ...
   strlcpy(gUser, user, sizeof(gUser));

   char line[kMAXPATHLEN];
   if ((gReUseAllow & gAUTH_SSH_MSK) && gReUseRequired) {

      if (sshproto == 0) {

         // Ask for the RSA key
         NetSend(gRSAKey, kROOTD_RSAKEY);

         // Receive the key securely
         if (RpdRecvClientRSAKey()) {
            ErrorInfo("RpdSshAuth: could not import a valid key"
                      " - switch off reuse for this session");
            gReUseRequired = 0;
         }
      }

      // Set an entry in the auth tab file for later (re)use, if required ...
      int offset = -1;
      char *token = 0;
      if (gReUseRequired) {
         SPrintf(line, kMAXPATHLEN, "4 1 %d %d %s %s",
                 gRSAKey, gRemPid, gOpenHost.c_str(), gUser);
         offset = RpdUpdateAuthTab(1, line, &token);
      }
      // Comunicate login user name to client
      SPrintf(line, kMAXPATHLEN, "%s %d", gUser, offset);
      NetSend(strlen(line), kROOTD_SSH);   // Send message length first
      NetSend(line, kMESS_STRING);

      if (gReUseRequired && offset > -1) {
         // Send over the token
         if (!token || (token && RpdSecureSend(token) == -1)) {
            ErrorInfo
                ("RpdSshAuth: problems secure-sending token"
                 " - may result in corrupted token");
         }
         if (token) delete[] token;
      }
      gOffSet = offset;
   } else {
      // Comunicate login user name to client
      SPrintf(line, kMAXPATHLEN, "%s -1", gUser);
      NetSend(strlen(line), kROOTD_SSH);   // Send message length first
      NetSend(line, kMESS_STRING);
   }

   // Release allocated memory
   delete[] uniquePipe;
   delete[] pipeFile;
   delete[] authFile;

   return auth;
}

//______________________________________________________________________________
int RpdKrb5Auth(const char *sstr)
{
   // Authenticate via Kerberos.

   int auth = 0;

#ifdef R__KRB5
   NetSend(1, kROOTD_KRB5);
   // TAuthenticate will respond to our encouragement by sending krb5
   // authentication through the socket

   int retval;

   if (gDebug > 2)
      ErrorInfo("RpdKrb5Auth: analyzing ... %s", sstr);

   if (gClientProtocol > 8) {
      int lenU, ofs, opt;
      char dumm[256];
      // Decode subject string
      sscanf(sstr, "%d %d %d %d %255s", &gRemPid, &ofs, &opt, &lenU, dumm);
      gReUseRequired = (opt & kAUTH_REUSE_MSK);
#ifdef R__SSL
      if (gRSASSLKey) {
         // Determine type of RSA key required
         gRSAKey = (opt & kAUTH_RSATY_MSK) ? 2 : 1;
      } else
         gRSAKey = 1;
#else
      gRSAKey = 1;
#endif
   }

   // Init context
   retval = krb5_init_context(&gKcontext);
   if (retval) {
      ErrorInfo("RpdKrb5Auth: %s while initializing krb5",
            error_message(retval));
      return auth;
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
      return auth;
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
      return auth;
   }

   // get client name
   char *cname;
   if ((retval =
        krb5_unparse_name(gKcontext, ticket->enc_part2->client, &cname))) {
      ErrorInfo("RpdKrb5Auth: unparse failed: %s", error_message(retval));
      RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, 0);
      return auth;
   }
   if (gDebug > 2)
         ErrorInfo("RpdKrb5Auth: name in ticket is: %s",cname);

   using std::string;
   std::string user = std::string(cname);
   free(cname);
   std::string reply = std::string("authenticated as ").append(user);

   // set user name
   // avoid using 'erase' (it is buggy with some compilers)
   snprintf(gUser,64,"%s",user.c_str());
   char *pc = 0;
   // cut off realm
   if ((pc = (char *)strstr(gUser,"@")))
      *pc = '\0';
   // drop instances, if any
   if ((pc = (char *)strstr(gUser,"/")))
      *pc = '\0';

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

   if (gDebug > 2)
      ErrorInfo("RpdKrb5Auth: using ticket file: %s ... ",getenv("KRB5CCNAME"));


   // If the target user is not the owner of the principal
   // check if the user is authorized by the target user
   if (targetUser != gUser) {
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
   }

   // Get credentials if in a PROOF session
   if (gClientProtocol >= 9 &&
      (gService == kPROOFD || gClientProtocol < 11)) {

      char *data = 0;
      int size = 0;
      if (gDebug > 2)
         ErrorInfo("RpdKrb5Auth: receiving forward cred ... ");

      {
         EMessageTypes kind;
         char bufLen[20];
         NetRecv(bufLen, 20, kind);

         if (kind != kROOTD_KRB5) {
            ErrorInfo("RpdKrb5Auth: protocol error, received"
                      " message of type %d instead of %d\n",
                      kind, kROOTD_KRB5);
         }

         size = atoi(bufLen);
         if (gDebug > 3)
            ErrorInfo("RpdKrb5Auth: got len '%s' %d ", bufLen, size);

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

      bool forwarding = true;
      krb5_creds **creds = 0;
      if ((retval = krb5_rd_cred(gKcontext, auth_context,
                                 &forwardCreds, &creds, 0))) {
         ErrorInfo("RpdKrb5Auth: rd_cred failed--%s", error_message(retval));
         forwarding = false;
      }
      if (data) delete[] data;

      struct passwd *pw = getpwnam(gUser);
      if (forwarding && pw) {
         Int_t fromUid = getuid();
         Int_t fromEUid = geteuid();

         if (setresuid(pw->pw_uid, pw->pw_uid, fromEUid) == -1) {
            ErrorInfo("RpdKrb5Auth: can't setuid for user %s", gUser);
            NetSend(kErrNotAllowed, kROOTD_ERR);
            RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, creds);
            return auth;
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
            return auth;
         }

         krb5_ccache cache = 0;
         char ccacheName[256];
         SPrintf(ccacheName,256,"%240s_root_%d",krb5_cc_default_name(context),getpid());
         if ((retval = krb5_cc_resolve(context, ccacheName, &cache))) {
            ErrorInfo("RpdKrb5Auth: cc_default failed--%s",
                      error_message(retval));
            NetSend(kErrNotAllowed, kROOTD_ERR);
            krb5_free_context(context);
            RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, creds);
            return auth;
         }
         {
            char *ccname = new char[strlen("KRB5CCNAME")+strlen(ccacheName)+2];
            SPrintf(ccname, strlen("KRB5CCNAME")+strlen(ccacheName)+2, "KRB5CCNAME=%.*s", strlen(ccacheName), ccacheName);
            putenv(ccname);
         }

         if (gDebug > 5)
            ErrorInfo("RpdKrb5Auth: working (1) on ticket to cache (%s) ... ",
                      krb5_cc_get_name(context,cache));

         // this is not working (why?)
         // this would mean that if a second user comes in, it will tremple
         // the existing one :(
         //       if ((retval = krb5_cc_gen_new(context,&cache))) {
         //          ErrorInfo("RpdKrb5Auth: cc_gen_new failed--%s",
         //                    error_message(retval));
         //          return auth;
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
            return auth;
         }

         if ((retval = krb5_cc_store_cred(context,cache, *creds))) {
            ErrorInfo("RpdKrb5Auth: cc_store_cred failed--%s",
                       error_message(retval));
            NetSend(kErrNotAllowed, kROOTD_ERR);
            krb5_free_context(context);
            RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, creds);
            return auth;
         }
         if (gDebug>5)
            ErrorInfo("RpdKrb5Auth: done ticket to cache (%s) ... ",
                      cacheName);

         if ((retval = krb5_cc_close(context,cache))) {
            ErrorInfo("RpdKrb5Auth: cc_close failed--%s",
                       error_message(retval));
            NetSend(kErrNotAllowed, kROOTD_ERR);
            krb5_free_context(context);
            RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, creds);
            return auth;
         }

         // free context
         krb5_free_context(context);

         //       if ( chown( cacheName, pw->pw_uid, pw->pw_gid) != 0 ) {
         //          ErrorInfo("RpdKrb5Auth: could not change the owner"
         //                    " ship of the cache file %s",cacheName);
         //       }

         if (setresuid(fromUid,fromEUid,pw->pw_uid) == -1) {
            ErrorInfo("RpdKrb5Auth: can't setuid back to original uid");
            NetSend(kErrNotAllowed, kROOTD_ERR);
            RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, creds);
            return auth;
         }
      }

      // free creds
      krb5_free_tgt_creds(gKcontext,creds);
   }

   NetSend(reply.c_str(), kMESS_STRING);

   // free allocated stuff
   RpdFreeKrb5Vars(gKcontext, server, ticket, auth_context, (krb5_creds **)0);

   // Authentication was successfull
   auth = 1;
   gSec = 2;

   if (gClientProtocol > 8) {

      char line[kMAXPATHLEN];
      if ((gReUseAllow & gAUTH_KRB_MSK) && gReUseRequired) {

         // Ask for the RSA key
         NetSend(gRSAKey, kROOTD_RSAKEY);

         // Receive the key securely
         if (RpdRecvClientRSAKey()) {
            ErrorInfo("RpdKrb5Auth: could not import a valid key"
                      " - switch off reuse for this session");
            gReUseRequired = 0;
         }

         // Set an entry in the auth tab file for later (re)use,
         // if required ...
         int offset = -1;
         char *token = 0;
         if (gReUseRequired) {
            SPrintf(line, kMAXPATHLEN, "2 1 %d %d %s %s",
                    gRSAKey, gRemPid, gOpenHost.c_str(), gUser);
            offset = RpdUpdateAuthTab(1, line, &token);
            if (gDebug > 2)
               ErrorInfo("RpdKrb5Auth: line:%s offset:%d", line, offset);
         }
         // Comunicate login user name to client
         SPrintf(line, kMAXPATHLEN, "%s %d", gUser, offset);
         NetSend(strlen(line), kROOTD_KRB5);   // Send message length first
         NetSend(line, kMESS_STRING);

         // Send Token
         if (gReUseRequired && offset > -1) {
            if (!token || (token && RpdSecureSend(token) == -1)) {
               ErrorInfo("RpdKrb5Auth: problems secure-sending token"
                         " - may result in corrupted token");
            }
            if (token) delete[] token;
         }
         gOffSet = offset;

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

   return auth;
}

//______________________________________________________________________________
int RpdSRPUser(const char *sstr)
{
   // Use Secure Remote Password protocol.
   // Check user id in $HOME/.srootdpass file.

   int auth = 0;

   if (!*sstr) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: bad user name");
      return auth;
   }

#ifdef R__SRP

   // Decode subject string
   char user[kMAXUSERLEN] = { 0 };
   if (gClientProtocol > 8) {
      int lenU, ofs, opt;
      char dumm[20];
      sscanf(sstr, "%d %d %d %d %127s %19s", &gRemPid, &ofs, &opt, &lenU, user, dumm);
      lenU = (lenU > kMAXUSERLEN-1) ? kMAXUSERLEN-1 : lenU;
      user[lenU] = '\0';
      gReUseRequired = (opt & kAUTH_REUSE_MSK);
#ifdef R__SSL
      if (gRSASSLKey) {
         // Determine type of RSA key required
         gRSAKey = (opt & kAUTH_RSATY_MSK) ? 2 : 1;
      } else
         gRSAKey = 1;
#else
      gRSAKey = 1;
#endif
   } else {
      SPrintf(user,kMAXUSERLEN,"%s",sstr);
   }

   struct passwd *pw = getpwnam(user);
   if (!pw) {
      NetSend(kErrNoUser, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: user %s unknown", user);
      return auth;
   }
   // Method cannot be attempted for anonymous users ... (ie data servers )...
   if (!strcmp(pw->pw_shell, "/bin/false")) {
      NetSend(kErrNotAllowed, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: no SRP for anonymous user '%s' ", user);
      return auth;
   }
   // If server is not started as root and user is not same as the
   // one who started rootd then authetication is not ok.
   uid_t uid = getuid();
   if (uid && uid != pw->pw_uid) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: user not same as effective user of rootd");
      return auth;
   }

   NetSend(auth, kROOTD_AUTH);

   strlcpy(gUser, user, sizeof(gUser));

   std::string srootdpass, srootdconf;
   if (gAltSRPPass.length()) {
      srootdpass = gAltSRPPass;
   } else {
      srootdpass = std::string(pw->pw_dir).append(gSRootdPass);
   }
   srootdconf = srootdpass + std::string(".conf");

   FILE *fp1 = fopen(srootdpass.c_str(), "r");
   if (!fp1) {
      NetSend(kErrFileOpen, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: error opening %s", srootdpass.c_str());
      return auth;
   }
   FILE *fp2 = fopen(srootdconf.c_str(), "r");
   if (!fp2) {
      NetSend(kErrFileOpen, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: error opening %s", srootdconf.c_str());
      if (fp1)
         fclose(fp1);
      return auth;
   }

   struct t_pw *tpw = t_openpw(fp1);
   if (!tpw) {
      NetSend(kErrFileOpen, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: unable to open password file %s",
                srootdpass.c_str());
      fclose(fp1);
      fclose(fp2);
      return auth;
   }

   struct t_conf *tcnf = t_openconf(fp2);
   if (!tcnf) {
      NetSend(kErrFileOpen, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: unable to open configuration file %s",
                srootdconf.c_str());
      t_closepw(tpw);
      fclose(fp1);
      fclose(fp2);
      return auth;
   }
#if R__SRP_1_1
   struct t_server *ts = t_serveropen(gUser, tpw, tcnf);
#else
   struct t_server *ts = t_serveropenfromfiles(gUser, tpw, tcnf);
#endif

   if (tcnf)
      t_closeconf(tcnf);
   if (tpw)
      t_closepw(tpw);
   if (fp2)
      fclose(fp2);
   if (fp1)
      fclose(fp1);

   if (!ts) {
      NetSend(kErrNoUser, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: user %s not found SRP password file", gUser);
      return auth;
   }

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
      return auth;
   }
   if (kind != kROOTD_SRPA) {
      NetSend(kErrFatal, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: expected kROOTD_SRPA message");
      return auth;
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
      return auth;
   }
   if (kind != kROOTD_SRPRESPONSE) {
      NetSend(kErrFatal, kROOTD_ERR);
      ErrorInfo("RpdSRPUser: expected kROOTD_SRPRESPONSE message");
      return auth;
   }

   unsigned char cbuf[20];
   t_fromhex((char *) cbuf, hexbuf);

   if (!t_serververify(ts, cbuf)) {

      // authentication successful
      if (gDebug > 0)
         ErrorInfo("RpdSRPUser: user %s authenticated", gUser);
      auth = 1;
      gSec = 1;

      if (gClientProtocol > 8) {

         char line[kMAXPATHLEN];
         if ((gReUseAllow & gAUTH_SRP_MSK) && gReUseRequired) {

            // Ask for the RSA key
            NetSend(gRSAKey, kROOTD_RSAKEY);

            // Receive the key securely
            if (RpdRecvClientRSAKey()) {
               ErrorInfo
                   ("RpdSRPAuth: could not import a valid key"
                    " - switch off reuse for this session");
               gReUseRequired = 0;
            }

            // Set an entry in the auth tab file for later (re)use, if required ...
            int offset = -1;
            char *token = 0;
            if (gReUseRequired) {
               SPrintf(line, kMAXPATHLEN, "1 1 %d %d %s %s",
                       gRSAKey, gRemPid, gOpenHost.c_str(), gUser);
               offset = RpdUpdateAuthTab(1, line, &token);
            }
            // Comunicate login user name to client
            SPrintf(line, kMAXPATHLEN, "%s %d", gUser, offset);
            NetSend(strlen(line), kROOTD_SRPUSER);   // Send message length first
            NetSend(line, kMESS_STRING);

            if (gReUseRequired && offset > -1) {
               // Send Token
               if (RpdSecureSend(token) == -1) {
                  ErrorInfo("RpdSRPUser: problems secure-sending token"
                            " - may result in corrupted token");
               }
               if (token) delete[] token;
            }
            gOffSet = offset;

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
         return auth;
      }
   }

   t_serverclose(ts);

#else
   NetSend(0, kROOTD_SRPUSER);
#endif
   return auth;
}

//______________________________________________________________________________
int RpdCheckHostsEquiv(const char *host, const char *ruser,
                       const char *user, int &errout)
{
   // Check if the requesting {host,user} can be granted immediate
   // login on the base of the information found in /etc/hosts.equiv
   // and/or $HOME/.rhosts. The two files must be trustable, i.e. owned
   // and modifiable only by 'root' and by 'user', respectively (0600).
   // Returns 1 in case access can be granted, 0 in any other case
   // (errout contains a code for error logging on the client side)
   //
   // NB: entries granting access in one of the two files cannot be
   //     overriden in the other file; so, system admins cannot close
   //     access from a host and user cannot stop access to their
   //     account if the administrator has decided so; as an example,
   //     if this entry is found in /etc/hosts.equiv
   //
   //     remote.host.dom auser
   //
   //     (allowing user named 'auser' from host 'remote.host.dom' to
   //     login to any non-root local account without specifying a
   //     password) the following entries in $home/.rhosts are ignored
   //
   //     remote.host.dom -auser
   //     -remote.host.dom
   //
   //     and access to 'auser' is always granted. This is a "feature"
   //     of ruserok.
   //

   int rc = 0;

   // Effective uid
   int rootuser = 0;
   if (!geteuid() && !getegid())
      rootuser = 1;

   // Check the files only if i) at least one exists; ii) those existing
   // have the right permission settings
   bool badfiles = 0;
   int  nfiles = 0;

   // Check system file /etc/hosts.equiv if non-root
   char hostsequiv[20] = { "/etc/hosts.equiv" };
   if (!rootuser) {

      // Get info about the file ...
      struct stat st;
      if (stat(hostsequiv,&st) == -1) {
         if (GetErrno() != ENOENT) {
            ErrorInfo("RpdCheckHostsEquiv: cannot stat /etc/hosts.equiv"
                      " (errno: %d)",GetErrno());
            badfiles = 1;
         } else
            if (gDebug > 1)
               ErrorInfo("RpdCheckHostsEquiv: %s does not exist",
                         hostsequiv);
      } else {

         // Require 'root' ownership
         if (st.st_uid || st.st_gid) {
            if (gDebug > 0)
               ErrorInfo("RpdCheckHostsEquiv: /etc/hosts.equiv not owned by"
                         " system (uid: %d, gid: %d)",st.st_uid,st.st_gid);
            badfiles = 1;
         } else {

            // Require WRITE permission only for owner
            if ((st.st_mode & S_IWGRP) || (st.st_mode & S_IWOTH)) {
               if (gDebug > 0)
                  ErrorInfo("RpdCheckHostsEquiv: group or others have write"
                            " permission on /etc/hosts.equiv: do not trust"
                            " it (g: %d, o: %d)",
                            (st.st_mode & S_IWGRP),(st.st_mode & S_IWOTH));
               badfiles = 1;
            } else
               // Good file
               nfiles++;
         }
      }
   }

   // Check local file
   char rhosts[kMAXPATHLEN] = {0};
   if (!badfiles) {

      struct passwd *pw = getpwnam(user);
      if (pw) {
         int ldir = strlen(pw->pw_dir);
         ldir = (ldir > kMAXPATHLEN - 9) ? (kMAXPATHLEN - 9) : ldir;
         memcpy(rhosts,pw->pw_dir,ldir);
         memcpy(rhosts+ldir,"/.rhosts",8);
         rhosts[ldir+8] = 0;
         if (gDebug > 2)
            ErrorInfo("RpdCheckHostsEquiv: checking for user file %s ...",rhosts);
      } else {
         if (gDebug > 0)
            ErrorInfo("RpdCheckHostsEquiv: cannot get user info with getpwnam"
                   " (errno: %d)",GetErrno());
         badfiles = 1;
      }

      if (!badfiles) {
         // Check the $HOME/.rhosts file ... ownership and protections
         struct stat st;
         if (stat(rhosts,&st) == -1) {
            if (GetErrno() != ENOENT) {
               ErrorInfo("RpdCheckHostsEquiv: cannot stat $HOME/.rhosts"
                      " (errno: %d)",GetErrno());
               badfiles = 1;
            } else
               ErrorInfo("RpdCheckHostsEquiv: %s/.rhosts does not exist",
                         pw->pw_dir);
         } else {

            // Only use file when its access rights are 0600
            if (!S_ISREG(st.st_mode) || S_ISDIR(st.st_mode) ||
                (st.st_mode & 0777) != (S_IRUSR | S_IWUSR)) {
               if (gDebug > 0)
                  ErrorInfo("RpdCheckHostsEquiv: unsecure permission setting"
                            " found for $HOME/.rhosts: 0%o (must be 0600)",
                            (st.st_mode & 0777));
               badfiles = 1;
            } else
               // Good file
               nfiles++;
         }
      }
   }

   // if files are not available or have wrong permissions or are
   // not accessible, give up
   if (!nfiles) {
      if (gDebug > 0)
         ErrorInfo("RpdCheckHostsEquiv: no files to check");
      errout = 1;
      if (badfiles) {
         if (gDebug > 0)
            ErrorInfo("RpdCheckHostsEquiv: config files cannot be used"
                      " (check permissions)");
         errout = 2;
      }
      return rc;
   }

   // Ok, now use ruserok to find out if {host,ruser,user}
   // is trusted
#if defined(__sgi) || defined(_AIX)
   if (ruserok((char*)host,rootuser,(char*)ruser,(char*)user) == 0) {
#else
   if (ruserok(host,rootuser,ruser,user) == 0) {
#endif
      if (gDebug > 0)
         ErrorInfo("RpdCheckHostsEquiv: remote user %s authorized to"
                   " access %s's area",ruser,user);
      rc = 1;
   } else {
      if (gDebug > 0)
         ErrorInfo("RpdCheckHostsEquiv: no special permission from"
                   " %s or %s",hostsequiv,rhosts);
      errout = 3;
   }

   return rc;
}

//______________________________________________________________________________
int RpdCheckSpecialPass(const char *passwd)
{
   // Check received user's password against password in $HOME/.rootdpass.
   // The password is retrieved in RpdUser and temporarly saved in gPasswd.
   // Returns 1 in case of success authentication, 0 otherwise.

   // Check inputs
   if (!passwd)
      return 0;

   // and the saved the password
   if (strlen(gPasswd) <= 0)
      return 0;

   // Ok, point to the saved passwd (retrieved in RpdUser)
   char *rootdpass = gPasswd;
   int n = 0;

   if (gClientProtocol > 8 && gSaltRequired > 0) {
      n = strlen(rootdpass);
      if (strncmp(passwd, rootdpass, n + 1) != 0) {
         if (gDebug > 0)
            ErrorInfo("RpdCheckSpecialPass: wrong password");
         rpdmemset((volatile void *)rootdpass,0,n);
         return 0;
      }
   } else {
#ifndef R__NOCRYPT
      char *pass_crypt = crypt(passwd, rootdpass);
#else
      char *pass_crypt = (char *)passwd;
#endif
      n = strlen(rootdpass);
      if (strncmp(pass_crypt, rootdpass, n+1) != 0) {
         if (gDebug > 0)
            ErrorInfo("RpdCheckSpecialPass: wrong password");
         rpdmemset((volatile void *)rootdpass,0,n);
         return 0;
      }
   }

   if (gDebug > 0)
      ErrorInfo
          ("RpdCheckSpecialPass: user %s authenticated via ~/.rootdpass",
           gUser);

   rpdmemset((volatile void *)rootdpass,0,n);
   return 1;
}

//______________________________________________________________________________
int RpdPass(const char *pass, int errheq)
{
   // Check user's password.

   char passwd[128];
   char *passw;
   char *pass_crypt;
   struct passwd *pw;
#ifdef R__SHADOWPW
   struct spwd *spw;
#endif
   int afs_auth = 0;
#ifdef R__AFS
   char *reason;
#endif

   if (gDebug > 2)
      ErrorInfo("RpdPass: Enter (pass length: %d)", (int)strlen(pass));

   int auth = 0;
   errheq = (errheq > -1 && errheq < 4) ? errheq : 0;
   if (!*gUser) {
      if (gClientProtocol > 11)
         NetSend(gUsrPwdErr[0][errheq], kROOTD_ERR);
      else
         NetSend(kErrFatal, kROOTD_ERR);
      if (gDebug > 0)
         ErrorInfo("RpdPass: user needs to be specified first");
      return auth;
   }

   if (!pass) {
      if (gClientProtocol > 11)
         NetSend(gUsrPwdErr[1][errheq], kROOTD_ERR);
      else
         NetSend(kErrNoPasswd, kROOTD_ERR);
      if (gDebug > 0)
         ErrorInfo("RpdPass: no password specified");
      return auth;
   }
   int n = strlen(pass);
   // Passwd length should be in the correct range ...
   if (!n) {
      if (gClientProtocol > 11)
         NetSend(gUsrPwdErr[1][errheq], kROOTD_ERR);
      else
         NetSend(kErrBadPasswd, kROOTD_ERR);
      if (gDebug > 0)
         ErrorInfo("RpdPass: null passwd not allowed");
      return auth;
   }
   if (n > (int) sizeof(passwd)) {
      if (gClientProtocol > 11)
         NetSend(gUsrPwdErr[1][errheq], kROOTD_ERR);
      else
         NetSend(kErrBadPasswd, kROOTD_ERR);
      if (gDebug > 0)
         ErrorInfo("RpdPass: passwd too long");
      return auth;
   }
   // Inversion is done in RpdUser, if needed
   strlcpy(passwd, pass, sizeof(passwd));

   // Special treatment for anonimous ...
   if (gAnon) {
      strlcpy(gPasswd, passwd, sizeof(gPasswd));
      goto authok;
   }
   // ... and SpecialPass ...
   if (RpdCheckSpecialPass(passwd)) {
      goto authok;
   }
   // Get local passwd info for gUser
   if (!(pw = getpwnam(gUser))) {
      ErrorInfo("RpdPass: getpwnam failed!");
      return auth;
   }

#ifdef R__AFS
   void *tok = GetAFSToken(gUser, passwd, 0, -1, &reason);
   afs_auth = (tok) ? 1 : 0;
   // We do not need the token anymore
   DeleteAFSToken(tok);
   if (!afs_auth) {
      if (gDebug > 0)
         ErrorInfo("RpdPass: AFS login failed for user %s: %s",
                    gUser, reason);
      // try conventional login...
#endif

#ifdef R__SHADOWPW
      // System V Rel 4 style shadow passwords
      if ((spw = getspnam(gUser)) == 0) {
         if (gDebug > 0)
            ErrorInfo("RpdPass: Shadow passwd not available for user %s",
                   gUser);
         passw = pw->pw_passwd;
      } else
         passw = spw->sp_pwdp;
#else
      passw = pw->pw_passwd;
#endif
#ifndef R__NOCRYPT
      if (gClientProtocol <= 8 || !gSaltRequired) {
         char salt[20] = {0};
         int lenS = 2;
         if (!strncmp(passw, "$1$", 3)) {
            // Shadow passwd
            char *pd = strstr(passw + 4, "$");
            lenS = (int) (pd - passw);
            strncpy(salt, passw, lenS);
         } else
            strncpy(salt, passw, lenS);
         salt[lenS] = 0;
         pass_crypt = crypt(passwd, salt);   // Comment this
      } else {
         pass_crypt = passwd;
      }
#else
      pass_crypt = passwd;
#endif
      n = strlen(passw);
      if (strncmp(pass_crypt, passw, n + 1) != 0) {
         if (gClientProtocol > 11)
            NetSend(gUsrPwdErr[1][errheq], kROOTD_ERR);
         else
            NetSend(kErrBadPasswd, kROOTD_ERR);
         if (gDebug > 0)
            ErrorInfo("RpdPass: invalid password for user %s", gUser);
         return auth;
      }
      if (gDebug > 2)
         ErrorInfo("RpdPass: valid password for user %s", gUser);
#ifdef R__AFS
   } else                            // afs_auth
      if (gDebug > 2)
         ErrorInfo("RpdPass: AFS login successful for user %s", gUser);
#endif

   authok:
   auth = afs_auth ? 5 : 1;
   gSec = 0;

   if (gClientProtocol > 8) {
      // Set an entry in the auth tab file for later (re)use, if required ...
      int offset = -1;
      char *token = 0;
      char line[kMAXPATHLEN];
      if ((gReUseAllow & gAUTH_CLR_MSK) && gReUseRequired) {

         SPrintf(line, kMAXPATHLEN, "0 1 %d %d %s %s",
                 gRSAKey, gRemPid, gOpenHost.c_str(), gUser);
         if (!afs_auth || gService == kPROOFD)
            offset = RpdUpdateAuthTab(1, line, &token);
         if (gDebug > 2)
            ErrorInfo("RpdPass: got offset %d", offset);

         // Comunicate login user name to client
         SPrintf(line, kMAXPATHLEN, "%s %d", gUser, offset);
         if (gDebug > 2)
            ErrorInfo("RpdPass: sending back line %s", line);
         NetSend(strlen(line), kROOTD_PASS);   // Send message length first
         NetSend(line, kMESS_STRING);

         if (offset > -1) {
            if (gDebug > 2)
               ErrorInfo("RpdPass: sending token %s (Crypt: %d)", token,
                         gCryptRequired);
            if (gCryptRequired) {
               // Send over the token
               if (RpdSecureSend(token) == -1) {
                  if (gDebug > 0)
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
            delete[] token;
         }
         gOffSet = offset;

      } else {
         // Comunicate login user name to client
         SPrintf(line, kMAXPATHLEN, "%s -1", gUser);
         if (gDebug > 2)
            ErrorInfo("RpdPass: sending back line %s", line);
         NetSend(strlen(line), kROOTD_PASS);   // Send message length first
         NetSend(line, kMESS_STRING);
      }
   }

   return auth;
}

//______________________________________________________________________________
int RpdGlobusInit()
{
   // Prepare for globus authentication: check hostcer.conf and get
   // the credential handle. This is run once at daemon start-up

#ifdef R__GLBS
   // Now we open the certificates and we check if we are able to
   // autheticate the client. In the affirmative case we initialize
   // our credentials and we send our subject name to the client ...
   // NB: we look first for a specific certificate for ROOT (default
   // location under /etc/grid-security/root); if this is does not
   // work we try to open the host certificate, which however may
   // require super-user privileges; finally we check if valid proxies
   // (for the user who started the server) are available.
   char *subject_name = 0;
   int certRc = GlbsToolCheckCert(&subject_name);
   if (certRc)
      certRc = GlbsToolCheckProxy(&subject_name);
   if (certRc) {
      ErrorInfo("RpdGlobusInit: no valid server credentials found: globus disabled");
      gHaveGlobus = 0;
      return 1;
   } else {

      // Save the subject name
      gGlobusSubjName = subject_name;
      delete [] subject_name;

      // Inquire Globus credentials:
      // This is looking to file X509_USER_CERT for valid a X509 cert (default
      // /etc/grid-security/hostcert.pem) and to dir X509_CERT_DIR for trusted CAs
      // (default /etc/grid-security/certificates).
      OM_uint32 majStat = 0;
      OM_uint32 minStat = 0;
      if ((majStat =
           globus_gss_assist_acquire_cred(&minStat, GSS_C_ACCEPT,
                                          &gGlbCredHandle)) !=
          GSS_S_COMPLETE) {
         GlbsToolError("RpdGlobusInit: gss_assist_acquire_cred", majStat,
                       minStat, 0);
         if (getuid() > 0)
            ErrorInfo("RpdGlobusInit: non-root: make sure you have"
                      " initialized (manually) your proxies");
         return 1;
      }
   }
#endif
   // Done
   return 0;
}

//______________________________________________________________________________
int RpdGlobusAuth(const char *sstr)
{
   // Authenticate via Globus.

   int auth = 0;

#ifndef R__GLBS

   if (sstr) { }  // use sstr
   NetSend(0, kROOTD_GLOBUS);
   return auth;

#else

   if (!gHaveGlobus) {
      // No valid credentials
      if (sstr) { }  // use sstr
      return auth;
   }

   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;
   OM_uint32 GssRetFlags = 0;
   gss_ctx_id_t GlbContextHandle = GSS_C_NO_CONTEXT;
   gss_cred_id_t GlbDelCredHandle = GSS_C_NO_CREDENTIAL;
   int GlbTokenStatus = 0;
   char *GlbClientName;
   FILE *FILE_SockFd;
   char *gridmap_default = "/etc/grid-security/grid-mapfile";
   EMessageTypes kind;
   int lSubj, offset = -1;
   char *user = 0;
   int ulen = 0;

   if (gDebug > 2)
      ErrorInfo("RpdGlobusAuth: contacted by host: %s", gOpenHost.c_str());

   // Tell the remote client that we may accept Globus credentials ...
   NetSend(1, kROOTD_GLOBUS);

   // Decode subject string
   char Subj[kMAXPATHLEN];
   int opt;
   char dumm[20];
   sscanf(sstr, "%d %d %d %d %4095s %19s", &gRemPid, &offset, &opt, &lSubj, Subj, dumm);

   Subj[lSubj] = '\0';
   gReUseRequired = (opt & kAUTH_REUSE_MSK);
#ifdef R__SSL
   if (gRSASSLKey) {
      // Determine type of RSA key required
      gRSAKey = (opt & kAUTH_RSATY_MSK) ? 2 : 1;
   } else
      gRSAKey = 1;
#else
   gRSAKey = 1;
#endif
   if (gDebug > 2)
      ErrorInfo("RpdGlobusAuth: gRemPid: %d, Subj: %s (%d %d)", gRemPid,
                Subj, lSubj, strlen(Subj));

   if (gClientProtocol < 17) {
      // GlbClientName will be determined from the security context ...
      // Now wait for client to communicate the issuer name of the certificate ...
      char *answer = new char[20];
      NetRecv(answer, (int) sizeof(answer), kind);
      if (kind != kMESS_STRING) {
         Error(gErr, kErrAuthNotOK,
                "RpdGlobusAuth: client_issuer_name:received unexpected"
                " type of message (%d)",kind);
         if (answer) delete[] answer;
         return auth;
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
         return auth;
      }
      if (gDebug > 2)
         ErrorInfo("RpdGlobusAuth: client issuer name is: %s",
                   client_issuer_name);
   }

   // Send our subject to the clients: it is needed to start
   // the handshake
   int sjlen = gGlobusSubjName.length() + 1;
   int bsnd = NetSend(sjlen, kROOTD_GLOBUS);
   if (gDebug > 2)
      ErrorInfo("RpdGlobusAuth: sent: %d (due >=%d))", bsnd, 2 * sizeof(sjlen));
   bsnd = NetSend(gGlobusSubjName.c_str(), sjlen, kMESS_STRING);
   if (gDebug > 2)
      ErrorInfo("RpdGlobusAuth: sent: %d (due >=%d))", bsnd, sjlen);

   // We need to associate a FILE* stream with the socket
   // It will automatically closed when the socket will be closed ...
   FILE_SockFd = fdopen(NetGetSockFd(), "w+");

   // Now we are ready to start negotiating with the Client
   if ((MajStat =
        globus_gss_assist_accept_sec_context(&MinStat, &GlbContextHandle,
                                             gGlbCredHandle, &GlbClientName,
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
      return auth;
   } else {
      auth = 1;
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
            ErrorInfo("RpdGlobusAuth: Pointer to del cred is %p", GlbDelCredHandle);
      } else {
         Error(gErr, kErrAuthNotOK,
               "RpdGlobusAuth: did not get delegated credentials (RetFlags: 0x%x)",
               GssRetFlags);
         return auth;
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
         return auth;
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
         ErrorInfo("RpdGlobusAuth: user: %s", user);
   }
   if (!strcmp(user, "anonymous"))
      user = strdup(AnonUser);
   if (!strcmp(user, AnonUser))
      gAnon = 1;

   // No reuse for anonymous users
   gReUseRequired = (gAnon == 1) ? 0 : gReUseRequired;

   // Fill gUser and free allocated memory
   ulen = strlen(user);
   strncpy(gUser, user, ulen + 1);

   char line[kMAXPATHLEN];
   if ((gReUseAllow & gAUTH_GLB_MSK) && gReUseRequired) {

      // Ask for the RSA key
      NetSend(gRSAKey, kROOTD_RSAKEY);

      // Receive the key securely
      if (RpdRecvClientRSAKey()) {
         ErrorInfo
             ("RpdGlobusAuth: could not import a valid key"
              " - switch off reuse for this session");
         gReUseRequired = 0;
      }

      // Store security context and related info for later use ...
      offset = -1;
      char *token = 0;
      if (gReUseRequired) {
         int ShmId = GlbsToolStoreContext(GlbContextHandle, user);
         if (ShmId > 0) {
            SPrintf(line, kMAXPATHLEN, "3 1 %d %d %s %s %d %s",
                    gRSAKey, gRemPid, gOpenHost.c_str(),
                    user, ShmId, GlbClientName);
            offset = RpdUpdateAuthTab(1, line, &token);
         } else if (gDebug > 0)
            ErrorInfo
                ("RpdGlobusAuth: unable to export context to shm for later use");
      }
      // Comunicate login user name to client (and token)
      SPrintf(line, kMAXPATHLEN, "%s %d", gUser, offset);
      NetSend(strlen(line), kROOTD_GLOBUS);   // Send message length first
      NetSend(line, kMESS_STRING);

      if (gReUseRequired && offset > -1) {
         // Send Token
         if (RpdSecureSend(token) == -1) {
            ErrorInfo("RpdGlobusAuth: problems secure-sending token"
                      " - may result in corrupted token");
         }
         if (token) delete[] token;
      }
      gOffSet = offset;
   } else {
      // Comunicate login user name to client (and token)
      SPrintf(line, kMAXPATHLEN, "%s %d", gUser, offset);
      NetSend(strlen(line), kROOTD_GLOBUS);   // Send message length first
      NetSend(line, kMESS_STRING);
   }

   // and free allocated memory
   free(user);
   free(GlbClientName);

   if (gDebug > 0)
      ErrorInfo("RpdGlobusAuth: client mapped to local user %s ", gUser);

   return auth;

#endif
}

//______________________________________________________________________________
int RpdRfioAuth(const char *sstr)
{
   // Check if user and group id specified in the request exist in the
   // passwd file. If they do then grant access. Very insecure: to be used
   // with care.

   int auth = 0;

   if (gDebug > 2)
      ErrorInfo("RpdRfioAuth: analyzing ... %s", sstr);

   if (!*sstr) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo("RpdRfioAuth: subject string is empty");
      return auth;
   }
   // Decode subject string
   unsigned int uid, gid;
   sscanf(sstr, "%u %u", &uid, &gid);

   // Now inquire passwd ...
   struct passwd *pw;
   if ((pw = getpwuid((uid_t) uid)) == 0) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo("RpdRfioAuth: uid %u not found", uid);
      return auth;
   }
   // Check if authorized
   char cuid[20];
   SPrintf(cuid, 20, "%u", uid);
   if (gUserIgnLen[5] > 0 && strstr(gUserIgnore[5], cuid) != 0) {
      NetSend(kErrNotAllowed, kROOTD_ERR);
      ErrorInfo
          ("RpdRfioAuth: user (%u,%s) not authorized to use (uid:gid) method",
           uid, pw->pw_name);
      return auth;
   }
   if (gUserAlwLen[5] > 0 && strstr(gUserAllow[5], cuid) == 0) {
      NetSend(kErrNotAllowed, kROOTD_ERR);
      ErrorInfo
          ("RpdRfioAuth: user (%u,%s) not authorized to use (uid:gid) method",
           uid, pw->pw_name);
      return auth;
   }

   // Now check group id ...
   if (gid != (unsigned int) pw->pw_gid) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo
          ("RpdRfioAuth: group id does not match (remote:%u,local:%u)",
           gid, (unsigned int) pw->pw_gid);
      return auth;
   }
   // Set username ....
   strlcpy(gUser, pw->pw_name, sizeof(gUser));


   // Notify, if required ...
   if (gDebug > 0)
      ErrorInfo("RpdRfioAuth: user %s authenticated (uid:%u, gid:%u)",
                gUser, uid, gid);

   // Set Auth flag
   auth = 1;
   gSec = 5;

   return auth;
}

//______________________________________________________________________________
void RpdAuthCleanup(const char *sstr, int opt)
{
   // Terminate correctly by cleaning up the auth table (and shared
   // memories in case of Globus) and closing the file.
   // Called upon receipt of a kROOTD_CLEANUP and on SIGPIPE.

   int rpid = 0, sec = -1, offs = -1, nw = 0;
   char usr[64] = {0};
   if (sstr)
      nw = sscanf(sstr, "%d %d %d %63s", &rpid, &sec, &offs, usr);

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
         // host specific cleanup
         RpdCleanupAuthTab(gOpenHost.c_str(), rpid, -1);
         ErrorInfo("RpdAuthCleanup: cleanup ('%s',%d) done",
                   gOpenHost.c_str(), rpid);
      } else if (nw == 4) {
         // (host,usr,method) specific cleanup
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

   // Size check done in RpdUpdateAuthTab(1,...)

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
      std::string temp;
      char cm[5];
      if (gNumAllow == 0)
         temp.append("none");
      for (i = 0; i < gNumAllow; i++) {
         SPrintf(cm, 5, " %3d",gAllowMeth[i]);
         temp.append(cm);
      }
      ErrorInfo
          ("RpdDefaultAuthAllow: default list of secure methods available: %s",
           temp.c_str());
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
   if (daemon == 0 || !daemon[0])
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
      servAddr.sin_zero[0] = 0;
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
      localAddr.sin_zero[0] = 0;
      localAddr.sin_family = AF_INET;
      localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
      localAddr.sin_port = htons(0);
      if (bind(sd, (struct sockaddr *) &localAddr, sizeof(localAddr)) < 0) {
         ErrorInfo("RpdCheckSshd: cannot bind to local port %u", gSshdPort);
         close(sd);
         return 0;
      }
      // connect to server
      if (connect(sd, (struct sockaddr *) &servAddr, sizeof(servAddr)) < 0) {
         ErrorInfo("RpdCheckSshd: cannot connect to local port %u",
                   gSshdPort);
         close(sd);
         return 0;
      }
      close(sd);
      // Sshd successfully contacted
      if (gDebug > 2)
         ErrorInfo("RpdCheckSshd: success!");
      rc = 1;
   }

   return rc;
}

//______________________________________________________________________________
int RpdUser(const char *sstr)
{
   // Check user id. If user id is not equal to rootd's effective uid, user
   // will not be allowed access, unless effective uid = 0 (i.e. root).
   const int kMaxBuf = 256;
   char recvbuf[kMaxBuf];
   EMessageTypes kind;
   struct passwd *pw;
   if (gDebug > 2)
      ErrorInfo("RpdUser: Enter ... %s", sstr);

   int auth = 0;

   // Nothing can be done if empty message
   if (!*sstr) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo("RpdUser: received empty string");
      return auth;
   }
   // Parse input message
   char user[kMAXUSERLEN] = {0};
   char ruser[kMAXUSERLEN] = {0};
   if (gClientProtocol > 8) {
      int ulen, ofs, opt, rulen;
      // Decode subject string
      int nw = sscanf(sstr, "%d %d %d %d %63s %d %63s",
                            &gRemPid, &ofs, &opt, &ulen, user, &rulen, ruser);
      ulen = (ulen >= kMAXUSERLEN) ? kMAXUSERLEN-1 : ulen;
      rulen = (rulen >= kMAXUSERLEN) ? kMAXUSERLEN-1 : rulen;
      user[ulen] = '\0';
      if (nw > 5)
         ruser[rulen] = '\0';
      gReUseRequired = (opt & kAUTH_REUSE_MSK);
      gCryptRequired = (opt & kAUTH_CRYPT_MSK);
      gSaltRequired  = (opt & kAUTH_SSALT_MSK);
      gOffSet = ofs;
#ifdef R__SSL
      if (gRSASSLKey) {
         // Determine type of RSA key required
         gRSAKey = (opt & kAUTH_RSATY_MSK) ? 2 : 1;
      } else
         gRSAKey = 1;
#else
      gRSAKey = 1;
#endif
   } else {
      SPrintf(user,kMAXUSERLEN,"%s",sstr);
   }
   if (gDebug > 2)
      ErrorInfo("RpdUser: gReUseRequired: %d gCryptRequired: %d gRSAKey: %d",
                gReUseRequired, gCryptRequired, gRSAKey);

   ERootdErrors err = kErrNoUser;
   if (gService == kROOTD) {
      // Default anonymous account ...
      if (!strcmp(user, "anonymous")) {
         user[0] = '\0';
         strlcpy(user, "rootd", sizeof(user));
      }
   }

   if ((pw = getpwnam(user)) == 0) {
      NetSend(err, kROOTD_ERR);
      ErrorInfo("RpdUser: user %s unknown", user);
      return auth;
   }

   // If server is not started as root and user is not same as the
   // one who started rootd then authetication is not ok.
   uid_t uid = getuid();
   if (uid && uid != pw->pw_uid) {
      NetSend(kErrBadUser, kROOTD_ERR);
      ErrorInfo("RpdUser: user not same as effective user of rootd");
      return auth;
   }

   // Check if the administrator allows authentication
   char cuid[20];
   SPrintf(cuid, 20, "%d", (int)pw->pw_uid);
   if (gUserIgnLen[0] > 0 && strstr(gUserIgnore[0], cuid) != 0) {
      NetSend(kErrNotAllowed, kROOTD_ERR);
      ErrorInfo
          ("RpdUser: user (%d,%s) not authorized to use UsrPwd method",
           uid, pw->pw_name);
      return auth;
   }
   if (gUserAlwLen[0] > 0 && strstr(gUserAllow[0], cuid) == 0) {
      NetSend(kErrNotAllowed, kROOTD_ERR);
      ErrorInfo
          ("RpdUser: user (%d,%s) not authorized to use UsrPwd method",
           uid, pw->pw_name);
      return auth;
   }

   // Check /etc/hosts.equiv and/or $HOME/.rhosts
   int errheq = 0;
   if (gCheckHostsEquiv && strlen(ruser)) {
      if (RpdCheckHostsEquiv(gOpenHost.c_str(),ruser,user,errheq)) {
         auth = 3;
         strlcpy(gUser, user, sizeof(gUser));
         return auth;
      }
   }

   // Check if of type anonymous ...
   if (!strcmp(pw->pw_shell, "/bin/false")) {
      err = kErrNoAnon;
      gAnon = 1;
      gReUseRequired = 0;
   }

   // Check if authorized
   // If not anonymous, try to get passwd
   // (if our system uses shadow passwds and we are not superuser
   // we cannot authenticate users ...)
   //   char *passw = 0;
   gPasswd[0] = 0;
   char *passw = gPasswd;
   int errrdp = 0;
   if (gAnon == 0) {

      // Check ROOT specific passwd first
      int rcsp = RpdRetrieveSpecialPass(user,gRootdPass.c_str(),
                                        gPasswd,sizeof(gPasswd));
      if (rcsp < 0)
         errrdp = (rcsp == -2) ? 3 : 0;

      if (!passw[0] || !strcmp(passw, "x")) {
#ifdef R__AFS
         gSaltRequired = 0;
#else

#ifdef R__SHADOWPW
         struct spwd *spw = 0;
         // System V Rel 4 style shadow passwords
         if ((spw = getspnam(user)) == 0) {
            if (gDebug > 0) {
               ErrorInfo("RpdUser: Shadow passwd not accessible for user %s",user);
               ErrorInfo("RpdUser: trying normal system passwd");
            }
         } else
            passw = spw->sp_pwdp;
#else
         passw = pw->pw_passwd;
#endif
         // Check if successful
         if (!passw[0] || !strcmp(passw, "x")) {
            if (gClientProtocol > 11)
               NetSend(gUsrPwdErr[errrdp][errheq], kROOTD_ERR);
            else
               NetSend(kErrNotAllowed, kROOTD_ERR);
            ErrorInfo("RpdUser: passwd hash not available for user %s", user);
            ErrorInfo
                ("RpdUser: user %s cannot be authenticated with this method",
                 user);
            return auth;
         }
#endif
      }
   }
   // Ok: Save username and go to next steps
   strlcpy(gUser, user, sizeof(gUser));

   // Salt vars
   char salt[30] = { 0 };
   char ctag[11] = { 0 };
   int  rtag = 0;
   int lenS = 0;

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
            NetSend(gRSAKey, kROOTD_RSAKEY);

            // Receive the key securely
            if (RpdRecvClientRSAKey()) {
               ErrorInfo("RpdUser: could not import a valid key -"
                         " switch off reuse for this session");
               gReUseRequired = 0;
            }

            // We get a random tag
            if (gClientProtocol > 11) {
               RpdInitRand();
               rtag = rpd_rand();
               SPrintf(ctag, 11, "#%08x#",rtag);
            }

            if (gSaltRequired) {
               // The crypt man page says that alternative salts can be in the form '$1$...$';
               // but on Ubuntu 10.04 the salt are in the form '$j$...$' where j is 6 or other.
               if (passw[0] == '$' && passw[2] == '$') {
                  // Shadow passwd
                  char *pd = strstr(passw + 4, "$");
                  lenS = (int) (pd - passw);
                  strncpy(salt, passw, lenS);
                  salt[lenS] = 0;
               } else {
                  lenS = 2;
                  strncpy(salt, passw, lenS);
                  salt[lenS] = 0;
               }
               if (gDebug > 2)
                  ErrorInfo("RpdUser: salt: '%s' ",salt);

               // We add the random tag here
               if (gClientProtocol > 11) {
                  strncpy(&salt[lenS],ctag,10);
                  salt[lenS+10] = 0;
               }

               // Send it over encrypted
               if (RpdSecureSend(salt) == -1) {
                  ErrorInfo("RpdUser: problems secure-sending salt -"
                            " may result in corrupted salt");
               }
            } else {
               if (gClientProtocol > 11) {
                  // We send the random tag here
                  if (RpdSecureSend(ctag) == -1) {
                     ErrorInfo("RpdUser: problems secure-sending rndmtag -"
                               " may result in corrupted rndmtag");
                  }
               } else
                  NetSend(0, kMESS_ANY);
            }
         } else {
            // We continue the authentication process in clear
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
      return auth;
   }
   if (kind != kROOTD_PASS) {
      NetSend(kErrFatal, kROOTD_ERR);
      ErrorInfo("RpdUser: received wrong message type: %d (expecting: %d)",
                kind, (int) kROOTD_PASS);
      return auth;
   }
   if (!strncmp(recvbuf,"-1",2)) {
      if (gDebug > 0)
         ErrorInfo("RpdUser: client did not send a password - return");
      return auth;
   }
   // Get passwd
   char *passwd = 0;
   int lpwd = 0;
   if (gAnon == 0 && gClientProtocol > 8 && gCryptRequired) {

      // Receive encrypted pass or its hash
      if (RpdSecureRecv(&passwd) == -1) {
         ErrorInfo
             ("RpdUser: problems secure-receiving pass hash - %s",
              "may result in authentication failure");
      }
      // Length of the password buffer
      lpwd = strlen(passwd);

      // Check the random tag, if any
      if (strlen(ctag)) {

         // Check first that there is enough space for the tag
         int plen = lpwd;
         if (plen > 9 &&
             passwd[plen-1] == '#' && passwd[plen-10] == '#') {
            if (strncmp(ctag,&passwd[plen-10],10)) {
               // The tag does not match; failure
               if (gClientProtocol > 11)
                  NetSend(gUsrPwdErr[2][errheq], kROOTD_ERR);
               else
                  NetSend(kErrBadPasswd, kROOTD_ERR);
               ErrorInfo("RpdUser: rndm tag mis-match"
                         " (%s vs %s) - Failure",&passwd[plen-10],ctag);
               delete[] passwd;
               return auth;
            }

            // Tag ok: drop it
            plen -= 10;
            passwd[plen] = 0;

         } else {
            // The tag is not there or incomplete; failure
            if (gClientProtocol > 11)
               NetSend(gUsrPwdErr[2][errheq], kROOTD_ERR);
            else
               NetSend(kErrBadPasswd, kROOTD_ERR);
            ErrorInfo("RpdUser: rndm tag missing or incomplete"
                      " (pw length: %d) - Failure", plen);
            delete[] passwd;
            return auth;
         }
      }

      // If we required an hash check that we got it
      // (the client sends the passwd if the crypt version is different)
      if (gSaltRequired && lenS) {
         if (strncmp(passwd,salt,lenS))
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
   auth = RpdPass(passwd,errheq);

   // Erase memory used for password
   passwd = (char *)rpdmemset((volatile void *)passwd,0,lpwd);
   delete[] passwd;

   return auth;
}

//______________________________________________________________________________
int RpdGuessClientProt(const char *buf, EMessageTypes kind)
{
   // Try a guess of the client protocol from what they sent over
   // the net ...

   if (gDebug > 2)
      ErrorInfo("RpdGuessClientProt: Enter: buf: '%s', kind: %d", buf,
                (int) kind);

   // Assume same version as us.
   int proto = 9;

   // Clear authentication
   if (kind == kROOTD_USER) {
      char usr[64], rest[256];
      int ns = sscanf(buf, "%63s %255s", usr, rest);
      if (ns == 1)
         proto = 8;
   }
   // SRP authentication
   if (kind == kROOTD_SRPUSER) {
      char usr[64], rest[256];
      int ns = sscanf(buf, "%63s %255s", usr, rest);
      if (ns == 1)
         proto = 8;
   }
   // Kerberos authentication
   if (kind == kROOTD_KRB5) {
      if (!buf[0])
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

   unsigned int iimx[4][4] = {
      { 0x0, 0xffffff08, 0xafffffff, 0x2ffffffe }, // Opt = 0
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
   char *buf = new char[Len + 1];

   // Init Random machinery ...
   if (!gRandInit)
      RpdInitRand();

   // randomize
   int k = 0;
   int i, j, l, m, frnd;
   while (k < Len) {
      frnd = rpd_rand();
      for (m = 7; m < 32; m += 7) {
         i = 0x7F & (frnd >> m);
         j = i / 32;
         l = i - j * 32;
         if ((iimx[Opt][j] & (1 << l))) {
            buf[k] = i;
            k++;
         }
         if (k == Len)
            break;
      }
   }

   // NULL terminated
   buf[Len] = 0;
   if (gDebug > 2)
      ErrorInfo("RpdGetRandString: got '%s' ", buf);

   return buf;
}

//______________________________________________________________________________
int RpdGetRSAKeys(const char *pubkey, int Opt)
{
   // Get public key from file pubkey (Opt == 1) or string pubkey (Opt == 0).

   char str[kMAXPATHLEN] = { 0 };
   int keytype = 0;

   if (gDebug > 2)
      ErrorInfo("RpdGetRSAKeys: enter: string len: %d, opt %d ",
                 gPubKeyLen, Opt);

   if (!pubkey)
      return keytype;

   char *theKey = 0;
   FILE *fKey = 0;
   // Parse input type
   if (Opt == 1) {

      // Ok, now open it
      fKey = fopen(pubkey, "r");
      if (!fKey) {
         if (GetErrno() == EACCES) {
            struct passwd *pw = getpwuid(getuid());
            char *usr = 0;
            if (pw)
               usr = pw->pw_name;
            ErrorInfo("RpdGetRSAKeys: access to key file %s denied"
                      " to user: %s", pubkey, (usr ? usr : (char *)"????"));
         } else
            ErrorInfo("RpdGetRSAKeys: cannot open key file"
                      " %s (errno: %d)", pubkey, GetErrno());
         return 0;
      }
      // Check first the permissions: should be 0600
      struct stat st;
      if (fstat(fileno(fKey), &st) == -1) {
         ErrorInfo("RpdGetRSAKeys: cannot stat descriptor %d"
                   " %s (errno: %d)", fileno(fKey), GetErrno());
         fclose(fKey);
         return 0;
      }
      if (!S_ISREG(st.st_mode) || S_ISDIR(st.st_mode) ||
          (st.st_mode & 0777) != (S_IRUSR | S_IWUSR)) {
         ErrorInfo("RpdGetRSAKeys: key file %s: wrong permissions"
                   " 0%o (should be 0600)", pubkey, (st.st_mode & 0777));
         fclose(fKey);
         return 0;
      }
      gPubKeyLen = fread((void *)str,1,sizeof(str),fKey);
      if (gDebug > 2)
         ErrorInfo("RpdGetRSAKeys: length of the read key: %d",gPubKeyLen);

      // This the key
      theKey = str;
   } else {
      // the key is the argument
      theKey = (char *)pubkey;
   }

   if (gPubKeyLen > 0) {

      // Skip spaces at beginning, if any
      int k = 0;
      while (theKey[k] == 32) k++;

      keytype = gRSAKey;

      // The format of keytype 1 is #<hex_n>#<hex_d>#
      char *pd1 = 0, *pd2 = 0, *pd3 = 0;
      pd1 = strstr(theKey, "#");
      if (pd1) pd2 = strstr(pd1 + 1, "#");
      if (pd2) pd3 = strstr(pd2 + 1, "#");
      if (keytype == 1) {
         if (!pd1 || !pd2 || !pd3) {
            if (gDebug > 0)
               ErrorInfo("RpdGetRSAKeys: bad format for keytype %d"
                         " - exit", keytype);
            keytype = 0;
         }
      }
      if (keytype == 1) {

         if (gDebug > 2)
            ErrorInfo("RpdGetRSAKeys: keytype %d ", keytype);

         // Get <hex_n> ...
         int l1 = (int) (pd2 - pd1 - 1);
         char *n_exp_RSA = new char[l1 + 1];
         strncpy(n_exp_RSA, pd1 + 1, l1);
         n_exp_RSA[l1] = 0;
         if (gDebug > 2)
            ErrorInfo("RpdGetRSAKeys: got %d bytes for n_exp_RSA",
                      strlen(n_exp_RSA));
         // Now <hex_d>
         int l2 = (int) (pd3 - pd2 - 1);
         char *d_exp_RSA = new char[l2 + 1];
         strncpy(d_exp_RSA, pd2 + 1, l2);
         d_exp_RSA[l2] = 0;
         if (gDebug > 2)
            ErrorInfo("RpdGetRSAKeys: got %d bytes for d_exp_RSA",
                      strlen(d_exp_RSA));

         rsa_num_sget(&gRSA_n, n_exp_RSA);
         rsa_num_sget(&gRSA_d, d_exp_RSA);

         delete[] n_exp_RSA;
         delete[] d_exp_RSA;

      } else if (keytype == 2){

#ifdef R__SSL
         // try SSL
         if (gDebug > 2)
            ErrorInfo("RpdGetRSAKeys: keytype %d ", keytype);

         // Now set the key locally in BF form
         BF_set_key(&gBFKey, gPubKeyLen, (const unsigned char *)theKey);
#else
         if (gDebug > 0) {
            ErrorInfo("RpdGetRSAKeys: not compiled with SSL support:"
                   " you should not have got here!");
         }
#endif
      }
   }

   if (fKey)
      fclose(fKey);

   return keytype;
}

//______________________________________________________________________________
int RpdSavePubKey(const char *PubKey, int OffSet, char *user)
{
   // Save RSA public key into file for later use by other rootd/proofd.
   // Return: 0 if ok
   //         1 if not ok
   //         2 if not ok because file already exists and cannot be
   //           overwritten

   int retval = 0;

   if (gRSAKey == 0 || OffSet < 0)
      return 1;

   std::string pukfile = gRpdKeyRoot;
   pukfile.append(ItoA(OffSet));

   // Unlink the file first
   if (unlink(pukfile.c_str()) == -1) {
      if (GetErrno() != ENOENT)
         // File exists and cannot overwritten by this process
         return 2;
   }

   // Create file
   int ipuk = -1;
   ipuk = open(pukfile.c_str(), O_WRONLY | O_CREAT, 0600);
   if (ipuk == -1) {
      ErrorInfo("RpdSavePubKey: cannot open file %s (errno: %d)",
                pukfile.c_str(),GetErrno());
      if (GetErrno() == ENOENT)
         return 2;
      else
         return 1;
   }

   // If root process set ownership of the pub key to the user
   if (getuid() == 0) {
      struct passwd *pw = getpwnam(user);
      if (pw) {
         if (fchown(ipuk,pw->pw_uid,pw->pw_gid) == -1) {
            ErrorInfo("RpdSavePubKey: cannot change ownership"
                      " of %s (errno: %d)",pukfile.c_str(),GetErrno());
            retval = 1;
         }
      } else {
         ErrorInfo("RpdSavePubKey: getpwnam failure (errno: %d)",GetErrno());
         retval = 1;
      }
   }

   // Write the key if no error occured
   if (retval == 0) {
      while (write(ipuk, PubKey, gPubKeyLen) < 0 && GetErrno() == EINTR)
         ResetErrno();
   }

   // close the file
   close(ipuk);

   // Over
   return retval;
}

//______________________________________________________________________________
int RpdSecureSend(char *str)
{
   // Encode null terminated str using the session private key indcated by Key
   // and sends it over the network.
   // Returns number of bytes sent.or -1 in case of error.

   char buftmp[kMAXSECBUF];
   char buflen[20];

   int slen = strlen(str) + 1;

   int ttmp = 0;
   int nsen = -1;

   if (gRSAKey == 1) {
      strncpy(buftmp, str, slen);
      buftmp[slen] = 0;
      ttmp = rsa_encode(buftmp, slen, gRSA_n, gRSA_d);
   } else if (gRSAKey == 2) {
#ifdef R__SSL
      ttmp = strlen(str);
      if ((ttmp % 8) > 0)            // It should be a multiple of 8!
         ttmp = ((ttmp + 8)/8) * 8;
      unsigned char iv[8];
      memset((void *)&iv[0],0,8);
      BF_cbc_encrypt((const unsigned char *)str, (unsigned char *)buftmp,
                     strlen(str), &gBFKey, iv, BF_ENCRYPT);
#else
      ErrorInfo("RpdSecureSend: Not compiled with SSL support:"
                " you should not have got here! - return");
#endif
   } else {
      ErrorInfo("RpdSecureSend: Unknown key option (%d) - return",
                gRSAKey);
   }

   // Send the buffer now
   SPrintf(buflen, 20, "%d", ttmp);
   NetSend(buflen, kROOTD_ENCRYPT);
   nsen = NetSendRaw(buftmp, ttmp);
   if (gDebug > 4)
      ErrorInfo("RpdSecureSend: sent %d bytes (expected: %d) - keytype: %d",
                 nsen, ttmp, gRSAKey);

   return nsen;
}

//______________________________________________________________________________
int RpdSecureRecv(char **str)
{
   // Receive buffer and decode it in str using key indicated by Key type.
   // Return number of received bytes or -1 in case of error.

   char buftmp[kMAXSECBUF];
   char buflen[20];

   int nrec = -1;
   // We must get a pointer ...
   if (!str)
      return nrec;

   if (gDebug > 2)
      ErrorInfo("RpdSecureRecv: enter ... (key is %d)", gRSAKey);

   EMessageTypes kind;
   NetRecv(buflen, 20, kind);
   int len = atoi(buflen);
   if (gDebug > 4)
      ErrorInfo("RpdSecureRecv: got len '%s' %d ", buflen, len);
   if (!strncmp(buflen, "-1", 2))
      return nrec;

   // receive the buffer
   nrec = NetRecvRaw(buftmp,len);

   // decode it
   if (gRSAKey == 1) {
      rsa_decode(buftmp, len, gRSA_n, gRSA_d);
      if (gDebug > 2)
         ErrorInfo("RpdSecureRecv: Local: decoded string is %d bytes long",
                   strlen(buftmp));

      // Prepare output
      const size_t strSize = strlen(buftmp) + 1;
      *str = new char[strSize];
      strlcpy(*str, buftmp, strSize);
   } else if (gRSAKey == 2) {
#ifdef R__SSL
      unsigned char iv[8];
      memset((void *)&iv[0],0,8);
      *str = new char[nrec + 1];
      BF_cbc_encrypt((const unsigned char *)buftmp, (unsigned char *)(*str),
                      nrec, &gBFKey, iv, BF_DECRYPT);
      (*str)[nrec] = '\0';
#else
      ErrorInfo("RpdSecureRecv: Not compiled with SSL support:"
                " you should not have got here! - return");
#endif
   } else {
      ErrorInfo("RpdSecureRecv: Unknown key option (%d) - return",
                gRSAKey);
   }

   return nrec;
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

#ifdef R__NOCRYPT
   // Generate a random salt
   char *rsalt = RpdGetRandString(3,8);
   if (rsalt) {
      gRndmSalt = std::string(rsalt);
      delete[] rsalt;
   } else {
      if (gDebug > 0)
         ErrorInfo("RpdGenRSAKeys: could not generate random salt");
   }
#endif

#ifdef R__SSL
   // Generate also the SSL key
   if (gDebug > 2)
      ErrorInfo("RpdGenRSAKeys: Generate RSA SSL keys");

   // Init SSL ...
   SSL_library_init();

   //  ... and its error strings
   SSL_load_error_strings();

   // Load Ciphers
   OpenSSL_add_all_ciphers();

   // Number of bits for key
   Int_t nbits = 1024;

   // Public exponent
   Int_t pubex = 17;

   // Init random engine
   char *rbuf = RpdGetRandString(0,40);
   RAND_seed(rbuf,strlen(rbuf));

   // Generate Key
   gRSASSLKey = RSA_generate_key(nbits,pubex,0,0);

   // Bio for exporting the pub key
   BIO *bkey = BIO_new(BIO_s_mem());

   // Write public key to BIO
   PEM_write_bio_RSAPublicKey(bkey,gRSASSLKey);

   // Read key from BIO to buf
   Int_t sbuf = 2*RSA_size(gRSASSLKey);
   char *kbuf = new char[sbuf];
   BIO_read(bkey,(void *)kbuf,sbuf);
   BIO_free(bkey);

   // Prepare export
   gRSAPubExport[1].len = sbuf;
   gRSAPubExport[1].keys = new char[gRSAPubExport[1].len + 2];
   strncpy(gRSAPubExport[1].keys,kbuf,gRSAPubExport[1].len);
   gRSAPubExport[1].keys[gRSAPubExport[1].len-1] = '\0';
   delete[] kbuf;
   if (gDebug > 2)
      ErrorInfo("RpdGenRSAKeys: SSL: export pub:\n%.*s",
           gRSAPubExport[1].len,gRSAPubExport[1].keys);

   // We have at least one key
   gRSAInit = 1;

#endif

   // Sometimes some bunch is not decrypted correctly
   // That's why we make retries to make sure that encryption/decryption
   // works as expected
   bool notOK = 1;
   rsa_NUMBER p1, p2, rsa_n, rsa_e, rsa_d;
   int l_n = 0, l_d = 0;
#if R__RSADEB
   Int_t l_e = 0;
   char buf[rsa_STRLEN];
#endif
   char buf_n[rsa_STRLEN], buf_e[rsa_STRLEN], buf_d[rsa_STRLEN];

   int nAttempts = 0;
   int thePrimeLen = kPRIMELENGTH;
   int thePrimeExp = kPRIMEEXP + 5;   // Prime probability = 1-0.5^thePrimeExp
   while (notOK && nAttempts < kMAXRSATRIES) {

      nAttempts++;
      if (gDebug > 2 && nAttempts > 1) {
            ErrorInfo("RpdGenRSAKeys: retry no. %d",nAttempts);
         srand(rpd_rand());
      }

      // Valid pair of primes
      p1 = rsa_genprim(thePrimeLen, thePrimeExp);
      p2 = rsa_genprim(thePrimeLen+1, thePrimeExp);

      // Retry if equal
      int nPrimes = 0;
      while (rsa_cmp(&p1, &p2) == 0 && nPrimes < kMAXRSATRIES) {
         nPrimes++;
         if (gDebug > 2)
            ErrorInfo("RpdGenRSAKeys: equal primes: regenerate (%d times)",nPrimes);
         srand(rpd_rand());
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
         if (gDebug > 0)
            ErrorInfo("RpdGenRSAKeys: genrsa: attempt %d to generate"
                      " keys failed",nAttempts);
         continue;
      }

      // Determine their lengths
      rsa_num_sput(&rsa_n, buf_n, rsa_STRLEN);
      l_n = strlen(buf_n);
      rsa_num_sput(&rsa_e, buf_e, rsa_STRLEN);
#if R__RSADEB
      l_e = strlen(buf_e);
#endif
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
      char test[2 * rsa_STRLEN] = "ThisIsTheStringTest01203456-+/";
      Int_t lTes = 31;
      char *dumT = RpdGetRandString(0, lTes - 1);
      strncpy(test, dumT, lTes);
      delete[]dumT;
      char buf[2 * rsa_STRLEN];
      if (gDebug > 3)
         ErrorInfo("RpdGenRSAKeys: local: test string: '%s' ", test);

      // Private/Public
      strncpy(buf, test, lTes);
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

      if (strncmp(test, buf, lTes))
         continue;

      // Public/Private
      strncpy(buf, test, lTes);
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

      if (strncmp(test, buf, lTes))
         continue;

      notOK = 0;
   }

   if (notOK) {
      ErrorInfo("RpdGenRSAKeys: unable to generate good RSA key pair"
                " (%d attempts)- return",kMAXRSATRIES);
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
   gRSAPubExport[0].len = l_n + l_d + 4;
   if (gRSAPubExport[0].keys)
      delete[] gRSAPubExport[0].keys;
   gRSAPubExport[0].keys = new char[gRSAPubExport[0].len];

   gRSAPubExport[0].keys[0] = '#';
   memcpy(gRSAPubExport[0].keys + 1, buf_n, l_n);
   gRSAPubExport[0].keys[l_n + 1] = '#';
   memcpy(gRSAPubExport[0].keys + l_n + 2, buf_d, l_d);
   gRSAPubExport[0].keys[l_n + l_d + 2] = '#';
   gRSAPubExport[0].keys[l_n + l_d + 3] = 0;
#if R__RSADEB
   if (gDebug > 2)
      ErrorInfo("RpdGenRSAKeys: local: export pub: '%s'",
                gRSAPubExport[0].keys);
#else
   if (gDebug > 2)
      ErrorInfo("RpdGenRSAKeys: local: export pub length: %d bytes",
                gRSAPubExport[0].len);
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
   int key = gRSAKey - 1;
   NetSend(gRSAPubExport[key].keys, gRSAPubExport[key].len, kROOTD_RSAKEY);

   // Receive length of message with encode client public key
   EMessageTypes kind;
   char buflen[40];
   NetRecv(buflen, 20, kind);
   gPubKeyLen = atoi(buflen);
   if (gDebug > 3)
      ErrorInfo("RpdRecvClientRSAKey: got len '%s' %d ", buflen, gPubKeyLen);

   int nrec = 0;

   if (gRSAKey == 1) {

      // Receive and decode encoded public key
      nrec = NetRecvRaw(gPubKey, gPubKeyLen);

      rsa_decode(gPubKey, gPubKeyLen, gRSAPriKey.n, gRSAPriKey.e);
      if (gDebug > 2)
         ErrorInfo("RpdRecvClientRSAKey: Local: decoded string is %d bytes long ",
            strlen(gPubKey));
      gPubKeyLen = strlen(gPubKey);

   } else if (gRSAKey == 2) {
#ifdef R__SSL
      int ndec = 0;
      int lcmax = RSA_size(gRSASSLKey);
      char btmp[kMAXSECBUF];
      int nr = gPubKeyLen;
      int kd = 0;
      while (nr > 0) {
         // Receive and decode encoded public key
         nrec += NetRecvRaw(btmp, lcmax);
         if ((ndec = RSA_private_decrypt(lcmax,(unsigned char *)btmp,
                                    (unsigned char *)&gPubKey[kd],
                                    gRSASSLKey,
                                    RSA_PKCS1_PADDING)) < 0) {
            char errstr[120];
            ERR_error_string(ERR_get_error(), errstr);
            ErrorInfo("RpdRecvClientRSAKey: SSL: error: '%s' ",errstr);
         }
         nr -= lcmax;
         kd += ndec;
      }
      gPubKeyLen = kd;
#else
      if (gDebug > 0)
         ErrorInfo("RpdRecvClientRSAKey: not compiled with SSL support"
                   ": you should not have got here!");
      return 1;
#endif
   } else {
      if (gDebug > 0)
         ErrorInfo("RpdRecvClientRSAKey: unknown key type (%d)", gRSAKey);
   }


   // Import Key and Determine key type
   if (RpdGetRSAKeys(gPubKey, 0) != gRSAKey) {
      ErrorInfo("RpdRecvClientRSAKey:"
                " could not import a valid key (type %d)",gRSAKey);
      char *elogfile = new char[gRpdKeyRoot.length() + 11];
      SPrintf(elogfile, gRpdKeyRoot.length() + 11, "%.*serr.XXXXXX", (int)gRpdKeyRoot.length(), gRpdKeyRoot.c_str());
      mode_t oldumask = umask(0700);
      int ielog = mkstemp(elogfile);
      umask(oldumask);
      if (ielog != -1) {
         char line[kMAXPATHLEN] = {0};
         //
         SPrintf(line,kMAXPATHLEN,
                 " + RpdRecvClientRSAKey: error importing key\n + type: %d\n"
                 " + length: %d\n + key: %s\n + (%d bytes were received)",
                 gRSAKey, gPubKeyLen, gPubKey, nrec);
         while (write(ielog, line, strlen(line)) < 0 && GetErrno() == EINTR)
            ResetErrno();
         close (ielog);
      }
      delete [] elogfile;
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
      if (read(fd, &seed, sizeof(seed))) {;}
      close(fd);
   } else {
      if (gDebug > 2)
         ErrorInfo("RpdInitRand: %s not available: using time()", randdev);
      seed = time(0);   //better use times() + win32 equivalent
   }
   srand(seed);
}

//______________________________________________________________________________
int RpdAuthenticate()
{
   // Handle user authentication.
   char buf[kMAXRECVBUF];
   EMessageTypes kind;

//#define R__DEBUG
#ifdef R__DEBUG
   int debug = 1;
   while (debug)
      ;
#endif

   // Reset gAuth (if we have been called this means that we need
   // to check at least that a valid authentication exists ...)
   int auth = 0;

   while (!auth) {

      // Receive next
      if (!gClientOld) {
         if (NetRecv(buf, kMAXRECVBUF, kind) < 0) {
            Error(gErr, -1, "RpdAuthenticate: error receiving message");
            return auth;
         }
      } else {
         strlcpy(buf,gBufOld, sizeof(buf));
         kind = gKindOld;
         gBufOld[0] = '\0';
         gClientOld = 0;
      }

      // If this is a rootd contacted via a TXNetFile we need to
      // receive again the buffer
      if (gService == kROOTD && kind == kROOTD_PROTOCOL) {
         if (NetRecv(buf, kMAXRECVBUF, kind) < 0) {
            Error(gErr, -1, "RpdAuthenticate: error receiving message");
            return auth;
         }
      }

      // Decode the method ...
      gAuthProtocol = RpdGetAuthMethod(kind);

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
      if (gAuthProtocol != -1 && gClientProtocol > 8) {

         // Check if accepted ...
         if (RpdCheckAuthAllow(gAuthProtocol, gOpenHost.c_str())) {
            if (gNumAllow>0) {
               if (gAuthListSent == 0) {
                  if (gDebug > 0)
                     ErrorInfo("Authenticate: %s method not"
                               " accepted from host: %s",
                                gAuthMeth[gAuthProtocol].c_str(),
                                gOpenHost.c_str());
                  NetSend(kErrNotAllowed, kROOTD_ERR);
                  RpdSendAuthList();
                  gAuthListSent = 1;
                  goto next;
               } else {
                  Error(gErr,kErrNotAllowed,"Authenticate: method not"
                       " in the list sent to the client");
                  return auth;
               }
            } else {
               Error(gErr,kErrConnectionRefused,"Authenticate:"
                       " connection refused from host %s", gOpenHost.c_str());
               return auth;
            }
         }

         // Then check if a previous authentication exists and is valid
         // ReUse does not apply for RFIO
         if (kind != kROOTD_RFIO && (auth = RpdReUseAuth(buf, kind)))
            goto next;
      }

      // Reset global variable
      auth = 0;

      switch (kind) {
         case kROOTD_USER:
            auth = RpdUser(buf);
            break;
         case kROOTD_SRPUSER:
            auth = RpdSRPUser(buf);
            break;
         case kROOTD_PASS:
            auth = RpdPass(buf);
            break;
         case kROOTD_KRB5:
            auth = RpdKrb5Auth(buf);
            break;
         case kROOTD_GLOBUS:
            auth = RpdGlobusAuth(buf);
            break;
         case kROOTD_SSH:
            auth = RpdSshAuth(buf);
            break;
         case kROOTD_RFIO:
            auth = RpdRfioAuth(buf);
            break;
         case kROOTD_CLEANUP:
            RpdAuthCleanup(buf,1);
            ErrorInfo("RpdAuthenticate: authentication stuff cleaned - exit");
            // Fallthrough next case now to free the keys
         case kROOTD_BYE:
            RpdFreeKeys();
            return auth;
            break;
         default:
            Error(gErr,-1,"RpdAuthenticate: received bad opcode %d", kind);
            return auth;
      }

      if (gClientProtocol > 8) {

         // If failure prepare or continue negotiation
         // Don't do this if this was a SSH notification failure
         // because in such a case it was already done in the
         // appropriate daemon child
         int doneg = (gAuthProtocol != -1 || kind == kROOTD_PASS) &&
                     (gRemPid > 0 || kind != kROOTD_SSH);
         if (gDebug > 2 && doneg)
            ErrorInfo("RpdAuthenticate: kind:%d meth:%d auth:%d gNumLeft:%d",
                      kind, gAuthProtocol, auth, gNumLeft);

         // If authentication failure, check if other methods could be tried ...
         if (auth == 0 && doneg) {
            if (gNumLeft > 0) {
               if (gAuthListSent == 0) {
                  RpdSendAuthList();
                  gAuthListSent = 1;
               } else
                  NetSend(-1, kROOTD_NEGOTIA);
            } else {
               NetSend(0, kROOTD_NEGOTIA);
               Error(gErr, -1, "RpdAuthenticate: authentication failed");
               return auth;
            }
         }
      }
next:
      continue;
   }

   return auth;
}
//______________________________________________________________________________
void RpdFreeKeys()
{
   // Free space allocated for encryption keys

   if (gRSAPubExport[0].keys) delete[] gRSAPubExport[0].keys;
   if (gRSAPubExport[1].keys) delete[] gRSAPubExport[1].keys;
#ifdef R__SSL
   RSA_free(gRSASSLKey);
#endif
}

//______________________________________________________________________________
int RpdProtocol(int ServType)
{
   // Receives client protocol and returns daemon protocol.
   // Returns:  0 if ok
   //          -1 if any error occured
   //          -2 if special action (e.g. cleanup): no need to continue

   int rc = 0;

//#define R__DEBUG
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
   int lbuf[2];
   if (NetRecvRaw(lbuf, sizeof(lbuf)) < 0) {
      NetSend(kErrFatal, kROOTD_ERR);
      ErrorInfo("RpdProtocol: error receiving message");
      return -1;
   }

   // if kind is {kROOTD_PROTOCOL, kROOTD_CLEANUP, kROOTD_SSH}
   // receive the rest
   kind = (EMessageTypes) ntohl(lbuf[1]);
   int len = ntohl(lbuf[0]);
   if (gDebug > 1)
      ErrorInfo("RpdProtocol: kind: %d %d",kind,len);
   if (kind == kROOTD_PROTOCOL || kind == kROOTD_CLEANUP ||
       kind == kROOTD_SSH) {
      // Receive the rest
      char *buf = 0;
      len -= sizeof(int);
      if (gDebug > 1)
         ErrorInfo("RpdProtocol: len: %d",len);
      if (len) {
         buf = new char[len];
         if (NetRecvRaw(buf, len) < 0) {
            NetSend(kErrFatal, kROOTD_ERR);
            ErrorInfo("RpdProtocol: error receiving message");
            delete[] buf;
            return -1;
         }
         strlcpy(proto, buf, sizeof(proto));
      } else {
         // Empty buffer
         proto[0] = '\0';
      }
      if (gDebug > 1)
         ErrorInfo("RpdProtocol: proto buff: %s", buf ? buf : "---");
      // Copy buffer for later use
      readbuf = 0;
      if (buf) delete[] buf;
   } else if (ServType == kROOTD && kind == 0 && len == 0) {
      // TNetFile via TXNetFile: receive client protocol
      // read first next 12 bytes and discard them
      int llen = 12;
      char *buf = new char[llen];
      if (NetRecvRaw(buf, llen) < 0) {
         NetSend(kErrFatal, kROOTD_ERR);
         ErrorInfo("RpdProtocol: error receiving message");
         if (buf) delete[] buf;
         return -1;
      }
      if (buf) delete[] buf;
      // Send back the 'type'
      int type = htonl(8);
      if (NetSendRaw(&type,sizeof(type)) < 0) {
         NetSend(kErrFatal, kROOTD_ERR);
         ErrorInfo("RpdProtocol: error sending type to TXNetFile");
         return -1;
      }
      // Now read the client protocol
      llen = 4;
      buf = new char[llen];
      if (NetRecvRaw(buf,llen) < 0) {
         NetSend(kErrFatal, kROOTD_ERR);
         ErrorInfo("RpdProtocol: error receiving message");
         delete[] buf;
         return -1;
      }
      strlcpy(proto,buf, sizeof(proto));
      kind = kROOTD_PROTOCOL;
      readbuf = 0;
      delete[] buf;
   } else {
      // Need to open parallel sockets first
      int size = ntohl(lbuf[1]);
      // Read port
      int port;
      if (NetRecvRaw(&port, sizeof(int)) < 0) {
         NetSend(kErrFatal, kROOTD_ERR);
         ErrorInfo("RpdProtocol: error receiving message");
         return -1;
      }
      port = ntohl(port);
      if (gDebug > 0)
         ErrorInfo("RpdProtocol: port = %d, size = %d", port, size);
      if (size > 1)
         NetParOpen(port, size);
   }

   int done = 0;
   gClientOld = 0;
   while (!done) {

      // Receive next
      if (readbuf) {
         if (NetRecv(proto, kMAXRECVBUF, kind) < 0) {
            ErrorInfo("RpdProtocol: error receiving message");
            return -1;
         }
      }
      readbuf = 1;

      switch(kind) {

         case kROOTD_CLEANUP:
            RpdAuthCleanup(proto,1);
            ErrorInfo("RpdProtocol: authentication stuff cleaned");
            done = 1;
            rc = -2;
            break;
         case kROOTD_BYE:
            RpdFreeKeys();
            NetClose();
            done = 1;
            rc = -2;
            break;
         case kROOTD_PROTOCOL:

            if (strlen(proto) > 0) {
               gClientProtocol = atoi(proto);
            } else {
               if (ServType == kROOTD) {
                  // This is an old (TNetFile,TFTP) client:
                  // send our protocol first ...
                  if (NetSend(gServerProtocol, kROOTD_PROTOCOL) < 0) {
                     ErrorInfo("RpdProtocol: error sending kROOTD_PROTOCOL");
                     rc = -1;
                  }
                  // ... and receive protocol via kROOTD_PROTOCOL2
                  if (NetRecv(proto, kMAXRECVBUF, kind) < 0) {
                     ErrorInfo("RpdProtocol: error receiving message");
                     rc = -1;
                  }
                  if (kind != kROOTD_PROTOCOL2) {
                     strlcpy(gBufOld, proto, sizeof(gBufOld));
                     gKindOld = kind;
                     gClientOld = 1;
                     gClientProtocol = 0;
                  } else
                     gClientProtocol = atoi(proto);
               } else
                  gClientProtocol = 0;
            }
            if (!gClientOld) {
               // send our protocol
               // if we do not require authentication say it here
               Int_t protoanswer = gServerProtocol;
               if (!gRequireAuth && gClientProtocol > 10)
                  protoanswer += 1000;
               // Notify
               if (gDebug > 0) {
                  ErrorInfo("RpdProtocol: gClientProtocol = %d",
                            gClientProtocol);
                  ErrorInfo("RpdProtocol: Sending gServerProtocol = %d",
                            protoanswer);
               }
               if (NetSend(protoanswer, kROOTD_PROTOCOL) < 0) {
                  ErrorInfo("RpdProtocol: error sending kROOTD_PROTOCOL");
                  rc = -1;
               }
            }
            done = 1;
            break;
         case kROOTD_SSH:
            // Failure notification ...
            RpdSshAuth(proto);
            NetSend(kErrAuthNotOK, kROOTD_ERR);
            ErrorInfo("RpdProtocol: SSH failure notified");
            rc = -2;
            done = 1;
            break;
         default:
            ErrorInfo("RpdProtocol: received bad option (%d)",kind);
            rc = -1;
            done = 1;
            break;
      } // Switch

   } // done

   return rc;
}

//______________________________________________________________________________
int RpdLogin(int ServType, int auth)
{
   // Authentication was successful, set user environment.

//   if (gDebug > 2)
      ErrorInfo("RpdLogin: enter: Server: %d, gUser: %s, auth: %d",
                ServType, gUser, auth);

   // Login only if requested
   if (gDoLogin == 0)
      return -2;

   struct passwd *pw = getpwnam(gUser);

   if (!pw) {
      ErrorInfo("RpdLogin: user %s does not exist locally\n", gUser);
      return -1;
   }

   if (getuid() == 0) {

#ifdef R__GLBS
      if (ServType == 2) {
         // We need to change the ownership of the shared memory segments used
         // for credential export to allow proofserv to destroy them
         struct shmid_ds shm_ds;
         if (gShmIdCred > 0) {
            if (shmctl(gShmIdCred, IPC_STAT, &shm_ds) == -1) {
               ErrorInfo("RpdLogin: can't get info about shared memory"
                         " segment %d (errno: %d)",gShmIdCred,GetErrno());
               return -1;
            }
            shm_ds.shm_perm.uid = pw->pw_uid;
            shm_ds.shm_perm.gid = pw->pw_gid;
            if (shmctl(gShmIdCred, IPC_SET, &shm_ds) == -1) {
               ErrorInfo("RpdLogin: can't change ownership of shared"
                         " memory segment %d (errno: %d)",
                         gShmIdCred,GetErrno());
               return -1;
            }
         }
      }
#endif
      //
      // Anonymous users are confined to their corner
      if (gAnon) {
         // We need to do it before chroot, otherwise it does not work
         if (chdir(pw->pw_dir) == -1) {
            ErrorInfo("RpdLogin: can't change directory to %s (errno: %d)",
                      pw->pw_dir, errno);
            return -1;
         }
         if (chroot(pw->pw_dir) == -1) {
            ErrorInfo("RpdLogin: can't chroot to %s", pw->pw_dir);
            return -1;
         }
      }

      // set access control list from /etc/initgroup
      initgroups(gUser, pw->pw_gid);

      // set uid and gid
      if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1) {
         ErrorInfo("RpdLogin: can't setgid for user %s", gUser);
         return -1;
      }
      if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1) {
         ErrorInfo("RpdLogin: can't setuid for user %s", gUser);
         return -1;
      }
   }

   if (ServType == 2) {
      // set HOME env
      char *home = new char[8+strlen(pw->pw_dir)];
      SPrintf(home, 8+strlen(pw->pw_dir), "HOME=%s", pw->pw_dir);
      putenv(home);
   }

   // Change user's HOME, if required (for anon it is already done)
   if (gDoLogin == 2 && !gAnon) {
      if (chdir(pw->pw_dir) == -1) {
         ErrorInfo("RpdLogin: can't change directory to %s (errno: %d)",
                   pw->pw_dir, errno);
         return -1;
      }
   }

   umask(022);

   // Notify authentication to client ...
   NetSend(auth, kROOTD_AUTH);
   // Send also new offset if it changed ...
   if (auth == 2) NetSend(gOffSet, kROOTD_AUTH);

   if (gDebug > 0)
      ErrorInfo("RpdLogin: user %s logged in", gUser);

   return 0;
}

//______________________________________________________________________________
int RpdInitSession(int servtype, std::string &user,
                   int &cproto, int &meth, int &type, std::string &ctoken)
{
   // Perform the action needed to commence the new session:
   // Version called by TServerSocket.
   //   - set debug flag
   //   - check authentication table
   //   - Inquire protocol
   //   - authenticate the client
   // Returns logged-in user, the remote client procotol cproto,
   // the authentication protocol (ROOT internal) number is returned
   // in meth, type indicates the kind of authentication:
   //       0 = new authentication
   //       1 = existing authentication
   //       2 = existing authentication with updated offset
   // and the crypted-token in ctoken (used later for cleaning).
   // Called just after opening the connection

   std::string pwd;
   int auth = RpdInitSession(servtype,user,cproto,meth,pwd);
   if (auth == 1)
      if (gExistingAuth)
         type = 1;
      else
         type = 0;
   else if (auth == 2)
      type = 2;
   ctoken = gCryptToken;

   return auth;
}
//______________________________________________________________________________
int RpdInitSession(int servtype, std::string &user,
                   int &cproto, int &anon, std::string &passwd)
{
   // Perform the action needed to commence the new session:
   //   - set debug flag
   //   - check authentication table
   //   - Inquire protocol
   //   - authenticate the client
   //   - login the client
   // Returns 1 for a PROOF master server, 0 otherwise
   // Returns logged-in user, the remote client procotol cproto, the
   // client kind of user anon and, if anonymous user, the client passwd.
   // If TServerSocket (servtype==kSOCKD), the protocol number is returned
   // in anon.
   // Called just after opening the connection

   if (gDebug > 2)
      ErrorInfo("RpdInitSession: %s", gServName[servtype].c_str());

   int retval = 0;

   // CleanUp authentication table, if needed or required ...
   RpdInitAuth();

   // Get Host name
   NetGetRemoteHost(gOpenHost);

   if (servtype == kPROOFD) {

      // find out if we are supposed to be a master or a slave server
      char  msg[80];
      if (NetRecv(msg, sizeof(msg)) < 0) {
         ErrorInfo("RpdInitSession: Cannot receive master/slave status");
         return -1;
      }

      retval = !strcmp(msg, "master") ? 1 : 0;

      if (gDebug > 0)
         ErrorInfo("RpdInitSession: PROOF master/slave = %s", msg);
   }

   // Get protocol first
   // Failure typically indicate special actions like cleanup
   // which do not need additional work
   // The calling program will then decide what to do
   int rcp = RpdProtocol(servtype);
   if (rcp != 0) {
      if (rcp == -1)
         ErrorInfo("RpdInitSession: error getting remote protocol");
      else if (rcp != -2)
         ErrorInfo("RpdInitSession: unknown error from RpdProtocol");
      return rcp;
   }

   // Check if authentication is required
   // Old clients do not support no authentication mode
   bool runAuth = (gClientProtocol < 11 || gRequireAuth) ? 1 : 0;

   // user authentication (does not return in case of failure)
   int auth = 0;
   if (runAuth) {
      auth = RpdAuthenticate();
      if (auth == 0) {
         ErrorInfo("RpdInitSession: unsuccessful authentication attempt");
         return -1;
      }
   } else {
      auth = RpdNoAuth(servtype);
   }

   // Login the user (if in rootd/proofd environment)
   if (gDoLogin > 0) {
      if (RpdLogin(servtype,auth) != 0) {
         ErrorInfo("RpdInitSession: unsuccessful login attempt");
         // Notify failure to client ...
         NetSend(0, kROOTD_AUTH);
         return -1;
      }
   } else {
      // Notify authentication to client ...
      NetSend(auth, kROOTD_AUTH);
      // Send also new offset if it changed ...
      if (auth == 2)
         NetSend(gOffSet, kROOTD_AUTH);
      if (gDebug > 0)
         ErrorInfo("RpdInitSession: User '%s' authenticated", gUser);
      retval = auth;
   }

   // Output vars
   user = std::string(gUser);
   cproto = gClientProtocol;
   if (servtype == kSOCKD)
      anon = gSec;
   else
      anon = gAnon;
   if (gAnon)
      passwd = std::string(gPasswd);

   return retval;
}

//______________________________________________________________________________
int RpdInitSession(int servtype, std::string &user, int &rid)
{
   // Perform the action needed to commence the new session:
   //   - set debug flag
   //   - check authentication table
   //   - Inquire protocol
   //   - authenticate the client
   //   - login the client
   // Returns 1 for a PROOF master server, 0 otherwise
   // Returns logged-in user and remote process id in rid
   // Called just after opening the connection

   int dum1 = 0, dum2 = 0;
   std::string dum3;
   rid = gRemPid;
   return RpdInitSession(servtype,user,dum1,dum2,dum3);

}


//______________________________________________________________________________
int RpdNoAuth(int servtype)
{
   // Perform entrance formalities in case of no authentication
   // mode, i.e. get target user and check if authorized
   // Don't return if something goes wrong

   if (gDebug > 1)
      ErrorInfo("RpdNoAuth: no authentication required");

   // Special value for this case
   int auth = 0;

   // Receive target username
   if (servtype == kROOTD || servtype == kPROOFD) {

      char buf[kMAXPATHLEN];
      EMessageTypes kind;
      if (NetRecv(buf, kMAXPATHLEN, kind) < 0) {
         NetSend(kErrBadMess, kROOTD_ERR);
         ErrorInfo("RpdNoAuth: error receiving target user");
         goto quit;
      }

      if (kind == kROOTD_BYE)
         goto quit;

      if (kind != kROOTD_USER) {
         NetSend(kErrBadOp, kROOTD_ERR);
         ErrorInfo("RpdNoAuth: protocol error:"
          " received msg type: %d, expecting: %d", kind, kROOTD_USER);
         goto quit;
      }

      // Decode buffer
      char ruser[kMAXUSERLEN], user[kMAXUSERLEN];
      int nw = sscanf(buf,"%64s %64s",ruser,user);
      if (nw <= 0 || !strcmp(ruser,"-1")) {
         NetSend(kErrBadMess, kROOTD_ERR);
         ErrorInfo("RpdNoAuth: received uncorrect information: %s", buf);
         goto quit;
      }
      // If target user not send, assume user == ruser
      if (nw == 1)
         snprintf(user,kMAXUSERLEN,"%s",ruser);

      struct passwd *pw = 0;
      if ((pw = getpwnam(user)) == 0) {
         NetSend(kErrNoUser, kROOTD_ERR);
         ErrorInfo("RpdNoAuth: user %s unknown", user);
         goto quit;
      }

      // If server is not started as root, require user to be the same
      // as the one who started rootd
      uid_t uid = getuid();
      if (uid && uid != pw->pw_uid) {
         NetSend(kErrBadUser, kROOTD_ERR);
         ErrorInfo("RpdNoAuth: user not same as effective user of rootd");
         goto quit;
      }

      if (gDebug > 2)
         ErrorInfo("RpdNoAuth: remote user: %s, target user: %s",ruser,user);

      SPrintf(gUser, 63, "%s", user);
   }

   auth = 4;

   quit:
   return auth;
}

//______________________________________________________________________________
int RpdSetUid(int uid)
{
   // Change current user id to uid (and gid).

   if (gDebug > 2)
      ErrorInfo("RpdSetUid: enter ...uid: %d", uid);

   struct passwd *pw = getpwuid(uid);

   if (!pw) {
      ErrorInfo("RpdSetUid: uid %d does not exist locally", uid);
      return -1;
   } else if (chdir(pw->pw_dir) == -1) {
      ErrorInfo("RpdSetUid: can't change directory to %s", pw->pw_dir);
      return -1;
   }

   if (getuid() == 0) {

      // set access control list from /etc/initgroup
      initgroups(pw->pw_name, pw->pw_gid);

      // set uid and gid
      if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1) {
         ErrorInfo("RpdSetUid: can't setgid for uid %d", uid);
         return -1;
      }
      if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1) {
         ErrorInfo("RpdSetUid: can't setuid for uid %d", uid);
         return -1;
      }
   }

   if (gDebug > 0)
      ErrorInfo("RpdSetUid: uid set (%d,%s)", uid, pw->pw_name);

   return 0;
}

//_____________________________________________________________________________
void RpdInit(EService serv, int pid, int sproto, unsigned int options,
             int rumsk, int sshp, const char *tmpd, const char *asrpp, int login)
{
   // Change defaults job control options.

   gService        = serv;
   gParentId       = pid;
   gServerProtocol = sproto;
   gReUseAllow     = rumsk;
   gSshdPort       = sshp;
   gDoLogin        = login;

   // Parse options
   gCheckHostsEquiv= (bool)((options & kDMN_HOSTEQ) != 0);
   gRequireAuth    = (bool)((options & kDMN_RQAUTH) != 0);
   gSysLog         = (bool)((options & kDMN_SYSLOG) != 0);

   if (tmpd && strlen(tmpd)) {
      gTmpDir      = tmpd;
      gRpdAuthTab  = gTmpDir + gAuthTab;
      gRpdKeyRoot  = gTmpDir + gKeyRoot;
   }
   // Auth Tab and public key files are exclusive to this family
   gRpdAuthTab.append(".");
   gRpdAuthTab.append(ItoA(getuid()));
   gRpdKeyRoot.append(ItoA(getuid()));
   gRpdKeyRoot.append("_");

   if (asrpp && strlen(asrpp))
      gAltSRPPass  = asrpp;

#ifdef R__GLBS
   // Init globus
   if (RpdGlobusInit() != 0)
      ErrorInfo("RpdInit: failure initializing globus authentication");
#endif

   if (gDebug > 0) {
      ErrorInfo("RpdInit: gService= %s, gSysLog= %d, gSshdPort= %d",
                 gServName[gService].c_str(), gSysLog, gSshdPort);
      ErrorInfo("RpdInit: gParentId= %d", gParentId);
      ErrorInfo("RpdInit: gRequireAuth= %d, gCheckHostEquiv= %d",
                 gRequireAuth, gCheckHostsEquiv);
      ErrorInfo("RpdInit: gReUseAllow= 0x%x", gReUseAllow);
      ErrorInfo("RpdInit: gServerProtocol= %d", gServerProtocol);
      ErrorInfo("RpdInit: gDoLogin= %d", gDoLogin);
      if (tmpd)
         ErrorInfo("RpdInit: gTmpDir= %s", gTmpDir.c_str());
      if (asrpp)
         ErrorInfo("RpdInit: gAltSRPPass= %s", gAltSRPPass.c_str());
#ifdef R__GLBS
      ErrorInfo("RpdInit: gHaveGlobus: %d", (int) gHaveGlobus);
#endif
   }
}


//______________________________________________________________________________
int SPrintf(char *buf, size_t size, const char *va_(fmt), ...)
{
   // Acts like snprintf with some printout in case of error if required
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
      strlcpy(str,"-1", sizeof(str));
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


//______________________________________________________________________________
int RpdRetrieveSpecialPass(const char *usr, const char *fpw, char *pass, int lpwmax)
{
   // Retrieve specific ROOT password from $HOME/fpw, if any.
   // To avoid problems with NFS-root-squashing, if 'root' changes temporarly the
   // uid/gid to those of the target user (usr).
   // If OK, returns pass length and fill 'pass' with the password, null-terminated.
   // ('pass' is allocated externally to contain max lpwmax bytes).
   // If the file does not exists, return 0 and an empty pass.
   // If any problems with the file occurs, return a negative
   // code, -2 indicating wrong file permissions.
   // If any problem with changing ugid's occurs, prints a warning trying anyhow
   // to read the password hash.

   int rc = -1;
   int len = 0, n = 0, fid = -1;

   // Check inputs
   if (!usr || !pass) {
      if (gDebug > 0)
         ErrorInfo("RpdRetrieveSpecialPass: invalid arguments:"
                   " us:%p, sp:%p", usr, pass);
      return rc;
   }

   struct passwd *pw = getpwnam(usr);
   if (!pw) {
      if (gDebug > 0)
         ErrorInfo("RpdRetrieveSpecialPass: user '%s' does not exist", usr);
      return rc;
   }

   // target and actual uid
   int uid = pw->pw_uid;
   int ouid = getuid();

   // Temporary change to target user ID to avoid NFS squashing problems
   if (ouid == 0) {

      // set access control list from /etc/initgroup
      if (initgroups(pw->pw_name, pw->pw_gid) == -1)
         ErrorInfo("RpdRetrieveSpecialPass: can't initgroups for uid %d"
                   " (errno: %d)", uid, GetErrno());
      // set uid and gid
      if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1)
         ErrorInfo("RpdRetrieveSpecialPass: can't setgid for gid %d"
                   " (errno: %d)", pw->pw_gid, GetErrno());
      if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1)
         ErrorInfo("RpdRetrieveSpecialPass: can't setuid for uid %d"
                   " (errno: %d)", uid, GetErrno());
   }

   // The file now
   char rootdpass[kMAXPATHLEN];
   SPrintf(rootdpass, kMAXPATHLEN, "%s/%s", pw->pw_dir, fpw);

   if (gDebug > 0)
      ErrorInfo
         ("RpdRetrieveSpecialPass: checking file %s for user %s",rootdpass,
           pw->pw_name);


   if ((fid = open(rootdpass, O_RDONLY)) == -1) {
      ErrorInfo("RpdRetrieveSpecialPass: cannot open password file"
                " %s (errno: %d)", rootdpass, GetErrno());
      rc = -1;
      goto back;
   }
   // Check first the permissions: should be 0600
   struct stat st;
   if (fstat(fid, &st) == -1) {
      ErrorInfo("RpdRetrieveSpecialPass: cannot stat descriptor %d"
                  " %s (errno: %d)", fid, GetErrno());
      close(fid);
      rc = -1;
      goto back;
   }
   if (!S_ISREG(st.st_mode) || S_ISDIR(st.st_mode) ||
       (st.st_mode & (S_IWGRP | S_IWOTH | S_IRGRP | S_IROTH)) != 0) {
      ErrorInfo("RpdRetrieveSpecialPass: pass file %s: wrong permissions"
                " 0%o (should be 0600)", rootdpass, (st.st_mode & 0777));
      ErrorInfo("RpdRetrieveSpecialPass: %d %d",
                S_ISREG(st.st_mode),S_ISDIR(st.st_mode));
      close(fid);
      rc = -2;
      goto back;
   }

   if ((n = read(fid, pass, lpwmax - 1)) <= 0) {
      close(fid);
      ErrorInfo("RpdRetrieveSpecialPass: cannot read password file"
                " %s (errno: %d)", rootdpass, GetErrno());
      rc = -1;
      goto back;
   }
   close(fid);

   // Get rid of special trailing chars
   len = n;
   while (len-- && (pass[len] == '\n' || pass[len] == 32))
      pass[len] = 0;

   // Null-terminate
   pass[++len] = 0;
   rc = len;

   back:
   // Change back uid's
   if (ouid == 0) {
      // set uid and gid
      if (setresgid(0, 0, 0) == -1)
         ErrorInfo("RpdRetrieveSpecialPass: can't re-setgid for gid 0"
                   " (errno: %d)", GetErrno());
      if (setresuid(0, 0, 0) == -1)
         ErrorInfo("RpdRetrieveSpecialPass: can't re-setuid for uid 0"
                   " (errno: %d)", GetErrno());
   }

   // We are done
   return rc;
}

} // namespace ROOT
