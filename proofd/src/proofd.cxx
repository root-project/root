// @(#)root/proofd:$Name:  $:$Id: proofd.cxx,v 1.34 2003/08/29 10:41:28 rdm Exp $
// Author: Fons Rademakers   02/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/* Parts of this file are copied from the MIT krb5 distribution and
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
 *
 */

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Proofd                                                               //
//                                                                      //
// PROOF, Parallel ROOT Facility, front-end daemon.                     //
// This small server is started either by inetd when a client requests  //
// a connection to a PROOF server or by hand (i.e. from the command     //
// line). By default proofd uses port 1093 (allocated by IANA,          //
// www.iana.org, to proofd). If we don't want the PROOF server          //
// to run on this specific node, e.g. because the system is being       //
// shutdown or there are already too many servers running, we send      //
// the client a re-route message and close the connection. Otherwise    //
// we authenticate the user and exec the proofserv program.             //
// To run proofd via inetd add the following line to /etc/services:     //
//                                                                      //
// proofd     1093/tcp                                                  //
//                                                                      //
// and to /etc/inetd.conf:                                              //
//                                                                      //
// proofd stream tcp nowait root /usr/local/root/bin/proofd -i \        //
//    /usr/local/root                                                   //
//                                                                      //
// Force inetd to reread its conf file with "kill -HUP <pid inetd>".    //
//                                                                      //
// If xinetd is used instead, a file named 'proofd' should be created   //
// under /etc/xinetd.d with content                                     //
//                                                                      //
// # default: off                                                       //
// # description: The proof daemon                                      //
// #                                                                    //
// service proofd                                                       //
// {                                                                    //
//      disable         = no                                            //
//      flags           = REUSE                                         //
//      socket_type     = stream                                        //
//      wait            = no                                            //
//      user            = root                                          //
//      server          = /usr/local/bin/proofd                         //
//      server_args     = -i -d 0 /usr/local/root                       //
// }                                                                    //
//                                                                      //
// and xinetd restarted (/sbin/service xinetd restart).                 //
//                                                                      //
// You can also start proofd by hand running directly under your        //
// private account (no root system priviliges needed). For example to   //
// start proofd listening on port 5252 just type:                       //
//                                                                      //
// prootf -p 5252 $ROOTSYS                                              //
//                                                                      //
// Notice: no & is needed. Proofd will go in background by itself.      //
//                                                                      //
// Proofd arguments:                                                    //
//   -i                says we were started by inetd                    //
//   -p port#          specifies a different port to listen on          //
//   -b tcpwindowsize  specifies the tcp window size in bytes (e.g. see //
//                     http://www.psc.edu/networking/perf_tune.html)    //
//                     Default is 65535. Only change default for pipes  //
//                     with a high bandwidth*delay product.             //
//   -d level          level of debug info written to syslog            //
//                     0 = no debug (default)                           //
//                     1 = minimum                                      //
//                     2 = medium                                       //
//                     3 = maximum                                      //
//   -f                do not run as daemon, run in the foreground      //
//   -S keytabfile     use this keytab file, instead of the default     //
//                     (option only supported when compiled with        //
//                     Kerberos5 support)                               //
//   -T <tmpdir>       specifies the directory path to be used to place //
//                     temporary files; default is /usr/tmp.            //
//                     Useful if not running as root.                   //
//   -t period         defines the period (in hours) for cleaning of    //
//                     the authentication table <tmpdir>/rpdauthtab     //
//   -G gridmapfile    defines the gridmap file to be used for globus   //
//                     authentication if different from globus default  //
//                     (/etc/grid-security/gridmap); (re)defines the    //
//                     GRIDMAP environment variable.                    //
//   -C hostcertfile   defines a file where to find information for the //
//                     local host Globus information (see GLOBUS.README //
//                     for details)                                     //
//   -s <sshd_port>    specifies the port number for the sshd daemon    //
//                     (deafult is 22)                                  //
//   rootsys_dir       directory which must contain bin/proofserv and   //
//                     proof/etc/proof.conf. If not specified ROOTSYS   //
//                     or built-in (as specified to ./configure) is     //
//                     tried.                                           //
//                                                                      //
//  When your system uses shadow passwords you have to compile proofd   //
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
//  To use AFS for authentication compile proofd with the -DR__AFS      //
//  flag. In that case you also need to link with the AFS libraries.    //
//  See the Makefiles for more details.                                 //
//                                                                      //
//  To use Secure Remote Passwords (SRP) for authentication compile     //
//  proofd with the -DR__SRP flag. In that case you also need to link   //
//  with the SRP and gmp libraries. See the Makefile for more details.  //
//  SRP is described at: http://srp.stanford.edu/.                      //
//                                                                      //
//  See README.AUTH for more details on the authentication features.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// Protocol changes (see gProtocol):
// 6: added support for kerberos5 authentication
// 7: added support for Globus, SSH and uid/gid authentication and negotiation

#include "config.h"
#include "RConfig.h"

#include <ctype.h>
#include <fcntl.h>
#include <pwd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#include <sys/un.h>

#if defined(linux)
#   include <features.h>
#   if __GNU_LIBRARY__ == 6
#      ifndef R__GLIBC
#         define R__GLIBC
#      endif
#   endif
#endif
#if defined(__MACH__) && !defined(__APPLE__) || \
    (defined(__CYGWIN__) && defined(__GNUC__))
#   define R__GLIBC
#endif

#if defined(__FreeBSD__) && (__FreeBSD__ < 4)
#include <sys/file.h>
#define lockf(fd, op, sz)   flock((fd), (op))
#define F_LOCK             (LOCK_EX | LOCK_NB)
#define F_ULOCK             LOCK_UN
#endif

#if defined(linux) || defined(__sun) || defined(__sgi) || \
    defined(_AIX) || defined(__FreeBSD__) || defined(__MACH__)
#include <grp.h>
#include <sys/types.h>
#endif

#if defined(__sun) || defined(R__GLIBC)
#include <crypt.h>
#endif

#if defined(__osf__) || defined(__sgi)
extern "C" char *crypt(const char *, const char *);
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

#if defined(__sun)
#if defined(R__SUNGCC3)
extern "C" int gethostname(char *, unsigned int);
#else
extern "C" int gethostname(char *, int);
#endif
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

#ifdef R__KRB5
extern "C" {
   #include <com_err.h>
   #include <krb5.h>
   int krb5_net_write(krb5_context, int, const char *, int);
}
#include <string>
extern krb5_deltat krb5_clockskew;
#endif

#include "proofdp.h"

#ifdef R__KRB5
extern krb5_keytab  gKeytab; // to allow specifying on the command line
extern krb5_context gKcontext;
#endif

//--- Globals ------------------------------------------------------------------

const int   kMaxSlaves       = 32;
const char *gAuthMeth[kMAXSEC]= {"UsrPwdClear","SRP","Krb5","Globus","SSH","UidGidClear"};

char    gFilePA[40]              = { 0 };

int     gAuthListSent            = 0;
int     gCleaningPeriod          = 24;       // period for Auth table cleanup (default 1 day = 24 hours)
char    gConfDir[kMAXPATHLEN]    = { 0 };
int     gDebug                   = 0;
int     gForegroundFlag          = 0;
int     gInetdFlag               = 0;
int     gMaster                  =-1;
int     gPort                    = 0;
int     gProtocol                = 7;       // increase when protocol changes
char    gRcFile[kMAXPATHLEN]     = { 0 };
int     gRootLog                 = 0;
char    gRpdAuthTab[kMAXPATHLEN] = { 0 };   // keeps track of authentication info

#ifdef R__GLBS
  int          gShmIdCred        = -1;     // global, to pass the shm ID to proofserv
#endif

using namespace ROOT;

//--- Machine specific routines ------------------------------------------------

#if !defined(__hpux) && !defined(linux) && !defined(__FreeBSD__) || \
    defined(R__WINGCC)
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
#if defined(linux) && !defined(R__WINGCC) && !defined(HAS_SETRESUID)
extern "C" {
   int setresgid(gid_t r, gid_t e, gid_t s);
   int setresuid(uid_t r, uid_t e, uid_t s);
}
#endif

//
//--- Error handlers -----------------------------------------------------------

//______________________________________________________________________________
void Err(int level, const char *msg)
{
   Perror((char *)msg);
   if (level > -1) NetSend(level, kROOTD_ERR);
}
//______________________________________________________________________________
void ErrFatal(int level, const char *msg)
{
   Perror((char *)msg);
   if (level > -1) NetSend(msg, kMESS_STRING);
   exit(1);
}
//______________________________________________________________________________
void ErrSys(int level, const char *msg)
{
   Perror((char *)msg);
   ErrFatal(level, msg);
}

//--- Proofd routines ----------------------------------------------------------

//______________________________________________________________________________
void ProofdProtocol()
{
   // Return proofd protocol version id.

   NetSend(gProtocol, kROOTD_PROTOCOL);
}

//______________________________________________________________________________
void ProofdLogin()
{
   // Authentication was successful, set user environment.

   if (gDebug > 2) ErrorInfo("ProofdLogin: enter ... gUser: %s", gUser);

   struct passwd *pw = getpwnam(gUser);

   if (chdir(pw->pw_dir) == -1)
      Error(ErrFatal,-1,"ProofdLogin: can't change directory to %s", pw->pw_dir);

   if (getuid() == 0) {

#ifdef R__GLBS
      // We need to change the ownership of the shared memory segments used
      // for credential export to allow proofserv to destroy them
      struct shmid_ds shm_ds;
      if (gShmIdCred > 0) {
        if ( shmctl(gShmIdCred,IPC_STAT,&shm_ds) == -1)
           Error(ErrFatal,-1,"ProofdLogin: can't get info about shared memory segment %d", gShmIdCred);
        shm_ds.shm_perm.uid = pw->pw_uid;
        shm_ds.shm_perm.gid = pw->pw_gid;
        if ( shmctl(gShmIdCred,IPC_SET,&shm_ds) == -1)
           Error(ErrFatal,-1,"ProofdLogin: can't change ownership of shared memory segment %d", gShmIdCred);
      }
#endif
      // set access control list from /etc/initgroup
      initgroups(gUser, pw->pw_gid);

      // set uid and gid
      if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1)
         Error(ErrFatal,-1,"ProofdLogin: can't setgid for user %s", gUser);
      if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1)
         Error(ErrFatal,-1,"ProofdLogin: can't setuid for user %s", gUser);
   }

   // set HOME env
   char *home = new char[6+strlen(pw->pw_dir)];
   sprintf(home, "HOME=%s", pw->pw_dir);
   putenv(home);

   umask(022);

  // Notify authentication to client ...
   NetSend(gAuth, kROOTD_AUTH);
   // Send also new offset if it changed ...
   if (gAuth==2) NetSend(gOffSet, kROOTD_AUTH);

   if (gDebug > 0)
      ErrorInfo("ProofdLogin: user %s authenticated", gUser);
}

//______________________________________________________________________________
void ProofdPass(const char *pass)
{
   // Check user's password.

   // Evaluate credentials ...
   RpdPass(pass);

   // Login, if ok ...
   if (gAuth==1) ProofdLogin();

   return;
}

//______________________________________________________________________________
void ProofdUser(const char *sstr)
{
   // Check user's password.
   gAuth = 0;

   // Evaluate credentials ...
   RpdUser(sstr);

   // Login, if ok ...
   if (gAuth==1) ProofdLogin();

   return;
}

//______________________________________________________________________________
void ProofdKrb5Auth(const char *sstr)
{
   // Authenticate via Kerberos.

   // Reset global variable
   gAuth = 0;

   // Evaluate credentials ...
   RpdKrb5Auth(sstr);

   // Login, if ok ...
   if (gAuth==1) ProofdLogin();

}

//______________________________________________________________________________
void ProofdSshAuth(const char *sstr)
{
   // Authenticate via SSH.

   // Reset global variable
   gAuth = 0;

   // Evaluate credentials ...
   RpdSshAuth(sstr);

   // Login, if ok ...
   if (gAuth==1) ProofdLogin();

   return;
}

//______________________________________________________________________________
void ProofdSRPUser(const char *sstr)
{
   // Use Secure Remote Password protocol.
   // Check user id in $HOME/.srootdpass file.

   // Reset global variable
   gAuth = 0;

   // Evaluate credentials ...
   RpdSRPUser(sstr);

   // Login, if ok ...
   if (gAuth==1) ProofdLogin();

   return;
}

//______________________________________________________________________________
void ProofdRfioAuth(const char *sstr)
{
   // Authenticate via Rfio

   // Reset global variable
   gAuth = 0;

   // Evaluate credentials ...
   RpdRfioAuth(sstr);

   // ... and login
   if (gAuth==1) ProofdLogin();

}

//______________________________________________________________________________
void ProofdGlobusAuth(const char *sstr){
   // Authenticate via Globus

   // Reset global variable
   gAuth = 0;

   // Evaluate credentials ...
   RpdGlobusAuth(sstr);

   // Login, if ok ...
   if (gAuth==1) ProofdLogin();

   return;
}

//______________________________________________________________________________
void CheckGlobus(char *rcfile)
{
   // Create a resource table and read the (possibly) three resource files, i.e
   // $ROOTSYS/system<name> (or ROOTETCDIR/system<name>), $HOME/<name> and
   // ./<name>. ROOT always reads ".rootrc" (in TROOT::InitSystem()). You can
   // read additional user defined resource files by creating addtional TEnv
   // object.

   char line[kMAXPATHLEN];
   int  sGlobus =-1, uGlobus =-1, lGlobus =-1, pGlobus=-1;
   char lRcFile[kMAXPATHLEN] = { 0 }, uRcFile[kMAXPATHLEN] = { 0 },
        sRcFile[kMAXPATHLEN] = { 0 }, ProofCf[kMAXPATHLEN] = { 0 };
   char namenv[256], valenv[256], dummy[512];
   char sname[128] = "system";
   char s[kMAXPATHLEN] = { 0 };

   if (gDebug > 2) ErrorInfo("CheckGlobus: Enter: rcfile: %s", s);

   strcat(sname, rcfile);
#ifdef ROOTETCDIR
   strcat(s, sname);
   sprintf(s, "%s/%s", ROOTETCDIR, sname);
#else
   char etc[kMAXPATHLEN];
   sprintf(etc, "%s/etc", gConfDir);
   sprintf(s, "%s/%s", etc, sname);
   // Check file existence and readibility
   if (access(s, F_OK) || access(s, R_OK)) {
      // for backward compatibility check also $ROOTSYS/system<name> if
      // $ROOTSYS/etc/system<name> does not exist
      sprintf(s, "%s/%s", gConfDir, sname);
      if (access(s, F_OK) || access(s, R_OK)) {
         // for backward compatibility check also $ROOTSYS/<name> if
         // $ROOTSYS/system<name> does not exist
         sprintf(s, "%s/%s", gConfDir, rcfile);
      }
   }
#endif
   // Check in the system environment ...
   if (gDebug > 2) ErrorInfo("CheckGlobus: checking system: %s", s);
   if (!access(s, F_OK) && !access(s, R_OK)) {
      FILE *fs = fopen(s, "r");
      while (fgets(line, sizeof(line), fs)) {
         if (line[0] == '#') continue;   // skip comment lines
         sscanf(line, "%s %s %s", namenv, valenv, dummy);
         if (!strcmp(namenv, "Proofd.Authentication:")) {
            int sec = atoi(valenv);
            if (gDebug > 2) ErrorInfo("CheckGlobus: %s: %s (%d)", namenv, valenv, sec);
            if ((sGlobus != 1) && (sec == 3)) sGlobus = 1;
         }
      }
      fclose(fs);
      strcpy(sRcFile, s);
   }
   if (gDebug > 2) ErrorInfo("CheckGlobus: system: %d: %s", sGlobus, sRcFile);

   // Check in the user environment ...
   if (getenv("HOME")) {
      sprintf(s, "%s/%s", getenv("HOME"), rcfile);
      if (gDebug > 2) ErrorInfo("CheckGlobus: checking user: %s", s);
      if (!access(s, F_OK) && !access(s, R_OK)) {
         FILE *fs= fopen(s, "r");
         while (fgets(line, sizeof(line), fs)) {
            if (line[0] == '#') continue;   // skip comment lines
            sscanf(line, "%s %s %s", namenv, valenv, dummy);
            if (!strcmp(namenv, "Proofd.Authentication:")) {
               int sec = atoi(valenv);
               if (gDebug > 2) ErrorInfo("CheckGlobus: %s: %s (%d)", namenv, valenv, sec);
               if ((uGlobus != 1) && (sec == 3)) uGlobus = 1;
            }
         }
         fclose(fs);
         strcpy(uRcFile, s);
      }
   }
   if (gDebug > 2) ErrorInfo("CheckGlobus: user: %d: %s", uGlobus, uRcFile);

   // Check in the local environment ...
   sprintf(s,"%s",rcfile);
   if (gDebug > 2) ErrorInfo("CheckGlobus: checking local: %s", s);

   if (!access(s, F_OK) && !access(s, R_OK)) {
      FILE *fs = fopen(s, "r");
      while (fgets(line, sizeof(line), fs)) {
         if (line[0] == '#') continue;   // skip comment lines
         sscanf(line, "%s %s %s", namenv, valenv, dummy);
         if (!strcmp(namenv, "Proofd.Authentication:")) {
            int sec = atoi(valenv);
            if (gDebug > 2) ErrorInfo("CheckGlobus: %s: %s (%d)", namenv, valenv, sec);
            if ((lGlobus != 1) && (sec == 3)) lGlobus = 1;
         }
      }
      fclose(fs);
      strcpy(lRcFile,s);
   }
   if (gDebug > 2) ErrorInfo("CheckGlobus: local: %d: %s", lGlobus, lRcFile);

   // Check finally, the proof.conf files to see if there are any
   // specific instructions there. The system one first:
   sprintf(s, "%s/proof/etc/proof.conf", gConfDir);
   if (gDebug > 2) ErrorInfo("CheckGlobus: checking system proof.conf: %s", s);

   if (!access(s, F_OK) && !access(s, R_OK)) {
      pGlobus = 0;
      FILE *fs = fopen(s,"r");
      while (fgets(line, sizeof(line), fs)) {
         if (line[0] == '#') continue;   // skip comment lines
         if (line[strlen(line)-1] == '\n') line[strlen(line)-1] = '\0';
         char wd[12][64];
         int nw = sscanf(line, "%s %s %s %s %s %s %s %s %s %s %s %s",
                         wd[0],wd[1],wd[2],wd[3],wd[4],wd[5],wd[6],wd[7],
                         wd[8],wd[9],wd[10],wd[11]);
         // find all slave servers
         if (nw >= 2 && !strcmp(wd[0], "slave" )) {
            for (int i = 2; i < nw; i++) {
               if (!strncmp(wd[i],"globus", 6)) pGlobus = 1;
            }
            if (gDebug > 2) ErrorInfo("CheckGlobus: %s", line);
         }
      }
      fclose(fs);
      strcpy(ProofCf, s);
   }

   // Now the user one:
   if (getenv("HOME")) {
      sprintf(s, "%s/.proof.conf", getenv("HOME"));
      if (gDebug > 2) ErrorInfo("CheckGlobus: checking user proof.conf: %s", s);

      if (!access(s, F_OK) && !access(s, R_OK)) {
         pGlobus = 0;
         FILE *fs = fopen(s, "r");
         while (fgets(line, sizeof(line), fs)) {
            if (line[0] == '#') continue;   // skip comment lines
            if (line[strlen(line)-1] == '\n') line[strlen(line)-1] = '\0';
            char wd[12][64];
            int nw = sscanf(line, "%s %s %s %s %s %s %s %s %s %s %s %s",
                            wd[0],wd[1],wd[2],wd[3],wd[4],wd[5],wd[6],wd[7],
                            wd[8],wd[9],wd[10],wd[11]);
            // find all slave servers
            if (nw >= 2 && !strcmp(wd[0], "slave" )) {
               for (int i = 2; i < nw; i++) {
                  if (!strncmp(wd[i], "globus", 6)) pGlobus = 1;
               }
               if (gDebug > 2) ErrorInfo("CheckGlobus: %s", line);
            }
         }
         fclose(fs);
         strcpy(ProofCf, s);
      }
   }

   if (gDebug > 2) ErrorInfo("CheckGlobus: proof.conf: %d: %s", pGlobus, ProofCf);

   // Now fill the globals ...
   if (lGlobus != -1) {
      gGlobus = lGlobus; strcpy(gRcFile, lRcFile);
   } else if (uGlobus != -1) {
      gGlobus = uGlobus; strcpy(gRcFile, uRcFile);
   } else if (sGlobus != -1) {
      gGlobus = sGlobus; strcpy(gRcFile, sRcFile);
   } else if (pGlobus != -1) {
      gGlobus = pGlobus; strcpy(gRcFile, ProofCf);
   }

   if (gDebug > 2) ErrorInfo("CheckGlobus: exit: %d: %s", gGlobus, gRcFile);

   return;
}

//______________________________________________________________________________
bool ProofdReUseAuth(const char *sstr, int kind)
{
   // Check the requiring subject has already authenticated during this session
   // and its 'ticket' is still valid.
   // Not implemented for SRP and Krb5 (yet)

   if (RpdReUseAuth(sstr, kind)) {

      // Already authenticated ...we can login now
      ProofdLogin();
      return 1;

   } else {
      return 0;
   }
}

//______________________________________________________________________________
void Authenticate()
{
   // Handle user authentication.

   const int kMaxBuf = 1024;
   char recvbuf[kMaxBuf];
   EMessageTypes kind;
   int           Meth;

   while (!gAuth) {

      if (NetRecv(recvbuf, kMaxBuf, kind) < 0)
         Error(ErrFatal,-1,"Authenticate: error receiving message");

      // Decode the method ...
      Meth = RpdGetAuthMethod(kind);

      if (gDebug > 2) {
         if (kind != kROOTD_PASS) {
            ErrorInfo("Authenticate got: %d -- %s", kind, recvbuf);
         } else {
            ErrorInfo("Authenticate got: %d ", kind);
         }
      }

      // Guess the client procotol
      gClientProtocol = RpdGuessClientProt(recvbuf, kind);

      // If authentication required, check if we accept the method proposed; if not
      // send back the list of accepted methods, if any ...
      if (Meth != -1  && gClientProtocol > 8 ) {

         // Check if accepted ...
         if (RpdCheckAuthAllow(Meth, gOpenHost)) {
            if (gNumAllow>0) {
               if (gAuthListSent == 0) {
                  if (gDebug > 0) ErrorInfo("Authenticate: %s method not accepted from host: %s", gAuthMeth[Meth], gOpenHost);
                  NetSend(kErrNotAllowed, kROOTD_ERR);
                  RpdSendAuthList();
                  gAuthListSent = 1;
                  goto next;
               } else {
                  Error(ErrFatal,kErrNotAllowed, "Authenticate: method not in the list sent to the client");
               }
            } else
               Error(ErrFatal,kErrConnectionRefused, "Authenticate: connection refused from host %s", gOpenHost);
         }

         // Then check if a previous authentication exists and is valid
         // ReUse does not apply for RFIO
         if (kind != kROOTD_RFIO && ProofdReUseAuth(recvbuf, kind))
            goto recvauth;
      }

      switch (kind) {
         case kROOTD_USER:
            ProofdUser(recvbuf);
            break;
         case kROOTD_SRPUSER:
            ProofdSRPUser(recvbuf);
            break;
         case kROOTD_PASS:
            ProofdPass(recvbuf);
            break;
         case kROOTD_KRB5:
            ProofdKrb5Auth(recvbuf);
            break;
         case kROOTD_PROTOCOL:
            ProofdProtocol();
            break;
         case kROOTD_GLOBUS:
            ProofdGlobusAuth(recvbuf);
            break;
         case kROOTD_SSH:
            ProofdSshAuth(recvbuf);
            break;
         case kROOTD_RFIO:
            ProofdRfioAuth(recvbuf);
            break;
         case kROOTD_CLEANUP:
            RpdCleanup(recvbuf);
            ErrorInfo("Authenticate: authentication stuff cleaned - exit");
            exit(1);
            break;
         default:
            Error(ErrFatal,-1,"Authenticate: received bad opcode %d", kind);
      }

      if (gClientProtocol > 8) {

         if (gDebug > 2)
            ErrorInfo("Authenticate: here we are: kind:%d -- Meth:%d -- gAuth:%d -- gNumLeft:%d", kind, Meth, gAuth, gNumLeft);

         // If authentication failure, check if other methods could be tried ...
         if ((Meth != -1 || kind==kROOTD_PASS) && gAuth == 0) {
            if (gNumLeft > 0) {
               if (gAuthListSent == 0) {
                  RpdSendAuthList();
                  gAuthListSent = 1;
               } else
                  NetSend(-1, kROOTD_NEGOTIA);
            } else
               Error(ErrFatal, -1, "Authenticate: authentication failed");
         }
      }

recvauth:
      // If authentication successfull, receive info for later authentications
      if (gAuth == 1 && gClientProtocol > 8) {

         sprintf(gFilePA,"%s/proofauth.%ld", gTmpDir, (long)getpid());
         if (gDebug > 2) ErrorInfo("Authenticate: file with hostauth info is: %s", gFilePA);

         FILE *fpa = fopen(gFilePA, "w");
         if (fpa == 0) {
            ErrorInfo("Authenticate: error creating file: %s", gFilePA);
            goto next;
         }

         // Receive buffer
         EMessageTypes kindauth;
         int nr = NetRecv(recvbuf, kMaxBuf, kindauth);
         if (nr < 0 || kindauth != kPROOF_SENDHOSTAUTH)
            ErrorInfo("Authenticate: SENDHOSTAUTH: received: %d (%d bytes)", kindauth, nr);
         if (gDebug > 2) ErrorInfo("Authenticate: received: (%d) %s", nr, recvbuf);
         while (strcmp(recvbuf, "END")) {
            // Clean buffer
            recvbuf[nr+1] = '\0';
            // Write it to file
            fprintf(fpa, "%s\n", recvbuf);
            // Get the next one
            nr = NetRecv(recvbuf, kMaxBuf, kindauth);
            if (nr < 0 || kindauth != kPROOF_SENDHOSTAUTH)
               ErrorInfo("Authenticate: SENDHOSTAUTH: received: %d (%d bytes)", kindauth, nr);
            if (gDebug > 2) ErrorInfo("Authenticate: received: (%d) %s", nr, recvbuf);
         }
         // Close suth file
         fclose(fpa);
      }
next:
      continue;
   }

   if (gMaster == 1 && gAuth == 1 && gGlobus != -1 && kind != kROOTD_GLOBUS &&
       gClientProtocol > 8) {
      ErrorInfo("Authenticate: WARNING: got non-Globus authentication request");
      ErrorInfo("Authenticate: while later actions MAY need Globus credentials...");
      ErrorInfo("Authenticate: (source: %s)", gRcFile);
   }
}

//______________________________________________________________________________
const char *RerouteUser()
{
   // Look if user should be rerouted to another server node.

   char conffile[256];
   FILE *proofconf;

   sprintf(conffile, "%s/etc/proof.conf", gConfDir);
   if ((proofconf = fopen(conffile, "r")) != 0) {
      // read configuration file
      static char user_on_node[32];
      struct stat statbuf;
      char line[256];
      char node_name[kMaxSlaves][32];
      int  nnodes = 0;
      int  i;

      strcpy(user_on_node, "any");

      while (fgets(line, sizeof(line), proofconf) != 0) {
         char word[4][64];
         if (line[0] == '#') continue;  // skip comment lines
         int nword = sscanf(line, "%s %s %s %s",
                            word[0], word[1], word[2], word[3]);

         //
         // all available nodes must be configured by a line
         //    node <name>
         //
         if (nword >= 2 && strcmp(word[0], "node") == 0) {
            if (gethostbyname(word[1]) != 0) {
               if (nnodes < kMaxSlaves) {
                  strcpy(node_name[nnodes], word[1]);
                  nnodes++;
               }
            }
            continue;
         }

         //
         // users can be preferrably rerouted to a specific node
         //    user <name> on <node>
         //
         if (nword >= 4 && strcmp(word[0], "user") == 0 &&
             strcmp(word[1], gUser) == 0 && strcmp(word[2], "on") == 0) {
            // user <name> on <node>
            strcpy(user_on_node, word[3]);
            continue;
         }
      }
      fclose(proofconf);

      // make sure the node is running
      for (i = 0; i < nnodes; i++) {
         if (strcmp(node_name[i], user_on_node) == 0) {
            return user_on_node;
         }
      }

      //
      // get the node name from next.node update by a daemon monitoring
      // the system load; make sure the file is not completely out of date
      //
      sprintf(conffile, "%s/etc/next.node", gConfDir);
      if (stat(conffile, &statbuf) == -1) {
         return 0;
      } else if (difftime(time(0), statbuf.st_mtime) < 600 &&
                 (proofconf = fopen(conffile, "r")) != 0) {
         if (fgets(line, sizeof(line), proofconf) != 0) {
            sscanf(line, " %s ", user_on_node);
            for (i = 0; i < nnodes; i++) {
               if (strcmp(node_name[i], user_on_node) == 0) {
                  fclose(proofconf);
                  return user_on_node;
               }
            }
         }
         fclose(proofconf);
      }
   }
   return 0;
}

//______________________________________________________________________________
void ProofdExec()
{
   // Authenticate the user and exec the proofserv program.
   // gConfdir is the location where the PROOF config files and binaries live.

#ifdef R__GLBS
   char *argvv[12];
#else
   char *argvv[8];
#endif
   char  arg0[256];
   char  msg[80];
   char  rpid[20] = {0};

#ifdef R__DEBUG
   int debug = 1;
   while (debug)
      ;
#endif

   // Set debug level in RPDUtil ...
   RpdSetDebugFlag(gDebug);

   // CleanUp authentication table, if needed or required ...
   RpdCheckSession(gCleaningPeriod);

   // Get Host name
   const char *OpenHost = NetRemoteHost();
   strcpy(gOpenHost, OpenHost);
   if (gDebug > 0)
      ErrorInfo("ProofdExec: gOpenHost = %s", gOpenHost);

   // Set auth tab flag in RPDUtil ...
   RpdSetAuthTabFile(gRpdAuthTab);

   if (gDebug > 0)
      ErrorInfo("ProofdExec: gConfDir = %s", gConfDir);

   // find out if we are supposed to be a master or a slave server
   if (NetRecv(msg, sizeof(msg)) < 0)
      Error(ErrFatal,-1,"Cannot receive master/slave status");

   gMaster = !strcmp(msg, "master") ? 1 : 0;

   if (gDebug > 0)
      ErrorInfo("ProofdExec: master/slave = %s", msg);

   // user authentication
   Authenticate();

   // only reroute in case of master server
   const char *node_name;
   if (gMaster && (node_name = RerouteUser()) != 0) {
      // send a reroute request to the client passing the IP address

      char host_name[32];
      gethostname(host_name, sizeof(host_name));

      // make sure that we are not already on the target node
      if (strcmp(host_name, node_name) != 0) {
         struct hostent *host = gethostbyname(host_name);
         struct hostent *node;    // gethostbyname(node_name) would overwrite

         if (host != 0) {
            struct in_addr *host_addr = (struct in_addr*)(host->h_addr);
            char host_numb[32];
            strcpy(host_numb, inet_ntoa(*host_addr));

            if ((node = gethostbyname(node_name)) != 0) {
               struct in_addr *node_addr = (struct in_addr*)(node->h_addr);
               char node_numb[32];
               strcpy(node_numb, inet_ntoa(*node_addr));

               //
               // compare the string representation of the IP addresses
               // to avoid possible problems with host name aliases
               //
               if (strcmp(host_numb, node_numb) != 0) {
                  sprintf(msg, "Reroute:%s", node_numb);
                  NetSend(msg);
                  exit(0);
               }
            }
         }
      }
   }
   if (gDebug > 0)
      ErrorInfo("ProofdExec: send Okay (gSockFd: %d)",gSockFd);

   NetSend("Okay");

#ifdef R__GLBS
   // to pass over shm id to proofserv
   char  cShmIdCred[16];
   sprintf(cShmIdCred,"%d",gShmIdCred);
#endif

   // start server version
   sprintf(arg0, "%s/bin/proofserv", gConfDir);
   argvv[0] = arg0;
   argvv[1] = (char *)(gMaster ? "proofserv" : "proofslave");
   argvv[2] = gConfDir;
   argvv[3] = gFilePA;
   argvv[4] = gOpenHost;
   sprintf(rpid, "%d", gRemPid);
   argvv[5] = rpid;
   argvv[6] = gUser;
#ifdef R__GLBS
   argvv[7] = cShmIdCred;
   argvv[8] = 0;
   argvv[9] = 0;
   argvv[10] = 0;
   if (getenv("X509_CERT_DIR")!=0)  argvv[8] = strdup(getenv("X509_CERT_DIR"));
   if (getenv("X509_USER_CERT")!=0) argvv[9] = strdup(getenv("X509_USER_CERT"));
   if (getenv("X509_USER_KEY")!=0)  argvv[10] = strdup(getenv("X509_USER_KEY"));
   argvv[11] = 0;
#else
   argvv[7] = 0;
#endif

#ifndef ROOTPREFIX
   char *rootsys = new char[9+strlen(gConfDir)];
   sprintf(rootsys, "ROOTSYS=%s", gConfDir);
   putenv(rootsys);
#endif
#ifndef ROOTLIBDIR
   char *ldpath = new char[21+strlen(gConfDir)];
#   if defined(__hpux) || defined(_HIUX_SOURCE)
   sprintf(ldpath, "SHLIB_PATH=%s/lib", gConfDir);
#   elif defined(_AIX)
   sprintf(ldpath, "LIBPATH=%s/lib", gConfDir);
#   else
   sprintf(ldpath, "LD_LIBRARY_PATH=%s/lib", gConfDir);
#   endif
   putenv(ldpath);
#endif

   if (gDebug > 0)
#ifdef R__GLBS
      ErrorInfo("ProofdExec: execv(%s, %s, %s, %s,\n %s, %s, %s, %s, %s, %s, %s)",
                argvv[0], argvv[1], argvv[2], argvv[3], argvv[4],
                argvv[5], argvv[6], argvv[7], argvv[8], argvv[9], argvv[10]);
#else
      ErrorInfo("ProofdExec: execv(%s, %s, %s, %s, %s, %s, %s)",
                argvv[0], argvv[1], argvv[2], argvv[3],
                argvv[4], argvv[5], argvv[6]);
#endif

   if (!gInetdFlag) {
      // Duplicate the socket onto the descriptors 0, 1 and 2
      // and close the original socket descriptor (like inetd).
      dup2(gSockFd, 0);
      close(gSockFd);
      dup2(0, 1);
      dup2(0, 2);
   }

   // Start proofserv
   execv(arg0, argvv);

   // tell client that exec failed
   sprintf(msg, "Cannot start PROOF server --- make sure %s exists!", arg0);
   NetSend(msg);
}

//______________________________________________________________________________
int main(int argc, char **argv)
{
   char *s;
   int   tcpwindowsize = 65535;

   // Error Handlers
   gErrSys   = ErrSys;
   gErrFatal = ErrFatal;
   gErr      = Err;

   ErrorInit(argv[0]);

#ifdef R__KRB5
   const char *kt_fname;

   int retval = krb5_init_context(&gKcontext);
   if (retval)
      Error(ErrFatal, -1, "%s while initializing krb5", error_message(retval));
#endif

#ifdef R__GLBS
   char GridMap[kMAXPATHLEN] = { 0 };
#endif

   // Define service
   strcpy(gService, "proofd");

   while (--argc > 0 && (*++argv)[0] == '-')
      for (s = argv[0]+1; *s != 0; s++)
         switch (*s) {
            case 'i':
               gInetdFlag = 1;
               break;

            case 'f':
               gForegroundFlag = 1;
               break;

            case 'p':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-p requires a port number as argument\n");
                  Error(ErrFatal, -1, "-p requires a port number as argument");
               }
               gPort = atoi(*++argv);
               break;

            case 'd':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-d requires a debug level as argument\n");
                  Error(ErrFatal, -1, "-d requires a debug level as argument");
               }
               gDebug = atoi(*++argv);
               break;

            case 'b':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-b requires a buffersize in bytes as argument\n");
                  Error(ErrFatal, -1, "-b requires a buffersize in bytes as argument");
               }
               tcpwindowsize = atoi(*++argv);
               break;

            case 't':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-t requires as argument a period (in hours) as argument of cleaning for the auth table\n");
                  Error(ErrFatal, -1, "-t requires as argument a period (in hours) as argument of cleaning for the auth table");
               }
               gCleaningPeriod = atoi(*++argv);
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
                  Error(ErrFatal,-1,"-S requires a path to your keytab\n");
               }
               kt_fname = *++argv;
               if ((retval = krb5_kt_resolve(gKcontext, kt_fname, &gKeytab)))
                  Error(ErrFatal, -1, "%s while resolving keytab file %s",
                        error_message(retval), kt_fname);
               break;
#endif

#ifdef R__GLBS
            case 'G':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-G requires a file name for the gridmap file\n");
                  Error(ErrFatal, -1, "-G requires a file name for the gridmap file");
               }
               sprintf(GridMap, "%s", *++argv);
               if (setenv("GRIDMAP",GridMap,1) ){
                  Error(ErrFatal, -1, "%s while setting the GRIDMAP environment variable");
               }
               break;

            case 'C':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-C requires a file name for the host certificates file location\n");
                  Error(ErrFatal, -1, "-C requires a file name for the host certificates file location");
               }
               sprintf(gHostCertConf, "%s", *++argv);
               break;
#endif
            default:
               if (!gInetdFlag)
                  fprintf(stderr, "unknown command line option: %c\n", *s);
               Error(ErrFatal, -1, "unknown command line option: %c", *s);
         }

   if (argc > 0) {
      strncpy(gConfDir, *argv, kMAXPATHLEN-1);
      gConfDir[kMAXPATHLEN-1] = 0;
      sprintf(gExecDir, "%s/bin", gConfDir);
      sprintf(gAuthAllow, "%s/etc/rpdauth.allow", gConfDir);
   } else {
      // try to guess the config directory...
#ifndef ROOTPREFIX
      if (getenv("ROOTSYS")) {
         strcpy(gConfDir, getenv("ROOTSYS"));
         sprintf(gExecDir, "%s/bin", gConfDir);
         sprintf(gAuthAllow, "%s/etc/rpdauth.allow", gConfDir);
         if (gDebug > 0) ErrorInfo("main: no config directory specified using ROOTSYS (%s)", gConfDir);
      } else {
         if (!gInetdFlag)
            fprintf(stderr, "no config directory specified\n");
         Error(ErrFatal, -1, "no config directory specified");
      }
#else
      strcpy(gConfDir, ROOTPREFIX);
#endif
#ifdef ROOTBINDIR
      strcpy(gExecDir, ROOTBINDIR);
#endif
#ifdef ROOTETCDIR
      sprintf(gAuthAllow, "%s/rpdauth.allow", ROOTETCDIR);
#endif
   }

   // make sure needed files exist
   char arg0[256];
   sprintf(arg0, "%s/bin/proofserv", gConfDir);
   if (access(arg0, X_OK) == -1) {
      if (!gInetdFlag)
         fprintf(stderr, "incorrect config directory specified (%s)\n", gConfDir);
      Error(ErrFatal, -1, "incorrect config directory specified (%s)", gConfDir);
   }

   // dir for temporary files
   if (strlen(gTmpDir) <= 0) {
      sprintf(gTmpDir, "/usr/tmp");
      if (access(gTmpDir, R_OK) || access(gTmpDir, W_OK)) {
         sprintf(gTmpDir, "/tmp");
      }
   }

   // authentication tab file
   sprintf(gRpdAuthTab, "%s/rpdauthtab", gTmpDir);

   // Log to stderr if not started as daemon ...
   if (gForegroundFlag) RpdSetRootLogFlag(1);

   // Check if at any level there is request for Globus Authetication
   // for proofd or rootd
   sprintf(gRcFile, "%s", ".rootrc");
   CheckGlobus(gRcFile);
   if (gDebug > 0)
      ErrorInfo("main: gGlobus: %d, gRcFile: %s", gGlobus, gRcFile);

   if (!gInetdFlag) {

      // Start proofd up as a daemon process (in the background).
      // Also initialize the network connection - create the socket
      // and bind our well-know address to it.

      if (!gForegroundFlag) DaemonStart(1, 0, kPROOFD);

      NetInit(gService, gPort, tcpwindowsize);
   }

   if (gDebug > 0)
      ErrorInfo("main: pid = %d, gInetdFlag = %d", getpid(), gInetdFlag);

   // Concurrent server loop.
   // The child created by NetOpen() handles the client's request.
   // The parent waits for another request. In the inetd case,
   // the parent from NetOpen() never returns.

   while (1) {
      if (NetOpen(gInetdFlag,kPROOFD) == 0) {
         ProofdExec();     // child processes client's requests
         NetClose();       // then we are done
         exit(0);
      }

      // parent waits for another client to connect

   }

#ifdef R__KRB5
   // never called... needed?
   krb5_free_context(gKcontext);
#endif
}
