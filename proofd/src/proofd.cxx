// @(#)root/proofd:$Name:  $:$Id: proofd.cxx,v 1.22 2002/01/22 10:53:28 rdm Exp $
// Author: Fons Rademakers   02/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
//   rootsys_dir       directory which must contain bin/proofserv and   //
//                     proof/etc/proof.conf                             //
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
//////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include <ctype.h>
#include <fcntl.h>
#include <pwd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>

#if defined(linux)
#   include <features.h>
#   if __GNU_LIBRARY__ == 6
#      ifndef R__GLIBC
#         define R__GLIBC
#      endif
#   endif
#endif
#ifdef __MACH__
#   define R__GLIBC
#endif

#if defined(__FreeBSD__) && (__FreeBSD__ < 4)
#include <sys/file.h>
#define lockf(fd, op, sz)   flock((fd), (op))
#define	F_LOCK             (LOCK_EX | LOCK_NB)
#define	F_ULOCK             LOCK_UN
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

#if defined(sun)
extern "C" int gethostname(char *, int);
#ifndef SHADOWPW
#define SHADOWPW
#endif
#endif

#ifdef SHADOWPW
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

#include "proofdp.h"


//--- Globals ------------------------------------------------------------------

const char kProofdService[] = "proofd";
const char kRootdPass[]     = ".rootdpass";
const char kSRootdPass[]    = ".srootdpass";
const int  kMaxSlaves       = 32;
const int  kMAXPATHLEN      = 1024;

int  gInetdFlag             = 0;
int  gPort                  = 0;
int  gDebug                 = 0;
int  gSockFd                = -1;
int  gAuth                  = 0;
char gUser[64]              = { 0 };
char gPasswd[64]            = { 0 };
char gConfDir[kMAXPATHLEN]  = { 0 };


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


//--- Proofd routines ----------------------------------------------------------

//______________________________________________________________________________
void ProofdUser(const char *user)
{
   // Check user id. If user id is not equal to proofd's effective uid, user
   // will not be allowed access, unless effective uid = 0 (i.e. root).
   // Code almost identical to RootdUser().

   if (!*user)
      ErrorFatal("ProofdUser: bad user name");

   struct passwd *pw;
   if ((pw = getpwnam(user)) == 0)
      ErrorFatal("ProofdUser: user %s unknown", user);

   // If server is not started as root and user is not same as the
   // one who started proofd then authetication is not ok.
   uid_t uid = getuid();
   if (uid && uid != pw->pw_uid)
      ErrorFatal("ProofdUser: user not same as effective user of proofd");

   strcpy(gUser, user);

   NetSend(gAuth, kROOTD_AUTH);
}

//______________________________________________________________________________
void ProofdSRPUser(const char *user)
{
   // Use Secure Remote Password protocol.
   // Check user id in $HOME/.srootdpass file.
   // Code almost identical to RootdSRPUsr().

   if (!*user)
      ErrorFatal("ProofdSRPUser: bad user name");

   if (kSRootdPass[0]) { }  // remove compiler warning

#ifdef R__SRP

   char srootdpass[kMAXPATHLEN], srootdconf[kMAXPATHLEN];

   struct passwd *pw = getpwnam(user);
   if (!pw)
      ErrorFatal("ProofdSRPUser: user %s unknown", user);

   // If server is not started as root and user is not same as the
   // one who started proofd then authetication is not ok.
   uid_t uid = getuid();
   if (uid && uid != pw->pw_uid)
      ErrorFatal("ProofdSRPUser: user not same as effective user of proofd");

   NetSend(gAuth, kROOTD_AUTH);

   strcpy(gUser, user);

   sprintf(srootdpass, "%s/%s", pw->pw_dir, kSRootdPass);
   sprintf(srootdconf, "%s/%s.conf", pw->pw_dir, kSRootdPass);

   FILE *fp1 = fopen(srootdpass, "r");
   if (!fp1) {
      NetSend(2, kROOTD_AUTH);
      ErrorInfo("ProofdSRPUser: error opening %s", srootdpass);
      return;
   }
   FILE *fp2 = fopen(srootdconf, "r");
   if (!fp2) {
      NetSend(2, kROOTD_AUTH);
      ErrorInfo("ProofdSRPUser: error opening %s", srootdconf);
      if (fp1) fclose(fp1);
      return;
   }

   struct t_pw *tpw = t_openpw(fp1);
   if (!tpw) {
      NetSend(2, kROOTD_AUTH);
      ErrorInfo("ProofdSRPUser: unable to open password file %s", srootdpass);
      fclose(fp1);
      fclose(fp2);
      return;
   }

   struct t_conf *tcnf = t_openconf(fp2);
   if (!tcnf) {
      NetSend(2, kROOTD_AUTH);
      ErrorInfo("ProofdSRPUser: unable to open configuration file %s", srootdconf);
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
   if (!ts)
      ErrorFatal("ProofdSRPUser: user %s not found SRP password file", gUser);

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
      ErrorFatal("ProofdSRPUser: error receiving A from client");
   if (kind != kROOTD_SRPA)
      ErrorFatal("ProofdSRPUser: expected kROOTD_SRPA message");

   unsigned char buf[MAXPARAMLEN];
   struct t_num A;
   A.data = buf;
   A.len  = t_fromb64((char*)A.data, hexbuf);

   // send B to client
   NetSend(t_tob64(hexbuf, (char*)B->data, B->len), kROOTD_SRPB);

   t_servergetkey(ts, &A);

   // receive response from client
   if (NetRecv(hexbuf, MAXHEXPARAMLEN, kind) < 0)
      ErrorFatal("ProofdSRPUser: error receiving response from client");
   if (kind != kROOTD_SRPRESPONSE)
      ErrorFatal("ProofdSRPUser: expected kROOTD_SRPRESPONSE message");

   unsigned char cbuf[20];
   t_fromhex((char*)cbuf, hexbuf);

   if (!t_serververify(ts, cbuf)) {
      // authentication successful

      gAuth = 1;

      if (chdir(pw->pw_dir) == -1)
         ErrorFatal("ProofdSRPUser: can't change directory to %s", pw->pw_dir);

      if (getuid() == 0) {

         // set access control list from /etc/initgroup
         initgroups(gUser, pw->pw_gid);

         if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1)
            ErrorFatal("ProofdSRPUser: can't setgid for user %s", gUser);

         if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1)
            ErrorFatal("ProofdSRPUser: can't setuid for user %s", gUser);

      }

      char *home = new char[6+strlen(pw->pw_dir)];
      sprintf(home, "HOME=%s", pw->pw_dir);
      putenv(home);

      umask(022);

      NetSend(gAuth, kROOTD_AUTH);

      if (gDebug > 0)
         ErrorInfo("ProofdSRPUser: user %s authenticated", gUser);

   } else
      ErrorFatal("ProofdSRPUser: authentication failed for user %s", gUser);

   t_serverclose(ts);

#else
   NetSend(2, kROOTD_AUTH);
#endif
}

//______________________________________________________________________________
int ProofdCheckSpecialPass(const char *passwd)
{
   // Check user's password against password in $HOME/.rootdpass. If matches
   // skip other authentication mechanism. Returns 1 in case of success
   // authentication, 0 otherwise. Almost identical to RootdCheckSpecialPass().

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
      ErrorInfo("ProofdCheckSpecialPass: user %s authenticated via ~/.rootdpass", gUser);

   return 1;
}

//______________________________________________________________________________
void ProofdPass(const char *pass)
{
   // Check user's password, if ok, change to user's id and to user's directory.
   // Almost identical to RootdPass().

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
      ErrorFatal("ProofdPass: user needs to be specified first");

   int i;
   int n = strlen(pass);

   if (!n)
      ErrorFatal("ProofdPass: null passwd not allowed");

   if (n > (int)sizeof(passwd))
      ErrorFatal("ProofdPass: passwd too long");

   for (i = 0; i < n; i++)
      passwd[i] = ~pass[i];
   passwd[i] = '\0';

   pw = getpwnam(gUser);

   if (ProofdCheckSpecialPass(passwd))
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
      ErrorInfo("ProofdPass: AFS login failed for user %s: %s", gUser, reason);
      // try conventional login...
#endif

#ifdef R__SHADOWPW
   // System V Rel 4 style shadow passwords
   if ((spw = getspnam(gUser)) == 0) {
      ErrorInfo("ProofdPass: Shadow passwd not available for user %s", gUser);
      passw = pw->pw_passwd;
   } else
      passw = spw->sp_pwdp;
#else
   passw = pw->pw_passwd;
#endif
   pass_crypt = crypt(passwd, passw);
   n = strlen(passw);

   if (strncmp(pass_crypt, passw, n+1) != 0)
      ErrorFatal("ProofdPass: invalid password for user %s", gUser);

#ifdef R__AFS
   }  // afs_auth
#endif

skipauth:
   gAuth = 1;

   if (chdir(pw->pw_dir) == -1)
      ErrorFatal("ProofdPass: can't change directory to %s", pw->pw_dir);

   if (getuid() == 0) {

      // set access control list from /etc/initgroup
      initgroups(gUser, pw->pw_gid);

      if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1)
         ErrorFatal("ProofdPass: can't setgid for user %s", gUser);

      if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1)
         ErrorFatal("ProofdPass: can't setuid for user %s", gUser);

   }

   char *home = new char[6+strlen(pw->pw_dir)];
   sprintf(home, "HOME=%s", pw->pw_dir);
   putenv(home);

   umask(022);

   NetSend(gAuth, kROOTD_AUTH);

   if (gDebug > 0)
      ErrorInfo("ProofdPass: user %s authenticated", gUser);
}

//______________________________________________________________________________
void Authenticate()
{
   // Handle user authentication.

   const int kMaxBuf = 1024;
   char recvbuf[kMaxBuf];
   EMessageTypes kind;

   while (!gAuth) {
      if (NetRecv(recvbuf, kMaxBuf, kind) < 0)
         ErrorFatal("Authenticate: error receiving message");

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
         default:
            ErrorFatal("Authenticate: received bad opcode %d", kind);
      }
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

   char *argvv[4];
   char  arg0[256];
   char  msg[80];
   int   master;

#ifdef R__DEBUG
   int debug = 1;
   while (debug)
      ;
#endif

   if (gDebug > 0)
      ErrorInfo("ProofdExec: gConfDir = %s", gConfDir);

   // find out if we are supposed to be a master or a slave server
   if (NetRecv(msg, sizeof(msg)) < 0)
      ErrorFatal("Cannot receive master/slave status");

   master = !strcmp(msg, "master") ? 1 : 0;

   if (gDebug > 0)
      ErrorInfo("ProofdExec: master/slave = %s", msg);

   // user authentication
   Authenticate();

   // only reroute in case of master server
   const char *node_name;
   if (master && (node_name = RerouteUser()) != 0) {
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
      ErrorInfo("ProofdExec: send Okay");

   NetSend("Okay");

   // start server version
   sprintf(arg0, "%s/bin/proofserv", gConfDir);
   argvv[0] = arg0;
   argvv[1] = (char *)(master ? "proofserv" : "proofslave");
   argvv[2] = gConfDir;
   argvv[3] = 0;
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
      ErrorInfo("ProofdExec: execv(%s, %s, %s)", argvv[0], argvv[1], argvv[2]);

   if (!gInetdFlag) {
      // Duplicate the socket onto the descriptors 0, 1 and 2
      // and close the original socket descriptor (like inetd).
      dup2(gSockFd, 0);
      close(gSockFd);
      dup2(0, 1);
      dup2(0, 2);
   }

   execv(arg0, argvv);

   // tell client that exec failed
   sprintf(msg,
   "Cannot start PROOF server --- make sure %s exists!", arg0);
   NetSend(msg);
}

//______________________________________________________________________________
int main(int argc, char **argv)
{
   char *s;
   int   tcpwindowsize = 65535;

   ErrorInit(argv[0]);

   while (--argc > 0 && (*++argv)[0] == '-')
      for (s = argv[0]+1; *s != 0; s++)
         switch (*s) {
            case 'i':
               gInetdFlag = 1;
               break;

            case 'p':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-p requires a port number as argument\n");
                  ErrorFatal("-p requires a port number as argument");
               }
               gPort = atoi(*++argv);
               break;

            case 'd':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-d requires a debug level as argument\n");
                  ErrorFatal("-d requires a debug level as argument");
               }
               gDebug = atoi(*++argv);
               break;

            case 'b':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "-b requires a buffersize in bytes as argument\n");
                  ErrorFatal("-b requires a buffersize in bytes as argument");
               }
               tcpwindowsize = atoi(*++argv);
               break;

            default:
               if (!gInetdFlag)
                  fprintf(stderr, "unknown command line option: %c\n", *s);
               ErrorFatal("unknown command line option: %c", *s);
         }

   if (argc > 0) {
      strncpy(gConfDir, *argv, kMAXPATHLEN-1);
      gConfDir[kMAXPATHLEN-1] = 0;
   } else {
      if (!gInetdFlag)
         fprintf(stderr, "no config directory specified\n");
      ErrorFatal("no config directory specified");
   }

   if (!gInetdFlag) {

      // Start proofd up as a daemon process (in the background).
      // Also initialize the network connection - create the socket
      // and bind our well-know address to it.

      DaemonStart(1);

      NetInit(kProofdService, gPort, tcpwindowsize);
   }

   if (gDebug > 0)
      ErrorInfo("main: pid = %d, gInetdFlag = %d", getpid(), gInetdFlag);

   // Concurrent server loop.
   // The child created by NetOpen() handles the client's request.
   // The parent waits for another request. In the inetd case,
   // the parent from NetOpen() never returns.

   while (1) {
      if (NetOpen(gInetdFlag) == 0) {
         ProofdExec();     // child processes client's requests
         NetClose();       // then we are done
         exit(0);
      }

      // parent waits for another client to connect

   }
}
