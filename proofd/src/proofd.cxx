// @(#)root/proofd:$Name:  $:$Id: proofd.cxx,v 1.12 2000/12/13 12:08:00 rdm Exp $
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
//   -i              says we were started by inetd                      //
//   -p port#        specifies a different port to listen on            //
//   rootsys_dir     directory which must contain bin/proofserv and     //
//                   proof/etc/proof.conf                               //
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

#if defined(__FreeBSD__) && (__FreeBSD__ < 4)
#include <sys/file.h>
#define lockf(fd, op, sz)   flock((fd), (op))
#define	F_LOCK             (LOCK_EX | LOCK_NB)
#define	F_ULOCK             LOCK_UN
#endif

#if defined(linux) || defined(__sun) || defined(__sgi) || \
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
   int initgroups(const char *name, int basegid);
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
const char kProofdPass[]    = ".rootdpass";
const char kSProofdPass[]   = ".srootdpass";
const int  kMaxSlaves       = 32;
const int  kMAXPATHLEN      = 1024;

int  gInetdFlag             = 0;
int  gPort                  = 0;
int  gDebug                 = 0;
int  gSockFd                = -1;
int  gAuth                  = 0;
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

char *check_pass()
{
   // Check user's password, if ok, change to user's id and to user's directory.

   char   user_pass[64];
   char   new_user_pass[68];
   static char user_name[32];
   char   pass_word[32];
   char  *pass_crypt;
   char  *passw;
   struct passwd *pw;
#ifdef SHADOWPW
   struct spwd *spw;
#endif
   int    n, i;

   if ((n = NetRecv(new_user_pass, sizeof(new_user_pass))) < 0)
      ErrorFatal("Cannot receive authentication");

   for (i = 0; i < n-1; i++)
      user_pass[i] = ~new_user_pass[i];
   user_pass[i] = '\0';

   if (sscanf(user_pass, "%s %s", user_name, pass_word) != 2)
      ErrorFatal("Bad authentication record");

   if ((pw = getpwnam(user_name)) == 0)
      ErrorFatal("Passwd: User %s unknown", user_name);

#ifdef SHADOWPW
   // System V Rel 4 style shadow passwords
   if ((spw = getspnam(user_name)) == NULL)
      ErrorFatal("Passwd: User %s password unavailable", user_name);
   passw = spw->sp_pwdp;
#else
   passw = pw->pw_passwd;
#endif
   pass_crypt = crypt(pass_word, passw);
   n = strlen(passw);
#if 0
   // no passwd checking for time being.......... rdm
   if (strncmp(pass_crypt, passw, n+1) != 0)
      ErrorFatal("Passwd: Invalid password for user %s", user_name);
#endif

   // set access control list from /etc/initgroup
   initgroups(user_name, pw->pw_gid);

   if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1)
      ErrorFatal("Cannot setgid for user %s", user_name);

   if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1)
      ErrorFatal("Cannot setuid for user %s", user_name);

   if (chdir(pw->pw_dir) == -1)
      ErrorFatal("Cannot change directory to %s", pw->pw_dir);

   char *home = new char[6+strlen(pw->pw_dir)];
   sprintf(home, "HOME=%s", pw->pw_dir);
   putenv(home);

   return user_name;
}

char *reroute_user(const char *user_name)
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
            struct hostent *hp;

            if ((hp = gethostbyname(word[1])) != 0) {
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
             strcmp(word[1], user_name) == 0 && strcmp(word[2], "on") == 0) {
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

   char *argvv[5];
   char  arg0[256], arg1[32];
   char *user_name;
   char *node_name;
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
   user_name = check_pass();

   // only reroute in case of master server
   if (master && (node_name = reroute_user(user_name)) != 0) {
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
   sprintf(arg1, "%d", gSockFd);
   argvv[0] = arg0;
   argvv[1] = arg1;
   argvv[2] = (char *)(master ? "proofserv" : "proofslave");
   argvv[3] = gConfDir;
   argvv[4] = 0;
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
      ErrorInfo("ProofdExec: execv(%s, %s, %s, %s)", argvv[0], argvv[1],
                argvv[2], argvv[3]);

   execv(arg0, argvv);

   // tell client that exec failed
   sprintf(msg,
   "Cannot start PROOF server --- make sure %s exists!", arg0);
   NetSend(msg);
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
                  ErrorFatal("-p requires a port number as argument");
               gPort = atoi(*++argv);
               break;

            case 'd':
               if (--argc <= 0)
                  gDebug = 0;
               else
                  gDebug = atoi(*++argv);
               break;

            default:
               ErrorFatal("unknown command line option: %s", *s);
         }

   if (argc > 0) {
      strncpy(gConfDir, *argv, kMAXPATHLEN-1);
      gConfDir[kMAXPATHLEN-1] = 0;
   } else
      ErrorFatal("no config directory specified");

   if (!gInetdFlag) {

      // Start proofd up as a daemon process (in the background).
      // Also initialize the network connection - create the socket
      // and bind our well-know address to it.

      DaemonStart(1);

      NetInit(kProofdService, gPort);
   }

   if (gDebug > 0)
      ErrorInfo("main: pid = %d, gInetdFlag = %d", getpid(), gInetdFlag);

   // Concurrent server loop.
   // The child created by NetOpen() handles the client's request.
   // The parent waits for another request. In the inetd case,
   // the parent from NetOpen() never returns.

   while (1) {
      if ((childpid = NetOpen(gInetdFlag)) == 0) {
         ProofdExec();     // child processes client's requests
         NetClose();       // then we are done
         exit(0);
      }

      // parent waits for another client to connect

   }

   // not reached
   return 0;
}
