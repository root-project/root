// @(#)root/proofd:$Name:  $:$Id: proofd.cxx,v 1.58 2004/02/19 00:11:19 rdm Exp $
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
//      server          = /usr/local/root/bin/proofd                    //
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
// 8: change in Kerberos authentication protocol
// 9: change authentication cleaning protocol


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
#endif

#include "proofdp.h"

#ifdef R__KRB5
#include "Krb5Auth.h"
namespace ROOT {
   extern krb5_keytab  gKeytab; // to allow specifying on the command line
   extern krb5_context gKcontext;
}
#endif

#ifdef R__GLBS
namespace ROOT {
   extern int gShmIdCred;  // global, to pass the shm ID to proofserv
}
#endif

//--- Globals ------------------------------------------------------------------

const int kMaxSlaves             = 32;

char    gFilePA[40]              = { 0 };

char    gConfDir[kMAXPATHLEN]    = { 0 };
int     gDebug                   = 0;
int     gForegroundFlag          = 0;
int     gInetdFlag               = 0;
int     gMaster                  =-1;
int     gProtocol                = 9;       // increase when protocol changes
char    gRcFile[kMAXPATHLEN]     = { 0 };
int     gRootLog                 = 0;
char    gRpdAuthTab[kMAXPATHLEN] = { 0 };   // keeps track of authentication info

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
const char *RerouteUser()
{
   // Look if user should be rerouted to another server node.

   char conffile[1024];
   FILE *proofconf;

   conffile[0] = 0;
   if (getenv("HOME")) {
      sprintf(conffile, "%s/.proof.conf", getenv("HOME"));
      if (access(conffile, R_OK))
         conffile[0] = 0;
   }
   if (conffile[0] == 0)
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

   if (gDebug > 0)
      ErrorInfo("ProofdExec: gOpenHost = %s", gOpenHost);

   if (gDebug > 0)
      ErrorInfo("ProofdExec: gConfDir = %s", gConfDir);
#if 0
   // Login
   ProofdLogin();
#endif

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
   argvv[3] = gTmpDir;
   argvv[4] = gOpenHost;
   sprintf(rpid, "%d", gRemPid);
   argvv[5] = rpid;
   argvv[6] = gUser;
#ifdef R__GLBS
   argvv[7] = cShmIdCred;
   argvv[8] = 0;
   argvv[9] = 0;
   argvv[10] = 0;
   if (getenv("X509_CERT_DIR"))  argvv[8] = strdup(getenv("X509_CERT_DIR"));
   if (getenv("X509_USER_CERT")) argvv[9] = strdup(getenv("X509_USER_CERT"));
   if (getenv("X509_USER_KEY"))  argvv[10] = strdup(getenv("X509_USER_KEY"));
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
   char *ldpath;
#   if defined(__hpux) || defined(_HIUX_SOURCE)
   if (getenv("SHLIB_PATH")) {
      ldpath = new char[32+strlen(gConfDir)+strlen(getenv("SHLIB_PATH"))];
      sprintf(ldpath, "SHLIB_PATH=%s/lib:%s", gConfDir, getenv("SHLIB_PATH"));
   } else {
      ldpath = new char[32+strlen(gConfDir)];
      sprintf(ldpath, "SHLIB_PATH=%s/lib", gConfDir);
   }
#   elif defined(_AIX)
   if (getenv("LIBPATH")) {
      ldpath = new char[32+strlen(gConfDir)+strlen(getenv("LIBPATH"))];
      sprintf(ldpath, "LIBPATH=%s/lib:%s", gConfDir, getenv("LIBPATH"));
   } else {
      ldpath = new char[32+strlen(gConfDir)];
      sprintf(ldpath, "LIBPATH=%s/lib", gConfDir);
   }
#   else
   if (getenv("LD_LIBRARY_PATH")) {
      ldpath = new char[32+strlen(gConfDir)+strlen(getenv("LD_LIBRARY_PATH"))];
      sprintf(ldpath, "LD_LIBRARY_PATH=%s/lib:%s", gConfDir, getenv("LD_LIBRARY_PATH"));
   } else {
      ldpath = new char[32+strlen(gConfDir)];
      sprintf(ldpath, "LD_LIBRARY_PATH=%s/lib", gConfDir);
   }
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
   if (retval) {
      fprintf(stderr, "%s while initializing krb5\n",
            error_message(retval));
      Error(ErrFatal, -1, "%s while initializing krb5",
            error_message(retval));
   }
#endif

#ifdef R__GLBS
   char GridMap[kMAXPATHLEN] = { 0 };
#endif

   // Define service
   strcpy(gService, "proofd");

   // Set Server Protocol
   gServerProtocol = gProtocol;

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
                     fprintf(stderr,"-p requires a port number as argument\n");
                  Error(ErrFatal,-1,"-p requires a port number as argument");
               }
               char *p;
               gPortA = strtol(*++argv, &p, 10);
               if (*p == '-')
                  gPortB = strtol(++p, &p, 10);
               else if (*p == '\0')
                  gPortB = gPortA;
               if (*p != '\0' || gPortB < gPortA || gPortB < 0) {
                  if (!gInetdFlag)
                     fprintf(stderr, "invalid port number or range: %s\n",
                                     *argv);
                  Error(ErrFatal,kErrFatal,"invalid port number or range: %s",
                                     *argv);
               }
               break;

            case 'd':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr,"-d requires a debug level as argument\n");
                  Error(ErrFatal,-1,"-d requires a debug level as argument");
               }
               gDebug = atoi(*++argv);
               break;

            case 'b':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr,"-b requires a buffersize in bytes as"
                                    " argument\n");
                  Error(ErrFatal,-1,"-b requires a buffersize in bytes as"
                                    " argument");
               }
               tcpwindowsize = atoi(*++argv);
               break;

            case 'T':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr,"-T requires a dir path for temporary"
                                    " files [/usr/tmp]\n");
                  Error(ErrFatal,kErrFatal,"-T requires a dir path for"
                                    " temporary files [/usr/tmp]");
               }
               sprintf(gTmpDir, "%s", *++argv);
               break;

            case 's':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr,"-s requires as argument a port number"
                                    " for the sshd daemon\n");
                  Error(ErrFatal,kErrFatal,"-s requires as argument a port"
                                    " number for the sshd daemon");
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
                     fprintf(stderr,"-G requires a file name for the gridmap"
                                    " file\n");
                  Error(ErrFatal,-1,"-G requires a file name for the gridmap"
                                    " file");
               }
               sprintf(GridMap, "%s", *++argv);
               if (setenv("GRIDMAP",GridMap,1) ){
                  Error(ErrFatal,-1,"%s while setting the GRIDMAP environment"
                                    " variable");
               }
               break;

            case 'C':
               if (--argc <= 0) {
                  if (!gInetdFlag)
                     fprintf(stderr,"-C requires a file name for the host"
                                    " certificates file location\n");
                  Error(ErrFatal,-1,"-C requires a file name for the host"
                                    " certificates file location");
               }
               sprintf(gHostCertConf, "%s", *++argv);
               break;
#endif
            default:
               if (!gInetdFlag)
                  fprintf(stderr, "unknown command line option: %c\n", *s);
               Error(ErrFatal, -1, "unknown command line option: %c", *s);
         }

   // dir for temporary files
   if (strlen(gTmpDir) == 0) {
      strcpy(gTmpDir, "/usr/tmp");
      if (access(gTmpDir, W_OK) == -1) {
         strcpy(gTmpDir, "/tmp");
      }
   }

   // authentication tab file
   sprintf(gRpdAuthTab, "%s/rpdauthtab", gTmpDir);

   // Set auth tab flag in RPDUtil ...
   RpdSetAuthTabFile(gRpdAuthTab);

   if (argc > 0) {
      strncpy(gConfDir, *argv, kMAXPATHLEN-1);
      gConfDir[kMAXPATHLEN-1] = 0;
      sprintf(gExecDir, "%s/bin", gConfDir);
      sprintf(gSystemDaemonRc, "%s/etc/system%s", gConfDir, kDaemonRc);
   } else {
      // try to guess the config directory...
#ifndef ROOTPREFIX
      if (getenv("ROOTSYS")) {
         strcpy(gConfDir, getenv("ROOTSYS"));
         sprintf(gExecDir, "%s/bin", gConfDir);
         sprintf(gSystemDaemonRc, "%s/etc/system%s", gConfDir, kDaemonRc);
         if (gDebug > 0)
            ErrorInfo("main: no config directory specified using ROOTSYS (%s)",
                      gConfDir);
      } else {
         if (!gInetdFlag)
            fprintf(stderr, "proofd: no config directory specified\n");
         Error(ErrFatal, -1, "main: no config directory specified");
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

   // make sure needed files exist
   char arg0[256];
   sprintf(arg0, "%s/bin/proofserv", gConfDir);
   if (access(arg0, X_OK) == -1) {
      if (!gInetdFlag)
         fprintf(stderr,"proofd: incorrect config directory specified (%s)\n",
                        gConfDir);
      Error(ErrFatal,-1,"main: incorrect config directory specified (%s)",
                        gConfDir);
   }

   // Log to stderr if not started as daemon ...
   if (gForegroundFlag) RpdSetRootLogFlag(1);

   if (!gInetdFlag) {

      // Start proofd up as a daemon process (in the background).
      // Also initialize the network connection - create the socket
      // and bind our well-know address to it.

      if (!gForegroundFlag) DaemonStart(1, 0, kPROOFD);

      NetInit(gService, gPortA, gPortB, tcpwindowsize);
   }

   if (gDebug > 0)
      ErrorInfo("main: pid = %d, gInetdFlag = %d", getpid(), gInetdFlag);

   // Concurrent server loop.
   // The child created by NetOpen() handles the client's request.
   // The parent waits for another request. In the inetd case,
   // the parent from NetOpen() never returns.

   while (1) {
      if (NetOpen(gInetdFlag, kPROOFD) == 0) {

         // Init Session (get protocol, run authentication, login, ...)
         gMaster = RpdInitSession(gDebug,kPROOFD);

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
