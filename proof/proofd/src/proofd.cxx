// @(#)root/proofd:$Id$
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
// prootd -p 5252 $ROOTSYS                                              //
//                                                                      //
// Notice: no & is needed. Proofd will go in background by itself.      //
//                                                                      //
// Proofd arguments:                                                    //
//   -A [<rootauthrc>] Tells proofserv to read user's $HOME/.rootauthrc,//
//                     if any; by default such private file is ignored  //
//                     to allow complete control on the authentication  //
//                     directives to the cluster administrator, via the //
//                     system.rootauthrc file; if the optional argument //
//                     <rootauthrc> is given and points to a valid file,//
//                     this file takes the highest priority (private    //
//                     user's file being still read with next-to-highest//
//                     priority) providing a mean to use non-standard   //
//                     file names for authentication directives.        //
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
//   -G gridmapfile    defines the gridmap file to be used for globus   //
//                     authentication if different from globus default  //
//                     (/etc/grid-security/gridmap); (re)defines the    //
//                     GRIDMAP environment variable.                    //
//   -h                print usage message                              //
//   -i                says we were started by inetd                    //
//   -noauth           do not require client authentication             //
//   -p port#          specifies a different port to listen on          //
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
//   rootsys_dir       directory which must contain bin/proofserv and   //
//                     proof/etc/proof.conf. If not specified ROOTSYS   //
//                     or built-in (as specified to ./configure) is     //
//                     tried. (*MUST* be the last argument).            //
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
// 10: modified SSH protocol + support for server 'no authentication' mode
// 11: added support for openSSL keys for encryption
// 12: major authentication re-organization
// 13: support for SSH authentication via SSH tunnel
// 14: add env setup message

#include "RConfigure.h"
#include "RConfig.h"

#include <ctype.h>
#include <fcntl.h>
#include <pwd.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/param.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#include <sys/un.h>
#include "snprintf.h"

#if defined(__CYGWIN__) && defined(__GNUC__)
#   define cygwingcc
#endif
#ifdef __linux__
#define linux
#endif
#if defined(linux) || defined(__sun) || defined(__sgi) || \
    defined(_AIX) || defined(__FreeBSD__) || defined(__APPLE__) || \
    defined(__MACH__) || defined(cygwingcc) || defined(__OpenBSD__)
#include <grp.h>
#include <sys/types.h>
#include <signal.h>
#define ROOT_SIGNAL_INCLUDED
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
#endif
#endif

#include "proofdp.h"
extern "C" {
#include "rsadef.h"
#include "rsalib.h"
}

// General globals
int     gDebug                   = 0;

//--- Local Globals ---------------------------------------------------------

const int kMaxSlaves             = 32;

static std::string gAuthrc;
static std::string gConfDir;
static std::string gOpenHost;
static std::string gRootBinDir;
static std::string gRpdAuthTab;   // keeps track of authentication info
static std::string gTmpDir;
static std::string gUser;
static EService gService         = kPROOFD;
static int gProtocol             = 14;       // increase when protocol changes
static int gRemPid               = -1;      // remote process ID
static std::string gReadHomeAuthrc = "0";
static int gInetdFlag            = 0;
static int gMaster               =-1;
static int gRequireAuth          = 1;

using namespace ROOT;

//--- Error handlers -----------------------------------------------------------

//______________________________________________________________________________
void Err(int level, const char *msg, int size)
{
   Perror((char *)msg, size);
   if (level > -1) NetSend(level, kROOTD_ERR);
}
//______________________________________________________________________________
void ErrFatal(int level, const char *msg, int size)
{
   Perror((char *)msg, size);
   if (level > -1) NetSend(msg, kMESS_STRING);
   exit(1);
}
//______________________________________________________________________________
void ErrSys(int level, const char *msg, int size)
{
   Perror((char *)msg, size);
   ErrFatal(level, msg, size);
}

//--- Proofd routines ----------------------------------------------------------

#if defined(__sun)
//______________________________________________________________________________
extern "C" { void ProofdTerm(int)
{
   // Termination upon receipt of a SIGTERM or SIGINT.

   ErrorInfo("ProofdTerm: rootd.cxx: got a SIGTERM/SIGINT");
   // Terminate properly
   RpdAuthCleanup(0,0);
   // Close network connection
   NetClose();
   // exit
   exit(0);
}}
#else
//______________________________________________________________________________
static void ProofdTerm(int)
{
   // Termination upon receipt of a SIGTERM or SIGINT.

   ErrorInfo("ProofdTerm: rootd.cxx: got a SIGTERM/SIGINT");
   // Terminate properly
   RpdAuthCleanup(0,0);
   // Close network connection
   NetClose();
   // exit
   exit(0);
}
#endif

//______________________________________________________________________________
const char *RerouteUser()
{
   // Look if user should be rerouted to another server node.

   std::string conffile = "proof.conf";
   FILE *proofconf;

   if (getenv("HOME")) {
      conffile.insert(0,"/.");
      conffile.insert(0,getenv("HOME"));
      // string::insert is buggy on some compilers (eg gcc 2.96):
      // new length correct but data not always null terminated
      conffile[conffile.length()] = 0;
   }
   if (!(proofconf = fopen(conffile.c_str(), "r"))) {
      conffile = "/etc/";
      conffile.insert(0,gConfDir);
      // string::insert is buggy on some compilers (eg gcc 2.96):
      // new length correct but data not always null terminated
      conffile[conffile.length()] = 0;
   }
   if (proofconf || (proofconf = fopen(conffile.c_str(), "r")) != 0) {

      // read configuration file
      static char user_on_node[32];
      struct stat statbuf;
      char line[256];
      char node_name[kMaxSlaves][32];
      int  nnodes = 0;
      int  i;

      strncpy(user_on_node, "any", 32);
      user_on_node[31] = 0;

      while (fgets(line, sizeof(line), proofconf) != 0) {
         char word[4][64];
         if (line[0] == '#') continue;  // skip comment lines
         // coverity[secure_coding]
         int nword = sscanf(line, "%63s %63s %63s %63s",
                            word[0], word[1], word[2], word[3]);

         //
         // all available nodes must be configured by a line
         //    node <name>
         //
         if (nword >= 2 && strcmp(word[0], "node") == 0) {
            if (gethostbyname(word[1]) != 0) {
               if (nnodes < kMaxSlaves) {
                  if (strlen(word[1]) < 32) {
                     strncpy(node_name[nnodes], word[1], 32);
                     node_name[nnodes][31] = 0;
                  }
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
             strcmp(word[1], gUser.c_str()) == 0 &&
             strcmp(word[2], "on") == 0) {
            // user <name> on <node>
            if (strlen(word[3]) < 32) {
               strncpy(user_on_node, word[3], 32);
               user_on_node[31] = 0;
            }
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
      conffile = gConfDir + "/etc/next.node";
      proofconf = fopen(conffile.c_str(), "r");
      if (proofconf) {
         if (fstat(fileno(proofconf), &statbuf) == 0 &&
             difftime(time(0), statbuf.st_mtime) < 600) {
            if (fgets(line, sizeof(line), proofconf) != 0) {
               strncpy(user_on_node, line, 32);
               user_on_node[31] = 0;
               for (i = 0; i < nnodes; i++) {
                  if (strcmp(node_name[i], user_on_node) == 0) {
                     fclose(proofconf);
                     return user_on_node;
                  }
               }
            }
         }
         fclose(proofconf);
      }
   }
   return 0;
}

//______________________________________________________________________________
int RpdProofGetAuthSetup(char **abuf)
{
   // Receive buffer for final setup of authentication related stuff
   // This is base 64 string to decoded by proofserv, if needed
   int nrec = -1;

   if (RpdGetOffSet() > -1) {
      if ((nrec = RpdSecureRecv(abuf)) < 0) {
         ErrorInfo("RpdProofGetAuthSetup: sec: problems receiving buf");
         return -1;
      }
   } else {
      // No key: receive plainly
      EMessageTypes kind;
      char buflen[20];
      if (NetRecv(buflen, 20, kind) < 0) {
         ErrorInfo("RpdProofGetAuthSetup: plain: problems receiving buf length");
         return -1;
      }
      int len = atoi(buflen);

      // receive the buffer
      *abuf = new char[len + 1];
      if ((nrec = NetRecvRaw(*abuf, len)) < 0) {
         ErrorInfo("RpdProofGetAuthSetup: plain: problems receiving buf");
         delete[] *abuf;
         return -1;
      }
      (*abuf)[len] = 0;
   }

   if (gDebug > 1)
      ErrorInfo("RpdProofGetAuthSetup: proto: %d len: %d",
                RpdGetAuthProtocol(), nrec);

   return nrec;
}

//______________________________________________________________________________
void ProofdExec()
{
   // Authenticate the user and exec the proofserv program.
   // gConfdir is the location where the PROOF config files and binaries live.

   char *argvv[3];
   std::string arg0;
   std::string msg;

#ifdef R__DEBUG
   int debug = 1;
   while (debug)
      ;
#endif

   // Remote Host
   NetGetRemoteHost(gOpenHost);

   // Socket descriptor
   int sockFd = NetGetSockFd();

   if (gDebug > 0)
      ErrorInfo("ProofdExec: gOpenHost = %s", gOpenHost.c_str());

   if (gDebug > 0)
      ErrorInfo("ProofdExec: gConfDir = %s", gConfDir.c_str());

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
            if (strlen(inet_ntoa(*host_addr)) < 32) {
               strncpy(host_numb, inet_ntoa(*host_addr), 32);
               host_numb[31] = 0;
            }

            if ((node = gethostbyname(node_name)) != 0) {
               struct in_addr *node_addr = (struct in_addr*)(node->h_addr);
               char node_numb[32];
               strncpy(node_numb, inet_ntoa(*node_addr), 32);
               node_numb[31] = 0;
               //
               // compare the string representation of the IP addresses
               // to avoid possible problems with host name aliases
               //
               if (strcmp(host_numb, node_numb) != 0) {
                  msg = std::string("Reroute:").append(node_numb);
                  NetSend(msg.c_str());
                  exit(0);
               }
            }
         }
      }
   }

   // Receive buffer for final setup of authentication related stuff
   // This is base 64 string to decoded by proofserv, if needed
   if (RpdGetClientProtocol() > 12 && gRequireAuth == 1) {
      char *authbuff = 0;
      int lab = 0;
      if ((lab = RpdProofGetAuthSetup(&authbuff)) > 0) {
         // Save it in an environment variable
         char *rootproofauthsetup = new char[20 + strlen(authbuff)];
         memset(rootproofauthsetup, 0, 20 + strlen(authbuff));
         snprintf(rootproofauthsetup, 20 + strlen(authbuff), "ROOTPROOFAUTHSETUP=%s", authbuff);
         putenv(rootproofauthsetup);
      } else if (lab < 0) {
         ErrorInfo("ProofdExec: problems receiving auth buffer");
      }
      if (authbuff) delete[] authbuff;
   }

   if(RpdGetClientProtocol() >= 16) {
      void          *vb = 0;
      Int_t          len = 0;
      EMessageTypes  kind = kMESS_ANY;

      int rc = NetRecvAllocate(vb, len, kind);

      if (rc < 0) {
         ErrorInfo("ProofdExec: error receiving kPROOF_SETENV message");
         return;
      }

      if (kind != kPROOF_SETENV) {
         ErrorInfo("ProofdExec: expecting kPROOF_SETENV, got %d", kind);
         return;

      }

      char *buf = (char *) vb;
      char *end = buf + len;
      const char name[] = "PROOF_ALLVARS=";
      int alen = strlen(name)+len;
      char *all = new char[alen]; // strlen("PROOF_ALLVARS=") = 14
      strlcpy(all, name, alen);
      while (buf < end) {
         if (gDebug > 0) ErrorInfo("ProofdExec: setting: %s", buf);
         char *p = index(buf, '=');
         if (p) {
            if (buf != (char *) vb) strlcat(all, ",", alen); // skip the first one
            strlcat(all, buf, alen);
            putenv(buf);
         }
         buf += strlen(buf) + 1;
      }
      if (gDebug > 0) ErrorInfo("ProofdExec: setting: %s", all);
      putenv(all);
   }

   if (gDebug > 0)
      ErrorInfo("ProofdExec: send Okay (SockFd: %d)", sockFd);
   NetSend("Okay");

   // Find a free filedescriptor outside the standard I/O range
   if (sockFd == 0 || sockFd == 1 || sockFd == 2) {
      Int_t fd;
      struct stat stbuf;
      for (fd = 3; fd < NOFILE; fd++) {
         ResetErrno();
         if (fstat(fd, &stbuf) == -1 && GetErrno() == EBADF) {
            if (dup2(sockFd, fd) < 0)
               ErrorInfo("ProofdExec: problems executing 'dup2' (errno: %d)", errno);
            close(sockFd);
            sockFd = fd;
            close(2);
            close(1);
            close(0);
            RpdSetSysLogFlag(1);   //syslog only from here
            break;
         }
      }

      if (fd == NOFILE) {
         NetSend("Cannot start proofserver -- no free filedescriptor");
         return;
      }
   }

   //
   // Set environments vars for proofserv
   // Config dir
   char *rootconf = new char[13+gConfDir.length()];
   memset(rootconf, 0, 13 + gConfDir.length());
   snprintf(rootconf, 13 + gConfDir.length(), "ROOTCONFDIR=%s", gConfDir.c_str());
   putenv(rootconf);
   if (gDebug > 0)
      ErrorInfo("ProofdExec: setting: %s", rootconf);
   // Temp dir
   char *roottmp = new char[12+gTmpDir.length()];
   memset(roottmp, 0, 12 + gTmpDir.length());
   snprintf(roottmp, 12+gTmpDir.length(), "ROOTTMPDIR=%s", gTmpDir.c_str());
   putenv(roottmp);
   if (gDebug > 0)
      ErrorInfo("ProofdExec: setting: %s", roottmp);
   // User, host, rpid
   char *rootentity = new char[gUser.length()+gOpenHost.length()+33];
   memset(rootentity, 0, gUser.length()+gOpenHost.length()+33);
   snprintf(rootentity, gUser.length()+gOpenHost.length()+33,
           "ROOTENTITY=%s:%d@%s", gUser.c_str(), gRemPid, gOpenHost.c_str());
   putenv(rootentity);
   if (gDebug > 2)
      ErrorInfo("ProofdExec: setting: %s", rootentity);
   // Open socket
   char *rootopensock = new char[33];
   memset(rootopensock, 0, 33);
   snprintf(rootopensock, 33, "ROOTOPENSOCK=%d", sockFd);
   putenv(rootopensock);
   if (gDebug > 0)
      ErrorInfo("ProofdExec: setting: %s", rootopensock);
   // ReadHomeAuthrc option
   char *roothomeauthrc = new char[20];
   memset(roothomeauthrc, 0, 20);
   snprintf(roothomeauthrc, 20, "ROOTHOMEAUTHRC=%s", gReadHomeAuthrc.c_str());
   putenv(roothomeauthrc);
   if (gDebug > 0)
      ErrorInfo("ProofdExec: setting: %s", roothomeauthrc);

#ifdef R__GLBS
   // ID of shm with exported credentials
   char *shmidcred = new char[25];
   memset(shmidcred, 0, 25);
   snprintf(shmidcred, 25, "ROOTSHMIDCRED=%d", RpdGetShmIdCred());
   putenv(shmidcred);
   if (gDebug > 0)
      ErrorInfo("ProofdExec: setting: %s", shmidcred);
#endif

   // start server version
   arg0 = gRootBinDir + "/proofserv";
   argvv[0] = (char *)arg0.c_str();
   argvv[1] = (char *)(gMaster ? "proofserv" : "proofslave");
   argvv[2] = 0;

#ifndef ROOTPREFIX
   char *rootsys = new char[9+gConfDir.length()];
   memset(rootsys, 0, 9 + gConfDir.length());
   snprintf(rootsys, 9+gConfDir.length(), "ROOTSYS=%s", gConfDir.c_str());
   putenv(rootsys);
   if (gDebug > 0)
      ErrorInfo("ProofdExec: setting: %s", rootsys);
#endif
#ifndef ROOTLIBDIR
   char *oldpath, *ldpath;
#   if defined(__hpux) || defined(_HIUX_SOURCE)
   if ((oldpath = getenv("SHLIB_PATH")) && strlen(oldpath) > 0) {
      ldpath = new char[32+gConfDir.length()+strlen(oldpath)];
      memset(ldpath, 0, 32+gConfDir.length()+strlen(oldpath));
      snprintf(ldpath, 32+gConfDir.length()+strlen(oldpath),
                      "SHLIB_PATH=%s/lib:%s", gConfDir.c_str(), oldpath);
   } else {
      ldpath = new char[32+gConfDir.length()];
      memset(ldpath, 0, 32+gConfDir.length());
      snprintf(ldpath, 32+gConfDir.length(), "SHLIB_PATH=%s/lib", gConfDir.c_str());
   }
#   elif defined(_AIX)
   if ((oldpath = getenv("LIBPATH")) && strlen(oldpath) > 0) {
      ldpath = new char[32+gConfDir.length()+strlen(oldpath)];
      memset(ldpath, 0, 32+gConfDir.length()+strlen(oldpath));
      snprintf(ldpath, 32+gConfDir.length()+strlen(oldpath),
                       "LIBPATH=%s/lib:%s", gConfDir.c_str(), oldpath);
   } else {
      ldpath = new char[32+gConfDir.length()];
      memset(ldpath, 0, 32+gConfDir.length());
      snprintf(ldpath, 32+gConfDir.length(), "LIBPATH=%s/lib", gConfDir.c_str());
   }
#   elif defined(__APPLE__)
   if ((oldpath = getenv("DYLD_LIBRARY_PATH")) && strlen(oldpath) > 0) {
      ldpath = new char[32+gConfDir.length()+strlen(oldpath)];
      memset(ldpath, 0, 32+gConfDir.length()+strlen(oldpath));
      snprintf(ldpath, 32+gConfDir.length()+strlen(oldpath),
                      "DYLD_LIBRARY_PATH=%s/lib:%s", gConfDir.c_str(), oldpath);
   } else {
      ldpath = new char[32+gConfDir.length()];
      memset(ldpath, 0, 32+gConfDir.length());
      snprintf(ldpath, 32+gConfDir.length(), "DYLD_LIBRARY_PATH=%s/lib", gConfDir.c_str());
   }
#   else
   if ((oldpath = getenv("LD_LIBRARY_PATH")) && strlen(oldpath) > 0) {
      ldpath = new char[32+gConfDir.length()+strlen(oldpath)];
      memset(ldpath, 0, 32+gConfDir.length()+strlen(oldpath));
      snprintf(ldpath, 32+gConfDir.length()+strlen(oldpath),
                      "LD_LIBRARY_PATH=%s/lib:%s", gConfDir.c_str(), oldpath);
   } else {
      ldpath = new char[32+gConfDir.length()];
      memset(ldpath, 0, 32+gConfDir.length());
      snprintf(ldpath, 32+gConfDir.length(), "LD_LIBRARY_PATH=%s/lib", gConfDir.c_str());
   }
#   endif
   putenv(ldpath);
   if (gDebug > 0)
      ErrorInfo("ProofdExec: setting: %s", ldpath);
#endif

   // Check if a special file for authentication directives
   // has been given for later use in TAuthenticate; if yes,
   // set the corresponding environment variable
   char *authrc = 0;
   if (gAuthrc.length()) {
      if (gDebug > 0)
         ErrorInfo("ProofdExec: setting ROOTAUTHRC to %s",gAuthrc.c_str());
      authrc = new char[15+gAuthrc.length()];
      memset(authrc, 0, 15 + gAuthrc.length());
      snprintf(authrc, 15+gAuthrc.length(), "ROOTAUTHRC=%s", gAuthrc.c_str());
      putenv(authrc);
      if (gDebug > 0)
         ErrorInfo("ProofdExec: setting: %s", authrc);
   }

   // For backward compatibility
   char *keyfile = new char[15+strlen(RpdGetKeyRoot())];
   memset(keyfile, 0, 15+strlen(RpdGetKeyRoot()));
   snprintf(keyfile, 15+strlen(RpdGetKeyRoot()), "ROOTKEYFILE=%s",RpdGetKeyRoot());
   putenv(keyfile);
   if (gDebug > 2)
      ErrorInfo("ProofdExec: setting: %s", keyfile);

   if (gDebug > 0)
      ErrorInfo("ProofdExec: execv(%s, %s)", argvv[0], argvv[1]);

   // Start proofserv
   execv(arg0.c_str(), argvv);

   // tell client that exec failed
   msg = "Cannot start PROOF server --- make sure " + arg0 + " exists!";
   NetSend(msg.c_str());
}


//______________________________________________________________________________
void Usage(const char* name, int rc)
{
   fprintf(stderr, "\nUsage: %s [options] [rootsys-dir]\n", name);
   fprintf(stderr, "\nOptions:\n");
   fprintf(stderr, "\t-A [<rootauthrc>] Use $HOME/.rootauthrc or specified file\n");
   fprintf(stderr, "\t                  (see documentation)\n");
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
   int checkhostsequiv = 1;
   int tcpwindowsize  = 65535;
   int sshdport       = 22;
   int port1          = 0;
   int port2          = 0;
   int reuseallow     = 0x1F;
   int foregroundflag = 0;
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
   signal(SIGTERM, ProofdTerm);
   signal(SIGINT, ProofdTerm);

   while (--argc > 0 && (*++argv)[0] == '-')
      for (s = argv[0]+1; *s != 0; s++)
         switch (*s) {

            case 'A':
               gReadHomeAuthrc = std::string("1");
               // Next argument may be the name of a file with the
               // authentication directives to be used
               if((*(argv+1)) && (*(argv+1))[0] != '-') {
                  gAuthrc = std::string(*(argv+1));
                  struct stat st;
                  if (stat(gAuthrc.c_str(),&st) == -1 || !S_ISREG(st.st_mode)) {
                     // Not a regular file: discard it
                     gAuthrc.erase();
                  } else {
                     // Got a regular file as argument: go to next
                     argc--;
                     argv++;
                  }
               }
               break;

            case 'b':
               if (--argc <= 0) {
                  Error(ErrFatal,-1,"-b requires a buffersize in bytes as"
                                    " argument");
               }
               tcpwindowsize = atoi(*++argv);
               break;
#ifdef R__GLBS
            case 'C':
               if (--argc <= 0) {
                  Error(ErrFatal,-1,"-C requires a file name for the host"
                                    " certificates file location");
               }
               hostcertconf = std::string(*++argv);
               break;
#endif
            case 'd':
               if (--argc <= 0) {
                  Error(ErrFatal,-1,"-d requires a debug level as argument");
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
               Error(ErrFatal, kErrFatal,"Option '-E' is now dummy"
                          " - ignored (see proofd/src/proofd.cxx for"
                          " additional details)");
               break;

            case 'f':
               if (gInetdFlag) {
                  Error(ErrFatal,-1,"-i and -f options are incompatible");
               }
               foregroundflag = 1;
               break;
#ifdef R__GLBS
            case 'G':
               if (--argc <= 0) {
                  Error(ErrFatal,-1,"-G requires a file name for the gridmap"
                                    " file");
               }
               gridmap = std::string(*++argv);
               break;
#endif
            case 'h':
               Usage(progname, 0);
               break;

            case 'i':
               if (foregroundflag) {
                  Error(ErrFatal,-1,"-i and -f options are incompatible");
               }
               gInetdFlag = 1;
               break;

            case 'n':
               if (!strncmp(argv[0]+1,"noauth",6)) {
                  gRequireAuth = 0;
                  s += 5;
               }
               break;

            case 'p':
               if (--argc <= 0) {
                  Error(ErrFatal,-1,"-p requires a port number as argument");
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
                  Error(ErrFatal,-1,"-S requires a path to your keytab\n");
               }
               RpdSetKeytabFile((const char *)(*++argv));
               break;
#endif
            case 'T':
               if (--argc <= 0) {
                  Error(ErrFatal,kErrFatal,"-T requires a dir path for"
                                    " temporary files [/usr/tmp]");
               }
               gTmpDir = std::string(*++argv);
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
   if (!gTmpDir.length())
      gTmpDir = "/usr/tmp";
   if (access(gTmpDir.c_str(), W_OK) == -1)
      gTmpDir = "/tmp";

   if (argc > 0) {
      gConfDir = std::string(*argv);
   } else {
      // try to guess the config directory...
#ifndef ROOTDATADIR
      if (getenv("ROOTSYS")) {
         gConfDir = getenv("ROOTSYS");
         if (gDebug > 0)
            ErrorInfo("main: no config directory specified using ROOTSYS (%s)",
                      gConfDir.c_str());
      } else {
         Error(ErrFatal, -1, "main: no config directory specified");
      }
#else
      gConfDir = ROOTDATADIR;
#endif
   }
#ifdef ROOTBINDIR
   gRootBinDir= ROOTBINDIR;
#endif
#ifdef ROOTETCDIR
   rootetcdir= ROOTETCDIR;
#endif

   // Define gRootBinDir if not done already
   if (!gRootBinDir.length())
      gRootBinDir = std::string(gConfDir).append("/bin");

   // make sure it contains the executable we want to run
   std::string arg0 = std::string(gRootBinDir).append("/proofserv");
   if (access(arg0.c_str(), X_OK) == -1) {
      Error(ErrFatal,-1,"main: incorrect config directory specified (%s)",
                        gConfDir.c_str());
   }
   // Make it available to all the session via env
   if (gRootBinDir.length()) {
      char *tmp = new char[15 + gRootBinDir.length()];
      snprintf(tmp, 15 + gRootBinDir.length(), "ROOTBINDIR=%s", gRootBinDir.c_str());
      putenv(tmp);
   }

   // Define rootetcdir if not done already
   if (!rootetcdir.length())
      rootetcdir = std::string(gConfDir).append("/etc");
   // Make it available to all the session via env
   if (rootetcdir.length()) {
      char *tmp = new char[15 + rootetcdir.length()];
      snprintf(tmp, 15 + rootetcdir.length(), "ROOTETCDIR=%s", rootetcdir.c_str());
      putenv(tmp);
   }

   // If specified, set the special daemonrc file to be used
   if (daemonrc.length()) {
      char *tmp = new char[15+daemonrc.length()];
      snprintf(tmp, 15+daemonrc.length(), "ROOTDAEMONRC=%s", daemonrc.c_str());
      putenv(tmp);
   }
#ifdef R__GLBS
   // If specified, set the special gridmap file to be used
   if (gridmap.length()) {
      char *tmp = new char[15+gridmap.length()];
      snprintf(tmp, 15+gridmap.length(), "GRIDMAP=%s", gridmap.c_str());
      putenv(tmp);
   }
   // If specified, set the special hostcert.conf file to be used
   if (hostcertconf.length()) {
      char *tmp = new char[15+hostcertconf.length()];
      snprintf(tmp, 15+hostcertconf.length(), "ROOTHOSTCERT=%s", hostcertconf.c_str());
      putenv(tmp);
   }
#endif

   // Parent ID
   int proofdparentid = -1;      // Parent process ID
   if (!gInetdFlag)
      proofdparentid = getpid(); // Identifies this family
   else
      proofdparentid = getppid(); // Identifies this family

   // default job options
   unsigned int options = kDMN_RQAUTH | kDMN_HOSTEQ | kDMN_SYSLOG ;
   // modify them if required
   if (!gRequireAuth)
      options &= ~kDMN_RQAUTH;
   if (!checkhostsequiv)
      options &= ~kDMN_HOSTEQ;
   if (foregroundflag)
      options &= ~kDMN_SYSLOG;
   RpdInit(gService, proofdparentid, gProtocol, options,
           reuseallow, sshdport,
           gTmpDir.c_str(),altSRPpass.c_str(),2);

   // Generate Local RSA keys for the session
   if (RpdGenRSAKeys(0)) {
      Error(Err, -1, "proofd: unable to generate local RSA keys");
   }

   if (!gInetdFlag) {

      // Start proofd up as a daemon process (in the background).
      // Also initialize the network connection - create the socket
      // and bind our well-know address to it.

      if (!foregroundflag)
         DaemonStart(1, 0, gService);

      NetInit(gService, port1, port2, tcpwindowsize);
   }

   if (gDebug > 0)
      ErrorInfo("main: pid = %d, gInetdFlag = %d", getpid(), gInetdFlag);

   // Concurrent server loop.
   // The child created by NetOpen() handles the client's request.
   // The parent waits for another request. In the inetd case,
   // the parent from NetOpen() never returns.

   while (1) {
      if (NetOpen(gInetdFlag, gService) == 0) {

         // Init Session (get protocol, run authentication, login, ...)
         if ((gMaster = RpdInitSession(gService, gUser, gRemPid)) < 0) {
            if (gMaster == -1)
               Error(ErrFatal, -1, "proofd: failure initializing session");
            else if (gMaster == -2)
               // special session (eg. cleanup): just exit
               exit(0);
         }

         ProofdExec();     // child processes client's requests
         NetClose();       // then we are done
         exit(0);
      }

      // parent waits for another client to connect

   }

}
