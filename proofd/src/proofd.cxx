// @(#)root/proofd:$Name:  $:$Id: proofd.cxx,v 1.1.1.1 2000/05/16 17:00:48 rdm Exp $
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
// Proof, Parallel ROOT Facility, front-end daemon.                     //
// This small server is started by inetd when a client requests         //
// a connection to a Proof server. If we don't want the Proof server    //
// to run on this specific node, e.g. because the system is being       //
// shutdown or there are already too many servers running, we send      //
// the client a re-route message and close the connection. Otherwise    //
// we receive the client's version key and exec the appropriate         //
// server version.                                                      //
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

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#if defined(__linux) || defined(__linux__) || defined(__sun) || defined(__sgi) || defined(_AIX)
#include <grp.h>
#include <sys/types.h>
#endif

#if defined(__sun) || defined(R__GLIBC)
#include <crypt.h>
#endif

#if defined(__osf__) || defined(__sgi)
extern "C" char *crypt(const char *, const char *);
#endif

#ifdef __alpha
extern "C" int initgroups(char *name, int basegid);
#endif

#if defined(__sgi) && !defined(__GNUG__) && (!defined(SGI_REL) || (SGI_REL<62))
extern "C" {
   int seteuid(int euid);
   int setegid(int egid);
}
#endif

#if defined(_AIX)
extern "C" {
   int initgroups(char *name, int basegid);
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

#include "MessageTypes.h"


const int kMaxSlaves = 32;

static int sockin  = 0;
static int sockout = 1;

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


void Send(char *msg)
{
   // Simulate TSocket::Send(const char *str).

   int hdr[2];
   int hlen = sizeof(kMESS_STRING) + strlen(msg)+1;  // including \0
   hdr[0] = htonl(hlen);
   hdr[1] = htonl(kMESS_STRING);
   if (send(sockout, (const char *)hdr, sizeof(hdr), 0) != sizeof(hdr))
      exit(1);
   hlen -= sizeof(kMESS_STRING);
   if (send(sockout, msg, hlen, 0) != hlen)
      exit(1);
}

int Recv(char *msg, int max)
{
   // Simulate TSocket::Recv(char *str, int max).

   int n, hdr[2];

   if ((n = recv(sockin, (char *)hdr, sizeof(hdr), 0)) < 0)
      return -1;

   int hlen = ntohl(hdr[0]) - sizeof(kMESS_STRING);
   if (hlen > max) hlen = max;
   if ((n = recv(sockin, msg, hlen, 0)) < 0)
      return -1;

   return hlen;
}

void fatal_error(char *msg)
{
   Send(msg);
   exit(1);
}

char *check_pass()
{
   // Check user's password, if ok, change to user's id and to user's directory.

   char   user_pass[64];
   char   new_user_pass[68];
   static char user_name[32];
   char   pass_word[32];
   char  *pass_crypt;
   char  *passw;
   char   msg[80];
   struct passwd *pw;
#ifdef SHADOWPW
   struct spwd *spw;
#endif
   int    n, i;

   if ((n = Recv(new_user_pass, sizeof(new_user_pass))) < 0) {
      fatal_error("Cannot receive authentication");
   }

   for (i = 0; i < n-1; i++)
      user_pass[i] = ~new_user_pass[i];
   user_pass[i] = '\0';

   if (sscanf(user_pass, "%s %s", user_name, pass_word) != 2) {
      fatal_error("Bad authentication record");
   }

   if ((pw = getpwnam(user_name)) == 0) {
      sprintf(msg, "Passwd: User %s unknown", user_name);
      fatal_error(msg);
   }
#ifdef SHADOWPW
   // System V Rel 4 style shadow passwords
   if ((spw = getspnam(user_name)) == NULL) {
      sprintf(msg, "Passwd: User %s password unavailable", user_name);
      fatal_error(msg);
   }
   passw = spw->sp_pwdp;
#else
   passw = pw->pw_passwd;
#endif
   pass_crypt = crypt(pass_word, passw);
   n = strlen(passw);
#if 0
   // no passwd checking for time being.......... rdm
   if (strncmp(pass_crypt, passw, n+1) != 0) {
      sprintf(msg, "Passwd: Invalid password for user %s", user_name);
      fatal_error(msg);
   }
#endif

   // set access control list from /etc/initgroup
   initgroups(user_name, pw->pw_gid);

   if (setresgid(pw->pw_gid, pw->pw_gid, 0) == -1) {
      sprintf(msg, "Cannot setgid for user %s", user_name);
      fatal_error(msg);
   }

   if (setresuid(pw->pw_uid, pw->pw_uid, 0) == -1) {
      sprintf(msg, "Cannot setuid for user %s", user_name);
      fatal_error(msg);
   }


   if (chdir(pw->pw_dir) == -1) {
      sprintf(msg, "Cannot change directory to %s", pw->pw_dir);
      fatal_error(msg);
   }
   return user_name;
}

char *reroute_user(char *confdir, char *user_name)
{
   // Look if user should be rerouted to another server node.

   char conffile[256];
   FILE *proofconf;

   sprintf(conffile, "%s/etc/proof.conf", confdir);
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
         // all running nodes must be configured by a line
         //    node <name>
         //
         if (nword >= 2 && strcmp(word[0], "node") == 0) {
            struct hostent *hp;

            if ((hp = gethostbyname(word[1])) != 0) {
               strcpy(node_name[nnodes], word[1]);
               nnodes++;
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
      sprintf(conffile, "%s/etc/next.node", confdir);
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

int main(int /* argc */, char **argv)
{
   // Arguments for master:  "proofserv" confdir
   // Arguments for slave :  "proofslave" confdir
   char *argvv[4];
   char  arg0[256];
   char *user_name;
   char *node_name;
   char  vtag[80];
   char  msg[80];


   //
   // Make this process the process group leader and disassociate from
   // control terminal - fork is executed to ensure a unique process id
   // and to make sure our process already isn't a process group leader
   // in which case the call to setsid would fail
   //
   if (fork() != 0) exit(0);   // parent exits
   setsid();

   // user authentication
   user_name = check_pass();

   // only reroute in case of master server
   if (!strcmp("proofserv", argv[0]) &&
       (node_name = reroute_user(argv[1], user_name)) != 0) {
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
                  Send(msg);
                  exit(0);
               }
            }
         }
      }
   }
   Send("Okay");

   // receive version tag
   if (Recv(vtag, sizeof(vtag)) < 0)
      fatal_error("Error receiving version tag");

   // start server version
   sprintf(arg0, "%s/bin/proofserv.%s", argv[1], vtag);
   argvv[0] = arg0;
   argvv[1] = argv[0];
   argvv[2] = argv[1];
   argvv[3] = 0;
#if defined(__linux)
   sprintf(msg, "LD_LIBRARY_PATH=%s/lib", argv[1]);
   putenv(msg);
#endif
   execv(arg0, argvv);

   // tell client that exec failed
   sprintf(msg,
   "Cannot start Proof server version %s --- update your ROOT version!", vtag);
   Send(msg);

   return 0;
}
