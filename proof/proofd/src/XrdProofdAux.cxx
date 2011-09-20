// @(#)root/proofd:$Id$
// Author: G. Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdAux                                                          //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// Small auxilliary classes used in XrdProof                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "XrdProofdPlatform.h"

#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPriv.hh"
#ifdef OLDXRDOUC
#  include "XrdOuc/XrdOucError.hh"
#  include "XrdOuc/XrdOucLogger.hh"
#else
#  include "XrdSys/XrdSysError.hh"
#  include "XrdSys/XrdSysLogger.hh"
#endif

#include "XrdProofdAux.h"
#include "XrdProofdConfig.h"
#include "XrdProofdProtocol.h"

// Tracing
#include "XrdProofdTrace.h"

// Local definitions
#ifdef XPD_MAXLEN
#undefine XPD_MAXLEN
#endif
#define XPD_MAXLEN 1024

XrdSysRecMutex XrdProofdAux::fgFormMutex;

//______________________________________________________________________________
const char *XrdProofdAux::AdminMsgType(int type)
{
   // Translates the admin message type in a human readable string.
   // Must be consistent with the values in XProofProtocol.h

   static const char *msgtypes[] = { "Undef",
     "QuerySessions", "SessionTag", "SessionAlias", "GetWorkers", "QueryWorkers",
     "CleanupSessions", "QueryLogPaths", "ReadBuffer", "QueryROOTVersions",
     "ROOTVersion", "GroupProperties", "SendMsgToUser", "ReleaseWorker",
     "Exec", "GetFile", "PutFile", "CpFile"};

   if (type < 1000 || type >= kUndef) {
      return msgtypes[0];
   } else {
      int t = type - 999;
      return msgtypes[t];
   }
}

//______________________________________________________________________________
const char *XrdProofdAux::ProofRequestTypes(int type)
{
   // Translates the proof request type in a human readable string.
   // Must be consistent with the values in XProofProtocol.h.
   // The reserved ones are for un

   static const char *reqtypes[] = { "Undef",
      "XP_login", "XP_auth", "XP_create", "XP_destroy", "XP_attach", "XP_detach",
      "XP_3107", "XP_3108", "XP_3109", "XP_3110",
      "XP_urgent", "XP_sendmsg", "XP_admin", "XP_interrupt", "XP_ping",
      "XP_cleanup", "XP_readbuf", "XP_touch", "XP_ctrlc", "XR_direct" };

   if (type < 3101 || type >= kXP_Undef) {
      return reqtypes[0];
   } else {
      int t = type - 3100;
      return reqtypes[t];
   }
}

//______________________________________________________________________________
char *XrdProofdAux::Expand(char *p)
{
   // Expand path 'p' relative to:
   //     $HOME               if begins with ~/
   //     <user>'s $HOME      if begins with ~<user>/
   //     $PWD                if does not begin with '/' or '~'
   //   getenv(<ENVVAR>)      if it begins with $<ENVVAR>)
   // The returned array of chars is the result of reallocation
   // of the input one.
   // If something is inconsistent, for example <ENVVAR> does not
   // exists, the original string is untouched

   // Make sure there soething to expand
   if (!p || strlen(p) <= 0 || p[0] == '/')
      return p;

   char *po = p;

   // Relative to the environment variable
   if (p[0] == '$') {
      // Resolve env
      XrdOucString env(&p[1]);
      int isl = env.find('/');
      env.erase(isl);
      char *p1 = (isl > 0) ? (char *)(p + isl + 2) : 0;
      if (getenv(env.c_str())) {
         int lenv = strlen(getenv(env.c_str()));
         int lp1 = p1 ? strlen(p1) : 0;
         po = (char *) malloc(lp1 + lenv + 2);
         if (po) {
            memcpy(po, getenv(env.c_str()), lenv);
            if (p1) {
               memcpy(po+lenv+1, p1, lp1);
               po[lenv] = '/';
            }
            po[lp1 + lenv + 1] = 0;
            free(p);
         } else
            po = p;
      }
      return po;
   }

   // Relative to the local location
   if (p[0] != '~') {
      if (getenv("PWD")) {
         int lpwd = strlen(getenv("PWD"));
         int lp = strlen(p);
         po = (char *) malloc(lp + lpwd + 2);
         if (po) {
            memcpy(po, getenv("PWD"), lpwd);
            memcpy(po+lpwd+1, p, lp);
            po[lpwd] = '/';
            po[lpwd+lp+1] = 0;
            free(p);
         } else
            po = p;
      }
      return po;
   }

   // Relative to $HOME or <user>'s $HOME
   if (p[0] == '~') {
      char *pu = p+1;
      char *pd = strchr(pu,'/');
      *pd++ = '\0';
      // Get the correct user structure
      XrdProofUI ui;
      int rc = 0;
      if (strlen(pu) > 0) {
         rc = XrdProofdAux::GetUserInfo(pu, ui);
      } else {
         rc = XrdProofdAux::GetUserInfo(getuid(), ui);
      }
      if (rc == 0) {
         int ldir = ui.fHomeDir.length();
         int lpd = strlen(pd);
         po = (char *) malloc(lpd + ldir + 2);
         if (po) {
            memcpy(po, ui.fHomeDir.c_str(), ldir);
            memcpy(po+ldir+1, pd, lpd);
            po[ldir] = '/';
            po[lpd + ldir + 1] = 0;
            free(p);
         } else
            po = p;
      }
      return po;
   }

   // We are done
   return po;
}

//______________________________________________________________________________
void XrdProofdAux::Expand(XrdOucString &p)
{
   // Expand path 'p' relative to:
   //     $HOME               if begins with ~/
   //     <user>'s $HOME      if begins with ~<user>/
   //     $PWD                if does not begin with '/' or '~'
   //   getenv(<ENVVAR>)      if it begins with $<ENVVAR>)
   // The input string is updated with the result.
   // If something is inconsistent, for example <ENVVAR> does not
   // exists, the original string is untouched

   char *po = strdup((char *)p.c_str());
   po = Expand(po);
   p = po;
   SafeFree(po);
}

//__________________________________________________________________________
long int XrdProofdAux::GetLong(char *str)
{
   // Extract first integer from string at 'str', if any

   // Reposition on first digit
   char *p = str;
   while ((*p < 48 || *p > 57) && (*p) != '\0')
      p++;
   if (*p == '\0')
      return LONG_MAX;

   // Find the last digit
   int j = 0;
   while (*(p+j) >= 48 && *(p+j) <= 57)
      j++;
   *(p+j) = '\0';

   // Convert now
   return strtol(p, 0, 10);
}

//__________________________________________________________________________
int XrdProofdAux::GetGroupInfo(const char *grp, XrdProofGI &gi)
{
   // Get information about group with 'gid' in a thread safe way.
   // Retur 0 on success, -errno on error

   // Make sure input is defined
   if (!grp || strlen(grp) <= 0)
      return -EINVAL;

   // Call getgrgid_r ...
   struct group gr;
   struct group *pgr = 0;
   char buf[2048];
#if defined(__sun) && !defined(__GNUC__)
   pgr = getgrnam_r(grp, &gr, buf, sizeof(buf));
#else
   getgrnam_r(grp, &gr, buf, sizeof(buf), &pgr);
#endif
   if (pgr) {
      // Fill output
      gi.fGroup = grp;
      gi.fGid = (int) gr.gr_gid;
      // Done
      return 0;
   }

   // Failure
   if (errno != 0)
      return ((int) -errno);
   else
      return -ENOENT;
}

//__________________________________________________________________________
int XrdProofdAux::GetGroupInfo(int gid, XrdProofGI &gi)
{
   // Get information about group with 'gid' in a thread safe way.
   // Retur 0 on success, -errno on error

   // Make sure input make sense
   if (gid <= 0)
      return -EINVAL;

   // Call getgrgid_r ...
   struct group gr;
   struct group *pgr = 0;
   char buf[2048];
#if defined(__sun) && !defined(__GNUC__)
   pgr = getgrgid_r((gid_t)gid, &gr, buf, sizeof(buf));
#else
   getgrgid_r((gid_t)gid, &gr, buf, sizeof(buf), &pgr);
#endif
   if (pgr) {
      // Fill output
      gi.fGroup = gr.gr_name;
      gi.fGid = gid;
      // Done
      return 0;
   }

   // Failure
   if (errno != 0)
      return ((int) -errno);
   else
      return -ENOENT;
}

//__________________________________________________________________________
int XrdProofdAux::GetUserInfo(const char *usr, XrdProofUI &ui)
{
   // Get information about user 'usr' in a thread safe way.
   // Return 0 on success, -errno on error

   // Make sure input is defined
   if (!usr || strlen(usr) <= 0)
      return -EINVAL;

   // Call getpwnam_r ...
   struct passwd pw;
   struct passwd *ppw = 0;
   char buf[2048];
#if defined(__sun) && !defined(__GNUC__)
   ppw = getpwnam_r(usr, &pw, buf, sizeof(buf));
#else
   getpwnam_r(usr, &pw, buf, sizeof(buf), &ppw);
#endif
   if (ppw) {
      // Fill output
      ui.fUid = (int) pw.pw_uid;
      ui.fGid = (int) pw.pw_gid;
      ui.fHomeDir = pw.pw_dir;
      ui.fUser = usr;
      // Done
      return 0;
   }

   // Failure
   if (errno != 0)
      return ((int) -errno);
   else
      return -ENOENT;
}

//__________________________________________________________________________
int XrdProofdAux::GetUserInfo(int uid, XrdProofUI &ui)
{
   // Get information about user with 'uid' in a thread safe way.
   // Retur 0 on success, -errno on error

   // Make sure input make sense
   if (uid < 0)
      return -EINVAL;

   // Call getpwuid_r ...
   struct passwd pw;
   struct passwd *ppw = 0;
   char buf[2048];
#if defined(__sun) && !defined(__GNUC__)
   ppw = getpwuid_r((uid_t)uid, &pw, buf, sizeof(buf));
#else
   getpwuid_r((uid_t)uid, &pw, buf, sizeof(buf), &ppw);
#endif
   if (ppw) {
      // Fill output
      ui.fUid = uid;
      ui.fGid = (int) pw.pw_gid;
      ui.fHomeDir = pw.pw_dir;
      ui.fUser = pw.pw_name;
      // Done
      return 0;
   }

   // Failure
   if (errno != 0)
      return ((int) -errno);
   else
      return -ENOENT;
}

//__________________________________________________________________________
int XrdProofdAux::Write(int fd, const void *buf, size_t nb)
{
   // Write nb bytes at buf to descriptor 'fd' ignoring interrupts
   // Return the number of bytes written or -1 in case of error

   if (fd < 0)
      return -1;

   const char *pw = (const char *)buf;
   int lw = nb;
   int nw = 0, written = 0;
   while (lw) {
      if ((nw = write(fd, pw + written, lw)) < 0) {
         if (errno == EINTR) {
            errno = 0;
            continue;
         } else {
            break;
         }
      }
      // Count
      written += nw;
      lw -= nw;
   }

   // Done
   return written;
}

//_________________________________________________________________________________
void XrdProofdAux::LogEmsgToFile(const char *flog, const char *emsg, const char *pfx)
{
   // Logs error message 'emsg' to file 'flog' using standard technology
   XPDLOC(AUX, "Aux::LogEmsgToFile")

   if (flog && strlen(flog)) {
      // Open the file in write-only, append mode
      int logfd = open(flog, O_WRONLY|O_APPEND, 0644);
      if (logfd > 0) {
         fcntl(logfd, F_SETFD, FD_CLOEXEC);
         // Attach a logger to the file
         XrdSysLogger logger(logfd, 0);
         XrdSysError error(&logger, "xpd");
         // Log the message
         if (emsg && strlen(emsg) > 0) error.Emsg("-E", pfx, emsg);
         // Make sure that it is written to file
         if (fsync(logfd) != 0)
            TRACE(XERR, "problem syncing file "<<flog<<" - errno: "<<errno);
         // Free the descriptor
         if (close(logfd) != 0)
            TRACE(XERR, "problem closing file "<<flog<<" - errno: "<<errno);
      } else {
         TRACE(XERR, "file "<<flog<<" could not be opened - errno: "<<errno);
      }
   } else {
      TRACE(XERR, "file path undefined!");
   }
   // Done
   return;
}

//_____________________________________________________________________________
int XrdProofdAux::AssertDir(const char *path, XrdProofUI ui, bool changeown)
{
   // Make sure that 'path' exists and is owned by the entity
   // described by 'ui'.
   // If changeown is TRUE it tries to acquire the privileges before.
   // Return 0 in case of success, -1 in case of error
   XPDLOC(AUX, "Aux::AssertDir")

   TRACE(DBG, path);

   if (!path || strlen(path) <= 0)
      return -1;

   struct stat st;
   if (stat(path,&st) != 0) {
      if (errno == ENOENT) {

         {  XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
            if (XpdBadPGuard(pGuard, ui.fUid) && changeown) {
               TRACE(XERR, "could not get privileges to create dir");
               return -1;
            }

            if (mkdir(path, 0755) != 0) {
               TRACE(XERR, "unable to create dir: "<<path<<" (errno: "<<errno<<")");
               return -1;
            }
         }
         if (stat(path,&st) != 0) {
            TRACE(XERR, "unable to stat dir: "<<path<<" (errno: "<<errno<<")");
            return -1;
         }
      } else {
         // Failure: stop
         TRACE(XERR, "unable to stat dir: "<<path<<" (errno: "<<errno<<")");
         return -1;
      }
   }

   // Make sure the ownership is right
   if (changeown &&
      ((int) st.st_uid != ui.fUid || (int) st.st_gid != ui.fGid)) {

      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (XpdBadPGuard(pGuard, ui.fUid)) {
         TRACE(XERR, "could not get privileges to change ownership");
         return -1;
      }

      // Set ownership of the path to the client
      if (chown(path, ui.fUid, ui.fGid) == -1) {
         TRACE(XERR, "cannot set user ownership on path (errno: "<<errno<<")");
         return -1;
      }
   }

   // We are done
   return 0;
}

//_____________________________________________________________________________
int XrdProofdAux::AssertBaseDir(const char *path, XrdProofUI ui)
{
   // Make sure that the base dir of 'path' is either owned by 'ui' or
   // gives full permissions to 'ui'. 
   // If 'path' is a directory, go through the paths inside it recursively.
   // Return 0 in case of success, -1 in case of error
   XPDLOC(AUX, "Aux::AssertBaseDir")

   TRACE(DBG, path);
  
   if (!path || strlen(path) <= 0)
      return -1;

   XrdOucString base(path);
   if (base.endswith("/")) base.erasefromend(1);
   int isl = base.rfind('/');
   if (isl != 0) base.erase(isl);
   TRACE(DBG, "base: " <<base);
   
   struct stat st;
   if (stat(base.c_str(), &st) != 0) {
      // Failure: stop
      TRACE(XERR, "unable to stat base path: "<<base<<" (errno: "<<errno<<")");
      return -1;
   }

   // Check ownership and permissions
   if (ui.fUid != (int) st.st_uid) {
      unsigned pa = (st.st_mode & S_IRWXG);
      if (ui.fGid != (int) st.st_gid) 
         pa |= (st.st_mode & S_IRWXO);
      else
         pa |= S_IRWXO;
      if (pa != 0077) {
         TRACE(XERR, "effective user has not full permissions on base path: "<<base);
         return -1;
      }
   }
   
   // Done
   return 0;
}

//_____________________________________________________________________________
int XrdProofdAux::ChangeOwn(const char *path, XrdProofUI ui)
{
   // Change the ownership of 'path' to the entity described by 'ui'.
   // If 'path' is a directory, go through the paths inside it recursively.
   // Return 0 in case of success, -1 in case of error
   XPDLOC(AUX, "Aux::ChangeOwn")

   TRACE(DBG, path);

   if (!path || strlen(path) <= 0)
      return -1;

   struct stat st;
   if (stat(path,&st) != 0) {
      // Failure: stop
      TRACE(XERR, "unable to stat path: "<<path<<" (errno: "<<errno<<")");
      return -1;
   }

   // If is a directory apply this on it
   if (S_ISDIR(st.st_mode)) {
      // Loop over the dir
      DIR *dir = opendir(path);
      if (!dir) {
         TRACE(XERR,"cannot open "<<path<< "- errno: "<< errno);
         return -1;
      }
      XrdOucString proot(path);
      if (!proot.endswith('/')) proot += "/";

      struct dirent *ent = 0;
      while ((ent = readdir(dir))) {
         if (ent->d_name[0] == '.' || !strcmp(ent->d_name, "..")) continue;
         XrdOucString fn(proot);
         fn += ent->d_name;

         struct stat xst;
         if (stat(fn.c_str(),&xst) == 0) {
            // If is a directory apply this on it
            if (S_ISDIR(xst.st_mode)) {
               if (XrdProofdAux::ChangeOwn(fn.c_str(), ui) != 0) {
                  TRACE(XERR, "problems changing recursively ownership of: "<<fn);
                  return -1;
               }
            } else {
               // Get the privileges, if needed
               XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
               if (XpdBadPGuard(pGuard, ui.fUid)) {
                  TRACE(XERR, "could not get privileges to change ownership");
                  return -1;
               }
               // Set ownership of the path to the client
               if (chown(fn.c_str(), ui.fUid, ui.fGid) == -1) {
                  TRACE(XERR, "cannot set user ownership on path (errno: "<<errno<<")");
                  return -1;
               }
            }
         } else {
            TRACE(XERR, "unable to stat dir: "<<fn<<" (errno: "<<errno<<")");
         }
      }
      // Close the directory
      closedir(dir);

   } else if (((int) st.st_uid != ui.fUid) || ((int) st.st_gid != ui.fGid)) {
      // Get the privileges, if needed
      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (XpdBadPGuard(pGuard, ui.fUid)) {
         TRACE(XERR, "could not get privileges to change ownership");
         return -1;
      }
      // Set ownership of the path to the client
      if (chown(path, ui.fUid, ui.fGid) == -1) {
         TRACE(XERR, "cannot set user ownership on path (errno: "<<errno<<")");
         return -1;
      }
   }

   // We are done
   return 0;
}

//_____________________________________________________________________________
int XrdProofdAux::ChangeMod(const char *path, unsigned int mode)
{
   // Change the permission mode of 'path' to 'mode'.
   // If 'path' is a directory, go through the paths inside it recursively.
   // Return 0 in case of success, -1 in case of error
   XPDLOC(AUX, "Aux::ChangeMod")

   TRACE(HDBG, "path: "<<path);

   if (!path || strlen(path) <= 0)
      return -1;

   struct stat st;
   if (stat(path,&st) != 0) {
      // Failure: stop
      TRACE(XERR, "unable to stat path: "<<path<<" (errno: "<<errno<<")");
      return -1;
   }

   // Change the path first; then do it recursively, if needed
   {  // Get the privileges, if needed
      XrdSysPrivGuard pGuard(st.st_uid, st.st_gid);
      if (XpdBadPGuard(pGuard, st.st_uid)) {
         TRACE(XERR, "could not get privileges to change ownership");
         return -1;
      }
      // Set ownership of the path to the client
      if (chmod(path, mode) == -1) {
         TRACE(XERR, "cannot change permissions on path (errno: "<<errno<<")");
         return -1;
      }
   }

   // If is a directory apply this on it
   if (S_ISDIR(st.st_mode)) {
      // Loop over the dir
      DIR *dir = opendir(path);
      if (!dir) {
         TRACE(XERR,"cannot open "<<path<< "- errno: "<< errno);
         return -1;
      }
      XrdOucString proot(path);
      if (!proot.endswith('/')) proot += "/";

      struct dirent *ent = 0;
      while ((ent = readdir(dir))) {
         if (ent->d_name[0] == '.' || !strcmp(ent->d_name, "..")) continue;
         XrdOucString fn(proot);
         fn += ent->d_name;

         struct stat xst;
         if (stat(fn.c_str(),&xst) == 0) {
            {  // Get the privileges, if needed
               TRACE(HDBG,"getting {"<<xst.st_uid<<", "<< xst.st_gid<<"} identity");
               XrdSysPrivGuard pGuard(xst.st_uid, xst.st_gid);
               if (XpdBadPGuard(pGuard, xst.st_uid)) {
                  TRACE(XERR, "could not get privileges to change ownership");
                  return -1;
               }
               // Set the permission mode of the path
               if (chmod(fn.c_str(), mode) == -1) {
                  TRACE(XERR, "cannot change permissions on path (errno: "<<errno<<")");
                  return -1;
               }
            }
            // If is a directory apply this on it
            if (S_ISDIR(xst.st_mode)) {
               if (XrdProofdAux::ChangeMod(fn.c_str(), mode) != 0) {
                  TRACE(XERR, "problems changing recursively permissions of: "<<fn);
                  return -1;
               }
            }
         } else {
            TRACE(XERR, "unable to stat dir: "<<fn<<" (errno: "<<errno<<")");
         }
      }
      // Close the directory
      closedir(dir);
   }

   // We are done
   return 0;
}

//_____________________________________________________________________________
int XrdProofdAux::ChangeToDir(const char *dir, XrdProofUI ui, bool changeown)
{
   // Change current directory to 'dir'.
   // If changeown is TRUE it tries to acquire the privileges before.
   // Return 0 in case of success, -1 in case of error
   XPDLOC(AUX, "Aux::ChangeToDir")

   TRACE(DBG, "changing to " << ((dir) ? dir : "**undef***"));

   if (!dir || strlen(dir) <= 0)
      return -1;

   if (changeown && ((int) geteuid() != ui.fUid || (int) getegid() != ui.fGid)) {

      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (XpdBadPGuard(pGuard, ui.fUid)) {
         TRACE(XERR, changeown << ": could not get privileges; {uid,gid} req: {"<< ui.fUid <<","<<ui.fGid<<
                     "}, {euid,egid}: {" << geteuid() <<","<<getegid()<<"}, {uid,gid}: {"<<getuid()<<","<<getgid() << "}; errno: "<<errno);
         return -1;
      }
      if (chdir(dir) == -1) {
         TRACE(XERR, changeown << ": can't change directory to '"<< dir<<"'; {ui.fUid,ui.fGid}: {"<< ui.fUid <<","<<ui.fGid<<
                     "}, {euid,egid}: {" << geteuid() <<","<<getegid()<<"}, {uid,gid}: {"<<getuid()<<","<<getgid() << "}; errno: "<<errno);
         return -1;
      }
   } else {
      if (chdir(dir) == -1) {
         TRACE(XERR, changeown << ": can't change directory to "<< dir << 
                     ", euid: " << geteuid() <<", uid:"<<getuid()<<"; errno: "<<errno);
         return -1;
      }
   }

   // We are done
   return 0;
}

//_____________________________________________________________________________
int XrdProofdAux::SymLink(const char *path, const char *link)
{
   // Create a symlink 'link' to 'path'
   // Return 0 in case of success, -1 in case of error
   XPDLOC(AUX, "Aux::SymLink")

   TRACE(DBG, path<<" -> "<<link);

   if (!path || strlen(path) <= 0 || !link || strlen(link) <= 0)
      return -1;

   // Remove existing link, if any
   if (unlink(link) != 0 && errno != ENOENT) {
      TRACE(XERR, "problems unlinking existing symlink "<< link<<
                    " (errno: "<<errno<<")");
      return -1;
   }
   if (symlink(path, link) != 0) {
      TRACE(XERR, "problems creating symlink " << link<<
                    " (errno: "<<errno<<")");
      return -1;
   }

   // We are done
   return 0;
}

//______________________________________________________________________________
int XrdProofdAux::CheckIf(XrdOucStream *s, const char *host)
{
   // Check existence and match condition of an 'if' directive
   // If none (valid) is found, return -1.
   // Else, return number of chars matching.
   XPDLOC(AUX, "")

   // There must be an 'if'
   char *val = s ? s->GetWord() : 0;
   if (!val || strncmp(val,"if",2)) {
      if (val)
         // allow the analysis of the token
         s->RetToken();
      return -1;
   }

   // check value if any
   val = s->GetWord();
   if (!val)
      return -1;

   // Deprecate
   TRACE(ALL,  ">>> Warning: 'if' conditions at the end of the directive are deprecated ");
   TRACE(ALL,  ">>> Please use standard Scalla/Xrootd 'if-else-fi' constructs");
   TRACE(ALL,  ">>> (see http://xrootd.slac.stanford.edu/doc/xrd_config/xrd_config.htm)");

   // Notify
   TRACE(DBG, "Aux::CheckIf: <pattern>: " <<val);

   // Return number of chars matching
   XrdOucString h(host);
   return h.matches((const char *)val);
}

//______________________________________________________________________________
int XrdProofdAux::GetNumCPUs()
{
   // Find out and return the number of CPUs in the local machine.
   // Return -1 in case of failure.
   XPDLOC(AUX, "Aux::GetNumCPUs")

   static int ncpu = -1;

   // Use cached value, if any
   if (ncpu > 0)
      return ncpu;
   ncpu = 0;

   XrdOucString emsg;

#if defined(linux)
   // Look for in the /proc/cpuinfo file
   XrdOucString fcpu("/proc/cpuinfo");
   FILE *fc = fopen(fcpu.c_str(), "r");
   if (!fc) {
      if (errno == ENOENT) {
         TRACE(XERR, "/proc/cpuinfo missing!!! Something very bad going on");
      } else {
         XPDFORM(emsg, "cannot open %s; errno: %d", fcpu.c_str(), errno);
         TRACE(XERR, emsg);
      }
      return -1;
   }
   // Read lines and count those starting with "processor"
   char line[2048] = { 0 };
   while (fgets(line, sizeof(line), fc)) {
      if (!strncmp(line, "processor", strlen("processor")))
         ncpu++;
   }
   // Close the file
   fclose(fc);

#elif defined(__sun)

   // Run "psrinfo" in popen and count lines
   FILE *fp = popen("psrinfo", "r");
   if (fp != 0) {
      char line[2048] = { 0 };
      while (fgets(line, sizeof(line), fp))
         ncpu++;
      pclose(fp);
   }

#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)

   // Run "sysctl -n hw.ncpu" in popen and decode the output
   FILE *fp = popen("sysctl -n hw.ncpu", "r");
   if (fp != 0) {
      char line[2048] = { 0 };
      while (fgets(line, sizeof(line), fp))
         ncpu = XrdProofdAux::GetLong(&line[0]);
      pclose(fp);
   }
#endif

   TRACE(DBG, "# of cores found: "<<ncpu);

   // Done
   return (ncpu <= 0) ? (int)(-1) : ncpu ;
}

#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)
//__________________________________________________________________________
int XrdProofdAux::GetMacProcList(kinfo_proc **plist, int &nproc)
{
   // Returns a list of all processes on the system.  This routine
   // allocates the list and puts it in *plist and counts the
   // number of entries in 'nproc'. Caller is responsible for 'freeing'
   // the list.
   // On success, the function returns 0.
   // On error, the function returns an errno value.
   //
   // Adapted from: reply to Technical Q&A 1123,
   //               http://developer.apple.com/qa/qa2001/qa1123.html
   //
   XPDLOC(AUX, "Aux::GetMacProcList")

   int rc = 0;
   kinfo_proc *res;
   bool done = 0;
   static const int name[] = {CTL_KERN, KERN_PROC, KERN_PROC_ALL, 0};

   TRACE(DBG, "enter");

   // Declaring name as const requires us to cast it when passing it to
   // sysctl because the prototype doesn't include the const modifier.
   size_t len = 0;

   if (!plist || (*plist))
      return EINVAL;
   nproc = 0;

   // We start by calling sysctl with res == 0 and len == 0.
   // That will succeed, and set len to the appropriate length.
   // We then allocate a buffer of that size and call sysctl again
   // with that buffer.  If that succeeds, we're done.  If that fails
   // with ENOMEM, we have to throw away our buffer and loop.  Note
   // that the loop causes use to call sysctl with 0 again; this
   // is necessary because the ENOMEM failure case sets length to
   // the amount of data returned, not the amount of data that
   // could have been returned.

   res = 0;
   do {
      // Call sysctl with a 0 buffer.
      len = 0;
      if ((rc = sysctl((int *)name, (sizeof(name)/sizeof(*name)) - 1,
                       0, &len, 0, 0)) == -1) {
         rc = errno;
      }

      // Allocate an appropriately sized buffer based on the results
      // from the previous call.
      if (rc == 0) {
         res = (kinfo_proc *) malloc(len);
         if (!res)
            rc = ENOMEM;
      }

      // Call sysctl again with the new buffer.  If we get an ENOMEM
      // error, toss away our buffer and start again.
      if (rc == 0) {
         if ((rc = sysctl((int *)name, (sizeof(name)/sizeof(*name)) - 1,
                          res, &len, 0, 0)) == -1) {
            rc = errno;
         }
         if (rc == 0) {
            done = 1;
         } else if (rc == ENOMEM) {
            if (res)
               free(res);
            res = 0;
            rc = 0;
         }
      }
   } while (rc == 0 && !done);

   // Clean up and establish post conditions.
   if (rc != 0 && !res) {
      free(res);
      res = 0;
   }
   *plist = res;
   if (rc == 0)
      nproc = len / sizeof(kinfo_proc);

   // Done
   return rc;
}
#endif

//______________________________________________________________________________
int XrdProofdAux::GetProcesses(const char *pn, std::map<int,XrdOucString> *pmap)
{
   // Get from the process table list of PIDs for processes named "proofserv'
   // For {linux, sun, macosx} it uses the system info; for other systems it
   // invokes the command shell 'ps ax' via popen.
   // Return the number of processes found, or -1 if some error occured.
   XPDLOC(AUX, "Aux::GetProcesses")

   int np = 0;

   // Check input consistency
   if (!pn || strlen(pn) <= 0 || !pmap) {
      TRACE(XERR, "invalid inputs");
      return -1;
   }
   TRACE(DBG, "process name: "<<pn);

   XrdOucString emsg;

#if defined(linux) || defined(__sun)
   // Loop over the "/proc" dir
   DIR *dir = opendir("/proc");
   if (!dir) {
      emsg = "cannot open /proc - errno: ";
      emsg += errno;
      TRACE(DBG, emsg.c_str());
      return -1;
   }

   struct dirent *ent = 0;
   while ((ent = readdir(dir))) {
      if (DIGIT(ent->d_name[0])) {
         XrdOucString fn("/proc/", 256);
         fn += ent->d_name;
#if defined(linux)
         fn += "/status";
         // Open file
         FILE *ffn = fopen(fn.c_str(), "r");
         if (!ffn) {
            emsg = "cannot open file ";
            emsg += fn; emsg += " - errno: "; emsg += errno;
            TRACE(HDBG, emsg);
            continue;
         }
         // Read info
         bool ok = 0;
         int pid = -1;
         char line[2048] = { 0 };
         while (fgets(line, sizeof(line), ffn)) {
            // Check name
            if (strstr(line, "Name:")) {
               if (strstr(line, pn)) {
                  // Good one
                  ok = 1;
               }
               // We are done with this proc file
               break;
            }
         }
         if (ok) {
            fclose(ffn);
            fn.replace("/status", "/cmdline");
            // Open file
            if (!(ffn = fopen(fn.c_str(), "r"))) {
               emsg = "cannot open file ";
               emsg += fn; emsg += " - errno: "; emsg += errno;
               TRACE(HDBG, emsg);
               continue;
            }
            // Read the command line
            XrdOucString cmd;
            char buf[256];
            char *p = &buf[0];
            int pos = 0, ltot = 0, nr = 1;
            errno = 0;
            while (nr > 0) {
               while ((nr = read(fileno(ffn), p + pos, 1)) == -1 && errno == EINTR) {
                  errno = 0;
               }
               ltot += nr;
               if (ltot == 254) {
                  buf[255] = 0;
                  cmd += buf;
                  pos = 0;
                  ltot = 0;
               } else if (nr > 0) {
                  if (*p == 0) *p = ' ';
                  p += nr;
               }
            }
            // Null terminate
            buf[ltot] = 0;
            cmd += buf;
            // Good one: take the pid
            pid = strtol(ent->d_name, 0, 10);
            pmap->insert(std::make_pair(pid, cmd));
            np++;
         }
         // Close the file
         fclose(ffn);
#elif defined(__sun)
         fn += "/psinfo";
         // Open file
         int ffd = open(fn.c_str(), O_RDONLY);
         if (ffd <= 0) {
            emsg = "cannot open file ";
            emsg += fn; emsg += " - errno: "; emsg += errno;
            TRACE(HDBG, emsg);
            continue;
         }
         // Get the information
         psinfo_t psi;
         if (read(ffd, &psi, sizeof(psinfo_t)) != sizeof(psinfo_t)) {
            emsg = "cannot read ";
            emsg += fn; emsg += ": errno: "; emsg += errno;
            TRACE(XERR, emsg);
            close(ffd);
            continue;
         }
         // Check name
         if (strstr(psi.pr_fname, pn)) {
            // Build command line
            XrdOucString cmd(psi.pr_fname);
            if (cmd.length() > 0) cmd += " ";
            cmd += psi.pr_psargs;
            // Good one: take the pid
            int pid = strtol(ent->d_name, 0, 10);
            pmap->insert(std::make_pair(pid, cmd));
            np++;
         }
         // Close the file
         close(ffd);
#endif
      }
   }
   // Close the directory
   closedir(dir);

#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)

   // Get the proclist
   kinfo_proc *pl = 0;
   int ern = 0;
   if ((ern = XrdProofdAux::GetMacProcList(&pl, np)) != 0) {
      emsg = "cannot get the process list: errno: ";
      emsg += ern;
      TRACE(XERR, emsg);
      return -1;
   }

   // Loop over the list
   int ii = np;
   while (ii--) {
      if (strstr(pl[ii].kp_proc.p_comm, pn)) {
         // Good one: take the pid
         pmap->insert(std::make_pair(pl[ii].kp_proc.p_pid, XrdOucString(pl[ii].kp_proc.p_comm)));
         np++;
      }
   }
   // Cleanup
   free(pl);
#else

   // For the remaining cases we use 'ps' via popen to localize the processes

   // Build command
   XrdOucString cmd = "ps ax -ww | grep proofserv 2>/dev/null";

   // Run it ...
   XrdOucString pids = ":";
   FILE *fp = popen(cmd.c_str(), "r");
   if (fp != 0) {
      char line[2048] = { 0 };
      while (fgets(line, sizeof(line), fp)) {
         int pid = (int) XrdProofdAux::GetLong(&line[from]);
         pmap->insert(std::make_pair(pid, XrdOucString(line)));
         np++;
      }
      pclose(fp);
   } else {
      // Error executing the command
      return -1;
   }

#endif

   // Done
   return np;
}

//_____________________________________________________________________________
int XrdProofdAux::GetIDFromPath(const char *path, XrdOucString &emsg)
{
   // Extract an integer from a file

   emsg = "";
   // Get the ID
   int id = -1;
   FILE *fid = fopen(path, "r");
   if (fid) {
      char line[64];
      if (fgets(line, sizeof(line), fid))
         sscanf(line, "%d", &id);
      fclose(fid);
   } else if (errno != ENOENT) {
      XPDFORM(emsg, "GetIDFromPath: error reading id from: %s (errno: %d)",
                path, errno);
   }
   // Done
   return id;
}

//______________________________________________________________________________
bool XrdProofdAux::HasToken(const char *s, const char *tokens)
{
   // Returns true is 's' contains at least one of the comma-separated tokens
   // in 'tokens'. Else returns false.

   if (s && strlen(s) > 0) {
      XrdOucString tks(tokens), tok;
      int from = 0;
      while ((from = tks.tokenize(tok, from, ',')) != -1)
         if (strstr(s, tok.c_str())) return 1;
   }
   return 0;
}

//______________________________________________________________________________
int XrdProofdAux::VerifyProcessByID(int pid, const char *pname)
{
   // Check if a process named 'pname' and process 'pid' is still
   // in the process table.
   // For {linux, sun, macosx} it uses the system info; for other systems it
   // invokes the command shell 'ps ax' via popen.
   // Return 1 if running, 0 if not running, -1 if the check could not be run.
   XPDLOC(AUX, "Aux::VerifyProcessByID")

   int rc = 0;

   TRACE(DBG, "pid: "<<pid);

   // Check input consistency
   if (pid < 0) {
      TRACE(XERR, "invalid pid");
      return -1;
   }

   XrdOucString emsg;

   // Name
   const char *pn = (pname && strlen(pname) > 0) ? pname : "proofserv";

#if defined(linux)
   // Look for the relevant /proc dir
   XrdOucString fn("/proc/");
   fn += pid;
   fn += "/stat";
   FILE *ffn = fopen(fn.c_str(), "r");
   if (!ffn) {
      if (errno == ENOENT) {
         TRACE(DBG, "process does not exists anymore");
         return 0;
      } else {
         XPDFORM(emsg, "cannot open %s; errno: %d", fn.c_str(), errno);
         TRACE(XERR, emsg);
         return -1;
      }
   }
   // Read status line
   char line[2048] = { 0 };
   if (fgets(line, sizeof(line), ffn)) {
      if (XrdProofdAux::HasToken(line, pn))
         // Still there
         rc = 1;
   } else {
      XPDFORM(emsg, "cannot read %s; errno: %d", fn.c_str(), errno);
      TRACE(XERR, emsg);
      fclose(ffn);
      return -1;
   }
   // Close the file
   fclose(ffn);

#elif defined(__sun)

   // Look for the relevant /proc dir
   XrdOucString fn("/proc/");
   fn += pid;
   fn += "/psinfo";
   int ffd = open(fn.c_str(), O_RDONLY);
   if (ffd <= 0) {
      if (errno == ENOENT) {
         TRACE(DBG, "VerifyProcessByID: process does not exists anymore");
         return 0;
      } else {
         XPDFORM(emsg, "cannot open %s; errno: %d", fn.c_str(), errno);
         TRACE(XERR, emsg);
         return -1;
      }
   }
   // Get the information
   psinfo_t psi;
   if (read(ffd, &psi, sizeof(psinfo_t)) != sizeof(psinfo_t)) {
      XPDFORM(emsg, "cannot read %s; errno: %d", fn.c_str(), errno);
      TRACE(XERR, emsg);
      close(ffd);
      return -1;
   }

   // Verify now
   if (XrdProofdAux::HasToken(psi.pr_fname, pn))
      // The process is still there
      rc = 1;

   // Close the file
   close(ffd);

#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)

   // Get the proclist
   kinfo_proc *pl = 0;
   int np;
   int ern = 0;
   if ((ern = XrdProofdAux::GetMacProcList(&pl, np)) != 0) {
      XPDFORM(emsg, "cannot get the process list: errno: %d", ern);
      TRACE(XERR, emsg);
      return -1;
   }

   // Loop over the list
   while (np--) {
      if (pl[np].kp_proc.p_pid == pid &&
         XrdProofdAux::HasToken(pl[np].kp_proc.p_comm, pn)) {
         // Process still exists
         rc = 1;
         break;
      }
   }
   // Cleanup
   free(pl);
#else
   // Use the output of 'ps ax' as a backup solution
   XrdOucString cmd = "ps ax | grep proofserv 2>/dev/null";
   if (pname && strlen(pname))
      cmd.replace("proofserv", pname);
   FILE *fp = popen(cmd.c_str(), "r");
   if (fp != 0) {
      char line[2048] = { 0 };
      while (fgets(line, sizeof(line), fp)) {
         if (pid == XrdProofdAux::GetLong(line)) {
            // Process still running
            rc = 1;
            break;
         }
      }
      pclose(fp);
   } else {
      // Error executing the command
      return -1;
   }
#endif
   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofdAux::KillProcess(int pid, bool forcekill, XrdProofUI ui, bool changeown)
{
   // Kill the process 'pid'.
   // A SIGTERM is sent, unless 'kill' is TRUE, in which case a SIGKILL is used.
   // If add is TRUE (default) the pid is added to the list of processes
   // requested to terminate.
   // Return 0 on success, -1 if not allowed or other errors occured.
   XPDLOC(AUX, "Aux::KillProcess")

   TRACE(DBG, "pid: "<<pid<< ", forcekill: "<< forcekill);

   XrdOucString msg;
   if (pid > 0) {
      // We need the right privileges to do this
      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (XpdBadPGuard(pGuard, ui.fUid) && changeown) {
         TRACE(XERR, "could not get privileges");
         return -1;
      } else {
         bool signalled = 1;
         if (forcekill) {
            // Hard shutdown via SIGKILL
            if (kill(pid, SIGKILL) != 0) {
               if (errno != ESRCH) {
                  XPDFORM(msg, "kill(pid,SIGKILL) failed for process %d; errno: %d", pid, errno);
                  TRACE(XERR, msg);
                  return -1;
               }
               signalled = 0;
            }
         } else {
            // Softer shutdown via SIGTERM
            if (kill(pid, SIGTERM) != 0) {
               if (errno != ESRCH) {
                  XPDFORM(msg, "kill(pid,SIGTERM) failed for process %d; errno: %d", pid, errno);
                  TRACE(XERR, msg);
                  return -1;
               }
               signalled = 0;
            }
         }
         // Notify failure
         if (!signalled) {
            TRACE(DBG, "process ID "<<pid<<" not found in the process table");
         }
      }
   } else {
      return -1;
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdAux::RmDir(const char *path)
{
   // Remove directory at path and its content.
   // Returns 0 on success, -errno of the last error on failure
   XPDLOC(AUX, "Aux::RmDir")

   int rc = 0;

   TRACE(DBG, path);

   // Open dir
   DIR *dir = opendir(path);
   if (!dir) {
      TRACE(XERR, "cannot open dir "<<path<<" ; error: "<<errno);
      return -errno;
   }

   // Scan the directory
   XrdOucString entry;
   struct stat st;
   struct dirent *ent = 0;
   while ((ent = (struct dirent *)readdir(dir))) {
      // Skip the basic entries
      if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) continue;
      // Get info about the entry
      XPDFORM(entry, "%s/%s", path, ent->d_name);
      if (stat(entry.c_str(), &st) != 0) {
         TRACE(XERR, "cannot stat entry "<<entry<<" ; error: "<<errno);
         rc = -errno;
         break;
      }
      // Remove directories recursively
      if (S_ISDIR(st.st_mode)) {
         rc = XrdProofdAux::RmDir(entry.c_str());
         if (rc != 0) {
            TRACE(XERR, "problems removing"<<entry<<" ; error: "<<-rc);
            break;
         }
      } else {
         // Remove the entry
         if (unlink(entry.c_str()) != 0) {
            rc = -errno;
            TRACE(XERR, "problems removing"<<entry<<" ; error: "<<-rc);
            break;
         }
      }
   }
   // Close the directory
   closedir(dir);

   // If successful, remove the directory
   if (!rc && rmdir(path) != 0) {
      rc = -errno;
      TRACE(XERR, "problems removing"<<path<<" ; error: "<<-rc);
   }

   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofdAux::MvDir(const char *oldpath, const char *newpath)
{
   // Move content of directory at oldpath to newpath.
   // The destination path 'newpath' must exist.
   // Returns 0 on success, -errno of the last error on failure
   XPDLOC(AUX, "Aux::MvDir")

   int rc = 0;

   TRACE(DBG, "oldpath "<<oldpath<<", newpath: "<<newpath);

   // Open existing dir
   DIR *dir = opendir(oldpath);
   if (!dir) {
      TRACE(XERR, "cannot open dir "<<oldpath<<" ; error: "<<errno);
      return -errno;
   }

   // Assert destination dir
   struct stat st;
   if (stat(newpath, &st) != 0 || !S_ISDIR(st.st_mode)) {
      TRACE(XERR, "destination dir "<<newpath<<
                  " does not exist or is not a directory; errno: "<<errno);
      return -ENOENT;
   }

   // Scan the source directory
   XrdOucString srcentry, dstentry;
   struct dirent *ent = 0;
   while ((ent = (struct dirent *)readdir(dir))) {
      // Skip the basic entries
      if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) continue;
      // Get info about the entry
      XPDFORM(srcentry, "%s/%s", oldpath, ent->d_name);
      if (stat(srcentry.c_str(), &st) != 0) {
         TRACE(XERR, "cannot stat entry "<<srcentry<<" ; error: "<<errno);
         rc = -errno;
         break;
      }
      // Destination entry
      XPDFORM(dstentry, "%s/%s", newpath, ent->d_name);
      // Mv directories recursively
      if (S_ISDIR(st.st_mode)) {
         mode_t srcmode = st.st_mode;
         // Create dest sub-dir
         if (stat(dstentry.c_str(), &st) == 0) {
            if (!S_ISDIR(st.st_mode)) {
               TRACE(XERR, "destination path already exists and is not a directory: "<<dstentry);
               rc = -ENOTDIR;
               break;
            }
         } else {
            if (mkdir(dstentry.c_str(), srcmode) != 0) {
               TRACE(XERR, "cannot create entry "<<dstentry<<" ; error: "<<errno);
               rc = -errno;
               break;
            }
         }
         if ((rc = XrdProofdAux::MvDir(srcentry.c_str(), dstentry.c_str())) != 0) {
            TRACE(XERR, "problems moving "<<srcentry<<" to "<<dstentry<<"; error: "<<-rc);
            break;
         }
         if ((rc = XrdProofdAux::RmDir(srcentry.c_str())) != 0) {
            TRACE(XERR, "problems removing "<<srcentry<<"; error: "<<-rc);
            break;
         }
      } else {
         // Move the entry
         if (rename(srcentry.c_str(), dstentry.c_str()) != 0) {
            rc = -errno;
            TRACE(XERR, "problems moving "<<srcentry<<" to "<<dstentry<<"; error: "<<-rc);
            break;
         }
      }
   }
   // Close the directory
   closedir(dir);

   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofdAux::Touch(const char *path, int opt)
{
   // Set access (opt == 1), modify (opt =2 ) or access&modify (opt = 0, default)
   // times of path to current time.
   // Returns 0 on success, -errno on failure

   if (opt == 0) {
      if (utime(path, 0) != 0)
         return -errno;
   } else if (opt <= 2) {
      struct stat st;
      if (stat(path, &st) != 0)
         return -errno;
      struct utimbuf ut;
      if (opt == 1) {
         ut.actime = time(0);
         ut.modtime = st.st_mtime;
      } else if (opt == 2) {
         ut.modtime = time(0);
         ut.actime = st.st_atime;
      }
      if (utime(path, &ut) != 0)
         return -errno;
   } else {
      // Unknown option
      return -1;
   }
   // Done
   return 0;
}

//___________________________________________________________________________
int XrdProofdAux::ReadMsg(int fd, XrdOucString &msg)
{
   // Receive 'msg' from pipe fd
   XPDLOC(AUX, "Aux::ReadMsg")

   msg = "";
   if (fd > 0) {

      // Read message length
      int len = 0;
      if (read(fd, &len, sizeof(len)) != sizeof(len))
         return -errno;
      TRACE(HDBG,fd<<": len: "<<len);

      // Read message
      char buf[XPD_MAXLEN];
      int nr = -1;
      do {
         int wanted = (len > XPD_MAXLEN-1) ? XPD_MAXLEN-1 : len;
         while ((nr = read(fd, buf, wanted)) < 0 &&
               errno == EINTR)
            errno = 0;
         if (nr < wanted) {
            break;
         } else {
            buf[nr] = '\0';
            msg += buf;
         }
         // Update counters
         len -= nr;
      } while (nr > 0 && len > 0);

      TRACE(HDBG,fd<<": buf: "<<buf);

      // Done
      return 0;
   }
   // Undefined socket
   TRACE(XERR, "pipe descriptor undefined: "<<fd);
   return -1;
}

//______________________________________________________________________________
int XrdProofdAux::ParsePidPath(const char *path,
                               XrdOucString &before, XrdOucString &after)
{
   // Parse a path in the form of "<before>[.<pid>][.<after>]", filling 'rest'
   // and returning 'pid'.
   // Return 0 if pid is not defined; 'before' is filled with the string preceeding
   // <pid>, <after> with the string following <pid>.
   XPDLOC(AUX, "ParsePidPath")

   long int pid = -1;
   if (path && strlen(path)) {
      pid = 0;
      int from = 0;
      XrdOucString spid, s(path);
      bool nopid = 1;
      while ((from = s.tokenize(spid, from, '.')) != -1) {
         if (spid.length() > 0) {
            if (spid.isdigit()) {
               // Get pid
               pid = (int) spid.atoi();
               if (!XPD_LONGOK(pid)) {
                  // Substring is not a PID
                  pid = 0;
               }
            }
            if (nopid && pid > 0) {
               nopid = 0;
            } else if (nopid) {
               if (before.length() > 0) before += ".";
               before += spid;
            } else {
               if (after.length() > 0) after += ".";
               after += spid;
            }
         }
      }
      if (pid == 0 && before.length() == 0) {
         before = after;
         after = "";
      }
   }

   TRACE(HDBG,"path: "<<path<<" --> before: '"<<before<<"', pid: "<<pid<<", after: '"<<after<<"'");

   // Done
   return pid;
}

//______________________________________________________________________________
int XrdProofdAux::ParseUsrGrp(const char *path, XrdOucString &usr, XrdOucString &grp)
{
   // Parse a path in the form of "<usr>[.<grp>][.<pid>]", filling 'usr' and 'grp'.
   // Returns -1 on failure, 0 if the pid is not defined or the pid.

   XrdOucString rest, after;
   int pid = ParsePidPath(path, rest, after);

   if (pid >= 0 && rest.length() > 0) {
      // Fill 'usr' (everything until the last dot)
      usr = rest;
      int ip = STR_NPOS;
      if ((ip = rest.rfind('.')) != STR_NPOS) {
         usr.erase(ip);
         // Fill 'grp'
         grp = rest;
         grp.erase(0, ip + 1);
      }
   }
   // Done
   return pid;
}

//
// Functions to process directives for integer and strings
//

//______________________________________________________________________________
int DoDirectiveClass(XrdProofdDirective *d, char *val, XrdOucStream *cfg, bool rcf)
{
   // Generic class directive processor

   if (!d || !(d->fVal))
      // undefined inputs
      return -1;

   return ((XrdProofdConfig *)d->fVal)->DoDirective(d, val, cfg, rcf);
}

//______________________________________________________________________________
int DoDirectiveInt(XrdProofdDirective *d, char *val, XrdOucStream *cfg, bool rcf)
{
   // Process directive for an integer
   XPDLOC(AUX, "DoDirectiveInt")

   if (!d || !(d->fVal) || !val)
      // undefined inputs
      return -1;

   if (rcf && !d->fRcf)
      // Not re-configurable: do nothing
      return 0;

   // Check deprecated 'if' directive
   if (d->fHost && cfg)
      if (XrdProofdAux::CheckIf(cfg, d->fHost) == 0)
         return 0;

   long int v = strtol(val,0,10);
   *((int *)d->fVal) = v;

   TRACE(DBG, "set "<<d->fName<<" to "<<*((int *)d->fVal));

   return 0;
}

//______________________________________________________________________________
int DoDirectiveString(XrdProofdDirective *d, char *val, XrdOucStream *cfg, bool rcf)
{
   // Process directive for a string
   XPDLOC(AUX, "DoDirectiveString")

   if (!d || !(d->fVal) || !val)
      // undefined inputs
      return -1;

   if (rcf && !d->fRcf)
      // Not re-configurable: do nothing
      return 0;

   // Check deprecated 'if' directive
   if (d->fHost && cfg)
      if (XrdProofdAux::CheckIf(cfg, d->fHost) == 0)
         return 0;

   *((XrdOucString *)d->fVal) = val;

   TRACE(DBG, "set "<<d->fName<<" to "<<*((XrdOucString *)d->fVal));
   return 0;
}

//__________________________________________________________________________
int SetHostInDirectives(const char *, XrdProofdDirective *d, void *h)
{
   // Set host field for directive 'd' to (const char *h)

   const char *host = (const char *)h;

   if (!d || !host || strlen(host) <= 0)
      // Dataset root dir undefined: we cannot continue
      return 1;

   d->fHost = host;

   // Process next
   return 0;
}

//
// XrdProofdPipe: class implementing pipe functionality
//
//__________________________________________________________________________
XrdProofdPipe::XrdProofdPipe()
{
   // Constructor: create the pipe

   // Init pipe for the poller
   if (pipe(fPipe) != 0) {
      fPipe[0] = -1;
      fPipe[1] = -1;
   }
}

//__________________________________________________________________________
XrdProofdPipe::~XrdProofdPipe()
{
   // Destructor

   // Close the pipe
   Close();
}

//__________________________________________________________________________
void XrdProofdPipe::Close()
{
   // If open, close and invalidated the pipe descriptors

   if (IsValid()) {
      close(fPipe[0]);
      close(fPipe[1]);
      fPipe[0] = -1;
      fPipe[1] = -1;
   }
}

//__________________________________________________________________________
int XrdProofdPipe::Post(int type, const char *msg)
{
   // Post message on the pipe 
   XPDLOC(AUX, "Pipe::Post")


   if (IsValid()) {
      XrdOucString buf;
      if (msg && strlen(msg) > 0) {
         XPDFORM(buf, "%d %s", type, msg);
      } else {
         buf += type;
      }
      TRACE(HDBG, fPipe[1] << ": posting: type: "<<type<<", buf: "<<buf);
      int len = buf.length() + 1;
      XrdSysMutexHelper mh(fWrMtx);
      if (write(fPipe[1], &len, sizeof(len)) !=  sizeof(len))
         return -errno;
      if (write(fPipe[1], buf.c_str(), len) !=  len)
         return -errno;
      // Done
      return 0;
   }
   // Invalid pipe
   TRACE(XERR, "pipe is invalid");
   return -1;
}

//__________________________________________________________________________
int XrdProofdPipe::Recv(XpdMsg &msg)
{
   // Recv message from the pipe 
   XPDLOC(AUX, "Pipe::Recv")

   if (IsValid()) {
      XrdOucString buf;
      {  XrdSysMutexHelper mh(fRdMtx);
         if (XrdProofdAux::ReadMsg(fPipe[0], buf) != 0)
            return -1;
      }
      TRACE(HDBG, fPipe[0] << ": receiving: msg: "<< buf);
      msg.Init(buf.c_str());
      // Done
      return 0;
   }
   // Invalid pipe
   TRACE(XERR, "pipe is invalid");
   return -1;
}

//__________________________________________________________________________
int XrdProofdPipe::Poll(int to)
{
   // Poll over the read pipe for to secs; return whatever poll returns
   XPDLOC(AUX, "Pipe::Poll")

   if (IsValid()) {

      // Read descriptor
      struct pollfd fds_r;
      fds_r.fd = fPipe[0];
      fds_r.events = POLLIN;

      // We wait for processes to communicate a session status change
      int pollrc = 0;
      int xto = (to > 0) ? to * 1000 : -1;
      while ((pollrc = poll(&fds_r, 1, xto)) < 0 && (errno == EINTR)) {
         errno = 0;
      }
      // Done
      return (pollrc >= 0) ? pollrc : -errno;
   }
   // Invalid pipe
   TRACE(XERR, "pipe is invalid");
   return -1;
}

//
// XpdMsg: class to handle messages received over the pipe
//
//__________________________________________________________________________
int XpdMsg::Init(const char *buf)
{
   // Init from buffer
   XPDLOC(AUX, "Msg::Init")

   fType = -1;
   fBuf = "";
   fFrom = -1;

   TRACE(HDBG, "buf: "<< (const char *)(buf ? buf : "+++ empty +++"));

   if (buf && strlen(buf) > 0) {
      fBuf = buf;
      fFrom = 0;
      // Extract the type
      XrdOucString ctyp;
      if ((fFrom = fBuf.tokenize(ctyp, fFrom, ' ')) == -1 || ctyp.length() <= 0) {
         TRACE(XERR, "ctyp: "<<ctyp<<" fFrom: "<<fFrom);
         fBuf = "";
         fFrom = -1;
         return -1;
      }
      fType = ctyp.atoi();
      if (!XPD_LONGOK(fType)) {
         TRACE(XERR, "ctyp: "<<ctyp<<" fType: "<<fType);
         fBuf = "";
         fFrom = -1;
         return -1;
      }
      fBuf.erase(0,fFrom);
      while (fBuf.beginswith(' '))
         fBuf.erase(0, 1);
      fFrom = 0;
      TRACE(HDBG, fType<<", "<<fBuf);
   }
   // Done
   return 0;
}

//__________________________________________________________________________
int XpdMsg::Get(int &i)
{
   // Get next token and interpret it as an int
   XPDLOC(AUX, "Msg::Get")

   TRACE(HDBG,"int &i: "<<fFrom<<" "<<fBuf);

   int iold = i;
   XrdOucString tkn;
   if ((fFrom = fBuf.tokenize(tkn, fFrom, ' ')) == -1 || tkn.length() <= 0)
      return -1;
   i = tkn.atoi();
   if (!XPD_LONGOK(i)) {
      TRACE(XERR, "tkn: "<<tkn<<" i: "<<i);
      i = iold;
      return -1;
   }
   // Done
   return 0;
}

//__________________________________________________________________________
int XpdMsg::Get(XrdOucString &s)
{
   // Get next token
   XPDLOC(AUX, "Msg::Get")

   TRACE(HDBG,"XrdOucString &s: "<<fFrom<<" "<<fBuf);

   if ((fFrom = fBuf.tokenize(s, fFrom, ' ')) == -1 || s.length() <= 0) {
      TRACE(XERR, "s: "<<s<<" fFrom: "<<fFrom);
      return -1;
   }

   // Done
   return 0;
}

//__________________________________________________________________________
int XpdMsg::Get(void **p)
{
   // Get next token and interpret it as a pointer
   XPDLOC(AUX, "Msg::Get")

   TRACE(HDBG,"void **p: "<<fFrom<<" "<<fBuf);

   XrdOucString tkn;
   if ((fFrom = fBuf.tokenize(tkn, fFrom, ' ')) == -1 || tkn.length() <= 0) {
      TRACE(XERR, "tkn: "<<tkn<<" fFrom: "<<fFrom);
      return -1;
   }
   sscanf(tkn.c_str(), "%p", p);

   // Done
   return 0;
}


//
// Class to handle condensed multi-string specification, e.g <head>[01-25]<tail>
//

//__________________________________________________________________________
void XrdProofdMultiStr::Init(const char *s)
{
   // Init the multi-string handler.
   // Supported formats:
   //    <head>[1-4]<tail>   for  <head>1<tail>, ..., <head>4<tail> (4 items)
   //    <head>[a,b]<tail>   for  <head>a<tail>, <head>b<tail> (2 items)
   //    <head>[a,1-3]<tail> for  <head>a<tail>, <head>1<tail>, <head>2<tail>,
   //                             <head>3<tail> (4 items)
   //    <head>[01-15]<tail> for  <head>01<tail>, ..., <head>15<tail> (15 items)
   //
   // A dashed is possible only between numerically treatable values, i.e.
   // single letters ([a-Z] will take all tokens between 'a' and 'Z') or n-field
   // numbers ([001-999] will take all numbers 1 to 999 always using 3 spaces).
   // Mixed values (e.g. [a-034]) are not allowed.

   fN = 0;
   if (s && strlen(s)) {
      XrdOucString kernel(s);
      // Find begin of kernel
      int ib = kernel.find('[');
      if (ib == STR_NPOS) return;
      // Find end of kernel
      int ie = kernel.find(']', ib + 1);
      if (ie == STR_NPOS) return;
      // Check kernel length (it must not be empty)
      if (ie == ib + 1) return;
      // Fill head and tail
      fHead.assign(kernel, 0, ib -1);
      fTail.assign(kernel, ie + 1);
      // The rest is the kernel
      XrdOucString tkns(kernel, ib + 1, ie - 1);
      // Tokenize the kernel filling the list
      int from = 0;
      XrdOucString tkn;
      while ((from = tkns.tokenize(tkn, from, ',')) != -1) {
         if (tkn.length() > 0) {
            XrdProofdMultiStrToken t(tkn.c_str());
            if (t.IsValid()) {
               fN += t.N();
               fTokens.push_back(t);
            }
         }
      }
      // Reset everything if nothing found
      if (!IsValid()) {
         fHead = "";
         fTail = "";
      }
   }
}

//__________________________________________________________________________
bool XrdProofdMultiStr::Matches(const char *s)
{
   // Return true if 's' is compatible with this multi-string 

   if (s && strlen(s)) {
      XrdOucString str(s);
      if (fHead.length() <= 0 || str.beginswith(fHead)) {
         if (fTail.length() <= 0 || str.endswith(fTail)) {
            str.replace(fHead,"");
            str.replace(fTail,"");
            std::list<XrdProofdMultiStrToken>::iterator it = fTokens.begin();
            for (; it != fTokens.end(); it++) {
               if ((*it).Matches(str.c_str()))
                  return 1;
            }
         }
      }
   }
   // Done
   return 0;
}

//__________________________________________________________________________
XrdOucString XrdProofdMultiStr::Export()
{
   // Return a string with comma-separated elements

   XrdOucString str(fN * (fHead.length() + fTail.length() + 4)) ;
   str = "";
   if (fN > 0) {
      std::list<XrdProofdMultiStrToken>::iterator it = fTokens.begin();
      for (; it != fTokens.end(); it++) {
         int n = (*it).N(), j = -1;
         while (n--) {
            str += fHead;
            str += (*it).Export(j);
            str += fTail;
            str += ",";
         }
      }
   }
   // Remove last ','
   if (str.endswith(','))
      str.erase(str.rfind(','));
   // Done
   return str;
}

//__________________________________________________________________________
XrdOucString XrdProofdMultiStr::Get(int i)
{
   // Return i-th combination (i : 0 -> fN-1)

   XrdOucString str;

   if (i >= 0) {
      std::list<XrdProofdMultiStrToken>::iterator it = fTokens.begin();
      for (; it != fTokens.end(); it++) {
         int n = (*it).N(), j = -1;
         if ((i + 1) > n) {
            i -= n;
         } else {
            j = i;
            str = fHead;
            str += (*it).Export(j);
            str += fTail;
            break;
         }
      }
   }

   // Done
   return str;
}

//__________________________________________________________________________
void XrdProofdMultiStrToken::Init(const char *s)
{
   // Init the multi-string token.
   // Supported formats:
   //    [1-4]   for  1, ..., 4 (4 items)
   //    [a,b]   for  a, b<tail> (2 items)
   //    [a,1-3] for  a, 1, 2, 3 (4 items)
   //    [01-15] for  01, ..., 15 (15 items)
   //
   // A dashed is possible only between numerically treatable values, i.e.
   // single letters ([a-Z] will take all tokens between 'a' and 'Z') or n-field
   // numbers ([001-999] will take all numbers 1 to 999 always using 3 spaces).
   // Mixed values (e.g. [a-034]) are not allowed.
   XPDLOC(AUX, "MultiStrToken::Init")

   fIa = LONG_MAX;
   fIb = LONG_MAX;
   fType = kUndef;
   fN = 0;
   bool bad = 0;
   XrdOucString emsg;
   if (s && strlen(s)) {
      fA = s;
      // Find the dash, if any
      int id = fA.find('-');
      if (id == STR_NPOS) {
         // Simple token, nothing much to do
         fN = 1;
         fType = kSimple;
         return;
      }
      // Define the extremes
      fB.assign(fA, id + 1);
      fA.erase(id);
      if (fB.length() <= 0) {
         if (fA.length() > 0) {
            // Simple token, nothing much to do
            fN = 1;
            fType = kSimple;
         }
         // Invalid
         return;
      }
      // Check validity
      char *a = (char *)fA.c_str();
      char *b = (char *)fB.c_str();
      if (fA.length() == 1 && fB.length() == 1) {
         LETTOIDX(*a, fIa);
         if (fIa != LONG_MAX) {
            LETTOIDX(*b, fIb);
            if (fIb != LONG_MAX && fIa <= fIb) {
               // Ordered single-letter extremes: OK
               fType = kLetter;
               fN = fIb - fIa + 1;
               return;
            }
         } else if (DIGIT(*a) && DIGIT(*b) &&
                   (fIa = *a) <= (fIb = *b)) {
            // Ordered single-digit extremes: OK
            fType = kDigit;
            fN = fIb - fIa + 1;
            return;
         }
         // Not-supported single-field extremes
         emsg = "not-supported single-field extremes";
         bad = 1;
      }
      if (!bad) {
         fIa = fA.atoi();
         if (fIa != LONG_MAX && fIa != LONG_MIN) {
            fIb = fB.atoi();
            if (fIb != LONG_MAX && fIb != LONG_MIN && fIb >= fIa) {
               fType = kDigits;
               fN = fIb - fIa + 1;
               return;
            }
            // Not-supported single-field extremes
            emsg = "non-digit or wrong-ordered extremes";
            bad = 1;
         } else {
            // Not-supported single-field extremes
            emsg = "non-digit extremes";
            bad = 1;
         }
      }
   }
   // Print error message, if any
   if (bad) {
      TRACE(XERR, emsg);
      fA = "";
      fB = "";
      fIa = LONG_MAX;
      fIb = LONG_MAX;
   }
   // Done
   return;
}

//__________________________________________________________________________
bool XrdProofdMultiStrToken::Matches(const char *s)
{
   // Return true if 's' is compatible with this token

   if (s && strlen(s)) {
      if (fType == kSimple)
         return ((fA == s) ? 1 : 0);
      // Multiple one: parse it
      XrdOucString str(s);
      long ls = LONG_MIN;
      if (fType != kDigits) {
         if (str.length() > 1)
            return 0;
         char *ps = (char *)s;
         if (fType == kDigit) {
            if (!DIGIT(*ps) || *ps < fIa || *ps > fIb)
               return 0;
         } else if (fType == kLetter) {
            LETTOIDX(*ps, ls);
            if (ls == LONG_MAX || ls < fIa || ls > fIb)
               return 0;
         }
      } else {
         ls = str.atoi();
         if (ls == LONG_MAX || ls < fIa || ls > fIb)
            return 0;
      }
      // OK
      return 1;
   }
   // Undefined
   return 0;
}

//__________________________________________________________________________
XrdOucString XrdProofdMultiStrToken::Export(int &next)
{
   // Export 'next' token; use next < 0 start from the first

   XrdOucString tkn(fA.length());

   // If simple, return the one we have
   if (fType == kSimple)
      return (tkn = fA);

   // Check if we still have something
   if (next > fIb - fIa)
      return tkn;

   // Check where we are
   if (next == -1)
      next = 0;

   // If letters we need to found the right letter
   if (fType == kLetter) {
      char c = 0;
      IDXTOLET(fIa + next, c);
      next++;
      return (tkn = c);
   }

   // If single digit, add the offset
   if (fType == kDigit) {
      tkn = (char)(fIa + next);
      next++;
      return tkn;
   }

   // If digits, check if we need to pad 0's
   XrdOucString tmp(fA.length());
   tmp.form("%ld", fIa + next);
   next++;
   int dl = fA.length() - tmp.length();
   if (dl <= 0) return tmp;
   // Add padding 0's
   tkn = "";
   while (dl--) tkn += "0";
   tkn += tmp;
   return tkn;
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt,
                                         int ns, const char *ss[5],
                                         int ni, int ii[6],
                                         int np, void *pp[5],
                                         int nu, unsigned int ui)
{
   // Recreate the string according to 'fmt', the up to 5 'const char *',
   // up to 6 'int' arguments, up to 5 'void *' and up to 1 unsigned integer.

   int len = 0;
   if (!fmt || (len = strlen(fmt)) <= 0) return;

   char si[32], sp[32];

   // Estimate length
   int i = ns;
   while (i-- > 0) { if (ss[i]) { len += strlen(ss[i]); } }
   i = ni + np;
   while (i-- > 0) { len += 32; }

   s.resize(len+1);

   int from = 0;
   s.assign(fmt, from);
   int nii = 0, nss = 0, npp = 0, nui = 0;
   int k = STR_NPOS;
   while ((k = s.find('%', from)) != STR_NPOS) {
      bool replaced = 0;
      if (s[k+1] == 's') {
         if (nss < ns) {
            s.replace("%s", ss[nss++], k, k + 1);
            replaced = 1;
         }
      } else if (s[k+1] == 'd') {
         if (nii < ni) {
            sprintf(si,"%d", ii[nii++]);
            s.replace("%d", si, k, k + 1);
            replaced = 1;
         }
      } else if (s[k+1] == 'u') {
         if (nui < nu) {
            sprintf(si,"%u", ui);
            s.replace("%u", si, k, k + 1);
            replaced = 1;
         }
      } else if (s[k+1] == 'p') {
         if (npp < np) {
            sprintf(sp,"%p", pp[npp++]);
            s.replace("%p", sp, k, k + 1);
            replaced = 1;
         }
      }
      if (!replaced) from = k + 1;
   }
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt,
                        const char *s0, const char *s1,
                        const char *s2, const char *s3, const char *s4)
{
   // Recreate the string according to 'fmt' and the 5 'const char *' arguments

   const char *ss[5] = {s0, s1, s2, s3, s4};
   int ii[6] = {0,0,0,0,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,5,ss,0,ii,0,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, int i0,
                                         int i1, int i2, int i3, int i4, int i5)
{
   // Recreate the string according to 'fmt' and the 5 'int' arguments

   const char *ss[5] = {0, 0, 0, 0, 0};
   int ii[6] = {i0,i1,i2,i3,i4,i5};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,0,ss,6,ii,5,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt,
                                         void *p0, void *p1, void *p2, void *p3, void *p4)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {0, 0, 0, 0, 0};
   int ii[6] = {0,0,0,0,0,0};
   void *pp[5] = {p0,p1,p2,p3,p4};

   XrdProofdAux::Form(s,fmt,0,ss,0,ii,5,pp);
}


//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, int i0, const char *s0,
                                     const char *s1, const char *s2, const char *s3)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0, s1, s2, s3, 0};
   int ii[6] = {i0,0,0,0,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,4,ss,1,ii,0,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, const char *s0,
                                     int i0, int i1, int i2, int i3)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,0,0,0,0};
   int ii[6] = {i0,i1,i2,i3,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,1,ss,4,ii,0,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, const char *s0,
                                     int i0, int i1, unsigned int ui)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,0,0,0,0};
   int ii[6] = {i0,i1,0,0,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,1,ss,2,ii,0,pp, 1, ui);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, const char *s0, const char *s1,
                                     int i0, int i1, int i2)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,s1,0,0,0};
   int ii[6] = {i0,i1,i2,0,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,2,ss,3,ii,0,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, int i0, int i1,
                                     const char *s0, const char *s1, const char *s2)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,s1,s2,0,0};
   int ii[6] = {i0,i1,0,0,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,3,ss,2,ii,0,pp);
}


//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, const char *s0,
                                         const char *s1, const char *s2,
                                         int i0, int i1,
                                         const char *s3, const char *s4)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,s1,s2,s3,s4};
   int ii[6] = {i0,i1,0,0,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,5,ss,2,ii,0,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, const char *s0,
                                         int i0, int i1, const char *s1,
                                         const char *s2, const char *s3)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,s1,s2,s3,0};
   int ii[6] = {i0,i1,0,0,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,4,ss,2,ii,0,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, const char *s0,
                                         const char *s1, const char *s2,
                                         int i0, unsigned int ui)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,s1,s2,0,0};
   int ii[6] = {i0,0,0,0,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,3,ss,1,ii,0,pp, 1, ui);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, int i0, int i1, int i2,
                                         const char *s0, const char *s1)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,s1,0,0,0};
   int ii[6] = {i0,i1,i2,0,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,2,ss,3,ii,0,pp);
}


//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, const char *s0,
                          const char *s1, const char *s2, const char *s3, int i0)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,s1,s2,s3,0};
   int ii[6] = {i0,0,0,0,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,4,ss,1,ii,0,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, int i0, int i1, int i2,
                                         int i3, const char *s0)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,0,0,0,0};
   int ii[6] = {i0,i1,i2,i3,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,1,ss,4,ii,0,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, int i0, int i1, void *p0)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {0,0,0,0,0};
   int ii[6] = {i0,i1,0,0,0,0};
   void *pp[5] = {p0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,0,ss,2,ii,1,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt,
                                         int i0, int i1, int i2, void *p0)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {0,0,0,0,0};
   int ii[6] = {i0,i1,i2,0,0,0};
   void *pp[5] = {p0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,0,ss,3,ii,1,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt,
                                         int i0, int i1, int i2, int i3, void *p0)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {0,0,0,0,0};
   int ii[6] = {i0,i1,i2,i3,0,0};
   void *pp[5] = {p0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,0,ss,4,ii,1,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, int i0, int i1,
                                                          void *p0, int i2, int i3)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {0,0,0,0,0};
   int ii[6] = {i0,i1,i2,i3,0,0};
   void *pp[5] = {p0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,0,ss,4,ii,1,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, void *p0, int i0, int i1)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {0,0,0,0,0};
   int ii[6] = {i0,i1,0,0,0,0};
   void *pp[5] = {p0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,0,ss,2,ii,1,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt,
                                         const char *s0, void *p0, int i0, int i1)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,0,0,0,0};
   int ii[6] = {i0,i1,0,0,0,0};
   void *pp[5] = {p0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,1,ss,2,ii,1,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt,
                                         void *p0, const char *s0, int i0)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,0,0,0,0};
   int ii[6] = {i0,0,0,0,0,};
   void *pp[5] = {p0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,1,ss,1,ii,1,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt,
                                         const char *s0, const char *s1, void *p0)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,s1,0,0,0};
   int ii[6] = {0,0,0,0,0,0};
   void *pp[5] = {p0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,2,ss,0,ii,1,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, int i0,
                                         const char *s0, const char *s1, int i1, int i2)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,s1,0,0,0};
   int ii[6] = {i0,i1,i2,0,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,2,ss,3,ii,0,pp);
}

//______________________________________________________________________________
void XrdProofdAux::Form(XrdOucString &s, const char *fmt, int i0,
                                         const char *s0, int i1, int i2)
{
   // Recreate the string according to 'fmt' and the 5 'void *' arguments

   const char *ss[5] = {s0,0,0,0,0};
   int ii[6] = {i0,i1,i2,0,0,0};
   void *pp[5] = {0,0,0,0,0};

   XrdProofdAux::Form(s,fmt,1,ss,3,ii,0,pp);
}

