// @(#)root/proofd:$Name:  $:$Id: XrdProofdAux.cxx,v 1.1 2007/06/12 13:51:03 ganis Exp $
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

#include "XrdProofdAux.h"
#include "XrdProofdProtocol.h"

// Tracing
#include "XrdProofdTrace.h"
extern XrdOucTrace *XrdProofdTrace;

// Local definitions
#ifdef XPD_LONG_MAX
#undefine XPD_LONG_MAX
#endif
#define XPD_LONG_MAX 2147483647

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
      return XPD_LONG_MAX;

   // Find the last digit
   int j = 0;
   while (*(p+j) >= 48 && *(p+j) <= 57)
      j++;
   *(p+j) = '\0';

   // Convert now
   return strtol(p, 0, 10);
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
   if (uid <= 0)
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

//_____________________________________________________________________________
int XrdProofdAux::AssertDir(const char *path, XrdProofUI ui)
{
   // Make sure that 'path' exists and is owned by the entity
   // described by 'ui'
   // Return 0 in case of success, -1 in case of error

   MTRACE(ACT, MHEAD, "AssertDir: enter");

   if (!path || strlen(path) <= 0)
      return -1;

   struct stat st;
   if (stat(path,&st) != 0) {
      if (errno == ENOENT) {

         {  XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
            if (XpdBadPGuard(pGuard, ui.fUid)) {
               MERROR(MHEAD, "AsserDir: could not get privileges");
               return -1;
            }

            if (mkdir(path, 0755) != 0) {
               MERROR(MHEAD, "AssertDir: unable to create dir: "<<path<<
                             " (errno: "<<errno<<")");
               return -1;
            }
         }
         if (stat(path,&st) != 0) {
            MERROR(MHEAD, "AssertDir: unable to stat dir: "<<path<<
                          " (errno: "<<errno<<")");
            return -1;
         }
      } else {
         // Failure: stop
         MERROR(MHEAD, "AssertDir: unable to stat dir: "<<path<<
                       " (errno: "<<errno<<")");
         return -1;
      }
   }

   // Make sure the ownership is right
   if ((int) st.st_uid != ui.fUid || (int) st.st_gid != ui.fGid) {

      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (XpdBadPGuard(pGuard, ui.fUid)) {
         MERROR(MHEAD, "AsserDir: could not get privileges");
         return -1;
      }

      // Set ownership of the path to the client
      if (chown(path, ui.fUid, ui.fGid) == -1) {
         MERROR(MHEAD, "AssertDir: cannot set user ownership"
                       " on path (errno: "<<errno<<")");
         return -1;
      }
   }

   // We are done
   return 0;
}

//_____________________________________________________________________________
int XrdProofdAux::ChangeToDir(const char *dir, XrdProofUI ui)
{
   // Change current directory to 'dir'.
   // Return 0 in case of success, -1 in case of error

   MTRACE(ACT, MHEAD, "ChangeToDir: enter: changing to " << ((dir) ? dir : "**undef***"));

   if (!dir || strlen(dir) <= 0)
      return -1;

   if ((int) geteuid() != ui.fUid) {

      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (XpdBadPGuard(pGuard, ui.fUid)) {
         MTRACE(XERR, "xpd:child: ", "ChangeToDir: could not get privileges");
         return -1;
      }
      if (chdir(dir) == -1) {
         MTRACE(XERR, "xpd:child: ", "ChangeToDir: can't change directory to "<< dir);
         return -1;
      }
   } else {
      if (chdir(dir) == -1) {
         MTRACE(XERR, "xpd:child: ", "ChangeToDir: can't change directory to "<< dir);
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

   MTRACE(ACT, MHEAD, "SymLink: enter");

   if (!path || strlen(path) <= 0 || !link || strlen(link) <= 0)
      return -1;

   // Remove existing link, if any
   if (unlink(link) != 0 && errno != ENOENT) {
      MERROR(MHEAD, "SymLink: problems unlinking existing symlink "<< link<<
                    " (errno: "<<errno<<")");
      return -1;
   }
   if (symlink(path, link) != 0) {
      MERROR(MHEAD, "SymLink: problems creating symlink " << link<<
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

   // There must be an 'if'
   char *val = s ? s->GetToken() : 0;
   if (!val || strncmp(val,"if",2)) {
      if (val)
         // allow the analysis of the token
         s->RetToken();
      return -1;
   }

   // check value if any
   val = s->GetToken();
   if (!val)
      return -1;

   // Notify
   MTRACE(DBG, MHEAD, "CheckIf: <pattern>: " <<val);

   // Return number of chars matching
   XrdOucString h(host);
   return h.matches((const char *)val);
}

//______________________________________________________________________________
int XrdProofdAux::GetNumCPUs()
{
   // Find out and return the number of CPUs in the local machine.
   // Return -1 in case of failure.

   static int ncpu = -1;

   // Use cached value, if any
   if (ncpu > 0)
      return ncpu;

#if defined(linux)
   // Look for in the /proc/cpuinfo file
   XrdOucString fcpu("/proc/cpuinfo");
   FILE *fc = fopen(fcpu.c_str(), "r");
   if (!fc) {
      if (errno == ENOENT) {
         MTRACE(DBG, MHEAD, "GetNumCPUs: /proc/cpuinfo missing!!! Something very bad going on");
      } else {
         XrdOucString emsg("GetNumCPUs: cannot open ");
         emsg += fcpu;
         emsg += ": errno: ";
         emsg += errno;
         MTRACE(XERR, MHEAD, emsg.c_str());
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

   // Done
   return (ncpu <= 0) ? (int)(-1) : ncpu ;
}
