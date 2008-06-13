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
int XrdProofdAux::AssertDir(const char *path, XrdProofUI ui, bool changeown)
{
   // Make sure that 'path' exists and is owned by the entity
   // described by 'ui'.
   // If changeown is TRUE it tries to acquire the privileges before.
   // Return 0 in case of success, -1 in case of error

   MTRACE(ACT, MHEAD, "AssertDir: enter");

   if (!path || strlen(path) <= 0)
      return -1;

   struct stat st;
   if (stat(path,&st) != 0) {
      if (errno == ENOENT) {

         {  XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
            if (XpdBadPGuard(pGuard, ui.fUid) && changeown) {
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
   if (changeown &&
      ((int) st.st_uid != ui.fUid || (int) st.st_gid != ui.fGid)) {

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
int XrdProofdAux::ChangeToDir(const char *dir, XrdProofUI ui, bool changeown)
{
   // Change current directory to 'dir'.
   // If changeown is TRUE it tries to acquire the privileges before.
   // Return 0 in case of success, -1 in case of error

   MTRACE(ACT, MHEAD, "ChangeToDir: enter: changing to " << ((dir) ? dir : "**undef***"));

   if (!dir || strlen(dir) <= 0)
      return -1;

   if (changeown && (int) geteuid() != ui.fUid) {

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

   // Deprecate
   MPRINT(MHEAD, ">>> Warning: 'if' conditions at the end of the directive are deprecated ");
   MPRINT(MHEAD, ">>> Please use standard Scalla/Xrootd 'if-else-fi' constructs");
   MPRINT(MHEAD, ">>> (see http://xrootd.slac.stanford.edu/doc/xrd_config/xrd_config.htm)");

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
   ncpu = 0;

#if defined(linux)
   // Look for in the /proc/cpuinfo file
   XrdOucString fcpu("/proc/cpuinfo");
   FILE *fc = fopen(fcpu.c_str(), "r");
   if (!fc) {
      if (errno == ENOENT) {
         MPRINT(MHEAD, "GetNumCPUs: /proc/cpuinfo missing!!! Something very bad going on");
      } else {
         XrdOucString emsg("GetNumCPUs: cannot open ");
         emsg += fcpu;
         emsg += ": errno: ";
         emsg += errno;
         MPRINT(MHEAD, emsg.c_str());
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

   MPRINT(MHEAD, "GetNumCPUs: # of cores found: "<<ncpu);

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

   int rc = 0;
   kinfo_proc *res;
   bool done = 0;
   static const int name[] = {CTL_KERN, KERN_PROC, KERN_PROC_ALL, 0};

   MTRACE(ACT, MHEAD, "GetMacProcList: enter ");

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

//
// Functions to process directives for integer and strings
//

//______________________________________________________________________________
int DoDirectiveInt(XrdProofdDirective *d, char *val, XrdOucStream *cfg, bool)
{
   // Process directive for an integer

   if (!d || !(d->fVal) || !val)
      // undefined inputs
      return -1;

   // Check deprecated 'if' directive
   if (d->fHost && cfg)
      if (XrdProofdAux::CheckIf(cfg, d->fHost) == 0)
         return 0;

   int v = strtol(val,0,10);
   *((int *)d->fVal) = v;

   MTRACE(DBG,MHEAD,"DoDirectiveInt: set "<<d->fName<<" to "<<*((int *)d->fVal));

   return 0;
}

//______________________________________________________________________________
int DoDirectiveString(XrdProofdDirective *d, char *val, XrdOucStream *cfg, bool)
{
   // Process directive for a string

   if (!d || !(d->fVal) || !val)
      // undefined inputs
      return -1;

   // Check deprecated 'if' directive
   if (d->fHost && cfg)
      if (XrdProofdAux::CheckIf(cfg, d->fHost) == 0)
         return 0;

   *((XrdOucString *)d->fVal) = val;

   MTRACE(DBG,MHEAD,"DoDirectiveString: set "<<d->fName<<" to "<<*((XrdOucString *)d->fVal));
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

