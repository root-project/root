/******************************************************************************/
/*                                                                            */
/*                     X r d S y s P r i v . c c                              */
/*                                                                            */
/* (c) 2006 G. Ganis (CERN)                                                   */
/*     All Rights Reserved. See XrdInfo.cc for complete License Terms         */
/******************************************************************************/
// $Id$

const char *XrdSysPrivCVSID = "$Id$";

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdSysPriv                                                           //
//                                                                      //
// Author: G. Ganis, CERN, 2006                                         //
//                                                                      //
// Implementation of a privileges handling API following the paper      //
//   "Setuid Demystified" by H.Chen, D.Wagner, D.Dean                   //
// also quoted in "Secure programming Cookbook" by J.Viega & M.Messier. //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdSys/XrdSysPriv.hh"

#if !defined(WINDOWS)
#include <stdio.h>
#include "XrdSys/XrdSysHeaders.hh"
#include <unistd.h>
#include <pwd.h>
#include <errno.h>

#define NOUC ((uid_t)(-1))
#define NOGC ((gid_t)(-1))
#define XSPERR(x) ((x == 0) ? -1 : -x)

// Some machine specific stuff
#if defined(__sgi) && !defined(__GNUG__) && (SGI_REL<62)
extern "C" {
   int seteuid(int euid);
   int setegid(int egid);
   int geteuid();
   int getegid();
}
#endif

#if defined(_AIX)
extern "C" {
   int seteuid(uid_t euid);
   int setegid(gid_t egid);
   uid_t geteuid();
   gid_t getegid();
}
#endif

#if !defined(HAVE_SETRESUID)
static int setresgid(gid_t r, gid_t e, gid_t)
{
   if (r != NOGC && setgid(r) == -1)
      return XSPERR(errno);
   return ((e != NOGC) ? setegid(e) : 0);
}

static int setresuid(uid_t r, uid_t e, uid_t)
{
   if (r != NOUC && setuid(r) == -1)
      return XSPERR(errno);
   return ((e != NOUC) ? seteuid(e) : 0);
}

static int getresgid(gid_t *r, gid_t *e, gid_t *)
{
  *r = getgid();
  *e = getegid();
  return 0;
}

static int getresuid(uid_t *r, uid_t *e, uid_t *)
{
  *r = getuid();
  *e = geteuid();
  return 0;
}

#else
#if (defined(__linux__) || \
    (defined(__CYGWIN__) && defined(__GNUC__))) && !defined(linux)
#   define linux
#endif
#if defined(linux) && !defined(HAVE_SETRESUID)
extern "C" {
   int setresgid(gid_t r, gid_t e, gid_t s);
   int setresuid(uid_t r, uid_t e, uid_t s);
   int getresgid(gid_t *r, gid_t *e, gid_t *s);
   int getresuid(uid_t *r, uid_t *e, uid_t *s);
}
#endif
#endif
#endif // not WINDOWS

bool XrdSysPriv::fDebug = 0; // debug switch

// Gloval mutex
XrdSysRecMutex XrdSysPriv::fgMutex;

//______________________________________________________________________________
int XrdSysPriv::Restore(bool saved)
{
   // Restore the 'saved' (saved = TRUE) or 'real' entity as effective.
   // Return 0 on success, < 0 (== -errno) if any error occurs.

#if !defined(WINDOWS)
   // Get the UIDs
   uid_t ruid = 0, euid = 0, suid = 0;
   if (getresuid(&ruid, &euid, &suid) != 0)
      return XSPERR(errno);

   // Set the wanted value
   uid_t uid = saved ? suid : ruid;

   // Act only if a change is needed
   if (euid != uid) {

      // Set uid as effective
      if (setresuid(NOUC, uid, NOUC) != 0)
         return XSPERR(errno);

      // Make sure the new effective UID is the one wanted
      if (geteuid() != uid)
         return XSPERR(errno);
   }

   // Get the GIDs
   uid_t rgid = 0, egid = 0, sgid = 0;
   if (getresgid(&rgid, &egid, &sgid) != 0)
      return XSPERR(errno);

   // Set the wanted value
   gid_t gid = saved ? sgid : rgid;

   // Act only if a change is needed
   if (egid != gid) {

      // Set newuid as effective, saving the current effective GID
      if (setresgid(NOGC, gid, NOGC) != 0)
         return XSPERR(errno);

      // Make sure the new effective GID is the one wanted
      if (getegid() != gid)
         return XSPERR(errno);
   }

#endif
   // Done
   return 0;
}

//______________________________________________________________________________
int XrdSysPriv::ChangeTo(uid_t newuid, gid_t newgid)
{
   // Change effective to entity newuid. Current entity is saved.
   // Real entity is not touched. Use RestoreSaved to go back to
   // previous settings.
   // Return 0 on success, < 0 (== -errno) if any error occurs.

#if !defined(WINDOWS)
   // Current UGID 
   uid_t oeuid = geteuid();
   gid_t oegid = getegid();

   // Restore privileges, if needed
   if (oeuid && XrdSysPriv::Restore(0) != 0)
      return XSPERR(errno);

   // Act only if a change is needed
   if (newgid != oegid) {

      // Set newgid as effective, saving the current effective GID
      if (setresgid(NOGC, newgid, oegid) != 0)
         return XSPERR(errno);

      // Get the GIDs
      uid_t rgid = 0, egid = 0, sgid = 0;
      if (getresgid(&rgid, &egid, &sgid) != 0)
         return XSPERR(errno);

      // Make sure the new effective GID is the one wanted
      if (egid != newgid)
         return XSPERR(errno);
   }

   // Act only if a change is needed
   if (newuid != oeuid) {

      // Set newuid as effective, saving the current effective UID
      if (setresuid(NOUC, newuid, oeuid) != 0)
         return XSPERR(errno);

      // Get the UIDs
      uid_t ruid = 0, euid = 0, suid = 0;
      if (getresuid(&ruid, &euid, &suid) != 0)
         return XSPERR(errno);

      // Make sure the new effective UID is the one wanted
      if (euid != newuid)
         return XSPERR(errno);
   }

#endif
   // Done
   return 0;
}

//______________________________________________________________________________
int XrdSysPriv::ChangePerm(uid_t newuid, gid_t newgid)
{
   // Change permanently to entity newuid. Requires super-userprivileges.
   // Provides a way to drop permanently su privileges.
   // Return 0 on success, < 0 (== -errno) if any error occurs.

   // Atomic action
   XrdSysPriv::fgMutex.Lock();
#if !defined(WINDOWS)
   // Get UIDs
   uid_t cruid = 0, ceuid = 0, csuid = 0;
   if (getresuid(&cruid, &ceuid, &csuid) != 0) {
      XrdSysPriv::fgMutex.UnLock();
      return XSPERR(errno);
   }

   // Get GIDs
   uid_t crgid = 0, cegid = 0, csgid = 0;
   if (getresgid(&crgid, &cegid, &csgid) != 0) {
      XrdSysPriv::fgMutex.UnLock();
      return XSPERR(errno);
   }
   // Restore privileges, if needed
   if (ceuid && XrdSysPriv::Restore(0) != 0) {
      XrdSysPriv::fgMutex.UnLock();
      return XSPERR(errno);
   }
   // Act only if needed
   if (newgid != cegid || newgid != crgid) {

      // Set newgid as GID, all levels
      if (setresgid(newgid, newgid, newgid) != 0) {
         XrdSysPriv::fgMutex.UnLock();
         return XSPERR(errno);
      }
      // Get GIDs
      uid_t rgid = 0, egid = 0, sgid = 0;
      if (getresgid(&rgid, &egid, &sgid) != 0) {
         XrdSysPriv::fgMutex.UnLock();
         return XSPERR(errno);
      }
      // Make sure the new GIDs are all equal to the one asked
      if (rgid != newgid || egid != newgid) {
         XrdSysPriv::fgMutex.UnLock();
         return XSPERR(errno);
      }
   }

   // Act only if needed
   if (newuid != ceuid || newuid != cruid) {

      // Set newuid as UID, all levels
      if (setresuid(newuid, newuid, newuid) != 0) {
         XrdSysPriv::fgMutex.UnLock();
         return XSPERR(errno);
      }
      // Get UIDs
      uid_t ruid = 0, euid = 0, suid = 0;
      if (getresuid(&ruid, &euid, &suid) != 0) {
         XrdSysPriv::fgMutex.UnLock();
         return XSPERR(errno);
      }
      // Make sure the new UIDs are all equal to the one asked 
      if (ruid != newuid || euid != newuid) {
         XrdSysPriv::fgMutex.UnLock();
         return XSPERR(errno);
      }
   }
#endif
   // Release the mutex
   XrdSysPriv::fgMutex.UnLock();

   // Done
   return 0;
}

//______________________________________________________________________________
void XrdSysPriv::DumpUGID(const char *msg)
{
   // Dump current entity

#if !defined(WINDOWS)
   XrdSysPriv::fgMutex.Lock();
   // Get the UIDs
   uid_t ruid = 0, euid = 0, suid = 0;
   if (getresuid(&ruid, &euid, &suid) != 0)
      return;

   // Get the GIDs
   uid_t rgid = 0, egid = 0, sgid = 0;
   if (getresgid(&rgid, &egid, &sgid) != 0)
      return;

   cout << "XrdSysPriv: "  << endl; 
   cout << "XrdSysPriv: dump values: " << (msg ? msg : "") << endl; 
   cout << "XrdSysPriv: "  << endl; 
   cout << "XrdSysPriv: real       = (" << ruid <<","<< rgid <<")" << endl; 
   cout << "XrdSysPriv: effective  = (" << euid <<","<< egid <<")" << endl; 
   cout << "XrdSysPriv: saved      = (" << suid <<","<< sgid <<")" << endl; 
   cout << "XrdSysPriv: "  << endl; 
   XrdSysPriv::fgMutex.UnLock();
#endif
}

//
// Guard class
//______________________________________________________________________________
XrdSysPrivGuard::XrdSysPrivGuard(uid_t uid, gid_t gid)
{
   // Constructor. Create a guard object for temporarly change to privileges
   // of {'uid', 'gid'}

   dum = 1;
   valid = 0;

   Init(uid, gid);
}

//______________________________________________________________________________
XrdSysPrivGuard::XrdSysPrivGuard(const char *usr)
{
   // Constructor. Create a guard object for temporarly change to privileges
   // of 'usr'

   dum = 1;
   valid = 0;

#if !defined(WINDOWS)
   if (usr && strlen(usr) > 0) {
      struct passwd *pw = getpwnam(usr);
      if (pw)
         Init(pw->pw_uid, pw->pw_gid);
   }
#else
   if (usr) { }
#endif
}

//______________________________________________________________________________
XrdSysPrivGuard::~XrdSysPrivGuard()
{
   // Destructor. Restore state and unlock the global mutex.

   if (!dum) {
      XrdSysPriv::Restore();
      XrdSysPriv::fgMutex.UnLock();
   }
}

//______________________________________________________________________________
void XrdSysPrivGuard::Init(uid_t uid, gid_t gid)
{
   // Init a change of privileges guard. Act only if superuser.
   // The result of initialization can be tested with the Valid() method.

   dum = 1;
   valid = 1;

   // Debug hook
   if (XrdSysPriv::fDebug)
      XrdSysPriv::DumpUGID("before Init()");

#if !defined(WINDOWS)
   XrdSysPriv::fgMutex.Lock();
   uid_t ruid = 0, euid = 0, suid = 0;
   gid_t rgid = 0, egid = 0, sgid = 0;
   if (getresuid(&ruid, &euid, &suid) == 0 &&
       getresgid(&rgid, &egid, &sgid) == 0) {
      if ((euid != uid) || (egid != gid)) {
         if (!ruid) {
            // Change temporarly identity
            if (XrdSysPriv::ChangeTo(uid, gid) != 0)
               valid = 0;
            dum = 0;
         } else {
            // Change requested but not enough privileges
            valid = 0;
         }
      }
   } else {
      // Something bad happened: memory corruption?
      valid = 0;
   }
   // Unlock if no action
   if (dum)
      XrdSysPriv::fgMutex.UnLock();
#endif
   // Debug hook
   if (XrdSysPriv::fDebug)
      XrdSysPriv::DumpUGID("after Init()");
}
