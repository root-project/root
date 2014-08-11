// @(#)root/rpdutils:$Id$
// Author: Gerardo Ganis, March 2011

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// rpdpriv                                                              //
//                                                                      //
// Implementation of a privileges handling API following the paper      //
//   "Setuid Demystified" by H.Chen, D.Wagner, D.Dean                   //
// also quoted in "Secure programming Cookbook" by J.Viega & M.Messier. //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "rpdpriv.h"
#include "RConfigure.h"

#if !defined(WINDOWS)
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <pwd.h>
#include <errno.h>
#include <iostream>
using namespace std;

#define NOUC ((uid_t)(-1))
#define NOGC ((gid_t)(-1))
#define RPPERR(x) ((x == 0) ? -1 : -x)

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

#if !defined(R__HAS_SETRESUID)
static int setresgid(gid_t r, gid_t e, gid_t)
{
   if (r != NOGC && setgid(r) == -1)
      return RPPERR(errno);
   return ((e != NOGC) ? setegid(e) : 0);
}

static int setresuid(uid_t r, uid_t e, uid_t)
{
   if (r != NOUC && setuid(r) == -1)
      return RPPERR(errno);
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
#if defined(linux) && !defined(R__HAS_SETRESUID)
extern "C" {
   int setresgid(gid_t r, gid_t e, gid_t s);
   int setresuid(uid_t r, uid_t e, uid_t s);
   int getresgid(gid_t *r, gid_t *e, gid_t *s);
   int getresuid(uid_t *r, uid_t *e, uid_t *s);
}
#endif
#endif
#endif // not WINDOWS

bool rpdpriv::debug = 0; // debug switch

//______________________________________________________________________________
int rpdpriv::restore(bool saved)
{
   // Restore the 'saved' (saved = TRUE) or 'real' entity as effective.
   // Return 0 on success, < 0 (== -errno) if any error occurs.

#if !defined(WINDOWS)
   // Get the UIDs
   uid_t ruid = 0, euid = 0, suid = 0;
   if (getresuid(&ruid, &euid, &suid) != 0)
      return RPPERR(errno);

   // Set the wanted value
   uid_t uid = saved ? suid : ruid;

   // Act only if a change is needed
   if (euid != uid) {

      // Set uid as effective
      if (setresuid(NOUC, uid, NOUC) != 0)
         return RPPERR(errno);

      // Make sure the new effective UID is the one wanted
      if (geteuid() != uid)
         return RPPERR(errno);
   }

   // Get the GIDs
   uid_t rgid = 0, egid = 0, sgid = 0;
   if (getresgid(&rgid, &egid, &sgid) != 0)
      return RPPERR(errno);

   // Set the wanted value
   gid_t gid = saved ? sgid : rgid;

   // Act only if a change is needed
   if (egid != gid) {

      // Set newuid as effective, saving the current effective GID
      if (setresgid(NOGC, gid, NOGC) != 0)
         return RPPERR(errno);

      // Make sure the new effective GID is the one wanted
      if (getegid() != gid)
         return RPPERR(errno);
   }

#endif
   // Done
   return 0;
}

//______________________________________________________________________________
int rpdpriv::changeto(uid_t newuid, gid_t newgid)
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
   if (oeuid && rpdpriv::restore(0) != 0)
      return RPPERR(errno);

   // Act only if a change is needed
   if (newgid != oegid) {

      // Set newgid as effective, saving the current effective GID
      if (setresgid(NOGC, newgid, oegid) != 0)
         return RPPERR(errno);

      // Get the GIDs
      uid_t rgid = 0, egid = 0, sgid = 0;
      if (getresgid(&rgid, &egid, &sgid) != 0)
         return RPPERR(errno);

      // Make sure the new effective GID is the one wanted
      if (egid != newgid)
         return RPPERR(errno);
   }

   // Act only if a change is needed
   if (newuid != oeuid) {

      // Set newuid as effective, saving the current effective UID
      if (setresuid(NOUC, newuid, oeuid) != 0)
         return RPPERR(errno);

      // Get the UIDs
      uid_t ruid = 0, euid = 0, suid = 0;
      if (getresuid(&ruid, &euid, &suid) != 0)
         return RPPERR(errno);

      // Make sure the new effective UID is the one wanted
      if (euid != newuid)
         return RPPERR(errno);
   }

#endif
   // Done
   return 0;
}

//______________________________________________________________________________
int rpdpriv::changeperm(uid_t newuid, gid_t newgid)
{
   // Change permanently to entity newuid. Requires super-userprivileges.
   // Provides a way to drop permanently su privileges.
   // Return 0 on success, < 0 (== -errno) if any error occurs.

   // Atomic action
#if !defined(WINDOWS)
   // Get UIDs
   uid_t cruid = 0, ceuid = 0, csuid = 0;
   if (getresuid(&cruid, &ceuid, &csuid) != 0) {
      return RPPERR(errno);
   }

   // Get GIDs
   uid_t crgid = 0, cegid = 0, csgid = 0;
   if (getresgid(&crgid, &cegid, &csgid) != 0) {
      return RPPERR(errno);
   }
   // Restore privileges, if needed
   if (ceuid && rpdpriv::restore(0) != 0) {
      return RPPERR(errno);
   }
   // Act only if needed
   if (newgid != cegid || newgid != crgid) {

      // Set newgid as GID, all levels
      if (setresgid(newgid, newgid, newgid) != 0) {
         return RPPERR(errno);
      }
      // Get GIDs
      uid_t rgid = 0, egid = 0, sgid = 0;
      if (getresgid(&rgid, &egid, &sgid) != 0) {
         return RPPERR(errno);
      }
      // Make sure the new GIDs are all equal to the one asked
      if (rgid != newgid || egid != newgid) {
         return RPPERR(errno);
      }
   }

   // Act only if needed
   if (newuid != ceuid || newuid != cruid) {

      // Set newuid as UID, all levels
      if (setresuid(newuid, newuid, newuid) != 0) {
         return RPPERR(errno);
      }
      // Get UIDs
      uid_t ruid = 0, euid = 0, suid = 0;
      if (getresuid(&ruid, &euid, &suid) != 0) {
         return RPPERR(errno);
      }
      // Make sure the new UIDs are all equal to the one asked
      if (ruid != newuid || euid != newuid) {
         return RPPERR(errno);
      }
   }
#endif

   // Done
   return 0;
}

//______________________________________________________________________________
void rpdpriv::dumpugid(const char *msg)
{
   // Dump current entity

#if !defined(WINDOWS)
   // Get the UIDs
   uid_t ruid = 0, euid = 0, suid = 0;
   if (getresuid(&ruid, &euid, &suid) != 0)
      return;

   // Get the GIDs
   uid_t rgid = 0, egid = 0, sgid = 0;
   if (getresgid(&rgid, &egid, &sgid) != 0)
      return;

   cout << "rpdpriv: "  << endl;
   cout << "rpdpriv: dump values: " << (msg ? msg : "") << endl;
   cout << "rpdpriv: "  << endl;
   cout << "rpdpriv: real       = (" << ruid <<","<< rgid <<")" << endl;
   cout << "rpdpriv: effective  = (" << euid <<","<< egid <<")" << endl;
   cout << "rpdpriv: saved      = (" << suid <<","<< sgid <<")" << endl;
   cout << "rpdpriv: "  << endl;
#endif
}

//
// Guard class
//______________________________________________________________________________
rpdprivguard::rpdprivguard(uid_t uid, gid_t gid)
{
   // Constructor. Create a guard object for temporarly change to privileges
   // of {'uid', 'gid'}

   dum = 1;
   valid = 0;

   init(uid, gid);
}

//______________________________________________________________________________
rpdprivguard::rpdprivguard(const char *usr)
{
   // Constructor. Create a guard object for temporarly change to privileges
   // of 'usr'

   dum = 1;
   valid = 0;

#if !defined(WINDOWS)
   if (usr && strlen(usr) > 0) {
      struct passwd *pw = getpwnam(usr);
      if (pw)
         init(pw->pw_uid, pw->pw_gid);
   }
#else
   if (usr) { }
#endif
}

//______________________________________________________________________________
rpdprivguard::~rpdprivguard()
{
   // Destructor. Restore state and unlock the global mutex.

   if (!dum) {
      rpdpriv::restore();
   }
}

//______________________________________________________________________________
void rpdprivguard::init(uid_t uid, gid_t gid)
{
   // Init a change of privileges guard. Act only if superuser.
   // The result of initialization can be tested with the Valid() method.

   dum = 1;
   valid = 1;

   // Debug hook
   if (rpdpriv::debug)
      rpdpriv::dumpugid("before init()");

#if !defined(WINDOWS)
   uid_t ruid = 0, euid = 0, suid = 0;
   gid_t rgid = 0, egid = 0, sgid = 0;
   if (getresuid(&ruid, &euid, &suid) == 0 &&
       getresgid(&rgid, &egid, &sgid) == 0) {
      if ((euid != uid) || (egid != gid)) {
         if (!ruid) {
            // Change temporarly identity
            if (rpdpriv::changeto(uid, gid) != 0)
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
#endif
   // Debug hook
   if (rpdpriv::debug)
      rpdpriv::dumpugid("after init()");
}
