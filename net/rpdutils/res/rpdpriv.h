// @(#)root/rpdutils:$Id$
// Author: Gerardo Ganis, March 2011

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_rpdpriv
#define ROOT_rpdpriv

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// rpdpriv                                                              //
//                                                                      //
// Implementation of a privileges handling API following the paper      //
//   "Setuid Demystified" by H.Chen, D.Wagner, D.Dean                   //
// also quoted in "Secure programming Cookbook" by J.Viega & M.Messier. //
//                                                                      //
// NB: this not thread-safe: it is meant to be used in single-threaded  //
//     applications                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#if !defined(WINDOWS)
#  include <sys/types.h>
#else
#  define uid_t unsigned int
#  define gid_t unsigned int
#endif

class rpdpriv
{
 friend class rpdprivguard;
 private:

   rpdpriv();

   static bool debug;

   static int  changeto(uid_t uid, gid_t gid);
   static void dumpugid(const char *msg = 0);
   static int  restore(bool saved = 1);

 public:
   virtual ~rpdpriv() { }
   static int changeperm(uid_t uid, gid_t gid);
};

//
// Guard class;
// Usage:
//
//    {  rpdprivguard priv(tempuid);
//
//       // Work as tempuid (maybe superuser)
//       ...
//
//    }
//
class rpdprivguard
{
 public:
   rpdprivguard(uid_t uid, gid_t gid);
   rpdprivguard(const char *user);
   virtual ~rpdprivguard();
   bool isvalid() const { return valid; }
 private:
   bool dum;
   bool valid;
   void init(uid_t uid, gid_t gid);
};

#ifndef rpdbadpguard
#define rpdbadpguard(g,u) (!(g.isvalid()) && (geteuid() != (uid_t)u))
#endif

#endif
