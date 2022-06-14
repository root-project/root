#ifndef __SYS_PRIV_H__
#define __SYS_PRIV_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d S y s P r i v . h h                              */
/*                                                                            */
/* (c) 2006 G. Ganis (CERN)                                                   */
/*                                                                            */
/* This file is part of the XRootD software suite.                            */
/*                                                                            */
/* XRootD is free software: you can redistribute it and/or modify it under    */
/* the terms of the GNU Lesser General Public License as published by the     */
/* Free Software Foundation, either version 3 of the License, or (at your     */
/* option) any later version.                                                 */
/*                                                                            */
/* XRootD is distributed in the hope that it will be useful, but WITHOUT      */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or      */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public       */
/* License for more details.                                                  */
/*                                                                            */
/* You should have received a copy of the GNU Lesser General Public License   */
/* along with XRootD in a file called COPYING.LESSER (LGPL license) and file  */
/* COPYING (GPL license).  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                            */
/* The copyright holder's institutional names and contributor's names may not */
/* be used to endorse or promote products derived from this software without  */
/* specific prior written permission of the institution or contributor.       */
/*     All Rights Reserved. See XrdInfo.cc for complete License Terms         */
/******************************************************************************/

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
// NB: this class can only used via XrdSysPrivGuard (see below)         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#if !defined(WINDOWS)
#  include <sys/types.h>
#else
#  define uid_t unsigned int
#  define gid_t unsigned int
#endif

#include "XrdSys/XrdSysPthread.hh"

class XrdSysPriv
{
 friend class XrdSysPrivGuard;
 private:
   // Ownership cannot be changed by thread, so there must be an overall
   // locking
   static XrdSysRecMutex fgMutex;

   XrdSysPriv();

   static bool fDebug;

   static int ChangeTo(uid_t uid, gid_t gid);
   static void DumpUGID(const char *msg = 0);
   static int Restore(bool saved = 1);

 public:
   virtual ~XrdSysPriv() { }
   static int ChangePerm(uid_t uid, gid_t gid);
};

//
// Guard class;
// Usage:
//
//    {  XrdSysPrivGuard priv(tempuid);
//
//       // Work as tempuid (maybe superuser)
//       ...
//
//    }
//
class XrdSysPrivGuard
{
 public:
   XrdSysPrivGuard(uid_t uid, gid_t gid);
   XrdSysPrivGuard(const char *user);
   virtual ~XrdSysPrivGuard();
   bool Valid() const { return valid; }
 private:
   bool dum;
   bool valid;
   void Init(uid_t uid, gid_t gid);
};

#endif
