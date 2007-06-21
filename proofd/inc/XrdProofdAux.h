// @(#)root/proofd:$Name:  $:$Id: XrdProofdAux.h,v 1.1 2007/06/12 13:51:03 ganis Exp $
// Author: G. Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdAux
#define ROOT_XrdProofdAux

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdAux                                                          //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// Small auxilliary classes used in XrdProof                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Xrd/XrdProtocol.hh"
#include "XProofProtocol.h"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucString.hh"

//
// User Info class
//
class XrdProofUI {
public:
   XrdOucString fUser;
   XrdOucString fHomeDir;
   XrdOucString fWorkDir;
   int          fUid;
   int          fGid;

   XrdProofUI() { fUid = -1; fGid = -1; }
   XrdProofUI(const XrdProofUI &ui) { fUser = ui.fUser;
                                      fHomeDir = ui.fHomeDir;
                                      fWorkDir = ui.fWorkDir;
                                      fUid = ui.fUid; fGid = ui.fGid; }
   ~XrdProofUI() { }

   void Reset() { fUser = ""; fHomeDir = ""; fWorkDir = ""; fUid = -1; fGid = -1; }
};

//
// File container (e.g. for config files)
//
class XrdProofdFile {
public:
   XrdOucString  fName;  // File name
   time_t        fMtime; // File mofification time last time we accessed it
   XrdProofdFile(const char *fn = 0, time_t mtime = 0) : fName(fn), fMtime(mtime) { }
};

//
// User priority
//
class XrdProofdPriority {
public:
   XrdOucString            fUser;          // User to who this applies (wild cards accepted)
   int                     fDeltaPriority; // Priority change
   XrdProofdPriority(const char *usr, int dp) : fUser(usr), fDeltaPriority(dp) { }
};

//
// Small class to describe a process
//
class XrdProofdPInfo {
public:
   int pid;
   XrdOucString pname;
   XrdProofdPInfo(int i, const char *n) : pid(i) { pname = n; }
};

//
// Static methods
//
class XrdOucStream;
class XrdProofdAux {
public:
   XrdProofdAux() { }

   static int AssertDir(const char *path, XrdProofUI ui);
   static int ChangeToDir(const char *dir, XrdProofUI ui);
   static int CheckIf(XrdOucStream *s, const char *h);
   static char *Expand(char *p);
   static void Expand(XrdOucString &path);
   static long int GetLong(char *str);
   static int GetNumCPUs();
   static int GetUserInfo(const char *usr, XrdProofUI &ui);
   static int GetUserInfo(int uid, XrdProofUI &ui);
   static int SymLink(const char *path, const char *link);
   static int Write(int fd, const void *buf, size_t nb);
};

// Useful definitions
#ifndef SafeDel
#define SafeDel(x) { if (x) { delete x; x = 0; } }
#endif
#ifndef SafeDelArray
#define SafeDelArray(x) { if (x) { delete[] x; x = 0; } }
#endif
#ifndef SafeFree
#define SafeFree(x) { if (x) free(x); x = 0; }
#endif

#ifndef INRANGE
#define INRANGE(x,y) ((x >= 0) && (x < (int)y->size()))
#endif

#ifndef DIGIT
#define DIGIT(x) (x >= 48 && x <= 57)
#endif

#ifndef XPDSWAP
#define XPDSWAP(a,b,t) { t = a ; a = b; b = t; }
#endif

#ifndef XpdBadPGuard
#define XpdBadPGuard(g,u) (!(g.Valid()) && (geteuid() != (uid_t)u))
#endif

#undef MHEAD
#define MHEAD "--- Proofd: "

#undef  TRACELINK
#define TRACELINK fLink

#undef  RESPONSE
#define RESPONSE fResponse

#endif
