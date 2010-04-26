// @(#)root/proofd:$Id$
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
#include <list>
#include <map>
#include <stdarg.h>

#ifdef OLDXRDOUC
#  include "XrdSysToOuc.h"
#  include "XrdOuc/XrdOucSemWait.hh"
#else
#  include "XrdSys/XrdSysSemWait.hh"
#endif

#include "Xrd/XrdProtocol.hh"
#include "XProofProtocol.h"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucString.hh"

//
// User Info class
//
class XrdProofUI {
public:
   XrdOucString fUser;       // User name
   XrdOucString fGroup;      // PROOF group name
   XrdOucString fHomeDir;    // Unix home
   int          fUid;        // Unix user ID
   int          fGid;        // Unix group ID

   XrdProofUI() { fUid = -1; fGid = -1; }
   XrdProofUI(const XrdProofUI &ui) { fUser = ui.fUser;
                                      fGroup = ui.fGroup;
                                      fHomeDir = ui.fHomeDir;
                                      fUid = ui.fUid; fGid = ui.fGid; }
   ~XrdProofUI() { }

   void Reset() { fUser = ""; fHomeDir = ""; fGroup = ""; fUid = -1; fGid = -1; }
};

//
// Group Info class
//
class XrdProofGI {
public:
   XrdOucString fGroup;
   int          fGid;

   XrdProofGI() { fGid = -1; }
   XrdProofGI(const XrdProofGI &gi) { fGroup = gi.fGroup; fGid = gi.fGid; }
   ~XrdProofGI() { }

   void Reset() { fGroup = ""; fGid = -1; }
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
// Class to handle configuration directives
//
class XrdProofdDirective;
class XrdOucStream;
typedef int (*XrdFunDirective_t)(XrdProofdDirective *, char *,
                                 XrdOucStream *cfg, bool reconfig);
class XrdProofdDirective {
public:
   void              *fVal;
   XrdOucString       fName;
   XrdFunDirective_t  fFun;
   bool               fRcf;
   const char        *fHost; // needed to support old 'if' construct

   XrdProofdDirective(const char *n, void *v, XrdFunDirective_t f, bool rcf = 1) :
                      fVal(v), fName(n), fFun(f), fRcf(rcf) { }

   int DoDirective(char *val, XrdOucStream *cfg, bool reconfig)
                      { return (*fFun)(this, val, cfg, reconfig); }
};
// Function of general interest
int DoDirectiveClass(XrdProofdDirective *, char *val, XrdOucStream *cfg, bool rcf);
int DoDirectiveInt(XrdProofdDirective *, char *val, XrdOucStream *cfg, bool rcf);
int DoDirectiveString(XrdProofdDirective *, char *val, XrdOucStream *cfg, bool rcf);
// To set the host field in a loop over the hash list
int SetHostInDirectives(const char *, XrdProofdDirective *d, void *h);

//
// Class to handle condensed multi-string specification, e.g <head>[01-25]<tail>
//
class XrdProofdMultiStrToken {
private:
   long         fIa;
   long         fIb;
   XrdOucString fA;
   XrdOucString fB;
   int          fType;
   int          fN;     // Number of combinations

   void Init(const char *s);
public:
   enum ETokenType { kUndef, kSimple, kLetter, kDigit, kDigits };

   XrdProofdMultiStrToken(const char *s = 0) { Init(s); }
   virtual ~XrdProofdMultiStrToken() { }

   XrdOucString Export(int &next);
   bool IsValid() const { return (fType == kUndef) ? 0 : 1; }
   bool Matches(const char *s);
   int  N() const { return fN; }
};

class XrdProofdMultiStr {
private:
   XrdOucString fHead;
   XrdOucString fTail;
   std::list<XrdProofdMultiStrToken> fTokens;
   int          fN;     // Number of combinations

   void Init(const char *s);
public:
   XrdProofdMultiStr(const char *s) { Init(s); }
   virtual ~XrdProofdMultiStr() { }

   XrdOucString Get(int i);
   bool IsValid() const { return (fTokens.size() > 0 ? 1 : 0); }
   bool Matches(const char *s);
   int  N() const { return fN; }

   XrdOucString Export();
};

//
// Class to handle message buffers received via a pipe
//
class XpdMsg {
   int          fType;
   XrdOucString fBuf;
   int          fFrom;
public:
   XpdMsg(const char *buf = 0) { Init(buf); }
   virtual ~XpdMsg() { }

   const char *Buf() const {return fBuf.c_str(); }

   int Init(const char *buf);
   void Reset() { fFrom = 0; }

   int Get(int &i);
   int Get(XrdOucString &s);
   int Get(void **p);

   int Type() const { return fType; }
};

//
// Class describing a pipe
//
class XrdProofdPipe {
   XrdSysRecMutex fRdMtx;   // Mutex for read operations
   XrdSysRecMutex fWrMtx;   // Mutex for write operations
   int            fPipe[2]; // pipe descriptors
public:
   XrdProofdPipe();
   virtual ~XrdProofdPipe();

   void Close();
   bool IsValid() const { return (fPipe[0] > 0 && fPipe[1] > 0) ? 1 : 0; }

   int Poll(int to = -1);

   int Post(int type, const char *msg);
   int Recv(XpdMsg &msg);
};

//
// Container for DS information
//
class XrdProofdDSInfo {
public:
   XrdOucString  fType;  // Backend type
   XrdOucString  fUrl;   // URL from where to take the information
   bool          fLocal; // TRUE if on the local file system
   bool          fRW;    // TRUE if users can modify their area
   XrdOucString  fOpts;  // Options for this source
   XrdProofdDSInfo(const char *t, const char *u, bool local, bool rw,
                   const char *o = "Ar:Av:") : 
                   fType(t), fUrl(u), fLocal(local), fRW(rw), fOpts(o) { }
};

//
// Static methods
//
#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)
typedef struct kinfo_proc kinfo_proc;
#endif
class XrdOucStream;
class XrdProofdAux {
   static XrdSysRecMutex fgFormMutex;
public:
   XrdProofdAux() { }

   static const char *AdminMsgType(int type);
   static int AssertDir(const char *path, XrdProofUI ui, bool changeown);
   static int ChangeMod(const char *path, unsigned int mode);
   static int ChangeOwn(const char *path, XrdProofUI ui);
   static int ChangeToDir(const char *dir, XrdProofUI ui, bool changeown);
   static int CheckIf(XrdOucStream *s, const char *h);
   static char *Expand(char *p);
   static void Expand(XrdOucString &path);
   // String form functions
   static void Form(XrdOucString &s, const char *fmt, int ns, const char *ss[5], int ni, int ii[5],
                                     int np, void *pp[5]);
   static void Form(XrdOucString &s, const char *fmt, const char *s0, const char *s1 = 0,
                                     const char *s2 = 0, const char *s3 = 0, const char *s4 = 0);
   static void Form(XrdOucString &s, const char *fmt, int i0, int i1 = 0, int i2 = 0,
                                                      int i3 = 0, int i4 = 0);
   static void Form(XrdOucString &s, const char *fmt, void *p0, void *p1 = 0, void *p2 = 0,
                                                      void *p3 = 0, void *p4 = 0);
   static void Form(XrdOucString &s, const char *fmt, int i0, const char *s0,
                                     const char *s1 = 0, const char *s2 = 0, const char *s3 = 0);
   static void Form(XrdOucString &s, const char *fmt, const char *s0,
                                     int i0, int i1 = 0, int i2 = 0, int i3 = 0);
   static void Form(XrdOucString &s, const char *fmt, const char *s0, const char *s1,
                                     int i0, int i1, int i2);
   static void Form(XrdOucString &s, const char *fmt, int i0, int i1,
                                     const char *s0, const char *s1, const char *s2);
   static void Form(XrdOucString &s, const char *fmt, const char *s0, const char *s1,
                                                      const char *s2, int i0, int i1 = 0);
   static void Form(XrdOucString &s, const char *fmt, int i0, int i1, int i2,
                                                      const char *s0, const char *s1);

   static void Form(XrdOucString &s, const char *fmt, const char *s0, const char *s1, const char *s2,
                                                      const char *s3, int i1);
   static void Form(XrdOucString &s, const char *fmt, int i0, int i1, int i2, int i3, const char *s0);

   static void Form(XrdOucString &s, const char *fmt, int i0, int i1, void *p0);
   static void Form(XrdOucString &s, const char *fmt, int i0, int i1, int i2, void *p0);
   static void Form(XrdOucString &s, const char *fmt, int i0, int i1, int i2, int i3, void *p0);
   static void Form(XrdOucString &s, const char *fmt, int i0, int i1, void *p0, int i2, int i3 = 0);
   static void Form(XrdOucString &s, const char *fmt, void *p0, int i0, int i1);
   static void Form(XrdOucString &s, const char *fmt, const char *s0, void *p0, int i0, int i1);
   static void Form(XrdOucString &s, const char *fmt, void *p0, const char *s0, int i0);
   static void Form(XrdOucString &s, const char *fmt, const char *s0, const char *s1, void *p0);
   static void Form(XrdOucString &s, const char *fmt, int i0, const char *s0, const char *s1,
                                                      int i1, int i2 = 0);
   static void Form(XrdOucString &s, const char *fmt, int i0, const char *s0, int i1, int i2 = 0);

   static int GetIDFromPath(const char *path, XrdOucString &emsg);
   static long int GetLong(char *str);
#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)
   static int GetMacProcList(kinfo_proc **plist, int &nproc);
#endif
   static int GetNumCPUs();
   static int GetGroupInfo(const char *grp, XrdProofGI &gi);
   static int GetGroupInfo(int gid, XrdProofGI &gi);
   static int GetProcesses(const char *pn, std::map<int,XrdOucString> *plist);
   static int GetUserInfo(const char *usr, XrdProofUI &ui);
   static int GetUserInfo(int uid, XrdProofUI &ui);
   static bool HasToken(const char *s, const char *tokens);
   static int KillProcess(int pid, bool forcekill, XrdProofUI ui, bool changeown);
   static int MvDir(const char *oldpath, const char *newpath);
   static int ParsePidPath(const char *path, XrdOucString &before, XrdOucString &after);
   static int ParseUsrGrp(const char *path, XrdOucString &usr, XrdOucString &grp);
   static const char *ProofRequestTypes(int type);
   static int ReadMsg(int fd, XrdOucString &msg);
   static int RmDir(const char *path);
   static int SymLink(const char *path, const char *link);
   static int Touch(const char *path, int opt = 0);
   static int VerifyProcessByID(int pid, const char *pname = "proofserv");
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

#ifndef LETTOIDX
#define LETTOIDX(x, ilet) \
        if (x >= 97 && x <= 122) ilet = x - 96; \
        if (x >= 65 && x <= 90) ilet = x - 38;
#endif
#ifndef IDXTOLET
#define IDXTOLET(ilet, x) \
        if ((ilet) >= 1 && (ilet) <= 26) x = (ilet) + 96; \
        if ((ilet) >= 27 && (ilet) <= 52) x = (ilet) + 38;
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

#ifndef XPDFORM
#define XPDFORM XrdProofdAux::Form
#endif

#endif
