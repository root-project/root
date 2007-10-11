// @(#)root/proofd:$Id$
// Author: Gerardo Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofGroup
#define ROOT_XrdProofGroup

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofGroup                                                        //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// Class describing groups                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef OLDXRDOUC
#  include "XrdSysToOuc.h"
#  include "XrdOuc/XrdOucPthread.hh"
#else
#  include "XrdSys/XrdSysPthread.hh"
#endif
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdProofdAux.h"

class XrdProofGroup {
friend class XrdProofGroupMgr;
private:
   XrdOucString  fName;    // group name

   XrdOucString  fMembers; // comma-separated list of members
   XrdOucString  fActives; // comma-separated list of active (i.e. non-idle) members
   int           fSize;    // Number of members
   int           fActive;  // Number of active (i.e. non-idle) members

   // Properties
   float         fPriority; // Arbitrary number indicating the priority of this group
   int           fFraction; // Resource fraction in % (nominal)
   float         fFracEff;  // Resource fraction in % (effective)

   XrdSysRecMutex *fMutex; // Local mutex

   void          AddMember(const char *usr) { XrdSysMutexHelper mhp(fMutex);
                                              fMembers += usr; fMembers += ","; fSize++; }
   XrdProofGroup(const char *n, const char *m = 0);

public:
   ~XrdProofGroup();

   inline int    Active() const { XrdSysMutexHelper mhp(fMutex); return fActive; }
   bool          HasMember(const char *usr);

   inline const char *Members() const { XrdSysMutexHelper mhp(fMutex); return fMembers.c_str(); }
   inline const char *Name() const { XrdSysMutexHelper mhp(fMutex); return fName.c_str(); }
   inline int    Size() const { XrdSysMutexHelper mhp(fMutex); return fSize; }

   inline int    Fraction() const { XrdSysMutexHelper mhp(fMutex); return fFraction; }
   inline float  FracEff() const { XrdSysMutexHelper mhp(fMutex); return fFracEff; }
   inline float  Priority() const { XrdSysMutexHelper mhp(fMutex); return fPriority; }
   void          SetFracEff(float f) { XrdSysMutexHelper mhp(fMutex); fFracEff = f; }
   void          SetFraction(int f) { XrdSysMutexHelper mhp(fMutex); fFraction = f; }
   void          SetPriority(float p) { XrdSysMutexHelper mhp(fMutex); fPriority = p; }

   void          Count(const char *usr, int n = 1);
   void          Print();
};


//
// Group Manager class
//
class XrdProofGroupMgr {
private:
   XrdOucString              fIterator; // Keeps track of groups already processed 
   XrdOucHash<XrdProofGroup> fGroups;  // Keeps track of groups managed by this instance
   XrdSysRecMutex            fMutex;   // Mutex to protect access to fGroups

   XrdProofdFile             fCfgFile; // Last used group configuration file
   XrdProofdFile             fPriorityFile; // Last used file with priorities

public:
   XrdProofGroupMgr(const char *fn = 0);
   ~XrdProofGroupMgr() { }

   int            Config(const char *fn);
   int            ReadPriorities();

   XrdProofGroup *Apply(int (*f)(const char *, XrdProofGroup *, void *), void *arg);

   XrdOucString   Export(const char *grp);
   int            Num() { return fGroups.Num(); }
   void           Print(const char *grp);

   XrdProofGroup *GetGroup(const char *grp);
   XrdProofGroup *GetUserGroup(const char *usr, const char *grp = 0);

   // Pseudo-iterator functionality
   void           ResetIter() { fIterator = "getnextgrp:"; }
   XrdProofGroup *Next();
};

// Auc structures for scan through operations
typedef struct {
   float prmax;
   float prmin;
   int nofrac;
   float totfrac;
} XpdGroupGlobal_t;

typedef struct {
   int opt;
   XpdGroupGlobal_t *glo;
   float cut;
   float norm;
} XpdGroupEff_t;

#endif
