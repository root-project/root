// @(#)root/proofd:$Name:  $:$Id: XrdProofGroup.h,v 1.1 2007/06/12 13:51:03 ganis Exp $
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

#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucPthread.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdProofdAux.h"

class XrdProofGroupProperty {
   XrdOucString  fName;       // property name
   int           fNominal;    // property nominal value
   int           fEffective;  // property effective value
   int           fState;      // relative state (minimal, maximal)

public:
   XrdProofGroupProperty(const char *name, int nom, int eff, int st = 0) :
                 fName(name), fNominal(nom), fEffective(eff), fState(st) { }
   ~XrdProofGroupProperty() { }

   const char   *Name() const { return fName.c_str(); }
   int           Nominal() const { return fNominal; }
   int           Effective() const { return fEffective; }
   int           State() const { return fState; }

   void          SetEffective(int eff) { fEffective = eff; }
   void          SetState(int st) { fState = st; }
};

class XrdProofGroup {
friend class XrdProofGroupMgr;
private:
   XrdOucString  fName;    // group name

   XrdOucString  fMembers; // comma-separated list of members
   XrdOucString  fActives; // comma-separated list of active (i.e. non-idle) members
   int           fSize;    // Number of members
   int           fActive;  // Number of active (i.e. non-idle) members

   // Properties
   int           fPriority; // Arbitrary number indicating the priority of this group
   int           fFraction; // Resource fraction in % (nominal)
   float         fFracEff;  // Resource fraction in % (effective)

   XrdOucRecMutex *fMutex; // Local mutex

   void          AddMember(const char *usr) { XrdOucMutexHelper mhp(fMutex);
                                              fMembers += usr; fMembers += ","; fSize++; }
   XrdProofGroup(const char *n, const char *m = 0);

public:
   ~XrdProofGroup();

   inline int    Active() const { XrdOucMutexHelper mhp(fMutex); return fActive; }
   bool          HasMember(const char *usr);
   inline const char *Members() const { XrdOucMutexHelper mhp(fMutex); return fMembers.c_str(); }
   inline const char *Name() const { XrdOucMutexHelper mhp(fMutex); return fName.c_str(); }
   inline int    Size() const { XrdOucMutexHelper mhp(fMutex); return fSize; }

   inline int    Fraction() const { XrdOucMutexHelper mhp(fMutex); return fFraction; }
   inline float  FracEff() const { XrdOucMutexHelper mhp(fMutex); return fFracEff; }
   inline int    Priority() const { XrdOucMutexHelper mhp(fMutex); return fPriority; }
   void          SetFracEff(float f) { XrdOucMutexHelper mhp(fMutex); fFracEff = f; }
   void          SetFraction(int f) { XrdOucMutexHelper mhp(fMutex); fFraction = f; }
   void          SetPriority(int p) { XrdOucMutexHelper mhp(fMutex); fPriority = p; }

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
   XrdOucRecMutex            fMutex;   // Mutex to protect access to fGroups

   XrdProofdFile             fCfgFile; // Last used group configuration file

public:
   XrdProofGroupMgr(const char *fn = 0);
   ~XrdProofGroupMgr() { }

   int            Config(const char *fn);

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
   int prmax;
   int prmin;
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
