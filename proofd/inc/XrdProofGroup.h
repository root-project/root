// @(#)root/proofd:$Name:  $:$Id:$
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
private:
   XrdOucString  fName;    // group name

   XrdOucString  fMembers; // comma-separated list of members
   XrdOucString  fActives; // comma-separated list of active (i.e. non-idle) members
   int           fSize;    // Number of members
   int           fActive;  // Number of active (i.e. non-idle) members

   XrdOucHash<XrdProofGroupProperty> fProperties; // list of properties identified by name

   XrdOucRecMutex *fMutex; // Local mutex

   void          AddMember(const char *usr) { XrdOucMutexHelper mhp(fMutex);
                                              fMembers += usr; fMembers += ","; fSize++; }
   void          AddProperty(XrdProofGroupProperty *p);

   XrdProofGroup(const char *n, const char *m = 0);

   static XrdOucHash<XrdProofGroup> fgGroups;  // keeps track of group of users
   static XrdOucRecMutex fgMutex; // Mutex to protect access to fgGroups

   static XrdProofdFile fgCfgFile; // Last used group configuration file

public:
   ~XrdProofGroup();

   static int    Config(const char *fn);

   inline int    Active() const { XrdOucMutexHelper mhp(fMutex); return fActive; }
   bool          HasMember(const char *usr);
   inline const char *Members() const { XrdOucMutexHelper mhp(fMutex); return fMembers.c_str(); }
   inline const char *Name() const { XrdOucMutexHelper mhp(fMutex); return fName.c_str(); }
   inline int    Size() const { XrdOucMutexHelper mhp(fMutex); return fSize; }

   void          Count(const char *usr, int n = 1);
   void          Print();

   XrdProofGroupProperty *GetProperty(const char *p);

   static XrdProofGroup *Apply(int (*f)(const char *, XrdProofGroup *, void *), void *arg);

   static XrdOucString Export(const char *grp);
   static void         Print(const char *grp);

   static XrdProofGroup *GetGroup(const char *grp);
   static XrdProofGroup *GetUserGroup(const char *usr, const char *grp = 0);
};

#endif
