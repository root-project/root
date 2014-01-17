// @(#)root/base:$Id$
// Author: Fons Rademakers   15/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMemberInspector
#define ROOT_TMemberInspector

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMemberInspector                                                     //
//                                                                      //
// Abstract base class for accessing the datamembers of a class.        //
// Classes derived from this class can be given as argument to the      //
// ShowMembers() methods of ROOT classes. This feature facilitates      //
// the writing of class browsers and inspectors.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"

class TObject;
class TClass;

class TMemberInspector {
private:
   class TParentBuf;
   TParentBuf* fParent; // current inspection "path"

   TMemberInspector(const TMemberInspector&);            // Not implemented.
   TMemberInspector &operator=(const TMemberInspector&); // Not implemented.

public:
   TMemberInspector();
   virtual ~TMemberInspector();

   virtual void Inspect(TClass *cl, const char *parent, const char *name, const void *addr);
   virtual void Inspect(TClass *cl, const char *parent, const char *name, const void *addr, Bool_t /* isTransient */) { Inspect(cl,parent,name,addr); }

   const char* GetParent() const;
   Ssiz_t GetParentLen() const;
   void AddToParent(const char* name);
   void RemoveFromParent(Ssiz_t startingAt);

   template <class T>
   void InspectMember(const T& obj, const char* name, Bool_t isTransient) {
      Ssiz_t len = GetParentLen();
      AddToParent(name);
      obj.IsA()->CallShowMembers(&obj, *this, isTransient);
      RemoveFromParent(len);
   }

   void InspectMember(const TObject& obj, const char* name, Bool_t isTransient);
   void InspectMember(const char* topclassname, const void* pobj, const char* name,
                      Bool_t transient);
   void InspectMember(TClass* cl, const void* pobj, const char* name,
                      Bool_t isTransient);
   
   void GenericShowMembers(const char *topClassName, const void *obj,
                           Bool_t transientMember);

   ClassDef(TMemberInspector,0)  //ABC for inspecting class data members
};

#endif
