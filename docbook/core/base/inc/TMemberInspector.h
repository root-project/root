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

public:
   TMemberInspector();
   virtual ~TMemberInspector();

   virtual void Inspect(TClass *cl, const char *parent, const char *name, const void *addr) = 0;

   const char* GetParent() const;
   Ssiz_t GetParentLen() const;
   void AddToParent(const char* name);
   void RemoveFromParent(Ssiz_t startingAt);

   template <class T>
   void InspectMember(T& obj, const char* name) {
      Ssiz_t len = GetParentLen();
      AddToParent(name);
      obj.ShowMembers(*this);
      RemoveFromParent(len);
   }

   void InspectMember(TObject& obj, const char* name);
   void InspectMember(const char* topclassname, void* pobj, const char* name,
                      Bool_t transient);
   void InspectMember(TClass* cl, void* pobj, const char* name);
   
   void GenericShowMembers(const char *topClassName, void *obj,
                           Bool_t transientMember);

   ClassDef(TMemberInspector,0)  //ABC for inspecting class data members
};

#endif
