// @(#)root/base:$Id$
// Author: Fons Rademakers   15/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TMemberInspector.h"
#include "TInterpreter.h"
#include "TClassEdit.h"
#include "TClass.h"


class TMemberInspector::TParentBuf {
private:
   std::vector<char> fBuf;
   Ssiz_t fLen;
public:
   TParentBuf(): fBuf(1024), fLen(0) {}
   Ssiz_t GetLength() const { return fLen; }
   void Append(const char*);
   void Remove(Ssiz_t startingAt);
   operator const char*() const { return &fBuf[0]; }
};

void TMemberInspector::TParentBuf::Append(const char* add)
{
   // Add "add" to string
   if (!add || !add[0]) return;
   Ssiz_t addlen = strlen(add);
   fBuf.reserve(fLen + addlen);
   const char* i = add;
   while (*i) {
      fBuf[fLen++] = *i;
      ++i;
   }
   fBuf[fLen] = 0;
}

void TMemberInspector::TParentBuf::Remove(Ssiz_t startingAt)
{
   // Remove characters starting at "startingAt"
   fLen = startingAt;
   fBuf[startingAt] = 0;
}

ClassImp(TMemberInspector)

TMemberInspector::TMemberInspector() {
   // Construct a member inspector

   fParent = new TParentBuf();
}

TMemberInspector::~TMemberInspector() {
   // Destruct a member inspector
   delete fParent;
}

const char* TMemberInspector::GetParent() const
{
   // Get the parent string.
   return *fParent;
}

Ssiz_t TMemberInspector::GetParentLen() const
{
   // Get the length of the parent string.
   return fParent->GetLength();
}

void TMemberInspector::AddToParent(const char* name)
{
   // Append "name" to the parent string.
   fParent->Append(name);
}

void TMemberInspector::RemoveFromParent(Ssiz_t startingAt)
{
   // Remove trailing characters starting at "startingAt".
   fParent->Remove(startingAt);
}



void TMemberInspector::GenericShowMembers(const char *topClassName, void *obj,
                                          Bool_t transientMember) {
   // Call ShowMember() on obj.

   // This could be faster if we implemented this either as a templated
   // function or by rootcint-generated code using the typeid (i.e. the
   // difference is a lookup in a TList instead of in a map).
   
   // To avoid a spurrious error message in case the data member is
   // transient and does not have a dictionary we check first.
   if (transientMember) {
      if (!TClassEdit::IsSTLCont(topClassName)) {
         ClassInfo_t *b = gInterpreter->ClassInfo_Factory(topClassName);
         Bool_t isloaded = gInterpreter->ClassInfo_IsLoaded(b);
         gInterpreter->ClassInfo_Delete(b);
         if (!isloaded) return;
      }
   }
   
   TClass *top = TClass::GetClass(topClassName);
   if (top) {
      top->CallShowMembers(obj, *this);
   } else {
      // This might be worth an error message
   }
   
}


void TMemberInspector::InspectMember(TObject& obj, const char* name) {
   InspectMember<TObject>(obj, name);
}

void TMemberInspector::InspectMember(const char* topclassname, void* pobj,
                                     const char* name, Bool_t transient) {
   Ssiz_t len = fParent->GetLength();
   fParent->Append(name);
   GenericShowMembers(topclassname, pobj, transient);
   fParent->Remove(len);
}

void TMemberInspector::InspectMember(TClass* cl, void* pobj, const char* name) {
   Ssiz_t len = fParent->GetLength();
   fParent->Append(name);
   cl->CallShowMembers(pobj, *this);
   fParent->Remove(len);
}
