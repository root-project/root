// @(#)root/meta:$Name:  $:$Id: TClass.cxx,v 1.29 2001/02/02 11:24:12 brun Exp $
// Author: Rene Brun   07/01/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  The ROOT global object gROOT contains a list of all defined         //
//  classes. This list is build when a reference to a class dictionary  //
//  is made. When this happens, the static "class"::Dictionary()        //
//  function is called to create a TClass object describing the         //
//  class. The Dictionary() function is defined in the ClassDef         //
//  macro and stored (at program startup or library load time) together //
//  with the class name in the TClassTable singleton object.            //
//  For a description of all dictionary classes see TDictionary.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//*-*x7.5 macros/layout_class

#include <iostream.h>

#include "TROOT.h"
#include "TClass.h"
#include "TObjArray.h"
#include "TBaseClass.h"
#include "TBrowser.h"
#include "TDataMember.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TDataType.h"
#include "TRealData.h"
#include "TVirtualPad.h"
#include "TInterpreter.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TError.h"
#include "TMapFile.h"
#include "TStreamerInfo.h"
#include "Api.h"

#ifndef WIN32
extern long G__globalvarpointer;
#endif

Int_t  TClass::fgClassCount;
Bool_t TClass::fgCallingNew = kFALSE;


class TBuildRealData : public TMemberInspector {

private:
   TObject *fRealDataObject;
   TClass  *fRealDataClass;

public:
   TBuildRealData(TObject *obj, TClass *cl)
      { fRealDataObject = obj; fRealDataClass = cl; }
   void Inspect(TClass *cl, const char *parent, const char *name, void *addr);
};

//______________________________________________________________________________
void TBuildRealData::Inspect(TClass *cl, const char *pname, const char *mname, void *add)
{
   // This method is called from ShowMembers() via BuildRealdata().

   TRealData   *rd;
   TDataMember *dm = cl->GetDataMember(mname);
   if (!dm || !dm->IsPersistent()) return;

   char rname[256];
   strcpy(rname,pname);
   strcat(rname,mname);
   Int_t offset = Int_t((Long_t)add - (Long_t)fRealDataObject);

   // Data Member is a pointer
   if (dm->IsaPointer()) {     // pointer to class object
      if (!dm->IsBasic()) {
         rd = new TRealData(rname,offset,dm);
         fRealDataClass->GetListOfRealData()->Add(rd);
      } else {                 // pointer to basic data type
         rd = new TRealData(rname,offset,dm);
         fRealDataClass->GetListOfRealData()->Add(rd);
      }
   } else {
     // Data Member is a basic data type
     rd = new TRealData(rname,offset,dm);
     if (!dm->IsBasic()) rd->SetIsObject(kTRUE);
     fRealDataClass->GetListOfRealData()->Add(rd);
   }
}


ClassImp(TClass)

//______________________________________________________________________________
TClass::TClass() : TDictionary()
{
   // Default ctor.

   fBase         = 0;
   fData         = 0;
   fMethod       = 0;
   fRealData     = 0;
   fClassInfo    = 0;
   fAllPubData   = 0;
   fAllPubMethod = 0;
   fStreamerInfo = 0;
   fCheckSum     = 0;

   ResetInstanceCount();
}

//______________________________________________________________________________
TClass::TClass(const char *name) : TDictionary()
{
   // Create a TClass object. This object contains the full dictionary
   // of a class. It has list to baseclasses, datamembers and methods.
   // Use this ctor to create a standalone TClass object. Most useful
   // to get a TClass interface to an interpreted class. Used by TTabCom.
   // Normally you would use gROOT->GetClass("class") to get access to a
   // TClass object for a certain class.

   if (!gROOT)
      ::Fatal("TClass::TClass", "ROOT system not initialized");

   fName           = name;
   fClassVersion   = 0;
   fDeclFileName   = 0;
   fImplFileName   = 0;
   fDeclFileLine   = -1;    // -1 for standalone TClass (checked in dtor)
   fImplFileLine   = 0;
   fBase           = 0;
   fData           = 0;
   fMethod         = 0;
   fRealData       = 0;
   fClassInfo      = 0;
   fAllPubData     = 0;
   fAllPubMethod   = 0;
   fCheckSum       = 0;
   fStreamerInfo   = 0;

   ResetInstanceCount();

   if (!fClassInfo) {
      if (!gInterpreter)
         ::Fatal("TClass::TClass", "gInterpreter not initialized");

      gInterpreter->SetClassInfo(this);   // sets fClassInfo pointer
      if (!fClassInfo)
         ::Warning("TClass::TClass", "no dictionary for class %s is available", name);
   }
}

//______________________________________________________________________________
TClass::TClass(const char *name, Version_t cversion,
               const char *dfil, const char *ifil, Int_t dl, Int_t il)
        : TDictionary()
{
   // Create a TClass object. This object contains the full dictionary
   // of a class. It has list to baseclasses, datamembers and methods.

   if (!gROOT)
      ::Fatal("TClass::TClass", "ROOT system not initialized");

   fName           = name;
   fClassVersion   = cversion;
   fDeclFileName   = dfil;
   fImplFileName   = ifil;
   fDeclFileLine   = dl;
   fImplFileLine   = il;
   fBase           = 0;
   fData           = 0;
   fMethod         = 0;
   fRealData       = 0;
   fClassInfo      = 0;
   fAllPubData     = 0;
   fAllPubMethod   = 0;
   fCheckSum       = 0;
   fStreamerInfo   = new TObjArray(fClassVersion+1);

   ResetInstanceCount();

   if (!fClassInfo) {
      if (!gInterpreter)
         ::Fatal("TClass::TClass", "gInterpreter not initialized");

      gInterpreter->SetClassInfo(this);   // sets fClassInfo pointer
      if (!fClassInfo)
         ::Warning("TClass::TClass", "no dictionary for class %s is available", name);
   }

   fgClassCount++;
   SetUniqueID(fgClassCount);

   //in case a class with the same name had been created by TStreamerInfo
   //we must delete the old class, importing only the StreamerInfo structure
   //from the old dummy class.
   TClass *oldcl = (TClass*)gROOT->GetListOfClasses()->FindObject(name);
   if (oldcl) {
      if (oldcl->CanIgnoreTObjectStreamer()) {
         printf("old class: %s has IgnoreTobjectStreamer bit\n",name);
         IgnoreTObjectStreamer();
      } 
      if (oldcl->CanBypassStreamer()) {
         printf("old class: %s has BypassStreamer bit\n",name);
         BypassStreamer();
      } 
      
      TStreamerInfo *info;
      TIter next(oldcl->GetStreamerInfos());
      while ((info = (TStreamerInfo*)next())) {
         info->SetClass(this);
         if (info->CanBypassStreamer()) BypassStreamer();
         fStreamerInfo->Add(info);
      }
      oldcl->GetStreamerInfos()->Clear();
      gROOT->GetListOfClasses()->Remove(oldcl);
      delete oldcl;
   }
   gROOT->GetListOfClasses()->Add(this);
}

//______________________________________________________________________________
TClass::~TClass()
{
   // TClass dtor. Deletes all list that might have been created.

   // Not owning lists, don't call Delete()
   // But this still need to be done first because the TList desctructor
   // does access the object contained (via GetObject()->TestBit(kCanDelete))
   delete fAllPubData;
   delete fAllPubMethod;

   if (fBase)
      fBase->Delete();
   delete fBase;

   if (fData)
      fData->Delete();
   delete fData;

   if (fMethod)
      fMethod->Delete();
   delete fMethod;

   if (fRealData)
      fRealData->Delete();
   delete fRealData;

   if (fStreamerInfo)
      fStreamerInfo->Delete();
   delete fStreamerInfo;

   if (fDeclFileLine >= 0)
      gROOT->GetListOfClasses()->Remove(this);

   delete fClassInfo;
}

//______________________________________________________________________________
void TClass::Browse(TBrowser *b)
{
   // This method is called by a browser to get the class information.

   if (!fClassInfo) return;

   if (b) {
      if (!fRealData) BuildRealData();

      b->Add(GetListOfDataMembers(), "Data Members");
      b->Add(GetListOfRealData(), "Real Data Members");
      b->Add(GetListOfMethods(), "Methods");
      b->Add(GetListOfBases(), "Base Classes");
   }
}

//______________________________________________________________________________
void TClass::BuildRealData(void *pointer)
{
   // Build a full list of persistent data members.
   // Scans the list of all data members in the class itself and also
   // in all base classes. For each persistent data member, inserts a
   // TRealData object in the list fRealData.
   //
   // if pointer is NULL, uses the object at pointer
   // otherwise creates a temporary object object of this class

   if (fRealData) return;
   if (!fClassInfo) return;

   TObject *realDataObject = (TObject*)pointer;

   fRealData = new TList;

   if ((!pointer) && (Property() & kIsAbstract)) return;
   
   // Create an instance of this class
   if (!realDataObject) {
      if (!strcmp(GetName(),"TROOT")) realDataObject = gROOT;
      else                            realDataObject = (TObject*)New();
   }

   // The following statement will call recursively all the subclasses
   // of this class.
   if (realDataObject) {
      char parent[256];
      parent[0] = 0;
      TBuildRealData brd(realDataObject, this);
      if (InheritsFrom(TObject::Class())) {
         realDataObject->ShowMembers(brd, parent);
      } else {
         G__CallFunc func;
         void *address;
         long  offset;
         func.SetFunc(fClassInfo->GetMethod("ShowMembers",
                      "TMemberInspector&,char*", &offset).InterfaceMethod());
         func.SetArg((long)&brd);
         func.SetArg((long)parent);
         address = (void*)((long)realDataObject + offset);
         func.Exec(address);
      }
      // take this opportunity to build the real data for base classes
      // In case one base class is abstract, it would not be possible later
      // to create the list of real data for this abstract class
      TBaseClass *base;
      TIter       next(GetListOfBases());
      while ((base = (TBaseClass *) next())) {
         TClass *c = base->GetClassPointer();
         if (c) c->BuildRealData((char*)realDataObject + base->GetDelta());
      }
   }

   if( !pointer && realDataObject && realDataObject != gROOT) {
      if (InheritsFrom(TObject::Class())) {
         realDataObject->SetBit(kZombie); //this info useful in object destructor
         delete realDataObject;
      } else {
         Destructor(realDataObject);
         //::operator delete(realDataObject);
      }
   }
}

//______________________________________________________________________________
void TClass::BypassStreamer(Bool_t bypass)
{
//  When the class kBypassStreamer bit is set, the automatically
//  generated Streamer can call directly TClass::WriteBuffer
//  This option can be set for critical classes in collections.
//  It saves a non negligible overhead in time at the expense of losing
//  the control for additional operations in the Streamer function.

   if (bypass) SetBit(kBypassStreamer);
   else        ResetBit(kBypassStreamer);
}


//______________________________________________________________________________
Int_t TClass::Compare(const TObject *obj) const
{
  // Compare to other object. Returns 0<, 0 or >0 depending on
   // whether "this" is lexicographically less than, equal to, or
   // greater than obj.

   return strcmp(fName.Data(), obj->GetName());
}

//______________________________________________________________________________
void TClass::Draw(Option_t *option)
{
   // Draw detailed class inheritance structure.
   // If a class B inherits from a class A, the description of B is drawn
   // on the right side of the description of A.
   // Member functions overridden by B are shown in class A with a blue line
   // erasing the corresponding member function

   if (!fClassInfo) return;

   TVirtualPad *padsav = gPad;

   // Should we create a new canvas?
   TString opt=option;
   if (!padsav || !opt.Contains("same")) {
      TVirtualPad *padclass = (TVirtualPad*)(gROOT->GetListOfCanvases())->FindObject("R__class");
      if (!padclass)
         gROOT->ProcessLineFast("new TCanvas(\"R__class\",\"class\",20,20,1000,750);");
      else
         padclass->cd();
   }

   if (gPad) gPad->DrawClassObject(this,option);

   if (padsav) padsav->cd();
}

//______________________________________________________________________________
char *TClass::EscapeChars(char *text) const
{
   // Introduce an escape character (@) in front of a special chars.
   // You need to use the result immediately before it is being overwritten.

   static char name[128];
   Int_t nch = strlen(text);
   if (nch > 127) nch = 127;
   Int_t icur = -1;
   for (Int_t i=0;i<nch;i++) {
      icur++;
      if ( text[i] == '\"' || text[i] == '['
        || text[i] == ']'  || text[i] == '&'
        || text[i] == '#'  || text[i] == '!'
        || text[i] == '^'  || text[i] == '<'
        || text[i] == '?'  || text[i] == '>') { name[icur] = '@'; icur++; }
      name[icur] = text[i];
   }
   name[icur+1] = 0;
   return name;
}

//______________________________________________________________________________
TClass *TClass::GetBaseClass(const char *classname)
{
   // Return pointer to the base class "classname". Returns 0 in case
   // "classname" is not a base class. Takes care of multiple inheritance.

   // check if class name itself is equal to classname
   if (strcmp(GetName(), classname) == 0) return (TClass*)this;

   if (!fClassInfo) return 0;

   TObjLink *lnk = GetListOfBases() ? fBase->FirstLink() : 0;

   // otherwise look at inheritance tree
   while (lnk) {
      TClass     *c, *c1;
      TBaseClass *base = (TBaseClass*) lnk->GetObject();
      c = base->GetClassPointer();
      if (c) {
         if (strcmp(c->GetName(), classname) == 0) return c;
         c1 = c->GetBaseClass(classname);
         if (c1) return c1;
      }
      lnk = lnk->Next();
   }
   return 0;
}

//______________________________________________________________________________
TClass *TClass::GetBaseClass(const TClass *cl)
{
   // Return pointer to the base class "cl". Returns 0 in case "cl"
   // is not a base class. Takes care of multiple inheritance.

   // check if class name itself is equal to classname
   if (cl == this) return (TClass*)this;

   if (!fClassInfo) return 0;

   TObjLink *lnk = GetListOfBases() ? fBase->FirstLink() : 0;

   // otherwise look at inheritance tree
   while (lnk) {
      TClass     *c, *c1;
      TBaseClass *base = (TBaseClass*) lnk->GetObject();
      c = base->GetClassPointer();
      if (c) {
         if (cl == c) return c;
         c1 = c->GetBaseClass(cl);
         if (c1) return c1;
      }
      lnk = lnk->Next();
   }
   return 0;
}

//______________________________________________________________________________
Int_t TClass::GetBaseClassOffset(const TClass *cl)
{
   // Return data member offset to the base class "cl".
   // Returns -1 in case "cl" is not a base class.
   // Takes care of multiple inheritance.

   // check if class name itself is equal to classname
   if (cl == this) return 0;

   if (!fClassInfo) return -1;

   TClass     *c;
   Int_t      off;
   TBaseClass *inh;
   TIter      next(GetListOfBases());

   // otherwise look at inheritance tree
   while ((inh = (TBaseClass *) next())) {
      c = inh->GetClassPointer();
      if (c) {
         if (cl == c) return inh->GetDelta();
         off = c->GetBaseClassOffset(cl);
         if (off != -1) return off + inh->GetDelta();
      }
   }
   return -1;
}

//______________________________________________________________________________
TClass *TClass::GetBaseDataMember(const char *datamember)
{
   // Return pointer to (base) class that contains datamember.

   if (!fClassInfo) return 0;

   // Check if data member exists in class itself
   TDataMember *dm = GetDataMember(datamember);
   if (dm) return (TClass*)this;

   // if datamember not found in class, search in next base classes
   TBaseClass *inh;
   TIter       next(GetListOfBases());
   while ((inh = (TBaseClass *) next())) {
      TClass *c = inh->GetClassPointer();
      if (c) {
         TClass *cdm = c->GetBaseDataMember(datamember);
         if (cdm) return cdm;
      }
   }

   return 0;
}

//______________________________________________________________________________
TDataMember *TClass::GetDataMember(const char *datamember)
{
   // Return pointer to datamember object with name "datamember".

   if (!fClassInfo) return 0;

   // Strip off leading *'s and trailing [
   char memb[64];
   char *s = (char*)datamember;
   while (*s == '*') s++;
   strcpy(memb, s);
   if ((s = strchr(memb, '['))) *s = 0;

   TDataMember *dm;
   TIter   next(GetListOfDataMembers());

   while ((dm = (TDataMember *) next()))
      if (strcmp(memb, dm->GetName()) == 0)
         return dm;
   return 0;
}

//______________________________________________________________________________
TList *TClass::GetListOfBases()
{
   // Return list containing the TBaseClass(es) of a class.

   if (!fClassInfo) return 0;

   if (!fBase) {
      if (!gInterpreter)
         Fatal("GetListOfBases", "gInterpreter not initialized");

      gInterpreter->CreateListOfBaseClasses(this);
   }
   return fBase;
}

//______________________________________________________________________________
TList *TClass::GetListOfDataMembers()
{
   // Return list containing the TDataMembers of a class.

   if (!fClassInfo) return 0;

   if (!fData) {
      if (!gInterpreter)
         Fatal("GetListOfDataMembers", "gInterpreter not initialized");

      gInterpreter->CreateListOfDataMembers(this);
   }
   return fData;
}

//______________________________________________________________________________
TList *TClass::GetListOfMethods()
{
   // Return list containing the TMethods of a class.

   if (!fClassInfo) return 0;

   if (!fMethod) {
      if (!gInterpreter)
         Fatal("GetListOfMethods", "gInterpreter not initialized");

      gInterpreter->CreateListOfMethods(this);
   }
   return fMethod;
}

//______________________________________________________________________________
TList *TClass::GetListOfAllPublicMethods()
{
   // Returns a list of all public methods of this class and its base classes.
   // Refers to a subset of the methods in GetListOfMethods() so don't do
   // GetListOfAllPublicMethods()->Delete().
   // Algorithm used to get the list is:
   // - put all methods of the class in the list (also protected and private
   //   ones).
   // - loop over all base classes and add only those methods not already in the
   //   list (also protected and private ones).
   // - once finished, loop over resulting list and remove all private and
   //   protected methods.

   if (!fAllPubMethod) {
      fAllPubMethod = new TList;

      // put all methods in the list
      fAllPubMethod->AddAll(GetListOfMethods());

      // loop over all base classes and add new methods
      TIter nextBaseClass(GetListOfBases());
      TBaseClass *pB;
      TMethod    *p;
      while ((pB = (TBaseClass*) nextBaseClass())) {
         if (!pB->GetClassPointer()) continue;
         TIter next(pB->GetClassPointer()->GetListOfAllPublicMethods());
         TList temp;
         while ((p = (TMethod*) next()))
            if (!fAllPubMethod->Contains(p->GetName()))
               temp.Add(p);
         fAllPubMethod->AddAll(&temp);
         temp.Clear();
      }

      // loop over list and remove all non-public methods
      TIter next(fAllPubMethod);
      while ((p = (TMethod*) next()))
         if (!(p->Property() & kIsPublic)) fAllPubMethod->Remove(p);
   }
   return fAllPubMethod;
}

//______________________________________________________________________________
TList *TClass::GetListOfAllPublicDataMembers()
{
   // Returns a list of all public data members of this class and its base
   // classes. Refers to a subset of the data members in GetListOfDatamembers()
   // so don't do GetListOfAllPublicDataMembers()->Delete().

   if (!fAllPubData) {
      fAllPubData = new TList;
      TIter next(GetListOfDataMembers());
      TDataMember *p;

       while ((p = (TDataMember*) next()))
          if (p->Property() & kIsPublic) fAllPubData->Add(p);

       TIter next_BaseClass(GetListOfBases());
       TBaseClass *pB;
       while ((pB = (TBaseClass*) next_BaseClass())) {
          if (!pB->GetClassPointer()) continue;
          fAllPubData->AddAll(pB->GetClassPointer()->GetListOfAllPublicDataMembers() );
       }
   }
   return fAllPubData;
}

//______________________________________________________________________________
void TClass::GetMenuItems(TList *list)
{
   // Returns list of methods accessible by context menu.

   if (!fClassInfo) return;

   // get the base class
   TIter nextBase(GetListOfBases(), kIterBackward);
   TBaseClass *baseClass;
   while ((baseClass = (TBaseClass *) nextBase())) {
      TClass *base = baseClass->GetClassPointer();
      if (base) base->GetMenuItems(list);
   }

   // remove methods redefined in this class with no menu
   TMethod *method, *m;
   TIter next(GetListOfMethods(), kIterBackward);
   while ((method = (TMethod*)next())) {
      m = (TMethod*)list->FindObject(method->GetName());
      if (method->IsMenuItem()) {
         if (!m) list->AddFirst(method);
      } else {
         if (m) list->Remove(m);
      }
   }
}

//______________________________________________________________________________
TMethod *TClass::GetMethodAny(const char *method)
{
   // Return pointer to method without looking at parameters.
   // Does not look in (possible) base classes.

   if (!fClassInfo) return 0;

   TMethod *m;
   TIter    next(GetListOfMethods());

   while ((m = (TMethod *) next())) {
      if (strcmp(method, m->GetName()) == 0) return m;
   }
   return 0;
}

//______________________________________________________________________________
TMethod *TClass::GetMethod(const char *method, const char *params)
{
   // Find the best method (if there is one) matching the parameters.
   // The params string must contain argument values, like "3189, \"aap\", 1.3".
   // The function invokes GetClassMethod to search for a possible method
   // in the class itself or in its base classes. Returns 0 in case method
   // is not found.

   if (!fClassInfo) return 0;

   if (!gInterpreter)
      Fatal("GetMethod", "gInterpreter not initialized");

   Long_t faddr = (Long_t)gInterpreter->GetInterfaceMethod(this, (char *)method,
                                                           (char *)params);
   if (!faddr) return 0;

   // loop over all methods in this class (and its baseclasses) till
   // we find a TMethod with the same faddr

   TMethod *m = GetClassMethod(faddr);
   if (m) return m;

   TBaseClass *base;
   TIter       next(GetListOfBases());
   while ((base = (TBaseClass *) next())) {
      TClass *c = base->GetClassPointer();
      if (c) {
         m = c->GetClassMethod(faddr);
         if (m) return m;
      }
   }
   Error("GetMethod",
       "\nDid not find matching TMethod <%s> with \"%s\" for %s",
       method,params,GetName());
   return 0;
}

//______________________________________________________________________________
TMethod *TClass::GetMethodWithPrototype(const char *method, const char *proto)
{
   // Find the method with a given prototype. The proto string must be of the
   // form: "char*,int,double". Returns 0 in case method is not found.

   if (!fClassInfo) return 0;

   if (!gInterpreter)
      Fatal("GetMethod", "gInterpreter not initialized");

   Long_t faddr = (Long_t)gInterpreter->GetInterfaceMethodWithPrototype(this,
                                                 (char *)method, (char *)proto);
   if (!faddr) return 0;

   // loop over all methods in this class (and its baseclasses) till
   // we find a TMethod with the same faddr

   TMethod *m = GetClassMethod(faddr);
   if (m) return m;

   TBaseClass *base;
   TIter       next(GetListOfBases());
   while ((base = (TBaseClass *) next())) {
      TClass *c = base->GetClassPointer();
      if (c) {
         m = c->GetClassMethod(faddr);
         if (m) return m;
      }
   }
   Error("GetMethod", "Did not find matching TMethod (should never happen)");
   return 0;
}

//______________________________________________________________________________
TMethod *TClass::GetClassMethod(Long_t faddr)
{
   // Look for a method in this class that has the interface function
   // address faddr.

   if (!fClassInfo) return 0;

   TMethod *m;
   TIter    next(GetListOfMethods());
   while ((m = (TMethod *) next())) {
      if (faddr == (Long_t)m->InterfaceMethod())
         return m;
   }
   return 0;
}

//______________________________________________________________________________
const char *TClass::GetTitle() const
{
   // Return the description of the class.

   if (!fClassInfo) return 0;
   return GetClassInfo()->Title();
}

//______________________________________________________________________________
Int_t TClass::GetNdata()
{
   // Return the number of data members of this class
   // Note that in case the list of data members is not yet created, it will be done
   // by GetListOfDataMembers().

   if (!fClassInfo) return 0;

   TList *lm = GetListOfDataMembers();
   if (lm)
      return lm->GetSize();
   else
      return 0;
}

//______________________________________________________________________________
Int_t TClass::GetNmethods()
{
   // Return the number of methods of this class
   // Note that in case the list of methods is not yet created, it will be done
   // by GetListOfMethods().

   if (!fClassInfo) return 0;

   TList *lm = GetListOfMethods();
   if (lm)
      return lm->GetSize();
   else
      return 0;
}

//______________________________________________________________________________
TStreamerInfo *TClass::GetStreamerInfo(Int_t version)
{
   // returns a pointer to the TStreamerInfo object for version
   // If the object doest not exist, it is created

   if (version <= 0) version = fClassVersion;
   TStreamerInfo *sinfo = (TStreamerInfo*)fStreamerInfo->UncheckedAt(version);
   //if (sinfo) return sinfo;
   if (!sinfo) {
      sinfo = new TStreamerInfo(this,"");
      fStreamerInfo->AddAt(sinfo,fClassVersion);
      if (gDebug > 0) printf("Creating StreamerInfo for class: %s, version: %d\n",GetName(),fClassVersion);
      sinfo->Build();
   } else {
      if (!sinfo->GetOffsets()) sinfo->BuildOld();
   }
   return sinfo;
}

//______________________________________________________________________________
void TClass::IgnoreTObjectStreamer(Bool_t ignore)
{
//  When the class kIgnoreTObjectStreamer bit is set, the automatically
//  generated Streamer will not call TObject::Streamer.
//  This option saves the TObject space overhead on the file.
//  However, the information (fBits, fUniqueID) of TObject is lost.

   if ( ignore &&  TestBit(kIgnoreTObjectStreamer)) return;
   if (!ignore && !TestBit(kIgnoreTObjectStreamer)) return;
   TStreamerInfo *sinfo = (TStreamerInfo*)fStreamerInfo->UncheckedAt(fClassVersion);
   if (sinfo) {
      Error("IgnoreTObjectStreamer","Must be called before the creation of StreamerInfo");
      return;
   }
   if (ignore) SetBit  (kIgnoreTObjectStreamer);
   else        ResetBit(kIgnoreTObjectStreamer);
}

//______________________________________________________________________________
Bool_t TClass::InheritsFrom(const char *classname) const
{
   // Return kTRUE if this class inherits from a class with name "classname".

   if (strcmp(GetName(), classname) == 0) return kTRUE;

   if (!fClassInfo) return kFALSE;

   // cast const away (only for member fBase which can be set in GetListOfBases())
   if (((TClass *)this)->GetBaseClass(classname)) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TClass::InheritsFrom(const TClass *cl) const
{
   // Return kTRUE if this class inherits from class cl.

   if (cl == this) return kTRUE;

   if (!fClassInfo) return kFALSE;

   // cast const away (only for member fBase which can be set in GetListOfBases())
   if (((TClass *)this)->GetBaseClass(cl)) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
void *TClass::DynamicCast(const TClass *cl, void *obj, Bool_t up)
{
   // Cast obj of this class type up to baseclass cl if up is true.
   // Cast obj of this class type down from baseclass cl if up is false.
   // If this class is not a baseclass of cl return 0, else the pointer
   // to the cl part of this (up) or to this (down).

   if (cl == this) return obj;

   if (!fClassInfo) return 0;

   Int_t off;
   if ((off = GetBaseClassOffset(cl)) != -1) {
      if (up)
         return (void*)((Long_t)obj+off);
      else
         return (void*)((Long_t)obj-off);
   }
   return 0;
}

//______________________________________________________________________________
void *TClass::New(Bool_t defConstructor)
{
   // Return a pointer to a newly allocated object of this class.
   // The class must have a default constructor.

   if (!fClassInfo) return 0;

   fgCallingNew = defConstructor;
   void *p = GetClassInfo()->New();
   fgCallingNew = kFALSE;
   if (!p) Error("New", "no default ctor for class %s", GetName());

   return p;
}

//______________________________________________________________________________
void TClass::Destructor(void *obj, Bool_t dtorOnly)
{
   // Explicitely call destructor for object.

   if (!fClassInfo) return;

   G__CallFunc func;
   void *address;
   long  offset;
   char  dtor[64];
   sprintf(dtor, "~%s", GetName());
   func.SetFunc(fClassInfo->GetMethod(dtor, "", &offset).InterfaceMethod());
   address = (void*)((long)obj + offset);
   if (dtorOnly) {
#ifdef WIN32
      long saveglobalvar = G__getgvp();
      G__setgvp((long)address);
      func.Exec(address);
      G__setgvp(saveglobalvar);
#else
      long saveglobalvar = G__globalvarpointer;
      G__globalvarpointer = (long)address;
      func.Exec(address);
      G__globalvarpointer = saveglobalvar;
#endif
   } else
      func.Exec(address);
}

//______________________________________________________________________________
Int_t TClass::Size() const
{
   // Return size of object of this class.

   if (!fClassInfo) return 0;
   return GetClassInfo()->Size();
}

//______________________________________________________________________________
Long_t TClass::Property() const
{
   // Get property description word. For meaning of bits see EProperty.

   if (!fClassInfo) return 0;
   return GetClassInfo()->Property();
}

//______________________________________________________________________________
TClass *TClass::Load(TBuffer &b)
{
   // Load class description from I/O buffer and return class object.

   char s[80];

   b.ReadString(s, 80);
   TClass *cl = gROOT->GetClass(s, kTRUE);
   if (!cl)
      ::Error("TClass::Load", "dictionary of class %s not found", s);

   return cl;
}

//______________________________________________________________________________
void TClass::Store(TBuffer &b) const
{
   // Store class description on I/O buffer.

   b.WriteString(GetName());
}

//______________________________________________________________________________
TClass *CreateClass(const char *cname, Version_t id, const char *dfil,
                    const char *ifil, Int_t dl, Int_t il)
{
   // Global function called by a class' static Dictionary() method
   // (see the ClassDef macro).

   // When called via TMapFile (e.g. Update()) make sure that the dictionary
   // gets allocated on the heap and not in the mapped file.
   if (gMmallocDesc) {
      void *msave  = gMmallocDesc;
      gMmallocDesc = 0;
      TClass *cl   = new TClass(cname, id, dfil, ifil, dl, il);
      gMmallocDesc = msave;
      return cl;
   }
   return new TClass(cname, id, dfil, ifil, dl, il);
}

//______________________________________________________________________________
Bool_t TClass::IsCallingNew()
{
   // Static method returning the defConstructor flag passed to TClass::New().
   // This function cannot be inline (problems with NT linker).

   return fgCallingNew;
}

//______________________________________________________________________________
TStreamerInfo *TClass::SetStreamerInfo(Int_t version, const char *info)
{
   // Info is a string describing the names and types of attributes
   // written by the class Streamer function.
   // If info is an empty string (when called by TObject::StreamerInfo)
   // the default Streamer info string is build. This corresponds to
   // the case of an automatically generated Streamer.
   // In case of user defined Streamer function, it is the user responsability
   // to implement a StreamerInfo function (override TObject::StreamerInfo).
   // The user must call IsA()->SetStreamerInfo(info) from this function.

   // info is specified, nothing to do, except that we should verify
   // that it contains a valid descriptor.

/*
   TDataMember *dm;
   Int_t nch = strlen(info);
   Bool_t update = kTRUE;
   if (nch != 0) {
      //decode strings like "TObject;TAttLine;fA;fB;Int_t i,j,k;"
      char *save, *temp, *blank, *colon, *comma;
      save = new char[10000];
      temp = save;
      strcpy(temp,info);
      //remove heading and trailing blanks
      while (*temp == ' ') temp++;
      while (save[nch-1] == ' ') {nch--; save[nch] = 0;}
      if (nch == 0) {delete [] save; return;}
      if (save[nch-1] != ';') {save[nch] = ';'; save[nch+1] = 0;}
      //remove blanks around , or ;
      while ((blank = strstr(temp,"; "))) strcpy(blank+1,blank+2);
      while ((blank = strstr(temp," ;"))) strcpy(blank,  blank+1);
      while ((blank = strstr(temp,", "))) strcpy(blank+1,blank+2);
      while ((blank = strstr(temp," ,"))) strcpy(blank,  blank+1);
      while ((blank = strstr(temp,"  "))) strcpy(blank,  blank+1);
      //loop on tokens separated by ;
      char *final = new char[1000];
      char token[100];
      while ((colon=strchr(temp,';'))) {
         *colon = 0;
         strcpy(token,temp);
         blank = strchr(token,' ');
         if (blank) {
            *blank = 0;
            if (!gROOT->GetType(token)) {
               Error("SetStreamerInfo","Illegal type: %s in %s",token,info);
               return;
            }
            while (blank) {
               strcat(final,token);
               strcat(final," ");
               comma = strchr(blank+1,','); if (comma) *comma=0;
               strcat(final,blank+1);
               strcat(final,";");
               blank = comma;
            }

         } else {
            if (gROOT->GetClass(token,update)) {
               //a class name
               strcat(final,token); strcat(final,";");
            } else {
               //a data member name
               dm = (TDataMember*)GetListOfDataMembers()->FindObject(token);
               if (dm) {
                  strcat(final,dm->GetFullTypeName());
                  strcat(final," ");
                  strcat(final,token); strcat(final,";");
               } else {
                  Error("SetStreamerInfo","Illegal name: %s in %s",token,info);
                  return;
               }
            }
            update = kFALSE;
         }
         temp = colon+1;
         if (*temp == 0) break;
      }
 ////     fStreamerInfo = final;
      delete [] final;
      delete [] save;
      return;
   }

   //info is empty. Let's build the default Streamer descriptor

   char *temp = new char[10000];
   temp[0] = 0;
   char local[100];

   //add list of base classes
   TIter nextb(GetListOfBases());
   TBaseClass *base;
   while ((base = (TBaseClass*) nextb())) {
      sprintf(local,"%s;",base->GetName());
      strcat(temp,local);
   }

   //add list of data members and types
   TIter nextd(GetListOfDataMembers());
   while ((dm = (TDataMember *) nextd())) {
      if (dm->IsEnum()) continue;
      if (!dm->IsPersistent()) continue;
      Long_t property = dm->Property();
      if (property & kIsStatic) continue;
      TClass *acl = gROOT->GetClass(dm->GetTypeName(),update);
      update = kFALSE;
      if (acl) {
         if (acl->GetClassVersion() == 0) continue;
      }

      // dm->GetArrayIndex() returns an empty string if it does not
      // applies
      const char * index = dm->GetArrayIndex();
      if (strlen(index)==0)
         sprintf(local,"%s %s;",dm->GetFullTypeName(),dm->GetName());
      else
         sprintf(local,"%s %s[%s];",dm->GetFullTypeName(),dm->GetName(),index);
      strcat(temp,local);
   }
   //fStreamerInfo = temp;
   delete [] temp;
*/
   return 0;
}

//______________________________________________________________________________
UInt_t TClass::GetCheckSum() const
{
//   Compute and/or return the class check sum.
//  The class ckecksum is used by the automatic schema evolution algorithm
//  to uniquely identify a class version.
//  The check sum is built from the names/types of base classes and
//  data members
//  Algorithm from Victor Perevovchikov (perev@bnl.gov)

   int il;

   UInt_t id = fCheckSum;
   if (id) return id;

   TString name = GetName();
   TString type;
   il = name.Length();
   for (int i=0; i<il; i++) id = id*3+name[i];

   TList *tlb = ((TClass*)this)->GetListOfBases();
   if (tlb) {   // Loop over bases

     TIter nextBase(tlb);

     TBaseClass *tbc=0;
     while((tbc=(TBaseClass*)nextBase())) {
       name = tbc->GetName();
       il = name.Length();
       for (int i=0; i<il; i++) id = id*3+name[i];
     }/*EndBaseLoop*/
   }
   TList *tlm = ((TClass*)this)->GetListOfDataMembers();
   if (tlm) {   // Loop over members
     TIter nextMemb(tlm);
     TDataMember *tdm=0;
     Long_t prop = 0;
     while((tdm=(TDataMember*)nextMemb())) {
       if (!tdm->IsPersistent())        continue;
               //  combine properties
       prop = (tdm->Property());
       TDataType* tdt = tdm->GetDataType();
       if (tdt) prop |= tdt->Property();

       if ( prop&kIsStatic)             continue;
       name = tdm->GetName(); il = name.Length();
       if ( prop&kIsEnum) id = id*3 + 1;

       int i;
       for (i=0; i<il; i++) id = id*3+name[i];
       type = tdm->GetFullTypeName(); il = type.Length();
       for (i=0; i<il; i++) id = id*3+type[i];

       int dim = tdm->GetArrayDim();
       if (prop&kIsArray) {
         for (int i=0;i<dim;i++) id = id*3+tdm->GetMaxIndex(i);}
     }/*EndMembLoop*/
   }
   ((TClass*)this)->fCheckSum =id;
   return id;
}

//______________________________________________________________________________
void TClass::SetStreamer(const char *name, Streamer_t p)
{
// store pointer to function to Stream non basic member name

   if (!fRealData) return;
   TIter next(fRealData);
   TRealData *rd;
   while ((rd = (TRealData*)next())) {
      if (strcmp(rd->GetName(),name) == 0) { rd->SetStreamer(p); break;}
   }
}

//______________________________________________________________________________
Int_t TClass::ReadBuffer(TBuffer &b, void *pointer, Int_t version, UInt_t start, UInt_t count)
{
// function called by the Streamer functions to deserialize information
// from buffer b into object at p.
// This function assumes that the class version and the byte count information
// have been read.
//   version  is the version number of the class
//   start    is the starting position in the buffer b
//   count    is the number of bytes for this object in the buffer

   //the StreamerInfo should exist at this point
   TStreamerInfo *sinfo = (TStreamerInfo*)fStreamerInfo->UncheckedAt(version);
   if (sinfo == 0) {
      BuildRealData(pointer);
      sinfo = new TStreamerInfo(this,"");
      fStreamerInfo->AddAt(sinfo,version);
      if (gDebug > 0) printf("Creating StreamerInfo for class: %s, version: %d\n",GetName(),version);
      sinfo->Build();
   } else if (!fRealData) {
      BuildRealData(pointer);
      sinfo->BuildOld();
   }

   //deserialize the object
   sinfo->ReadBuffer(b, (char*)pointer,-1);

   //check that the buffer position correesponds to the byte count
   b.CheckByteCount(start,count,this);
   return 0;
}

//______________________________________________________________________________
Int_t TClass::ReadBuffer(TBuffer &b, void *pointer)
{
// function called by the Streamer functions to deserialize information
// from buffer b into object at p


   // read the class version from the buffer
   UInt_t R__s, R__c;
   Version_t version = b.ReadVersion(&R__s, &R__c);

   //the StreamerInfo should exist at this point
   TStreamerInfo *sinfo = (TStreamerInfo*)fStreamerInfo->UncheckedAt(version);
   if (sinfo == 0) {
      BuildRealData(pointer);
      sinfo = new TStreamerInfo(this,"");
      fStreamerInfo->AddAt(sinfo,version);
      if (gDebug > 0) printf("Creating StreamerInfo for class: %s, version: %d\n",GetName(),version);
      sinfo->Build();
   } else if (!sinfo->GetOffsets()) {
      BuildRealData(pointer);
      sinfo->BuildOld();
   }

   //deserialize the object
   sinfo->ReadBuffer(b, (char*)pointer,-1);

   //check that the buffer position correesponds to the byte count
   b.CheckByteCount(R__s, R__c,this);

   if (gDebug > 2) printf(" ReadBuffer for class: %s has read %d bytes\n",GetName(),R__c);

   return 0;
}

//______________________________________________________________________________
Int_t TClass::WriteBuffer(TBuffer &b, void *pointer, const char *info)
{
// function called by the Streamer functions to serialize object at p
// to buffer b. The optional argument info may be specified to give an
// alternative StreamerInfo instead of using the default StreamerInfo
// automatically built from the class definition.
// For more information, see class TStreamerInfo.

   //build the StreamerInfo if first time for the class
   TStreamerInfo *sinfo = (TStreamerInfo*)fStreamerInfo->UncheckedAt(fClassVersion);
   if (sinfo == 0) {
      BuildRealData(pointer);
      sinfo = new TStreamerInfo(this,info);
      fStreamerInfo->AddAt(sinfo,fClassVersion);
      if (gDebug > 0) printf("Creating StreamerInfo for class: %s, version: %d\n",GetName(),fClassVersion);
      sinfo->Build();
   } else if (!sinfo->GetOffsets()) {
      BuildRealData(pointer);
      sinfo->BuildOld();
   }

   //write the class version number and reserve space for the byte count
   UInt_t R__c = b.WriteVersion(this, kTRUE);

   //serialize the object
   sinfo->WriteBuffer(b, (char*)pointer,-1);

   //write the byte count at the start of the buffer
   b.SetByteCount(R__c, kTRUE);

   if (gDebug > 2) printf(" WriteBuffer for class: %s has written %d bytes\n",GetName(),R__c);

   return 0;
}
