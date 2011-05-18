// @(#)root/tree:$Id$
// Author: Axel Naumann   14/10/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBranchBrowsable.h"
#include "TBranchElement.h"
#include "TBranchObject.h"
#include "TMethod.h"
#include "TBrowser.h"
#include "TTree.h"
#include "TLeafObject.h"
#include "TClonesArray.h"
#include "TVirtualPad.h"
#include "TClass.h"
#include "TBaseClass.h"
#include "TDataMember.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TVirtualCollectionProxy.h"
#include "TRef.h"
#include <algorithm>

R__EXTERN TTree *gTree;

ClassImp(TVirtualBranchBrowsable);

//______________________________________________________________________________
//
// TVirtualBranchBrowsable is a base class (not really abstract, but useless
// by itself) for helper objects that extend TBranch's browsing support.
// Each registered derived class's generator method is called, which fills
// all created helper objects into a list which can then be browsed.
// For details of what these browser helper objects can do, see e.g. 
// TMethodBrowsable, which allows methods to show up in the TBrowser.
//
// Only registered helper objects are created. By default, only 
// TMethodBrowsable, TNonSplitBrowsable, and TCollectionPropertyBrowsable
// are registered (see RegisterDefaultGenerators). You can prevent any of 
// their objects to show up in the browser by unregistering the generator:
//   TMethodBrowsable::Unregister()
// will stop creating browsable method helper objects from that call on.
// Note that these helper objects are cached (in TBranch::fBrowsables);
// already created (and thus cached) browsables will still appear in the
// browser even after unregistering the corresponding generator.
//
// You can implement your own browsable objects and thier generator; see
// e.g. the simple TCollectionPropertyBrowsable. Note that you will have
// to register your generator just like any other, and that you should 
// implement the following methods for your own class, mainly for 
// consistency reasons:
//   static void Register() { 
//     TVirtualBranchBrowsable::RegisterGenerator(GetBrowsables); }
//   static void Unregister() { 
//     TVirtualBranchBrowsable::UnregisterGenerator(GetBrowsables); }
// where GetBrowsables is a static member function of your class, that 
// creates the browsable helper objects, and has the signature
//   static Int_t GetBrowsables(TList& list, const TBranch* branch,
//                              const TVirtualBranchBrowsable* parent=0);
// It has to return the number of browsable helper objects for parent
// (or, if NULL, for branch) which are added to the list.
//______________________________________________________________________________

std::list<TVirtualBranchBrowsable::MethodCreateListOfBrowsables_t> 
   TVirtualBranchBrowsable::fgGenerators;
Bool_t TVirtualBranchBrowsable::fgGeneratorsSet=kFALSE;

//______________________________________________________________________________
TVirtualBranchBrowsable::TVirtualBranchBrowsable(const TBranch* branch, TClass* type, 
                                                 Bool_t typeIsPointer, 
                                                 const TVirtualBranchBrowsable* parent /*=0*/):
fBranch(branch), fParent(parent), fLeaves(0), fClass(type), fTypeIsPointer(typeIsPointer) 
{
   // constructor setting all members according to parameters.
   if (!fgGeneratorsSet) RegisterDefaultGenerators();
   if (!branch) 
      Warning("TVirtualBranchBrowsable", "branch is NULL!");
}

//______________________________________________________________________________
TVirtualBranchBrowsable::~TVirtualBranchBrowsable() 
{
   // Destructor. Delete our leaves.
   delete fLeaves;
}

//______________________________________________________________________________
void TVirtualBranchBrowsable::Browse(TBrowser *b) 
{
// Calls TTree::Draw on the method if return type is not a class;
// otherwise expands returned object's "folder"

   if (!fClass) {
      TString name;
      GetScope(name);

      // If this is meant to be run on the collection
      // we need to "move" the "@" from branch.@member
      // to branch@.member
      name.ReplaceAll(".@","@.");
      name.ReplaceAll("->@","@->");

      TTree* tree=0;
      if (!fBranch) {
         Warning("Browse", "branch not set - might access wrong tree!");
         tree=gTree;
      } else tree=fBranch->GetTree();
      tree->Draw(name, "", b ? b->GetDrawOption() : "");
      if (gPad) gPad->Update();
   } else 
      if (GetLeaves()) GetLeaves()->Browse(b);
}

//______________________________________________________________________________
Int_t TVirtualBranchBrowsable::FillListOfBrowsables(TList& li, const TBranch* branch, 
   const TVirtualBranchBrowsable* parent /* =0 */) 
{
// Askes all registered generators to fill their browsables into
// the list. The browsables are generated for a given parent,
// or (if 0), for a given branch. The branch is passed down to
// leaves of TVirtualBranchBrowsable, too, as we need to access
// the branch's TTree to be able to traw.
   if (!fgGeneratorsSet) RegisterDefaultGenerators();
   std::list<MethodCreateListOfBrowsables_t>::iterator iGenerator;
   Int_t numCreated=0;
   for (iGenerator=fgGenerators.begin(); iGenerator!=fgGenerators.end(); iGenerator++)
      numCreated+=(*(*iGenerator))(li, branch, parent);
   return numCreated;
}

//______________________________________________________________________________
TClass* TVirtualBranchBrowsable::GetCollectionContainedType(const TBranch* branch, 
                                                            const TVirtualBranchBrowsable* parent,
                                                            TClass* &contained) 
{
// Check whether the branch (or the parent) contains a collection. 
// If it does, set "contained" to the contained type (if we can 
// retrieve it) and return the TClass for the collection. Set 
// "contained" to the branch's (or parent's) contained object's 
// class for non-collections, returning 0.
//
// Only one of "branch" or "parent" can ge given (depending on whether
// we are creating browsable objects for a branch or for another
// browsable object)
   contained=0;
   TClass* type=0;
   if (parent)
      type=parent->GetClassType();
   else if (branch) {
      if (branch->IsA()==TBranchElement::Class()) {
         // could be a split TClonesArray
         TBranchElement* be=(TBranchElement*) branch;

         // this is the contained type - if !=0
         const char* clonesname=be->GetClonesName();
         if (clonesname && strlen(clonesname))
            contained=TClass::GetClass(clonesname);

         // check if we're in a sub-branch of this class
         // we can only find out asking the streamer given our ID
         ULong_t *elems=0;
         TStreamerElement *element=0;
         if (be->GetID()>=0 && be->GetInfo() 
            && (be->GetID() < be->GetInfo()->GetNdata())
            && ((elems=be->GetInfo()->GetElems()))
            && ((element=(TStreamerElement *)elems[be->GetID()]))) {
            // if contained is set (i.e. GetClonesName was successful),
            // this element containes the container, otherwise it's the 
            // contained
            if (contained)
               // we have all we need
               return element->GetClassPointer();
            else 
               type=element->GetClassPointer();
         } else if (clonesname && strlen(clonesname)) {
            // we have a clones name, and the TCA is not split:
            contained=TClass::GetClass(clonesname);
            return TClass::GetClass(be->GetClassName());
         } else 
            type=TClass::GetClass(be->GetClassName());
      } else if (branch->IsA()==TBranchObject::Class()) {
         // could be an unsplit TClonesArray
         TBranchObject* bo=(TBranchObject*)branch;
         const char* clonesname=bo->GetClassName();
         contained=0;
         if (!clonesname || !strlen(clonesname)) return 0;
         type=TClass::GetClass(clonesname);
      }
   } else {
      if (gTree) gTree->Warning("GetCollectionContainedType", "Neither branch nor parent given!");
      return 0;
   }

   if (!type) return 0;

   TBranch* branchNonCost=const_cast<TBranch*>(branch);
   if (type->InheritsFrom(TClonesArray::Class()) 
      && branch->IsA()==TBranchObject::Class()
      && branchNonCost->GetListOfLeaves()
      && branchNonCost->GetListOfLeaves()->GetEntriesFast()==1) {
      // load first entry of the branch. Yes, this is bad, and might have
      // unexpected side effects for the user, esp as already looking at
      // (and not just drawing) a branch triggeres it.
      // To prove just how ugly it is, we'll also have to const_cast the
      // branch...
      if (branch->GetReadEntry()==-1) branchNonCost->GetEntry(0);
      // now get element
      TLeafObject* lo = (TLeafObject*)branchNonCost->GetListOfLeaves()->First();
      if (lo) {
         TObject* objContainer = lo->GetObject();
         if (objContainer && objContainer->IsA()==TClonesArray::Class()) {
            contained = ((TClonesArray*)objContainer)->GetClass();
         }
      }
      return type;
   } else    if (type->InheritsFrom(TClonesArray::Class()) 
      && branch->IsA()==TBranchElement::Class()
      && branchNonCost->GetListOfLeaves()
      && branchNonCost->GetListOfLeaves()->GetEntriesFast()==1) {
      // load first entry of the branch. Yes, this is bad, and might have
      // unexpected side effects for the user, esp as already looking at
      // (and not just drawing) a branch triggeres it.
      // To prove just how ugly it is, we'll also have to const_cast the
      // branch...
      
      //if (branch->GetReadEntry()==-1) branchNonCost->GetEntry(0);
      // now get element
      //TLeafObject* lo=(TLeafElement*)branchNonCost->GetListOfLeaves()->First();
      //TObject* objContainer=(TObject*)((TBranchElement*)branch)->GetValuePointer();

      //if (objContainer && objContainer->IsA()==TClonesArray::Class())
      //   contained=((TClonesArray*)objContainer)->GetClass();

      // Currently we can peer into the nested TClonesArray, we need
      // to update TBranchElement::GetValuePointer.
      return type;
   } else if (type->InheritsFrom(TCollection::Class())) {
      // some other container, and we don't know what the contained type is
      return type;
   } else if (type->GetCollectionProxy()) {
      contained=type->GetCollectionProxy()->GetValueClass();
      return type;
   } else if (type->InheritsFrom(TRef::Class()))
      // we don't do TRefs, so return contained and container as 0
      return 0;
   else contained=type;
   return 0;
}

//______________________________________________________________________________
TList* TVirtualBranchBrowsable::GetLeaves() const 
{
// Return list of leaves. If not set up yet we'll create them.
   if (!fLeaves) {
      TList* leaves=new TList();
      leaves->SetOwner();
      FillListOfBrowsables(*leaves, GetBranch(), this);
      const_cast<TVirtualBranchBrowsable*>(this)->fLeaves=leaves;
   }
   return fLeaves; 
}

//______________________________________________________________________________
std::list<TVirtualBranchBrowsable::MethodCreateListOfBrowsables_t>& TVirtualBranchBrowsable::GetRegisteredGenerators() 
{
   // returns the list of registered generator methods
   return fgGenerators;
}

//______________________________________________________________________________
void TVirtualBranchBrowsable::GetScope(TString & scope) const 
{
// Returns the full name for TTree::Draw to draw *this.
// Recursively appends, starting at the top TBranch,
// all method / object names with proper reference operators (->, .)
// depending on fTypeIsPointer.

   if (fParent)
      fParent->GetScope(scope);
   else {
      scope=fBranch->GetName();
      Ssiz_t pos = scope.First('[');
      if (pos != kNPOS) {
         scope.Remove(pos);
      }
      if (!scope.EndsWith(".")) scope+=".";
      const TBranch* mother=fBranch;
      while (mother != mother->GetMother() && (mother=mother->GetMother())) {
         TString nameMother(mother->GetName());
         if (!nameMother.EndsWith(".")) {
            scope.Prepend(".");
            scope.Prepend(nameMother);
         } else {
            if (mother != mother->GetMother()) {
               // If the mother is the top level mother
               // and its ends ends with a ., the name is already
               // embedded!
               scope.Prepend(nameMother);
            }
         }
      }
   }
   if (GetName() && GetName()[0]=='.')
      scope+=(GetName()+1);
   else
      scope+=GetName();
   if (fClass && !scope.EndsWith(".")) { // otherwise we're a leaf, and no delimiter is appended
      if (fTypeIsPointer)
         scope+="->";
      else scope+=".";
   }
}


//______________________________________________________________________________
void TVirtualBranchBrowsable::RegisterDefaultGenerators() 
{
// Adds the default generators. The user can remove any of them as follows:
//   TMethodBrowsable::Unregister();
// which will cause the browser not to show any methods.
   if (fgGeneratorsSet) return;
   // can't call RegisterGenerator - would be recusive infite loop
   fgGenerators.push_back(&TMethodBrowsable::GetBrowsables);
   fgGenerators.push_back(&TNonSplitBrowsable::GetBrowsables);
   fgGenerators.push_back(&TCollectionPropertyBrowsable::GetBrowsables);
   fgGeneratorsSet=kTRUE;
}

void TVirtualBranchBrowsable::RegisterGenerator(MethodCreateListOfBrowsables_t generator) 
{
   // Adds a generator to be called when browsing branches.
   // Called by the Register method, which should be implemented 
   // for all derived classes (see e.g. TMethodBrowsable::Register())
   if (!fgGeneratorsSet) RegisterDefaultGenerators();
   // make sure we're not adding another copy
   fgGenerators.remove(generator);
   fgGenerators.push_back(generator);
}

void TVirtualBranchBrowsable::UnregisterGenerator(MethodCreateListOfBrowsables_t generator) 
{
   // Removes a generator from the list of generators to be called when 
   // browsing branches. The user can remove any of the generators as follows:
   //   TMethodBrowsable::Unregister();
   // which will cause the browser not to show any methods.
   if (!fgGeneratorsSet) RegisterDefaultGenerators();
   fgGenerators.remove(generator);
}


ClassImp(TMethodBrowsable);

//______________________________________________________________________________
//
//  This helper object allows the browsing of methods of objects stored in
//  branches. They will be depicted by a leaf (or a branch, in case the method
//  returns an object) with a red exclamation mark. Only a subset of all 
//  methods will be shown in the browser (see IsMethodBrowsable for the
//  criteria a method has to satisfy). 
//
//  Obviously, methods are only available if the library is loaded which 
//  contains the dictionary for the class to be browsed!
//
//  If a branch contains a collection, TMethodBrowsable tries to find out 
//  what the contained element is (it will only create methods for the 
//  contained elements, but never for the collection). If it fails to extract
//  the type of the contained elements, or if there is no guarantee that the
//  type has any other common denominator than TObject (e.g. in the case of
//  a TObjArray, which can hold any object deriving from TObject) no methods
//  will be added.
//______________________________________________________________________________


//______________________________________________________________________________
TMethodBrowsable::TMethodBrowsable(const TBranch* branch, TMethod* m,
                                   const TVirtualBranchBrowsable* parent /* =0 */):
   TVirtualBranchBrowsable(branch, 0, kFALSE, parent), fMethod(m) 
{
// Constructor.
// Links a TBranchElement to a TMethod, allowing the TBrowser to
// browse simple methods.
//
// The c'tor sets the name for a method "Class::Method(params) const"
// to "Method(params)", title to TMethod::GetPrototype
   TString name(m->GetName());
   name+="()";
   if (name.EndsWith(" const")) name.Remove(name.Length()-6);
   SetName(name);

   name=m->GetPrototype();
   if (m->GetCommentString() && strlen(m->GetCommentString()))
      name.Append(" // ").Append(m->GetCommentString());
   SetTitle(name);

   TString plainReturnType(m->GetReturnTypeName());
   if (plainReturnType.EndsWith("*")) {
      SetTypeIsPointer();
      plainReturnType.Remove(plainReturnType.Length()-1);
      plainReturnType.Strip();
      if(plainReturnType.BeginsWith("const")) {
         plainReturnType.Remove(0,5);
         plainReturnType.Strip();
      }   
   }
   SetType(TClass::GetClass(plainReturnType));
}

//______________________________________________________________________________
void TMethodBrowsable::GetBrowsableMethodsForClass(TClass* cl, TList& li) 
{
// Given a class, this methods fills list with TMethodBrowsables
// for the class and its base classes, and returns the number of 
// added elements. If called from a TBranch::Browse overload, "branch" 
// should be set to the calling TBranch, otherwise "parent" should 
// be set to the TVirtualBranchBrowsable being browsed, and branch
// should be the branch of the parent.

   if (!cl) return;
   TList allClasses;
   allClasses.Add(cl);
   
   if (cl->IsLoaded()) {
      for(TObjLink* lnk=allClasses.FirstLink();
          lnk; lnk=lnk->Next()) {
         cl=(TClass*)lnk->GetObject();
         TList* bases=cl->GetListOfBases();
         TBaseClass* base;
         TIter iB(bases);
         while ((base=(TBaseClass*)iB())) {
            TClass* bc=base->GetClassPointer();
            if (bc) allClasses.Add(bc);
         }
      }
   } else {
      TVirtualStreamerInfo *info = cl->GetStreamerInfo();
      for(int el = 0; el < info->GetElements()->GetEntries(); ++el) {
         TStreamerElement *element = (TStreamerElement *)info->GetElements()->UncheckedAt(el);
         if (element->IsBase()) {
            TClass *bc = element->GetClassPointer();
            if (bc) allClasses.Add(bc);
         }
      }
   }

   TList allMethods;
   TIter iC(&allClasses);
   while ((cl=(TClass*)iC())) {
      TList* methods=cl->GetListOfMethods();
      if (!methods) continue;
      TMethod* method=0;
      TIter iM(methods);
      while ((method=(TMethod*)iM()))
         if (method && !allMethods.FindObject(method->GetName()))
            allMethods.Add(method);
   }

   TIter iM(&allMethods);
   TMethod* m=0;
   while ((m=(TMethod*)iM())) {
      if (TMethodBrowsable::IsMethodBrowsable(m)) {
         li.Add(m);
      }
   }
}


//______________________________________________________________________________
Int_t TMethodBrowsable::GetBrowsables(TList& li, const TBranch* branch, 
                                      const TVirtualBranchBrowsable* parent /*=0*/) 
{
// This methods fills list with TMethodBrowsables
// for the branch's or parent's class and its base classes, and returns 
// the number of added elements. If called from a TBranch::Browse 
// overload, "branch" should be set to the calling TBranch, otherwise 
// "parent" should be set to the TVirtualBranchBrowsable being browsed.
   TClass* cl;
   // we don't care about collections, so only use the TClass argument,
   // and not the return value
   GetCollectionContainedType(branch, parent, cl);
   if (!cl) return 0;

   TList listMethods;
   GetBrowsableMethodsForClass(cl, listMethods);
   TMethod* method=0;
   TIter iMethods(&listMethods);
   while ((method=(TMethod*)iMethods())) {
      li.Add(new TMethodBrowsable(branch, method, parent));
   }
   return listMethods.GetSize();
}

//______________________________________________________________________________
Bool_t TMethodBrowsable::IsMethodBrowsable(const TMethod* m) 
{
// A TMethod is browsable if it is const, public and not pure virtual,
// if does not have any parameter without default value, and if it has 
// a (non-void) return value.
// A method called *, Get*, or get* will not be browsable if there is a 
// persistent data member called f*, _*, or m*, as data member access is 
// faster than method access. Examples: if one of fX, _X, or mX is a 
// persistent data member, the methods GetX(), getX(), and X() will not 
// be browsable.

   if (m->GetNargs()-m->GetNargsOpt()==0
       && (m->Property() & kIsConstant 
           & ~kIsPrivate & ~kIsProtected & ~kIsPureVirtual )
       && m->GetReturnTypeName()
       && strcmp("void",m->GetReturnTypeName())
       && !strstr(m->GetName(),"DeclFile")
       && !strstr(m->GetName(),"ImplFile")
       && strcmp(m->GetName(),"IsA")
       && strcmp(m->GetName(),"Class")
       && strcmp(m->GetName(),"CanBypassStreamer")
       && strcmp(m->GetName(),"Class_Name")
       && strcmp(m->GetName(),"ClassName")
       && strcmp(m->GetName(),"Clone")
       && strcmp(m->GetName(),"DrawClone")
       && strcmp(m->GetName(),"GetName")
       && strcmp(m->GetName(),"GetDrawOption")
       && strcmp(m->GetName(),"GetIconName")
       && strcmp(m->GetName(),"GetOption")
       && strcmp(m->GetName(),"GetTitle")
       && strcmp(m->GetName(),"GetUniqueID")
       && strcmp(m->GetName(),"Hash")
       && strcmp(m->GetName(),"IsFolder")
       && strcmp(m->GetName(),"IsOnHeap")
       && strcmp(m->GetName(),"IsSortable")
       && strcmp(m->GetName(),"IsZombie")) {

      // look for matching data member
      TClass* cl=m->GetClass();
      if (!cl) return kTRUE;
      TList* members=cl->GetListOfDataMembers();
      if (!members) return kTRUE;
      const char* baseName=m->GetName();
      if (!strncmp(m->GetName(), "Get", 3) ||
          !strncmp(m->GetName(), "get", 3))
         baseName+=3;
      if (!baseName[0]) return kTRUE;
      
      TObject* mem=0;
      const char* arrMemberNames[3]={"f%s","_%s","m%s"};
      for (Int_t i=0; !mem && i<3; i++)
         mem=members->FindObject(Form(arrMemberNames[i],baseName));
      return (!mem ||! ((TDataMember*)mem)->IsPersistent());
   };
   return kFALSE;
}

//______________________________________________________________________________
void TMethodBrowsable::Register() 
{
   // Wrapper for the registration method. Needed against MSVC, which 
   // assigned different addr to the same method, depending on what
   // translation unit you're in...
   TVirtualBranchBrowsable::RegisterGenerator(GetBrowsables);
}

//______________________________________________________________________________
void TMethodBrowsable::Unregister() 
{
   // Wrapper for the registration method. Needed against MSVC, which 
   // assigned different addr to the same method, depending on what
   // translation unit you're in...
   TVirtualBranchBrowsable::UnregisterGenerator(GetBrowsables);
}


ClassImp(TNonSplitBrowsable);

//______________________________________________________________________________
//
// Allows a TBrowser to browse non-split branches as if they were split. The
// generator extracts the necessary information from the streamer info in 
// memory (which does not have to be the same as the one on file, in case
// a library was loaded containing the dictionary for this type), i.e. it 
// also works without loading the class's library.
//
// Just as with TMethodBrowsables, if the generator finds a collection it 
// only takes the contained objects into account, not the collections. If
// it identifies a collection, but cannot extract the contained type, or the 
// contained type can be anything deriving from a TObject (like for TObjArray)
// or is not limited at all, no browser helper objects are created.
//______________________________________________________________________________

//______________________________________________________________________________
TNonSplitBrowsable::TNonSplitBrowsable(const TStreamerElement* element, const TBranch* branch, 
                                       const TVirtualBranchBrowsable* parent /* =0 */):
   TVirtualBranchBrowsable(branch, element->GetClassPointer(), 
   element->IsaPointer(), parent) 
{
// Constructor. Creates a TNonSplitBrowsable from a TStreamerElement, containing branch 
// and (if applicable) parent TVirtualBranchBrowsable.
   SetNameTitle(element->GetName(), element->GetTitle());
}


//______________________________________________________________________________
Int_t TNonSplitBrowsable::GetBrowsables(TList& li, const TBranch* branch,
                                        const TVirtualBranchBrowsable* parent /* =0 */) 
{
// Given either a branch "branch" or a "parent" TVirtualBranchBrowsable, we fill
// "list" with objects of type TNonSplitBrowsable which represent the members
// of class "cl" (and its base classes' members).

   // branch has to be unsplit, i.e. without sub-branches
   if (parent==0 
       && (branch==0 ||
           (const_cast<TBranch*>(branch)->GetListOfBranches() 
            && const_cast<TBranch*>(branch)->GetListOfBranches()->GetEntries()!=0)
           )
       ) {
      return 0;
   }
   // we only expand our own parents
   //if (parent && parent->IsA()!=TNonSplitBrowsable::Class()) return 0;

   TClass* clContained=0;
   GetCollectionContainedType(branch, parent, clContained);
   TVirtualStreamerInfo* streamerInfo= clContained?clContained->GetStreamerInfo():0;
   if (!streamerInfo 
      || !streamerInfo->GetElements() 
      || !streamerInfo->GetElements()->GetSize())  return 0;

   if (!branch && parent) branch=parent->GetBranch();

   // we simply add all of our and the bases' members into one big list
   TList myStreamerElementsToCheck;
   myStreamerElementsToCheck.AddAll(streamerInfo->GetElements());

   Int_t numAdded=0;
   TStreamerElement* streamerElement=0;
   for (TObjLink *link = myStreamerElementsToCheck.FirstLink();
        link;
        link = link->Next() ) {
      streamerElement = (TStreamerElement*)link->GetObject();
      if (streamerElement->IsBase()) {
         // this is a base class place holder
         // continue with the base class's streamer info
         TClass* base=streamerElement->GetClassPointer();
         if (!base || !base->GetStreamerInfo()) continue;

         // add all of the base class's streamer elements 
         // (which in turn can be a base, which will be 
         // unfolded in a later iteration) to the list
         TObjArray* baseElements=base->GetStreamerInfo()->GetElements();
         if (!baseElements) continue;
         TIter iBaseSE(baseElements);
         TStreamerElement* baseSE=0;
         while ((baseSE=(TStreamerElement*)iBaseSE()))
            // we should probably check whether we're replacing something here...
            myStreamerElementsToCheck.Add(baseSE);
      } else if (!strcmp(streamerElement->GetName(),"This") 
         && !strcmp(clContained->GetName(), streamerElement->GetTypeName())) {
         // this is a collection of the real elements. 
         // So get the class ptr for these elements...
         TClass* clElements=streamerElement->GetClassPointer();
         TVirtualCollectionProxy* collProxy=clElements?clElements->GetCollectionProxy():0;
         clElements=collProxy?collProxy->GetValueClass():0;
         if (!clElements) continue;

         // now loop over the class's streamer elements
         streamerInfo= clElements->GetStreamerInfo();
         TIter iElem(streamerInfo->GetElements());
         TStreamerElement* elem=0;
         while ((elem=(TStreamerElement*)iElem())) {
            TNonSplitBrowsable* nsb=new TNonSplitBrowsable(elem, branch, parent);
            li.Add(nsb);
            numAdded++;
         }
      } else {
         // we have a basic streamer element
         TNonSplitBrowsable* nsb=new TNonSplitBrowsable(streamerElement, branch, parent);
         li.Add(nsb);
         numAdded++;
      }
   }
   return numAdded;
}

//______________________________________________________________________________
void TNonSplitBrowsable::Register() 
{
   // Wrapper for the registration method. Needed against MSVC, which 
   // assigned different addr to the same method, depending on what
   // translation unit you're in...
   TVirtualBranchBrowsable::RegisterGenerator(GetBrowsables);
}

//______________________________________________________________________________
void TNonSplitBrowsable::Unregister() 
{
   // Wrapper for the registration method. Needed against MSVC, which 
   // assigned different addr to the same method, depending on what
   // translation unit you're in...
   TVirtualBranchBrowsable::UnregisterGenerator(GetBrowsables);
}


ClassImp(TCollectionPropertyBrowsable);

//______________________________________________________________________________
//
// A tiny browser helper object (and its generator) for adding a virtual 
// (as in "not actually part of the class", not in C++ virtual) "@size()" 
// method to a collection. For all collections that derive from
// TCollection, or have a TVirtualCollectionProxy associated with them,
// a leaf is created that allows access to the number of elements in the 
// collection. For TClonesArrays and types with an associated 
// TVirtualCollectionProxy, this forwards to TTreeFormula's 
// "@branch.size()" functionality. For all other collections, a method call
// to the appropriate collection's member function is executed when drawing.
//
// These objects are of course only created for elements containing a 
// collection; the generator has no effect on any other elements.
//______________________________________________________________________________


//______________________________________________________________________________
void TCollectionPropertyBrowsable::Browse(TBrowser *b) 
{
   // Browses a TCollectionPropertyBrowsable. The only difference to
   // the generic TVirtualBranchBrowsable::Browse is our fDraw
   GetBranch()->GetTree()->Draw(GetDraw(), "", b ? b->GetDrawOption() : "");
   if (gPad) gPad->Update();
}

//______________________________________________________________________________
Int_t TCollectionPropertyBrowsable::GetBrowsables(TList& li, const TBranch* branch, 
                                                  const TVirtualBranchBrowsable* parent /* =0 */) 
{
// If the element to browse (given by either parent of branch) contains
// a collection (TClonesArray or something for which a TVirtualCollectionProxy
// exists), we will add some special objects to the browser. For now there is
// just one object "@size", returning the size of the collection (as in
// std::list::size(), or TClonesArray::GetEntries()).
// The objects we create are simply used to forward strings (like "@size") to
// TTreeFormula via our Browse method. These strings are stored in fName.
   TClass* clContained=0;
   TClass* clCollection=GetCollectionContainedType(branch, parent, clContained);
   if (!clCollection || !clContained) return 0;

   // Build the fDraw string. Start with our scope.
   TString scope;
   if (parent) {
      parent->GetScope(scope);
      branch=parent->GetBranch();
   } else if (branch){
      scope=branch->GetName();
      scope+=".";
      const TBranch* mother=branch;
      while (mother != mother->GetMother() && (mother=mother->GetMother())) {
         TString nameMother(mother->GetName());
         if (!nameMother.EndsWith(".")) {
            scope.Prepend(".");
            scope.Prepend(nameMother);
         } else {
            if (mother != mother->GetMother()) {
               // If the mother is the top level mother
               // and its ends ends with a ., the name is already
               // embedded!
               scope.Prepend(nameMother);
            }
         }
      }
   } else {
      if (gTree)
         gTree->Warning("GetBrowsables", "Neither branch nor parent is set!");
      return 0;
   }

   // remove trailing delimiters
   if (scope.EndsWith(".")) scope.Remove(scope.Length()-1, 1);
   else if (scope.EndsWith("->")) scope.Remove(scope.Length()-2, 2);

   // now prepend "@" to the last element of the scope,
   // to access the collection and not its containees
   Ssiz_t lastDot=scope.Last('.');
   Ssiz_t lastArrow=scope.Last('>'); // assuming there's no branch name containing ">"...
   Ssiz_t lastPart=lastDot;
   if (lastPart==kNPOS || (lastArrow!=kNPOS && lastPart<lastArrow))
      lastPart=lastArrow;
   if (lastPart==kNPOS) lastPart=0;
   else lastPart++;

   TString size_title("size of ");
   size_title += clCollection->GetName();
   if (clContained) {
      size_title += " of ";
      size_title += clContained->GetName();
   }

   if (clCollection->GetCollectionProxy() || clCollection==TClonesArray::Class()) {
   // the collection is one for which TTree::Draw supports @coll.size()

      TCollectionPropertyBrowsable* cpb;
      if ( clCollection->GetCollectionProxy() && 
           ( (clCollection->GetCollectionProxy()->GetValueClass()==0) 
           ||(clCollection->GetCollectionProxy()->GetValueClass()->GetCollectionProxy()!=0 
              && clCollection->GetCollectionProxy()->GetValueClass()->GetCollectionProxy()->GetValueClass()==0)
            )) {
         // If the contained type is not a class, we need an explitcit handle to get to the data.
         cpb = new TCollectionPropertyBrowsable("values", "values in the container", 
                                                scope, branch, parent);
         li.Add(cpb);
      }
      scope.Insert(lastPart, "@");
      cpb = new TCollectionPropertyBrowsable("@size", size_title, 
                                            scope+".size()", branch, parent);
      li.Add(cpb);
      return 1;
   } // if a collection proxy or TClonesArray
   else if (clCollection->InheritsFrom(TCollection::Class())) {
      // generic TCollection - we'll build size() ourselves, by mapping
      // it to the proper member function of the collection
      if (clCollection->InheritsFrom(TObjArray::Class()))
         scope+="@.GetEntries()";
      else scope+="@.GetSize()";
      TCollectionPropertyBrowsable* cpb=
         new TCollectionPropertyBrowsable("@size", size_title, scope, branch, parent);
      li.Add(cpb);
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
void TCollectionPropertyBrowsable::Register() 
{
   // Wrapper for the registration method. Needed against MSVC, which 
   // assigned different addr to the same method, depending on what
   // translation unit you're in...
   TVirtualBranchBrowsable::RegisterGenerator(GetBrowsables);
}

//______________________________________________________________________________
void TCollectionPropertyBrowsable::Unregister() 
{
   // Wrapper for the registration method. Needed against MSVC, which 
   // assigned different addr to the same method, depending on what
   // translation unit you're in...
   TVirtualBranchBrowsable::UnregisterGenerator(GetBrowsables);
}


ClassImp(TCollectionMethodBrowsable);

//______________________________________________________________________________
//
// TCollectionMethodBrowsable extends TCollectionPropertyBrowsable by showing
// all methods of the collection itself. If none are available - e.g. for STL 
// classes like std::list, a TVirtualBranchBrowsable object is reated instead. 
// The methods' names will have a "@" prepended, to distinguish them from the 
// contained elements' methods.
//
// This browser helper object is not part of the default list of registered
// generators (see TVirtualBranchBrowsable::RegisterDefaultGenerators()). 
// If you want to use it, you should call 
//   TCollectionMethodBrowsable::Register();
// As it extends the functionality of TVirtualBranchBrowsable, one might want 
// to unregister the generator of the "@size()" method by calling
//   TCollectionPropertyBrowsable::Unregister();
//______________________________________________________________________________


//______________________________________________________________________________
TCollectionMethodBrowsable::TCollectionMethodBrowsable(const TBranch* branch, TMethod* m, 
                                                       const TVirtualBranchBrowsable* parent /*=0*/):
TMethodBrowsable(branch, m, parent) 
{
   // Contructor, see TMethodBrowsable's constructor.
   // Prepends "@" to the name to make this method work on the container.
   SetName(TString("@")+GetName());
}

//______________________________________________________________________________
Int_t TCollectionMethodBrowsable::GetBrowsables(TList& li, const TBranch* branch, 
                                                const TVirtualBranchBrowsable* parent /*=0*/) 
{
// This methods fills list with TMethodBrowsables
// for the branch's or parent's collection class and its base classes, 
// and returns the number of added elements. If called from a TBranch::Browse 
// overload, "branch" should be set to the calling TBranch, otherwise 
// "parent" should be set to the TVirtualBranchBrowsable being browsed.
   TClass* clContained=0;
   // we don't care about the contained class, but only about the collections, 
   TClass* clContainer=GetCollectionContainedType(branch, parent, clContained);
   if (!clContainer || !clContained) return 0;

   TList listMethods;
   GetBrowsableMethodsForClass(clContainer, listMethods);
   TMethod* method=0;
   TIter iMethods(&listMethods);
   while ((method=(TMethod*)iMethods()))
      li.Add(new TCollectionMethodBrowsable(branch, method, parent));

   // if we have no methods, and if the class has a collection proxy, just add
   // the corresponding TCollectionPropertyBrowsable instead.
   // But only do that if TCollectionPropertyBrowsable is not generatated anyway
   // - we don't need two of them.
   if (!listMethods.GetSize() && clContainer->GetCollectionProxy()) {
      std::list<MethodCreateListOfBrowsables_t>& listGenerators=GetRegisteredGenerators();
      std::list<MethodCreateListOfBrowsables_t>::iterator iIsRegistered
         = std::find(listGenerators.begin(), listGenerators.end(), &TCollectionPropertyBrowsable::GetBrowsables);
      if (iIsRegistered==listGenerators.end()) {
         TCollectionPropertyBrowsable::GetBrowsables(li, branch, parent);
         return 1;
      }
   }
   return listMethods.GetSize();
}

//______________________________________________________________________________
void TCollectionMethodBrowsable::Register() 
{
   // Wrapper for the registration method. Needed against MSVC, which 
   // assigned different addr to the same method, depending on what
   // translation unit you're in...
   TVirtualBranchBrowsable::RegisterGenerator(GetBrowsables);
}

//______________________________________________________________________________
void TCollectionMethodBrowsable::Unregister() 
{
   // Wrapper for the registration method. Needed against MSVC, which 
   // assigned different addr to the same method, depending on what
   // translation unit you're in...
   TVirtualBranchBrowsable::UnregisterGenerator(GetBrowsables);
}
