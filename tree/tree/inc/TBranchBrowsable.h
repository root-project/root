// @(#)root/tree:$Id$
// Author: Axel Naumann   14/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranchBrowsable
#define ROOT_TBranchBrowsable

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TList
#include "TList.h"
#endif

#include <list>

class TMethod;
class TBowser;
class TClass;
class TBranch;
class TBranchElement;
class TString;
class TStreamerElement;

class TVirtualBranchBrowsable: public TNamed {
public: 

   // these methods are registered in RegisterGenerator, and 
   // called to create the list of browsables. See e.g. 
   // TMethodBrowsable::Register
   typedef Int_t (*MethodCreateListOfBrowsables_t)
      (TList&, const TBranch* branch, const TVirtualBranchBrowsable* parent); 

   ~TVirtualBranchBrowsable();

   void Browse(TBrowser *b);
   const char *GetIconName() const {
      // return icon shown when browsing a TVirtualBranchBrowsable
      if (IsFolder()) return "TBranchElement-folder";
      else return "TBranchElement-leaf"; 
   }
   void GetScope(TString & scope) const;
   Bool_t IsFolder() const {
      // check whether we have sub-elements
      return (GetLeaves() && GetLeaves()->GetSize()); }

   static Int_t FillListOfBrowsables(TList& list, const TBranch* branch,
                                     const TVirtualBranchBrowsable* parent=0);

   const TBranch* GetBranch() const { 
      // return the parent branch (might be many levels up)
      return fBranch; }
   const TVirtualBranchBrowsable* GetParent() const { 
      // return the parent TVirtualBranchBrowsable
      return fParent; }
   TClass* GetClassType() const { 
      // return the type of this browsable object
      return fClass; }
   Bool_t TypeIsPointer() const { 
      // return whether the type of this browsable object is a pointer
      return fTypeIsPointer; }
   TList* GetLeaves() const; 

   // static void Register()   has to be implemented for all derived classes!
   // static void Unregister() has to be implemented for all derived classes!

protected:
   TVirtualBranchBrowsable(const TBranch* b, TClass* type, Bool_t typeIsPointer, 
      const TVirtualBranchBrowsable* parent=0);
   static TClass* GetCollectionContainedType(const TBranch* b, 
      const TVirtualBranchBrowsable* parent, TClass* &contained);
   static std::list<MethodCreateListOfBrowsables_t>& GetRegisteredGenerators();
   static void RegisterGenerator(MethodCreateListOfBrowsables_t generator);
   static void UnregisterGenerator(MethodCreateListOfBrowsables_t generator);
   void SetType(TClass* type) { 
      // sets the type of this browsable object
      fClass=type; }
   void SetTypeIsPointer(Bool_t set=kTRUE) { 
      // sets whether the type of this browsable object is a pointer
      fTypeIsPointer=set; }

private:
   static void RegisterDefaultGenerators();
   const TBranch    *fBranch; // pointer to the branch element representing the top object
   const TVirtualBranchBrowsable *fParent; // parent method if this method is member of a returned class
   TList            *fLeaves; // pointer to leaves
   TClass           *fClass; // pointer to TClass representing our type (i.e. return type for methods), 0 if basic type
   Bool_t            fTypeIsPointer; // return type is pointer to class
   static std::list<MethodCreateListOfBrowsables_t> fgGenerators; // list of MethodCreateListOfBrowsables_t called by CreateListOfBrowsables
   static Bool_t     fgGeneratorsSet; // have we set the generators yet? empty is not good enough - user might have removed them
   ClassDef(TVirtualBranchBrowsable, 0); // Base class for helper objects used for browsing
};


class TMethodBrowsable: public TVirtualBranchBrowsable {
public:
   ~TMethodBrowsable() {};

   static Int_t GetBrowsables(TList& list, const TBranch* branch,
                              const TVirtualBranchBrowsable* parent=0);
   const char *GetIconName() const {
      // return our special icons
      if (IsFolder()) return "TMethodBrowsable-branch"; 
      return "TMethodBrowsable-leaf";}
   static Bool_t IsMethodBrowsable(const TMethod* m);
   static void Register();
   static void Unregister();

protected:
   static void GetBrowsableMethodsForClass(TClass* cl, TList& list);
   TMethodBrowsable(const TBranch* branch, TMethod* m, 
      const TVirtualBranchBrowsable* parent=0);

private:
   TMethod         *fMethod; // pointer to a method
   ClassDef(TMethodBrowsable,0); // Helper object to browse methods
};


class TNonSplitBrowsable: public TVirtualBranchBrowsable {
public:
   ~TNonSplitBrowsable() {}

   static Int_t GetBrowsables(TList& list, const TBranch* branch, 
                              const TVirtualBranchBrowsable* parent=0);
   static void Register();
   static void Unregister();

protected:
   TNonSplitBrowsable(const TStreamerElement* element, const TBranch* branch, 
      const TVirtualBranchBrowsable* parent=0);

private:
   ClassDef(TNonSplitBrowsable, 0); // Helper object to browse unsplit objects
};


class TCollectionPropertyBrowsable: public TVirtualBranchBrowsable {
public:
   ~TCollectionPropertyBrowsable() {}

   void Browse(TBrowser *b);
   static Int_t GetBrowsables(TList& list, const TBranch* branch, 
                              const TVirtualBranchBrowsable* parent=0);
   const char* GetDraw() const {
      // return the string passed to TTree::Draw
      return fDraw.Data(); }
   static void Register();
   static void Unregister();

protected:
   TCollectionPropertyBrowsable(const char* name, const char* title, 
      const char* draw, const TBranch* branch, const TVirtualBranchBrowsable* parent=0): 
   TVirtualBranchBrowsable(branch, 0, kFALSE, parent), fDraw(draw) {
      // constructor, which sets the name and title according to the parameters
      // (and thus differently than our base class TVirtualBranchBrowsable)
      SetNameTitle(name, title);
   }

private:
   TString fDraw; // string to send to TTree::Draw(), NOT by GetScope()!
   ClassDef(TCollectionPropertyBrowsable, 0); // Helper object to add browsable collection properties
};

class TCollectionMethodBrowsable: public TMethodBrowsable {
public:
   ~TCollectionMethodBrowsable() {};

   static Int_t GetBrowsables(TList& list, const TBranch* branch, 
                              const TVirtualBranchBrowsable* parent=0);
   static void Register();
   static void Unregister();

protected:
   TCollectionMethodBrowsable(const TBranch* branch, TMethod* m, 
      const TVirtualBranchBrowsable* parent=0);

   ClassDef(TCollectionMethodBrowsable,0); // Helper object to browse a collection's methods
};

#endif // defined ROOT_TBranchBrowsable
