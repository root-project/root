// @(#)root/tree:$Name:  $:$Id: TMethodBrowsable.cxx,v 1.142 2004/08/24 10:41:58 brun Exp $
// Author: Axel Naumann   14/10/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMethodBrowsable                                                     //
//                                                                      //
// A helper object to browse methods (see                               //
// TBranchElement::GetBrowsableMethods)                                 //
//////////////////////////////////////////////////////////////////////////

#include "TMethodBrowsable.h"
#include "TBranchElement.h"
#include "TMethod.h"
#include "TBrowser.h"
#include "TTree.h"
#include "TPad.h"
#include "TClass.h"
#include "TBaseClass.h"

ClassImp(TMethodBrowsable);

//______________________________________________________________________________
TMethodBrowsable::TMethodBrowsable(TBranchElement* be, TMethod* m,
                                   TMethodBrowsable* parent /* =0 */):
      TNamed("", m->GetPrototype()), 
      fBranchElement(be), fParent(parent), fMethod(m), 
      fReturnClass(0), fReturnLeafs(0), fReturnIsPointer(kFALSE) {
// standard constructor.
// Links a TBranchElement with a TMethod, allowing the TBrowser to
// Browser simple methods.
//
// The c'tor sets the name for a method "Class::Method(params) const"
// to "Method(params)", title to TMethod::GetPrototype
   TString name(m->GetName());
   name+=m->GetSignature();
   if (name.EndsWith(" const")) name.Remove(name.Length()-6);
   SetName(name);

   TString plainReturnType(m->GetReturnTypeName());
   if (plainReturnType.EndsWith("*")) {
      fReturnIsPointer=kTRUE;
      plainReturnType.Remove(plainReturnType.Length()-1);
      plainReturnType.Strip();
   }
   fReturnClass=gROOT->GetClass(plainReturnType);
}

//______________________________________________________________________________
void TMethodBrowsable::Browse(TBrowser *b) {
// Calls TTree::Draw on the method if return type is not a class;
// otherwise expands returned object's "folder"

   if (!fReturnClass) {
      TString name;
      GetScope(name);
      fBranchElement->GetTree()->Draw(name, "", b ? b->GetDrawOption() : "");
      if (gPad) gPad->Update();
   } else {
      if (!fReturnLeafs)
         fReturnLeafs=GetMethodBrowsables(fBranchElement, fReturnClass, this);
      if (fReturnLeafs)
         fReturnLeafs->Browse(b);
   }
}

//______________________________________________________________________________
TList* TMethodBrowsable::GetMethodBrowsables(TBranchElement* be, TClass* cl,
                                             TMethodBrowsable* parent /* =0 */) {
// Given a class, this methods returns a list of TMethodBrowsables
// for the class and its base classes.
// This list has to be deleted by the caller!

   if (!cl) return 0;

   TList allClasses;
   allClasses.Add(cl);
   
   for(TObjLink* lnk=allClasses.FirstLink();
       lnk; lnk=lnk->Next()){
      cl=(TClass*)lnk->GetObject();
      TList* bases=cl->GetListOfBases();
      TBaseClass* base;
      TIter iB(bases);
      while ((base=(TBaseClass*)iB())) {
         TClass* bc=base->GetClassPointer();
         if (bc) allClasses.Add(bc);
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
   TList* browsableMethods=new TList();
   browsableMethods->SetOwner();
   while ((m=(TMethod*)iM()))
      if (TMethodBrowsable::IsMethodBrowsable(m))
         browsableMethods->Add(new TMethodBrowsable(be, m, parent));

   return browsableMethods;
}

//______________________________________________________________________________
void TMethodBrowsable::GetScope(TString & scope) {
// Returns the full name for TTree::Draw to draw *this.
// Recursively appends, starting form the top TBranchElement,
// all method names with proper reference operators (->, .)
// depending on fReturnIsPointer.

   if (fParent)
      fParent->GetScope(scope);
   else {
      scope=fBranchElement->GetName();
      scope+=".";
   }
   scope+=GetName();
   if (fReturnClass) // otherwise we're a leaf, and we are the one drawn
      if (fReturnIsPointer)
         scope+="->";
      else scope+=".";
}

//______________________________________________________________________________
Bool_t TMethodBrowsable::IsMethodBrowsable(TMethod* m) {
// A TMethod is browsable if it is const, public and not pure virtual,
// if does not have any parameter without default value, and if it has 
// a (non-void) return value.

   return (m->GetNargs()-m->GetNargsOpt()==0
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
           && strcmp(m->GetName(),"IsZombie")); 
}
