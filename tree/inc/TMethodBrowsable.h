// @(#)root/tree:$Name:  $:$Id: TMethodBrowsable.h,v 1.34 2004/07/29 10:54:54 brun Exp $
// Author: Axel Naumann   14/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMethodBrowsable
#define ROOT_TMethodBrowsable

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMethodBrowsable                                                     //
//                                                                      //
// A helper object to browse methods                                    //
// (see TBranchElement::GetBrowsableMethods)                            //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TMethod;
class TBowser;
class TClass;
class TBranchElement;
class TString;

class TMethodBrowsable: public TNamed {
 public:
   TMethodBrowsable(TBranchElement* be, TMethod* m, 
                    TMethodBrowsable* parent=0);
   ~TMethodBrowsable() {};

   void Browse(TBrowser *b);
   const char *GetIconName() const {
      if (fReturnClass)
         return "TMethodBrowsable-branch"; 
      return "TMethodBrowsable-leaf";}
   static TList* GetMethodBrowsables(TBranchElement* be, TClass* cl, 
                                     TMethodBrowsable* parent=0);
   Bool_t IsFolder() const { 
      return (fReturnClass); }
   static Bool_t IsMethodBrowsable(TMethod* m);

 private:
   void GetScope(TString & scope);
   
   TBranchElement   *fBranchElement; // pointer to the branch element representing the top object
   TMethodBrowsable *fParent; // parent method if this method is member of a returned class
   TMethod          *fMethod; // pointer to a method
   TClass           *fReturnClass; // pointer to TClass representing return type
   TList            *fReturnLeafs; // pointer to return type's leafs
   Bool_t            fReturnIsPointer; // return type is pointer to class
   ClassDef(TMethodBrowsable,0) // Helper object to browse methods
};

#endif
