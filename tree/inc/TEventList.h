// @(#)root/tree:$Name:  $:$Id: TEventList.h,v 1.2 2000/11/21 20:47:45 brun Exp $
// Author: Rene Brun   11/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEventList
#define ROOT_TEventList


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEventList                                                           //
//                                                                      //
// A list of selected entries in a TTree.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TDirectory;

class TEventList : public TNamed {

protected:
        Int_t            fN;           //  Number of elements in the list
        Int_t            fSize;        //  Size of array
        Int_t            fDelta;       //  Increment size
        Int_t            *fList;       //[fN]Array of elements
        TDirectory       *fDirectory;  //! Pointer to directory holding this tree

public:
        TEventList();
        TEventList(const char *name, const char *title="",Int_t initsize=0, Int_t delta = 0);
        TEventList(const TEventList &list);
        virtual          ~TEventList();
        virtual void     Add(const TEventList *list);
        virtual Bool_t   Contains(Int_t entry);
        virtual void     Enter(Int_t entry);
        TDirectory      *GetDirectory() const {return fDirectory;}
        virtual Int_t    GetEntry(Int_t index) const;
        virtual Int_t    GetIndex(Int_t entry) const;
        virtual Int_t   *GetList() const { return fList; }
        virtual Int_t    GetN() const { return fN; }
        virtual Int_t    GetSize() const { return fSize; }
        virtual void     Print(Option_t *option="") const;
        virtual void     Reset(Option_t *option="");
        virtual void     Resize(Int_t delta=0);
        virtual void     SetDelta(Int_t delta=100) {fDelta = delta;}
        virtual void     SetDirectory(TDirectory *dir);
        virtual void     SetName(const char *name); // *MENU*
        virtual void     Sort();
        virtual void     Subtract(const TEventList *list);

        TEventList& operator=(const TEventList &list);
 friend TEventList  operator+(const TEventList &list1, const TEventList &list2);
 friend TEventList  operator-(const TEventList &list1, const TEventList &list2);

        ClassDef(TEventList,2)  //A list of selected entries in a TTree.
};

#endif

