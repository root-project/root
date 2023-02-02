// @(#)root/base:$Id$
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TNamed
#define ROOT_TNamed


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNamed                                                               //
//                                                                      //
// The basis for a named object (name, title).                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TObject.h"
#include "TString.h"


class TNamed : public TObject {

protected:
   TString   fName;            //object identifier
   TString   fTitle;           //object title

public:
   TNamed(): fName(), fTitle() { }
   TNamed(const char *name, const char *title) : fName(name), fTitle(title) { }
   TNamed(const TString &name, const TString &title) : fName(name), fTitle(title) { }
   TNamed(const TNamed &named);
   TNamed& operator=(const TNamed& rhs);
   virtual ~TNamed();
            void     Clear(Option_t *option ="") override;
            TObject *Clone(const char *newname="") const override;
            Int_t    Compare(const TObject *obj) const override;
            void     Copy(TObject &named) const override;
   virtual  void     FillBuffer(char *&buffer);
            const char  *GetName() const override { return fName; }
            const char  *GetTitle() const override { return fTitle; }
            ULong_t  Hash() const override { return fName.Hash(); }
            Bool_t   IsSortable() const override { return kTRUE; }
   virtual  void     SetName(const char *name); // *MENU*
   virtual  void     SetNameTitle(const char *name, const char *title);
   virtual  void     SetTitle(const char *title=""); // *MENU*
            void     ls(Option_t *option="") const override;
            void     Print(Option_t *option="") const override;
   virtual  Int_t    Sizeof() const;

   ClassDefOverride(TNamed,1)  //The basis for a named object (name, title)
};

#endif
