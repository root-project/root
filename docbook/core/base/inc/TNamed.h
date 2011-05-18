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


#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


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
   virtual ~TNamed() { }
   virtual void     Clear(Option_t *option ="");
   virtual TObject *Clone(const char *newname="") const;
   virtual Int_t    Compare(const TObject *obj) const;
   virtual void     Copy(TObject &named) const;
   virtual void     FillBuffer(char *&buffer);
   virtual const char  *GetName() const { return fName; }
   virtual const char  *GetTitle() const { return fTitle; }
   virtual ULong_t  Hash() const { return fName.Hash(); }
   virtual Bool_t   IsSortable() const { return kTRUE; }
   virtual void     SetName(const char *name); // *MENU*
   virtual void     SetNameTitle(const char *name, const char *title);
   virtual void     SetTitle(const char *title=""); // *MENU*
   virtual void     ls(Option_t *option="") const;
   virtual void     Print(Option_t *option="") const;
   virtual Int_t    Sizeof() const;

   ClassDef(TNamed,1)  //The basis for a named object (name, title)
};

#endif
