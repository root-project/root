// @(#)root/graf:$Name:  $:$Id: TLink.cxx,v 1.1.1.1 2000/05/16 17:00:49 rdm Exp $
// Author: Rene Brun   05/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdio.h>

#include "TVirtualPad.h"
#include "TClass.h"
#include "TROOT.h"
#include "TLink.h"

ClassImp(TLink)

//______________________________________________________________________________
//
// Special TText object used to show hyperlinks.
// In the example below created by TObject::Inspect, TLinks are used
// to show pointers to other objects.
// Clicking on one link, inspect the corresponding object.
//Begin_Html
/*
<img src="gif/link.gif">
*/
//End_Html
//

//______________________________________________________________________________
TLink::TLink() : TText()
{
//*-*-*-*-*-*-*-*-*-*-*Link default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================

   fLink  = 0;
}

//______________________________________________________________________________
TLink::TLink(Double_t x, Double_t y, void *pointer)
           : TText(x, y, "")
{
//*-*-*-*-*-*-*-*-*-*Constructor to define a link object*-*-*-*-*-*-*-*
//*-*                ========================================
// pointer points to any kind of object.
//
   fLink  = pointer;
   static char line[16];
   sprintf(line,"->%lx ", (Long_t)pointer);
   SetTitle(line);
}

//______________________________________________________________________________
TLink::~TLink()
{
//*-*-*-*-*-*-*-*-*-*-*Link default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =============================

}

//______________________________________________________________________________
void TLink::ExecuteEvent(Int_t event, Int_t, Int_t)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//  This member function is called when a link is clicked with the locator
//
//  If mouse is clicked on a link text, the object pointed by the link
//  is Inspected
//

  if (event == kMouseMotion)
     gPad->SetCursor(kHand);

  if (event != kButton1Up) return;

  TObject *idcur = (TObject*)fLink;
  if (!idcur) return;
  TClass *cl = gROOT->GetClass(GetName());
  if (!cl) return;

//*-*- make a special case for top of Collections
//*-*-    if status word is 0 take first member
//*-*-    otherwise go back to parent of linked list
  if (cl->InheritsFrom(TCollection::Class())) {
     TList *lh = (TList*)idcur;
     if (!TestBit(kObjIsParent))  idcur = lh->First();
     else                         idcur = lh->GetParent();
     if (!idcur) return;
     idcur->Inspect();
     return;
  }

//*-*- check if link points to a TObject
   TClass *c1 = (TClass*)cl->GetBaseClass("TObject");
   if (!c1) return;

   idcur->Inspect();
}
