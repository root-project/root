// @(#)root/guibuilder:$Name:  $:$Id: TGFrame.cxx,v 1.78 2004/09/13 09:10:08 rdm Exp $
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBuilder                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGuiBuilder.h"
#include "TGuiBldDragManager.h"
#include "TGShutter.h"
#include "TGLayout.h"
#include "TGResourcePool.h"
#include "TGButton.h"
#include "TROOT.h"

ClassImp(TGuiBuilder)


////////////////////////////////////////////////////////////////////////////////
TGuiBuilder::TGuiBuilder(const TGWindow *p) : TVirtualGuiBld(),
             TGMainFrame(p ? p : gClient->GetDefaultRoot(), 1, 1)
{
   // ctor

   SetCleanup(kTRUE);
   fEditDisabled = kTRUE;

   if (gDragManager) ((TGuiBldDragManager *)gDragManager)->SetBuilder(this);

   fShutter = new TGShutter(this, kSunkenFrame);
   AddFrame(fShutter, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   AddSection("Projects");
   AddSection("Standard");
   AddSection("Containers");
//   AddSection("Extended");

   TGuiBldAction *act = new TGuiBldAction("TGMainFrame", "Main Frame", kGuiBldProj);
   act->fAct = "new TGMainFrame(gClient->GetRoot(), 300, 300)";
   act->fPic = "bld_mainframe.xpm";
   AddAction(act, "Projects");

   // Standard
   act = new TGuiBldAction("TGTextButton", "Text Button", kGuiBldCtor);
   act->fAct = "new TGTextButton()";
   act->fPic = "bld_textbutton.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGCheckButton", "Check Button", kGuiBldCtor);
   act->fAct = "new TGCheckButton()";
   act->fPic = "bld_checkbutton.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGRadioButton", "Radio Button", kGuiBldCtor);
   act->fAct = "new TGRadioButton()";
   act->fPic = "bld_radiobutton.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGTextEntry", "Text Entry", kGuiBldCtor);
   act->fAct = "new TGTextEntry()";
   act->fPic = "bld_entry.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGNumberEntry", "Number Entry", kGuiBldCtor);
   act->fAct = "new TGNumberEntry()";
   act->fPic = "bld_numberentry.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGLabel", "Text Label", kGuiBldCtor);
   act->fAct = "new TGLabel()";
   act->fPic = "bld_label.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGHorizontal3DLine", "Horizontal ScrollBar", kGuiBldCtor);
   act->fAct = "new TGHorizontal3DLine()";
   act->fPic = "bld_hseparator.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGVertical3DLine", "Vertical ScrollBar", kGuiBldCtor);
   act->fAct = "new TGVertical3DLine()";
   act->fPic = "bld_vseparator.xpm";
   AddAction(act, "Standard");

   // Containers
   act = new TGuiBldAction("TGHorizontalFrame", "Horizontal Frame", kGuiBldCtor);
   act->fAct = "new TGHorizontalFrame()";
   act->fPic = "bld_hbox.xpm";
   AddAction(act, "Containers");

   act = new TGuiBldAction("TGVerticalFrame", "Vertical Frame", kGuiBldCtor);
   act->fAct = "new TGVerticalFrame()";
   act->fPic = "bld_vbox.xpm";
   AddAction(act, "Containers");

   MapSubwindows();
   Resize(80, 300);
   SetWindowName("GUI Builder");
   MapRaised();
}

//______________________________________________________________________________
TGuiBuilder::~TGuiBuilder()
{
   // destructor

}

//______________________________________________________________________________
void TGuiBuilder::AddAction(TGuiBldAction *act, const char *sect)
{
   //

   if (!act || !sect) return;

   TGLayoutHints *l = new TGLayoutHints(kLHintsTop | kLHintsCenterX, 5, 5, 5, 0);
   TGShutterItem *item = fShutter->GetItem(sect);
   TGButton *btn = 0;

   if (!item) return;
   TGCompositeFrame *cont = (TGCompositeFrame *)item->GetContainer();

   const TGPicture *pic = fClient->GetPicture(act->fPic);

   if (pic) {
      btn = new TGPictureButton(cont, pic);
   } else {
      btn = new TGTextButton(cont, act->GetName());
   }

   btn->SetToolTipText(act->GetTitle());
   btn->SetUserData((void*)act);
   btn->Connect("Clicked()", "TGuiBuilder", this, "HandleButtons()");

   cont->AddFrame(btn, l);
   cont->MapSubwindows();
   cont->Resize();  // invoke Layout()
}

//______________________________________________________________________________
void TGuiBuilder::AddSection(const char *sect)
{
   //

   static int id = 10000;
   TGShutterItem *item = new TGShutterItem(fShutter, new TGHotString(sect), id++);
   fShutter->AddItem(item);
}

//______________________________________________________________________________
void TGuiBuilder::HandleButtons()
{
   //

   TGButton *btn = (TGButton *)gTQSender;
   TGuiBldAction *act  = (TGuiBldAction *)btn->GetUserData();

   if (act) {
      fAction = act;
      if (fAction->fType == kGuiBldProj) ExecuteAction();
   }
}

//______________________________________________________________________________
TGFrame *TGuiBuilder::ExecuteAction()
{
   //

   if (!fAction || fAction->fAct.IsNull()) return 0;

   TGFrame *ret = 0;
   TGWindow *root = (TGWindow*)fClient->GetRoot();

   switch (fAction->fType) {
      case kGuiBldProj:
         root->SetEditable(kFALSE);
         ret = (TGFrame *)gROOT->ProcessLineFast(fAction->fAct.Data());
         ret->MapRaised();
         ret->SetEditable(kTRUE);
         break;
      default:
         ret = (TGFrame *)gROOT->ProcessLineFast(fAction->fAct.Data());
         break;
   }

   fAction = 0;
   return ret;
}
