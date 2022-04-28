// @(#)root/gui:$Id$
// Author: Fons Rademakers   12/02/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootContextMenu
#define ROOT_TRootContextMenu


#include "TContextMenuImp.h"
#include "TGMenu.h"

class TRootDialog;


class TRootContextMenu : public TGPopupMenu, public TContextMenuImp {

private:
   TRootDialog *fDialog;    ///< dialog prompting for command line arguments
   TList       *fTrash;     ///< list of objects to be deleted before refilling menu

   TRootContextMenu(const TRootContextMenu&);
   TRootContextMenu& operator=(const TRootContextMenu&);
   void CreateMenu(TObject *object);

public:
   TRootContextMenu(TContextMenu *c = nullptr, const char *name = "ROOT Context Menu");
   virtual ~TRootContextMenu();

   void   DisplayPopup(Int_t x, Int_t y) override;
   void   Dialog(TObject *object, TMethod *method) override;
   void   Dialog(TObject *object, TFunction *function) override;
   void   DrawEntry(TGMenuEntry *entry) override;
   TRootDialog   *GetDialog() const { return fDialog; };
   Bool_t HandleButton(Event_t *event) override;
   Bool_t HandleCrossing(Event_t *event) override;
   Bool_t HandleMotion(Event_t *event) override;
   virtual void   OnlineHelp();
   void   RecursiveRemove(TObject *obj) override;

   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;

protected:
   TGPopupMenu * FindHierarchy(const char *commentstring, TString &last_component);
   void AddEntrySorted(TGPopupMenu *current, const char *s, Int_t id, void *ud = nullptr,
                       const TGPicture *p = nullptr, Bool_t sorted = kTRUE);

   ClassDefOverride(TRootContextMenu,0)  //ROOT native GUI context sensitive popup menu
};

#endif
