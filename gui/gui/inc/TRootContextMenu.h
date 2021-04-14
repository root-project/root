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

   virtual void   DisplayPopup(Int_t x, Int_t y);
   virtual void   Dialog(TObject *object, TMethod *method);
   virtual void   Dialog(TObject *object, TFunction *function);
   virtual void   DrawEntry(TGMenuEntry *entry);
   TRootDialog   *GetDialog() const { return fDialog; };
   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual void   OnlineHelp();
   virtual void   RecursiveRemove(TObject *obj);

   Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

protected:
   TGPopupMenu * FindHierarchy(const char *commentstring, TString &last_component);
   void AddEntrySorted(TGPopupMenu *current, const char *s, Int_t id, void *ud = nullptr,
                       const TGPicture *p = nullptr, Bool_t sorted = kTRUE);

   ClassDef(TRootContextMenu,0)  //ROOT native GUI context sensitive popup menu
};

#endif
