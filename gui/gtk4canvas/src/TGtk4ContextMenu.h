// Author: Sergey Linev   13/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGtk4ContextMenu
#define ROOT_TGtk4ContextMenu

#include "TContextMenuImp.h"
#include "TObject.h"
#include "TList.h"
#include <map>


class TGtk4ContextMenu : public TObject, public TContextMenuImp {

protected:

   TList     fTrash;
   std::map<int, void*> fCustomArg;

/*
   QAction* addMenuAction(QMenu *menu, QSignalMapper *map, const QString &text, int id, void *arg = nullptr);

   void AddColorElements(int colindx, QFormLayout *layout);
   void UpdateColorElements();

   void SetLineAttributesDialog();
   void SetFillAttributesDialog();
   void SetTextAttributesDialog();
   void SetMarkerAttributesDialog();
*/

   void executeMenu(int id);

public:
   TGtk4ContextMenu(TContextMenu *c = nullptr, const char *name = "ROOT Context Menu");
   ~TGtk4ContextMenu() override;

   void   DisplayPopup(Int_t x, Int_t y) override;
   void   Dialog(TObject *object, TMethod *method) override;
   void   Dialog(TObject *object, TFunction *function) override;

   void   RecursiveRemove(TObject *obj) override;
};

#endif
