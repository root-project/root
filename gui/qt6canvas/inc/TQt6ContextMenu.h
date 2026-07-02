// Author: Sergey Linev   2/07/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_QRootContextMenu
#define ROOT_QRootContextMenu

#include "TContextMenuImp.h"
#include "TObject.h"

class QRootContextMenu : public TObject, public TContextMenuImp {

private:

   QRootContextMenu(const QRootContextMenu&) = delete;
   QRootContextMenu& operator=(const QRootContextMenu&)  = delete;

public:
   QRootContextMenu(TContextMenu *c = nullptr, const char *name = "ROOT Context Menu");
   ~QRootContextMenu() override;

   void   DisplayPopup(Int_t x, Int_t y) override;
   void   Dialog(TObject *object, TMethod *method) override;
   void   Dialog(TObject *object, TFunction *function) override;

   void   RecursiveRemove(TObject *obj) override;

   ClassDefOverride(QRootContextMenu,0)  // Qt6 GUI context sensitive popup menu
};

#endif
