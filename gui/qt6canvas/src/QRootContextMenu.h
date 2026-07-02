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

#include <QObject>
#include <QString>
#include "TContextMenuImp.h"
#include "TObject.h"
#include "TList.h"
#include <map>

class QSignalMapper;
class TList;
class QAction;
class QMenu;

class QRootContextMenu : public QObject, public TObject, public TContextMenuImp {
   Q_OBJECT

public slots:
   void executeMenu(int id);

protected:
   TObject       *fMenuObj = nullptr;      // object use to fill menu

   TList     fTrash;
   std::map<int, void*> fCustomArg;
   QAction* addMenuAction(QMenu *menu, QSignalMapper *map, const QString &text, int id, void *arg = nullptr);

public:
   QRootContextMenu(TContextMenu *c = nullptr, const char *name = "ROOT Context Menu");
   ~QRootContextMenu() override;

   void   DisplayPopup(Int_t x, Int_t y) override;
   void   Dialog(TObject *object, TMethod *method) override;
   void   Dialog(TObject *object, TFunction *function) override;

   void   RecursiveRemove(TObject *obj) override;
};

#endif
