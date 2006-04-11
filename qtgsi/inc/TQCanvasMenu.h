// @(#)root/qtgsi:$Name:$:$Id:$
// Author: Denis Bertini, M. AL-Turany  01/11/2000

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQCanvasMenu
#define ROOT_TQCanvasMenu

////////////////////////////////////////////////////////////////////////////////
//
// TQCanvasMenu
//
// This class provides an interface to Qt based context sensitive popup menus.
// These menus pop up when the user hits the right mouse button, and
// are destroyed when the menu pops downs.
//
////////////////////////////////////////////////////////////////////////////////

#include "qobject.h"
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TMethodArg
#include "TMethodArg.h"
#endif
#ifndef ROOT_TQRootDialog
#include "TQRootDialog.h"
#endif

class QPopupMenu;
class QAction;
class QMouseEvent;
class QResizeEvent;
class QPaintEvent;
class TCanvas;
class TObject;
class QWidget;
class TMethod;


class TQCanvasMenu : public QObject {
   Q_OBJECT
public:
   TQCanvasMenu(QWidget* parent = 0, TCanvas *canvas = 0);
   TQCanvasMenu(QWidget* parent, QWidget *tabWin, TCanvas *canvas) ;
   ~TQCanvasMenu();
   void Popup(TObject *obj, double x, double y, QMouseEvent *e);
   void Dialog(TObject *obj, TMethod* method);
   char* CreateDialogTitle( TObject *object, TMethod *method );
   char* CreateArgumentTitle(TMethodArg *argument);

public slots:
   void Execute(int id);

protected:
   TObject* fCurrObj;         // current selected object
   QPopupMenu  *fPopup;       // Qt popup menu
   TList fMethods;            // list of Root metheds associated with the selected object
   TCanvas *fc;               // pointer to the ROOT canvas
   TQRootDialog *fDialog;     // the TQRootDialog which is used to prompt for
                              //the arguments of an object's member function.
   QWidget *fParent,*fTabWin; //parents widgets
   double fMousePosX;         // mouse position in user coordinate
   double fMousePosY;         // mouse position in user coordinate
};

#endif
