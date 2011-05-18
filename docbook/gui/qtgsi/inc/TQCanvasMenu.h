// @(#)root/qtgsi:$Id$
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

#ifndef __CINT__
#include "qobject.h"
# if (QT_VERSION > 0x039999) // Added by cholm@nbi.dk - for Qt 3
#  include <q3popupmenu.h>
typedef Q3PopupMenu QPopupMenu;
# else 
class QPopupMenu;
# endif
#else
class QPopupMenu;
#endif

#ifndef ROOT_TList
#include "TList.h"
#endif

class TCanvas;
class TObject;
class TMethodArg;
class TQRootDialog;

class QObject;
class QAction;
class QMouseEvent;
class QResizeEvent;
class QPaintEvent;
class QWidget;

class TQCanvasMenu : public QObject {
#ifndef __CINT__
   Q_OBJECT
#endif
private:
   TQCanvasMenu(const TQCanvasMenu &c);
   TQCanvasMenu& operator=(const TQCanvasMenu&) {return *this;} 
      
public:
   TQCanvasMenu(QWidget* parent = 0, TCanvas *canvas = 0);
   TQCanvasMenu(QWidget* parent, QWidget *tabWin, TCanvas *canvas) ;
   virtual ~TQCanvasMenu();
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

   ClassDef(TQCanvasMenu,1)  //interface to Qt based context sensitive popup menus
};
   
#endif
