// @(#)root/qtgsi:$Name:  $:$Id: TQRootDialog.h,v 1.1 2006/04/11 16:33:46 rdm Exp $
// Author: Denis Bertini, M. Al-Turany  01/11/2000

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQRootDialog
#define ROOT_TQRootDialog

////////////////////////////////////////////////////////////////////////////
//
// TQRootDialog
//
// A TQRootDialog is used to prompt for the arguments of an object's
// member function. It is called by the TQCanvasMenu class
// @see TQCanvasMenu, QList, QVBox, TMethod, TCanvas
// @authors Denis Bertini <d.bertini@gsi.de>
//	   M. AL-Turany  <m.al-turany@gsi.de>
////////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "qvbox.h"
#include "qlist.h"
#include "qlineedit.h"
#endif

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TMethod;
class TCanvas;

class QLineEdit;
class QVBox;
class QWidget;
#ifdef __CINT__
class QList<QLineEdit>;
#endif

class TQRootDialog: public QVBox
{
#ifndef __CINT__
   Q_OBJECT
#endif
public:
   TQRootDialog(QWidget *parent, const char *name, WFlags f=0,
                TObject *obj=0,TMethod *meth=0);
   virtual ~TQRootDialog();
   void Add(const char* argname, const char* value, const char* type);
   void Popup();
   void SetTCanvas(TCanvas* aCanvas){fCurCanvas=aCanvas;}

public slots:
   void   Receive(){ExecuteMethod();} // to be replaced
   void   ExecuteMethod();

protected:
   void closeEvent( QCloseEvent* ce);
   QVBox *fArgBox;         // Box layout
   QLineEdit *fLineEdit;   // LineEdit widget for arguments
   TObject *fCurObj;       // Selected object
   TMethod *fCurMethod;    // method to be executed
   TCanvas* fCurCanvas;    // current canvas
   QWidget* fParent;       // parent widget
   QList<QLineEdit> fList; // list of widget corresponding to the number of arguments
      
   ClassDef(TQRootDialog,1)  //prompt for the arguments of an object's member function
};

#endif
