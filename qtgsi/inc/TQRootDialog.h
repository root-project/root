// @(#)root/qtgsi:$Name:$:$Id:$
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

#include "qvbox.h"
#include "qlist.h"
#ifndef ROOT_TCanvas
#include "TCanvas.h"
#endif

class QLineEdit;
class TMethod;


class TQRootDialog: public QVBox
{
   Q_OBJECT

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
   QList<QLineEdit> fList; // list of widget corresponding to the number of arguments
   TObject *fCurObj;       // Selected object
   TMethod *fCurMethod;    // method to be executed
   TCanvas* fCurCanvas;    // current canvas
   QWidget* fParent;       // parent widget
};

#endif
