// @(#)root/qtgsi:$Id$
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
// @see TQCanvasMenu, QList, QVBoxLayout, TMethod, TCanvas
// @authors Denis Bertini <d.bertini@gsi.de>
//    M. AL-Turany  <m.al-turany@gsi.de>
////////////////////////////////////////////////////////////////////////////

#include "TQtGSIIncludes.h"
#include "TObject.h"

class TMethod;
class TCanvas;

class TQRootDialog : public QWidget
{
#ifndef __CINT__
   Q_OBJECT
#endif
private:
   TQRootDialog(const TQRootDialog &);
   TQRootDialog& operator=(const TQRootDialog &);

public:
   TQRootDialog(QWidget *parent, const QString& title,
                TObject *obj = nullptr, TMethod *method = nullptr);
   virtual ~TQRootDialog();
   void Add(const char* argname, const char* value, const char* type);
   void Popup();
   void SetTCanvas(TCanvas* aCanvas){fCurCanvas=aCanvas;}

public slots:
   void   Receive(){ExecuteMethod();} // to be replaced
   void   ExecuteMethod();

protected:
   void closeEvent( QCloseEvent* ce);
   QLineEdit *fLineEdit;   // LineEdit widget for arguments
   TObject *fCurObj;       // Selected object
   TMethod *fCurMethod;    // method to be executed
   TCanvas* fCurCanvas;    // current canvas
   QWidget* fParent;       // parent widget
   QList<QLineEdit*> fList; // list of widget corresponding to the number of arguments
   ClassDef(TQRootDialog,1)  //prompt for the arguments of an object's member function
};

#endif
