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
// @see TQCanvasMenu, QList, QVBox, TMethod, TCanvas
// @authors Denis Bertini <d.bertini@gsi.de>
//    M. AL-Turany  <m.al-turany@gsi.de>
////////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
# include "qlineedit.h"
# include "qnamespace.h"
# if (QT_VERSION > 0x039999) // Added by cholm@nbi.dk - for Qt 4
#  include "qlist.h"
#  include "q3vbox.h"
typedef Q3VBox QVBox;
# else
#  include "qvbox.h"
# endif
#endif

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TMethod;
class TCanvas;

class QCloseEvent;
class QLineEdit;
class QWidget;
#ifdef __CINT__
template <typename T> class QList;
class QLineEdit;
class QList<QLineEdit*>;
class QVBox;
#if QTVERS > 3
class WindowFlags;
typedef WindowFlags Qt::WFlags;
#endif
#endif

class TQRootDialog: public QVBox
{
#ifndef __CINT__
   Q_OBJECT
#endif
private:
   TQRootDialog(const TQRootDialog &);
   TQRootDialog& operator=(const TQRootDialog &);

public:
   TQRootDialog(QWidget *parent, const char *name, Qt::WFlags f=0,
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
# if (QT_VERSION > 0x039999) // Added by cholm@nbi.dk - for Qt 4
   QList<QLineEdit*> fList; // list of widget corresponding to the number of arguments
#else
   QList<QLineEdit> fList; // list of widget corresponding to the number of arguments
#endif
   ClassDef(TQRootDialog,1)  //prompt for the arguments of an object's member function
};

#endif
