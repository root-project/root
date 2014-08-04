// @(#)root/qtgsi:$Id$
// Author: Denis Bertini, M. Al-Turany  01/11/2000

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdlib.h>
#include "qevent.h"
#include "qdialog.h"
#include "qpushbutton.h"
#include "qlabel.h"
#include "qobject.h"
#include "qlineedit.h"
#if (QT_VERSION > 0x039999) // Added by cholm@nbi.dk - for Qt 4
# include "q3hbox.h"
typedef Q3HBox QHBox;
#endif

#include "TQRootDialog.h"
#include "TMethod.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TClass.h"
#include "TObjString.h"

using namespace Qt;

ClassImp(TQRootDialog)

//______________________________________________________________________________
TQRootDialog::TQRootDialog(QWidget *wparent, const char *wname, WFlags f,
                         TObject* obj, TMethod *method ) :
   QVBox(wparent,wname, f | WType_Modal | WStyle_Dialog   ),
   fLineEdit(0),
   fParent(wparent)
{
   // ctor
   fCurObj=obj;
   fCurMethod=method;
   setSizePolicy(QSizePolicy(QSizePolicy::Expanding,
                            QSizePolicy::Expanding));
   fArgBox = new QVBox(this, "args");
   fArgBox->setSizePolicy(QSizePolicy(QSizePolicy::Expanding,
                 QSizePolicy::Expanding));
   QHBox *hbox = new QHBox(this,"buttons");
   QPushButton *bOk = new QPushButton("Apply",hbox,"Apply");
   QPushButton *bCancel = new QPushButton("Cancel",hbox,"Close");
   connect(bCancel,SIGNAL (clicked()), this, SLOT(close()));
   connect(bOk,SIGNAL( clicked() ), this, SLOT( ExecuteMethod() ));
}

//______________________________________________________________________________
void TQRootDialog::ExecuteMethod()
{
   // Execute ROOT methods.

   Bool_t deletion = kFALSE;
   TVirtualPad *psave =  gROOT->GetSelectedPad();

   //if (fCurObj)
   TObjArray tobjlist(fCurMethod->GetListOfMethodArgs()->LastIndex()+1);
#if (QT_VERSION > 0x039999) // Added by cholm@nbi.dk - for Qt 4
   typedef QList<QLineEdit*>::iterator iter;
   for (iter st = fList.begin(); st != fList.end(); ++st) {
     QString s = (*st)->text();
      TObjString *t = new TObjString( (const char*) s );
      tobjlist.AddLast((TObject*) t) ;
   }
#else
   for ( QLineEdit* st = fList.first(); st; st = fList.next()) {
      QString s = st->text();
      TObjString *t = new TObjString( (const char*) s );
      tobjlist.AddLast((TObject*) t) ;
   }
#endif
   // handle command if existing object
   if ( fCurObj ) {
      if( strcmp(fCurMethod->GetName(),"Delete") == 0  ) {
         if (fCurObj) {
            delete fCurObj;
            fCurObj=0;
            deletion = kTRUE;
         }
      }
      else if (  strcmp(fCurMethod->GetName(),"SetCanvasSize") == 0 ) {
         int value[2] = {0,0};
         int l=0;
#if (QT_VERSION > 0x039999) // Added by cholm@nbi.dk - for Qt 4
         for (iter st = fList.begin(); st != fList.end(); ++st) {
            QString s = (*st)->text();
            value[l++] = atoi ( s );
         }
#else
         for ( QLineEdit* st = fList.first(); st; st = fList.next()) {
            QString s = st->text();
            value[l++] = atoi ( s );
         }
#endif
         fParent->resize(value[0], value[1]);
      }
      else {
         // here call cint call
         fCurObj->Execute( fCurMethod, &tobjlist);
      }
   } // ! fCurrent Obj

   if (!deletion ) {
      gROOT->SetSelectedPad(psave);
      gROOT->GetSelectedPad()->Modified();
      gROOT->GetSelectedPad()->Update();
   }
   else {
      gROOT->SetSelectedPad( gPad );
      gROOT->GetSelectedPad()->Update();
   }
}

//______________________________________________________________________________
TQRootDialog::~TQRootDialog()
{
   // dtor

   if (fArgBox) delete fArgBox;
   if (fLineEdit) delete fLineEdit;
#if (QT_VERSION > 0x039999) // Added by cholm@nbi.dk - for Qt 4
   // Perhaps we need to deallocate all?
   fList.erase(fList.begin(),fList.end());
#else
   fList.remove();
#endif
}

//______________________________________________________________________________
void TQRootDialog::Add(const char* argname, const char* value, const char* /*type*/)
{
   // Add widgets for arguments.

   QString s;
   s = value;
   new QLabel(argname,fArgBox);
   QLineEdit* lineEdit = new  QLineEdit(fArgBox);
   if (fLineEdit) {
      fLineEdit->setGeometry(10,10, 130, 30);
      fLineEdit->setFocus();
      fLineEdit->setText(s);
   }
   fList.append( lineEdit );
}

//______________________________________________________________________________
void TQRootDialog::Popup()
{
   // Show the dialog.

   show();
}

//______________________________________________________________________________
void TQRootDialog::closeEvent( QCloseEvent* ce )
{
   // Handle close event.

   ce->accept();
}
