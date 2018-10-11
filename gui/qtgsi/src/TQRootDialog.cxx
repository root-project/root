// @(#)root/qtgsi:$Id$
// Author: Denis Bertini, M. Al-Turany  01/11/2000

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TQRootDialog.h"

#include "TMethod.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TClass.h"
#include "TObjString.h"

#include <stdlib.h>

using namespace Qt;

ClassImp(TQRootDialog);

////////////////////////////////////////////////////////////////////////////////
/// ctor

TQRootDialog::TQRootDialog(QWidget *wparent, const QString& title, TObject* obj, TMethod *method)
   : QWidget(wparent)
   , fLineEdit(nullptr)
   , fCurObj(obj)
   , fCurMethod(method)
   , fCurCanvas(nullptr)
   , fParent(wparent)
{
   setObjectName(title);

   QPushButton *apply  = new QPushButton("Apply");
   QPushButton *cancel = new QPushButton("Cancel");
   QHBoxLayout *hbox   = new QHBoxLayout(fParent);

   hbox->addWidget(apply);
   hbox->addWidget(cancel);

   setLayout(hbox);

   connect(apply, SIGNAL(clicked()), fParent, SLOT(ExecuteMethod()));
   connect(cancel,SIGNAL(clicked()), fParent, SLOT(close()));
}

////////////////////////////////////////////////////////////////////////////////
/// dtor

TQRootDialog::~TQRootDialog()
{
   if (fLineEdit)
      delete fLineEdit;
   fList.erase(fList.begin(), fList.end());
}

////////////////////////////////////////////////////////////////////////////////
/// Execute ROOT methods.

void TQRootDialog::ExecuteMethod()
{
   Bool_t deletion = false;
   TVirtualPad *psave = gROOT->GetSelectedPad();

   TObjArray tobjlist(fCurMethod->GetListOfMethodArgs()->LastIndex()+1);

   for (auto str : fList)
      tobjlist.AddLast((TObject*) new TObjString(str->text().toAscii().data()));

   // handle command if existing object
   if (fCurObj) {
      if(strcmp(fCurMethod->GetName(),"Delete") == 0) {
            delete fCurObj;
            fCurObj = nullptr;
            deletion = true;
      } else if (strcmp(fCurMethod->GetName(),"SetCanvasSize") == 0 ) {
         int i = 0, value[2] = {0, 0};
         for (auto str : fList)
            value[i++] = atoi(str->text().toAscii().data());
         fParent->resize(value[0], value[1]);
      } else {
         // here call cint call
         fCurObj->Execute(fCurMethod, &tobjlist);
      }
   } // ! fCurrent Obj

   if (!deletion) {
      gROOT->SetSelectedPad(psave);
      gROOT->GetSelectedPad()->Modified();
      gROOT->GetSelectedPad()->Update();
   } else {
      gROOT->SetSelectedPad(gPad);
      gROOT->GetSelectedPad()->Update();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add widgets for arguments.

void TQRootDialog::Add(const char* argname, const char* value, const char* /*type*/)
{
   QString s;
   s = value;
   new QLabel(argname, this);
   QLineEdit* lineEdit = new  QLineEdit(this);
   if (fLineEdit) {
      fLineEdit->setGeometry(10,10, 130, 30);
      fLineEdit->setFocus();
      fLineEdit->setText(s);
   }
   fList.append(lineEdit);
}

////////////////////////////////////////////////////////////////////////////////
/// Show the dialog.

void TQRootDialog::Popup()
{
   show();
}

////////////////////////////////////////////////////////////////////////////////
/// Handle close event.

void TQRootDialog::closeEvent( QCloseEvent* ce )
{
   ce->accept();
}
