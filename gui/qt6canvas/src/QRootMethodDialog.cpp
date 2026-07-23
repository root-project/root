// Original code derived from QRootDialog
// from https://go4.gsi.de project
// Author : Denis Bertini 01.11.2000

// Author: Sergey Linev, GSI  30/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "QRootMethodDialog.h"

#include <QGridLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>

#include "TString.h"
#include "TROOT.h"
#include "TVirtualPad.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TMethodCall.h"
#include "TObjString.h"
#include "TContextMenu.h"

#include <iostream>

QRootMethodDialog::QRootMethodDialog() : QDialog()
{
   QGridLayout *gridLayout = new QGridLayout(this);
   gridLayout->setSpacing(1);
   gridLayout->setContentsMargins(1,1,1,1);

   QHBoxLayout *buttLayout = new QHBoxLayout();

   QPushButton *bOk = new QPushButton(this);
   bOk->setText("Apply");
   QObject::connect(bOk, &QPushButton::clicked, this, &QRootMethodDialog::accept);
   buttLayout->addWidget(bOk);

   QPushButton *bCancel = new QPushButton(this);
   bCancel->setText("Cancel");
   QObject::connect(bCancel, &QPushButton::clicked, this, &QRootMethodDialog::reject);
   buttLayout->addWidget(bCancel);

   argLayout = new QVBoxLayout();

   setSizePolicy(QSizePolicy(QSizePolicy::Expanding,
                            QSizePolicy::Expanding));

   gridLayout->addLayout(argLayout, 0, 0);
   gridLayout->addLayout(buttLayout, 1, 0, Qt::AlignBottom);
}

void QRootMethodDialog::addArg(const char *argname, const char *value, const char *)
{
   QLabel* lbl = new QLabel(argname);
   argLayout->addWidget(lbl);

   QLineEdit* le = new QLineEdit();
   le->setGeometry(10,10, 130, 30);
   le->setFocus();
   le->setText(value);
   argLayout->addWidget(le);

   fArgs.push_back(le);
}

QString QRootMethodDialog::getArg(int n)
{
   if ((n<0) || (n>=fArgs.size())) return QString("");
   return fArgs[n]->text();
}


void QRootMethodDialog::methodDialog(TContextMenu *menu, TObject *object, TFunction* func)
{
   if (!menu || !object || !func)
      return;

   setWindowTitle(menu->CreateDialogTitle(object, func));

   // iterate through all arguments and create appropriate input-data objects:
   // inputlines, option menus...
   TIter next(func->GetListOfMethodArgs());

   while (auto argument = (TMethodArg *) next()) {
      TString argTitle = menu->CreateArgumentTitle(argument);
      TString type = argument->GetTypeName();
      TDataType *datatype = gROOT->GetType(type);
      TString basictype;

      if (datatype) {
         basictype = datatype->GetTypeName();
      } else {
         if (type.CompareTo("enum") != 0)
            std::cout << "*** Warning in Dialog(): data type is not basic type, assuming (int)\n";
         basictype = "int";
      }

      if (TString(argument->GetTitle()).Index("*") != kNPOS) {
         basictype += "*";
         type = "char*";
      }

      TDataMember *m = argument->GetDataMember();
      if (m && m->GetterMethod()) {

         m->GetterMethod()->Init(object->IsA(), m->GetterMethod()->GetMethodName(), "");

         // Get the current value and form it as a text:

         TString val;

         if (basictype == "char*") {
            char *tdefval = nullptr;
            m->GetterMethod()->Execute(object, "", &tdefval);
            if (tdefval) val = tdefval;
         } else
         if ((basictype == "float") ||
             (basictype == "double")) {
            Double_t ddefval = 0.;
            m->GetterMethod()->Execute(object, "", ddefval);
            val = TString::Format("%g", ddefval);
         } else
         if ((basictype == "char") ||
             (basictype == "int")  ||
             (basictype == "bool")  ||
             (basictype == "long") ||
             (basictype == "short")) {
            Long_t ldefval = 0;
            m->GetterMethod()->Execute(object, "", ldefval);
            val = TString::Format("%ld", ldefval);
         }

         // Find out whether we have options ...

         TList *opt;
         if ((opt = m->GetOptions()) != nullptr) {
            // should stop dialog
            // workaround JAM: do not stop dialog, use textfield (for time display toggle)
            addArg(argTitle.Data(), val.Data(), type.Data());
            //return;
         } else {
            // we haven't got options - textfield ...
            addArg(argTitle.Data(), val.Data(), type.Data());
         }
      } else {    // if m not found ...
         TString argDflt;
         if (argument->GetDefault())
            argDflt = argument->GetDefault();

         if ((argDflt.Length() > 1) &&
             (argDflt[0]=='\"') && (argDflt[argDflt.Length()-1]=='\"')) {
            // cut "" from the string argument
            argDflt.Remove(0,1);
            argDflt.Remove(argDflt.Length()-1,1);
         }

         addArg(argTitle.Data(), argDflt.Data(), type.Data());
      }
   }

   if (exec() != QDialog::Accepted)
      return;

   TObjArray tobjlist(func->GetListOfMethodArgs()->LastIndex() + 1);
   for (int n = 0; n <= func->GetListOfMethodArgs()->LastIndex(); n++) {
      QString s = getArg(n);
      tobjlist.AddLast(new TObjString(s.toLatin1().constData()));
   }

   menu->Execute(object, func, &tobjlist);
}
