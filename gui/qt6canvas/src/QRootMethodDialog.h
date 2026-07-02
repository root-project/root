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


#ifndef QRootMethodDialog_h
#define QRootMethodDialog_h

#include <QtCore/QVector>
#include <QDialog>

class QLineEdit;
class QVBoxLayout;
class TObject;
class TFunction;
class TContextMenu;

class QRootMethodDialog: public QDialog {
   Q_OBJECT

   public:
      QRootMethodDialog();

      void addArg(const char *argname, const char *value, const char *type);

      QString getArg(int n);

      void methodDialog(TContextMenu *menu, TObject *object, TFunction* func);

   signals:
      void MenuCommandExecuted(TObject *obj, const char *method_name);

   protected:
      QVBoxLayout *argLayout{nullptr};

      QVector<QLineEdit*> fArgs;
};

#endif
