// Author: Sergey Linev, GSI  26/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_QCanvasWidget_h
#define ROOT_QCanvasWidget_h

#include <QWidget>
#include <QMenuBar>
#include "ui_QCanvasWidget.h"

#include <string>
#include <vector>

class TH1F;
class TH2I;

class QCanvasWidget : public QWidget, public Ui::QCanvasWidget {
   Q_OBJECT

   QMenuBar *fMenuBar = nullptr;

   QString fLastFileDir;

protected:

   std::vector<std::string> GetSupportedFileFormats();

   void FillSaveMenu(QMenu *menu);

public:
   QCanvasWidget(QWidget *parent = nullptr, const char *name = nullptr);

   virtual ~QCanvasWidget();

   QPaintWidget *GetPaintWidget() const { return fPaintWidget; }

   void SaveCanvas(const QString &fname);

public slots:

   void NewCanvas();
   void OpenRootFile();
   void CloseCanvas();

   void SaveCanvasAs();
   void PrintCanvas();
   void QuitRoot();
};

#endif
