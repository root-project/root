// Author: Sergey Linev, GSI  26/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_QCanvasWidget
#define ROOT_QCanvasWidget

#include <QWidget>
#include <QAction>
#include "ui_QCanvasWidget.h"

#include <string>
#include <vector>

class TH1F;
class TH2I;

class QCanvasWidget : public QWidget, public Ui::QCanvasWidget {
   Q_OBJECT

   QAction *fViewToolbar = nullptr;
   QAction *fViewEventStatus = nullptr;
   QAction *fViewToolTip = nullptr;

   QString fLastFileDir;

protected:

   std::vector<std::string> GetSupportedFileFormats();

   void FillSaveMenu(QMenu *menu);

public:
   QCanvasWidget(QWidget *parent = nullptr, const char *name = nullptr);

   virtual ~QCanvasWidget();

   QPaintWidget *GetPaintWidget() const { return fPaintWidget; }

   void SaveCanvas(const QString &fname);

   void ApplyCanvasStatusBits();

public slots:

   void NewCanvas();
   void OpenRootFile();
   void CloseCanvas();

   void SaveCanvasAs();
   void PrintCanvas();
   void QuitRoot();

   void ShowColors();
   void ShowMarkers();
   void IconifyCanvas();

   void SetViewToolbar(bool);
   void SetViewEventStatus(bool);
   void SetViewToolTip(bool);

   void CanvasStatusEventSlot(const QString &);
};

#endif
