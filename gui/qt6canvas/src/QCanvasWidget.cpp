// Author: Sergey Linev, GSI  26/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "QCanvasWidget.h"

#include "TCanvas.h"
#include "TROOT.h"
#include "TMarker.h"
#include "TApplication.h"
#include "TTimer.h"
#include "TFile.h"
#include "TError.h"
#include "TColorWheel.h"

#include <QMessageBox>
#include <QPushButton>
#include <QFileDialog>


enum { kFileNewCanvas, kFileOpen, kFileSaveAs, kFilePrint, kOptionInterrupt, kOptionRefresh,
       kInspectRoot, kToolsBrowser, kToolModify, kToolArc, kToolLine, kToolArrow, kToolDiamond,
       kToolEllipse, kToolPad, kToolPave, kToolPLabel, kToolPText, kToolPsText, kToolGraph,
       kToolCurlyLine, kToolCurlyArc, kToolLatex, kToolMarker, kToolCutG };

struct QToolBarData_t {
   const char *fPixmap;
   const char *fTipText;
   Bool_t      fStayDown;
   Int_t       fId;
   void       *fButton;
};


static QToolBarData_t qToolBarData[] = {
   // { filename,      tooltip,            staydown,  id,              button}
   { "newcanvas.xpm",  "New",              kFALSE,    kFileNewCanvas,  0 },
   { "open.xpm",       "Open",             kFALSE,    kFileOpen,       0 },
   { "save.xpm",       "Save As",          kFALSE,    kFileSaveAs,     0 },
   { "printer.xpm",    "Print",            kFALSE,    kFilePrint,      0 },
   { "",               "",                 kFALSE,    -1,              0 },
   { "interrupt.xpm",  "Interrupt",        kFALSE,    kOptionInterrupt,0 },
   { "refresh2.xpm",   "Refresh",          kFALSE,    kOptionRefresh,  0 },
   { "",               "",                 kFALSE,    -1,              0 },
   { "inspect.xpm",    "Inspect",          kFALSE,    kInspectRoot,    0 },
//   { "browser.xpm",    "Browser",          kFALSE,    kToolsBrowser,   0 },
   { "",                "",                kFALSE,    -1,              0 },
   { "pointer.xpm",    "Modify",           kFALSE,    kToolModify,     0 },
   { "arc.xpm",        "Arc",              kFALSE,    kToolArc,        0 },
   { "line.xpm",       "Line",             kFALSE,    kToolLine,       0 },
   { "arrow.xpm",      "Arrow",            kFALSE,    kToolArrow,      0 },
   { "diamond.xpm",    "Diamond",          kFALSE,    kToolDiamond,    0 },
   { "ellipse.xpm",    "Ellipse",          kFALSE,    kToolEllipse,    0 },
   { "pad.xpm",        "Pad",              kFALSE,    kToolPad,        0 },
   { "pave.xpm",       "Pave",             kFALSE,    kToolPave,       0 },
   { "pavelabel.xpm",  "Pave Label",       kFALSE,    kToolPLabel,     0 },
   { "pavetext.xpm",   "Pave Text",        kFALSE,    kToolPText,      0 },
   { "pavestext.xpm",  "Paves Text",       kFALSE,    kToolPsText,     0 },
   { "graph.xpm",      "Graph",            kFALSE,    kToolGraph,      0 },
   { "curlyline.xpm",  "Curly Line",       kFALSE,    kToolCurlyLine,  0 },
   { "curlyarc.xpm",   "Curly Arc",        kFALSE,    kToolCurlyArc,   0 },
   { "latex.xpm",      "Text/Latex",       kFALSE,    kToolLatex,      0 },
   { "marker.xpm",     "Marker",           kFALSE,    kToolMarker,     0 },
   { "cut.xpm",        "Graphical Cut",    kFALSE,    kToolCutG,       0 },
   { 0,                0,                  kFALSE,    0,               0 }
};



/** \class QCanvasWidget
    \ingroup qt6canvas

Qt widget which display canvas and provides menu, toolbar and status bar
Actual graphics shown in the \ref QPaintWidget
*/


////////////////////////////////////////////////////////////////////////////////
/// constructor

QCanvasWidget::QCanvasWidget(QWidget *parent, const char *name) : QWidget(parent)
{
   setupUi(this);

   setAttribute(Qt::WA_DeleteOnClose);

   setObjectName(name);

   connect(fPaintWidget, &QPaintWidget::CanvasStatusEvent, this, &QCanvasWidget::CanvasStatusEventSlot);

   // fMenuBar = new QMenuBar(fMenuFrame);
   fMenuBar->setMinimumWidth(50);
   fMenuBar->setNativeMenuBar(kFALSE); // disable putting this to screen menu. for MAC style WMs

   auto fileMenu = fMenuBar->addMenu("F&ile");
   fileMenu->addAction("&New canvas", this, &QCanvasWidget::NewCanvas);
   fileMenu->addAction("Open ...", this, &QCanvasWidget::OpenRootFile);
   fileMenu->addAction("Cl&ose canvas", this, &QCanvasWidget::CloseCanvas);

   fileMenu->addSeparator();

   auto saveMenu = fileMenu->addMenu("&Save");
   FillSaveMenu(saveMenu);
   connect(saveMenu, &QMenu::aboutToShow, this, [this, saveMenu]() {
      saveMenu->clear();
      FillSaveMenu(saveMenu);
   });

   fileMenu->addAction("Save &As...", this, &QCanvasWidget::SaveCanvasAs);

   fileMenu->addSeparator();

   fileMenu->addAction("Print...", this, &QCanvasWidget::PrintCanvas);
   fileMenu->addSeparator();

   fileMenu->addAction("Quit ROOT", this, &QCanvasWidget::QuitRoot);


   fMenuBar->addMenu("&Edit");

   auto viewMenu = fMenuBar->addMenu("&View");
   auto act = viewMenu->addAction("&Editor");
   act->setEnabled(false);

   fViewToolbar = viewMenu->addAction("&Toolbar");
   fViewToolbar->setCheckable(true);
   connect(fViewToolbar, &QAction::toggled, this, &QCanvasWidget::SetViewToolbar);
   SetViewToolbar(false);

   fViewEventStatus = viewMenu->addAction("Event &Statusbar");
   fViewEventStatus->setCheckable(true);
   connect(fViewEventStatus, &QAction::toggled, this, &QCanvasWidget::SetViewEventStatus);
   SetViewEventStatus(false);

   fViewToolTip = viewMenu->addAction("T&ooltip Info");
   fViewToolTip->setCheckable(true);
   connect(fViewToolTip, &QAction::toggled, this, &QCanvasWidget::SetViewToolTip);
   SetViewToolTip(false);

   viewMenu->addSeparator();

   viewMenu->addAction("&Colors", this, &QCanvasWidget::ShowColors);

   act = viewMenu->addAction("&Fonts");
   act->setEnabled(false);

   viewMenu->addAction("&Markers", this, &QCanvasWidget::ShowMarkers);

   viewMenu->addSeparator();
   viewMenu->addAction("&Iconify", this, &QCanvasWidget::IconifyCanvas);


   fMenuBar->addMenu("&Options");

   fMenuBar->addMenu("&Tools");

   auto spacer = new QWidget(this);
   spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
   fMenuBar->setCornerWidget(spacer, Qt::TopRightCorner);

   fMenuBar->addMenu("&Help");

   // configure tool bar

   //fToolBar->setStyleSheet("QToolBar { layout-spacing: 0px; }"
   //                        "QToolBar QToolButton { padding: 0px; margin: 0px; border: none; }");
   fToolBar->setIconSize(QSize(16, 16));
   for (int i = 0; qToolBarData[i].fPixmap; i++) {
      auto btn = &(qToolBarData[i]);
      if (strlen(btn->fPixmap) == 0) {
         fToolBar->addSeparator();
         continue;
      }

      TString iconname = TROOT::GetIconPath();
      iconname.Append("/");
      iconname.Append(btn->fPixmap);

      QIcon buttonIcon(iconname.Data());

      fToolBar->addAction(buttonIcon, btn->fTipText, [btn, this]() {
         auto canv = GetPaintWidget()->getCanvas();

         switch(btn->fId) {
            case kFileNewCanvas: NewCanvas(); break;
            case kFileOpen: OpenRootFile(); break;
            case kFileSaveAs: SaveCanvasAs(); break;
            case kFilePrint: PrintCanvas(); break;
            case kOptionInterrupt: gROOT->SetInterrupt(); break;
            case kOptionRefresh:
               if (canv) {
                  canv->Modified(kTRUE);
                  // trigger update of the paint widget, paint will be called
                  canv->Update();
               }
               break;
            case kInspectRoot:
               if (canv) {
                  canv->cd();
                  gROOT->Inspect();
                  canv->Update();
               }
               break;

            case kToolsBrowser: break;
            case kToolModify: gROOT->SetEditorMode(); break;
            case kToolArc:
               gROOT->SetEditorMode("Arc");
               break;
            case kToolLine:
               gROOT->SetEditorMode("Line");
               break;
            case kToolArrow:
               gROOT->SetEditorMode("Arrow");
               break;
            case kToolDiamond:
               gROOT->SetEditorMode("Diamond");
               break;
            case kToolEllipse:
               gROOT->SetEditorMode("Ellipse");
               break;
            case kToolPad:
               gROOT->SetEditorMode("Pad");
               break;
            case kToolPave:
               gROOT->SetEditorMode("Pave");
               break;
            case kToolPLabel:
               gROOT->SetEditorMode("PaveLabel");
               break;
            case kToolPText:
               gROOT->SetEditorMode("PaveText");
               break;
            case kToolPsText:
               gROOT->SetEditorMode("PavesText");
               break;
            case kToolGraph:
               gROOT->SetEditorMode("PolyLine");
               break;
            case kToolCurlyLine:
               gROOT->SetEditorMode("CurlyLine");
               break;
            case kToolCurlyArc:
               gROOT->SetEditorMode("CurlyArc");
               break;
            case kToolLatex:
               gROOT->SetEditorMode("Text");
               break;
            case kToolMarker:
               gROOT->SetEditorMode("Marker");
               break;
            case kToolCutG:
               gROOT->SetEditorMode("CutG");
               break;
         }
      });
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

QCanvasWidget::~QCanvasWidget()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create new canvas.

void QCanvasWidget::NewCanvas()
{
   gROOT->MakeDefCanvas();
}

////////////////////////////////////////////////////////////////////////////////
/// Open ROOT file dialog

void QCanvasWidget::OpenRootFile()
{
   QFileDialog fd( this,
                   "Select a ROOT file(s) to open them",
                   fLastFileDir,
                   QString("Root files (*.root);;Root xml files (*.xml);;All files (*.*)"));

   fd.setFileMode(QFileDialog::ExistingFiles);

   if (fd.exec() != QDialog::Accepted)
      return;

   QStringList list = fd.selectedFiles();
   for (auto &fileName : list) {
      fLastFileDir = QFileInfo(fileName).absolutePath();
      TFile::Open(fileName.toLatin1().constData(), "update");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Close the canvas.

void QCanvasWidget::CloseCanvas()
{
   close();
}

////////////////////////////////////////////////////////////////////////////////
/// Return vector with supported file formats with description and required option

std::vector<std::string> QCanvasWidget::GetSupportedFileFormats()
{
   return {
      "Post Script", "ps", "ps",
      "Post Script Portrait", "ps", "Portrait",
      "Post Script Landscape", "ps", "Landscape",
      "Encapsulated Post Script", "eps", "eps",
      "Encapsulated Post Script preview", "eps", "Preview",
      "GIF format", "gif", "gif",
      "PDF format", "pdf", "pdf",
      "SVG format", "svg", "svg",
      "XPM format", "xpm", "xpm",
      "PNG format", "png", "png",
      "JPG format", "jpg", "jpg",
      "BMP format", "bmp", "bmp",
      "TIFF format", "tiff", "tiff",
      "C++ Macro", "C", "cxx",
      "json file", "json", "json",
      "html file", "html", "html",
      "root file", "root", "root" };
}

////////////////////////////////////////////////////////////////////////////////
/// Fill save menu with predefined items

void QCanvasWidget::FillSaveMenu(QMenu *menu)
{
   auto canv = GetPaintWidget()->getCanvas();

   auto fmts = GetSupportedFileFormats();

   for (std::size_t pos = 0; pos < fmts.size(); pos += 3) {
      auto ext = fmts[pos + 1];
      if ((ext != "C") && (ext != fmts[pos + 2]))
         continue;

      QString filename = canv ? canv->GetName() : "c1";
      filename += ".";
      filename += ext.c_str();
      auto act1 = menu->addAction(filename);
      connect(act1, &QAction::triggered, this, [this, filename]() {
         SaveCanvas(filename);
      });
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save canvas as file

void QCanvasWidget::SaveCanvas(const QString &fname)
{
   auto canv = GetPaintWidget()->getCanvas();
   if (canv)
      canv->SaveAs(fname.toLatin1().constData());
}

////////////////////////////////////////////////////////////////////////////////
/// Save canvas as

void QCanvasWidget::SaveCanvasAs()
{
   auto canv = GetPaintWidget()->getCanvas();
   if (!canv)
      return;

   auto fmts = GetSupportedFileFormats();

   QFileDialog fd( this, QString("Save ") + canv->GetName() + " As", fLastFileDir);
   fd.setFileMode( QFileDialog::AnyFile );
   fd.setAcceptMode(QFileDialog::AcceptSave);

   QStringList flt;

   for (std::size_t pos = 0; pos < fmts.size(); pos += 3) {
      QString filter = QString(fmts[pos].c_str()) + " (*." + fmts[pos+1].c_str() + ")";
      flt << filter;
   }

   fd.setNameFilters(flt);

   QString filename0 = canv->GetName();

   fd.selectFile(filename0 + ".png");

   QObject::connect(&fd, &QFileDialog::filterSelected, [filename0, &fd](const QString &fltr) {
      QStringList flst = fd.selectedFiles();
      if (flst.size() > 1)
         return;
      if ((flst.size() == 1) && (flst[0].indexOf(filename0 + ".") != 0))
         return;

      auto p = fltr.indexOf("(*."), p2 = fltr.lastIndexOf(")");
      fd.selectFile(filename0 + fltr.mid(p + 2, p2 - p - 2));
   });

   if (fd.exec() != QDialog::Accepted)
      return;

   QStringList flst = fd.selectedFiles();
   if (flst.isEmpty()) return;

   QString filename = flst[0];

   fLastFileDir = fd.directory().path();

   std::string opt;

   for (std::size_t pos = 0; pos < fmts.size(); pos += 3) {
      QString filter = QString(fmts[pos].c_str()) + " (*." + fmts[pos+1].c_str() + ")";
      if (filter != fd.selectedNameFilter())
         continue;

      if (!filename.endsWith(QString(".") + fmts[pos+1].c_str())) {
         filename.append(".");
         filename.append(fmts[pos+1].c_str());
      }
      opt = fmts[pos + 2];
      break;
   }

   if (!opt.empty())
      canv->Print(filename.toLatin1().constData(), opt.c_str());
}

////////////////////////////////////////////////////////////////////////////////
/// Print the canvas.

void QCanvasWidget::PrintCanvas()
{
   ::Info("QCanvasWidget::PrintCanvas", "To be implemented");
}

////////////////////////////////////////////////////////////////////////////////
/// Quit ROOT

void QCanvasWidget::QuitRoot()
{
   // set flag which sometimes checked in TSystem::ProcessEvents
   gROOT->SetInterrupt(kTRUE);

   if (gApplication)
      TTimer::SingleShot(100, "TApplication",  gApplication, "Terminate()");
}

////////////////////////////////////////////////////////////////////////////////
/// Set visibility of tool bar

void QCanvasWidget::SetViewToolbar(bool on)
{
   fToolBar->setVisible(on);
   fViewToolbar->setChecked(on);
}

////////////////////////////////////////////////////////////////////////////////
/// Set visibility of event status

void QCanvasWidget::SetViewEventStatus(bool on)
{
   fStatusBar->setVisible(on);
   fViewEventStatus->setChecked(on);
}

////////////////////////////////////////////////////////////////////////////////
/// Status event handler

void QCanvasWidget::CanvasStatusEventSlot(const QString &msg)
{
   if (fStatusBar && fStatusBar->isVisible())
      fStatusBar->showMessage(msg);
}

////////////////////////////////////////////////////////////////////////////////
/// Set visibility of tooltip

void QCanvasWidget::SetViewToolTip(bool on)
{
   fViewToolTip->setChecked(on);

   fPaintWidget->SetShowToolTip(on);
}

////////////////////////////////////////////////////////////////////////////////
/// Show new canvas with ROOT colors wheel

void QCanvasWidget::ShowColors()
{
   TVirtualPad::TContext ctxt;

   auto wheel = new TColorWheel();
   wheel->Draw();
   gPad->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Show new canvas with ROOT markers

void QCanvasWidget::ShowMarkers()
{
   TVirtualPad::TContext ctxt;

   auto m = new TCanvas("markers","Marker Types",600,200);
   TMarker::DisplayMarkerTypes();
   m->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Iconify canvas widget

void QCanvasWidget::IconifyCanvas()
{
   showMinimized();
}

////////////////////////////////////////////////////////////////////////////////
/// Apply canvas status bits like show tooltip or menu bar

void QCanvasWidget::ApplyCanvasStatusBits()
{
   auto canv = GetPaintWidget()->getCanvas();
   if (!canv)
      return;

   SetViewEventStatus(canv->TestBit(TCanvas::kShowEventStatus));

   // canv->TestBit(TCanvas::kShowEditor);

   SetViewToolbar(canv->TestBit(TCanvas::kShowToolBar));

   SetViewToolTip(canv->TestBit(TCanvas::kShowToolTips));

   fMenuBar->setVisible(canv->TestBit(TCanvas::kMenuBar));


}
