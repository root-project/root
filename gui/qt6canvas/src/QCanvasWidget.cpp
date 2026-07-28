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
#include "TApplication.h"
#include "TTimer.h"
#include "TFile.h"
#include "TError.h"

#include <QMessageBox>
#include <QPushButton>
#include <QFileDialog>

////////////////////////////////////////////////////////////////////////////////
/// constructor

QCanvasWidget::QCanvasWidget(QWidget *parent, const char *name) : QWidget(parent)
{
   setupUi(this);

   setAttribute(Qt::WA_DeleteOnClose);

   setObjectName(name);

   fPaintWidget->SetStatusBar(fStatusBar);

   fMenuBar = new QMenuBar(fMenuFrame);
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

   fMenuBar->addMenu("&View");

   fMenuBar->addMenu("&Options");

   fMenuBar->addMenu("&Tools");

   auto spacer = new QWidget(this);
   spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
   fMenuBar->setCornerWidget(spacer, Qt::TopRightCorner);

   fMenuBar->addMenu("&Help");

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


