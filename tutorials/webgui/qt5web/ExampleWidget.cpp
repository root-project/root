#include "ExampleWidget.h"

#include "TCanvas.h"
#include "TClass.h"
#include "TH1.h"
#include "TROOT.h"
#include "TWebCanvas.h"

#include <QMessageBox>

ExampleWidget::ExampleWidget(QWidget *parent, const char* name) :
   QWidget(parent)
{
   setupUi(this);

   setAttribute(Qt::WA_DeleteOnClose);

   setObjectName(name);

   // create sample histogram

   fHisto = new TH1F("h1","title", 100, -5, 5);
   fHisto->FillRandom("gaus", 10000);
   fHisto->SetDirectory(nullptr);

   gPad = fxCanvasWidget->getCanvas();
   fHisto->Draw();
}

ExampleWidget::~ExampleWidget()
{
}

void ExampleWidget::InfoButton_clicked()
{
   QMessageBox::information(this,"QtRoot standalone example","Demo how QRootCanvas can be inserted into the QWidget");
}

void ExampleWidget::ExitButton_clicked()
{
   // when widget closed, application automatically exit
   close();
}
