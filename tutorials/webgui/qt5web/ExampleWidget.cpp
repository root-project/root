#include "ExampleWidget.h"

#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TMath.h"

#include <ROOT/RCanvas.hxx>
#include <ROOT/TObjectDrawable.hxx>

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

   gPad = fxTCanvasWidget->getCanvas();
   fHisto->Draw();

   static constexpr int nth2points = 40;
   fHisto2 = std::make_shared<TH2I>("gaus2", "Example of TH1", nth2points, -5, 5, nth2points, -5, 5);
   fHisto2->SetDirectory(nullptr);
   for (int n=0;n<nth2points;++n) {
      for (int k=0;k<nth2points;++k) {
         double x = 10.*n/nth2points-5.;
         double y = 10.*k/nth2points-5.;
         fHisto2->SetBinContent(fHisto2->GetBin(n+1, k+1), (int) (1000*TMath::Gaus(x)*TMath::Gaus(y)));
      }
   }

   fxRCanvasWidget->getCanvas()->Draw<ROOT::Experimental::TObjectDrawable>(fHisto2, "colz");
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
