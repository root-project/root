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

   // configure TCanvas output

   fCanvas = new TCanvas(kFALSE);
   fCanvas->SetName("Canvas");
   fCanvas->SetTitle("title");
   fCanvas->ResetBit(TCanvas::kShowEditor);
   fCanvas->SetCanvas(fCanvas);
   fCanvas->SetBatch(kTRUE); // mark canvas as batch

   TWebCanvas *web = new TWebCanvas(fCanvas, "title", 0, 0, 800, 600, kFALSE);

   // web->AddCustomClass("CustomClass");
   // web->SetCustomScripts("<custom JSROOT code to draw CustomClass>");

   fCanvas->SetCanvasImp(web);

   SetPrivateCanvasFields(fCanvas, true);

   web->SetCanCreateObjects(kFALSE); // not yet create objects on server side

   // web->SetUpdatedHandler([this]() { ProcessCanvasUpdated(); });
   // web->SetActivePadChangedHandler([this](TPad *pad){ ProcessActivePadChanged(pad); });
   // web->SetPadClickedHandler([this](TPad *pad, int x, int y) { ProcessPadClicked(pad, x, y); });
   // web->SetPadDblClickedHandler([this](TPad *pad, int x, int y) { ProcessPadDblClicked(pad, x, y); });

   ROOT::Experimental::RWebDisplayArgs args("qt5");
   args.SetDriverData(this); // it is parent widget for created QWebEngineView element
   args.SetUrlOpt("noopenui");
   web->ShowWebWindow(args);

   fView = findChild<QWebEngineView*>("RootWebView");
   if (!fView) {
      printf("FAIL TO FIND QWebEngineView - ROOT Qt5Web plugin does not work properly !!!!!\n");
      exit(11);
   }

   canvasGridLayout->addWidget(fView);

   // QObject::connect(fView, SIGNAL(drop(QDropEvent*)), this, SLOT(dropView(QDropEvent*)));

   fCanvas->SetCanvasSize(fView->width(), fView->height());

   gPad = fCanvas;
   fHisto->Draw();

//   fxQCanvas->setObjectName("example");
//   fxQCanvas->getCanvas()->SetName("example");
//   fxQCanvas->setEditorFrame(EditorFrame);
//   fxQCanvas->buildEditorWindow();

//   fxQCanvas->getCanvas()->cd();
//   fHisto->Draw("colz");
}

ExampleWidget::~ExampleWidget()
{
   if (fCanvas) {
      SetPrivateCanvasFields(fCanvas, false);
      gROOT->GetListOfCanvases()->Remove(fCanvas);
      fCanvas->Close();
      delete fCanvas;
      fCanvas = nullptr;
   }
}


void ExampleWidget::SetPrivateCanvasFields(TCanvas *canv, bool on_init)
{
   Long_t offset = TCanvas::Class()->GetDataMemberOffset("fCanvasID");
   if (offset > 0) {
      Int_t *id = (Int_t *)((char*) canv + offset);
      if (*id == canv->GetCanvasID()) *id = on_init ? 111222333 : -1;
   } else {
      printf("ERROR: Cannot modify fCanvasID data member\n");
   }

   offset = TCanvas::Class()->GetDataMemberOffset("fMother");
   if (offset > 0) {
      TPad **moth = (TPad **)((char*) canv + offset);
      if (*moth == canv->GetMother()) *moth = on_init ? canv : nullptr;
   } else {
      printf("ERROR: Cannot set fMother data member in canvas\n");
   }
}



void ExampleWidget::InfoButton_clicked()
{
   QMessageBox::information(this,"QtRoot standalone example","Demo how QRootCanvas can be inserted into the QWidget");
}

void ExampleWidget::resizeEvent(QResizeEvent * e)
{
   fCanvas->SetCanvasSize(fView->width(), fView->height());
}

void ExampleWidget::ExitButton_clicked()
{
   // when widget closed, application automatically exit
   close();
}


