#include "mainwidget.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QApplication>

#include "TQt6Canvas.h"
#include "TCanvas.h"
#include "TRandom.h"
#include "TMath.h"
#include "TLatex.h"
#include "TH1I.h"
#include "TH2I.h"
#include "TGraph.h"

using namespace ROOT::Experimental;

MainWidget::MainWidget(QWidget *parent) : QWidget(parent)
{
   auto *mainLayout = new QHBoxLayout(this);

   // left side - buttons
   auto *leftPanel = new QWidget(this);
   leftPanel->setFixedWidth(160);
   auto *leftLayout = new QVBoxLayout(leftPanel);
   leftLayout->setContentsMargins(0, 0, 0, 0);

   const char *btns[8] = {"TLatex", "TH1", "TH2", "TGraph", "Toggle menu", "Toggle toobar", "Toggle statusbar", "Exit"};

   for (int i = 0; i < 8; ++i) {
      auto *btn = new QPushButton(tr("%1").arg(btns[i]), leftPanel);
      btn->setMinimumHeight(25);
      leftLayout->addWidget(btn);
      connect(btn, &QPushButton::clicked, this, [this, i] {
         button_clicked(i);
         // emit buttonClicked(i);
      });
      if (i == 6) {
         auto box1 = new QCheckBox("Update", leftPanel);
         leftLayout->addWidget(box1);
         connect(box1, &QCheckBox::toggled, this, [this](bool on) { toggle_timer(on); });
         leftLayout->addStretch(); // keep buttons at the top
      }
   }

   // right side - placeholder for canvas
   m_placeholder = new QWidget(this);
   m_placeholderLayout = new QVBoxLayout(m_placeholder);
   m_placeholderLayout->setContentsMargins(0, 0, 0, 0);

   mainLayout->addWidget(leftPanel);
   mainLayout->addWidget(m_placeholder, 1); // stretch factor 1: takes remaining space

   m_canvas = TQt6Canvas::EmbedQt6Canvas(m_placeholder, "qt6canv");

   button_clicked(0);

   // configure update timer
   QObject::connect(&m_timer, &QTimer::timeout, [this]() { m_counter++; fill_canvas_object(); });
   m_timer.setSingleShot(false);
}

void MainWidget::toggle_timer(bool on)
{
   if (on)
      m_timer.start(200);
   else
      m_timer.stop();
}

void MainWidget::fill_canvas_object()
{
   if (!m_canvas)
      return;

   TObject *obj = nullptr;
   TIter iter(m_canvas->GetListOfPrimitives());
   while (obj = iter()) {
      // skip TFrame object
      if (!obj->InheritsFrom("TFrame"))
         break;
   }

   switch (m_clicked_button) {
   case 0:
      if (auto ltx = dynamic_cast<TLatex *>(obj))
         if (m_counter > 0) {
            ltx->SetTextAlign(12);
            ltx->SetX(0.2);
            ltx->SetTitle(TString::Format("TLatex : #color[%d]{%d}", 2 + m_counter % 7, m_counter));
            ltx->SetY(0.5 + 0.2*TMath::Sin(m_counter * 0.15));
         }
      break;
   case 1:
      if (auto h1 = dynamic_cast<TH1 *>(obj)) {
         h1->Reset();
         auto meanx = TMath::Sin(m_counter * 0.4);
         for (int n = 0; n < 300; ++n) {
            auto x = gRandom->Gaus(meanx);
            h1->Fill(x);
         }
         if (m_counter > 0)
            h1->SetTitle(TString::Format("TH1 example cnt: %d", m_counter));
      }
      break;
   case 2:
      if (auto h2 = dynamic_cast<TH2 *>(obj)) {
         h2->Reset();
         auto meanx = TMath::Sin(m_counter * 0.15);
         auto meany = -TMath::Sin(m_counter * 0.37);
         for (int n = 0; n < 300; ++n) {
            auto x = gRandom->Gaus(meanx);
            auto y = gRandom->Gaus(meany);
            h2->Fill(x, y);
         }
         if (m_counter > 0)
            h2->SetTitle(TString::Format("TH2 example cnt: %d", m_counter));
      }
      break;
   case 3:
      if (auto gr = dynamic_cast<TGraph *>(obj)) {
         for (int i = 0; i < 100; ++i)
            gr->SetPoint(i, i + 10, TMath::Sin((i + m_counter) * 0.2));
         if (m_counter > 0)
            gr->SetTitle(TString::Format("TGraph example cnt: %d", m_counter));
      }
      break;
   }
   m_canvas->Update();
}

void MainWidget::button_clicked(int index)
{
   if (!m_canvas)
      return;

   if (index == 4) {
      // there is no toggle method in TCanvas class, use TCanvasImp
      Bool_t flag = !m_canvas->TestBit(TCanvas::kMenuBar);
      m_canvas->SetBit(TCanvas::kMenuBar, flag);
      m_canvas->GetCanvasImp()->ShowMenuBar(flag);
      return;
   } else if (index == 5) {
      m_canvas->ToggleToolBar();
      return;
   } else if (index == 6) {
      m_canvas->ToggleEventStatus();
      return;
   }

   m_clicked_button = index;
   m_counter = 0;

   m_canvas->Clear();
   m_canvas->Resize();

   switch (index) {
   case 0: {
      auto ltx = new TLatex(0.5, 0.5, "TLatex");
      ltx->SetTextAlign(22);
      ltx->SetTextSize(0.2);
      ltx->SetBit(kCanDelete);
      m_canvas->Range(0, 0, 1, 1); // set default ranges
      m_canvas->Add(ltx);
      break;
   }
   case 1: {
      auto h1 = new TH1I("h1", "Example of TH1I", 100, -5, 5);
      h1->SetDirectory(nullptr);
      h1->SetBit(kCanDelete);
      m_canvas->Add(h1);
      break;
   }
   case 2: {
      auto h2 = new TH2I("h2", "Example of TH2I", 40, -5, 5, 40, -5, 5);
      h2->SetDirectory(nullptr);
      h2->SetBit(kCanDelete);
      m_canvas->Add(h2, "colz");
      break;
   }
   case 3: {
      auto gr = new TGraph(100);
      gr->SetName("gr1");
      gr->SetTitle("Example of TGraph");
      gr->SetLineColor(kGreen);
      gr->SetLineWidth(2);
      gr->SetBit(kCanDelete);
      m_canvas->Add(gr, "al");
      break;
   }
   default: QApplication::quit(); return;
   }

   fill_canvas_object(); // also canvas->Update() will be called
}
