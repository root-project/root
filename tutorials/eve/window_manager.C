// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Demonstrates usage of EVE window-manager.

#include "TEveWindow.h"
#include "TEveViewer.h"
#include "TEveManager.h"
#include "TEveBrowser.h"
#include "TEveGedEditor.h"
#include "TGLEmbeddedViewer.h"
#include "TCanvas.h"
#include "TGTab.h"

void window_manager()
{
   TEveManager::Create();

   TEveUtil::Macro("pointset.C");

   PackTest();
   DetailTest();
   TabsTest();

   gEve->GetBrowser()->GetTabRight()->SetTab(1);
   gDebug = 1;
}

void PackTest()
{
   TEveWindowSlot  *slot  = 0;
   TEveWindowFrame *frame = 0;
   TEveViewer *v = 0;

   slot = TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());
   TEveWindowPack* pack1 = slot->MakePack();
   pack1->SetShowTitleBar(kFALSE);
   pack1->SetHorizontal();

   // Embedded viewer.
   slot = pack1->NewSlot();
   v = new TEveViewer("BarViewer");
   v->SpawnGLEmbeddedViewer(gEve->GetEditor());
   slot->ReplaceWindow(v);
   v->SetElementName("Bar Embedded Viewer");

   gEve->GetViewers()->AddElement(v);
   v->AddScene(gEve->GetEventScene());

   slot = pack1->NewSlot();
   TEveWindowPack* pack2 = slot->MakePack();
   pack2->SetShowTitleBar(kFALSE);

   slot = pack2->NewSlot();
   slot->StartEmbedding();
   TCanvas* can = new TCanvas("Root Canvas");
   can->ToggleEditor();
   slot->StopEmbedding();

   // SA viewer.
   slot = pack2->NewSlot();
   v = new TEveViewer("FooViewer");
   v->SpawnGLViewer(gEve->GetEditor());
   slot->ReplaceWindow(v);
   gEve->GetViewers()->AddElement(v);
   v->AddScene(gEve->GetEventScene());
}


void DetailTest()
{
   TEveWindowSlot* slot =
      TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());
   pack1 = slot->MakePack();
   pack1->SetShowTitleBar(kFALSE);
   pack1->SetElementName("Detail");
   pack1->SetHorizontal();

   // left slot
   slot = pack1->NewSlot();
   frame = slot->MakeFrame();
   frame->SetElementName("Latex Frame");
   frame->SetShowTitleBar(kFALSE);
   TGCompositeFrame* cf = frame->GetGUICompositeFrame();
   TGCompositeFrame* hf = new TGVerticalFrame(cf);
   hf->SetCleanup(kLocalCleanup);
   cf->AddFrame(hf, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
   {
      TGVerticalFrame* guiFrame = new TGVerticalFrame(hf);
      hf->AddFrame(guiFrame, new TGLayoutHints(kLHintsExpandX));
      guiFrame->SetCleanup(kDeepCleanup);

      guiFrame->AddFrame(new TGLabel(guiFrame, "Press Button:"),
                         new TGLayoutHints(kLHintsLeft, 2, 2, 0, 0));
      TGTextButton *b = new TGTextButton(guiFrame, "TestButton");
      guiFrame->AddFrame(b, new TGLayoutHints(kLHintsExpandX));
      TRootEmbeddedCanvas* ec =
         new TRootEmbeddedCanvas("Embeddedcanvas", hf, 220);
      hf->AddFrame(ec, new TGLayoutHints(kLHintsExpandY|kLHintsExpandX));
      double fontsize = 0.07;
      double x = 0.02;
      double y = 1 -1*fontsize;
      TLatex* latex = new TLatex(x, y, "Legend:");
      latex->SetTextSize(fontsize);
      latex->Draw();
      y -= 2*fontsize;
      // legend
      latex->DrawLatex(x, y, "greek letter #Delta#eta_{out}");
      y -= fontsize;
      latex->DrawLatex(x, y, "#color[5]{+} marker");
      y -= fontsize;
      latex->DrawLatex(x, y, "#color[5]{+} marker");
      y -= fontsize;
      latex->DrawLatex(x, y, "#color[4]{+} marker");
      y -= fontsize;
      latex->DrawLatex(x, y, "#color[5]{#bullet} marker");
      y -= fontsize;
      latex->DrawLatex(x, y, "#color[4]{#bullet} marker some text");
      y -= fontsize;
      latex->DrawLatex(x, y, "#color[2]{#Box} square");
      y -= fontsize;
      latex->DrawLatex(x, y, "#color[5]{#Box} color");
   }

   cf->MapSubwindows();
   cf->Layout();
   cf->MapWindow();

   // viewer slot
   TEveWindowSlot* slot2 = pack1->NewSlotWithWeight(3);
   TEveViewer*  viewer = new TEveViewer("DetailView", "DetailView");
   TGLEmbeddedViewer*  embeddedViewer =  viewer->SpawnGLEmbeddedViewer();
   slot2->ReplaceWindow(viewer);
   gEve->GetViewers()->AddElement(viewer);
   viewer->AddScene(gEve->GetEventScene());
}

void TabsTest()
{
   TRandom r(0);
   TEveWindowSlot  *slot  = 0;
   TEveWindowFrame *frame = 0;
   TEveViewer *v = 0;

   slot = TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());
   TEveWindowTab* tab1 = slot->MakeTab();
   tab1->SetElementName("Tabs");
   tab1->SetShowTitleBar(kFALSE);

   // horizontal text views
   slot = tab1->NewSlot();
   TEveWindowPack* pack1 = slot->MakePack();
   for(int i = 0; i<4;++i)
   {
      Int_t weight = r.Uniform(3, 7);
      slot = pack1->NewSlotWithWeight(weight);
      frame = slot->MakeFrame();
      frame->SetElementName(Form("FrameInPack %d", i));
      TGCompositeFrame* cf = frame->GetGUICompositeFrame();
      TGTextView* text_view = new TGTextView(cf, 200, 400);
      cf->AddFrame(text_view, new TGLayoutHints(kLHintsLeft    |
                                                kLHintsExpandX |
                                                kLHintsExpandY));

      for(Int_t l =0; l<weight; l++)
      {
         text_view->AddLine(Form("slot[%d] add line %d here ", i, l));
      }
      text_view->Update();
      text_view->SetWidth(text_view->ReturnLongestLineWidth()+20);
      text_view->Layout();

      cf->MapSubwindows();
      cf->Layout();
      cf->MapWindow();
   }

   // canvas tab
   slot = tab1->NewSlot();
   frame = slot->MakeFrame(new TRootEmbeddedCanvas());
   frame->SetElementName("Embedded Canvas");

   // neseted 2nd leveltabs
   slot = tab1->NewSlot();
   slot->SetShowTitleBar(kFALSE);
   TEveWindowTab* tab2 = slot->MakeTab();
   tab2->SetElementName("Nested");
   tab2->SetShowTitleBar(kFALSE);
   slot =  tab2->NewSlot();
   slot->SetShowTitleBar(kFALSE);
   slot =    tab2->NewSlot();
   slot->SetShowTitleBar(kFALSE);
}
