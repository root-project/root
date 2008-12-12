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

void test_windows()
{
   TEveManager::Create();

   TEveUtil::Macro("pointset_test.C");

   TEveWindowSlot  *slot  = 0;
   TEveWindowFrame *frame = 0;

   TEveViewer *v = 0;

   // ----------------------------------------------------------------

   slot = TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());

   TEveWindowPack* pack1 = slot->MakePack();
   pack1->SetHorizontal();

   slot = pack1->NewSlot();
   // Embedded viewer.
   v = new TEveViewer("BarViewer");
   v->SpawnGLEmbeddedViewer();
   slot->ReplaceWindow(v);
   v->SetElementName("Bar Embedded Viewer");

   gEve->GetViewers()->AddElement(v);
   v->AddScene(gEve->GetEventScene());

   slot = pack1->NewSlot();   
   TEveWindowPack* pack2 = slot->MakePack();

   slot = pack2->NewSlot();
   slot->StartEmbedding();
   new TCanvas("Root Canvas");
   slot->StopEmbedding();

   slot = pack2->NewSlot();
   // SA viewer.
   v = new TEveViewer("FooViewer");
   v->SpawnGLViewer(gEve->GetEditor());
   slot->ReplaceWindow(v);

   gEve->GetViewers()->AddElement(v);
   v->AddScene(gEve->GetEventScene());   

   // ----------------------------------------------------------------

   slot = TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());

   TEveWindowTab* tab1 = slot->MakeTab();
   slot = tab1->NewSlot();
   frame = slot->MakeFrame(new TRootEmbeddedCanvas());
   frame->SetElementName("Embedded Canvas");

   slot = tab1->NewSlot();

   TEveWindowTab* tab2 = slot->MakeTab();
   tab2->NewSlot();
   tab2->NewSlot();

   // ----------------------------------------------------------------

   gEve->GetBrowser()->GetTabRight()->SetTab(1);

   gDebug = 1;
}
