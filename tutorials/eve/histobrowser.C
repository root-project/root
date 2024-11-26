/// \file
/// \ingroup tutorial_eve
/// Demonstrates how to use EVE as a histogram browser.
///
/// \image html eve_histobrowser.png
/// \macro_code
///
/// \author Matevz Tadel

TGFileBrowser *g_hlt_browser = 0;
TCanvas       *g_hlt_canvas  = 0;

void histobrowser(const char* name="HLT Histos")
{
   TEveManager::Create();

   // --- Create special browser

   gEve->GetBrowser()->StartEmbedding(0);
   g_hlt_browser = gEve->GetBrowser()->MakeFileBrowser();
   gEve->GetBrowser()->StopEmbedding(name);

   // --- Fill and register some lists/folders/histos

   gDirectory = 0;
   TH1F* h;

   TList* l = new TList;
   l->SetName("Cilka");
   h = new TH1F("Foo", "Bar", 51, 0, 1);
   for (Int_t i=0; i<500; ++i)
      h->Fill(gRandom->Gaus(.63, .2));
   l->Add(h);
   g_hlt_browser->Add(l);

   TFolder* f = new TFolder("Booboayes", "Statisticos");
   h = new TH1F("Fooes", "Baros", 51, 0, 1);
   for (Int_t i=0; i<2000; ++i) {
      h->Fill(gRandom->Gaus(.7, .1));
      h->Fill(gRandom->Gaus(.3, .1));
   }
   f->Add(h);
   g_hlt_browser->Add(f);

   h = new TH1F("Fooesoto", "Barosana", 51, 0, 1);
   for (Int_t i=0; i<4000; ++i) {
      h->Fill(gRandom->Gaus(.25, .02), 0.04);
      h->Fill(gRandom->Gaus(.5, .1));
      h->Fill(gRandom->Gaus(.75, .02), 0.04);
   }
   g_hlt_browser->Add(h);

   // --- Add some macros.

   TMacro* m;

   m = new TMacro;
   m->AddLine("{ g_hlt_canvas->Clear();"
              "  g_hlt_canvas->cd();"
              "  g_hlt_canvas->Update(); }");
   m->SetName("Clear Canvas");
   g_hlt_browser->Add(m);

   m = new TMacro;
   m->AddLine("{ g_hlt_canvas->Clear();"
              "  g_hlt_canvas->Divide(2,2);"
              "  g_hlt_canvas->cd(1);"
              "  g_hlt_canvas->Update(); }");
   m->SetName("Split Canvas");
   g_hlt_browser->Add(m);

   // --- Create an embedded canvas

   gEve->GetBrowser()->StartEmbedding(1);
   g_hlt_canvas = new TCanvas;
   gEve->GetBrowser()->StopEmbedding("HLT Canvas");
}
