#include "TMemFile.h"
#include "TCanvas.h"
#include "TFrame.h"
#include "TH1.h"
#include "TError.h"
#include "TROOT.h"

void writedata(TFile &file)
{
   auto h = new TH1F("h", "", 100, 0, 10);
   h->FillRandom("gaus");

   auto c = new TCanvas("c1");
   c->cd();
   h->Draw();
   c->Draw();
   c->Update();
   file.WriteTObject(c);
   delete c;

   c = new TCanvas("c2");
   c->cd();
   h->Draw();
   file.WriteTObject(c);
   delete c;

   c = new TCanvas("c3");
   c->cd();
   file.WriteTObject(c);
   delete c;

#if 0
   c = new TCanvas("c4");
   c->cd();
   h->Draw();
   c->Draw();
   file.WriteTObject(c);
#endif

   delete h;
}

struct TCountDeletes : public TObject
{
   TObject *fValueMonitored = nullptr;
   TList fMonitorList;

   Int_t fSeen = 0;
   Int_t fFrameSeen = 0;

   void RecursiveRemove(TObject *obj) {
     if (obj == fValueMonitored)
         ++fSeen;
     fMonitorList.Remove(obj);
   }

   TCountDeletes() {
      gROOT->GetListOfCleanups()->Add(this);
   }
   ~TCountDeletes() {
      gROOT->GetListOfCleanups()->Remove(this);
      Info("~TCountDeletes", "Seen %d monitored object being deleted and %d object left undeleted.", fSeen, fMonitorList.GetEntries());
      fMonitorList.Clear("nodelete");
   }

   ClassDef(TCountDeletes, 0);
};

int readdata(TFile &file, const char *name, bool draw)
{
   TCountDeletes tcd;

   std::unique_ptr<TCanvas> c(file.Get<TCanvas>(name));
   if (!c) {
      Error("readdata", "Can not read the TCanvas from the TMemFile.");
      return 1;
   }

   tcd.fMonitorList.AddAll(c->GetListOfPrimitives());
   // Originally the content of the list of primitives (humm maybe except histograms)
   // set to not use RecursiveRemove.  In order to detect memory leak in that list
   // we need to turn it on .. but doing so, also prevents the double delete,
   // so we need to also remove the Canvas from the list of cleaups (indirectly)
   for(TObject *o : *c->GetListOfPrimitives())
      o->SetBit(kMustCleanup);
   if (draw) {
      c->Draw();
      c->Update();
   }
   gROOT->GetListOfCanvases()->Remove(c.get());

   auto f = (TFrame*)c->GetListOfPrimitives()->FindObject("TFrame");
   if (!f) {
      Info("readdata", "There is no frame");
   }
   if (f && f->TestBit(kCanDelete)) {
      Warning("readdata", "The frame is set as 'kCanDelete'");
   }
   if (f) 
      f->SetBit(kMustCleanup);
   tcd.fValueMonitored = f;
   // This was crashing in some cases.
   c.reset(); // force delete so that we can check the result below

   if (f && tcd.fSeen != 1) {
      Error("readdata", "The frame was deleted %d times", tcd.fSeen);
      return 4;
   }
   return 0;
}


int restoreCanvas()
{
   TMemFile f("withcanvas.root", "NEW");
   writedata(f);
   //TFile f("hsimplecanvas.root");
   //return readdata(f, "c1", true);
   return readdata(f, "c1", true) + readdata(f, "c1", false) +
      readdata(f, "c2", true) + readdata(f, "c2", false) +
      readdata(f, "c3", true) + readdata(f, "c3", false);
}
