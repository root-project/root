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

   auto c = new TCanvas("canvHistDrawUpdate");
   c->cd();
   h->Draw();
   c->Draw();
   c->Update();
   file.WriteTObject(c);
   delete c;

   c = new TCanvas("canvHistDrawOnly");
   c->cd();
   h->Draw();
   file.WriteTObject(c);
   delete c;

   c = new TCanvas("emptyCanvas");
   c->cd();
   file.WriteTObject(c);
   delete c;

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
   if (gROOT->GetListOfCanvases()->FindObject(c.get())) {
      Error("readdata", "Failed to remove the canvas '%s' from the list of canvases", name);
      return 5;
   }

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

   return readdata(f, "canvHistDrawUpdate", true) + readdata(f, "canvHistDrawUpdate", false) +
      readdata(f, "canvHistDrawOnly", true) + readdata(f, "canvHistDrawOnly", false) +
      readdata(f, "emptyCanvas", true) + readdata(f, "emptyCanvas", false);
}
