#include <TH1.h>
#include <TCanvas.h>
#include <TQObject.h>

void exec3()
{
   // Example of using signal/slot in TCanvas/TPad to get feedback
   // about processed events. Note that slots can be either functions
   // or class methods. Compare this with tutorials exec1.C and exec2.C.
   //Author: Ilka Antcheva
   
   TH1F *h = new TH1F("h","h",100,-3,3);
   h->FillRandom("gaus",1000);
   TCanvas *c1=new TCanvas("c1");
   h->Draw();
   c1->Update();
   c1->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)", 0, 0,
               "exec3event(Int_t,Int_t,Int_t,TObject*)");
}

void exec3event(Int_t event, Int_t x, Int_t y, TObject *selected)
{
   TCanvas *c = (TCanvas *) gTQSender;
   printf("Canvas %s: event=%d, x=%d, y=%d, selected=%s\n", c->GetName(),
          event, x, y, selected->IsA()->GetName());
}
