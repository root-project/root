#include "TH1.h"
#include "TDirectory.h"
#include "TList.h"

TH1F *h=0;
TH1F *h2=0;
TH1F *h3=0;

void runcopy()
{
   if (h) delete h;
   if (h2) delete h2;
   if (h3) delete h3;
   fprintf(stdout,"List has %d elements!\n",gDirectory->GetList()->GetSize());

   h = new TH1F("h", "", 100, 0, 100);
   h2 = (TH1F *) h->Clone("h2");
   h3 = (TH1F *) h->Clone("h3");

   // do something

   //h3->SetDirectory(0);
   *h3 = *h + *h2;
   //h3->SetName("h3");
   //h3->SetDirectory(gDirectory);

}

