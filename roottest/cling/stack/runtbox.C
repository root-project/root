
#include "TCanvas.h"
#include "TBox.h"

void thebox(Int_t pat, Double_t x1, Double_t y1,Double_t x2, Double_t  y2)
{
   TBox b;
   Double_t dx = (x2-x1)/3;
   Double_t dy = (y2-y1)/3;
   Double_t h  = (y2-y1)/3;
   b.DrawBox(x1+dx, y1+dy, x2-dx, y2-dy);
}

void runtbox()
{
   TCanvas *Pat = new TCanvas("Patterns", "",0,0,700,900);
   Double_t bh = 0.059;
   Double_t db = 0.01;
   Double_t y  = 0.995;
   Int_t i,j=3001;

   for (i=1; i<=5; i++) {
      thebox(j++, 0.81, y-bh, 0.99, y);
      y = y-bh-db;
   }
   fprintf(stdout,"all box were created without a crash\n");
   delete Pat;
}
