#include "TPolyLine.h"

void polyline()
{
   Double_t x[5] = {.2,.7,.6,.25,.2};
   Double_t y[5] = {.5,.1,.9,.7,.5};
   TPolyLine *pline = new TPolyLine(5,x,y);
   pline->SetFillColor(38);
   pline->SetLineColor(2);
   pline->SetLineWidth(4);
   pline->Draw("f");
   pline->Draw();
}
