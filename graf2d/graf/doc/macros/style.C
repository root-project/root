#include "TCanvas.h"
#include "TLatex.h"

TCanvas *style()
{
   TCanvas *S = new TCanvas("script","Style",750,250);

   TLatex Tl;
   Tl.SetTextSize(0.08);
   Tl.SetTextAlign(12);
   TLatex Tt;
   Tt.SetTextSize(0.08);
   Tt.SetTextFont(82);
   Tt.SetTextAlign(12);

   // Draw First Column
   float y, x1, x2;
   float step = 0.3;
   y = 0.80; x1 = 0.01; x2 = x1+0.7;

               Tt.DrawText(x1, y, "#font[12]{Times Italic} and #font[22]{Times bold} : ") ; Tl.DrawLatex(x2, y, "#font[12]{Times Italic} and #font[22]{Times bold}");
   y -= step ; Tt.DrawText(x1, y, "#color[2]{Red} and #color[4]{Blue} : ")                ; Tl.DrawLatex(x2, y, "#color[2]{Red} and #color[4]{Blue}");
   y -= step ; Tt.DrawText(x1, y, "#scale[1.2]{Bigger} and #scale[0.8]{Smaller} : ")      ; Tl.DrawLatex(x2, y, "#scale[1.2]{Bigger} and #scale[0.8]{Smaller}");
   return S;
}
