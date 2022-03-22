#include "TCanvas.h"
#include "TLatex.h"

TCanvas *subsupscript2()
{
   TCanvas *S = new TCanvas("script","Subscripts and Superscripts",400,250);

   TLatex Tl;
   Tl.SetTextSize(0.09);
   Tl.SetTextAlign(12);
   TLatex Tt;
   Tt.SetTextSize(0.09);
   Tt.SetTextFont(82);
   Tt.SetTextAlign(12);

   // Draw First Column
   float y, x1, x2;
   float step = 0.3;
   y = 0.80; x1 = 0.02; x2 = x1+0.8;

               Tl.DrawLatex(x2, y, "{}^{40}_{20}Ca")     ; Tt.DrawText(x1, y, "(1) {}^{40}_{20}Ca");
   y -= step ; Tl.DrawLatex(x2, y, "f_{E}/f_{E}")        ; Tt.DrawText(x1, y, "(2) f_{E}/f_{E}");
   y -= step ; Tl.DrawLatex(x2, y, "f_{E}/^{}f_{E}")     ; Tt.DrawText(x1, y, "(3) f_{E}/^{}f_{E}");


   return S;
}
