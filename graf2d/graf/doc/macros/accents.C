TCanvas *accents()
{
   TCanvas *S = new TCanvas("script","accents",400,250);

   TLatex Tl;
   Tl.SetTextSize(0.09);
   Tl.SetTextAlign(12);
   TLatex Tt;
   Tt.SetTextSize(0.09);
   Tt.SetTextFont(82);
   Tt.SetTextAlign(12);

   // Draw First Column
   float y, x1, x2;
   float step = 0.25;
   y = 0.85; x1 = 0.02; x2 = x1+0.3;

               Tt.DrawText(x1, y, "#hat")   ; Tl.DrawLatex(x2, y, "#hat{a}");
   y -= step ; Tt.DrawText(x1, y, "#check") ; Tl.DrawLatex(x2, y, "#check{a}");
   y -= step ; Tt.DrawText(x1, y, "#acute") ; Tl.DrawLatex(x2, y, "#acute{a}");
   y -= step ; Tt.DrawText(x1, y, "#grave") ; Tl.DrawLatex(x2, y, "#grave{a}");


   // Draw Second Column
   y = 0.85; x1 = 0.52; x2 = x1+0.3;
               Tt.DrawText(x1, y, "#dot")   ; Tl.DrawLatex(x2, y, "#dot{a}");
   y -= step ; Tt.DrawText(x1, y, "#ddot")  ; Tl.DrawLatex(x2, y, "#ddot{a}");
   y -= step ; Tt.DrawText(x1, y, "#tilde") ; Tl.DrawLatex(x2, y, "#tilde{a} ");

   return S;
}
