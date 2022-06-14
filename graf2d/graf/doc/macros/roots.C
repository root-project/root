TCanvas *roots()
{
   TCanvas *S = new TCanvas("script","Roots",400,150);

   TLatex Tl;
   Tl.SetTextSize(0.15);
   Tl.SetTextAlign(12);
   TLatex Tt;
   Tt.SetTextSize(0.15);
   Tt.SetTextFont(82);
   Tt.SetTextAlign(12);

   // Draw First Column
   float y, x1, x2;
   float step = 0.5;
   y = 0.75; x1 = 0.02; x2 = x1+0.8;

               Tl.DrawLatex(x2, y, "#sqrt{10}")    ; Tt.DrawText(x1, y, "#sqrt{10}");
   y -= step ; Tl.DrawLatex(x2, y, "#sqrt[3]{10}") ; Tt.DrawText(x1, y, "#sqrt[3]{10}");


   return S;
}
