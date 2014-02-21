TCanvas *subsupscript()
{
   TCanvas *S = new TCanvas("script","Subscripts and Superscripts",400,250);

   TLatex Tl;
   Tl.SetTextSize(0.09);
   TLatex Tt;
   Tt.SetTextSize(0.09);
   Tt.SetTextFont(82);

   // Draw First Column
   Tl.SetTextAlign(12);
   float y, x1, x2;
   float step = 0.3;
   y = 0.80; x1 = 0.02; x2 = x1+0.12;

               Tl.DrawLatex(x1, y, "x^{2y}")     ; Tt.DrawText(x2, y, "x^{2y}");
   y -= step ; Tl.DrawLatex(x1, y, "x_{2y}")     ; Tt.DrawText(x2, y, "x_{2y}");
   y -= step ; Tl.DrawLatex(x1, y, "x^{y^{2}}")  ; Tt.DrawText(x2, y, "x^{y^{2}}");


   // Draw Second Column
   y = 0.80; x1 = 0.52; x2 = x1+0.12;
               Tl.DrawLatex(x1, y, "x^{y_{1}}")  ; Tt.DrawText(x2, y, "x^{y_{1}}");
   y -= step ; Tl.DrawLatex(x1, y, "x^{y}_{1}")  ; Tt.DrawText(x2, y, "x^{y}_{1}");
   y -= step ; Tl.DrawLatex(x1, y, "x_{1}^{y}")  ; Tt.DrawText(x2, y, "x_{1}^{y}");

   return S;
}
