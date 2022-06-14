TCanvas *kerning()
{
   TCanvas *S = new TCanvas("script","Kerning",400,250);

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
   y = 0.80; x1 = 0.02; x2 = x1+0.1;

               Tl.DrawLatex(x2, y, "Positive k#kern[0.3]{e}#kern[0.3]{r}#kern[0.3]{n}#kern[0.3]{i}#kern[0.3]{n}#kern[0.3]{g}") ; Tt.DrawText(x1, y, "(1)");
   y -= step ; Tl.DrawLatex(x2, y, "Negative k#kern[-0.3]{e}#kern[-0.3]{r}#kern[-0.3]{n}#kern[-0.3]{i}#kern[-0.3]{n}#kern[-0.3]{g}") ; Tt.DrawText(x1, y, "(2)");
   y -= step ; Tl.DrawLatex(x2, y, "Vertical a#lower[0.2]{d}#lower[0.4]{j}#lower[0.1]{u}#lower[-0.1]{s}#lower[-0.3]{t}#lower[-0.4]{m}#lower[-0.2]{e}#lower[0.1]{n}t") ; Tt.DrawText(x1, y, "(3)");

   return S;
}
