TCanvas *splitline()
{
   TCanvas *F = new TCanvas("script","Splitline",700,100);

   TLatex Tl;
   Tl.SetTextSize(0.3);
   Tl.SetTextAlign(12);
   TLatex Tt;
   Tt.SetTextSize(0.3);
   Tt.SetTextFont(82);
   Tt.SetTextAlign(12);

   float y, x1, x2;
   y = 0.50; x1 = 0.02; x2 = x1+0.7;

   Tt.DrawText(x1, y, "#splitline{21 April 2003}{14:02:30}");
   Tl.DrawLatex(x2, y, "#splitline{21 April 2003}{14:02:30}");

   return F;
}
