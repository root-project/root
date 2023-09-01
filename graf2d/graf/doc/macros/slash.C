TCanvas *slash()
{
   TCanvas *F = new TCanvas("script","Slash",500,100);

   TLatex Tl;
   Tl.SetTextSize(0.3);
   Tl.SetTextAlign(12);
   TLatex Tt;
   Tt.SetTextSize(0.3);
   Tt.SetTextFont(82);
   Tt.SetTextAlign(12);

   float y, x1, x2;
   y = 0.50; x1 = 0.02; x2 = x1+0.7;

   Tt.DrawText(x1, y, "#slash{E}_{T}");
   Tl.DrawLatex(x2, y, "#slash{E}_{T}");

   return F;
}
