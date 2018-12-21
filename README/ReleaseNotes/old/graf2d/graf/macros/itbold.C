TCanvas *itbold()
{
   TCanvas *S = new TCanvas("script","Italic Bold",400,150);

   TLatex Tl;
   Tl.SetTextSize(0.18);
   Tl.SetTextAlign(12);
   TLatex Tt;
   Tt.SetTextSize(0.18);
   Tt.SetTextFont(82);
   Tt.SetTextAlign(12);

   // Draw First Column
   float y, x1, x2;
   float step = 0.5;
   y = 0.75; x1 = 0.02; x2 = x1+0.1;

               Tl.DrawLatex(x2, y, "#bf{bold}, #it{italic}, #bf{#it{bold italic}}, #bf{#bf{unbold}}") ; Tt.DrawText(x1, y, "(1)");
   y -= step ; Tl.DrawLatex(x2, y, "abc#alpha#beta#gamma, #it{abc#alpha#beta#gamma}") ; Tt.DrawText(x1, y, "(2)");

   return S;
}
