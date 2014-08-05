// Draw the Greek letters as a table and save the result as GIF, PS, PDF
// and SVG files.
// Lowercase Greek letters are obtained by adding a # to the name of the letter.
// For an uppercase Greek letter, just capitalize the first letter of the
// command name. Some letter have two representations. The name of the
// second one (the "variation") starts with "var".
//Author: Rene Brun
void latex4() {
   TCanvas *c1 = new TCanvas("greek","greek",600,700);

   TLatex l;
   l.SetTextSize(0.03);

   // Draw the columns titles
   l.SetTextAlign(22);
   l.DrawLatex(0.165, 0.95, "Lower case");
   l.DrawLatex(0.495, 0.95, "Upper case");
   l.DrawLatex(0.825, 0.95, "Variations");

   // Draw the lower case letters
   l.SetTextAlign(12);
   float y, x1, x2;
   y = 0.90; x1 = 0.07; x2 = x1+0.2;
                 l.DrawLatex(x1, y, "alpha : ")   ; l.DrawLatex(x2, y, "#alpha");
   y -= 0.0375 ; l.DrawLatex(x1, y, "beta : ")    ; l.DrawLatex(x2, y, "#beta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "gamma : ")   ; l.DrawLatex(x2, y, "#gamma");
   y -= 0.0375 ; l.DrawLatex(x1, y, "delta : ")   ; l.DrawLatex(x2, y, "#delta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "epsilon : ") ; l.DrawLatex(x2, y, "#epsilon");
   y -= 0.0375 ; l.DrawLatex(x1, y, "zeta : ")    ; l.DrawLatex(x2, y, "#zeta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "eta : ")     ; l.DrawLatex(x2, y, "#eta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "theta : ")   ; l.DrawLatex(x2, y, "#theta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "iota : ")    ; l.DrawLatex(x2, y, "#iota");
   y -= 0.0375 ; l.DrawLatex(x1, y, "kappa : ")   ; l.DrawLatex(x2, y, "#kappa");
   y -= 0.0375 ; l.DrawLatex(x1, y, "lambda : ")  ; l.DrawLatex(x2, y, "#lambda");
   y -= 0.0375 ; l.DrawLatex(x1, y, "mu : ")      ; l.DrawLatex(x2, y, "#mu");
   y -= 0.0375 ; l.DrawLatex(x1, y, "nu : ")      ; l.DrawLatex(x2, y, "#nu");
   y -= 0.0375 ; l.DrawLatex(x1, y, "xi : ")      ; l.DrawLatex(x2, y, "#xi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "omicron : ") ; l.DrawLatex(x2, y, "#omicron");
   y -= 0.0375 ; l.DrawLatex(x1, y, "pi : ")      ; l.DrawLatex(x2, y, "#pi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "rho : ")     ; l.DrawLatex(x2, y, "#rho");
   y -= 0.0375 ; l.DrawLatex(x1, y, "sigma : ")   ; l.DrawLatex(x2, y, "#sigma");
   y -= 0.0375 ; l.DrawLatex(x1, y, "tau : ")     ; l.DrawLatex(x2, y, "#tau");
   y -= 0.0375 ; l.DrawLatex(x1, y, "upsilon : ") ; l.DrawLatex(x2, y, "#upsilon");
   y -= 0.0375 ; l.DrawLatex(x1, y, "phi : ")     ; l.DrawLatex(x2, y, "#phi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "chi : ")     ; l.DrawLatex(x2, y, "#chi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "psi : ")     ; l.DrawLatex(x2, y, "#psi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "omega : ")   ; l.DrawLatex(x2, y, "#omega");

   // Draw the upper case letters
   y = 0.90; x1 = 0.40; x2 = x1+0.2;
                 l.DrawLatex(x1, y, "Alpha : ")   ; l.DrawLatex(x2, y, "#Alpha");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Beta : ")    ; l.DrawLatex(x2, y, "#Beta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Gamma : ")   ; l.DrawLatex(x2, y, "#Gamma");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Delta : ")   ; l.DrawLatex(x2, y, "#Delta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Epsilon : ") ; l.DrawLatex(x2, y, "#Epsilon");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Zeta : ")    ; l.DrawLatex(x2, y, "#Zeta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Eta : ")     ; l.DrawLatex(x2, y, "#Eta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Theta : ")   ; l.DrawLatex(x2, y, "#Theta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Iota : ")    ; l.DrawLatex(x2, y, "#Iota");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Kappa : ")   ; l.DrawLatex(x2, y, "#Kappa");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Lambda : ")  ; l.DrawLatex(x2, y, "#Lambda");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Mu : ")      ; l.DrawLatex(x2, y, "#Mu");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Nu : ")      ; l.DrawLatex(x2, y, "#Nu");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Xi : ")      ; l.DrawLatex(x2, y, "#Xi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Omicron : ") ; l.DrawLatex(x2, y, "#Omicron");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Pi : ")      ; l.DrawLatex(x2, y, "#Pi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Rho : ")     ; l.DrawLatex(x2, y, "#Rho");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Sigma : ")   ; l.DrawLatex(x2, y, "#Sigma");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Tau : ")     ; l.DrawLatex(x2, y, "#Tau");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Upsilon : ") ; l.DrawLatex(x2, y, "#Upsilon");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Phi : ")     ; l.DrawLatex(x2, y, "#Phi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Chi : ")     ; l.DrawLatex(x2, y, "#Chi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Psi : ")     ; l.DrawLatex(x2, y, "#Psi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Omega : ")   ; l.DrawLatex(x2, y, "#Omega");

   // Draw the variations
   x1 = 0.73; x2 = x1+0.2;
   y = 0.7500 ; l.DrawLatex(x1, y, "varepsilon : ") ; l.DrawLatex(x2, y, "#varepsilon");
   y = 0.6375 ; l.DrawLatex(x1, y, "vartheta : ")   ; l.DrawLatex(x2, y, "#vartheta");
   y = 0.2625 ; l.DrawLatex(x1, y, "varsigma : ")   ; l.DrawLatex(x2, y, "#varsigma");
   y = 0.1875 ; l.DrawLatex(x1, y, "varUpsilon : ") ; l.DrawLatex(x2, y, "#varUpsilon");
   y = 0.1500 ; l.DrawLatex(x1, y, "varphi : ")     ; l.DrawLatex(x2, y, "#varphi");
   y = 0.0375 ; l.DrawLatex(x1, y, "varomega : ")   ; l.DrawLatex(x2, y, "#varomega");

   // Save the picture in various formats
   c1->Print("greek.ps");
   c1->Print("greek.gif");
   c1->Print("greek.pdf");
   c1->Print("greek.svg");
}
