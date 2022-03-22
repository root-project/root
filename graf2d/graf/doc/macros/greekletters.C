#include "TCanvas.h"
#include "TLatex.h"

TCanvas *greekletters()
{
   TCanvas *Gl = new TCanvas("greek","greek",500,700);

   TLatex Tl;
   Tl.SetTextSize(0.03);

   // Draw the columns titles
   Tl.SetTextAlign(22);
   Tl.DrawLatex(0.165, 0.95, "Lower case");
   Tl.DrawLatex(0.495, 0.95, "Upper case");
   Tl.DrawLatex(0.825, 0.95, "Variations");

   // Draw the lower case letters
   Tl.SetTextAlign(12);
   float y, x1, x2;
   y = 0.90; x1 = 0.07; x2 = x1+0.2;
                 Tl.DrawLatex(x1, y, "alpha : ")   ; Tl.DrawLatex(x2, y, "#alpha");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "beta : ")    ; Tl.DrawLatex(x2, y, "#beta");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "gamma : ")   ; Tl.DrawLatex(x2, y, "#gamma");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "delta : ")   ; Tl.DrawLatex(x2, y, "#delta");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "epsilon : ") ; Tl.DrawLatex(x2, y, "#epsilon");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "zeta : ")    ; Tl.DrawLatex(x2, y, "#zeta");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "eta : ")     ; Tl.DrawLatex(x2, y, "#eta");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "theta : ")   ; Tl.DrawLatex(x2, y, "#theta");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "iota : ")    ; Tl.DrawLatex(x2, y, "#iota");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "kappa : ")   ; Tl.DrawLatex(x2, y, "#kappa");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "lambda : ")  ; Tl.DrawLatex(x2, y, "#lambda");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "mu : ")      ; Tl.DrawLatex(x2, y, "#mu");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "nu : ")      ; Tl.DrawLatex(x2, y, "#nu");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "xi : ")      ; Tl.DrawLatex(x2, y, "#xi");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "omicron : ") ; Tl.DrawLatex(x2, y, "#omicron");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "pi : ")      ; Tl.DrawLatex(x2, y, "#pi");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "rho : ")     ; Tl.DrawLatex(x2, y, "#rho");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "sigma : ")   ; Tl.DrawLatex(x2, y, "#sigma");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "tau : ")     ; Tl.DrawLatex(x2, y, "#tau");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "upsilon : ") ; Tl.DrawLatex(x2, y, "#upsilon");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "phi : ")     ; Tl.DrawLatex(x2, y, "#phi");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "chi : ")     ; Tl.DrawLatex(x2, y, "#chi");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "psi : ")     ; Tl.DrawLatex(x2, y, "#psi");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "omega : ")   ; Tl.DrawLatex(x2, y, "#omega");

   // Draw the upper case letters
   y = 0.90; x1 = 0.40; x2 = x1+0.2;
                 Tl.DrawLatex(x1, y, "Alpha : ")   ; Tl.DrawLatex(x2, y, "#Alpha");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Beta : ")    ; Tl.DrawLatex(x2, y, "#Beta");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Gamma : ")   ; Tl.DrawLatex(x2, y, "#Gamma");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Delta : ")   ; Tl.DrawLatex(x2, y, "#Delta");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Epsilon : ") ; Tl.DrawLatex(x2, y, "#Epsilon");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Zeta : ")    ; Tl.DrawLatex(x2, y, "#Zeta");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Eta : ")     ; Tl.DrawLatex(x2, y, "#Eta");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Theta : ")   ; Tl.DrawLatex(x2, y, "#Theta");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Iota : ")    ; Tl.DrawLatex(x2, y, "#Iota");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Kappa : ")   ; Tl.DrawLatex(x2, y, "#Kappa");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Lambda : ")  ; Tl.DrawLatex(x2, y, "#Lambda");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Mu : ")      ; Tl.DrawLatex(x2, y, "#Mu");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Nu : ")      ; Tl.DrawLatex(x2, y, "#Nu");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Xi : ")      ; Tl.DrawLatex(x2, y, "#Xi");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Omicron : ") ; Tl.DrawLatex(x2, y, "#Omicron");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Pi : ")      ; Tl.DrawLatex(x2, y, "#Pi");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Rho : ")     ; Tl.DrawLatex(x2, y, "#Rho");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Sigma : ")   ; Tl.DrawLatex(x2, y, "#Sigma");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Tau : ")     ; Tl.DrawLatex(x2, y, "#Tau");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Upsilon : ") ; Tl.DrawLatex(x2, y, "#Upsilon");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Phi : ")     ; Tl.DrawLatex(x2, y, "#Phi");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Chi : ")     ; Tl.DrawLatex(x2, y, "#Chi");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Psi : ")     ; Tl.DrawLatex(x2, y, "#Psi");
   y -= 0.0375 ; Tl.DrawLatex(x1, y, "Omega : ")   ; Tl.DrawLatex(x2, y, "#Omega");

   // Draw the variations
   x1 = 0.73; x2 = x1+0.2;
   y = 0.7500 ; Tl.DrawLatex(x1, y, "varepsilon : ") ; Tl.DrawLatex(x2, y, "#varepsilon");
   y = 0.6375 ; Tl.DrawLatex(x1, y, "vartheta : ")   ; Tl.DrawLatex(x2, y, "#vartheta");
   y = 0.2625 ; Tl.DrawLatex(x1, y, "varsigma : ")   ; Tl.DrawLatex(x2, y, "#varsigma");
   y = 0.1875 ; Tl.DrawLatex(x1, y, "varUpsilon : ") ; Tl.DrawLatex(x2, y, "#varUpsilon");
   y = 0.1500 ; Tl.DrawLatex(x1, y, "varphi : ")     ; Tl.DrawLatex(x2, y, "#varphi");
   y = 0.0375 ; Tl.DrawLatex(x1, y, "varomega : ")   ; Tl.DrawLatex(x2, y, "#varomega");

   return Gl;
}
