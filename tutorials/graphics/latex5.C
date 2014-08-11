// This draws the Mathematical Symbols letters as a table and save the result
// as GIF, PS, PDF and SVG files.
//Author: Rene Brun

void latex5() {
   TCanvas *c1 = new TCanvas("mathsymb","Mathematical Symbols",600,600);

   TLatex l;
   l.SetTextSize(0.03);

   // Draw First Column
   l.SetTextAlign(12);
   float y, step, x1, x2;
   y = 0.96; step = 0.0465; x1 = 0.02; x2 = x1+0.04;
               l.DrawLatex(x1, y, "#club")           ; l.DrawText(x2, y, "#club");
   y -= step ; l.DrawLatex(x1, y, "#voidn")          ; l.DrawText(x2, y, "#voidn");
   y -= step ; l.DrawLatex(x1, y, "#leq")            ; l.DrawText(x2, y, "#leq");
   y -= step ; l.DrawLatex(x1, y, "#approx")         ; l.DrawText(x2, y, "#approx");
   y -= step ; l.DrawLatex(x1, y, "#in")             ; l.DrawText(x2, y, "#in");
   y -= step ; l.DrawLatex(x1, y, "#supset")         ; l.DrawText(x2, y, "#supset");
   y -= step ; l.DrawLatex(x1, y, "#cap")            ; l.DrawText(x2, y, "#cap");
   y -= step ; l.DrawLatex(x1, y, "#ocopyright")     ; l.DrawText(x2, y, "#ocopyright");
   y -= step ; l.DrawLatex(x1, y, "#trademark")      ; l.DrawText(x2, y, "#trademark");
   y -= step ; l.DrawLatex(x1, y, "#times")          ; l.DrawText(x2, y, "#times");
   y -= step ; l.DrawLatex(x1, y, "#bullet")         ; l.DrawText(x2, y, "#bullet");
   y -= step ; l.DrawLatex(x1, y, "#voidb")          ; l.DrawText(x2, y, "#voidb");
   y -= step ; l.DrawLatex(x1, y, "#doublequote")    ; l.DrawText(x2, y, "#doublequote");
   y -= step ; l.DrawLatex(x1, y, "#lbar")           ; l.DrawText(x2, y, "#lbar");
   y -= step ; l.DrawLatex(x1, y, "#arcbottom")      ; l.DrawText(x2, y, "#arcbottom");
   y -= step ; l.DrawLatex(x1, y, "#downarrow")      ; l.DrawText(x2, y, "#downarrow");
   y -= step ; l.DrawLatex(x1, y, "#leftrightarrow") ; l.DrawText(x2, y, "#leftrightarrow");
   y -= step ; l.DrawLatex(x1, y, "#Downarrow")      ; l.DrawText(x2, y, "#Downarrow");
   y -= step ; l.DrawLatex(x1, y, "#Leftrightarrow") ; l.DrawText(x2, y, "#Leftrightarrow");
   y -= step ; l.DrawLatex(x1, y, "#void8")          ; l.DrawText(x2, y, "#void8");
   y -= step ; l.DrawLatex(x1, y, "#hbar")           ; l.DrawText(x2, y, "#hbar");

   // Draw Second Column
   y = 0.96; step = 0.0465; x1 = 0.27; x2 = x1+0.04;
               l.DrawLatex(x1, y, "#diamond")        ; l.DrawText(x2, y, "#diamond");
   y -= step ; l.DrawLatex(x1, y, "#aleph")          ; l.DrawText(x2, y, "#aleph");
   y -= step ; l.DrawLatex(x1, y, "#geq")            ; l.DrawText(x2, y, "#geq");
   y -= step ; l.DrawLatex(x1, y, "#neq")            ; l.DrawText(x2, y, "#neq");
   y -= step ; l.DrawLatex(x1, y, "#notin")          ; l.DrawText(x2, y, "#notin");
   y -= step ; l.DrawLatex(x1, y, "#subseteq")       ; l.DrawText(x2, y, "#subseteq");
   y -= step ; l.DrawLatex(x1, y, "#cup")            ; l.DrawText(x2, y, "#cup");
   y -= step ; l.DrawLatex(x1, y, "#copyright")      ; l.DrawText(x2, y, "#copyright");
   y -= step ; l.DrawLatex(x1, y, "#void3")          ; l.DrawText(x2, y, "#void3");
   y -= step ; l.DrawLatex(x1, y, "#divide")         ; l.DrawText(x2, y, "#divide");
   y -= step ; l.DrawLatex(x1, y, "#circ")           ; l.DrawText(x2, y, "#circ");
   y -= step ; l.DrawLatex(x1, y, "#infty")          ; l.DrawText(x2, y, "#infty");
   y -= step ; l.DrawLatex(x1, y, "#angle")          ; l.DrawText(x2, y, "#angle");
   y -= step ; l.DrawLatex(x1, y, "#cbar")           ; l.DrawText(x2, y, "#cbar");
   y -= step ; l.DrawLatex(x1, y, "#arctop")         ; l.DrawText(x2, y, "#arctop");
   y -= step ; l.DrawLatex(x1, y, "#leftarrow")      ; l.DrawText(x2, y, "#leftarrow");
   y -= step ; l.DrawLatex(x1, y, "#otimes")         ; l.DrawText(x2, y, "#otimes");
   y -= step ; l.DrawLatex(x1, y, "#Leftarrow")      ; l.DrawText(x2, y, "#Leftarrow");
   y -= step ; l.DrawLatex(x1, y, "#prod")           ; l.DrawText(x2, y, "#prod");
   y -= step ; l.DrawLatex(x1, y, "#Box")            ; l.DrawText(x2, y, "#Box");
   y -= step ; l.DrawLatex(x1, y, "#parallel")       ; l.DrawText(x2, y, "#parallel");

   // Draw Third Column
   y = 0.96; step = 0.0465; x1 = 0.52; x2 = x1+0.04;
               l.DrawLatex(x1, y, "#heart")          ; l.DrawText(x2, y, "#heart");
   y -= step ; l.DrawLatex(x1, y, "#Jgothic")        ; l.DrawText(x2, y, "#Jgothic");
   y -= step ; l.DrawLatex(x1, y, "#LT")             ; l.DrawText(x2, y, "#LT");
   y -= step ; l.DrawLatex(x1, y, "#equiv")          ; l.DrawText(x2, y, "#equiv");
   y -= step ; l.DrawLatex(x1, y, "#subset")         ; l.DrawText(x2, y, "#subset");
   y -= step ; l.DrawLatex(x1, y, "#supseteq")       ; l.DrawText(x2, y, "#supseteq");
   y -= step ; l.DrawLatex(x1, y, "#wedge")          ; l.DrawText(x2, y, "#wedge");
   y -= step ; l.DrawLatex(x1, y, "#oright")         ; l.DrawText(x2, y, "#oright");
   y -= step ; l.DrawLatex(x1, y, "#AA")             ; l.DrawText(x2, y, "#AA");
   y -= step ; l.DrawLatex(x1, y, "#pm")             ; l.DrawText(x2, y, "#pm");
   y -= step ; l.DrawLatex(x1, y, "#3dots")          ; l.DrawText(x2, y, "#3dots");
   y -= step ; l.DrawLatex(x1, y, "#nabla")          ; l.DrawText(x2, y, "#nabla");
   y -= step ; l.DrawLatex(x1, y, "#downleftarrow")  ; l.DrawText(x2, y, "#downleftarrow");
   y -= step ; l.DrawLatex(x1, y, "#topbar")         ; l.DrawText(x2, y, "#topbar");
   y -= step ; l.DrawLatex(x1, y, "#arcbar")         ; l.DrawText(x2, y, "#arcbar");
   y -= step ; l.DrawLatex(x1, y, "#uparrow")        ; l.DrawText(x2, y, "#uparrow");
   y -= step ; l.DrawLatex(x1, y, "#oplus")          ; l.DrawText(x2, y, "#oplus");
   y -= step ; l.DrawLatex(x1, y, "#Uparrow")        ; l.DrawText(x2, y, "#Uparrow");
   y -= step ; l.DrawLatex(x1, y-0.01, "#sum")       ; l.DrawText(x2, y, "#sum");
   y -= step ; l.DrawLatex(x1, y, "#perp")           ; l.DrawText(x2, y, "#perp");

   // Draw Fourth Column
   y = 0.96; step = 0.0465; x1 = 0.77; x2 = x1+0.04;
               l.DrawLatex(x1, y, "#spade")          ; l.DrawText(x2, y, "#spade");
   y -= step ; l.DrawLatex(x1, y, "#Rgothic")        ; l.DrawText(x2, y, "#Rgothic");
   y -= step ; l.DrawLatex(x1, y, "#GT")             ; l.DrawText(x2, y, "#GT");
   y -= step ; l.DrawLatex(x1, y, "#propto")         ; l.DrawText(x2, y, "#propto");
   y -= step ; l.DrawLatex(x1, y, "#notsubset")      ; l.DrawText(x2, y, "#notsubset");
   y -= step ; l.DrawLatex(x1, y, "#oslash")         ; l.DrawText(x2, y, "#oslash");
   y -= step ; l.DrawLatex(x1, y, "#vee")            ; l.DrawText(x2, y, "#vee");
   y -= step ; l.DrawLatex(x1, y, "#void1")          ; l.DrawText(x2, y, "#void1");
   y -= step ; l.DrawLatex(x1, y, "#aa")             ; l.DrawText(x2, y, "#aa");
   y -= step ; l.DrawLatex(x1, y, "#/")              ; l.DrawText(x2, y, "#/");
   y -= step ; l.DrawLatex(x1, y, "#upoint")         ; l.DrawText(x2, y, "#upoint");
   y -= step ; l.DrawLatex(x1, y, "#partial")        ; l.DrawText(x2, y, "#partial");
   y -= step ; l.DrawLatex(x1, y, "#corner")         ; l.DrawText(x2, y, "#corner");
   y -= step ; l.DrawLatex(x1, y, "#ltbar")          ; l.DrawText(x2, y, "#ltbar");
   y -= step ; l.DrawLatex(x1, y, "#bottombar")      ; l.DrawText(x2, y, "#bottombar");
   y -= step ; l.DrawLatex(x1, y, "#rightarrow")     ; l.DrawText(x2, y, "#rightarrow");
   y -= step ; l.DrawLatex(x1, y, "#surd")           ; l.DrawText(x2, y, "#surd");
   y -= step ; l.DrawLatex(x1, y, "#Rightarrow")     ; l.DrawText(x2, y, "#Rightarrow");
   y -= step ; l.DrawLatex(x1, y-0.015, "#int")      ; l.DrawText(x2, y, "#int");
   y -= step ; l.DrawLatex(x1, y, "#odot")           ; l.DrawText(x2, y, "#odot");

   // Save the picture in various formats
   c1->Print("mathsymb.ps");
   c1->Print("mathsymb.gif");
   c1->Print("mathsymb.pdf");
   c1->Print("mathsymb.svg");
}
