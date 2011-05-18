{
   TCanvas *Ms = new TCanvas("mathsymb","Mathematical Symbols",500,600);

   TLatex Tl;
   Tl.SetTextSize(0.03);

   // Draw First Column
   Tl.SetTextAlign(12);
   float y, step, x1, x2;
   y = 0.96; step = 0.0465; x1 = 0.02; x2 = x1+0.04;
               Tl.DrawLatex(x1, y, "#club")           ; Tl.DrawText(x2, y, "#club");
   y -= step ; Tl.DrawLatex(x1, y, "#voidn")          ; Tl.DrawText(x2, y, "#voidn");
   y -= step ; Tl.DrawLatex(x1, y, "#leq")            ; Tl.DrawText(x2, y, "#leq");
   y -= step ; Tl.DrawLatex(x1, y, "#approx")         ; Tl.DrawText(x2, y, "#approx");
   y -= step ; Tl.DrawLatex(x1, y, "#in")             ; Tl.DrawText(x2, y, "#in");
   y -= step ; Tl.DrawLatex(x1, y, "#supset")         ; Tl.DrawText(x2, y, "#supset");
   y -= step ; Tl.DrawLatex(x1, y, "#cap")            ; Tl.DrawText(x2, y, "#cap");
   y -= step ; Tl.DrawLatex(x1, y, "#ocopyright")     ; Tl.DrawText(x2, y, "#ocopyright");
   y -= step ; Tl.DrawLatex(x1, y, "#trademark")      ; Tl.DrawText(x2, y, "#trademark");
   y -= step ; Tl.DrawLatex(x1, y, "#times")          ; Tl.DrawText(x2, y, "#times");
   y -= step ; Tl.DrawLatex(x1, y, "#bullet")         ; Tl.DrawText(x2, y, "#bullet");
   y -= step ; Tl.DrawLatex(x1, y, "#voidb")          ; Tl.DrawText(x2, y, "#voidb");
   y -= step ; Tl.DrawLatex(x1, y, "#doublequote")    ; Tl.DrawText(x2, y, "#doublequote");
   y -= step ; Tl.DrawLatex(x1, y, "#lbar")           ; Tl.DrawText(x2, y, "#lbar");
   y -= step ; Tl.DrawLatex(x1, y, "#arcbottom")      ; Tl.DrawText(x2, y, "#arcbottom");
   y -= step ; Tl.DrawLatex(x1, y, "#downarrow")      ; Tl.DrawText(x2, y, "#downarrow");
   y -= step ; Tl.DrawLatex(x1, y, "#leftrightarrow") ; Tl.DrawText(x2, y, "#leftrightarrow");
   y -= step ; Tl.DrawLatex(x1, y, "#Downarrow")      ; Tl.DrawText(x2, y, "#Downarrow");
   y -= step ; Tl.DrawLatex(x1, y, "#Leftrightarrow") ; Tl.DrawText(x2, y, "#Leftrightarrow");
   y -= step ; Tl.DrawLatex(x1, y, "#void8")          ; Tl.DrawText(x2, y, "#void8");
   y -= step ; Tl.DrawLatex(x1, y, "#hbar")           ; Tl.DrawText(x2, y, "#hbar");

   // Draw Second Column
   y = 0.96; step = 0.0465; x1 = 0.27; x2 = x1+0.04;
               Tl.DrawLatex(x1, y, "#diamond")        ; Tl.DrawText(x2, y, "#diamond");
   y -= step ; Tl.DrawLatex(x1, y, "#aleph")          ; Tl.DrawText(x2, y, "#aleph");
   y -= step ; Tl.DrawLatex(x1, y, "#geq")            ; Tl.DrawText(x2, y, "#geq");
   y -= step ; Tl.DrawLatex(x1, y, "#neq")            ; Tl.DrawText(x2, y, "#neq");
   y -= step ; Tl.DrawLatex(x1, y, "#notin")          ; Tl.DrawText(x2, y, "#notin");
   y -= step ; Tl.DrawLatex(x1, y, "#subseteq")       ; Tl.DrawText(x2, y, "#subseteq");
   y -= step ; Tl.DrawLatex(x1, y, "#cup")            ; Tl.DrawText(x2, y, "#cup");
   y -= step ; Tl.DrawLatex(x1, y, "#copyright")      ; Tl.DrawText(x2, y, "#copyright");
   y -= step ; Tl.DrawLatex(x1, y, "#void3")          ; Tl.DrawText(x2, y, "#void3");
   y -= step ; Tl.DrawLatex(x1, y, "#divide")         ; Tl.DrawText(x2, y, "#divide");
   y -= step ; Tl.DrawLatex(x1, y, "#circ")           ; Tl.DrawText(x2, y, "#circ");
   y -= step ; Tl.DrawLatex(x1, y, "#infty")          ; Tl.DrawText(x2, y, "#infty");
   y -= step ; Tl.DrawLatex(x1, y, "#angle")          ; Tl.DrawText(x2, y, "#angle");
   y -= step ; Tl.DrawLatex(x1, y, "#cbar")           ; Tl.DrawText(x2, y, "#cbar");
   y -= step ; Tl.DrawLatex(x1, y, "#arctop")         ; Tl.DrawText(x2, y, "#arctop");
   y -= step ; Tl.DrawLatex(x1, y, "#leftarrow")      ; Tl.DrawText(x2, y, "#leftarrow");
   y -= step ; Tl.DrawLatex(x1, y, "#otimes")         ; Tl.DrawText(x2, y, "#otimes");
   y -= step ; Tl.DrawLatex(x1, y, "#Leftarrow")      ; Tl.DrawText(x2, y, "#Leftarrow");
   y -= step ; Tl.DrawLatex(x1, y, "#prod")           ; Tl.DrawText(x2, y, "#prod");
   y -= step ; Tl.DrawLatex(x1, y, "#Box")            ; Tl.DrawText(x2, y, "#Box");
   y -= step ; Tl.DrawLatex(x1, y, "#parallel")       ; Tl.DrawText(x2, y, "#parallel");

   // Draw Third Column
   y = 0.96; step = 0.0465; x1 = 0.52; x2 = x1+0.04;
               Tl.DrawLatex(x1, y, "#heart")          ; Tl.DrawText(x2, y, "#heart");
   y -= step ; Tl.DrawLatex(x1, y, "#Jgothic")        ; Tl.DrawText(x2, y, "#Jgothic");
   y -= step ; Tl.DrawLatex(x1, y, "#LT")             ; Tl.DrawText(x2, y, "#LT");
   y -= step ; Tl.DrawLatex(x1, y, "#equiv")          ; Tl.DrawText(x2, y, "#equiv");
   y -= step ; Tl.DrawLatex(x1, y, "#subset")         ; Tl.DrawText(x2, y, "#subset");
   y -= step ; Tl.DrawLatex(x1, y, "#supseteq")       ; Tl.DrawText(x2, y, "#supseteq");
   y -= step ; Tl.DrawLatex(x1, y, "#wedge")          ; Tl.DrawText(x2, y, "#wedge");
   y -= step ; Tl.DrawLatex(x1, y, "#oright")         ; Tl.DrawText(x2, y, "#oright");
   y -= step ; Tl.DrawLatex(x1, y, "#AA")             ; Tl.DrawText(x2, y, "#AA");
   y -= step ; Tl.DrawLatex(x1, y, "#pm")             ; Tl.DrawText(x2, y, "#pm");
   y -= step ; Tl.DrawLatex(x1, y, "#3dots")          ; Tl.DrawText(x2, y, "#3dots");
   y -= step ; Tl.DrawLatex(x1, y, "#nabla")          ; Tl.DrawText(x2, y, "#nabla");
   y -= step ; Tl.DrawLatex(x1, y, "#downleftarrow")  ; Tl.DrawText(x2, y, "#downleftarrow");
   y -= step ; Tl.DrawLatex(x1, y, "#topbar")         ; Tl.DrawText(x2, y, "#topbar");
   y -= step ; Tl.DrawLatex(x1, y, "#arcbar")         ; Tl.DrawText(x2, y, "#arcbar");
   y -= step ; Tl.DrawLatex(x1, y, "#uparrow")        ; Tl.DrawText(x2, y, "#uparrow");
   y -= step ; Tl.DrawLatex(x1, y, "#oplus")          ; Tl.DrawText(x2, y, "#oplus");
   y -= step ; Tl.DrawLatex(x1, y, "#Uparrow")        ; Tl.DrawText(x2, y, "#Uparrow");
   y -= step ; Tl.DrawLatex(x1, y-0.01, "#sum")       ; Tl.DrawText(x2, y, "#sum");
   y -= step ; Tl.DrawLatex(x1, y, "#perp")           ; Tl.DrawText(x2, y, "#perp");
   y -= step ; Tl.DrawLatex(x1, y, "#forall")         ; Tl.DrawText(x2, y, "#forall");  

   // Draw Fourth Column
   y = 0.96; step = 0.0465; x1 = 0.77; x2 = x1+0.04;
               Tl.DrawLatex(x1, y, "#spade")          ; Tl.DrawText(x2, y, "#spade");
   y -= step ; Tl.DrawLatex(x1, y, "#Rgothic")        ; Tl.DrawText(x2, y, "#Rgothic");
   y -= step ; Tl.DrawLatex(x1, y, "#GT")             ; Tl.DrawText(x2, y, "#GT");
   y -= step ; Tl.DrawLatex(x1, y, "#propto")         ; Tl.DrawText(x2, y, "#propto");
   y -= step ; Tl.DrawLatex(x1, y, "#notsubset")      ; Tl.DrawText(x2, y, "#notsubset");
   y -= step ; Tl.DrawLatex(x1, y, "#oslash")         ; Tl.DrawText(x2, y, "#oslash");
   y -= step ; Tl.DrawLatex(x1, y, "#vee")            ; Tl.DrawText(x2, y, "#vee");
   y -= step ; Tl.DrawLatex(x1, y, "#void1")          ; Tl.DrawText(x2, y, "#void1");
   y -= step ; Tl.DrawLatex(x1, y, "#aa")             ; Tl.DrawText(x2, y, "#aa");
   y -= step ; Tl.DrawLatex(x1, y, "#/")              ; Tl.DrawText(x2, y, "#/");
   y -= step ; Tl.DrawLatex(x1, y, "#upoint")         ; Tl.DrawText(x2, y, "#upoint");
   y -= step ; Tl.DrawLatex(x1, y, "#partial")        ; Tl.DrawText(x2, y, "#partial");
   y -= step ; Tl.DrawLatex(x1, y, "#corner")         ; Tl.DrawText(x2, y, "#corner");
   y -= step ; Tl.DrawLatex(x1, y, "#ltbar")          ; Tl.DrawText(x2, y, "#ltbar");
   y -= step ; Tl.DrawLatex(x1, y, "#bottombar")      ; Tl.DrawText(x2, y, "#bottombar");
   y -= step ; Tl.DrawLatex(x1, y, "#rightarrow")     ; Tl.DrawText(x2, y, "#rightarrow");
   y -= step ; Tl.DrawLatex(x1, y, "#surd")           ; Tl.DrawText(x2, y, "#surd");
   y -= step ; Tl.DrawLatex(x1, y, "#Rightarrow")     ; Tl.DrawText(x2, y, "#Rightarrow");
   y -= step ; Tl.DrawLatex(x1, y-0.015, "#int")      ; Tl.DrawText(x2, y, "#int");
   y -= step ; Tl.DrawLatex(x1, y, "#odot")           ; Tl.DrawText(x2, y, "#odot");
   y -= step ; Tl.DrawLatex(x1, y, "#exists")         ; Tl.DrawText(x2, y, "#exists");  

   return Ms;
}
