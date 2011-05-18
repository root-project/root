void feynman()
{
   //Draw Feynman diagrams
   // To see the output of this macro, click begin_html <a href="gif/feynman.gif">here</a>. end_html
   //Author: Otto Schaile
   
   TCanvas *c1 = new TCanvas("c1", "A canvas", 10,10, 600, 300);
   c1->Range(0, 0, 140, 60);
   Int_t linsav = gStyle->GetLineWidth();
   gStyle->SetLineWidth(3);
   TLatex t;
   t.SetTextAlign(22);
   t.SetTextSize(0.1);
   TLine * l;
   l = new TLine(10, 10, 30, 30); l->Draw();
   l = new TLine(10, 50, 30, 30); l->Draw();
   TCurlyArc *ginit = new TCurlyArc(30, 30, 12.5*sqrt(2), 135, 225);
   ginit->SetWavy();
   ginit->Draw();
   t.DrawLatex(7,6,"e^{-}");
   t.DrawLatex(7,55,"e^{+}");
   t.DrawLatex(7,30,"#gamma");

   TCurlyLine *gamma = new TCurlyLine(30, 30, 55, 30);
   gamma->SetWavy();
   gamma->Draw();
   t.DrawLatex(42.5,37.7,"#gamma");

   TArc *a = new TArc(70, 30, 15);
   a->Draw();
   t.DrawLatex(55, 45,"#bar{q}");
   t.DrawLatex(85, 15,"q");
   TCurlyLine *gluon = new TCurlyLine(70, 45, 70, 15);
   gluon->Draw();
   t.DrawLatex(77.5,30,"g");

   TCurlyLine *z0 = new TCurlyLine(85, 30, 110, 30);
   z0->SetWavy();
   z0->Draw();
   t.DrawLatex(100, 37.5,"Z^{0}");

   l = new TLine(110, 30, 130, 10); l->Draw();
   l = new TLine(110, 30, 130, 50); l->Draw();

   TCurlyArc *gluon1 = new TCurlyArc(110, 30, 12.5*sqrt(2), 315, 45);
   gluon1->Draw();

   t.DrawLatex(135,6,"#bar{q}");
   t.DrawLatex(135,55,"q");
   t.DrawLatex(135,30,"g");
   c1->Update();
   gStyle->SetLineWidth(linsav);
}
