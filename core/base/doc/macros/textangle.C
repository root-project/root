{
   TCanvas *Ta = new TCanvas("Ta","Text angle",0,0,300,326);
   Ta->Range(0,0,1,1);
   TLine *l = new TLine();
   l->SetLineColor(kRed);
   l->DrawLine(0.1,0.1,0.9,0.1);
   l->DrawLine(0.1,0.1,0.9,0.9);
   TMarker *m = new TMarker();
   m->SetMarkerStyle(20);
   m->SetMarkerColor(kBlue);
   m->DrawMarker(0.1,0.1);
   TArc *a = new TArc();
   a->SetFillStyle(0);
   a->SetLineColor(kBlue); a->SetLineStyle(3);
   a->DrawArc(0.1, 0.1, 0.2, 0.,45.,"only");
   TText *tt = new TText(0.1,0.1,"Text angle is 45 degrees");
   tt->SetTextAlign(11); tt->SetTextSize(0.1);
   tt->SetTextAngle(45);
   tt->Draw();
   TLatex *t1 = new TLatex(0.3,0.18,"45^{o}");
   t1->Draw();
   return Ta;
}
