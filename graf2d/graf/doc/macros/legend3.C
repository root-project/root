{
   TCanvas *c3 = new TCanvas("c2","c2",500,300);

   TLegend* leg = new TLegend(0.2, 0.2, .8, .8);
   TH1* h = new TH1F("", "", 1, 0, 1);

   leg-> SetNColumns(2);

   leg->AddEntry(h, "Column 1 line 1", "l");
   leg->AddEntry(h, "Column 2 line 1", "l");
   leg->AddEntry(h, "Column 1 line 2", "l");
   leg->AddEntry(h, "Column 2 line 2", "l");

   leg->Draw();
   return c3;
}

