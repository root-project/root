{
   auto *Ta = new TCanvas("Ta","Ta",0,0,500,200);
   Ta->Range(0,0,1,1);

   TLine lv;
   lv.SetLineStyle(3);
   lv.SetLineColor(kBlue);
   lv.DrawLine(0.33,0.0,0.33,1.0);
   lv.DrawLine(0.6,0.165,1.,0.165);
   lv.DrawLine(0.6,0.493,1.,0.493);
   lv.DrawLine(0.6,0.823,1.,0.823);

   // Horizontal alignment.
   auto *th1 = new TText(0.33,0.165,"Left adjusted");
   th1->SetTextAlign(11); th1->SetTextSize(0.12);
   th1->Draw();

   auto *th2 = new TText(0.33,0.493,"Center adjusted");
   th2->SetTextAlign(21); th2->SetTextSize(0.12);
   th2->Draw();

   auto *th3 = new TText(0.33,0.823,"Right adjusted");
   th3->SetTextAlign(31); th3->SetTextSize(0.12);
   th3->Draw();

   // Vertical alignment.
   auto *tv1 = new TText(0.66,0.165,"Bottom adjusted");
   tv1->SetTextAlign(11); tv1->SetTextSize(0.12);
   tv1->Draw();

   auto *tv2 = new TText(0.66,0.493,"Center adjusted");
   tv2->SetTextAlign(12); tv2->SetTextSize(0.12);
   tv2->Draw();

   auto *tv3 = new TText(0.66,0.823,"Top adjusted");
   tv3->SetTextAlign(13); tv3->SetTextSize(0.12);
   tv3->Draw();
}
