{
   gROOT->Reset();
   TCanvas *c1 = gROOT->FindObject("c1"); if (c1) c1->Delete();
   c1 = new TCanvas("c1","I/O strategies",20,10,750,930);
   c1->SetBorderSize(0);
   c1->Range(0,0,20.5,26);


   TPaveText title(2,22,18,25.5);
   title.SetFillColor(10);
   title.AddText(" ");
   TText *ser = title.AddText("We need a flexible I/O mechanism");
   ser.SetTextSize(0.04);
   ser.SetTextAlign(21);
   TText *t1 = title.AddText("From the same Object Model, one must be able to select");
   t1.SetTextSize(0.022);
   t1.SetTextAlign(11);
   t1.SetTextFont(72);
   TText *t2 = title.AddText("different storage models (performance)");
   t2.SetTextSize(0.022);
   t2.SetTextAlign(11);
   t2.SetTextFont(72);
   title.Draw();

   TPaveText event(1,16,20,21);
   event.SetTextAlign(12);
   event.SetFillColor(10);
   event.SetTextFont(82);
   event.SetTextSize(0.015);
   event.ReadFile("event.h");
   event.Draw();

   TPaveText m1(4,13,19,15);
   m1.SetTextSize(0.024);
   m1.SetTextFont(72);
   m1.SetFillColor(10);
   TText *t1 = m1.AddText("One single container (old raw data model)");
   t1.SetTextFont(62);
   t1.SetTextAlign(22);
   m1.AddText("All event components are serialized in the same buffer");
   m1.Draw();

   TPaveText m2(4,8,19,12);
   m2.SetTextAlign(12);
   m2.SetTextFont(72);
   m2.SetTextSize(0.025);
   m2.SetFillColor(10);
   TText *tm1 = m2.AddText("A few containers: (DST analysis)");
   tm1.SetTextFont(62);
   tm1.SetTextAlign(22);
   tm1.SetTextSize(0.03);
   m2.AddText("- Header");
   m2.AddText("- Tracks");
   m2.AddText("- Calorimeters");
   m2.AddText("- Histograms");
   m2.Draw();

   TPaveText m3(4,1,19,7);
   m3.SetTextAlign(12);
   m3.SetTextFont(72);
   m3.SetTextSize(0.025);
   m3.SetFillColor(10);
   TText *tm3 = m3.AddText("Many containers (like PAW ntuples):");
   tm3.SetTextFont(62);
   tm3.SetTextAlign(22);
   tm3.SetTextSize(0.03);
   m3.AddText("- Each Data Member has its container");
   m3.AddText("- Special classes (ClonesArray) for large collections");
   m3.AddText("- One event-histogram per container");
   m3.AddText("- Good for Physic Analysis stage");
   m3.Draw();

   TText arrow;
   arrow.SetTextFont(142);
   arrow.SetTextSize(0.07);
   arrow.DrawText(1.5,13.5,"~\375");
   arrow.DrawText(1.5,10,"~\375");
   arrow.DrawText(1.5, 5,"~\375");

   c1.Print("io.ps");
}
