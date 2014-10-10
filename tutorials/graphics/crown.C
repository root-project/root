//Draw crowns
//Author: Olivier Couet
TCanvas *crown(){
   TCanvas *c1 = new TCanvas("c1","c1",400,400);
   TCrown *cr1 = new TCrown(.5,.5,.3,.4);
   cr1->SetLineStyle(2);
   cr1->SetLineWidth(4);
   cr1->Draw();
   TCrown *cr2 = new TCrown(.5,.5,.2,.3,45,315);
   cr2->SetFillColor(38);
   cr2->SetFillStyle(3010);
   cr2->Draw();
   TCrown *cr3 = new TCrown(.5,.5,.2,.3,-45,45);
   cr3->SetFillColor(50);
   cr3->SetFillStyle(3025);
   cr3->Draw();
   TCrown *cr4 = new TCrown(.5,.5,.0,.2);
   cr4->SetFillColor(4);
   cr4->SetFillStyle(3008);
   cr4->Draw();
   return c1;
}
