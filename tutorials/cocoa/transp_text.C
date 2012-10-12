//This macro is based on labels1.C by Rene Brun.
//Updated by Timur Pocheptsov to use transparent text.


void transp_text()
{
   TCanvas *c1 = new TCanvas("c1","transparent text demo", 10, 10, 900, 500);

   //After we created a canvas, gVirtualX in principle should be initialized
   //and we can check its type:
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      std::cout<<"You can see the transparency ONLY in a pdf or png output (\"File\"->\"Save As\" ->...)\n"
                 "To have transparency in a canvas graphics, you need MacOSX version with cocoa enabled\n";
   }

   const Int_t nx = 20;
   char *people[nx] = {"Jean","Pierre","Marie","Odile",
      "Sebastien","Fons","Rene","Nicolas","Xavier","Greg",
      "Bjarne","Anton","Otto","Eddy","Peter","Pasha",
      "Philippe","Suzanne","Jeff","Valery"};
   c1->SetGrid();
   c1->SetBottomMargin(0.15);
   TH1F *h = new TH1F("h","test",nx,0,nx);
   h->SetFillColor(38);
   for (Int_t i=0;i<5000;i++) {
      h->Fill(gRandom->Gaus(0.5*nx,0.2*nx));
   }
   h->SetStats(0);
   for (i=1;i<=nx;i++) {
      h->GetXaxis()->SetBinLabel(i,people[i-1]);
   }
   h->Draw();
   
   TPaveText *pt = new TPaveText(0.3,0.3,0.98,0.98,"brNDC");
   
   //Create special transparent colors for both pavetext fill color and text color.
   new TColor(1001, 0.8, 0.8, 0.8, "transparent_gray", 0.85);
   pt->SetFillColor(1001);
   //Add new color with index 1002.
   new TColor(1002, 0., 0., 0., "transparent_black", 0.5);
   pt->SetTextColor(1002);
   pt->SetTextSize(0.5);
   pt->SetTextAlign(12);
   
   pt->AddText("Hello");
   pt->Draw();
}
