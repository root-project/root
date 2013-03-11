void transp()
{
   //This demo shows, how to use transparency.
   //On MacOS X you can see the transparency in a canvas,
   //you can save canvas contents as pdf/png
   //(and thus you'll have an image with transparency on every platform.

   //Unfortunately, these transparent colors can
   //not be saved with a histogram object in a file,
   //since ROOT just save color indices and our transparent
   //colors were created/added "on the fly".

   TCanvas * cnv = new TCanvas("trasnparency", "transparency demo", 600, 400);
   
   //After we created a canvas, gVirtualX in principle should be initialized
   //and we can check its type:
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      std::cout<<"You can see the transparency ONLY in a pdf or png output (\"File\"->\"Save As\" ->...)\n"
                 "To have transparency in a canvas graphics, you need MacOSX version with cocoa enabled\n";
   }

   TH1F * hist = new TH1F("a", "b", 10, -2., 3.);
   TH1F * hist2 = new TH1F("c", "d", 10, -3., 3.);
   hist->FillRandom("landau", 100000);
   hist2->FillRandom("gaus", 100000);

   //Add new color with index 1001.
   new TColor(1001, 1., 0., 0., "red", 0.85);
   hist->SetFillColor(1001);
   
   //Add new color with index 1002.
   new TColor(1002, 0., 1., 0., "green", 0.5);
   hist2->SetFillColor(1002);
   
   cnv->cd();
   hist2->Draw();
   hist->Draw("SAME");
}