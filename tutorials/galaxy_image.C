void galaxy_image()
{
   TCanvas *canv = new TCanvas("image", "n4254", 40, 40, 812, 700);
   canv->ToggleEventStatus();
   canv->SetRightMargin(0.2);
   canv->SetLeftMargin(0.01);
   canv->SetTopMargin(0.01);
   canv->SetBottomMargin(0.01);

   // read the pixel data from file "galaxy.root"
   // the size of the image is 401 X 401 pixels
   TFile *gal = new TFile("galaxy.root", "READ");
   TImage *img = (TImage*)gal->Get("n4254");
   img->Draw();

   // open the color editor
   img->StartPaletteEditor();

   // zoom the image
   img->Zoom(80, 80, 250, 250);
}
