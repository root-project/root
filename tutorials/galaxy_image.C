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
   TVectorD *data = (TVectorD*)gal->Get("galaxy");
   delete gal;

   // read a color palette, it was written to the file via the
   // color editor (Save button)
   TFile *fpal = new TFile("galaxy.pal.root", "READ");
   TImagePalette *palette = (TImagePalette*)fpal->Get("TImagePalette");
   delete fpal;

   // create an image and set the pixel data and the color palette
   TImage *img = TImage::Create();
   if (!img) {
      printf("Could not create an image... exit\n");
      return;
   }
   img->SetImage(*data, 401, palette);
   delete palette;

   img->Draw();

   // open the color editor
   img->StartPaletteEditor();

   // zoom the image
   img->Zoom(80, 80, 250, 250);
}
