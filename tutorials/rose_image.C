void rose_image()
{
   // Display image in a new canvas and pad.

   TImage *img = TImage::Open("rose512.jpg");
   if (!img) {
      printf("Could not create an image... exit\n");
      return;
   }

   img->SetConstRatio(0);
   img->SetImageQuality(TAttImage::kImgBest);
   img->Draw("N");
}
