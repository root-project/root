// Open a FITS file and retrieve the first plane of the image array
// as a TImage object
void FITS_tutorial3()
{
   printf("\n\n--------------------------------\n");
   printf("WELCOME TO FITS tutorial #3 !!!!\n");
   printf("--------------------------------\n");
   printf("We're gonna open a FITS file that contains several image\n");
   printf("extensions. The primary HDU contains no data.\n");
   printf("Data copyright: NASA\n\n");

   if (!gROOT->IsBatch()) {
      //printf("Press ENTER to start..."); getchar();
   }

   // Open extensions 1 to 5 from file
   //printf("Press ENTER to see a canvas with all images within the file:"); getchar();
   TString dir = gSystem->DirName(__FILE__);

   TCanvas *c = new TCanvas("c1", "FITS tutorial #1", 800, 700);
   c->Divide(2,3);
   for (int i=1; i <= 5; i++) {
      TFITSHDU *hdu = new TFITSHDU(dir+"/sample3.fits", i);
      if (hdu == 0) {
         printf("ERROR: could not access the HDU\n"); return;
      }

      TImage *im = hdu->ReadAsImage(0);
      c->cd(i);
      im->Draw();
      delete hdu;
   }
}


