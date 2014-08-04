// Open a FITS file and retrieve the first plane of the image array
// as a TImage object
void FITS_tutorial1()
{
   printf("\n\n--------------------------------\n");
   printf("WELCOME TO FITS tutorial #1 !!!!\n");
   printf("--------------------------------\n");
   printf("We're gonna open a FITS file that contains only the\n");
   printf("primary HDU, consisting on an image.\n");
   printf("The object you will see is a snapshot of the NGC7662 nebula,\n");
   printf("which was taken by the author on November 2009 in Barcelona (CATALONIA).\n\n");

   if (!gROOT->IsBatch()) {
      //printf("Press ENTER to start..."); getchar();
   }
   TString dir = gSystem->DirName(__FILE__);

   // Open primary HDU from file
   TFITSHDU *hdu = new TFITSHDU(dir+"/sample1.fits");
   if (hdu == 0) {
      printf("ERROR: could not access the HDU\n"); return;
   }
   printf("File successfully open!\n");

   // Dump the HDUs within the FITS file
   // and also their metadata
   //printf("Press ENTER to see summary of all data stored in the file:"); getchar();

   hdu->Print("F+");

   printf("....................................\n");
   // Here we get the exposure time.
   //printf("Press ENTER to retrieve the exposure time from the HDU metadata..."); getchar();
   printf("Exposure time = %s\n", hdu->GetKeywordValue("EXPTIME").Data());


   // Read the primary array as a matrix,
   // selecting only layer 0.
   // This function may be useful to
   // do image processing.
   printf("....................................\n");
   printf("We can read the image as a matrix of values.\n");
   printf("This feature is useful to do image processing, e.g:\n");
   printf("histogram equalization, custom filtering, ...\n");
   //printf("Press ENTER to continue..."); getchar();

   TMatrixD *mat = hdu->ReadAsMatrix(0);
   mat->Print();
   delete mat;

   // Read the primary array as an image,
   // selecting only layer 0.
   printf("....................................\n");
   printf("Now the primary array will be read both as an image and as a histogram,\n");
   printf("and they will be shown in a canvas.\n");
   //printf("Press ENTER to continue..."); getchar();

   TImage *im = hdu->ReadAsImage(0);

   // Read the primary array as a histogram.
   // Depending on array dimensions, returned
   // histogram will be 1D, 2D or 3D
   TH1 *hist = hdu->ReadAsHistogram();


   TCanvas *c = new TCanvas("c1", "FITS tutorial #1", 800, 300);
   c->Divide(2,1);
   c->cd(1);
   im->Draw();
   c->cd(2);
   hist->Draw("COL");

   // Clean up
   delete hdu;
}


