// Open a FITS file whose primary array represents
// a spectrum (flux vs wavelength)
void FITS_tutorial2()
{
   printf("\n\n--------------------------------\n");
   printf("WELCOME TO FITS tutorial #2 !!!!\n");
   printf("--------------------------------\n");
   printf("We're gonna open a FITS file that contains the\n");
   printf("primary HDU and a little data table.\n");
   printf("The primary HDU is an array of 2 rows by 2040 columns, and\n");
   printf("they represent a radiation spectrum. The first row contains\n");
   printf("the flux data, whereas the second row the wavelengths.\n");
   printf("Data copyright: NASA\n\n");

   if (!gROOT->IsBatch()) {
      //printf("Press ENTER to start..."); getchar();
   }
   TString dir = gSystem->DirName(__FILE__);

   // Open primary HDU from file
   TFITSHDU *hdu = new TFITSHDU(dir+"/sample2.fits");
   if (hdu == 0) {
      printf("ERROR: could not access the HDU\n"); return;
   }
   printf("File successfully open!\n");


   // Dump the HDUs within the FITS file
   // and also their metadata
   //printf("Press ENTER to see summary of all data stored in the file:"); getchar();
   hdu->Print("F+");

   printf("....................................\n");
   printf("We are going to generate a TGraph from vectors\n");
   //printf("within the primary array. Press ENTER to continue.."); getchar();

   TVectorD *Y = hdu->GetArrayRow(0);
   TVectorD *X = hdu->GetArrayRow(1);
   TGraph *gr = new TGraph(*X,*Y);

   // Show the graphic
   TCanvas *c = new TCanvas("c1", "FITS tutorial #2", 800, 800);
   gr->Draw("BA");


   // Clean up
   delete X;
   delete Y;
   delete hdu;
}


