// Open a FITS file whose primary array represents
// a spectrum (flux vs wavelength)
void FITS_tutorial4()
{
   printf("\n\n--------------------------------\n");
   printf("WELCOME TO FITS tutorial #4 !!!!\n");
   printf("--------------------------------\n");
   printf("We're gonna open a FITS file that contains the\n");
   printf("primary HDU and a little data table.\n");
   printf("The data table is extension #1 and it has 2 rows.\n");
   printf("We want to read only the rows that have the column\n");
   printf("named DATAMAX greater than 2e-15 (there's only 1\n");
   printf("matching row)\n");
   printf("Data copyright: NASA\n\n");

   if (!gROOT->IsBatch()) {
      //printf("Press ENTER to start..."); getchar();
   }
   TString dir = gSystem->DirName(__FILE__);

   //Open the table extension number 1)
   TFITSHDU *hdu = new TFITSHDU(dir+"/sample2.fits[1][DATAMAX > 2e-15]");
   if (hdu == 0) {
      printf("ERROR: could not access the HDU\n"); return;
   }
   //printf("Press ENTER to see information about the table's columns..."); getchar();
   hdu->Print("T");

   printf("\n\n........................................\n");
   printf("Press ENTER to see full table contents (maybe you should resize\n");
   //printf("this window as large as possible before)..."); getchar();
   hdu->Print("T+");

   printf("\n\n........................................\n");
   //printf("Press ENTER to get only the DATAMAX value of the matched row..."); getchar();
   TVectorD *v = hdu->GetTabRealVectorColumn("DATAMAX");
   printf("%lg\n", (*v)[0]);


   printf("Does the matched row have DATAMAX > 2e-15? :-)\n");

   //Clean up
   delete v;
   delete hdu;
}


