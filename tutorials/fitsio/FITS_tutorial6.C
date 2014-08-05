// Open a FITS file whose primary array represents
// a spectrum table (flux vs wavelength) and dump its columns
void FITS_tutorial6()
{
   TVectorD *v;

   printf("\n\n--------------------------------\n");
   printf("WELCOME TO FITS tutorial #6 !!!!\n");
   printf("--------------------------------\n");
   printf("We are going to open a table from a FITS file\n");
   printf("and dump its columns.\n\n");

   TString dir = gSystem->DirName(__FILE__);

   //Open the table
   TFITSHDU *hdu = new TFITSHDU(dir+"/sample4.fits[1]");
   if (hdu == 0) {
      printf("ERROR: could not access the HDU\n"); return;
   }

   //Show columns
   Int_t nColumns = hdu->GetTabNColumns();
   printf("The table has %d columns:\n", nColumns);
   for (Int_t i = 0; i < nColumns; i++) {
      printf("...Column %d: %s\n", i, hdu->GetColumnName(i).Data());
   }
   puts("");

   delete hdu;
}


