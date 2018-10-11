/// \file
/// \ingroup tutorial_FITS
/// Open a FITS file that contains a catalog of astronomical objects
/// and dump some of its columns
///
/// \macro_code
///
/// \author Elizabeth Buckley-Geer

void FITS_tutorial7()
{

   printf("\n\n--------------------------------\n");
   printf("WELCOME TO FITS tutorial #7 !!!!\n");
   printf("--------------------------------\n");
   printf("We are going to open a table from a FITS file\n");
   printf("and print out three columns for some of the objects.\n");
   printf("This table contains a logical data type so this tutorial tests\n");
   printf("that we can read it correctly\n\n");

   TString dir = gSystem->DirName(__FILE__);

   // Open the table
   TFITSHDU *hdu = new TFITSHDU(dir + "/sample5.fits[1]");
   if (hdu == 0) {
      printf("ERROR: could not access the HDU\n");
      return;
   }

   TVectorD *vec1;
   TVectorD *vec2;
   TVectorD *vec3;
   TVectorD *vec4;

   // Read the ra, dec, flux_g and brick_primary columns

   vec1 = hdu->GetTabRealVectorColumn("ra");
   vec2 = hdu->GetTabRealVectorColumn("dec");
   vec3 = hdu->GetTabRealVectorColumn("flux_g");
   vec4 = hdu->GetTabRealVectorColumn("brick_primary");

   Double_t gflux, ra, dec, bp;

   for (Int_t i = vec1->GetLwb(); i <= vec1->GetUpb(); i++) {

      bp = (*vec4)[i];
      gflux = (*vec3)[i];
      ra = (*vec1)[i];
      dec = (*vec2)[i];

      if (bp) {
         printf("RA %f DEC %f G-FLUX %f\n", ra, dec, gflux);
      }
   }

   // Clean up

   delete vec1;
   delete vec2;
   delete vec3;
   delete vec4;
   delete hdu;
}