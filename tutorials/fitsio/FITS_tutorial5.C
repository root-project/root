// Open a FITS file whose primary array represents
// a spectrum (flux vs wavelength)
void FITS_tutorial5()
{
   TVectorD *v;
   
   printf("\n\n--------------------------------\n");
   printf("WELCOME TO FITS tutorial #5 !!!!\n");
   printf("--------------------------------\n");
   printf("We're gonna open a FITS file that contains a\n");
   printf("table with 9 rows and 8 columns. Column 4 has name\n");
   printf("'mag' and contains a vector of 6 numeric components.\n");
   printf("The values of vectors in rows 1 and 2 (column 4) are:\n");
   printf("Row1: (99.0, 24.768, 23.215, 21.68, 21.076, 20.857)\n");
   printf("Row2: (99.0, 21.689, 20.206, 18.86, 18.32 , 18.128 )\n");
   printf("WARNING: when coding, row and column indices start from 0\n");
   
   if (!gROOT->IsBatch()) {
      //printf("Press ENTER to start..."); getchar();
      //printf("\n");
   }
    
   //Open the table
   TFITSHDU *hdu = new TFITSHDU("sample4.fits[1]");
   if (hdu == 0) {
      printf("ERROR: could not access the HDU\n"); return;
   }
  
   
   //Read vectors at rows 1 and 2 (indices 0 and 1)
   TVectorD *vecs[2];
   vecs[0] = hdu->GetTabRealVectorCell(0, "mag");
   vecs[1] = hdu->GetTabRealVectorCell(1, "mag");
   for (int iVec=0; iVec < 2; iVec++) {
      printf("Vector %d = (", iVec+1);
      v = vecs[iVec]; 
      for(int i=0; i < v->GetNoElements(); i++) {
         if (i>0) printf(", ");
         printf("%lg", (*v)[i]); //NOTE: the asterisk is for using the overloaded [] operator of the TVectorD object
      }
      printf(")\n");
   }
   
   printf("\nBONUS EXAMPLE: we're gonna dump all rows using\n");
   printf("the function GetTabRealVectorCells()\n");
   //printf("Press ENTER to continue..."); getchar();
   
   TObjArray *vectorCollection = hdu->GetTabRealVectorCells("mag");
   
   for (int iVec=0; iVec < vectorCollection->GetEntriesFast(); iVec++) {
      printf("Vector %d = (", iVec+1);
      v = (TVectorD *) (*vectorCollection)[iVec]; //NOTE: the asterisk is for using the overloaded [] operator of the TObjArray object
      for(int i=0; i < v->GetNoElements(); i++) {
         if (i>0) printf(", ");
         printf("%lg", (*v)[i]); //NOTE: the asterisk is for using the overloaded [] operator of the TVectorD object
      }
      printf(")\n");
   }
   
   
   //Clean up
   delete vecs[0];
   delete vecs[1];
   delete vectorCollection;  
   delete hdu;
}

 
