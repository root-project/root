/// \file
/// \ingroup tutorial_FITS
/// \notebook
///
/// Open a FITS file with columns containing variable-length arrays. 
/// 
/// ABOUT THE DATA: To illustrate such a case we use a Redistribution 
/// Matrix File (RMF) conform with the HEASARC OGIP specifications
/// OGIP definition
/// https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/ogip_92_007.html
/// RMF defintion
/// https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc3.1
///
/// Variable-length arrays are used to return the non-null row elements of a 
/// sparse matrix representing the PDF of an energy estimator. 
///
/// Such a format is a standard in spectral data for X-ray astronomy instruments
/// and recently has been adopted by gamma-ray astronomy instruments.
/// The file we use indeed represents the migration matrix of the MAGIC 
/// gamma-ray telescope and is taken from the publicly available data of the 
/// following publication 
/// https://ui.adsabs.harvard.edu/abs/2019A%26A...625A..10N/abstract
/// The file, included here as an example, is available in the github repository
/// associated with the publication
/// https://github.com/open-gamma-ray-astro/joint-crab
/// under this path 
/// https://github.com/open-gamma-ray-astro/joint-crab/blob/master/results/spectra/magic/rmf_obs5029747.fits
/// 
/// \macro_code
/// \macro_output
///
/// \author Cosimo Nigro

void FITS_tutorial8()
{ 
   // let us read the first header data unit, opening the file with a FITS viewer
   // we see that the following cells contain variable-length arrays with values 
   // row 10 column "F_CHAN": (14, 16)
   // row 10 column "N_CHAN": (1, 1)
   // row 10 column "MATRIX" : (0.49981028, 0.5001897)
   TFITSHDU* hdu = new TFITSHDU("rmf_obs5029747.fits", 1);
   int rownum = 9; // FITS tables are indexed starting from 1
   TString colname1 = "F_CHAN";
   TString colname2 = "N_CHAN";
   TString colname3 = "MATRIX";
   
   printf("reading in row %d, column %s \n", rownum+1, colname1.Data());
   TArrayD *arr1 = hdu->GetTabVarLengthVectorCell(rownum, colname1);
   printf("(%f, %f) \n", arr1->At(0), arr1->At(1)); 
   
   printf("reading in row %d, column %s \n", rownum+1, colname2.Data());
   TArrayD *arr2 = hdu->GetTabVarLengthVectorCell(rownum, colname2);
   printf("(%f, %f) \n", arr2->At(0), arr2->At(1)); 
   
   printf("reading in row %d, column %s \n", rownum+1, colname3.Data());
   TArrayD *arr3 = hdu->GetTabVarLengthVectorCell(rownum, colname3);
   printf("(%f, %f) \n", arr3->At(0), arr3->At(1)); 

}