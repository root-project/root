// @(#)root/graf2d:$Id$
// Author:  Claudi Martinez, July 19th 2010

/*************************************************************************
 * Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// IMPLEMENTATION NOTE:
// CFITSIO library uses standard C types ('int', 'long', ...). Since these
// types may depend on the compiler machine (in 32-bit CPU 'int' and 'long' are 32-bit),
// we use the standard C types too. Using types as Long_t (which is defined to be 64-bit),
// may lead to type size mismatch.

//______________________________________________________________________________
/* Begin_Html
<center><h2>FITS file interface class</h2></center>
TFITS is a class that allows extracting images and data from FITS files and contains
several methods to manage them.
End_Html */

#include "TFITS.h"
#include "TROOT.h"
#include "TASImage.h"
#include "TArrayI.h"
#include "TArrayD.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TH3D.h"
#include "TVectorD.h"
#include "TMatrixD.h"

#include "fitsio2.h"
#include <stdlib.h>


ClassImp(TFITSHDU)

/**************************************************************************/
// TFITSHDU
/**************************************************************************/

//______________________________________________________________________________
void TFITSHDU::CleanFilePath(const char *filepath_with_filter, TString &dst)
{
   // Clean path from possible filter and put the result in 'dst'.

   dst = filepath_with_filter;

   Ssiz_t ndx = dst.Index("[", 1, 0, TString::kExact);
   if (ndx != kNPOS) {
      dst.Resize(ndx);
   }
}


//______________________________________________________________________________
TFITSHDU::TFITSHDU(const char *filepath_with_filter)
{
   // TFITSHDU constructor from file path with HDU selection filter.

   _initialize_me();
   TString finalpath = filepath_with_filter;
   CleanFilePath(filepath_with_filter, fCleanFilePath);

   if (kFALSE == LoadHDU(finalpath)) {
      _release_resources();
      throw -1;
   }
}

//______________________________________________________________________________
TFITSHDU::TFITSHDU(const char *filepath, Int_t extension_number)
{
   // TFITSHDU constructor from filepath and extension number.

   _initialize_me();
   CleanFilePath(filepath, fCleanFilePath);

   //Add "by extension number" filter
   TString finalpath;
   finalpath.Form("%s[%d]", fCleanFilePath.Data(), extension_number);

   if (kFALSE == LoadHDU(finalpath)) {
      _release_resources();
      throw -1;
   }
}

//______________________________________________________________________________
TFITSHDU::TFITSHDU(const char *filepath, const char *extension_name)
{
   // TFITSHDU constructor from filepath and extension name.

   _initialize_me();
   CleanFilePath(filepath, fCleanFilePath);

   //Add "by extension number" filter
   TString finalpath;
   finalpath.Form("%s[%s]", fCleanFilePath.Data(), extension_name);


   if (kFALSE == LoadHDU(finalpath)) {
      _release_resources();
      throw -1;
   }
}

//______________________________________________________________________________
TFITSHDU::~TFITSHDU()
{
   // TFITSHDU destructor.

   _release_resources();
}

//______________________________________________________________________________
void TFITSHDU::_release_resources()
{
   // Release internal resources.

   if (fRecords) delete [] fRecords;

   if (fType == kImageHDU) {
      if (fSizes) delete fSizes;
      if (fPixels) delete fPixels;
   } else {
      //TODO
   }
}

//______________________________________________________________________________
void TFITSHDU::_initialize_me()
{
   // Do some initializations.

   fRecords = 0;
   fPixels = 0;
   fSizes = 0;
}

//______________________________________________________________________________
Bool_t TFITSHDU::LoadHDU(TString& filepath_filter)
{
   // Load HDU from fits file satisfying the specified filter.
   // Returns kTRUE if success. Otherwise kFALSE.
   // If filter == "" then the primary array is selected

   fitsfile *fp=0;
   int status = 0;
   char errdescr[FLEN_STATUS+1];

   // Open file with filter
   fits_open_file(&fp, filepath_filter.Data(), READONLY, &status);
   if (status) goto ERR;

   // Read HDU number
   int hdunum;
   fits_get_hdu_num(fp, &hdunum);
   fNumber = Int_t(hdunum);

   // Read HDU type
   int hdutype;
   fits_get_hdu_type(fp, &hdutype, &status);
   if (status) goto ERR;
   fType = (hdutype == IMAGE_HDU) ? kImageHDU : kTableHDU;

   //Read HDU header records
   int nkeys, morekeys;
   char keyname[FLEN_KEYWORD+1];
   char keyvalue[FLEN_VALUE+1];
   char comment[FLEN_COMMENT+1];

   fits_get_hdrspace(fp, &nkeys, &morekeys, &status);
   if (status) goto ERR;

   fRecords = new struct HDURecord[nkeys];

   for (int i = 1; i <= nkeys; i++) {
      fits_read_keyn(fp, i, keyname, keyvalue, comment, &status);
      if (status) goto ERR;
      fRecords[i-1].fKeyword = keyname;
      fRecords[i-1].fValue = keyvalue;
      fRecords[i-1].fComment = comment;
   }

   fNRecords = Int_t(nkeys);

   //Set extension name
   fExtensionName = "PRIMARY"; //Default
   for (int i = 0; i < nkeys; i++) {
      if (fRecords[i].fKeyword == "EXTNAME") {
         fExtensionName = fRecords[i].fValue;
         break;
      }
   }

   //Read HDU's data
   if (fType == kImageHDU) {
      //Image
      int param_ndims=0;
      long *param_dimsizes;

      //Read image number of dimensions
      fits_get_img_dim(fp, &param_ndims, &status);
      if (status) goto ERR;
      if (param_ndims > 0) {
         //Read image sizes in each dimension
         param_dimsizes = new long[param_ndims];
         fits_get_img_size(fp, param_ndims, param_dimsizes, &status);
         if (status) goto ERR;

         fSizes = new TArrayI(param_ndims);
         fSizes = new TArrayI(param_ndims);
         for (int i = 0; i < param_ndims; i++) { //Use for loop to copy values instead of passing array to constructor, since 'Int_t' size may differ from 'long' size
            fSizes->SetAt(param_dimsizes[i], i);
         }

         delete [] param_dimsizes;

         //Read pixels
         int anynul;
         long *firstpixel = new long[param_ndims];
         double nulval = 0;
         long npixels = 1;

         for (int i = 0; i < param_ndims; i++) {
            npixels *= (long) fSizes->GetAt(i); //Compute total number of pixels
            firstpixel[i] = 1; //Set first pixel to read from.
         }

         double *pixels = new double[npixels];

         fits_read_pix(fp, TDOUBLE, firstpixel, npixels,
                     (void *) &nulval, (void *) pixels, &anynul, &status);

         if (status) {
            delete [] firstpixel;
            delete [] pixels;
            goto ERR;
         }

         fPixels = new TArrayD(npixels, pixels);

         delete [] firstpixel;
         delete [] pixels;

      } else {
         //Null array
         fSizes = new TArrayI();
         fPixels = new TArrayD();
      }
   } else {
      //Table
      //TODO

   }

   // Close file
   fits_close_file(fp, &status);
   return kTRUE;

ERR:
   fits_get_errstatus(status, errdescr);
   Warning("LoadHDU", "error opening FITS file. Details: %s", errdescr);
   status = 0;
   if (fp) fits_close_file(fp, &status);
   return kFALSE;
}

//______________________________________________________________________________
struct TFITSHDU::HDURecord* TFITSHDU::GetRecord(const char *keyword)
{
   // Get record by keyword.

   for (int i = 0; i < fNRecords; i++) {
      if (fRecords[i].fKeyword == keyword) {
         return &fRecords[i];
      }
   }
   return 0;
}

//______________________________________________________________________________
TString& TFITSHDU::GetKeywordValue(const char *keyword)
{
   // Get the value of a given keyword. Return "" if not found.

   HDURecord *rec = GetRecord(keyword);
   if (rec) {
      return rec->fValue;
   } else {
      return *(new TString(""));
   }
}

//______________________________________________________________________________
void TFITSHDU::PrintHDUMetadata(const Option_t *) const
{
   // Print records.

   for (int i = 0; i < fNRecords; i++) {
      if (fRecords[i].fComment.Length() > 0) {
         printf("%-10s = %s / %s\n", fRecords[i].fKeyword.Data(), fRecords[i].fValue.Data(), fRecords[i].fComment.Data());
      } else {
         printf("%-10s = %s\n", fRecords[i].fKeyword.Data(), fRecords[i].fValue.Data());
      }
   }
}

//______________________________________________________________________________
void TFITSHDU::PrintFileMetadata(const Option_t *opt) const
{
   // Print HDU's parent file's metadata.

   fitsfile *fp=0;
   int status = 0;
   char errdescr[FLEN_STATUS+1];
   int hducount, extnum;
   int hdutype = IMAGE_HDU;
   const char *exttype;
   char extname[FLEN_CARD]="PRIMARY"; //room enough
   int verbose = (opt[0] ? 1 : 0);

   // Open file with no filters: current HDU will be the primary one.
   fits_open_file(&fp, fCleanFilePath.Data(), READONLY, &status);
   if (status) goto ERR;

   // Read HDU count
   fits_get_num_hdus(fp, &hducount, &status);
   if (status) goto ERR;
   printf("Total: %d HDUs\n", hducount);

   extnum = 0;
   while(hducount) {
      // Read HDU type
      fits_get_hdu_type(fp, &hdutype, &status);
      if (status) goto ERR;

      if (hdutype == IMAGE_HDU) {
         exttype="IMAGE";
      } else if (hdutype == ASCII_TBL) {
         exttype="ASCII TABLE";
      } else {
         exttype="BINARY TABLE";
      }

      //Read HDU header records
      int nkeys, morekeys;
      char keyname[FLEN_KEYWORD+1];
      char keyvalue[FLEN_VALUE+1];
      char comment[FLEN_COMMENT+1];

      fits_get_hdrspace(fp, &nkeys, &morekeys, &status);
      if (status) goto ERR;

      struct HDURecord *records = new struct HDURecord[nkeys];

      for (int i = 1; i <= nkeys; i++) {
         fits_read_keyn(fp, i, keyname, keyvalue, comment, &status);
         if (status) {
            delete [] records;
            goto ERR;
         }

         records[i-1].fKeyword = keyname;
         records[i-1].fValue = keyvalue;
         records[i-1].fComment = comment;

         if (strcmp(keyname, "EXTNAME") == 0) {
            //Extension name
            strcpy(extname, keyvalue);
         }
      }

      //HDU info
      printf("   [%d] %s (%s)\n", extnum, exttype, extname);

      //HDU records
      if (verbose) {
         for (int i = 0; i < nkeys; i++) {
            if (comment[0]) {
               printf("      %-10s = %s / %s\n", records[i].fKeyword.Data(), records[i].fValue.Data(), records[i].fComment.Data());
            } else {
               printf("      %-10s = %s\n", records[i].fKeyword.Data(), records[i].fValue.Data());
            }
         }
      }
      printf("\n");

      delete [] records;

      hducount--;
      extnum++;
      if (hducount){
         fits_movrel_hdu(fp, 1, &hdutype, &status);
         if (status) goto ERR;
      }
   }

   // Close file
   fits_close_file(fp, &status);
   return;

ERR:
   fits_get_errstatus(status, errdescr);
   Warning("PrintFileMetadata", "error opening FITS file. Details: %s", errdescr);
   status = 0;
   if (fp) fits_close_file(fp, &status);
}

//______________________________________________________________________________
void TFITSHDU::Print(const Option_t *opt) const
{
   // Print metadata.
   // Currently supported options:
   // ""  :  print HDU record data
   // "F" :  print FITS file's extension names, numbers and types
   // "F+":  print FITS file's extension names and types and their record data

   if ((opt[0] == 'F') || (opt[0] == 'f')) {
      PrintFileMetadata((opt[1] == '+') ? "+" : "");
   } else {
      PrintHDUMetadata("");
   }
}


//______________________________________________________________________________
TASImage *TFITSHDU::ReadAsImage(Int_t layer, TImagePalette *pal)
{
   // Read image HDU as a displayable image. Return 0 if conversion cannot be done.
   // If the HDU seems to be a multilayer image, 'layer' parameter can be used
   // to retrieve the specified layer (starting from 0)

   if (fType != kImageHDU) {
      Warning("ReadAsImage", "this is not an image HDU.");
      return 0;
   }

   if ((fSizes->GetSize() != 2) && (fSizes->GetSize() != 3)) {
      Warning("ReadAsImage", "could not convert image HDU to image because it has %d dimensions.", fSizes->GetSize());
      return 0;
   }

   Int_t width, height;
   UInt_t pixels_per_layer;

   width  = Int_t(fSizes->GetAt(0));
   height = Int_t(fSizes->GetAt(1));

   pixels_per_layer = UInt_t(width) * UInt_t(height);

   if (((fSizes->GetSize() == 2) && (layer > 0)) || ((fSizes->GetSize() == 3) && (layer > fSizes->GetAt(2)))) {
      Warning("ReadAsImage", "layer out of bounds.");
      return 0;
   }

   // Get the maximum and minimum pixel values in the layer to auto-stretch pixels
   Double_t maxval = 0, minval = 0;
   register UInt_t i;
   Double_t pixvalue;
   Int_t offset = layer * pixels_per_layer;

   for (i = 0; i < pixels_per_layer; i++) {
      pixvalue = fPixels->GetAt(offset + i);

      if (pixvalue > maxval) {
         maxval = pixvalue;
      }

      if ((i == 0) || (pixvalue < minval)) {
         minval = pixvalue;
      }
   }

   //Build the image stretching pixels into a range from 0.0 to 255.0
   TASImage *im = new TASImage(width, height);
   TArrayD *layer_pixels = new TArrayD(pixels_per_layer);

   Double_t factor = 255.0 / (maxval-minval);
   for (i = 0; i < pixels_per_layer; i++) {
      pixvalue = fPixels->GetAt(offset + i);
      layer_pixels->SetAt(factor * (pixvalue-minval), i) ;
   }

   if (pal == 0) {
      // Default palette: grayscale palette
      pal = new TImagePalette(256);
      for (i = 0; i < 256; i++) {
         pal->fPoints[i] = ((Double_t)i)/255.0;
         pal->fColorAlpha[i] = 0xffff;
         pal->fColorBlue[i] = pal->fColorGreen[i] = pal->fColorRed[i] = i << 8;
      }
      pal->fPoints[0] = 0;
      pal->fPoints[255] = 1.0;

      im->SetImage(*layer_pixels, UInt_t(width), pal);

      delete pal;

   } else {
      im->SetImage(*layer_pixels, UInt_t(width), pal);
   }

   delete layer_pixels;

   return im;
}

//______________________________________________________________________________
TMatrixD* TFITSHDU::ReadAsMatrix(Int_t layer)
{
   // Read image HDU as a matrix. Return 0 if conversion cannot be done
   // If the HDU seems to be a multilayer image, 'layer' parameter can be used
   // to retrieve the specified layer (starting from 0) in matrix form

   if (fType != kImageHDU) {
      Warning("ReadAsMatrix", "this is not an image HDU.");
      return 0;
   }


   if ((fSizes->GetSize() != 2) && (fSizes->GetSize() != 3)) {
      Warning("ReadAsMatrix", "could not convert image HDU to image because it has %d dimensions.", fSizes->GetSize());
      return 0;
   }

   if (((fSizes->GetSize() == 2) && (layer > 0)) || ((fSizes->GetSize() == 3) && (layer > fSizes->GetAt(2)))) {
      Warning("ReadAsMatrix", "layer out of bounds.");
      return 0;
   }

   Int_t width, height;
   UInt_t pixels_per_layer;
   Int_t offset;

   width  = Int_t(fSizes->GetAt(0));
   height = Int_t(fSizes->GetAt(1));

   pixels_per_layer = UInt_t(width) * UInt_t(height);
   offset = layer * pixels_per_layer;


   TMatrixD *mat = new TMatrixD(height, width);
   double *layer_pixels = new double[pixels_per_layer];

   for (UInt_t i = 0; i < pixels_per_layer; i++) {
      layer_pixels[i] = fPixels->GetAt(offset + i);
   }

   mat->Use(height, width, layer_pixels);

   return mat;
}


//______________________________________________________________________________
TH1 *TFITSHDU::ReadAsHistogram()
{
   // Read image HDU as a histogram. Return 0 if conversion cannot be done.
   // The returned object can be TH1D, TH2D or TH3D depending on data dimensionality.
   // Please, check condition (returnedValue->IsA() == TH*D::Class()) to
   // determine the object class.
   // NOTE: do not confuse with image histogram! This function interprets
   // the array as a histogram. It does not compute the histogram of pixel
   // values of an image! Here "pixels" are interpreted as number of entries.

   if (fType != kImageHDU) {
      Warning("ReadAsHistogram", "this is not an image HDU.");
      return 0;
   }

   TH1 *result=0;

   if ((fSizes->GetSize() != 1) && (fSizes->GetSize() != 2) && (fSizes->GetSize() != 3)) {
      Warning("ReadAsHistogram", "could not convert image HDU to histogram because it has %d dimensions.", fSizes->GetSize());
      return 0;
   }

   if (fSizes->GetSize() == 1) {
      //1D
      UInt_t Nx = UInt_t(fSizes->GetAt(0));
      UInt_t x;

      TH1D *h = new TH1D("", "", Int_t(Nx), 0, Nx-1);

      for (x = 0; x < Nx; x++) {
         Long_t nentries = Long_t(fPixels->GetAt(x));
         if (nentries < 0) nentries = 0; //Crop negative values
         h->Fill(x, nentries);
      }

      result = h;

   } else if (fSizes->GetSize() == 2) {
      //2D
      UInt_t Nx = UInt_t(fSizes->GetAt(0));
      UInt_t Ny = UInt_t(fSizes->GetAt(1));
      UInt_t x,y;

      TH2D *h = new TH2D("", "", Int_t(Nx), 0, Nx-1, Int_t(Ny), 0, Ny-1);

      for (y = 0; y < Ny; y++) {
         UInt_t offset = y * Nx;
         for (x = 0; x < Nx; x++) {
            Long_t nentries = Long_t(fPixels->GetAt(offset + x));
            if (nentries < 0) nentries = 0; //Crop negative values
            h->Fill(x,y, nentries);
         }
      }

      result = h;

   } else if (fSizes->GetSize() == 3) {
      //3D
      UInt_t Nx = UInt_t(fSizes->GetAt(0));
      UInt_t Ny = UInt_t(fSizes->GetAt(1));
      UInt_t Nz = UInt_t(fSizes->GetAt(2));
      UInt_t x,y,z;

      TH3D *h = new TH3D("", "", Int_t(Nx), 0, Nx-1, Int_t(Ny), 0, Ny-1, Int_t(Nz), 0, Nz-1);


      for (z = 0; z < Nz; z++) {
         UInt_t offset1 = z * Nx * Ny;
         for (y = 0; y < Ny; y++) {
            UInt_t offset2 = y * Nx;
            for (x = 0; x < Nx; x++) {
               Long_t nentries = Long_t(fPixels->GetAt(offset1 + offset2 + x));
               if (nentries < 0) nentries = 0; //Crop negative values
               h->Fill(x, y, z, nentries);
            }
         }
      }

      result = h;
   }

   return result;
}

//______________________________________________________________________________
TVectorD* TFITSHDU::GetArrayRow(UInt_t row)
{
   // Get a row from the image HDU when it's a 2D array.

   if (fType != kImageHDU) {
      Warning("GetArrayRow", "this is not an image HDU.");
      return 0;
   }

   if (fSizes->GetSize() != 2) {
      Warning("GetArrayRow", "could not get row from HDU because it has %d dimensions.", fSizes->GetSize());
      return 0;
   }

   UInt_t i, offset, W,H;

   W =  UInt_t(fSizes->GetAt(0));
   H =  UInt_t(fSizes->GetAt(1));


   if (row >= H) {
      Warning("GetArrayRow", "index out of bounds.");
      return 0;
   }

   offset = W * row;
   double *v = new double[W];

   for (i = 0; i < W; i++) {
      v[i] = fPixels->GetAt(offset+i);
   }

   TVectorD *vec = new TVectorD(W, v);

   delete [] v;

   return vec;
}

//______________________________________________________________________________
TVectorD* TFITSHDU::GetArrayColumn(UInt_t col)
{
   // Get a column from the image HDU when it's a 2D array.

   if (fType != kImageHDU) {
      Warning("GetArrayColumn", "this is not an image HDU.");
      return 0;
   }

   if (fSizes->GetSize() != 2) {
      Warning("GetArrayColumn", "could not get row from HDU because it has %d dimensions.", fSizes->GetSize());
      return 0;
   }

   UInt_t i, W,H;

   W =  UInt_t(fSizes->GetAt(0));
   H =  UInt_t(fSizes->GetAt(1));


   if (col >= W) {
      Warning("GetArrayColumn", "index out of bounds.");
      return 0;
   }

   double *v = new double[H];

   for (i = 0; i < H; i++) {
      v[i] = fPixels->GetAt(W*i+col);
   }

   TVectorD *vec = new TVectorD(H, v);

   delete [] v;

   return vec;
}
