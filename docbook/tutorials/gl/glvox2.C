// This macro demonstrates how to use "glcol" option for TH3
// and how to create user defined TRANSFER FUNCTION:
// transfer function maps bin value to voxel's opacity.
// codomain is [0, 1] (1. - non-transparent, 0.5 is semitransparent, etc.)
// To pass transparency function into painting algorithm, you have to:
// 1) Create TF1 object (with symbolic expression like "0.5 * (sin(x) + 1)":
// ...
// TF1 * tf = new TF1("TransferFunction", "0.5 * (sin(x) + 1)", -10., 10.);
// ...
// IMPORTANT, the name of TF1 object MUST be "TransferFunction".
// 2) Add this function into a hist's list of functions:
// ...
// TList * lof = hist->GetListOfFunctions();
// if (lof) lof->Add(tf);
// ...
// It's also possible to use your own function and pass it into TF1, please read
// TF1 documentation to learn how.
//
// This macro is to be compiled: TF1 is extremely slow with CINT's function
// as an argument.
//
//
// Author: Timur Pocheptsov
//

#if defined(__CINT__) && !defined(__MAKECINT__)

{
  //gSystem->AddIncludePath("-I$ROOTSYS/include");
   TString macroName;
   const char *rootSysPath = gSystem->Getenv("ROOTSYS");
   if (rootSysPath) {
      macroName = rootSysPath;
      if (!macroName.EndsWith("/") && !macroName.EndsWith("\\"))
         macroName += "/";
      macroName += "tutorials/gl/";
   }

   macroName += "glvox2.C++";
   gROOT->LoadMacro(macroName.Data());
   glvox2();
}

#else

#include "TStyle.h"
#include "TList.h"
#include "TH3.h"
#include "TF1.h"

namespace {

Double_t my_transfer_function(const Double_t *x, const Double_t * /*param*/)
{
   // Bin values in our example range from -2 to 1.
   // Let's make values from -2. to -1.5 more transparent:
   if (*x < -1.5)
      return 0.08;

   if (*x > -1.5 && *x < -0.5)
      return 0.015;

   if (*x < -0.5 && *x < 0.)
      return 0.02;

   if (*x > 0 && *x < 0.5)
      return 0.03;

   if (*x > 0.8)
      return 0.01;

   return 0.2;
}

}

void glvox2()
{
   //Create and fill TH3.
   const UInt_t nX = 30;
   const Double_t xMin = -1., xMax = 1., xStep = (xMax - xMin) / (nX - 1);

   const UInt_t nY = 30;
   const Double_t yMin = -1., yMax = 1., yStep = (yMax - yMin) / (nY - 1);

   const UInt_t nZ = 30;
   const Double_t zMin = -1., zMax = 1., zStep = (zMax - zMin) / (nZ - 1);

   TH3F *hist = new TH3F("glvoxel", "glvoxel", nX, -1., 1., nY, -1., 1., nZ, -1., 1.);

   //Fill the histogram to create a "sphere".
   for (UInt_t i = 0; i < nZ; ++i) {
      const Double_t z = zMin + i * zStep;

      for (UInt_t j = 0; j < nY; ++j) {
         const Double_t y = yMin + j * yStep;

         for (UInt_t k = 0; k < nX; ++k) {
            const Double_t x = xMin + k * xStep;

            const Double_t val = 1. - (x * x + y * y + z * z);
            hist->SetBinContent(k + 1, j + 1, i + 1, val);
         }
      }
   }

   //Now, specify the transfer function.
   TList * lf = hist->GetListOfFunctions();
   if (lf) {
      TF1 * tf = new TF1("TransferFunction", my_transfer_function);
      lf->Add(tf);
   }

   gStyle->SetCanvasPreferGL(1);
   gStyle->SetPalette(1);

   hist->Draw("glcol");
}

#endif
