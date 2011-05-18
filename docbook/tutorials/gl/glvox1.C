// This macro demonstrates how to use "glcol" option for TH3.
// Author: Timur Pocheptsov

void glvox1()
{
   //Create and fill TH3.
   const UInt_t nX = 30;
   const Double_t xMin = -1., xMax = 1., xStep = (xMax - xMin) / (nX - 1);

   const UInt_t nY = 30;
   const Double_t yMin = -1., yMax = 1., yStep = (yMax - yMin) / (nY - 1);

   const UInt_t nZ = 30;
   const Double_t zMin = -1., zMax = 1., zStep = (zMax - zMin) / (nZ - 1);

   TH3F *hist = new TH3F("glvoxel", "glvoxel", 30, -1., 1., 30, -1., 1., 30, -1., 1.);

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

   gStyle->SetCanvasPreferGL(1);
   gStyle->SetPalette(1);

   hist->Draw("glcol");
}
