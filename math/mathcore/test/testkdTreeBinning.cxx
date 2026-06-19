// ------------------------------------------------------------------------
//
// kdTreeBinning tutorial: bin the data in cells of equal content using a kd-tree
//
// Using TKDTree wrapper class as a data binning structure
//  Plot the 2D data using the TH2Poly class
//
//
// Author:   Bartolomeu Rabacal    11/2010
//
// ------------------------------------------------------------------------

#include <cmath>

#include "TKDTreeBinning.h"
#include "TH2D.h"
#include "TH2Poly.h"
#include "TRandom3.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TMarker.h"
#include <iostream>

bool verbose = false;
bool showGraphics = false;

using std::cout;
using std::cerr;
using std::endl;

// Returns the number of detected failures (0 on success) so that the caller can
// turn it into a non-zero process exit code: ROOT's Error() only prints, it does
// not by itself make the test fail under ctest.
int testkdTreeBinning()
{

   int nfail = 0;

   // -----------------------------------------------------------------------------------------------
   //  C r e a t e  r a n d o m  s a m p l e
   // -----------------------------------------------------------------------------------------------

   const UInt_t DATASZ = 10000;
   const UInt_t DATADIM = 2;
   const UInt_t NBINS = 50;

   Double_t smp[DATASZ * DATADIM];

   double mu[2] = {0,2};
   double sig[2] = {2,3};
   TRandom3 r;
   r.SetSeed(1);
   for (UInt_t i = 0; i < DATADIM; ++i)
      for (UInt_t j = 0; j < DATASZ; ++j)
         smp[DATASZ * i + j] = r.Gaus(mu[i], sig[i]);

   UInt_t h1bins = (UInt_t) std::sqrt(double(NBINS));

   TH2D* h1 = new TH2D("h1BinTest", "Regular binning", h1bins, -5., 5., h1bins, -5., 5.);
   for (UInt_t j = 0; j < DATASZ; ++j)
      h1->Fill(smp[j], smp[DATASZ + j]);


   // ---------------------------------------------------------------------------------------------
   // C r e a t e  K D T r e e B i n n i n g  o b j e c t  w i t h  T H 2 P o l y  p l o t t i n g
   // ---------------------------------------------------------------------------------------------

   TKDTreeBinning* kdBins = new TKDTreeBinning(DATASZ, DATADIM, smp, NBINS);

   UInt_t nbins = kdBins->GetNBins();
   UInt_t dim   = kdBins->GetDim();

   const Double_t* binsMinEdges = kdBins->GetBinsMinEdges();
   const Double_t* binsMaxEdges = kdBins->GetBinsMaxEdges();

   TH2Poly* h2pol = new TH2Poly("h2PolyBinTest", "KDTree binning", kdBins->GetDataMin(0), kdBins->GetDataMax(0), kdBins->GetDataMin(1), kdBins->GetDataMax(1));

   for (UInt_t i = 0; i < nbins; ++i) {
      UInt_t edgeDim = i * dim;
      h2pol->AddBin(binsMinEdges[edgeDim], binsMinEdges[edgeDim + 1], binsMaxEdges[edgeDim], binsMaxEdges[edgeDim + 1]);
   }

   for (UInt_t i = 1; i <= kdBins->GetNBins(); ++i)
      h2pol->SetBinContent(i, kdBins->GetBinDensity(i - 1));

   int ibinMin = kdBins->GetBinMinDensity();
   int ibinMax = kdBins->GetBinMaxDensity();

   std::cout << "Bin with minimum density: " << ibinMin << " density = " <<  kdBins->GetBinDensity(ibinMin) << " content = " << kdBins->GetBinContent(ibinMin)  << std::endl;
   std::cout << "Bin with maximum density: " << ibinMax << " density = " <<  kdBins->GetBinDensity(ibinMax) << " content = " << kdBins->GetBinContent(ibinMax) << std::endl;

   if (kdBins->GetBinContent(ibinMax) != DATASZ / NBINS) {
      Error("testkdTreeBinning","Wrong bin content");
      ++nfail;
   }

   // order bins by density
   kdBins->SortBinsByDensity(true);

   if (kdBins->GetBinMinDensity() != 0) {
      Error("testkdTreeBinning", "Wrong minimum bin after sorting");
      ++nfail;
   }
   if (kdBins->GetBinMaxDensity() != nbins - 1) {
      Error("testkdTreeBinning", "Wrong maximum bin after sorting");
      ++nfail;
   }

   if (showGraphics) {
      new TCanvas();
      h2pol->Draw("COLZ L");
      gPad->Update();
   }


   // test find bin
   int ntimes = (verbose) ? 2 : 200;
   double point[2] = {0,0};
//   double binCenter[2];
   gRandom->SetSeed(0);
   for (int itimes = 0; itimes < ntimes; itimes++) {

      // generate a random point in 2D
      point[0] = gRandom->Uniform(-5,5);
      point[1] = gRandom->Uniform(-5,5);
      // int inode = tree->FindNode(point);
      // inode = inode - tree->GetNNodes();

      int ibin = kdBins->FindBin(point);

      const double * binCenter = kdBins->GetBinCenter(ibin);
      const double * binMin = kdBins->GetBinMinEdges(ibin);
      const double * binMax = kdBins->GetBinMaxEdges(ibin);
      if (binCenter) {

         if (verbose) {
            std::cout << "**** point *** " << itimes << "\n";
            std::cout << " point x " << point[0] << " BIN CENTER is " << binCenter[0] << " min " << binMin[0] << " max " << binMax[0] << std::endl;
               std::cout << " point y " << point[1] << " BIN CENTER is " << binCenter[1] << " min " << binMin[1] << " max " << binMax[1] <<  std::endl;
         }

         bool ok = point[0] > binMin[0] && point[0] < binMax[0] && point[1] > binMin[1] && point[1] < binMax[1];
         if (!ok) {
            Error ("testkdTreeBinning::FindBin"," Point is not in the right bin " );
            std::cout << " point x " << point[0] << " BIN CENTER is " << binCenter[0] << " min " << binMin[0] << " max " << binMax[0] << std::endl;
            std::cout << " point y " << point[1] << " BIN CENTER is " << binCenter[1] << " min " << binMin[1] << " max " << binMax[1] <<  std::endl;
            ++nfail;
         }

         if (itimes < 2 && showGraphics ) {

            TMarker * m1 = new TMarker(point[0], point[1],20 );
            TMarker * m2 = new TMarker(binCenter[0], binCenter[1], 21);
            m1->Draw();
            m2->Draw();
         }

         delete [] binCenter;
      } else {
         Error("testkdTreeBinning::FindBin"," Bin %d is not existing ",ibin);
         ++nfail;
      }
   }

   return nfail;
}

// Regression test for the case where the requested number of bins does not
// divide the data size evenly. In that situation the kd-tree builds more
// terminal nodes (bins) than naively expected, and FindBin() must never return
// an index outside [0, GetNBins()). See
// https://github.com/root-project/root/issues/10784 about
// TKDTreeBinning::FindBin returning non-existent bins. Returns the number of
// detected failures (0 on success) so that the caller can turn it into a
// non-zero process exit code: ROOT's Error() only prints, it does not by
// itself make the test fail under ctest.
int testkdTreeBinningFindBinRange()
{

   int nfail = 0;

   const UInt_t DATASZ = 100500; // deliberately NOT a multiple of NBINS
   const UInt_t DATADIM = 5;
   const UInt_t NBINS = 1000;

   std::vector<Double_t> smp(DATASZ * DATADIM);
   TRandom3 r;
   r.SetSeed(1);
   for (UInt_t i = 0; i < DATADIM; ++i)
      for (UInt_t j = 0; j < DATASZ; ++j)
         smp[DATASZ * i + j] = r.Uniform(-1., 1.);

   TKDTreeBinning kdBins(DATASZ, DATADIM, smp, NBINS);

   const UInt_t nbins = kdBins.GetNBins();

   // The number of bins must match the number of terminal nodes of the kd-tree.
   if ((int)nbins != kdBins.GetTree()->GetNNodes() + 1) {
      Error("testkdTreeBinningFindBinRange", "GetNBins() (%u) != number of kd-tree terminal nodes (%d)", nbins,
            kdBins.GetTree()->GetNNodes() + 1);
      ++nfail;
   }

   // Every data point must be assigned to a valid bin.
   std::vector<Double_t> point(DATADIM);
   for (UInt_t j = 0; j < DATASZ; ++j) {
      for (UInt_t i = 0; i < DATADIM; ++i)
         point[i] = smp[DATASZ * i + j];
      UInt_t bin = kdBins.FindBin(point.data());
      if (bin >= nbins) {
         Error("testkdTreeBinningFindBinRange", "FindBin returned out-of-range bin %u (NBins = %u)", bin, nbins);
         ++nfail;
         break;
      }
   }

   // The total bin content must add up to the data size.
   Long64_t total = 0;
   for (UInt_t i = 0; i < nbins; ++i)
      total += kdBins.GetBinContent(i);
   if (total != (Long64_t)DATASZ) {
      Error("testkdTreeBinningFindBinRange", "Sum of bin contents (%lld) != data size (%u)", total, DATASZ);
      ++nfail;
   }

   return nfail;
}

int main(int argc, char **argv)
{
  // Parse command line arguments
  for (Int_t i=1 ;  i<argc ; i++) {
     std::string arg = argv[i] ;
     if (arg == "-g") {
      showGraphics = true;
     }
     if (arg == "-v") {
      showGraphics = true;
      verbose = true;
     }
     if (arg == "-h") {
        cerr << "Usage: " << argv[0] << " [-g] [-v]\n";
        cerr << "  where:\n";
        cerr << "     -g : graphics mode\n";
        cerr << "     -v : verbose  mode";
        cerr << endl;
        return -1;
     }
   }

   TApplication* theApp = nullptr;
   if ( showGraphics )
      theApp = new TApplication("App",&argc,argv);

   int nfail = testkdTreeBinning();

   nfail += testkdTreeBinningFindBinRange();

   if ( showGraphics )
   {
      theApp->Run();
      delete theApp;
      theApp = nullptr;
   }

   return nfail;
}

