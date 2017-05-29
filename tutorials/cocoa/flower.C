/// \file
/// \ingroup tutorial_cocoa
/// A demo to show transparency with TMultiGraph
/// (and a really interesting curve/equation). Point compression in TPadPainter.
/// You can see all three flowers ONLY with Cocoa (transparency).
///
/// The equation by Paul Burke: http://paulbourke.net/geometry/
///
/// \macro_code
///
/// \author Timur Pocheptsov

#include <cassert>
#include <vector>

#include "TMultiGraph.h"
#include "TVirtualX.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TError.h"
#include "TColor.h"
#include "TMath.h"

#include "customcolor.h"

namespace {

typedef std::vector<Double_t> vector_type;
typedef vector_type::size_type size_type;

//______________________________________________________________________
void create_flower(vector_type &xs, vector_type &ys, size_type nPoints, Double_t r)
{
   assert(nPoints > 100 && "create_flower, number of points is too small");

   xs.resize(nPoints + 1);
   ys.resize(nPoints + 1);

   const Double_t angle = 21. * TMath::Pi() / nPoints;

   for (size_type i = 0; i <= nPoints; ++i) {
      const Double_t u = i * angle;
      const Double_t p4 = TMath::Sin(17 * u / 3);
      const Double_t p8 = TMath::Sin(2 * TMath::Cos(3 * u) - 28 * u);
      const Double_t rr = r * (1 + TMath::Sin(11 * u / 5)) - 4 * p4 * p4 * p4 * p4 * p8 * p8 * p8 * p8 * p8 * p8 * p8 * p8;

      xs[i] = rr * TMath::Cos(u);
      ys[i] = rr * TMath::Sin(u);
   }
}

}//unnamed namespace.

void flower()
{
   //0. Indices for custom colors.
   Color_t indices[3] = {};
   if (ROOT::CocoaTutorials::FindFreeCustomColorIndices(indices) != 3) {
      ::Error("flower", "failed to create custom colors");
      return;
   }

   //1. I have to create a canvas to initialize gVirtualX.
   TCanvas * const cnv = new TCanvas("Chrysanthemum", "Chrysanthemum", 900, 900);
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      ::Error("flower", "This macro requires OS X version of ROOT with cocoa enabled");
      delete cnv;
      return;
   }

   cnv->cd();//Just to suppress a warning if compiled.

   vector_type xs, ys;

   //2. Create graphs and custom colors for each graph.
   create_flower(xs, ys, 300, 6);
   TGraph * const gr1 = new TGraph(Int_t(xs.size()), &xs[0], &ys[0]);
   new TColor(indices[0], 0., 0., 0.5, "custom_blue", 0.7);
   gr1->SetFillColor(indices[0]);
   gr1->SetName("part1");
   gr1->SetTitle("part1");

   create_flower(xs, ys, 500000, 8);
   TGraph * const gr2 = new TGraph(Int_t(xs.size()), &xs[0], &ys[0]);
   new TColor(indices[1], 0.5, 0., 0.5, "custom_purple", 0.5);
   gr2->SetFillColor(indices[1]);
   gr2->SetName("part2");
   gr2->SetTitle("part2");

   create_flower(xs, ys, 100000, 10);
   TGraph * const gr3 = new TGraph(Int_t(xs.size()), &xs[0], &ys[0]);

   //If you want to see the difference, change 0.2 to 1 in the next call:
   new TColor(indices[2], 1., 0., 0.4, "custom_magenta", 0.2);
   gr3->SetFillColor(indices[2]);
   gr3->SetName("part3");
   gr3->SetTitle("part3");

   //3. Create a final multigraph.

   //Otcveli, uzh davno ... nu ti ponEl.

   TMultiGraph * const flower = new TMultiGraph("Chrysanthemum", "Chrysanthemum");
   flower->Add(gr1);
   flower->Add(gr2);
   flower->Add(gr3);

   flower->Draw("AFP");
}
