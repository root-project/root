#include "TMultiGraph.h"
#include "TGraph.h"
#include "TMath.h"

#include "ExclusionGraphDemo.h"
#include "IOSPad.h"

namespace ROOT {
namespace iOS {
namespace Demos {

//______________________________________________________________________________
ExclusionGraphDemo::ExclusionGraphDemo()
                        : fMultiGraph(new TMultiGraph)
{
   fMultiGraph->SetTitle("Exclusion graphs");

   Double_t x1[kNPoints], x2[kNPoints], x3[kNPoints];
   Double_t y1[kNPoints], y2[kNPoints], y3[kNPoints];
   
   for (UInt_t i = 0; i < kNPoints; ++i) {
      x1[i]  = i * 0.1;
      x2[i]  = x1[i];
      x3[i]  = x1[i] + 0.5;
      y1[i] = 10 * TMath::Sin(x1[i]);
      y2[i] = 10 * TMath::Cos(x1[i]);
      y3[i] = 10 * TMath::Sin(x1[i]) - 2;
   }

   std::auto_ptr<TGraph> graph1(new TGraph(kNPoints, x1, y1));
   graph1->SetLineColor(2);
   graph1->SetLineWidth(1504);
   graph1->SetFillStyle(3005);

   std::auto_ptr<TGraph> graph2(new TGraph(kNPoints, x2, y2));
   graph2->SetLineColor(4);
   graph2->SetLineWidth(-2002);
   graph2->SetFillStyle(3004);
   graph2->SetFillColor(9);

   std::auto_ptr<TGraph> graph3(new TGraph(kNPoints, x3, y3));
   graph3->SetLineColor(5);
   graph3->SetLineWidth(-802);
   graph3->SetFillStyle(3002);
   graph3->SetFillColor(2);

   fMultiGraph->Add(graph1.get());
   fMultiGraph->Add(graph2.get());
   fMultiGraph->Add(graph3.get());
   
   graph1.release();
   graph2.release();
   graph3.release();
}

//______________________________________________________________________________
ExclusionGraphDemo::~ExclusionGraphDemo()
{
   //Just for auto_ptr's dtor.
}

//______________________________________________________________________________
void ExclusionGraphDemo::AdjustPad(Pad *pad)
{
   pad->SetFillColor(0);
}

//______________________________________________________________________________
void ExclusionGraphDemo::PresentDemo()
{
   fMultiGraph->Draw("AC");
}

}
}
}
