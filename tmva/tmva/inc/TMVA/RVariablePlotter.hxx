#ifndef TMVA_RVARIABLEPLOTTER
#define TMVA_RVARIABLEPLOTTER

#include <vector>
#include <string>

#include "TLegend.h"
#include "TH1D.h"
#include "THStack.h"

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/RInterface.hxx"

class RVariablePlotter {
private:
   std::vector<ROOT::RDF::RNode> fNodes;
   std::vector<std::string> fLabels;

public:
   RVariablePlotter(
         const std::vector<ROOT::RDF::RNode>& nodes,
         const std::vector<std::string>& labels) : fNodes(nodes), fLabels(labels)
   {
      if (fNodes.size() != fLabels.size())
         std::runtime_error("Number of given RDataFrame nodes does not match number of given class labels.");
      if (fNodes.size() == 0)
         std::runtime_error("Number of given RDataFrame nodes and number of given class labels cannot be zero.");
   }
   void Draw(const std::string& variable);
   void DrawLegend(float minX, float minY, float maxX, float maxY);
};

inline void RVariablePlotter::Draw(const std::string& variable) {
   // Make histograms
   const auto size = fNodes.size();
   std::vector<ROOT::RDF::RResultPtr<TH1D>> histos;
   for (std::size_t i = 0; i < size; i++) {
      // Trigger event loop with computing the histogram
      auto h = fNodes[i].Histo1D(variable);
      histos.push_back(h);
   }

   // Modify style and draw histograms
   THStack stack("","");
   for (unsigned int i = 0; i < histos.size(); i++) {
      histos[i]->SetLineColor(i + 1);
      if (i == 0) {
         histos[i]->SetTitle("");
         histos[i]->SetStats(false);
      }
      stack.Add(histos[i].GetPtr());
   }
   auto clone = (THStack*) stack.DrawClone("nostack");
   clone->GetXaxis()->SetTitle(variable.c_str());
   clone->GetYaxis()->SetTitle("Count");
}

inline void RVariablePlotter::DrawLegend(float minX = 0.1, float minY = 0.1, float maxX = 0.9, float maxY = 0.9) {
   TLegend l(minX, minY, maxX, maxY);
   std::vector<TH1D> histos(fLabels.size());
   for (unsigned int i = 0; i < fLabels.size(); i++) {
      histos[i].SetLineColor(i + 1);
      l.AddEntry(&histos[i], fLabels[i].c_str(), "l");
   }
   l.SetBorderSize(0);
   l.DrawClone();
}

#endif // TMVA_RVARIABLEPLOTTER
