/// \file
/// \ingroup tutorial_dataframe
/// \notebook
/// This tutorial shows how to implement a custom action.
/// As an example, we build a helper for filling a TGraph.
/// This helper, used in Multithread, may lead to unexpected results depending on how dots are connected.
/// However, it is possible to specify in the constructor if a sort on the x axis is needed.
///
/// \macro_code
///
/// \date July 2018
/// \author Enrico Guiraud, Danilo Piparo, Massimo Tumolo

class FillTGraph : public ROOT::Detail::RDF::RActionImpl<FillTGraph> {
public:
   /// This type is a requirement for every helper.
   using Result_t = TGraph;

private:
   std::unique_ptr<ROOT::TThreadedObject<TGraph>> fTo;
   bool isSortingRequired;

public:
   FillTGraph(FillTGraph &&) = default;
   FillTGraph(const FillTGraph &) = delete;

   FillTGraph(const std::shared_ptr<TGraph> &h, const bool &isSortingRequired = false)
      : fTo(new ROOT::TThreadedObject<TGraph>(*h)), isSortingRequired(isSortingRequired)
   {
      const auto nSlots = ROOT::IsImplicitMTEnabled() ? ROOT::GetImplicitMTPoolSize() : 1;
      fTo->SetAtSlot(nSlots, h);
      // Initialise all other slots
      for (unsigned int i = 0; i < nSlots; ++i) {
         fTo->GetAtSlot(i);
      }
   }

   std::shared_ptr<Result_t> GetResultPtr() const
   {
      auto graph = fTo->Get();
      if (isSortingRequired)
         graph->Sort();
      return graph;
   }

   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, double x0, double x1)
   {
      auto rawSlot = fTo->GetAtSlotRaw(slot);
      rawSlot->SetPoint(rawSlot->GetN(), x0, x1);
   }

   void Finalize() { fTo->Merge(); }

   TGraph &PartialUpdate(unsigned int slot) { return *fTo->GetAtSlotRaw(slot); }
};

void df021_createTGraph()
{
   // We enable implicit parallelism
   ROOT::EnableImplicitMT(8);

   std::vector<int> source(160);
   for (int i = 0; i < 160; ++i)
      source[i] = i;

   ROOT::RDataFrame d(160);
   auto dd = d.DefineSlotEntry("x1",
                               [&source](unsigned int slot, ULong64_t entry) {
                                  (void)slot;
                                  return source[entry];
                               })
                .DefineSlotEntry("x2", [&source](unsigned int slot, ULong64_t entry) {
                   (void)slot;
                   return source[entry];
                });

   using Helper_t = FillTGraph;

   auto sortedGraph = std::make_shared<TGraph>();

   // If true is not specified, its assumed unsorted.
   Helper_t sortedHelper(sortedGraph, true);

   auto sorted = dd.Book<int, int>(std::move(sortedHelper), {"x1", "x2"});

   sorted->DrawClone("APL");
}
