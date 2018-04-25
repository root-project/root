/// \file
/// \ingroup tutorial_tdataframe
/// \notebook -draw
/// This tutorial shows how to implement a custom action.
/// As an example, we build a helper for filling THns.
///
/// \macro_code
///
/// \date April 2018
/// \author Enrico Guiraud, Danilo Piparo

template <typename T, unsigned int NDIM, typename... ColumnTypes>
class THnHelper : public ROOT::Detail::TDF::TActionImpl<THnHelper<T, NDIM, ColumnTypes...>> {
public:
   // This is the list of the types of the columns
   using ColumnTypes_t = ROOT::TypeTraits::TypeList<ColumnTypes...>;
   using THn_t = THnT<T>;
   using Result_t = THn_t;

private:
   std::vector<std::shared_ptr<THn_t>> fHistos; // one per slot
   const ROOT::Detail::TDF::ColumnNames_t fColumnNames;

public:
   THnHelper(std::string_view name, std::string_view title, std::array<int, NDIM> nbins, std::array<double, NDIM> xmins,
             std::array<double, NDIM> xmax, unsigned int nSlots, ROOT::Detail::TDF::ColumnNames_t columnNames)
      : fHistos(nSlots, std::make_shared<THn_t>(std::string(name).c_str(), std::string(title).c_str(), NDIM,
                                                nbins.data(), xmins.data(), xmax.data())),
        fColumnNames(columnNames)
   {
   }
   THnHelper(THnHelper &&) = default;
   THnHelper(const THnHelper &) = delete;
   ROOT::Detail::TDF::ColumnNames_t GetColumnNames() const { return fColumnNames; }
   std::shared_ptr<THn_t> GetResultPtr() const { return fHistos[0]; }
   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, ColumnTypes... values)
   {
      std::array<double, sizeof...(ColumnTypes)> valuesArr{(double)values...};
      fHistos[slot]->Fill(valuesArr.data());
   }

   void Finalize()
   {
      auto &res = fHistos[0];
      for (auto slot : ROOT::TSeqU(1, fHistos.size())) {
         res->Add(fHistos[slot].get());
      }
   }
};

void tdf018_customActions()
{
   const auto nSlots = 4;
   ROOT::EnableImplicitMT(nSlots);

   ROOT::Experimental::TDataFrame d(128);
   auto genD = []() { return gRandom->Uniform(-5, 5); };
   auto genF = [&genD]() { return (float)genD(); };
   auto genI = [&genD]() { return (int)genD(); };
   auto dd = d.Define("x0", genD).Define("x1", genD).Define("x2", genF).Define("x3", genI);

   using Helper_t = THnHelper<float, 4, double, double, float, int>;

   Helper_t helper{"myThN",
                   "A THn with 4 dimensions",
                   {4, 4, 8, 2},
                   {-10., -10, -4., -6.},
                   {10., 10, 5., 7.},
                   nSlots,
                   {"x0", "x1", "x2", "x3", "x4"}};

   auto myTHnT = dd.Book(std::move(helper));

   myTHnT->Print();
}
