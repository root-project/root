// Test the usage of different instantiations of std::map, std::multimap, std::unordered_map, std::unordered_multimap
// Each container has a dictionary generated via an XML selection file
// We test storage of each container as:
// * An object in a file (a.k.a. row-wise)
// * An unsplit branch in a TTree
// * A split branch in a TTree

#include <gtest/gtest.h>

#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <ROOT/TestSupport.hxx>
#include <RtypesCore.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TH1F.h>
#include <TROOT.h>

// Test fixture based on different container types and key types

// Wrapper type used to instantiate pairs of concrete types to use in the
// TYPED_TEST_SUITE macro call
template <template <typename...> typename MapType>
struct MapWrapper {
   template <typename Key, typename Value>
   using wrapped_type = MapType<Key, Value>;
};

template <typename T>
struct cpp11ContainersUnorderedMap : public testing::Test {

   // Type of the key of the container type being tested
   using KeyType = typename T::first_type;

   // Container type being tested, not fully instantiated yet but used
   // to create different data members with the concretely instantiated type
   template <typename K, typename V>
   using ContType = typename T::second_type::template wrapped_type<K, V>;

   static inline bool fAddDirectoryStatus{};
   static inline unsigned fNEvents{100};

   ContType<KeyType, double> fDoubleCont{{1, 1.}, {2, 2.}, {3, 3.}, {4, 4.}};
   ContType<KeyType, TH1F> fHistoCont{{1, TH1F("h1", "ht", 100, -2, 2)}, {9, TH1F("h2", "ht", 10, -1.2, 1.2)}};

   std::vector<ContType<KeyType, TH1F>> fVecHistoCont{
      {{1, TH1F("h1", "ht", 100, -2, 2)}, {2, TH1F("h2", "ht", 10, -1.23, 1.23)}},
      {{7, TH1F("h3", "ht", 100, -23, 23)}, {8, TH1F("h4", "ht", 10, -1.92, 1.92)}}};

   ContType<KeyType, std::vector<TH1F>> fContHistoVec{
      {1, {TH1F("h1", "ht", 100, -2, 2), TH1F("h2", "ht", 10, -1.23, 1.23)}},
      {2, {TH1F("h3", "ht", 100, -23, 23), TH1F("h4", "ht", 10, -1.92, 1.92)}}};

   std::mt19937 fRNG{1};
   std::normal_distribution<double> fGaus{0., 1.};
   std::uniform_real_distribution<double> fUniform{1., 2.};

   static void SetUpTestSuite()
   {
      fAddDirectoryStatus = ROOT::Experimental::ObjectAutoRegistrationEnabled();
      TH1::AddDirectory(false);
   }

   static void TearDownTestSuite() { TH1::AddDirectory(fAddDirectoryStatus); }
};

using TestTypes = ::testing::Types<
   // int
   std::pair<int, MapWrapper<std::map>>, std::pair<int, MapWrapper<std::unordered_map>>,
   std::pair<int, MapWrapper<std::multimap>>, std::pair<int, MapWrapper<std::unordered_multimap>>,
   // float
   std::pair<float, MapWrapper<std::map>>, std::pair<float, MapWrapper<std::unordered_map>>,
   std::pair<float, MapWrapper<std::multimap>>, std::pair<float, MapWrapper<std::unordered_multimap>>,
   // double
   std::pair<double, MapWrapper<std::map>>, std::pair<double, MapWrapper<std::unordered_map>>,
   std::pair<double, MapWrapper<std::multimap>>, std::pair<double, MapWrapper<std::unordered_multimap>>,
   // Long64_t
   std::pair<Long64_t, MapWrapper<std::map>>, std::pair<Long64_t, MapWrapper<std::unordered_map>>,
   std::pair<Long64_t, MapWrapper<std::multimap>>, std::pair<Long64_t, MapWrapper<std::unordered_multimap>>>;
TYPED_TEST_SUITE(cpp11ContainersUnorderedMap, TestTypes);

// Helper functions for equality comparisons

template <typename T>
void check_eq(const T &a, const T &b)
{
   EXPECT_EQ(a, b);
}

template <>
void check_eq<>(const TH1F &a, const TH1F &b)
{
   EXPECT_STREQ(a.GetName(), b.GetName());
   EXPECT_STREQ(a.GetTitle(), b.GetTitle());

   EXPECT_EQ(a.GetNbinsX(), b.GetNbinsX());
   for (int i = 0; i < a.GetNbinsX(); ++i) {
      EXPECT_EQ(a.GetBinContent(i), b.GetBinContent(i));

      EXPECT_DOUBLE_EQ(a.GetBinError(i), b.GetBinError(i));
   }
}

// We want to match associative containers of the `key:value` type. The variadic
// template parameter list is used to help all compilers match this overload
template <typename KeyType, typename ValueType, template <typename... Args> typename Cont, typename... Args>
void check_eq(const Cont<KeyType, ValueType, Args...> &a, const Cont<KeyType, ValueType, Args...> &b)
{
   EXPECT_EQ(a.size(), b.size());
   for (const auto &kv : a) {
      const auto it = b.find(kv.first);
      EXPECT_NE(it, b.cend());
      check_eq(kv.second, it->second);
   }
}

template <typename T>
void check_eq(const std::vector<T> &a, const std::vector<T> &b)
{
   EXPECT_EQ(a.size(), b.size());
   for (decltype(a.size()) i{}; i < a.size(); i++)
      check_eq(a[i], b[i]);
}

// Helper functions to fill values in test fixture data members

template <class Cont>
void randomizeAssoCont(Cont &cont, std::uniform_real_distribution<double> &uniform, std::mt19937 &rng)
{
   // Copy value pairs into temporary vector
   using contPair_t = std::pair<typename Cont::key_type, typename Cont::mapped_type>;
   std::vector<contPair_t> contValues;
   contValues.reserve(cont.size());
   for (const auto &el : cont)
      contValues.push_back(el);

   // Note: sorting is fundamental as we want to deal with values in the same order whether
   // they are coming from an ordered or unordered container
   auto sortingFunction = [](const contPair_t &p1, const contPair_t &p2) { return p1.first < p2.first; };
   std::sort(contValues.begin(), contValues.end(), sortingFunction);

   cont.clear();
   for (auto &kv : contValues) {
      cont.insert(std::make_pair(kv.first, kv.second * uniform(rng)));
   }
}

template <typename Cont>
void fillHistoAssoCont(Cont &cont, std::normal_distribution<double> &gaus, std::mt19937 &rng, unsigned int n = 5000)
{

   // Copy value pairs into temporary vector
   using contPair_t = std::pair<typename Cont::key_type, typename Cont::mapped_type>;
   std::vector<contPair_t> contValues;
   contValues.reserve(cont.size());
   for (const auto &el : cont)
      contValues.push_back(el);

   // Note: sorting is fundamental as we want to deal with values in the same order whether
   // they are coming from an ordered or unordered container
   auto sortingFunction = [](const contPair_t &p1, const contPair_t &p2) { return p1.first < p2.first; };
   std::sort(contValues.begin(), contValues.end(), sortingFunction);

   cont.clear();
   for (auto &kv : contValues) {
      for (decltype(n) i{}; i < n; i++)
         kv.second.Fill(gaus(rng));
      cont.insert(kv);
   }
}

template <class NestedCont>
void fillHistoNestedAssoCont(std::vector<NestedCont> &cont, std::normal_distribution<double> &gaus, std::mt19937 &rng,
                             unsigned int n = 5000)
{
   for (auto &hCont : cont) {
      fillHistoAssoCont(hCont, gaus, rng, n);
   }
}

void fillHistoCont(std::vector<TH1F> &cont, std::normal_distribution<double> &gaus, std::mt19937 &rng,
                   unsigned int n = 5000)
{
   for (auto &h : cont)
      for (decltype(n) i{}; i < n; i++)
         h.Fill(gaus(rng));
}

template <class NestedCont>
void fillHistoNestedAssoCont(NestedCont &cont, std::normal_distribution<double> &gaus, std::mt19937 &rng,
                             unsigned int n = 5000)
{
   // Copy value pairs into temporary vector
   using contPair_t = std::pair<typename NestedCont::key_type, typename NestedCont::mapped_type>;
   std::vector<contPair_t> contValues;
   contValues.reserve(cont.size());
   for (const auto &el : cont)
      contValues.push_back(el);

   // Note: sorting is fundamental as we want to deal with values in the same order whether
   // they are coming from an ordered or unordered container
   auto sortingFunction = [](const contPair_t &p1, const contPair_t &p2) { return p1.first < p2.first; };
   std::sort(contValues.begin(), contValues.end(), sortingFunction);

   cont.clear();
   for (auto &kv : contValues) {
      fillHistoCont(kv.second, gaus, rng, n);
      cont.insert(kv);
   }
}

template <typename T>
void fillObj(T &obj, [[maybe_unused]] std::normal_distribution<double> &gaus,
             [[maybe_unused]] std::uniform_real_distribution<double> &uniform, [[maybe_unused]] std::mt19937 &rng)
{
   if constexpr (std::is_same_v<typename T::mapped_type, double>)
      randomizeAssoCont(obj, uniform, rng);
   else if constexpr (std::is_same_v<typename T::mapped_type, TH1F>)
      fillHistoAssoCont(obj, gaus, rng);
   else if constexpr (std::is_same_v<typename T::mapped_type, std::vector<TH1F>>)
      fillHistoNestedAssoCont(obj, gaus, rng);
   else
      throw std::runtime_error("Unrecognized type");
}

template <typename T>
void fillObj(std::vector<T> &obj, std::normal_distribution<double> &gaus,
             [[maybe_unused]] std::uniform_real_distribution<double> &uniform, std::mt19937 &rng)
{
   fillHistoNestedAssoCont(obj, gaus, rng);
}

// Row-wise storage tests

template <typename T>
void checkRowWise(const T &obj, std::string_view name, std::normal_distribution<double> &gaus,
                  std::uniform_real_distribution<double> &uniform, std::mt19937 &rng)
{
   const char *filename{"cpp11ContainersUnorderedMap_RowWise.root"};

   rng.seed(1);
   gaus.reset();
   uniform.reset();
   auto copy = obj;
   fillObj(copy, gaus, uniform, rng);

   // Write objects to file row-wise, i.e. as a whole object
   {
      auto f = std::make_unique<TFile>(filename, "RECREATE");
      f->WriteObject(&copy, name.data());
   }

   // Read back the object from file and check contents
   {
      auto f = std::make_unique<TFile>(filename);
      auto *objFromFile = f->Get<T>(name.data());
      EXPECT_NE(objFromFile, nullptr) << "Error in reading object " << name << " from file " << filename << "\n";

      check_eq(*objFromFile, copy);
   }
}

TYPED_TEST(cpp11ContainersUnorderedMap, RowWiseDoubleCont)
{
   checkRowWise(this->fDoubleCont, "fDoubleCont", this->fGaus, this->fUniform, this->fRNG);
}

TYPED_TEST(cpp11ContainersUnorderedMap, RowWiseHistoCont)
{
   checkRowWise(this->fHistoCont, "fHistoCont", this->fGaus, this->fUniform, this->fRNG);
}

TYPED_TEST(cpp11ContainersUnorderedMap, RowWiseVecHistoCont)
{
   checkRowWise(this->fVecHistoCont, "fVecHistoCont", this->fGaus, this->fUniform, this->fRNG);
}

TYPED_TEST(cpp11ContainersUnorderedMap, RowWiseContHistoVec)
{
   checkRowWise(this->fContHistoVec, "fContHistoVec", this->fGaus, this->fUniform, this->fRNG);
}

// Column-wise storage tests
template <typename T>
void checkColumnWise(const T &obj, std::string_view name, unsigned nEvents, std::normal_distribution<double> &gaus,
                     std::uniform_real_distribution<double> &uniform, std::mt19937 &rng)
{
   std::string filename{"cpp11ContainersUnorderedMap_ColumnWise.root"};
   std::string splitName = std::string{name} + "_split";
   std::string unsplitName = std::string{name} + "_unsplit";
   auto copyForWrite = obj;
   auto copyForRead = obj;

   // Write
   rng.seed(1);
   gaus.reset();
   uniform.reset();
   {
      auto f = std::make_unique<TFile>(filename.c_str(), "UPDATE");
      auto t = std::make_unique<TTree>("t", "Test Tree");
      t->Branch(splitName.c_str(), &copyForWrite, 16000, 99);
      t->Branch(unsplitName.c_str(), &copyForWrite, 16000, 0);

      for (unsigned i = 0; i < nEvents; ++i) {
         fillObj(copyForWrite, gaus, uniform, rng);
         t->Fill();
      }

      f->Write();
   }

   // Read
   rng.seed(1);
   gaus.reset();
   uniform.reset();
   {
      auto f = std::make_unique<TFile>(filename.c_str());
      std::unique_ptr<TTree> t{f->Get<TTree>("t")};
      TTreeReader r{t.get()};
      TTreeReaderValue<T> splitRV{r, splitName.c_str()};
      TTreeReaderValue<T> unsplitRV{r, unsplitName.c_str()};
      for (unsigned i = 0; i < nEvents; ++i) {
         EXPECT_TRUE(r.Next());
         fillObj(copyForRead, gaus, uniform, rng);
         check_eq(*splitRV, copyForRead);
         check_eq(*unsplitRV, copyForRead);
      }
   }
}

TYPED_TEST(cpp11ContainersUnorderedMap, ColumnWiseDoubleCont)
{
   checkColumnWise(this->fDoubleCont, "fDoubleCont", this->fNEvents, this->fGaus, this->fUniform, this->fRNG);
}

TYPED_TEST(cpp11ContainersUnorderedMap, ColumnWiseHistoCont)
{
   checkColumnWise(this->fHistoCont, "fHistoCont", this->fNEvents, this->fGaus, this->fUniform, this->fRNG);
}

TYPED_TEST(cpp11ContainersUnorderedMap, ColumnWiseVecHistoCont)
{
   checkColumnWise(this->fVecHistoCont, "fVecHistoCont", this->fNEvents, this->fGaus, this->fUniform, this->fRNG);
}

TYPED_TEST(cpp11ContainersUnorderedMap, ColumnWiseContHistoVec)
{
   checkColumnWise(this->fContHistoVec, "fContHistoVec", this->fNEvents, this->fGaus, this->fUniform, this->fRNG);
}

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
