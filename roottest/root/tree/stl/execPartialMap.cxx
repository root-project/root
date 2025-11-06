// This is testing the issue reported at https://github.com/root-project/root/issues/19963

#include <iostream>
#include <string>
#include <map>
#include <memory>

#include "Rtypes.h"
#include "TBuffer.h"

class AddressKey {
public:
   int id;
   std::string name;

   AddressKey(int i = 0, const std::string &n = "") : id(i), name(n) {}

   // Comparison operator for std::map
   bool operator<(const AddressKey &other) const { return (id < other.id) || (id == other.id && name < other.name); }

   static int fgStreamerCalled;

   ClassDef(AddressKey, 1) // Example class with custom Streamer
};

inline void AddressKey::Streamer(TBuffer &b)
{
   ++fgStreamerCalled;
   // Default StreamerInfo-based ROOT I/O for a class with one int member
   // This is the standard pattern for custom Streamer functions
   if (b.IsReading()) {
      b.ReadClassBuffer(AddressKey::Class(), this);
   } else {
      b.WriteClassBuffer(AddressKey::Class(), this);
   }
}

class AddressValue {
public:
   int fData = -1;

   static int fgStreamerCalled;

   ClassDef(AddressValue, 1) // Example class with custom Streamer
};

inline void AddressValue::Streamer(TBuffer &b)
{
   ++fgStreamerCalled;
   // Default StreamerInfo-based ROOT I/O for a class with one int member
   // This is the standard pattern for custom Streamer functions
   if (b.IsReading()) {
      b.ReadClassBuffer(AddressValue::Class(), this);
   } else {
      b.WriteClassBuffer(AddressValue::Class(), this);
   }
}

// Static member definitions
int AddressKey::fgStreamerCalled = 0;
int AddressValue::fgStreamerCalled = 0;

#ifdef __ROOTCLING__
#pragma link C++ class AddressKey-;
#pragma link C++ class AddressValue-;
#pragma link C++ class std::map<AddressKey,AddressValue>+;
#pragma link C++ class std::pair<const AddressKey,AddressValue>+;
#pragma link C++ class std::vector<std::pair<const AddressKey,AddressValue> >+;
#endif

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

TTree *CreateTree()
{
   TTree *tree = new TTree("tree", "A tree with AddressPrinter objects");
   std::map<AddressKey, AddressValue> vec;
   tree->Branch("map", &vec);

   // Fill with one element
   vec[0] = {AddressValue()};
   tree->Fill();

   // Fill with two elements
   vec[1] = {AddressValue()};
   tree->Fill();

   // Fill with one element again
   vec[2] = {AddressValue()};
   tree->Fill();

   tree->ResetBranchAddresses();
   return tree;
}

template <typename T>
struct References;

template <>
struct References<std::map<AddressKey, AddressValue>> {
   static constexpr std::array<int, 3> KeyCount = {1, 3, 6};
   static constexpr std::array<int, 3> ValueCount = {1, 3, 6};
};

template <>
struct References<std::vector<std::pair<const AddressKey, AddressValue>>> {
   static constexpr std::array<int, 3> KeyCount = {0, 0, 0};
   static constexpr std::array<int, 3> ValueCount = {1, 3, 6};
};

template <typename Collection = std::map<AddressKey, AddressValue>>
bool ReadTree(TTree *tree)
{
   constexpr std::array<int, 3> expectedReadKeyCountValues = References<Collection>::KeyCount;
   constexpr std::array<int, 3> expectedReadValueCountValues = References<Collection>::ValueCount;

   AddressKey::fgStreamerCalled = 0;
   AddressValue::fgStreamerCalled = 0;

   bool result = true;

   // std::vector<std::pair<const AddressKey,AddressValue> *map = nullptr;
   // std::map<AddressKey, AddressValue>  *m = nullptr;
   Collection *m = nullptr;
   tree->SetBranchAddress("map", &m);

   ULong64_t nentries = tree->GetEntries();
   for (ULong64_t i = 0; i < nentries; ++i) {
      std::cout << "Entry " << i << ":\n";
      // tree->GetEntry(i);
      tree->GetBranch("map")->TBranch::GetEntry(i);
      tree->GetBranch("map.second")->GetEntry(i);
      if (!m) {
         std::cerr << "ERROR: nullptr branch\n";
         continue;
      }
      std::cout << "AddressKey::Streamer called " << AddressKey::fgStreamerCalled << " times\n";
      std::cout << "AddressValue::Streamer called " << AddressValue::fgStreamerCalled << " times\n";
      int expectedKeyCount = (i < expectedReadKeyCountValues.size()) ? expectedReadKeyCountValues[i] : -1;
      if (AddressKey::fgStreamerCalled != expectedKeyCount) {
         std::cerr << "ERROR: AddressKey::fgStreamerCalled=" << AddressKey::fgStreamerCalled
                   << ", expected=" << expectedKeyCount << std::endl;
         result = false;
      }
      int expectedValueCount = (i < expectedReadValueCountValues.size()) ? expectedReadValueCountValues[i] : -1;
      if (AddressValue::fgStreamerCalled != expectedValueCount) {
         std::cerr << "ERROR: AddressValue::fgStreamerCalled=" << AddressValue::fgStreamerCalled
                   << ", expected=" << expectedValueCount << std::endl;
         result = false;
      }
   }
   tree->ResetBranchAddresses();
   delete m;
   return result;
}

int readwrite()
{
   std::cout << "Creating tree\n";
   std::unique_ptr<TFile> f(TFile::Open("objects.root", "RECREATE"));

   TTree *tree = CreateTree();
   f->Write();
   std::cout << "Reading tree\n";
   auto result = ReadTree(tree);
   std::cout << "Deleting tree\n";
   delete tree;
   std::cout << "done\n";
   return result ? 0 : 1;
}

template <typename Collection>
int justread()
{
   std::cout << "Open file\n";
   std::unique_ptr<TFile> f(TFile::Open("objects.root", "READ"));
   auto tree = f->Get<TTree>("tree");
   std::cout << "Reading tree\n";
   auto result = ReadTree<Collection>(tree);
   std::cout << "Deleting tree\n";
   delete tree;
   std::cout << "done\n";
   return result ? 0 : 2;
}

int execPartialMap()
{
   int ret1 = readwrite();
   AddressKey::fgStreamerCalled = 0;
   AddressValue::fgStreamerCalled = 0;
   int ret2 = justread<std::vector<std::pair<const AddressKey, AddressValue>>>();
   return ret1 + ret2;
}
