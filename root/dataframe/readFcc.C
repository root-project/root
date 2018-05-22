/*
// FFCC data model mock
class MyLorentzVector
{
public:
   float px;
   float py;
   float pz;
   float E;
};

class Core
{
public:
   MyLorentzVector p4;
};

class Electron
{
public:
   Core core
};
*/

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>

using floats = ROOT::VecOps::RVec<float>;

void case1(const char* filname, const char* treename = "t")
{
   std::cout << "Case 1" << std::endl;
   ROOT::RDataFrame d(treename, filname);
   d.Alias("q0", "electrons.core.p4.px")
      .Define("sqrtq0", [](float pxs){ return sqrt(pxs); }, {"q0"})
      .Histo1D<floats>("sqrtq0");
}

void case2(const char* filname, const char* treename = "t")
{
   std::cout << "Case 2" << std::endl;
   ROOT::RDataFrame d(treename, filname);
   d.Define("sqrtq0", [](float pxs){ return sqrt(pxs); }, {"electrons.core.p4.px"})
    .Histo1D<floats>("sqrtq0");
}

void case3(const char* filname, const char* treename = "t")
{
   std::cout << "Case 3" << std::endl;
   ROOT::RDataFrame d(treename, filname);
   d.Define("sqrtq0", "sqrt(electrons.core.p4.px)")
    .Histo1D<floats>("sqrtq0");
}


void readFcc()
{
   std::cout << "-- Mockup" << std::endl;
   case1("fccMockup.root");
   case2("fccMockup.root");
   case3("fccMockup.root");
   std::cout << "-- Skim" << std::endl;
   case1("fccSkim.root");
   case2("fccSkim.root");
   case3("fccSkim.root");
}

int main()
{
   readFcc();
}
