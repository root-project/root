#include <ROOT/TDataFrame.hxx>

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
   Core core;
};

#ifdef __ROOTCLING__
#pragma link C++ class std::vector<Electron>+;
#endif


void writeFcc()
{
   using colType = vector<Electron>;
   ROOT::Experimental::TDataFrame d(2);
   d.Define("electrons", [](){return colType(4);})
    .Snapshot<colType>("t",
                       "fccMockup.root",
                       {"electrons"},
                       {"RECREATE", ROOT::kLZ4, 4, 0, 0, false}); // non split!
}
