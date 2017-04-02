#include "TFile.h"

#include <array>

template<class T>
class A{};

class C {

 std::array<A<int>, 4> v;

};

int testArrOfTemplateInst(){

TFile f("f.root", "RECREATE");
C c;
f.WriteObject(&c, "c");
f.Close();

return 0;

}
