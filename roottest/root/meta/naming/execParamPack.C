{
TInterpreter::SuspendAutoParsing s(gInterpreter);
gInterpreter->Declare("namespace pos { class PixelROCName {}; }");
#include <map>
#include <vector>
//vector<int> v;
//using namespace pos;
gROOT->ProcessLine("std::map<const unsigned int, class std::map<unsigned int, class std::vector<pos::PixelROCName> > > m");
return 0;
}
