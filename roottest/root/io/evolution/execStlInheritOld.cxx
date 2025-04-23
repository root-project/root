#include <vector>

class Data {
public:
   int fValue;
   Data() : fValue(0) {}
   Data(const Data &rhs) : fValue(rhs.fValue) {}
   Data(int in) : fValue(in) {}
};

class Container : public vector<Data>
{

};

class ContainerInt : public vector<int>
{

};

#include "TFile.h"

void execStlInheritOld(const char *filename = "inheritstl.root")
{
   Container c;
   c.push_back(3);
   ContainerInt ci;
   ci.push_back(5);
   TFile *f = TFile::Open(filename,"RECREATE");
   f->WriteObject(&c,"cont");
   f->WriteObject(&ci,"contint");
   delete f;
}


