template <typename T> class Wrapper
{
public:
   T fValue;
};

#include "TFile.h"

template <typename T> void writeFile(const char *filename)
{
   TFile *f = TFile::Open(filename,"RECREATE");
   T obj;
   f->WriteObject(&obj,"obj");
   delete f;
}


