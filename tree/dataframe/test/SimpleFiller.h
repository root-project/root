#ifndef RDF_TEST_SIMPLEFILLER
#define RDF_TEST_SIMPLEFILLER

#include <TH1D.h>

class SimpleFiller {
   TH1D fHisto;

public:
   SimpleFiller() : fHisto("", "", 128, 0., 0.) {}
   SimpleFiller(const SimpleFiller &) = default;
   SimpleFiller(SimpleFiller &&) = default;

   void Fill(double x) { fHisto.Fill(x); }
   void Merge(const std::vector<SimpleFiller *> &others)
   {
      TList l;
      for (auto *o : others)
         l.Add(&o->GetHisto());
      fHisto.Merge(&l);
   }

   TH1D &GetHisto() { return fHisto; }
};

#endif // RDF_TEST_SIMPLEFILLER
