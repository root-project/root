#include "classes.h"

#include "memory"
#include "TFile.h"
#include "TH1F.h"

void w(const char *filename)
{
   std::unique_ptr<TFile> f(TFile::Open(filename, "RECREATE"));
   auto c = new Aconst("test");
   f->WriteObject(c, "c");
}

void r(const char* filename)
{
   std::unique_ptr<TFile> f(TFile::Open(filename));
   Aconst *c;
   f->GetObject("c", c);
}

int rwconst(bool write = true){

   auto c = TClass::GetClass("Aconst");
   auto si = c->GetStreamerInfo();
   // Here the output

   std::vector<const char *> filenames{"outconst.root"}; //, "out.xml"};
   for (auto filename : filenames){
      if (write) w(filename);
      r(filename);
   }
   return 0;
}
