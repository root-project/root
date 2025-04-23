#include "TFile.h"
#include "infoDumper.h"

const char * modelForReadDict2_code = R"(
#include <array>
namespace edm2 {

   class B{};

   class A{
   public:
      std::array<int,3> a0 {{3,6,9}};
      int a1[3] = {3,6,9};

      std::array<std::array<int,3>,3> a2 {{ {{1,2,3}},{{1,2,3}},{{1,2,3}} }};
      int a3[3][3] = {{1,2,3},{1,2,3},{1,2,3}};

      std::array<B,42> a4;
      B a5[42];

      std::array<float,3> a6 {{3,6,9}};
      float a7[3] = {3,6,9};

      std::array<std::array<float,3>,3> a8 {{ {{1,2,3}},{{1,2,3}},{{1,2,3}} }};
      float a9[3][3] = {{1,2,3},{1,2,3},{1,2,3}};
   };

}

)";



int modelReadDict2(const char* filename) {
   // This is with dictionaries
   unique_ptr<TFile> f (TFile::Open (filename));

   auto classFileName = "modelForReadDict2.cpp";
   {
      ofstream classFile (classFileName);
      if (classFile.is_open())
      {
         classFile <<modelForReadDict2_code;
      }
   }
   string line = ".L ";
   line += classFileName;
   line += "++";
   gErrorIgnoreLevel = 1001;
   gROOT->ProcessLine(line.c_str());
   gErrorIgnoreLevel = -1;
   auto className = "edm2::A";
   dumpInfo(className);

   return 0;
}

