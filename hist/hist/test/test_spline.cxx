#include "TSpline.h"
#include "TInterpreter.h"
#include "TSystem.h"

#include "gtest/gtest.h"

#include <fstream>
#include <string>
#include <cmath>

class FileDeleterRAII {
   const std::string fFileName;

public:
   FileDeleterRAII(const std::string &fileName) : fFileName(fileName) {}
   ~FileDeleterRAII() { gSystem->Unlink(fFileName.c_str()); }
};

// issue #12091
TEST(TSpline, Precision)
{  
   // Prepare the reference
   const int n = 11;
   Double_t x[n], y[n];
   for (unsigned int i = 0; i < n; ++i) {
      x[i] = i * 0.1;
      y[i] = sqrt(2.) * pow(x[i], 2); // simple quadratic polynomial
   }

   // Prepare the spline and save it as a macro
   const auto splineName = "splinePrecisionTest";
   TSpline5 s(splineName, 0, 1, y, n);
   std::string macroName = splineName;
   macroName += ".C";
   FileDeleterRAII fdRAII(macroName);
   s.SaveAs(macroName.c_str());

   // Interpret the spline macro and get out of the interpreter the function obtained
   std::ifstream ifs(macroName);
   std::string codeToJIT((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
   codeToJIT+=splineName;
   codeToJIT+=";";
   auto spline = (double (*)(double))gInterpreter->ProcessLine(codeToJIT.c_str());

   const auto tolerance = 0.000001;
   for (unsigned int i = 0; i < n; ++i) {
      const auto splineVal = spline(x[i]);
      const auto expectedVal = y[i];
      EXPECT_NEAR(y[i], spline(x[i]), tolerance)
         << "Spline value (" << splineVal << ") and expected value (" << expectedVal << ") differ more than allowed ("
         << tolerance << ")" << std::endl;
   }
}
