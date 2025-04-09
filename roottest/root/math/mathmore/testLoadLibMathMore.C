#include "Math/SpecFunc.h"
#include "Math/ProbFunc.h"

#include "Math/PdfFuncMathMore.h" // For ROOT::Math::MathMoreLibrary::Load()
void testLoadLibMathMore() {

   ROOT::Math::MathMoreLibrary::Load();
   cout<< ROOT::Math::cyl_bessel_i(1,2)   <<endl;
   cout<< ROOT::Math::noncentral_chisquared_pdf(2,3,1)  <<endl;

}
