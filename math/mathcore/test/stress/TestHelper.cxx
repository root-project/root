#include "gtest/gtest.h"
#include "TROOT.h"
#include "TestHelper.h"
#include "TMath.h"

bool OutsideBounds(double v1, double v2, double scale)
{
   // numerical double limit for epsilon
   double eps = scale * std::numeric_limits<double>::epsilon();

   double delta = TMath::Abs(v2 - v1);

   if (v1 == 0 || v2 == 0) {
      if (delta > eps) {
         return false;
      }
   }
   // skip case v1 or v2 is infinity
   else {
      // add also case when delta is small by default (relative error + absolute error)
      if (delta / TMath::Abs(v1) > eps && delta > eps) {
         return false;
      }
   }

   return true;
}

// Compared to ASSERT_NEAR, this function takes into account also the relative error
::testing::AssertionResult IsNear(std::string name, double v1, double v2, double scale)
{
   if (!OutsideBounds(v1, v2, scale)) {
      double eps = scale * std::numeric_limits<double>::epsilon();
      return testing::AssertionFailure() << std::endl
                                         << "Discrepancy in " << name.c_str() << "() :  " << std::endl
                                         << v1 << " != " << v2
                                         << " discr = " << int(TMath::Abs(v2 - v1) / TMath::Abs(v1) / eps)
                                         << "   (Allowed discrepancy is " << eps << ")" << std::endl
                                         << std::endl;
   } else {
      return testing::AssertionSuccess();
   }
}
