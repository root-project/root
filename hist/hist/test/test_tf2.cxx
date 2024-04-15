#include "TF2.h"

#include "gtest/gtest.h"

TEST(TF2, Save)
{
   TF2 linear("linear", "x+y", -10, 10, -10, 10);
   linear.SetNpx(20);
   linear.SetNpy(20);

   Double_t args[2];

   // store at exactly defined range
   linear.Save(-10, 10, -10, 10, 0, 0);

   // test exact position
   for (Double_t x = -10.; x <= 10.; x += 1.) {
      for (Double_t y = -10.; y <= 10.; y += 1.) {
         args[0] = x;
         args[1] = y;
         EXPECT_NEAR(x + y, linear.GetSave(args), 1e-10);
      }
   }

   // test approximation
   for (Double_t x = -10.; x <= 10.; x += 0.33) {
      for (Double_t y = -10.; y <= 10.; y += 0.44) {
         args[0] = x;
         args[1] = y;
         EXPECT_NEAR(x + y, linear.GetSave(args), 1e-10);
      }
   }

   // test boundaries
   for (Double_t x = -11.; x <= 11.; x += 22) {
      for (Double_t y = -11.; y <= 11.; y += 22) {
         args[0] = x;
         args[1] = y;
         EXPECT_EQ(0, linear.GetSave(args));
      }
   }


   // store at middle of bins
   linear.Save(0, 0, 0, 0, 0, 0);

   // test exact position
   for (Double_t x = -9.5; x <= 9.5; x += 1.) {
      for (Double_t y = -9.5; y <= 9.5; y += 1.) {
         args[0] = x;
         args[1] = y;
         EXPECT_NEAR(x + y, linear.GetSave(args), 1e-10);
      }
   }

   // test approximation
   for (Double_t x = -9.5; x <= 9.5; x += 0.33) {
      for (Double_t y = -9.5; y <= 9.5; y += 0.44) {
         args[0] = x;
         args[1] = y;
         EXPECT_NEAR(x + y, linear.GetSave(args), 1e-10);
      }
   }

   // test boundaries
   for (Double_t x = -11.; x <= 11.; x += 22) {
      for (Double_t y = -11.; y <= 11.; y += 22) {
         args[0] = x;
         args[1] = y;
         EXPECT_EQ(0, linear.GetSave(args));
      }
   }

}
