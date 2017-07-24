#include "TF1.h"
#include "TObjString.h"
// #include "TObject.h"

#include "gtest/gtest.h"

#include <iostream>

using namespace std;

Float_t delta = 0.00000000001;

void coeffNamesGeneric(TString *formula, TObjArray *coeffNames) {
   TF1 *cn0 = new TF1("cn0", *formula, 0, 1);
   ASSERT_EQ(cn0->GetNpar(), coeffNames->GetEntries());
   for (int i = 0; i < coeffNames->GetEntries(); i++) {
      TObjString *coeffObj = (TObjString *)coeffNames->At(i);
      TString coeffName = coeffObj->GetString();
      EXPECT_EQ(coeffName, TString(cn0->GetParName(i)));
   }
}

void test_coeffNames() {
   // cout << "About to start" << endl;
   
   TObjArray *coeffNames = new TObjArray();
   coeffNames->SetOwner(kTRUE);
   coeffNames->Add(new TObjString("sg"));
   coeffNames->Add(new TObjString("bg"));
   coeffNames->Add(new TObjString("Mean"));
   coeffNames->Add(new TObjString("Sigma"));
   coeffNames->Add(new TObjString("Slope"));
   TString *formula = new TString("NSUM([sg] * gaus, [bg] * expo)");

   // cout << "Almost done" << endl;
   
   coeffNamesGeneric(formula, coeffNames);
}

void test_normalization() {
   double xmin = -5;
   double xmax = 5;
   TF1 *n0 = new TF1("n0", "NSUM(.5 * gaus, .5 * (x+[0])**2)", xmin, xmax);
   EXPECT_NEAR(n0->Integral(xmin, xmax), 1, delta);
   n0->SetParameter(4,1); // should not affect integral
   EXPECT_NEAR(n0->Integral(xmin, xmax), 1, delta);
   n0->SetParameter(0,0);
   EXPECT_NEAR(n0->Integral(xmin, xmax), .5, delta);
   
   TF1 *n1 = new TF1("n1", "NSUM([sg] * gaus, [bg] * (x+[0])**2)", xmin, xmax);
   n1->SetParameter(0,.5);
   n1->SetParameter(1,.5);
   EXPECT_NEAR(n1->Integral(xmin, xmax), 1, delta);
   n0->SetParameter(0,0);
   EXPECT_NEAR(n0->Integral(xmin, xmax), .5, delta);

   TF1 *n2 = new TF1("n2", "NSUM([sg] * gaus, -0.5 * (x+[0])**2)", xmin, xmax);
   n2->SetParameter(0,.5);
   EXPECT_NEAR(n2->GetParameter(1), -.5, delta);
   EXPECT_NEAR(n2->Integral(xmin, xmax), 0, delta);
   n2->SetParameter(0,0);
   EXPECT_NEAR(n2->Integral(xmin, xmax), -.5, delta);
}

TEST(TF1, CoeffNames) {
   test_coeffNames();
}

TEST(TF1, Normalization) {
   test_normalization();
}
