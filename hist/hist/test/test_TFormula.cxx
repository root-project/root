#include "gtest/gtest.h"

#include "TFormula.h"

// Test that autoloading works (ROOT-9840)
TEST(TFormula, Interp)
{
  TFormula f("func", "TGeoBBox::DeclFileLine()");
}
