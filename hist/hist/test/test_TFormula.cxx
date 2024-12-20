#include "gtest/gtest.h"

#include "TFormula.h"

#include <locale.h>

#ifdef R__HAS_GEOM
// Test that autoloading works (ROOT-9840)
TEST(TFormula, Interp)
{
   TFormula f("func", "TGeoBBox::DeclFileLine()");
}
#endif

class localeRAII {
   std::string fLocale;

public:
   localeRAII() : fLocale(setlocale(LC_NUMERIC, nullptr)){};
   ~localeRAII() { setlocale(LC_NUMERIC, fLocale.c_str()); }
};

// #17225
TEST(TFormula, Locale)
{
   localeRAII lraii;
   setlocale(LC_NUMERIC, "de_AT.UTF-8");
   TFormula f0("f0", "gausn(x)");
   EXPECT_TRUE(f0.IsValid());
   TFormula f1("f1", "landau(x)");
   EXPECT_TRUE(f1.IsValid());
}
