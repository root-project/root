#include "ROOT/StringConv.hxx"
#include "TError.h"
#include "TString.h"
#include <limits>

int checkFail(std::string_view input)
{
   Long64_t res = -1;
   auto parseResult = ROOT::FromHumanReadableSize(input,res);
   if (parseResult == ROOT::EFromHumanReadableSize::kSuccess) {
      Error("FromHumanReadableSize","Parsing %s should have failed but got the value %lld instead of -1",
            std::string(input).c_str(),res);
      return 1;
   } else {
      return 0;
   }
}

int checkParsing(std::string_view input, Long64_t expected)
{
   Long64_t res = -1;
   auto parseResult = ROOT::FromHumanReadableSize(input,res);
   if (parseResult == ROOT::EFromHumanReadableSize::kParseFail) {
      Error("FromHumanReadableSize","Parsing %s failed.",
            std::string(input).c_str());
      return 1;
   }
   if (parseResult == ROOT::EFromHumanReadableSize::kOverflow) {
      Error("FromHumanReadableSize","Overflow of %s which does not fit in a long long.",
            std::string(input).c_str());
      return 1;
   }
   bool match = ( res == expected );
   if (!match && (abs(expected) >= std::numeric_limits<decltype(expected)>::max()/1000) ) {
      // We are close to the numeric limits, scale down the number to eliminate numerical imprecision.
      Long64_t resScaled = res / 1000;
      Long64_t expectedScaled = expected / 1000;
      match = (resScaled == expectedScaled);
   }
   if (!match) {
      Error("FromHumanReadableSize","Incorrectly parsed %s and got the value %lld instead of %lld",
            std::string(input).c_str(),res,expected);
      return 1;
   } else {
      return 0;
   }
}

int checkParsingSet(const char *unit, Long64_t exp)
{
   TString value;

   double values [] = {1.0,1.5,-2.33,7,-4};

   int num_failed = 0;

   value.Form("1%s",unit);
   num_failed += checkParsing(value,exp);
   value.Form("1 %s",unit);
   num_failed += checkParsing(value,exp);
   value.Form(" 1 %s",unit);
   num_failed += checkParsing(value,exp);

   for(auto &&val : values) {
     value.Form("%f%s %lld",val,unit,(long long)(val*exp));
      num_failed += checkParsing(value,val*exp);
   }
   value.Form("2%s",unit);
   num_failed += checkParsing(value,2*exp);
   value.Form("2.3%s",unit);
   // Avoid floating point imprecision on 32bit platform.
   Long64_t expectedValue;
   if (exp < std::numeric_limits<Long64_t>::max()/100)
     expectedValue = 23*exp/10;
   else
     expectedValue = 2.3*exp;
   num_failed += checkParsing(value,expectedValue);
   return num_failed;
}

int assertFromHumanReadable()
{
   printf("Checking FromHumanReadableSize\n");
   int num_failed = 0;
   num_failed += checkFail("wrong");
   num_failed += checkFail("K");
   num_failed += checkFail("8ZiB");
   num_failed += checkFail("9.3ZB");
   num_failed += checkParsing("1",1);
   num_failed += checkParsing("0K",0);
   num_failed += checkParsing("1Ki",1024);
   num_failed += checkParsing("1Kib",1024);
   num_failed += checkParsing("1KiB",1024);
   num_failed += checkParsing("1k",1000);
   num_failed += checkParsing("1.2 k",1200);
   num_failed += checkParsing("1.2 ki",1.2*1024);


   static const char *const suffix[][2] =
   { { "B",  "B"   },
      { "KB", "KiB" },
      { "MB", "MiB" },
      { "GB", "GiB" },
      { "TB", "TiB" },
      { "EB", "EiB" },
      { "ZB", "ZiB" } };
   static const unsigned int size = (sizeof(suffix) / sizeof(suffix[0]) );

   std::vector<std::pair<Long64_t,Long64_t> > exps;
   std::pair<Long64_t,Long64_t> p{1,1};

   exps.push_back(p);
   for(unsigned int i = 1; i < size; ++i) {
      p.first *= 1000;
      p.second*= 1024;
      exps.push_back(p);
   }

   for(unsigned int i = 1; i < size; ++i) {
      num_failed += checkParsingSet(suffix[i][0],exps[i].first);
      num_failed += checkParsingSet(suffix[i][1],exps[i].second);
   }
   return num_failed;
}


int checkToHumanReadable(Long64_t value, bool si, Double_t expectedCoeff, const char *expectedSuffix)
{

   Double_t coeff;
   const char *suffix = nullptr;

   ROOT::ToHumanReadableSize(value, si, &coeff, &suffix);
   if (!suffix) {
      Error("ToHumanReadableSize","Did not assign a suffix for %lld\n",value);
      return 1;
   }
   if (strcmp(expectedSuffix,suffix) != 0) {
      Error("ToHumanReadableSize","Did not get expected suffix for %lld, we got %g %s instead of %g %s\n",value,coeff,suffix,expectedCoeff,expectedSuffix);
      return 1;
   }
   if (expectedCoeff == 0 && coeff != 0) {
      Error("ToHumanReadableSize","Got un-expected coeff for %lld, we got %g %s instead of %g %s\n",value,coeff,suffix,expectedCoeff,expectedSuffix);
      return 1;
   }
   if ( (abs( coeff - expectedCoeff ) / expectedCoeff ) > 1e-10 ) {
      Error("ToHumanReadableSize","Got un-expected coeff for %lld, we got %g %s instead of %g %s\n",value,coeff,suffix,expectedCoeff,expectedSuffix);
      return 1;
   }
   return 0;
}

int checkToHumanReadable(Double_t value, bool si, Double_t expectedCoeff, const char *expectedSuffix)
{

   Double_t coeff;
   const char *suffix = nullptr;

   ROOT::ToHumanReadableSize(value, si, &coeff, &suffix);
   if (!suffix) {
      Error("ToHumanReadableSize(double)","Did not assign a suffix for %g\n",value);
      return 1;
   }
   if (strcmp(expectedSuffix,suffix) != 0) {
      Error("ToHumanReadableSize(double)","Did not get expected suffix for %g, we got %g %s instead of %g %s\n",value,coeff,suffix,expectedCoeff,expectedSuffix);
      return 1;
   }
   if (expectedCoeff == 0 && coeff != 0) {
      Error("ToHumanReadableSize(double)","Got un-expected coeff for %g, we got %g %s instead of %g %s\n",value,coeff,suffix,expectedCoeff,expectedSuffix);
      return 1;
   }
   if ( (abs( coeff - expectedCoeff ) / expectedCoeff ) > 1e-10 ) {
      Error("ToHumanReadableSize(double)","Got un-expected coeff for %g, we got %g %s instead of %g %s\n",value,coeff,suffix,expectedCoeff,expectedSuffix);
      return 1;
   }
   // printf("For %g (%d) got %g%s\n",value,si,coeff,suffix);

   return 0;
}

int assertToHumanReadable()
{
   printf("Checking ToHumanReadableSize\n");
   int num_failed = 0;

   num_failed += checkToHumanReadable(0LL,false,0,"B");
   num_failed += checkToHumanReadable(16LL,false,16,"B");
   num_failed += checkToHumanReadable(1024LL,false,1,"KiB");
   num_failed += checkToHumanReadable(0LL,true,0,"B");
   num_failed += checkToHumanReadable(16LL,true,16,"B");
   num_failed += checkToHumanReadable(1000LL,true,1,"KB");
   num_failed += checkToHumanReadable(1024LL,true,1.024,"KB");

   Long64_t value = 1024;
   double expectedCoeff = 1;
   double expectedSiCoeff = 1.024;

   static const char *const suffix[][2] =
   { { "B",  "B"   },
      { "KB", "KiB" },
      { "MB", "MiB" },
      { "GB", "GiB" },
      { "TB", "TiB" },
      { "EB", "EiB" },
      { "ZB", "ZiB" },
      { "YB", "YiB" } };
   static const unsigned int size = (sizeof(suffix) / sizeof(suffix[0]));

   for(unsigned int i = 1; i < size-2; ++i) {

      printf("Checking %lld vs %g%s an %g%s\n",value,expectedSiCoeff,suffix[i][1],expectedCoeff,suffix[i][0]);
      num_failed += checkToHumanReadable(value,false,expectedCoeff,suffix[i][1]);
      num_failed += checkToHumanReadable(value,true,expectedSiCoeff,suffix[i][0]);

      value = 2 * 1024 * value;
      expectedSiCoeff = 2 * expectedSiCoeff * 1.024;
      expectedCoeff = 2 * expectedCoeff;
   }
   value = 2* std::pow(1024, 6);
   expectedSiCoeff = 2.30584300921;
   expectedCoeff = 2;
   num_failed += checkToHumanReadable(value,false,expectedCoeff,suffix[size-2][1]);
   num_failed += checkToHumanReadable(value,true,expectedSiCoeff,suffix[size-2][0]);


   double dvalue = 16;
   expectedCoeff = 16;
   expectedSiCoeff = 16;
   for(unsigned int i = 0; i < size; ++i) {

      printf("Checking %g vs %g%s and %g%s\n",dvalue,expectedSiCoeff,suffix[i][1],expectedCoeff,suffix[i][0]);
      num_failed += checkToHumanReadable(dvalue,false,expectedCoeff,suffix[i][1]);
      num_failed += checkToHumanReadable(dvalue,true,expectedSiCoeff,suffix[i][0]);

      dvalue = 1.5 * 1024 * dvalue;
      expectedSiCoeff = 1.5 * expectedSiCoeff * 1.024;
      expectedCoeff = 1.5 * expectedCoeff;
   }


   return num_failed;
}

int assertHumanReadable() {
   int num_failed = assertFromHumanReadable();
   num_failed += assertToHumanReadable();
   return num_failed;
}
