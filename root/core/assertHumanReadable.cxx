#include "ROOT/StringConv.h"
#include "TError.h"
#include "TString.h"

int checkFail(std::string_view input)
{
   Long64_t res = -1;
   auto parseResult = ROOT::FromHumanReadableSize(input,res);
   if (parseResult == ROOT::EFromHumanReadableSize::kSuccess) {
      Error("FromHumanReadableSize","Parsing %s should have failed but got the value %lld instead of -1",
            input.to_string().c_str(),res);
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
            input.to_string().c_str());
      return 1;
   }
   if (parseResult == ROOT::EFromHumanReadableSize::kOverflow) {
      Error("FromHumanReadableSize","Overflow of %s which does not fit in a long long.",
            input.to_string().c_str());
      return 1;
   }
   if (res != expected) {
      Error("FromHumanReadableSize","Incorrectly parsed %s and got the value %lld instead of %lld",
            input.to_string().c_str(),res,expected);
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
      value.Form("%f%s %lld",val,unit,(long long)(exp*val));
      num_failed += checkParsing(value,val*exp);
   }
   value.Form("2%s",unit);
   num_failed += checkParsing(value,2*exp);
   value.Form("2.3%s",unit);
   num_failed += checkParsing(value,2.3*exp);
   return num_failed;
}

int assertFromHumanReadable()
{
   printf("Checking FromHumanReadableSize\n");
   int num_failed = 0;
   num_failed += checkFail("wrong");
   num_failed += checkFail("8ZB");
   num_failed += checkFail("9.3ZiB");
   num_failed += checkParsing("1",1);
   num_failed += checkParsing("1K",1024);
   num_failed += checkParsing("1Ki",1000);
   num_failed += checkParsing("1Kib",1000);
   num_failed += checkParsing("1KiB",1000);
   num_failed += checkParsing("1k",1024);
   num_failed += checkParsing("1.2 k",1.2*1024);


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
      p.first *= 1024;
      p.second*= 1000;
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
   num_failed += checkToHumanReadable(1024LL,false,1,"KB");
   num_failed += checkToHumanReadable(1024LL,true,1.024,"KiB");

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

      // printf("Checking %lld vs %g%s an %g%s\n",value,expectedSiCoeff,suffix[i][1],expectedCoeff,suffix[i][0]);
      num_failed += checkToHumanReadable(value,false,expectedCoeff,suffix[i][0]);
      num_failed += checkToHumanReadable(value,true,expectedSiCoeff,suffix[i][1]);

      value = 2 * 1024 * value;
      expectedSiCoeff = 2 * expectedSiCoeff * 1.024;
      expectedCoeff = 2 * expectedCoeff;
   }
   value = 2* std::pow(1024, 6);
   expectedSiCoeff = 2.30584300921;
   expectedCoeff = 2;
   num_failed += checkToHumanReadable(value,false,expectedCoeff,suffix[size-2][0]);
   num_failed += checkToHumanReadable(value,true,expectedSiCoeff,suffix[size-2][1]);


   double dvalue = 16;
   expectedCoeff = 16;
   expectedSiCoeff = 16;
   for(unsigned int i = 0; i < size; ++i) {

      // printf("Checking %g vs %g%s and %g%s\n",dvalue,expectedSiCoeff,suffix[i][1],expectedCoeff,suffix[i][0]);
      num_failed += checkToHumanReadable(dvalue,false,expectedCoeff,suffix[i][0]);
      num_failed += checkToHumanReadable(dvalue,true,expectedSiCoeff,suffix[i][1]);

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
