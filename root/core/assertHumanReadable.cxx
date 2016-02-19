#include "ROOT/StringConv.h"
#include "TError.h"
#include "TString.h"

int checkFail(std::string_view input)
{
   Long64_t res = ROOT::FromHumanReadableSize(input);
   if (res != -1) {
      Error("FromHumanReadableSize","Parsing %s should have failed but got the value %lld instead of -1",
            input.to_string().c_str(),res);
      return 1;
   } else {
      return 0;
   }
}

int checkParsing(std::string_view input, Long64_t expected)
{
   Long64_t res = ROOT::FromHumanReadableSize(input);
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
