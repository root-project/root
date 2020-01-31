#include "TBase64.h"

#include <iostream>
using namespace std;

void base64()
{
   TString binary1 = "This is test binary";

   TString coded = TBase64::Encode(binary1.Data(), binary1.Length());

   std::cout << "base64: " << coded << std::endl;

   TString binary2 = TBase64::Decode(coded);

   if (binary1 == binary2)
      std::cout << "base64: coding1 match" << std::endl;
   else
      std::cout << "base64: coding1 mismatch" << std::endl;


   binary1 = "Other test string";

   coded = TBase64::Encode(binary1.Data(), binary1.Length());

   std::cout << "base64: " << coded << std::endl;

   binary2 = TBase64::Decode(coded);

   if (binary1 == binary2)
      std::cout << "base64: coding2 match" << std::endl;
   else
      std::cout << "base64: coding2 mismatch" << std::endl;
}
