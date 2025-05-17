#include <tuple>
#include <limits>
#include "TFile.h"
#include "TError.h"

class ShortClass {
public:
   char fValue;
};

#ifdef __ROOTCLING__
#pragma link C++ class tuple<short, double, ShortClass, char, long>+;
#endif



bool tester(unsigned int component, long value, long expected)
{
   if (value != expected) {
      Error("read","Component %d incorrect read %ld instead of %ld",component, value, expected);
      return false;
   }
   return true;
}

bool tester(unsigned int component, char value, char expected)
{
   if (value != expected) {
      Error("read","Component %d incorrect read %d instead of %d",component, value, expected);
      return false;
   }
   return true;
}

bool tester(unsigned int component, short value, short expected)
{
   if (value != expected) {
      Error("read","Component %d incorrect read %d instead of %d",component, value, expected);
      return false;
   }
   return true;
}

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp)
{
   // the machine epsilon has to be scaled to the magnitude of the values used
   // and multiplied by the desired precision in ULPs (units in the last place)
   return std::abs(x-y) < std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
          // unless the result is subnormal
          || std::abs(x-y) < std::numeric_limits<T>::min();
}

bool tester(unsigned int component, double value, double expected)
{
   if (!almost_equal(value,expected,1)) {
      Error("read","Component %d incorrect read %f instead of %f", component, value, expected);
      return false;
   }
   return true;
}


void write(const char *filename = "tuple.root")
{
   Printf("Writing %s",filename);
   TFile f(filename,"RECREATE");

   tuple<short, double, ShortClass, char, long> data;
   std::get<0>(data) = 33;
   std::get<1>(data) = 200/3.0;
   std::get<2>(data).fValue = 13;
   std::get<3>(data) = 7;
   std::get<4>(data) = (long)2 ^ (long)7;

   f.WriteObject(&data,"tupleData");
}

int read(const char *filename = "tuple.root")
{
   Printf("Reading %s",filename);
   TFile f(filename,"READ");
   tuple<short, double, ShortClass, char, long> *data{nullptr};

   f.GetObject("tupleData",data);

   if (!data) {
      Error("read","Could not read the tuple object");
      return 1;
   }

   unsigned short result = 0;
   if (!tester(0, std::get<0>(*data), 33)) result += 2;
   if (!tester(1, std::get<1>(*data), 200/3.0)) result += 4;
   if (!tester(2, std::get<2>(*data).fValue, 13)) result += 8;
   if (!tester(3, std::get<3>(*data), 7)) result += 16;
   if (!tester(4, std::get<4>(*data), (long)2 ^ (long)7)) result += 32;

   return result;
}

int execTuple()
{
   write();
   return read() + read("tuple.macos.root") + read("tuple.x86_64.root");
}
