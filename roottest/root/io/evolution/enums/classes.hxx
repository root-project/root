#include <vector>
#include "TNamed.h"

#include <type_traits>
#include <iostream>
#include <iomanip>

#include <utility>

#if 0 /* old files */
#define CalArrayVersion 4
#else
#define CalArrayVersion 5
#endif

template <typename E>
void LoadEnumCollection(/* const */ std::vector<int> &in_onfile, std::vector<E> &enums)
{
   auto &onfile = reinterpret_cast<std::vector<E>&>(in_onfile);
   if (gDebug > 1 ) {
      std::cout << "Running LoadEnumCollection on: " << (void*)&onfile << " and " << (void*)&enums << '\n';
      std::cout << "Onfile number of elements: " << onfile.size() << '\n';
      for(auto in : onfile)
         std::cout << "Value: " << std::hex << static_cast<int>( in ) << '\n';
  
      auto &alt = reinterpret_cast<std::vector<int>&>(onfile);
      std::cout << "Alternate Onfile number of elements: " << alt.size() << '\n';
      for(auto in : alt)
         std::cout << "Value: " << std::hex << static_cast<int>( in ) << '\n';
   }   


   constexpr size_t delta = sizeof(int)/sizeof(E);
   const size_t nvalues = onfile.size() / delta;
   onfile.resize(nvalues);
   std::swap(onfile, enums);

   if (gDebug > 1) {
      std::cout << "Output size: " << enums.size() << '\n';
      for(auto out : enums)
         std::cout << "Value: " << static_cast<int>( out ) << '\n';
   }
};


enum class PadFlags : unsigned short {
   kConst,
   kOne,
   kTwo,
   kThree,
   kFour
};

enum class PadSubset : unsigned char
{
   kZero, kOne, kTwo, kThree, kFour, kFive = 5
};

template <typename Flags>
class CalArray {
public:
   std::vector<Flags> mFlags;
   PadSubset mPadSubset;
   virtual ~CalArray() = default;

   ClassDef(CalArray, CalArrayVersion);
};

template <typename Flags>
class CalArray2 : public CalArray<Flags> {
public:
   CalArray2(CalArray<Flags> in) : CalArray<Flags>(in) {}
   CalArray2() = default;
};

#include "TBuffer.h"

template <typename Flags>
inline void CalArray<Flags>::Streamer(TBuffer &R__b)
{
   // Stream an object of class CalArray<PadFlags>.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v <= 4) {
         {
            UInt_t start, count;
            /* Version_t vers = */ R__b.ReadVersion(&start, &count);

            std::vector<int> R__stl;
            R__stl.clear();
            int R__n;
            R__b >> R__n;
            R__stl.reserve(R__n);
            for (int R__i = 0; R__i < R__n; R__i++) {
               Int_t readtemp;
               R__b >> readtemp;
               R__stl.push_back(readtemp);
            }
            R__b.CheckByteCount(start, count, "stl collection of enums");

            mFlags.clear();
            auto data = reinterpret_cast<unsigned short*>(R__stl.data());
            for(int i = 0; i < R__n; ++i)
               mFlags.push_back(static_cast<PadFlags>( data[i] ));
         }
         int tmp;
         R__b >> tmp;
         mPadSubset = static_cast<PadSubset>(tmp);

         R__b.CheckByteCount(R__s, R__c, CalArray::IsA());
      } else {
         R__b.ReadClassBuffer(CalArray<Flags>::Class(), this, R__v, R__s, R__c);
      }
   } else {
      R__b.WriteClassBuffer(CalArray<Flags>::Class(),this);
   }
}

class Event {
public:
   PadFlags mMainFlag = PadFlags::kTwo;
   char mCanary1 = 101; //!
   PadFlags mSecondFlag = PadFlags::kOne;
   char mCanary2 = 102; //!
   std::vector<CalArray<PadFlags>> mData;
   std::vector<PadFlags> mFlags;
};

class UnevenEvent {
public:
   std::vector<PadSubset> mChar;
   std::vector<PadFlags> mPad;
};


