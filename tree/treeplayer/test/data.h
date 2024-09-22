#ifndef TTREEREADER_LEAF_TEST_DATA
#define TTREEREADER_LEAF_TEST_DATA

#include <vector>
class Data {
   unsigned int fUSize{};
   int fSize{};
   double *fArray{nullptr}; //[fSize]
   float *fUArray{nullptr}; //[fUSize]
   std::vector<double> fVec{};
   Double32_t fDouble32{};
   Float16_t fFloat16{};

public:
   Data() = default;
   Data(const Data &) = delete;
   Data &operator=(const Data &) = delete;
   Data(Data &&) = delete;
   Data &operator=(Data &&) = delete;
   ~Data()
   {
      delete[] fArray;
      fArray = nullptr;
      delete[] fUArray;
      fUArray = nullptr;
   }
   void Init()
   {
      fUSize = 2;
      fSize = 4;
      fArray = new double[4]{12., 13., 14., 15.};
      fUArray = new float[2]{42., 43.};
      fVec = {17., 18., 19., 20., 21., 22.};
      fDouble32 = 17.;
      fFloat16 = 44.;
   }
   std::vector<double> &GetVecMember() { return fVec; }
};

struct V {
   int a = -100;
};

struct W {
   int b;
   V v;
};

#endif
