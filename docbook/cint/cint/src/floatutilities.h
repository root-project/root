// Code provided by Theodor Rascanu (	rascanu@ikf.uni-frankfurt.de )

//class BitPatternCompare
namespace Cint {
   namespace FloatUtilities {
#if defined(CINT__BITPATTERN_IS_SUPPORTED)
      namespace BitPatternCompare
      {
         union FloatUnion
         {
            float value;
            struct
            {
               unsigned long fraction : 23;
               unsigned int exponent : 8;
               unsigned short sign : 1;
            };
         };
         
         union DoubleUnion
         {
            double value;
            struct
            {
               unsigned long fraction : 52;
               unsigned int exponent : 11;
               unsigned short sign : 1;
            };
         };
         
         //public:
         
         static bool isnan(float infloat);
         static bool isinf(float infloat);
         
         static bool isnan(double infloat);
         static bool isinf(double infloat);
         
      }
#endif
      namespace DirectCompare {

#if !defined(isnan) && !defined(__SUNPRO_CC)
         //since nan are always inequal to anything, even to itself
         bool isnan(const double &x) { return ((x) != (x)); }
         bool isnan(const float &x)  { return ((x) != (x)); }
#endif
         bool isinfornan(const double &x) { return ((x-x) != (x-x)); }
         bool isinfornan(const float &x)  { return ((x-x) != (x-x)); }

#ifndef isinf
         bool isinf(const double &x) { return ( (isnan(x) && isinfornan(x)) ? !isnan(x) : isinfornan(x) ); }
         bool isinf(const float &x) { return ( (isnan(x) && isinfornan(x)) ? !isnan(x) : isinfornan(x) ); }
#endif
      }
   }
}

#if defined(CINT__BITPATTERN_IS_SUPPORTED)
bool Cint::FloatUtilities::BitPatternCompare::isnan(float infloat)
{
   FloatUnion ux;
   FloatUnion nan;
   nan.exponent=(unsigned int)-1;
   ux.value=infloat;
   return ( ux.exponent == nan.exponent ) && ( ux.fraction != 0 );
}

bool Cint::FloatUtilities::BitPatternCompare::isinf(float infloat)
{
   FloatUnion ux;
   FloatUnion nan;
   nan.exponent=(unsigned int)-1;
   ux.value=infloat;
   return ( ux.exponent == nan.exponent ) && ( ux.fraction == 0 );
}

bool Cint::FloatUtilities::BitPatternCompare::isnan(double infloat)
{
   DoubleUnion ux;
   DoubleUnion nan;
   nan.exponent=(unsigned int)-1;
   ux.value=infloat;
   return ( ux.exponent == nan.exponent ) && ( ux.fraction != 0 );
}

bool Cint::FloatUtilities::BitPatternCompare::isinf(double infloat)
{
   DoubleUnion ux;
   DoubleUnion nan;
   nan.exponent=(unsigned int)-1;
   ux.value=infloat;
   return ( ux.exponent == nan.exponent ) && ( ux.fraction == 0 );
}
#endif
