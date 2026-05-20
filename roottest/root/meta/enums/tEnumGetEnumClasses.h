#ifndef tEnumGetEnumClasses_h
#define tEnumGetEnumClasses_h

enum enum1{enc1,enc2,enc3};

namespace ns1{
   enum enum2{enc1,enc2,enc3};
   namespace ns2{
      enum enum3{enc1,enc2,enc3};
   }
}

class class1{
public:
   enum enum4{enc1,enc2,enc3};
};

namespace ns3 {
   class class2{
      public:
         enum enum5{enc1,enc2,enc3};
   };
}


template <class T, int I>
class class3{
public:
   enum enum6{enc1,enc2,enc3};
};

namespace ns4 {
   template <class T, int I>
   class class4{
      public:
         enum enum6{enc1,enc2,enc3};
   };
}

/*
 * List of enums:
 * enum1
 * ns1::enum2
 * ns1::ns2::enum3
 * class1::enum4
 * class3<double,1>::enum6
 * ns4::class4<std::string,1>::enum6
 * */

#endif
