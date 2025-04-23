#ifndef CLASS6_H
#define CLASS6_H

#include "class3.h"

class class6 {

public:
   class6(int i);
   ~class6();
private:
   const int m_arrSize;
   int* m_arr1; //[m_arrSize]
   int* m_arr2; ///<[m_arrSize]
   int* m_arr3; // [m_arrSize]
   int* m_arr4; ///<      [m_arrSize]
   int m_transient1; //!This member is transient
   int m_transient2; ///<!This member is transient and we use doxygen comments
   int m_transient3; //  !This member is transient
   int m_transient4; ///<   !This member is transient and we use doxygen comments
   int m_transient5; //!<!This member is transient and we use doxygen comments
   int m_transient6; //!<   !This member is transient and we use doxygen comments
   class3 m_dontSplit1; //|| Please do not split me!
   class3 m_dontSplit2; ///<|| Please do not split me (Doxy)!
   class3 m_dontSplit3; //    || Please do not split me!
   class3 m_dontSplit4; ///<     || Please do not split me (Doxy)!
   Double32_t m_d321;     //[0,  0, 14] Hi I am opaque
   Double32_t m_d322;     ///<[0,  0, 14] Hi I am opaque in Doxygen
   Double32_t m_d323;     //     [0,  0, 14] Hi I am opaque
   Double32_t m_d324;     ///<  [0,  0, 14] Hi I am opaque in Doxygen
   int m_testComment; //  This is an important member and should not be transient !
   
};
#endif

class6::class6(int i):
  m_arrSize(i),
  m_arr1(new int[i]),
  m_arr2(new int[i]),
  m_arr3(new int[i]),
  m_arr4(new int[i]){};
  
class6::~class6()
{
   delete[] m_arr1;
   delete[] m_arr2;
   delete[] m_arr3;
   delete[] m_arr4;
}
