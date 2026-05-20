#ifndef _ARRAYHOLDER_
#define _ARRAYHOLDER_

#include <array>
#include <vector>
#include <string>
#include <sstream>

template<class T = int>
class ArrayHolderT {
  public:
   void Set(T a, T b, T c) {
      m_a[0]=a;m_a[1]=b;m_a[2]=c;
      for (auto& v : {a,b,c}) {
         auto vprime=v+.5;
         m_v.emplace_back(vprime);
      }
   };
    ArrayHolderT(T a, T b, T c) {
       Set(a,b,c);
   };
   ArrayHolderT() {};
   std::string ToString() {
      std::stringstream ss;
      for (auto&& el : m_a) ss << el << "-";
      for (auto&& el : m_v) ss << el << "-";
      ss << std::endl;
      return ss.str();
   }
    std::vector<double> m_v {10,11};
#ifdef ARRAYHOLDER_STDARRAY
    std::array<T, 3> m_a {{0,0,0}};
#else
    T m_a[3] {0,0,0};
#endif
};

using ArrayHolder = ArrayHolderT<int>;

class MetaArrayHolder {
  public:
   void Set(int a, int b, int c) {
      m_a[0]={a,b,c};
      m_a[1]={2*a,2*b,2*c};
      m_a[2]={3*a,3*b,3*c};
   };
   MetaArrayHolder(int a, int b, int c){Set(a,b,c);};
   MetaArrayHolder() {
      Set(0,0,0);
   };
   std::string ToString() {
      std::stringstream ss;
      for (auto& el : m_a)
         ss << el.ToString();
      return ss.str();
   }
#ifdef ARRAYHOLDER_STDARRAY
    std::array<ArrayHolder, 3> m_a;
#else
    ArrayHolder m_a[3];
#endif
};

class MetaArrayHolder2 {
  public:
   void Set(int a, int b, int c) {
      m_a2.Set(a,b,c);
      }
   MetaArrayHolder2(int a, int b, int c){Set(a,b,c);};
   MetaArrayHolder2() {
      Set(0,0,0);
   };
   std::string ToString() {
      std::stringstream ss;
      ss << "MetaArrayHolder2: " << m_a2.ToString();
      return ss.str();
   }
private:
    ArrayHolder m_a2;
};




#ifdef __ROOTCLING__
#pragma link C++ class ArrayHolderT<int>+;
#pragma link C++ class MetaArrayHolder+;
#pragma link C++ class MetaArrayHolder2+;
#pragma link C++ class vector<MetaArrayHolder2>+;
#endif

#endif
