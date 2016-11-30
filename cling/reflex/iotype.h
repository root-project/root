#ifndef IOTYPE_H
#define IOTYPE_H

#include "Rtypes.h"
#include <vector>
#include <cstdlib>

template <typename T>
class Template {
public:
   Template() {}
   void Set(double& v);
private:
   T m[12];
};

class CIoType {
public:
   void Set(double v);
private:
   std::vector<Double_t> vd;
   std::vector<Double32_t> vd32;
   std::vector<double> vdd32;
   std::vector<float> vf;

   double d;
   Double32_t d32;
   Double_t dd32;
   Float_t f;

protected:
//    std::vector<Template<Double_t> > vTd;
   std::vector<Template<Double32_t> > vTd32;
   std::vector<Template<Double_t> > vTdd32;
   std::vector<Template<float> > vTf;

   Template<std::vector<double> > Tvd;
   Template<std::vector<Double32_t> > Tvd32;
   Template<std::vector<Double_t> > Tvdd32;
   Template<std::vector<Float_t> > Tvf;

public:
//    Template<double> Td;
   Template<Double32_t> Td32;
   Template<double> Tdd32;
   Template<Float_t> Tf;
};

template <typename T>
struct SetterT {
   SetterT(T& /*what*/, double /*v*/) { printf("ERROR: generic setter is called!\n"); exit(1);}
};

template <typename T>
void SetT(T& what, double& v) { SetterT<T>(what, ++v); }

template <>
struct SetterT<double> {
   SetterT(double& what, double& v) { what = ++v; }
};

template <>
struct SetterT<float> {
   SetterT(float& what, double& v) { what = ++v; }
};

template <typename T>
struct SetterT<Template<T> > {
   SetterT(Template<T>& what, double& v) { what.Set(++v); }
};

template <typename T>
struct SetterT<std::vector<T> > {
   SetterT(std::vector<T>& what, double& v) {
      what.clear();
      size_t n = v;
      for (size_t i = 0; i < n; ++i) {
         T t;
         SetT(t, ++v);
         what.push_back(t);
      }
      v = n; // don't increase
   }
};

template<typename T>
void Template<T>::Set(double& v) {
   static const int N = sizeof(m) / sizeof(T);
   for (int i = 0; i < N; ++i) {
      SetT(m[i], ++v);
   }
}

inline void CIoType::Set(double x) {
   double v = x;
   SetT(vd, v);
   SetT(vd32, v = x);
   SetT(vdd32, v = x);
   SetT(vf, v = x);

   SetT(d, v = x);
   SetT(d32, v = x);
   SetT(dd32, v = x);
   SetT(f, v = x);

//    SetT(vTd, v = x);
   SetT(vTd32, v = x);
   SetT(vTdd32, v = x);
   SetT(vTf, v = x);

   SetT(Tvd, v = x);
   SetT(Tvd32, v = x);
   SetT(Tvdd32, v = x);
   SetT(Tvf, v = x);

//    SetT(Td, v = x);
   SetT(Td32, v = x);
   SetT(Tdd32, v = x);
   SetT(Tf, v = x);
}

template class Template<double>;
//template class Template<Double32_t>;
template class Template<float>;
template class Template<std::vector<double> >;
//template class Template<std::vector<Double32_t> >;
template class Template<std::vector<float> >;

#endif
