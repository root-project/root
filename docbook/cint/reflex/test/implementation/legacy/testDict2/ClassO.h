#ifndef DICT2_CLASSO_H
#define DICT2_CLASSO_H

template <typename T>
class ClassO {
public:
   void
   P(T* p) { _p = p; }

   void
   R(T& r) { _p = &r; }

   void
   cP(const T* cp) { _cpp = &cp; }

   void
   cR(const T& cr) { *_cpp = &cr; }

   void
   cPc(const T* const cpc) { *_cpp = cpc; }

private:
   T* _p;
   T** _pp;
   const T** _cpp;
};

#include <vector>

namespace {
struct __TestDict2ClassO_Instances__ {
   ClassO<int> m;
   ClassO<int*> n;
   ClassO<ClassO<int> > o;
   ClassO<ClassO<int*>*> p;
   std::vector<ClassO<int*> > _v;
   std::vector<ClassO<int*>*> _vp;
   std::vector<const ClassO<int>*> _vcp;
   std::vector<const ClassO<int const*>* const*> _vcpp;
};
}

#endif // DICT2_CLASSO_H
