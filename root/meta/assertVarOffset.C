// Test ROOT-7306 and friends.

#include <iostream>

int N1 = 42;
const int N2 = 13;
float f1 = 3.14;
const float f2 = 2.72;

template <class T>
struct allowed_delta {
   static constexpr T val = T();
};

template <>
struct allowed_delta<float> {
   static constexpr float val = 1E-3;
};
template <>
struct allowed_delta<double> {
   static constexpr double val = 1E-6;
};


template <class T>
void assertEqual(T expect, T actual, T allowed = allowed_delta<T>::val) {
   T delta = actual - expect;
   if (delta * delta > allowed * allowed)
      std::cerr << "ERROR: expected " << expect << ", got " << actual
                << std::endl;
}

int assertVarOffset() {
   assertEqual(N1, *(int*)gROOT->GetGlobal("N1")->GetAddress());
   assertEqual(N2, *(int*)gROOT->GetGlobal("N2")->GetAddress());
   assertEqual(f1, *(float*)gROOT->GetGlobal("f1")->GetAddress());
   assertEqual(f2, *(float*)gROOT->GetGlobal("f2")->GetAddress());
   return 0;
}
