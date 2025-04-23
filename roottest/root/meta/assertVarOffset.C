// Test ROOT-7306 and friends.
// One friend being ROOT-7459.

#include <iostream>

int N1 = 42;
const int N2 = 13;
float f1 = 3.14;
const float f2 = 2.72;

template <class T>
struct allowed_delta {
   using Delta_t = decltype (T() - T());
   static constexpr Delta_t val = Delta_t();
};

template <>
struct allowed_delta<float> {
   using Delta_t = float;
   static constexpr float val = 1E-3;
};
template <>
struct allowed_delta<double> {
   using Delta_t = double;
   static constexpr double val = 1E-6;
};


int exitCode = 0;

template <class T>
void assertEqual(T expect, T actual, typename allowed_delta<T>::Delta_t allowed = allowed_delta<T>::val) {
   auto delta = actual - expect;
   if (   (delta > 0 && delta > allowed)
       || (delta < 0 && -delta > allowed)) {
      std::cerr << "ERROR: expected " << expect << ", got " << actual
                << std::endl;
      exitCode = 1;
   }
}

int assertVarOffset() {
   assertEqual(&N1, (int*)gROOT->GetGlobal("N1")->GetAddress());
   assertEqual(N2, *(int*)gROOT->GetGlobal("N2")->GetAddress());
   assertEqual(f1, *(float*)gROOT->GetGlobal("f1")->GetAddress());
   assertEqual(f2, *(float*)gROOT->GetGlobal("f2")->GetAddress());
   return exitCode;
}
