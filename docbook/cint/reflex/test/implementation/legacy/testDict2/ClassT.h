#ifndef DICT2_CLASST_T
#define DICT2_CLASST_T

template <class T1> class ClassT1 {
public:
   T1 fA; };
template <class T1, class T2> class ClassT2 {
public:
   T1 fA; T2 fB; };
template <class T1, class T2, class T3, class T4> class ClassT4 {
public:
   T1 fA; T2 fB; T3 fC; T4 fD; };
template <class T1, class T2, class T3, class T4, class T5, class T6> class ClassT6 {
public:
   T1 fA; T2 fB; T3 fC; T4 fD; T5 fE; T6 fF; };

namespace {
struct __TestDict2_Instances__ {
   ClassT1<int> i1;
   ClassT2<int, int> i2;
   ClassT4<int, int, int, int> i3;
   ClassT6<int, int, int, int, int, int> i4;
};
}

#endif // DICT2_CLASST_H
