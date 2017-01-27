#include "ROOT/TDataFrame.hxx"

#include <iostream>
#include <memory>
#include <type_traits>
#include <typeinfo>
#include <vector>

struct Dummy {};

int freeFun1(int) { return 0; }
Dummy freeFun2(Dummy&, int*, const std::vector<Dummy>&) { return Dummy(); }

struct Functor1 {
   void operator()(int) { return; }
};

struct Functor2 {
   Dummy operator()(Dummy&, int*, const std::vector<Dummy>&) { return Dummy(); }
};

   

int main() {
// check TFunctionTraits with one arg, two args, return type, nodecaytypes
   using namespace ROOT::Internal::TDFTraitsUtils;

   // free function
   static_assert(std::is_same<TTypeList<int>,
                              TFunctionTraits<decltype(freeFun1)>::ArgTypes_t>::value,
                 "freeFun1 arg types");
   static_assert(std::is_same<TTypeList<int>,
                              TFunctionTraits<decltype(freeFun1)>::ArgTypesNoDecay_t>::value,
                 "freeFun1 arg types no decay");
   static_assert(std::is_same<int,
                              TFunctionTraits<decltype(freeFun1)>::RetType_t>::value,
                 "freeFun1 ret type");
   static_assert(std::is_same<TTypeList<Dummy, int*, std::vector<Dummy>>,
                              TFunctionTraits<decltype(freeFun2)>::ArgTypes_t>::value,
                 "freeFun2 arg types");
   static_assert(std::is_same<TTypeList<Dummy&, int*, const std::vector<Dummy>&>,
                              TFunctionTraits<decltype(freeFun2)>::ArgTypesNoDecay_t>::value,
                 "freeFun2 arg types no decay");
   static_assert(std::is_same<Dummy,
                              TFunctionTraits<decltype(freeFun2)>::RetType_t>::value,
                 "freeFun2 ret type");

   // function pointer
   auto freeFun1Ptr = &freeFun1;
   auto freeFun2Ptr = &freeFun2;
   static_assert(std::is_same<TTypeList<int>,
                              TFunctionTraits<decltype(freeFun1Ptr)>::ArgTypes_t>::value,
                 "freeFun1Ptr arg types");
   static_assert(std::is_same<TTypeList<int>,
                              TFunctionTraits<decltype(freeFun1Ptr)>::ArgTypesNoDecay_t>::value,
                 "freeFun1Ptr arg types no decay");
   static_assert(std::is_same<int,
                              TFunctionTraits<decltype(freeFun1Ptr)>::RetType_t>::value,
                 "freeFun1Ptr ret type");
   static_assert(std::is_same<TTypeList<Dummy, int*, std::vector<Dummy>>,
                              TFunctionTraits<decltype(freeFun2Ptr)>::ArgTypes_t>::value,
                 "freeFun2Ptr arg types");
   static_assert(std::is_same<TTypeList<Dummy&, int*, const std::vector<Dummy>&>,
                              TFunctionTraits<decltype(freeFun2Ptr)>::ArgTypesNoDecay_t>::value,
                 "freeFun2Ptr arg types no decay");
   static_assert(std::is_same<Dummy,
                              TFunctionTraits<decltype(freeFun2Ptr)>::RetType_t>::value,
                 "freeFun2Ptr ret type");

   // functor class
   static_assert(std::is_same<TTypeList<int>,
                              TFunctionTraits<Functor1>::ArgTypes_t>::value,
                 "Functor1 arg types");
   static_assert(std::is_same<TTypeList<int>,
                              TFunctionTraits<Functor1>::ArgTypesNoDecay_t>::value,
                 "Functor1 arg types no decay");
   static_assert(std::is_same<void,
                              TFunctionTraits<Functor1>::RetType_t>::value,
                 "Functor1 ret type");
   static_assert(std::is_same<TTypeList<Dummy, int*, std::vector<Dummy>>,
                              TFunctionTraits<Functor2>::ArgTypes_t>::value,
                 "Functor2 arg types");
   static_assert(std::is_same<TTypeList<Dummy&, int*, const std::vector<Dummy>&>,
                              TFunctionTraits<Functor2>::ArgTypesNoDecay_t>::value,
                 "Functor2 arg types no decay");
   static_assert(std::is_same<Dummy,
                              TFunctionTraits<Functor2>::RetType_t>::value,
                 "Functor2 ret type");

   // mutable lambda
   auto boolvec = std::make_shared<std::vector<bool>>(2, false);
   auto lambda1 = [boolvec](const bool b) mutable -> std::vector<bool>& {
      boolvec->at(0) = b;
      return *boolvec;
   };

   // lambda
   auto lambda2 = [](Dummy&, int*, const std::vector<Dummy>&) { return Dummy(); };
   static_assert(std::is_same<TTypeList<bool>,
                              TFunctionTraits<decltype(lambda1)>::ArgTypes_t>::value,
                 "lambda1 arg types");
   static_assert(std::is_same<TTypeList<bool>,
                              TFunctionTraits<decltype(lambda1)>::ArgTypesNoDecay_t>::value,
                 "lambda1 arg types no decay");
   static_assert(std::is_same<std::vector<bool>&,
                              TFunctionTraits<decltype(lambda1)>::RetType_t>::value,
                 "lambda1 ret type");
   static_assert(std::is_same<TTypeList<Dummy, int*, std::vector<Dummy>>,
                              TFunctionTraits<decltype(lambda2)>::ArgTypes_t>::value,
                 "lambda2 arg types");
   static_assert(std::is_same<TTypeList<Dummy&, int*, const std::vector<Dummy>&>,
                              TFunctionTraits<decltype(lambda2)>::ArgTypesNoDecay_t>::value,
                 "lambda2 arg types no decay");
   static_assert(std::is_same<Dummy,
                              TFunctionTraits<decltype(lambda2)>::RetType_t>::value,
                 "lambda2 ret type");

   // std::function
   // masking signature int(int) of freeFunc1
   std::function<int(double)> stdFun1(freeFun1);
   // masking signature of lambda2
   std::function<Dummy(Dummy&, int*, const std::vector<Dummy>&)> stdFun2(lambda2);
   static_assert(std::is_same<TTypeList<double>,
                              TFunctionTraits<decltype(stdFun1)>::ArgTypes_t>::value,
                 "stdFun1 arg types");
   static_assert(std::is_same<TTypeList<double>,
                              TFunctionTraits<decltype(stdFun1)>::ArgTypesNoDecay_t>::value,
                 "stdFun1 arg types no decay");
   static_assert(std::is_same<int,
                              TFunctionTraits<decltype(stdFun1)>::RetType_t>::value,
                 "stdFun1 ret type");
   static_assert(std::is_same<TTypeList<Dummy, int*, std::vector<Dummy>>,
                              TFunctionTraits<decltype(stdFun2)>::ArgTypes_t>::value,
                 "stdFun2 arg types");
   static_assert(std::is_same<TTypeList<Dummy&, int*, const std::vector<Dummy>&>,
                              TFunctionTraits<decltype(stdFun2)>::ArgTypesNoDecay_t>::value,
                 "stdFun2 arg types no decay");
   static_assert(std::is_same<Dummy,
                              TFunctionTraits<decltype(stdFun2)>::RetType_t>::value,
                 "stdFun2 ret type");

}
