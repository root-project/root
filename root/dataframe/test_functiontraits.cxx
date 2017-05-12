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
   using namespace ROOT::Internal::TDF;

   // free function
   static_assert(std::is_same<TTypeList<int>,
                              TFunctionTraits<decltype(freeFun1)>::Args_t>::value,
                 "freeFun1 arg types");
   static_assert(std::is_same<TTypeList<int>,
                              TFunctionTraits<decltype(freeFun1)>::ArgsNoDecay_t>::value,
                 "freeFun1 arg types no decay");
   static_assert(std::is_same<int,
                              TFunctionTraits<decltype(freeFun1)>::Ret_t>::value,
                 "freeFun1 ret type");
   static_assert(std::is_same<TTypeList<Dummy, int*, std::vector<Dummy>>,
                              TFunctionTraits<decltype(freeFun2)>::Args_t>::value,
                 "freeFun2 arg types");
   static_assert(std::is_same<TTypeList<Dummy&, int*, const std::vector<Dummy>&>,
                              TFunctionTraits<decltype(freeFun2)>::ArgsNoDecay_t>::value,
                 "freeFun2 arg types no decay");
   static_assert(std::is_same<Dummy,
                              TFunctionTraits<decltype(freeFun2)>::Ret_t>::value,
                 "freeFun2 ret type");

   // function pointer
   auto freeFun1Ptr = &freeFun1;
   auto freeFun2Ptr = &freeFun2;
   static_assert(std::is_same<TTypeList<int>,
                              TFunctionTraits<decltype(freeFun1Ptr)>::Args_t>::value,
                 "freeFun1Ptr arg types");
   static_assert(std::is_same<TTypeList<int>,
                              TFunctionTraits<decltype(freeFun1Ptr)>::ArgsNoDecay_t>::value,
                 "freeFun1Ptr arg types no decay");
   static_assert(std::is_same<int,
                              TFunctionTraits<decltype(freeFun1Ptr)>::Ret_t>::value,
                 "freeFun1Ptr ret type");
   static_assert(std::is_same<TTypeList<Dummy, int*, std::vector<Dummy>>,
                              TFunctionTraits<decltype(freeFun2Ptr)>::Args_t>::value,
                 "freeFun2Ptr arg types");
   static_assert(std::is_same<TTypeList<Dummy&, int*, const std::vector<Dummy>&>,
                              TFunctionTraits<decltype(freeFun2Ptr)>::ArgsNoDecay_t>::value,
                 "freeFun2Ptr arg types no decay");
   static_assert(std::is_same<Dummy,
                              TFunctionTraits<decltype(freeFun2Ptr)>::Ret_t>::value,
                 "freeFun2Ptr ret type");

   // functor class
   static_assert(std::is_same<TTypeList<int>,
                              TFunctionTraits<Functor1>::Args_t>::value,
                 "Functor1 arg types");
   static_assert(std::is_same<TTypeList<int>,
                              TFunctionTraits<Functor1>::ArgsNoDecay_t>::value,
                 "Functor1 arg types no decay");
   static_assert(std::is_same<void,
                              TFunctionTraits<Functor1>::Ret_t>::value,
                 "Functor1 ret type");
   static_assert(std::is_same<TTypeList<Dummy, int*, std::vector<Dummy>>,
                              TFunctionTraits<Functor2>::Args_t>::value,
                 "Functor2 arg types");
   static_assert(std::is_same<TTypeList<Dummy&, int*, const std::vector<Dummy>&>,
                              TFunctionTraits<Functor2>::ArgsNoDecay_t>::value,
                 "Functor2 arg types no decay");
   static_assert(std::is_same<Dummy,
                              TFunctionTraits<Functor2>::Ret_t>::value,
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
                              TFunctionTraits<decltype(lambda1)>::Args_t>::value,
                 "lambda1 arg types");
   static_assert(std::is_same<TTypeList<bool>,
                              TFunctionTraits<decltype(lambda1)>::ArgsNoDecay_t>::value,
                 "lambda1 arg types no decay");
   static_assert(std::is_same<std::vector<bool>&,
                              TFunctionTraits<decltype(lambda1)>::Ret_t>::value,
                 "lambda1 ret type");
   static_assert(std::is_same<TTypeList<Dummy, int*, std::vector<Dummy>>,
                              TFunctionTraits<decltype(lambda2)>::Args_t>::value,
                 "lambda2 arg types");
   static_assert(std::is_same<TTypeList<Dummy&, int*, const std::vector<Dummy>&>,
                              TFunctionTraits<decltype(lambda2)>::ArgsNoDecay_t>::value,
                 "lambda2 arg types no decay");
   static_assert(std::is_same<Dummy,
                              TFunctionTraits<decltype(lambda2)>::Ret_t>::value,
                 "lambda2 ret type");

   // std::function
   // masking signature int(int) of freeFunc1
   std::function<int(double)> stdFun1(freeFun1);
   // masking signature of lambda2
   std::function<Dummy(Dummy&, int*, const std::vector<Dummy>&)> stdFun2(lambda2);
   static_assert(std::is_same<TTypeList<double>,
                              TFunctionTraits<decltype(stdFun1)>::Args_t>::value,
                 "stdFun1 arg types");
   static_assert(std::is_same<TTypeList<double>,
                              TFunctionTraits<decltype(stdFun1)>::ArgsNoDecay_t>::value,
                 "stdFun1 arg types no decay");
   static_assert(std::is_same<int,
                              TFunctionTraits<decltype(stdFun1)>::Ret_t>::value,
                 "stdFun1 ret type");
   static_assert(std::is_same<TTypeList<Dummy, int*, std::vector<Dummy>>,
                              TFunctionTraits<decltype(stdFun2)>::Args_t>::value,
                 "stdFun2 arg types");
   static_assert(std::is_same<TTypeList<Dummy&, int*, const std::vector<Dummy>&>,
                              TFunctionTraits<decltype(stdFun2)>::ArgsNoDecay_t>::value,
                 "stdFun2 arg types no decay");
   static_assert(std::is_same<Dummy,
                              TFunctionTraits<decltype(stdFun2)>::Ret_t>::value,
                 "stdFun2 ret type");

}
