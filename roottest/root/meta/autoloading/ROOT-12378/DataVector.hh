#pragma once

namespace TestxAOD {
struct Jet_v1 {};
} // namesapce TestxAOD

template <typename T> struct DataVectorBase
{
   using Base = DataVectorBase<T>;
};

template <typename T, class BASE = typename DataVectorBase<T>::Base> struct TestDataVector : BASE {};


