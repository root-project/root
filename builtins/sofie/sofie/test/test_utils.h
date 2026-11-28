// Utilities shared by the tests of the generated models: reading the reference
// data of the models and comparing outputs. They don't depend on how the
// generated code is compiled and run.
#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"

constexpr float DEFAULT_TOLERANCE = 1e-3f;

/// Reference data for one model: the test inputs and the expected outputs,
/// read from the references/<Model>.ref file that generate_input_models.py
/// writes next to the generated ONNX models. Entries are keyed "input0",
/// "input1", ... (one per graph input, in graph order) and "output0", ...
/// (one per graph output).
class SofieReference {
public:
   const std::vector<float> &f32(std::string const &key) const { return at(fF32, key); }
   const std::vector<double> &f64(std::string const &key) const { return at(fF64, key); }
   const std::vector<int64_t> &i64(std::string const &key) const { return at(fI64, key); }
   const std::vector<uint8_t> &u8(std::string const &key) const { return at(fU8, key); }

   std::map<std::string, std::vector<float>> fF32;
   std::map<std::string, std::vector<double>> fF64;
   std::map<std::string, std::vector<int64_t>> fI64;
   std::map<std::string, std::vector<uint8_t>> fU8;

private:
   template <class Map>
   static const typename Map::mapped_type &at(Map const &m, std::string const &key)
   {
      auto it = m.find(key);
      if (it == m.end())
         throw std::runtime_error("no reference data entry \"" + key + "\" of this type");
      return it->second;
   }
};

inline SofieReference readReference(std::string const &modelName)
{
   const std::string path = "input_models/references/" + modelName + ".ref";
   std::ifstream in(path);
   if (!in)
      throw std::runtime_error("cannot open reference data file " + path +
                               " (it is written by the SofieGenerateModels_ONNX test)");
   SofieReference ref;
   std::string key, type;
   std::size_t count;
   while (in >> key >> type >> count) {
      bool ok = true;
      if (type == "f32") {
         auto &v = ref.fF32[key];
         v.resize(count);
         for (auto &x : v)
            ok &= bool(in >> x);
      } else if (type == "f64") {
         auto &v = ref.fF64[key];
         v.resize(count);
         for (auto &x : v)
            ok &= bool(in >> x);
      } else if (type == "i64") {
         auto &v = ref.fI64[key];
         v.resize(count);
         for (auto &x : v)
            ok &= bool(in >> x);
      } else if (type == "u8") {
         auto &v = ref.fU8[key];
         v.resize(count);
         for (auto &x : v) {
            int tmp;
            ok &= bool(in >> tmp);
            x = static_cast<uint8_t>(tmp);
         }
      } else {
         ok = false;
      }
      if (!ok)
         throw std::runtime_error("malformed reference data file " + path + " at entry \"" + key + "\"");
   }
   return ref;
}

/// Element-wise |output - expected| <= tolerance
template <typename T, typename U>
void expectNear(std::vector<T> const &output, std::vector<U> const &expected, float tolerance)
{
   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < output.size(); ++i)
      EXPECT_LE(std::abs(static_cast<double>(output[i]) - static_cast<double>(expected[i])), tolerance)
         << "at output index " << i;
}

/// Element-wise |output - expected| <= tolerance, for models with several outputs
template <typename T, typename U>
void expectNear(std::vector<std::vector<T>> const &output, std::vector<std::vector<U>> const &expected, float tolerance)
{
   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < output.size(); ++i) {
      SCOPED_TRACE("output tensor " + std::to_string(i));
      expectNear(output[i], expected[i], tolerance);
   }
}

/// Element-wise |actual - expected| <= tolerance over raw arrays, printing at
/// most maxPrint failing elements followed by a count of the suppressed rest
inline void
expectNearCapped(const float *actual, const float *expected, std::size_t n, double tolerance, std::size_t maxPrint = 10)
{
   std::size_t mismatchCount = 0;
   for (std::size_t i = 0; i < n; ++i) {
      const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
      if (diff > tolerance) {
         if (mismatchCount < maxPrint) {
            ADD_FAILURE() << "Mismatch at index " << i << " actual=" << actual[i] << " expected=" << expected[i]
                          << " diff=" << diff;
         }
         ++mismatchCount;
      }
   }
   if (mismatchCount > maxPrint) {
      ADD_FAILURE() << "Further mismatches suppressed (total mismatches: " << mismatchCount << ")";
   }
}

/// Element-wise exact equality
template <typename T, typename U>
void expectEqual(std::vector<T> const &output, std::vector<U> const &expected)
{
   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < output.size(); ++i)
      EXPECT_EQ(static_cast<int64_t>(output[i]), static_cast<int64_t>(expected[i])) << "at output index " << i;
}

/// Element-wise exact equality, for models with several outputs
template <typename T, typename U>
void expectEqual(std::vector<std::vector<T>> const &output, std::vector<std::vector<U>> const &expected)
{
   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < output.size(); ++i) {
      SCOPED_TRACE("output tensor " + std::to_string(i));
      expectEqual(output[i], expected[i]);
   }
}
