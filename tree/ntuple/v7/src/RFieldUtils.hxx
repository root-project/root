/// \file RFieldUtils.hxx
/// \ingroup NTuple ROOT7
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-11-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#ifndef ROOT7_RFieldUtils
#define ROOT7_RFieldUtils

#include <string>
#include <string_view>
#include <tuple>
#include <vector>

class TClass;

namespace ROOT {
namespace Experimental {
namespace Internal {

/// Applies type name normalization rules that lead to the final name used to create a RField, e.g. transforms
/// `const vector<T>` to `std::vector<T>`.  Specifically, `const` / `volatile` qualifiers are removed and `std::` is
/// added to fully qualify known types in the `std` namespace.  The same happens to `ROOT::RVec` which is normalized to
/// `ROOT::VecOps::RVec`.
std::string GetNormalizedTypeName(const std::string &typeName);

/// Possible settings for the "rntuple.streamerMode" class attribute in the dictionary.
enum class ERNTupleSerializationMode { kForceNativeMode, kForceStreamerMode, kUnset };

ERNTupleSerializationMode GetRNTupleSerializationMode(TClass *cl);

/// Parse a type name of the form `T[n][m]...` and return the base type `T` and a vector that contains,
/// in order, the declared size for each dimension, e.g. for `unsigned char[1][2][3]` it returns the tuple
/// `{"unsigned char", {1, 2, 3}}`. Extra whitespace in `typeName` should be removed before calling this function.
///
/// If `typeName` is not an array type, it returns a tuple `{T, {}}`. On error, it returns a default-constructed tuple.
std::tuple<std::string, std::vector<size_t>> ParseArrayType(std::string_view typeName);

/// Used in RFieldBase::Create() in order to get the comma-separated list of template types
/// E.g., gets {"int", "std::variant<double,int>"} from "int,std::variant<double,int>"
std::vector<std::string> TokenizeTypeList(std::string_view templateType);

} // namespace Internal

} // namespace Experimental
} // namespace ROOT

#endif
