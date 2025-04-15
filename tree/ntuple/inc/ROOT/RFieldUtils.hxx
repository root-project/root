/// \file RFieldUtils.hxx
/// \ingroup NTuple
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-11-19

#ifndef ROOT_RFieldUtils
#define ROOT_RFieldUtils

#include <string>
#include <string_view>
#include <typeinfo>
#include <tuple>
#include <vector>

class TClass;

namespace ROOT {
namespace Internal {

/// Applies RNTuple specific type name normalization rules (see specs) that help the string parsing in
/// RFieldBase::Create(). The normalization of templated types does not include full normalization of the
/// template arguments (hence "Prefix").
std::string GetCanonicalTypePrefix(const std::string &typeName);

/// Given a type name normalized by ROOT meta, renormalize it for RNTuple. E.g., insert std::prefix.
std::string GetRenormalizedTypeName(const std::string &metaNormalizedName);

/// Given a type info ask ROOT meta to demangle it, then renormalize the resulting type name for RNTuple. Useful to
/// ensure that e.g. fundamental types are normalized to the type used by RNTuple (e.g. int -> std::int32_t).
std::string GetRenormalizedDemangledTypeName(const std::type_info &ti);

/// Applies all RNTuple type normalization rules except typedef resolution.
std::string GetNormalizedUnresolvedTypeName(const std::string &origName);

/// Appends 'll' or 'ull' to the where necessary and strips the suffix if not needed.
std::string GetNormalizedInteger(const std::string &intTemplateArg);
std::string GetNormalizedInteger(long long val);
std::string GetNormalizedInteger(unsigned long long val);
long long ParseIntTypeToken(const std::string &intToken);
unsigned long long ParseUIntTypeToken(const std::string &uintToken);

/// Possible settings for the "rntuple.streamerMode" class attribute in the dictionary.
enum class ERNTupleSerializationMode {
   kForceNativeMode,
   kForceStreamerMode,
   kUnset
};

ERNTupleSerializationMode GetRNTupleSerializationMode(TClass *cl);

/// Parse a type name of the form `T[n][m]...` and return the base type `T` and a vector that contains,
/// in order, the declared size for each dimension, e.g. for `unsigned char[1][2][3]` it returns the tuple
/// `{"unsigned char", {1, 2, 3}}`. Extra whitespace in `typeName` should be removed before calling this function.
///
/// If `typeName` is not an array type, it returns a tuple `{T, {}}`. On error, it returns a default-constructed tuple.
std::tuple<std::string, std::vector<std::size_t>> ParseArrayType(const std::string &typeName);

/// Used in RFieldBase::Create() in order to get the comma-separated list of template types
/// E.g., gets {"int", "std::variant<double,int>"} from "int,std::variant<double,int>".
/// TODO(jblomer): Try to merge with TClassEdit::TSplitType
std::vector<std::string> TokenizeTypeList(std::string_view templateType);

} // namespace Internal
} // namespace ROOT

#endif
