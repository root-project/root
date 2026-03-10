/// \file ROOT/RNTupleAttrUtils.hxx
/// \ingroup NTuple
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2026-03-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#ifndef ROOT7_RNTuple_Attr_Utils
#define ROOT7_RNTuple_Attr_Utils

#include <cstddef>
#include <cstdint>

namespace ROOT::Experimental::Internal::RNTupleAttributes {

/*
Attribute RNTuples have a fixed Model schema that looks like this:

                +-----------+
                | MetaModel |
                +-----+-----+
                      |
       +--------------+---------------+
       |              |               |
+------+------+ +-----+------+ +------+------+
| _rangeStart | | _rangeLen  | | _userModel  |
+-------------+ +------------+ +------+------+
                                      |
                                +-----+-----+
                                | UserModel |
                                +-----------+

Where "_userModel" is an untyped record field containing all the user-defined Model's fields
as its children.

The order and name of the meta Model's fields is defined by the schema version.
*/

inline const std::uint16_t kSchemaVersionMajor = 1;
inline const std::uint16_t kSchemaVersionMinor = 0;

inline const char *const kRangeStartName = "_rangeStart";
inline const char *const kRangeLenName = "_rangeLen";
inline const char *const kUserModelName = "_userModel";

inline constexpr std::size_t kRangeStartIndex = 0;
inline constexpr std::size_t kRangeLenIndex = 1;
inline constexpr std::size_t kUserModelIndex = 2;

} // namespace ROOT::Experimental::Internal::RNTupleAttributes

#endif
