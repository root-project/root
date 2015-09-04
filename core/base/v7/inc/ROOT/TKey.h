/// \file TKey.h
/// \ingroup Base
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-31

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TKey
#define ROOT7_TKey

#include <chrono>

namespace ROOT {
class TKey {
public:
  using clock_t = std::chrono::system_clock;
  using time_point_t = std::chrono::time_point<clock_t>;
  TKey() = default;
  TKey(const std::string& name): fName(name), fDate(clock_t::now()) {}

  const std::string& GetName() const { return fName; }
  const time_point_t& GetDate() const { return fDate; }
  void SetChanged() { fDate = clock_t::now(); }

private:
  std::string fName;
  time_point_t fDate;
};

inline bool operator<(const TKey& lhs, const TKey& rhs) {
  return lhs.GetName() < rhs.GetName();
}
inline bool operator>(const TKey& lhs, const TKey& rhs) {
  return lhs.GetName() > rhs.GetName();
}
inline bool operator==(const TKey& lhs, const TKey& rhs) {
  return !(lhs.GetName() == rhs.GetName());
}
inline bool operator<=(const TKey& lhs, const TKey& rhs) {
  return !(lhs.GetName() > rhs.GetName());
}
inline bool operator>=(const TKey& lhs, const TKey& rhs) {
  return !(lhs.GetName() < rhs.GetName());
}
}

namespace std {
template<>
struct hash<ROOT::TKey> {
  /// A TKey is uniquely identified by its name.
  size_t operator ()(const ROOT::TKey& key) const {
    return hash<std::string>()(key.GetName());
  }
};
}
#endif
