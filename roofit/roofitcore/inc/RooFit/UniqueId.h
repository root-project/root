/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN, Jun 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef roofit_roofitcore_RooFit_UniqueId_h
#define roofit_roofitcore_RooFit_UniqueId_h

#include <atomic>

namespace RooFit {

/// A UniqueId can be added as a class member to enhance any class with a
/// unique identifier for each instantiated object.
///
/// Example:
/// ~~~{.cpp}
/// class MyClass {
///
/// public:
///    /// Return unique ID by reference.
///    /// Please always use the name `uniqueId` for the getter.
///    UniqueId<MyClass> const& uniqueId() const { return _uniqueId; }
///
/// private:
///    const UniqueId<MyClass> _uniqueId; //! should be non-persistent
///
/// };
/// ~~~

template <class Class>
struct UniqueId {
public:
   using Value_t = unsigned long;

   /// Create a new UniqueId with the next value from the static counter.
   UniqueId() : _val{++counter} {}

   // Disable all sorts of copying and moving to ensure uniqueness.
   UniqueId(const UniqueId &) = delete;
   UniqueId &operator=(const UniqueId &) = delete;
   UniqueId(UniqueId &&) = delete;
   UniqueId &operator=(UniqueId &&) = delete;

   operator Value_t() const { return _val; }

   /// Return numerical value of ID.
   /// Use only if necessary, as the UniqueId type information is lost and
   /// copying/moving is not prohibited for the value type.
   /// Please don't turn this into a cast operator, as a function with an
   /// explicit name is easier to track in the codebase.
   constexpr Value_t value() const { return _val; }

   bool operator==(UniqueId const &other) const { return _val == other._val; }
   bool operator<(UniqueId const &other) const { return _val < other._val; }

   /// Get an ID that is less than the ID of any object (similar to nullptr).
   static UniqueId const &nullid()
   {
      static const UniqueId nid{nullval};
      return nid;
   }

   static constexpr Value_t nullval = 0UL; ///< The value of the nullid.

private:
   UniqueId(Value_t val) : _val{val} {}

   Value_t _val; ///< Numerical value of the ID.

   static std::atomic<Value_t> counter; ///< The static object counter to get the next ID value.
};

template <class Class>
std::atomic<typename UniqueId<Class>::Value_t> UniqueId<Class>::counter{UniqueId<Class>::nullval};

/// A helper function to replace pointer comparisons with UniqueId comparisons.
/// With pointer comparisons, we can also have `nullptr`. In the UniqueId case,
/// this translates to the `nullid`.
template <class Class,
          class UniqueId_t = std::remove_reference_t<decltype(std::declval<std::remove_pointer_t<Class>>().uniqueId())>>
UniqueId_t const &getUniqueId(Class const *ptr)
{
   return ptr ? ptr->uniqueId() : UniqueId_t::nullid();
}

} // namespace RooFit

#endif
