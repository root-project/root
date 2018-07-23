/// \file RCodingStyle.hxx
/// \author Axel Naumann <axel@cern.ch>
/// \author Other Author <other@author>
/// \date 2018-07-06
// The above entries are mostly for giving some context.
// The "author" field gives a hint whom to contact in case of questions, also
// from within the team. The date shows whether this is ancient or only part
// of the latest release. It's the date of creation / last massive rewrite.

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// This file demonstrates the coding styles to be used for new ROOT files.
// See the accompanying source file for context.

// File names correspond to the name of the main class that they contain.
// Our users will expect that a file called "RCodingStyle.hxx" contains a class
// named `RCodingStyle`.

// All files start with a Doxygen header, followed by the copyright statement.

// The file must contain no trailing whitespace including no indentation-only lines.
// Except for extreme cases, line breaks should happen between 80 and 120 characters.

// The first non-comment, non-empty line must be the #include guard.
// Acceptable forms are:
//   ROOT_RCodingStyle, ROOT_RCodingStyle_hxx, ROOT_RCodingStyle_Hxx,
//   ROOT_RCODINGSTYLE, ROOT_RCODINGSTYLE_HXX
// The first is preferred.
#ifndef ROOT_RCodingStyle

// Especially within header files, reduce the number of included headers as
// much as possible, at the expense of more forward declarations.
// EXCEPTION: standard library classes must be #included, not forward declared.

// Include ROOT's headers first, in alphabetical order:
//   Rationale: make sure that ROOT headers are standalone.
// Do not use #include guards at the point of `#include`.
//   Rationale: compilers optimize this themselves, this is interfering with their optimization.
// Use <>, not "".
//   Rationale: "" looks at ./ first; as ROOT headers are included by user code
//   that adds context fragility.
#include <ROOT/TSeq.hxx>
// Include v6 ROOT headers last.
// You're welcome to state what's used from a header, i.e. why it's included.
#include <Rtypes.h> // for ULong64_t

// Include stdlib headers next, in alphabetical order.
#include <string>

// Include non-ROOT, non-stdlib headers last.
// Rationale: non-ROOT, non-stdlib headers often #define like mad. Reduce interference
// with ROOT or stdlib headers.
#include <png.h>

// Say NO to
// - `using namespace` in at global scope in headers
// - variables, types, functions etc at global scope
// - preprocessor macros except for #include guards

namespace ROOT {
// All new classes are within the namespace ROOT.

// Forward declare before class definitions.
namespace Experimental {
class RAxisConfig;
}

///\class RExampleClass
/// Classes and structs start with 'R'. And they are documented.
class RExampleClass {
public:
   // FIRST section: public types.

   /// Nested types are documented, too. Do not use `typedef` but `using`.
   /// Type aliases end with "_t".
   using SeqULL_t = TSeq<ULong64_t>;

   ///\class Nested
   /// This is a nested class. Only use `struct`s for simple aggregates without
   /// methods and all public members.
   struct Nested {
      std::string fNestedName; ///< The nested name.
   };

private:
   // SECOND section: private types.

   // Any type used by any public / protected interface must also be public / protected!
   // This is not enforced by C++ but required by this coding style.
   using Nesteds_t = std::vector<Nested>;

   // THIRD section: data members.

   int fIntMember = -2;                  ///< Use inline initialization at least for fundamental types.
   std::vector<ROOT::Experimental::RAxisConfig> fAxes; ///< This documents a regular member. Don't explicitly default initialize (`{}`)
   std::vector<Nested> fManyStrings;     ///<! This marks a member that does not get serialized.
   static constexpr const int kSeq = -7; ///< A static constexpr (const) variable starts with `k`.

   /// A static variable starts with `fg`. Avoid all non-constexpr static variables! Long data
   /// member documentation is just fine, but then put it in front of the data member.
   static std::string fgNameBadPleaseUseStaticFunctionStaticVariable;

protected:
   /// FOURTH section: non-public functions.

   /// It's a good habit to prevent slicing of classes that can be base classes,
   /// e.g. for `RExampleClass &Get()` you likely want to keep the derived object intact,
   /// and prevent `auto ret = Get()` (where `ret` is now an object of type `RExampleClass`)
   /// instead of `auto &ret = Get()`.
   RExampleClass(const RExampleClass &) = default;
   RExampleClass(RExampleClass &&) = default;

   /// Instead of static data members (whether private or public), use outlined static
   /// functions. See the implementation in the source file.
   static const std::string &StaticFunc();

public:
   // FIFTH section: methods, starting with constructors, then alphabetical order.

   /// Use `= default` inside the class whenever possible
   RExampleClass() = default;

   /// Classes with virtual functions must have a virtual destructor. If all other functions are
   /// pure virtual or inlined, the destructor should be outlined to pin the class's vtable to an
   /// object file. I.e. say `= default` in the source, not here.
   virtual ~RExampleClass();

   /// Virtual functions are still fine (though we try to avoid them as they come with a call
   /// and optimization penalty). See `RCodingStyle` for overriding virtual functions.
   virtual void AVirtualFunction() = 0;

   /// Static functions line up alphabetically with the others.
   static int GetSeq() { return kSeq; }
};

///\class RCodingStyle
/// This is exhaustive documentation of the class template, explaining the constraints
/// on the template parameters, e.g. `IDX` must be positive, `T` must be  copyable.
/// Disabled if `T` is of reference type.

template <int IDX, // Template parameters are all-capital letters
          class T, // We use "class", not "typename"; `T` is fine for generic class names
                   // we use enable-if through unnamed template parameters and provide a `assert`-style message
                   // to make the diagnostics more understandable.
          class = std::enable_if<"Must only be used with non-reference types" && !std::is_reference<T>::value>>
class RCodingStyle : public RExampleClass {
private:
   T fMember; ///< The wrapped value.
public:
   // Use static_assert excessively: it makes diagnostics much more readable,
   // and helps readers of your code.
   static_assert(IDX > 0, "IDX must be >0.");

   /// Defaulted default constructor.
   RCodingStyle() = default;

   // Use `= delete` instead of private, unimplemented.
   // To avoid copy and move construction and assignment, `delete` the copy
   // constructor and copy-assignment operator, see https://stackoverflow.com/a/15181645/6182509
   RCodingStyle(const RCodingStyle &) = delete;
   RCodingStyle &operator=(const RCodingStyle &) = delete;

   /// Virtual destructors of derived classes may be decorated with `virtual` (even though
   /// they "inherit" the virtuality of the virtual base class destructor).
   virtual ~RCodingStyle() {}

   /// Overridden virtual functions do not specify `virtual` but either `override` or `final`.
   void AVirtualFunction() override;

   /// "Free" (i.e. non-class member) operators should be declared as "hidden friends".
   /// This makes them available only when needed, instead of adding to the long list of
   /// "no matching overload for a+b, did you mean ...(hundreds of unrelated types)..."
   friend RCodingStyle operator+(const RCodingStyle &a, const RCodingStyle &b)
   {
      RCodingStyle ret{a};
      ret.fIntMember += b.fIntMember;
      return ret;
   }
};

/// Functions that access only public members of ROOT classes shall be in here.
/// \param a - first RExampleClass to add.
/// \param b - second RExampleClass to add.
double Add(const RExampleClass &a, const RExampleClass &b);

} // namespace ROOT
#endif // ROOT_RCodingStyle
// The file must end on a trailing new-line.
