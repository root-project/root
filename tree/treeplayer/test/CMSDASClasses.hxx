#ifndef CMS_DAS_CLASSES
#define CMS_DAS_CLASSES

#include <Rtypes.h>
#include <vector>

namespace DAS {
struct Weight {
   int v{1};
   int i{0};
   operator int() const { return v; }

   // https://root.cern/manual/io_custom_classes/#restrictions-on-types-root-io-can-handle
   Weight() = default;

   // For vector initialization in test
   Weight(int a, int b) : v(a), i(b) {}

   // https://root.cern/manual/io_custom_classes/#the-classdef-macro
   // NV avoids marking the methods in I/O as virtual (not needed for this struct)
   ClassDefNV(Weight, 1);
};

// Also this is a new type and it gets stored to disk, needs a dictionary
using Weights = std::vector<Weight>;

// Stored, needs a dictionary
struct AbstractEvent {
   // https://root.cern/manual/io_custom_classes/#restrictions-on-types-root-io-can-handle
   AbstractEvent() = default;

   virtual ~AbstractEvent() = default;
   // Destructor is defined, thus we need to implement rule of five
   // https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c21-if-you-define-or-delete-any-copy-move-or-destructor-function-define-or-delete-them-all
   AbstractEvent(const AbstractEvent &) = default;
   AbstractEvent &operator=(const AbstractEvent &) = default;
   AbstractEvent(AbstractEvent &&) = default;
   AbstractEvent &operator=(AbstractEvent &&) = default;

   Weights weights;

   inline int Weight() const { return weights.front(); }

   // https://root.cern/manual/io_custom_classes/#the-classdef-macro
   ClassDef(AbstractEvent, 1);
};

struct GenEvent : public AbstractEvent {
   // Default constructor for ROOT I/O
   GenEvent() = default;

   // This is a derived class of a pure virtual class, we need to override all virtual functions
   // https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c128-virtual-functions-should-specify-exactly-one-of-virtual-override-or-final
   ~GenEvent() override = default;
   // Destructor is defined, thus we need to implement rule of five
   // https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c21-if-you-define-or-delete-any-copy-move-or-destructor-function-define-or-delete-them-all
   GenEvent(const GenEvent &) = default;
   GenEvent &operator=(const GenEvent &) = default;
   GenEvent(GenEvent &&) = default;
   GenEvent &operator=(GenEvent &&) = default;

   // https://root.cern/manual/io_custom_classes/#the-classdef-macro
   // Override because this is a derived class
   ClassDefOverride(GenEvent, 1);
};

} // namespace DAS

#endif // CMS_DAS_CLASSES
