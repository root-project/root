/// \file ROOT/RNTupleAttributes.hxx
/// \ingroup NTuple ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2026-01-27
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#ifndef ROOT7_RNTuple_Attributes
#define ROOT7_RNTuple_Attributes

#include <memory>
#include <string_view>

#include <ROOT/REntry.hxx>
#include <ROOT/RNTupleFillContext.hxx>
#include <ROOT/RNTupleUtils.hxx>

namespace ROOT {

class RNTupleModel;
class RNTuple;

namespace Experimental {

namespace Internal {

namespace RNTupleAttributes {

inline const char *const kRangeStartName = "_rangeStart";
inline const char *const kRangeLenName = "_rangeLen";
inline const char *const kUserModelName = "_userModel";

} // namespace RNTupleAttributes

} // namespace Internal

// clang-format off
/**
\class ROOT::Experimental::RNTupleAttrSetWriter
\ingroup NTuple
\brief Class used to write an RNTupleAttrSet in the context of an RNTupleWriter.

An Attribute Set is written as a separate RNTuple linked to the "main" RNTuple that created it.
A RNTupleAttrSetWriter only lives as long as the RNTupleWriter that created it (or until CloseAttributeSet() is called).
Users should not use this class directly but rather via RNTupleAttrSetWriterHandle, which is the type returned by
RNTupleWriter::CreateAttributeSet().

~~~ {.cpp}
// TODO
~~~
*/
// clang-format on
class RNTupleAttrSetWriter final {
   friend class ROOT::RNTupleFillContext;

   /// Our own fill context.
   RNTupleFillContext fFillContext;
   /// Fill context of the main RNTuple being written (i.e. the RNTuple whose attributes we are).
   const RNTupleFillContext *fMainFillContext = nullptr;
   /// The model that the user provided on creation. Used to create user-visible entries.
   std::unique_ptr<RNTupleModel> fUserModel;

   std::shared_ptr<ROOT::NTupleSize_t> fRangeStartPtr;
   std::shared_ptr<ROOT::NTupleSize_t> fRangeLenPtr;

   /// Creates an RNTupleAttrSetWriter associated to the RNTupleWriter owning `mainFillContext` and writing
   /// using `sink`. `userModel` is the schema of the AttributeSet.
   static std::unique_ptr<RNTupleAttrSetWriter> Create(const RNTupleFillContext &mainFillContext,
                                                       std::unique_ptr<ROOT::Internal::RPageSink> sink,
                                                       std::unique_ptr<RNTupleModel> userModel);

   RNTupleAttrSetWriter(const RNTupleFillContext &mainFillContext, std::unique_ptr<ROOT::Internal::RPageSink> sink,
                        std::unique_ptr<RNTupleModel> metaModel, std::unique_ptr<RNTupleModel> userModel,
                        std::shared_ptr<ROOT::NTupleSize_t> rangeStartPtr,
                        std::shared_ptr<ROOT::NTupleSize_t> rangeLenPtr);

public:
   /// Returns the descriptor of the underlying attribute RNTuple. This is **NOT** the same descriptor as the
   /// main RNTuple being written!
   const ROOT::RNTupleDescriptor &GetDescriptor() const { return fFillContext.fSink->GetDescriptor(); }
   /// Returns the user-defined model used to create this attribute set.
   const ROOT::RNTupleModel &GetModel() const { return *fUserModel; }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleAttrSetWriterHandle
\ingroup NTuple
\brief Non-owning handle to an RNTupleAttrSetWriter

RNTupleAttrSetWriter can only be used through an RNTupleAttrSetWriterHandle, a weak_ptr-like object that allows safe
access to it. The lifetime of an attribute set writer is tied to its parent RNTupleWriter, so the handle handed out
by RNTupleWriter::CreateAttributeSet is invalidated as soon as the parent writer is destructed.

*/
// clang-format on
class RNTupleAttrSetWriterHandle final {
   friend class ROOT::RNTupleFillContext;

   std::weak_ptr<RNTupleAttrSetWriter> fWriter;

   explicit RNTupleAttrSetWriterHandle(const std::shared_ptr<RNTupleAttrSetWriter> &range) : fWriter(range) {}

public:
   RNTupleAttrSetWriterHandle(const RNTupleAttrSetWriterHandle &) = delete;
   RNTupleAttrSetWriterHandle &operator=(const RNTupleAttrSetWriterHandle &) = delete;
   RNTupleAttrSetWriterHandle(RNTupleAttrSetWriterHandle &&) = default;
   RNTupleAttrSetWriterHandle &operator=(RNTupleAttrSetWriterHandle &&other) = default;

   /// Retrieves the underlying pointer to the AttrSetWriter, throwing if it's invalid.
   /// This is NOT thread-safe and must be called from the same thread that created the AttrSetWriter.
   RNTupleAttrSetWriter *operator->()
   {
      auto ptr = fWriter.lock();
      if (R__unlikely(!ptr))
         throw ROOT::RException(R__FAIL("Tried to access invalid RNTupleAttrSetWriterHandle"));
      return ptr.get();
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
