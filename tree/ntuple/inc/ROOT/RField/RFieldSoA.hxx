/// \file ROOT/RField/RFieldSoA.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2026-03-03

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RField_SoA
#define ROOT_RField_SoA

#ifndef ROOT_RField
#error "Please include RField.hxx!"
#endif

#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleTypes.hxx>

#include <cstddef>
#include <memory>
#include <string_view>
#include <typeinfo>
#include <vector>

class TClass;

namespace ROOT {

namespace Experimental {

/// The SoA field provides I/O for an in-memory SoA layout linked to an on-disk collection of the underlying record.
/// As a concrete example, for an underlying record type
/// \code{.cpp}
/// struct PointRecord {
///    float px, py;
/// };
/// \endcode
/// a corresponding SoA layout will look like
/// \code{.cpp}
/// struct PointSoA {
///    ROOT::RVec<float> px;
///    ROOT::RVec<float> py;
/// };
/// \endcode
///
/// The SoA type has to be marked in the dictionary with the rntupleSoARecord attribute.
///
/// Since the on-disk representation is a collection of record type, the class version and checksum of the SoA type
/// itself is ignored.
class RSoAField : public RFieldBase {
   class RSoADeleter : public RDeleter {
   private:
      TClass *fSoAClass;

   public:
      explicit RSoADeleter(TClass *cl) : fSoAClass(cl) {}
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   TClass *fSoAClass = nullptr;
   std::vector<std::size_t> fSoAMemberOffsets;    ///< The offset of the RVec members in the SoA type
   std::vector<std::size_t> fRecordMemberIndexes; ///< Maps the SoA members to the members of the underlying record
   std::vector<RFieldBase *> fRecordMemberFields; ///< Direct access to the member fields of the underlying record
   std::size_t fMaxAlignment = 1;
   ROOT::Internal::RColumnIndex fNWritten;

   RSoAField(std::string_view fieldName, const RSoAField &source); ///< Used by CloneImpl
   RSoAField(std::string_view fieldName, TClass *clSoA);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const ROOT::RNTupleDescriptor &desc) final;

   void ConstructValue(void *where) const final;
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RSoADeleter>(fSoAClass); }

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final;

   void ReconcileOnDiskField(const RNTupleDescriptor &) final {}

   void CommitClusterImpl() final { fNWritten = 0; }

public:
   RSoAField(std::string_view fieldName, std::string_view className);
   RSoAField(RSoAField &&other) = default;
   RSoAField &operator=(RSoAField &&other) = default;
   ~RSoAField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final;
   size_t GetAlignment() const final { return fMaxAlignment; }
   /// For polymorphic classes (that declare or inherit at least one virtual method), return the expected dynamic type
   /// of any user object. If the class is not polymorphic, return nullptr.
   /// TODO(jblomer): use information in unique pointer field
   const std::type_info *GetPolymorphicTypeInfo() const;
   // TODO(jblomer)
   // void AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const final;
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RField_SoA
