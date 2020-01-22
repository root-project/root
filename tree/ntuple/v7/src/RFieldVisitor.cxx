/// \file RFieldVisitor.cxx
/// \ingroup NTuple ROOT7
/// \author Simon Leisibach <simon.leisibach@gmail.com>
/// \date 2019-06-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RField.hxx>
#include <ROOT/RFieldValue.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RNTupleView.hxx>

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


//----------------------------- RPrepareVisitor --------------------------------


void ROOT::Experimental::RPrepareVisitor::VisitField(const Detail::RFieldBase & /*field*/, int level)
{
   ++fNumFields;
   if (level > fDeepestLevel)
      fDeepestLevel = level;
}


//---------------------------- RPrintSchemaVisitor -----------------------------


void ROOT::Experimental::RPrintSchemaVisitor::SetDeepestLevel(int d)
{
   fDeepestLevel = d;
   fFlagForVerticalLines.resize(d - 1);
}

void ROOT::Experimental::RPrintSchemaVisitor::SetNumFields(int n)
{
   fNumFields = n;
   SetAvailableSpaceForStrings();
}

std::string ROOT::Experimental::RPrintSchemaVisitor::MakeKeyString(const Detail::RFieldBase &field, int level)
{
   std::string result{""};
   if (level == 1) {
      result += "Field ";
      result += std::to_string(field.GetLevelInfo().GetOrder());
   } else {
      if (field.GetLevelInfo().GetOrder() == field.GetLevelInfo().GetNumSiblings()) {
         fFlagForVerticalLines.at(level - 2) = false;
      } else {
         fFlagForVerticalLines.at(level - 2) = true;
      }
      for (int i = 0; i < level - 2; ++i) {
         if (fFlagForVerticalLines.at(i)) {
            result += "| ";
         } else {
            result += "  ";
         }
      }
      result += "|__Field ";
      result += RNTupleFormatter::HierarchialFieldOrder(field);
   }
   return result;
}

std::string ROOT::Experimental::RPrintSchemaVisitor::MakeValueString(const Detail::RFieldBase &field)
{
   std::string nameAndType{field.GetName() + " (" + field.GetType() + ")"};
   return nameAndType;
}

// Entire function only prints 1 Line, when if statement is disregarded.
void ROOT::Experimental::RPrintSchemaVisitor::VisitField(const Detail::RFieldBase &field, int level)
{
   if (level == 1) {
      for (int i = 0; i < fWidth; ++i) {
         fOutput << fFrameSymbol;
      }
      fOutput << std::endl;
   }
   fOutput << fFrameSymbol << ' ';
   fOutput << RNTupleFormatter::FitString(MakeKeyString(field, level), fAvailableSpaceKeyString);
   fOutput << " : ";
   fOutput << RNTupleFormatter::FitString(MakeValueString(field), fAvailableSpaceValueString);
   fOutput << fFrameSymbol << std::endl;
}

//------------------------ RValueVisitor --------------------------------

void ROOT::Experimental::RValueVisitor::VisitField(const Detail::RFieldBase &field, int level)
{
   if (fPrintOnlyValue) {
      fOutput << "no support for " << field.GetType();
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": no support for " << field.GetType();

   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitArrayField(const RFieldArray &field, int level)
{
   // When fPrintOnlyValue is true, int level gets a new meaning. It is used in VisitVectorField for multidimentional
   // vectors and array of vectors.
   if (fPrintOnlyValue) {
      fOutput << "[ ";

      // iterate through the elements of the array
      for (std::size_t i = 0; i < field.GetLength() - 1; ++i) {
         // The expression (i + fIndex*field.GetLength()) is passed as level. When passed to a field with basic data
         // type it is unused, but for multidimensional vectors it denotes the index of a subvector.
         field.FirstSubFieldAcceptVisitor(*this, i + fIndex * field.GetLength());
         fOutput << ", ";
         ++fCollectionIndex;
      }
      // don't print ", " for the last element
      field.FirstSubFieldAcceptVisitor(*this, field.GetLength() - 1 + fIndex * field.GetLength());
      fOutput << " ]";
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": [";
   // Get startIndex from next non-vector/non-array itemfield.
   SetCollectionIndex(field);

   for (std::size_t i = 0; i < field.GetLength() - 1; ++i) {
      field.FirstSubFieldAcceptVisitor(*this, i + fIndex * field.GetLength());
      fOutput << ", ";
      ++fCollectionIndex;
   }
   // don't print ", " for the last element
   field.FirstSubFieldAcceptVisitor(*this, field.GetLength() - 1 + fIndex * field.GetLength());

   if (field.GetFirstChild()->GetStructure() == ENTupleStructure::kRecord) {
      fOutput << std::endl;
      for (int i = 0; i < level; ++i)
         fOutput << "  ";
   }
   fOutput << "]";
   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;

   fPrintOnlyValue = false;
}

void ROOT::Experimental::RValueVisitor::VisitBoolField(const RField<bool> &field, int level)
{
   if (fPrintOnlyValue) {
      auto view = fReader->GetView<bool>(RNTupleFormatter::FieldHierarchy(field));
      if (view(fCollectionIndex) == 0) {
         fOutput << "false";
      } else {
         fOutput << "true";
      }
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";

   auto view = fReader->GetView<bool>(RNTupleFormatter::FieldHierarchy(field));
   if (view(fIndex) == 0) {
      fOutput << "false";
   } else {
      fOutput << "true";
   }

   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;
}

// Visited when encountering a field with a custom object with dictionary.
void ROOT::Experimental::RValueVisitor::VisitClassField(const RFieldClass &field, int level)
{
   if (fPrintOnlyValue) {
      fOutput << std::endl;
      // Create new RValueVisitor to have 2 different fCollectionIndexes (allows to display vector of objects which
      // contain vectors themselves)
      RValueVisitor visitor(fOutput, fReader, level /*fIndex*/, false, 0);
      field.NotVisitTopFieldTraverseValueVisitor(visitor, field.GetLevelInfo().GetLevel());
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": " << std::endl;
   // A custom object (represented by RFieldClass) should let its subfields display
   // its entries instead of displaying all by itself.
   // So the field of the custom object should only display its name and make an early quit here.
   return;
}

void ROOT::Experimental::RValueVisitor::VisitClusterSizeField(const RField<ROOT::Experimental::ClusterSize_t> &field,
                                                              int level)
{
   if (fPrintOnlyValue) {
      auto view = fReader->GetView<ClusterSize_t>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << view(fCollectionIndex);
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";

   auto view = fReader->GetView<ClusterSize_t>(RNTupleFormatter::FieldHierarchy(field));
   fOutput << view(fIndex);

   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitDoubleField(const RField<double> &field, int level)
{
   if (fPrintOnlyValue) {
      auto view = fReader->GetView<double>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << view(fCollectionIndex);
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";

   auto view = fReader->GetView<double>(RNTupleFormatter::FieldHierarchy(field));
   fOutput << view(fIndex);

   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitFloatField(const RField<float> &field, int level)
{
   if (fPrintOnlyValue) {
      auto view = fReader->GetView<float>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << view(fCollectionIndex);
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";

   auto view = fReader->GetView<float>(RNTupleFormatter::FieldHierarchy(field));
   fOutput << view(fIndex);

   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitIntField(const RField<int> &field, int level)
{
   if (fPrintOnlyValue) {
      auto view = fReader->GetView<std::int32_t>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << view(fCollectionIndex);
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";

   auto view = fReader->GetView<int>(RNTupleFormatter::FieldHierarchy(field));
   fOutput << view(fIndex);

   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitStringField(const RField<std::string> &field, int level)
{
   if (fPrintOnlyValue) {
      auto view = fReader->GetView<std::string>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << "\"" << view(fCollectionIndex) << "\"";
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";

   auto view = fReader->GetView<std::string>(RNTupleFormatter::FieldHierarchy(field));
   fOutput << "\"" << view(fIndex) << "\"";

   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitUInt32Field(const RField<std::uint32_t> &field, int level)
{
   if (fPrintOnlyValue) {
      auto view = fReader->GetView<std::uint32_t>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << view(fCollectionIndex);
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";

   auto view = fReader->GetView<std::uint32_t>(RNTupleFormatter::FieldHierarchy(field));
   fOutput << view(fIndex);

   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitUInt64Field(const RField<std::uint64_t> &field, int level)
{
   if (fPrintOnlyValue) {
      auto view = fReader->GetView<std::uint64_t>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << view(fCollectionIndex);
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";

   auto view = fReader->GetView<std::uint64_t>(RNTupleFormatter::FieldHierarchy(field));
   fOutput << view(fIndex);

   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitUInt8Field(const RField<std::uint8_t> &field, int level)
{
   if (fPrintOnlyValue) {
      auto view = fReader->GetView<std::uint8_t>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << "'" << view(fCollectionIndex) << "'";
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";

   auto view = fReader->GetView<std::uint8_t>(RNTupleFormatter::FieldHierarchy(field));
   fOutput << "'" << view(fIndex) << "'";

   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitVectorField(const RFieldVector &field, int level)
{
   if (fPrintOnlyValue) {
      fOutput << "{ ";
      RClusterIndex dummyCluster;
      ClusterSize_t nItems;
      field.GetCollectionInfo(level, &dummyCluster, &nItems);

      for (std::size_t i = 0; i < nItems - 1; ++i) {
         // The level parameter has a different meaning when fPrintOnlyValue = true. i + dummyCluster.GetIndex() is used
         // to get the index of the subvector when dealing with multidimensional vectors.
         field.FirstSubFieldAcceptVisitor(*this, i + dummyCluster.GetIndex() /*level*/);
         fOutput << ", ";
         ++fCollectionIndex;
      }
      // don't print ", " for the last element
      field.FirstSubFieldAcceptVisitor(*this, nItems - 1 + dummyCluster.GetIndex() /*level*/);
      fOutput << " }";
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": {";

   ClusterSize_t nItems;
   RClusterIndex dummyIndex;
   field.GetCollectionInfo(fIndex, &dummyIndex, &nItems);
   SetCollectionIndex(field);

   for (std::size_t i = 0; i < nItems - 1; ++i) {
      field.FirstSubFieldAcceptVisitor(*this, i + dummyIndex.GetIndex());
      fOutput << ", ";
      ++fCollectionIndex;
   }
   // don't print ", " for the last element
   field.FirstSubFieldAcceptVisitor(*this, nItems - 1 + dummyIndex.GetIndex());

   // Cosmetics for case where a vector of objects should be displayed.
   if (field.GetFirstChild()->GetStructure() == ENTupleStructure::kRecord) {
      fOutput << std::endl;
      for (int i = 0; i < level; ++i)
         fOutput << "  ";
   }
   fOutput << "}";
   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;

   fPrintOnlyValue = false;
}

// See RValueVisitor::VisitVectorField for comments
void ROOT::Experimental::RValueVisitor::VisitVectorBoolField(const RField<std::vector<bool>> &field, int level)
{
   if (fPrintOnlyValue) {
      fOutput << "{ ";
      RClusterIndex dummyIndex;
      ClusterSize_t nItems;
      field.GetCollectionInfo(level, &dummyIndex, &nItems);

      for (std::size_t i = 0; i < nItems - 1; ++i) {
         field.FirstSubFieldAcceptVisitor(*this, i + dummyIndex.GetIndex() /* level*/);
         fOutput << ", ";
         ++fCollectionIndex;
      }
      // don't print ", " for the last element
      field.FirstSubFieldAcceptVisitor(*this, nItems - 1 + dummyIndex.GetIndex() /* level*/);
      fOutput << " }";
      return;
   }

   for (int i = 0; i < level; ++i)
      fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": {";

   ClusterSize_t nItems;
   RClusterIndex dummyIndex;
   field.GetCollectionInfo(fIndex, &dummyIndex, &nItems);
   SetCollectionIndex(field);
   for (std::size_t i = 0; i < nItems - 1; ++i) {
      field.FirstSubFieldAcceptVisitor(*this, i + dummyIndex.GetIndex());
      fOutput << ", ";
      ++fCollectionIndex;
   }
   field.FirstSubFieldAcceptVisitor(*this, nItems - 1 + dummyIndex.GetIndex());

   if (field.GetFirstChild()->GetStructure() == ENTupleStructure::kRecord) {
      fOutput << std::endl;
      for (int i = 0; i < level; ++i)
         fOutput << "  ";
   }
   fOutput << "}";
   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fPrintOnlyValue = false;
}

// Quick formulas for calculating the startIndex of the first non-array/non-vector itemfield:
// vector of vector: itemFieldIndex
//                   MainFieldPtr->GetCollectionInfo(fIndex, &clusterIndex, &nItems)
//                   subfieldIndex = ConvertClusterIndexToGlobalIndex(clusterIndex)
//                   SubFieldPtr->GetCollectionInfo(subFieldIndex, &clusterIndex2, &nItems)
//                   itemFieldIndex = ConvertClusterIndexToGlobalIndex(clusterIndex2)
//                   (for higher dimensional vectors successivley plugIn the clusterIndex to GetCollectionInfo.
// array of array: fIndex * fLength1 * fLength2
// vector of array:  subfieldIndex * fLength2
//                   MainFieldPtr->GetCollectionInfo(fIndex, &clusterIndex, &nItems)
//                   subfieldIndex = ConvertClusterIndexToGlobalIndex(clusterIndex)
// array of vector:  itemFieldIndex
//                   SubFieldPtr->GetCollectionInfo(fIndex * fLength1, &clusterIndex, &nItems)
//                   itemFieldIndex = ConvertClusterIndexToGlobalIndex(clusterIndex)
void ROOT::Experimental::RValueVisitor::SetCollectionIndex(const Detail::RFieldBase &field)
{
   const Detail::RFieldBase *fieldPtr = &field;
   const Detail::RFieldBase *childPtr = field.GetFirstChild();
   if (childPtr == nullptr)
      assert(false);

   std::string typeName = field.GetType();
   if (typeName.compare(0, 11, "std::array<") == 0) {
      auto arrayFieldPtr = reinterpret_cast<const RFieldArray *>(fieldPtr);
      fPrintOnlyValue = true;
      fCollectionIndex = fIndex * arrayFieldPtr->GetLength();
   } else if (typeName.compare(0, 17, "std::vector<bool>") == 0) {
      auto boolVecFieldPtr = reinterpret_cast<const RField<std::vector<bool>> *>(fieldPtr);
      fPrintOnlyValue = true;
      ClusterSize_t nItems;
      RClusterIndex dummyIndex;
      boolVecFieldPtr->GetCollectionInfo(fIndex, &dummyIndex, &nItems);
      fCollectionIndex = ConvertClusterIndexToGlobalIndex(dummyIndex);
      return;
   } else if (typeName.compare(0, 12, "std::vector<") == 0) {
      auto vecFieldPtr = reinterpret_cast<const RFieldVector *>(fieldPtr);
      fPrintOnlyValue = true;
      ClusterSize_t nItems;
      RClusterIndex dummyIndex;
      vecFieldPtr->GetCollectionInfo(fIndex, &dummyIndex, &nItems);
      fCollectionIndex = ConvertClusterIndexToGlobalIndex(dummyIndex);
   } else {
      return;
   }
   std::string childTypeName = childPtr->GetType();
   while ((childTypeName.compare(0, 11, "std::array<") == 0) || (childTypeName.compare(0, 12, "std::vector<") == 0)) {
      fieldPtr = childPtr;
      childPtr = fieldPtr->GetFirstChild();
      typeName = fieldPtr->GetType();
      if (typeName.compare(0, 11, "std::array<") == 0) {
         auto arrayFieldPtr = reinterpret_cast<const RFieldArray *>(fieldPtr);
         fCollectionIndex = fCollectionIndex * arrayFieldPtr->GetLength();
      } else if (typeName.compare(0, 17, "std::vector<bool>") == 0) {
         auto boolVecFieldPtr = reinterpret_cast<const RField<std::vector<bool>> *>(fieldPtr);
         ClusterSize_t nItems;
         RClusterIndex dummyIndex;
         boolVecFieldPtr->GetCollectionInfo(fCollectionIndex, &dummyIndex, &nItems);
         fCollectionIndex = ConvertClusterIndexToGlobalIndex(dummyIndex);
         return;
      } else if (typeName.compare(0, 12, "std::vector<") == 0) {
         auto vecFieldPtr = reinterpret_cast<const RFieldVector *>(fieldPtr);
         ClusterSize_t nItems;
         RClusterIndex dummyIndex;
         vecFieldPtr->GetCollectionInfo(fCollectionIndex, &dummyIndex, &nItems);
         fCollectionIndex = ConvertClusterIndexToGlobalIndex(dummyIndex);
      } else {
         return;
      }
      childTypeName = childPtr->GetType();
   }
   return;
}

std::size_t ROOT::Experimental::RValueVisitor::ConvertClusterIndexToGlobalIndex(RClusterIndex clusterIndex) const
{
   auto &desc = fReader->GetDescriptor();
   std::size_t globalIndex = static_cast<std::size_t>(clusterIndex.GetIndex());
   for (std::size_t i = 0; i < clusterIndex.GetClusterId(); ++i) {
      globalIndex += desc.GetClusterDescriptor(i).GetNEntries();
   }
   return globalIndex;
}


//---------------------------- RNTupleFormatter --------------------------------


std::string ROOT::Experimental::RNTupleFormatter::FieldHierarchy(const Detail::RFieldBase &field)
{
   std::string qualifiedName{field.GetName()};
   if (!field.GetParent())
      return qualifiedName;
   const Detail::RFieldBase *parentField{field.GetParent()};
   for (int i = field.GetLevelInfo().GetLevel(); i > 1; --i) {
      qualifiedName = parentField->GetName() + "." + qualifiedName;
      parentField = parentField->GetParent();
   }
   return qualifiedName;
}

std::string ROOT::Experimental::RNTupleFormatter::FitString(const std::string &str, int availableSpace)
{
   int strSize{static_cast<int>(str.size())};
   if (strSize <= availableSpace)
      return str + std::string(availableSpace - strSize, ' ');
   else if (availableSpace < 3)
      return std::string(availableSpace, '.');
   return std::string(str, 0, availableSpace - 3) + "...";
}

std::string
ROOT::Experimental::RNTupleFormatter::HierarchialFieldOrder(const ROOT::Experimental::Detail::RFieldBase &field)
{
   std::string hierarchialOrder{std::to_string(field.GetLevelInfo().GetOrder())};
   const Detail::RFieldBase *parentPtr{field.GetParent()};
   // To avoid having the index of the RootField (-1) in the return value, it is checked if the grandparent is a nullptr
   // (in that case RootField is parent)
   while (parentPtr && (parentPtr->GetLevelInfo().GetOrder() != -1)) {
      hierarchialOrder = std::to_string(parentPtr->GetLevelInfo().GetOrder()) + "." + hierarchialOrder;
      parentPtr = parentPtr->GetParent();
   }
   return hierarchialOrder;
}
