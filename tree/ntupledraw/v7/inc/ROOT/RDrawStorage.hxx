/// \file ROOT/RDrawStorage.hxx
/// \ingroup NTupleDraw ROOT7
/// \author Simon Leisibach <simon.satoshi.rene.leisibach@cern.ch>
/// \date 2019-11-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RDrawStorage
#define ROOT7_RDrawStorage

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <TBox.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TLegend.h>
#include <TLine.h>
#include <TPad.h>
#include <TText.h>

#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {
namespace Detail {
class RDrawStorage;

// clang-format off
/**
\class ROOT::Experimental::Detail::RMetaDataBox
\ingroup NTupleDraw
\brief A TBox which contains metadata information of a RNTuple

A RMetaDataBox is drawn on the TCanvas showing the RNTuple storage layout and represents some metadata (header or footer) in the RNTuple. It also holds some data of the metadata it represents, like its byte size.
*/
// clang-format on
class RMetaDataBox : public TBox {
private:
   std::string fDescription; // e.g. "Header" or "Footer"
   const std::uint32_t fNBytesInStorage;
   RDrawStorage *fParent;

public:
   RMetaDataBox() : RMetaDataBox(0, 0, 0, 0, "", 0, nullptr) {}
   RMetaDataBox(double x1, double y1, double x2, double y2, std::string description, std::uint32_t nBytes,
                RDrawStorage *parent, std::int32_t color = kGray);
   std::uint32_t GetNBytesInStorage() const { return fNBytesInStorage; }
   void Dump() const final;    // *MENU*
   void Inspect() const final; // *MENU*
   ClassDef(RMetaDataBox, 1)
};

// clang-format off
/**
\class ROOT::Experimental::RPageBox
\ingroup NTupleDraw
\brief A TBox which represents a RPage

A RPageBox is drawn on the TCanvas showing the RNTuple storage layout and represents a RPage in the RNTuple. It also holds various data of a RPage, which allows the user to dump/inspect the RPageBox to obtain information about the RPage.
*/
// clang-format on
class RPageBox : public TBox {
private:
   std::string fFieldName;
   std::string fFieldType;
   std::string fColumnType;
   DescriptorId_t fFieldId;
   DescriptorId_t fColumnId;
   DescriptorId_t fClusterId;
   ClusterSize_t::ValueType fNElements;
   ClusterSize_t::ValueType fElementSizeOnDisk;
   NTupleSize_t fGlobalRangeStart;
   NTupleSize_t fClusterRangeStart;
   RClusterDescriptor::RLocator fLocator; // required for sorting
   RDrawStorage *fParent;
   std::size_t fPageBoxId;

public:
   RPageBox()
      : RPageBox(0, 0, 0, 0, "", "", 0, 0, 0, EColumnType::kUnknown, 0, 0, 0, RClusterDescriptor::RLocator(), nullptr)
   {
   }
   RPageBox(double x1, double y1, double x2, double y2, std::string fieldName, std::string fieldType,
            DescriptorId_t fieldId, DescriptorId_t columnId, DescriptorId_t clusterId, EColumnType columnType,
            ClusterSize_t::ValueType nElements, NTupleSize_t globalRangeStart, NTupleSize_t clusterRangeStart,
            RClusterDescriptor::RLocator locator, RDrawStorage *parent, std::size_t pageBoxId = 0);
   DescriptorId_t GetFieldId() const { return fFieldId; }
   DescriptorId_t GetClusterId() const { return fClusterId; }
   const RClusterDescriptor::RLocator &GetLocator() const { return fLocator; }
   void SetPageId(std::size_t pageId) { fPageBoxId = pageId; }
   void Dump() const final;    // *MENU*
   void Inspect() const final; // *MENU*
   ClassDef(RPageBox, 1)
};

// clang-format off
/**
\class ROOT::Experimental::RDrawStorage
\ingroup NTupleDraw
\brief Main class for drawing the storage layout of a RNTuple

This class coordinates the drawing process of the storage layout of a RNTuple. It also holds all generated unique pointers with a static member variable until ROOT is terminated, in order for the drawing to persist.
*/
// clang-format on
class RDrawStorage {
private:
   std::size_t fNEntries;
   std::size_t fTotalNumBytes;
   // E.g. if the total number of bytes in a file is 42*1024*1024 bytes = 42 MB, then fScalingFactorOfAxis is set to
   // 1024*1024, so that 42 can be displayed on the axis.
   std::size_t fScalingFactorOfAxis = 1;
   NTupleSize_t fNFields;
   NTupleSize_t fNColumns;
   NTupleSize_t fNClusters;
   /// For every page in the ntuple, there exists an entry in fPageBoxes.
   std::vector<std::unique_ptr<RPageBox>> fPageBoxes;
   std::vector<std::unique_ptr<TText>> fTexts;
   std::vector<std::unique_ptr<TLine>> fLines;
   // Not Stored as a unique_ptr because ROOT will sometimes delete fPad before the destructor of this class is called.
   TPad *fPad;
   // Not Stored as as unique_ptrs in a vector because ROOT will sometimes delete fTH1Fs before the destructor of this class is called.
   std::vector<TH1F*> fTH1Fs;
   std::string fNTupleName;
   std::string fAxisTitle; // for TLatex in RDrawStorage::Draw()
   std::unique_ptr<RMetaDataBox> fHeaderBox;
   std::unique_ptr<RMetaDataBox> fFooterBox;
   std::unique_ptr<TLegend> fLegend;
   void SetPageIds();

public:
   // holds all generated cavas pointers in case they need to be manually destructed in the future due to changes in
   // ROOT.
   std::vector<TCanvas*> fCanvasPtrs;
   RDrawStorage(RNTupleReader *reader);
   /// holds all created RDrawStorage instances until the lifetime
   static std::vector<RDrawStorage> fgDrawStorageVec;
   static std::int32_t GetColourFromFieldId(DescriptorId_t fieldId);
   std::size_t GetPageBoxSize() const { return fPageBoxes.size(); }
   std::size_t GetTotalNumBytes() const { return fTotalNumBytes; }
   std::size_t GetScalingFactorOfAxis() const { return fScalingFactorOfAxis; }
   NTupleSize_t GetNFields() const { return fNFields; }
   NTupleSize_t GetNColumns() const { return fNColumns; }
   NTupleSize_t GetNClusters() const { return fNClusters; }
   static void RPageBoxClicked();
   void Draw();
};
} // namespace Detail

// clang-format off
/**
\class ROOT::Experimental::RNTupleDraw
\ingroup NTupleDraw
\brief User interface for drawing the structure of an ntuple

This class acts as a user interface like RNTupleWriter and RNTupleReader. It acts as a delegeator to RDrawStorage instead of doing the drawing job itself. This way the lifetime of the data displayed on the canvas is tied to the termination of the ROOT program and not to the destruction of a RNTupleDraw or RNTupleReader instance.
*/
// clang-format on
class RNTupleDraw {
private:
   Detail::RDrawStorage *fStorage;
   bool fEmpty = false;
public:
   RNTupleDraw(const std::unique_ptr<RNTupleReader> &reader);
   static std::unique_ptr<RNTupleDraw> Open(const std::unique_ptr<RNTupleReader> &reader);
   void Draw();
};
} // namespace Experimental
} // namespace ROOT

#endif
