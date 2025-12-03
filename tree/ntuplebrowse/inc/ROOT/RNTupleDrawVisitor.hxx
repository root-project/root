/// \file ROOT/RNTupleDrawVisitor.hxx
/// \ingroup NTuple
/// \author Sergey Linev <S.Linev@gsi.de>, Jakob Blomer <jblomer@cern.ch>
/// \date 2025-07-24

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleDrawVisitor
#define ROOT_RNTupleDrawVisitor

#include <ROOT/RField.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleView.hxx>

#include <TH1F.h>

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace ROOT {

namespace Internal {

class RNTupleDrawVisitor : public ROOT::Detail::RFieldVisitor {
private:
   std::shared_ptr<ROOT::RNTupleReader> fNtplReader;
   std::unique_ptr<TH1> fHist;
   std::string fTitle;

   /** Test collected entries if it looks like integer values and one can use better binning */
   void TestHistBuffer();

   template <typename ViewT>
   void FillHistogramImpl(ViewT &view)
   {
      fHist = std::make_unique<TH1F>("hdraw", fTitle.c_str(), 100, 0, 0);
      fHist->SetDirectory(nullptr);

      auto bufsize = (fHist->GetBufferSize() - 1) / 2;
      int cnt = 0;
      if (bufsize > 10) {
         bufsize -= 3;
      } else {
         bufsize = -1;
      }

      for (auto i : view.GetFieldRange()) {
         fHist->Fill(view(i));
         if (++cnt == bufsize) {
            TestHistBuffer();
            ++cnt;
         }
      }
      if (cnt < bufsize)
         TestHistBuffer();

      fHist->BufferEmpty();
   }

   template <typename T>
   void FillHistogram(const ROOT::RIntegralField<T> &field)
   {
      auto view = fNtplReader->GetDirectAccessView<T>(field.GetOnDiskId());
      FillHistogramImpl(view);
   }

   template <typename T>
   void FillHistogram(const ROOT::RField<T> &field)
   {
      auto view = fNtplReader->GetView<T>(field.GetOnDiskId());
      FillHistogramImpl(view);
   }

   void FillStringHistogram(const ROOT::RField<std::string> &field)
   {
      std::map<std::string, std::uint64_t> values;

      std::uint64_t nentries = 0;

      auto view = fNtplReader->GetView<std::string>(field.GetOnDiskId());
      for (auto i : view.GetFieldRange()) {
         std::string v = view(i);
         nentries++;
         auto iter = values.find(v);
         if (iter != values.end())
            iter->second++;
         else if (values.size() >= 50)
            return;
         else
            values[v] = 0;
      }

      // now create histogram with labels
      fHist = std::make_unique<TH1F>("h", fTitle.c_str(), 3, 0, 3);
      fHist->SetDirectory(nullptr);
      fHist->SetStats(0);
      fHist->SetEntries(nentries);
      fHist->SetCanExtend(TH1::kAllAxes);
      for (auto &entry : values)
         fHist->Fill(entry.first.c_str(), entry.second);
      fHist->LabelsDeflate();
      fHist->Sumw2(false);
   }

public:
   RNTupleDrawVisitor(std::shared_ptr<ROOT::RNTupleReader> ntplReader, const std::string &title)
      : fNtplReader(ntplReader), fTitle(title)
   {
   }

   TH1 *MoveHist() { return fHist.release(); }

   void VisitField(const ROOT::RFieldBase & /* field */) final {}
   void VisitBoolField(const ROOT::RField<bool> &field) final { FillHistogram(field); }
   void VisitFloatField(const ROOT::RField<float> &field) final { FillHistogram(field); }
   void VisitDoubleField(const ROOT::RField<double> &field) final { FillHistogram(field); }
   void VisitCharField(const ROOT::RField<char> &field) final { FillHistogram(field); }
   void VisitInt8Field(const ROOT::RIntegralField<std::int8_t> &field) final { FillHistogram(field); }
   void VisitInt16Field(const ROOT::RIntegralField<std::int16_t> &field) final { FillHistogram(field); }
   void VisitInt32Field(const ROOT::RIntegralField<std::int32_t> &field) final { FillHistogram(field); }
   void VisitInt64Field(const ROOT::RIntegralField<std::int64_t> &field) final { FillHistogram(field); }
   void VisitStringField(const ROOT::RField<std::string> &field) final { FillStringHistogram(field); }
   void VisitUInt16Field(const ROOT::RIntegralField<std::uint16_t> &field) final { FillHistogram(field); }
   void VisitUInt32Field(const ROOT::RIntegralField<std::uint32_t> &field) final { FillHistogram(field); }
   void VisitUInt64Field(const ROOT::RIntegralField<std::uint64_t> &field) final { FillHistogram(field); }
   void VisitUInt8Field(const ROOT::RIntegralField<std::uint8_t> &field) final { FillHistogram(field); }
   void VisitCardinalityField(const ROOT::RCardinalityField &field) final
   {
      if (const auto f32 = field.As32Bit()) {
         FillHistogram(*f32);
      } else if (const auto f64 = field.As64Bit()) {
         FillHistogram(*f64);
      }
   }
}; // class RDrawVisitor

} // namespace Internal
} // namespace ROOT

#endif
