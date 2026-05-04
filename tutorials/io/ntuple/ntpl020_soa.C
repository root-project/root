/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Example of RNTuple I/O on collections with a SoA memory layout
///
/// RNTuple on-disk collections can be used in struct-of-arrays (SoA) memory layout.
/// An RNTuple SoA class consists of persistent members of type `RVec` corresponding to
/// and underlying record type, as shown in this tutorial.
///
/// NOTE: The RNTuple SoA I/O is still experimental at this point.
/// Functionality and interface are subject to changes.
///
/// \macro_code
///
/// \date April 2026
/// \author The ROOT Team

#include <ROOT/REntry.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RVec.hxx>

#include <TCanvas.h>
#include <TClass.h>
#include <TDictAttributeMap.h>
#include <TEllipse.h>
#include <TGraph.h>
#include <TRandom.h>

#include <iostream>
#include <memory>

constexpr const char *kFileName = "ntpl020_soa.root";
constexpr const char *kNTupleName = "ntpl";

// The SoA class for this tutorial. Contains a number of 2D points. All vectors have to have the same length.
// Note that RVecs can adopt memory.
struct PointSoA {
   ROOT::RVec<float> fX;
   ROOT::RVec<float> fY;
};

// The underlying record type for the SoA class. Members between the SoA class and the underlying record type
// are matched by name. Every member of type `T` in the underlying record type has to be of type `ROOT::RVec<T>` in
// the SoA class.
struct PointRecord {
   float fX;
   float fY;
};

void Write()
{
   // Create a model with a SoA field
   auto model = ROOT::RNTupleModel::CreateBare();
   model->AddField(ROOT::RFieldBase::Create("points", "PointSoA").Unwrap());

   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), kNTupleName, kFileName);
   auto entry = writer->GetModel().CreateBareEntry();

   for (auto nPoints : {100, 500, 1000, 10000}) {
      // We will use our own memory are to store the points
      auto memory = std::make_unique<float[]>(nPoints * 2);

      // Create random points in the unit square
      gRandom->RndmArray(2 * nPoints, memory.get());

      // Adopt the memory by a PointSoA object. First all x values, then all y values.
      PointSoA points{ROOT::RVec<float>(memory.get(), nPoints), ROOT::RVec<float>(memory.get() + nPoints, nPoints)};

      entry->BindRawPtr("points", &points);
      writer->Fill(*entry);
   }
}

void Read()
{
   auto reader = ROOT::RNTupleReader::Open(kNTupleName, kFileName);

   // Show on-disk layout: collection of underlying record type (AoS)
   reader->PrintInfo();

   // Used to draw the points
   auto canvas = new TCanvas("c", "", 1200, 1200);
   canvas->Divide(2, 2);

   // Read back the points in two steps: first get the size and then read into adopted RVecs.
   auto viewSize = reader->GetCollectionView("points");
   PointSoA points;
   auto viewSoA = reader->GetView("points", &points, "PointSoA");

   for (auto i : reader->GetEntryRange()) {
      const auto N = viewSize(i); // size of this entry's SoA collection
      auto memory = std::make_unique<float[]>(N * 2);

      points.fX = ROOT::RVec<float>(memory.get(), N);
      points.fY = ROOT::RVec<float>(memory.get() + N, N);
      viewSoA(i);

      // Use the raw memory area to draw the points
      canvas->cd(i + 1);
      auto *graph = new TGraph(N, &memory[0], &memory[N]);
      graph->SetTitle((std::to_string(N) + " Points").c_str());
      graph->SetMarkerStyle(29);
      graph->SetMarkerSize(1);
      graph->SetMarkerColor(kRed);
      graph->Draw("AP");
      auto *circle = new TEllipse(0.5, 0.5, 0.5, 0.5);
      circle->SetFillStyle(0);
      circle->SetLineColor(kBlue);
      circle->SetLineWidth(4);
      circle->Draw();

      // Use adopted RVec's to approximate PI
      points.fX -= 0.5;
      points.fY -= 0.5;
      auto isInCircle = points.fX * points.fX + points.fY * points.fY < 0.25;
      auto hits = ROOT::VecOps::Sum(isInCircle);
      float approxPI = 4.0 * static_cast<float>(hits) / static_cast<float>(N);
      std::cout << "Approximated PI with " << N << " points to " << approxPI << std::endl;
   }
   canvas->Update();
}

void ntpl020_soa()
{
   // Usually, the SoA class dictionary definition would mark it as a SoA class of the corresponding
   // underlying record type like this
   //     #pragma link C++ options=rntupleSoARecord(PointRecord) class PointSoA+;
   // For the interpreted classes in this tutorial, we mark the SoA class at runtime:
   auto cl = TClass::GetClass("PointSoA");
   cl->CreateAttributeMap();
   cl->GetAttributeMap()->AddProperty("rntuple.SoARecord", "PointRecord");

   Write();
   Read();
}
