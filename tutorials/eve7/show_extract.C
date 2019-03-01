/// \file
/// \ingroup tutorial_eve7
/// Helper script for showing of extracted / simplified geometries.
/// By default shows a simplified ALICE geometry.
///
/// \image html eve_show_extract.png
/// \macro_code
///
/// \author Matevz Tadel

#include "TFile.h"
#include "TKey.h"

#include "TGeoShape.h"

#include <ROOT/REveManager.hxx>
#include <ROOT/REveGeoShapeExtract.hxx>
#include <ROOT/REveGeoShape.hxx>

namespace REX = ROOT::Experimental;

void show_extract(const char* file="csg.root")
{
   // disable browser cache - all scripts and html files will be loaded every time, useful for development
   // gEnv->SetValue("WebGui.HttpMaxAge", 0);

   auto eveMng = REX::REveManager::Create();

   TFile::Open(file);

   TIter next(gDirectory->GetListOfKeys());

   const TString extract_class("ROOT::Experimental::REveGeoShapeExtract");

   REX::REveGeoShape *eve_shape = nullptr;

   while (auto key = (TKey *)next())
   {
      if (extract_class == key->GetClassName())
      {
         auto gse = (REX::REveGeoShapeExtract*) key->ReadObj();
         eve_shape = REX::REveGeoShape::ImportShapeExtract(gse, 0);
         eveMng->AddGlobalElement(eve_shape);
      }
   }

   if (!eve_shape)
   {
      Error("show_extract.C", "No keys of class '%s'.", extract_class.Data());
      return;
   }

   eve_shape->GetShape()->Draw("ogl");
}
