/*
 * Project: RooFit
 * Authors:
 *   Carsten D. Burgard, DESY/ATLAS, Dec 2021
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFitHS3/JSONIO.h>

#include <TClass.h>

#include <iostream>

namespace RooFit::JSONIO {

ImportMap &importers()
{
   static ImportMap _importers;
   return _importers;
}

namespace {

// An exporter that was registered by the name of its class, for which there might have been no TClass yet.
struct ExporterByName {
   std::string className;
   std::unique_ptr<const Exporter> exporter;
   bool topPriority = true;
};

std::vector<ExporterByName> &exportersByName()
{
   static std::vector<ExporterByName> _exportersByName;
   return _exportersByName;
}

} // namespace

ExportMap &exporters()
{
   static ExportMap _exporters;
   // The exporters that were registered by class name are registered with their TClass now, in the same order.
   std::vector<ExporterByName> byName;
   byName.swap(exportersByName());
   for (auto &item : byName) {
      registerExporter(TClass::GetClass(item.className.c_str()), std::move(item.exporter), item.topPriority);
   }
   return _exporters;
}

bool registerImporter(const std::string &key, std::unique_ptr<const Importer> f, bool topPriority)
{
   auto &vec = importers()[key];
   vec.insert(topPriority ? vec.begin() : vec.end(), std::move(f));
   return true;
}

bool registerExporter(const TClass *key, std::unique_ptr<const Exporter> f, bool topPriority)
{
   auto &vec = exporters()[key];
   vec.insert(topPriority ? vec.begin() : vec.end(), std::move(f));
   return true;
}

bool registerExporter(const std::string &key, std::unique_ptr<const Exporter> f, bool topPriority)
{
   exportersByName().push_back({key, std::move(f), topPriority});
   return true;
}

namespace {

template <class Map>
int removeByTypeName(Map &map, const std::string &needle)
{
   int n = 0;
   for (auto &element : map) {
      for (size_t i = element.second.size(); i > 0; --i) {
         auto *imp = element.second[i - 1].get();
         std::string name(typeid(*imp).name());
         if (name.find(needle) != std::string::npos) {
            element.second.erase(element.second.begin() + i - 1);
            ++n;
         }
      }
   }
   return n;
}

template <class Map, class KeyPrinter>
void printByTypeName(Map &map, KeyPrinter printKey)
{
   for (const auto &x : map) {
      for (const auto &ePtr : x.second) {
         // Passing *e directory to typeid results in clang warnings.
         auto const &e = *ePtr;
         std::cout << printKey(x.first) << "\t" << typeid(e).name() << std::endl;
      }
   }
}

} // namespace

int removeImporters(const std::string &needle)
{
   return removeByTypeName(importers(), needle);
}

int removeExporters(const std::string &needle)
{
   return removeByTypeName(exporters(), needle);
}

void printImporters()
{
   printByTypeName(importers(), [](auto const &key) { return key; });
}
void printExporters()
{
   printByTypeName(exporters(), [](auto const &key) { return key->GetName(); });
}

} // namespace RooFit::JSONIO
