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

#include <RooAbsPdf.h>

#include <TClass.h>

#include <iostream>

namespace RooFit::JSONIO {

ImportMap &importers()
{
   static ImportMap _importers;
   return _importers;
}

namespace {

auto &exportersToAdd()
{
   static std::map<const std::string, std::vector<std::unique_ptr<const Exporter>>> toAdd;
   return toAdd;
}

ExportMap &exportersImpl()
{
   static ExportMap _exporters;
   return _exporters;
}

} // namespace

ExportMap &exporters()
{
   // If there are exporters to be added, do this now.
   for (auto &item : exportersToAdd()) {
      TClass *klass = TClass::GetClass(item.first.c_str());
      auto &exporters = exportersImpl()[klass]; // registered exporters so far
      auto &toAdd = item.second;                // exporters to add

      // Find the nullptr separator
      auto nullIt = std::find(toAdd.begin(), toAdd.end(), nullptr);

      if (nullIt == toAdd.end()) {
         throw std::runtime_error("toAdd does not contain nullptr separator");
      }

      // Move elements after nullptr to the back
      exporters.insert(exporters.end(), std::make_move_iterator(nullIt + 1), std::make_move_iterator(toAdd.end()));

      // Move elements before nullptr to the front
      exporters.insert(exporters.begin(), std::make_move_iterator(toAdd.begin()), std::make_move_iterator(nullIt));

      toAdd.clear();
   }
   exportersToAdd().clear();
   return exportersImpl();
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
   auto &vec = exportersToAdd()[key];
   // The vector starts out with just a nullptr separator. Elements before it
   // will be added to the top of the exporter queue, and the elements after
   // get appended to the end.
   if (vec.empty()) {
      vec.emplace_back(nullptr);
   }
   vec.insert(topPriority ? vec.begin() : vec.end(), std::move(f));
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
