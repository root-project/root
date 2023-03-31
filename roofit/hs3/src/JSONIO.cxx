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

#include <RooFit/Detail/JSONInterface.h>

#include <RooAbsPdf.h>

#include <TClass.h>

#include <iostream>
#include <fstream>

namespace RooFit {
namespace JSONIO {

ImportMap &importers()
{
   static ImportMap _importers;
   return _importers;
}

ExportMap &exporters()
{
   static ExportMap _exporters;
   return _exporters;
}

ImportExpressionMap &pdfImportExpressions()
{
   static ImportExpressionMap _pdfFactoryExpressions;
   return _pdfFactoryExpressions;
}

ImportExpressionMap &functionImportExpressions()
{
   static ImportExpressionMap _funcFactoryExpressions;
   return _funcFactoryExpressions;
}

ExportKeysMap &exportKeys()
{
   static ExportKeysMap _exportKeys;
   return _exportKeys;
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

int removeImporters(const std::string &needle)
{
   int n = 0;
   for (auto &element : importers()) {
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

int removeExporters(const std::string &needle)
{
   int n = 0;
   for (auto &element : exporters()) {
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

void printImporters()
{
   for (const auto &x : importers()) {
      for (const auto &ePtr : x.second) {
         // Passing *e directory to typeid results in clang warnings.
         auto const &e = *ePtr;
         std::cout << x.first << "\t" << typeid(e).name() << std::endl;
      }
   }
}
void printExporters()
{
   for (const auto &x : exporters()) {
      for (const auto &ePtr : x.second) {
         // Passing *e directory to typeid results in clang warnings.
         auto const &e = *ePtr;
         std::cout << x.first->GetName() << "\t" << typeid(e).name() << std::endl;
      }
   }
}

void loadFactoryExpressions(const std::string &fname)
{
   auto &pdfFactoryExpressions = RooFit::JSONIO::pdfImportExpressions();
   auto &funcFactoryExpressions = RooFit::JSONIO::functionImportExpressions();

   // load a yml file defining the factory expressions
   std::ifstream infile(fname);
   if (!infile.is_open()) {
      std::cerr << "unable to read file '" << fname << "'" << std::endl;
      return;
   }

   std::unique_ptr<RooFit::Detail::JSONTree> tree = RooFit::Detail::JSONTree::create(infile);
   const RooFit::Detail::JSONNode &n = tree->rootnode();
   for (const auto &cl : n.children()) {
      std::string key = cl.key();
      if (!cl.has_child("class")) {
         std::cerr << "error in file '" << fname << "' for entry '" << key << "': 'class' key is required!"
                   << std::endl;
         continue;
      }
      std::string classname(cl["class"].val());
      TClass *c = TClass::GetClass(classname.c_str());
      if (!c) {
         std::cerr << "unable to find class " << classname << ", skipping." << std::endl;
      } else {
         RooFit::JSONIO::ImportExpression ex;
         ex.tclass = c;
         if (!cl.has_child("arguments")) {
            std::cerr << "class " << classname << " seems to have no arguments attached, skipping" << std::endl;
            continue;
         }
         for (const auto &arg : cl["arguments"].children()) {
            ex.arguments.push_back(arg.val());
         }
         if (c->InheritsFrom(RooAbsPdf::Class())) {
            pdfFactoryExpressions[key] = ex;
         } else if (c->InheritsFrom(RooAbsReal::Class())) {
            funcFactoryExpressions[key] = ex;
         } else {
            std::cerr << "class " << classname << " seems to not inherit from any suitable class, skipping"
                      << std::endl;
         }
      }
   }
}

void clearFactoryExpressions()
{
   // clear all factory expressions
   RooFit::JSONIO::pdfImportExpressions().clear();
   RooFit::JSONIO::functionImportExpressions().clear();
}

void printFactoryExpressions()
{
   // print all factory expressions
   for (auto it : RooFit::JSONIO::pdfImportExpressions()) {
      std::cout << it.first;
      std::cout << " " << it.second.tclass->GetName();
      for (auto v : it.second.arguments) {
         std::cout << " " << v;
      }
      std::cout << std::endl;
   }
   for (auto it : RooFit::JSONIO::functionImportExpressions()) {
      std::cout << it.first;
      std::cout << " " << it.second.tclass->GetName();
      for (auto v : it.second.arguments) {
         std::cout << " " << v;
      }
      std::cout << std::endl;
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// RooProxy-based export handling
///////////////////////////////////////////////////////////////////////////////////////////////////////

void loadExportKeys(const std::string &fname)
{
   auto &exportKeys = RooFit::JSONIO::exportKeys();

   // load a yml file defining the export keys
   std::ifstream infile(fname);
   if (!infile.is_open()) {
      std::cerr << "unable to read file '" << fname << "'" << std::endl;
      return;
   }

   std::unique_ptr<RooFit::Detail::JSONTree> tree = RooFit::Detail::JSONTree::create(infile);
   const RooFit::Detail::JSONNode &n = tree->rootnode();
   for (const auto &cl : n.children()) {
      std::string classname = cl.key();
      TClass *c = TClass::GetClass(classname.c_str());
      if (!c) {
         std::cerr << "unable to find class " << classname << ", skipping." << std::endl;
      } else {
         RooFit::JSONIO::ExportKeys ex;
         if (!cl.has_child("type")) {
            std::cerr << "class " << classname << "has not type key set, skipping" << std::endl;
            continue;
         }
         if (!cl.has_child("proxies")) {
            std::cerr << "class " << classname << "has no proxies identified, skipping" << std::endl;
            continue;
         }
         ex.type = cl["type"].val();
         for (const auto &k : cl["proxies"].children()) {
            ex.proxies[k.key()] = k.val();
         }
         exportKeys[c] = ex;
      }
   }
}

void clearExportKeys()
{
   // clear all export keys
   RooFit::JSONIO::exportKeys().clear();
}

void printExportKeys()
{
   // print all export keys
   for (const auto &it : RooFit::JSONIO::exportKeys()) {
      std::cout << it.first->GetName() << ": " << it.second.type;
      for (const auto &kv : it.second.proxies) {
         std::cout << " " << kv.first << "=" << kv.second;
      }
      std::cout << std::endl;
   }
}

} // namespace JSONIO
} // namespace RooFit
