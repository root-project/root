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

#ifndef RooFitHS3_JSONIO_h
#define RooFitHS3_JSONIO_h

#include <map>
#include <memory>
#include <string>
#include <vector>

class RooAbsArg;
class RooJSONFactoryWSTool;

class TClass;

namespace RooFit {
namespace Detail {
class JSONNode;
class JSONTree;
} // namespace Detail

namespace JSONIO {

class Importer {
public:
   virtual bool importPdf(RooJSONFactoryWSTool *, const RooFit::Detail::JSONNode &) const { return false; }
   virtual bool importFunction(RooJSONFactoryWSTool *, const RooFit::Detail::JSONNode &) const { return false; }
   virtual ~Importer() {}
};
class Exporter {
public:
   virtual std::string const &key() const = 0;
   virtual bool autoExportDependants() const { return true; }
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *, RooFit::Detail::JSONNode &) const
   {
      return false;
   }
   virtual ~Exporter() {}
};
struct ExportKeys {
   std::string type;
   std::map<std::string, std::string> proxies;
};
struct ImportExpression {
   TClass const *tclass = nullptr;
   std::vector<std::string> arguments;
};

using ImportMap = std::map<const std::string, std::vector<std::unique_ptr<const Importer>>>;
using ExportMap = std::map<TClass const *, std::vector<std::unique_ptr<const Exporter>>>;
using ExportKeysMap = std::map<TClass const *, ExportKeys>;
using ImportExpressionMap = std::map<const std::string, ImportExpression>;

ImportMap &importers();
ExportMap &exporters();
ImportExpressionMap &pdfImportExpressions();
ImportExpressionMap &functionImportExpressions();
ExportKeysMap &exportKeys();

template <class T>
static bool registerImporter(const std::string &key, bool topPriority = true)
{
   return registerImporter(key, std::make_unique<T>(), topPriority);
}
template <class T>
static bool registerExporter(const TClass *key, bool topPriority = true)
{
   return registerExporter(key, std::make_unique<T>(), topPriority);
}

bool registerImporter(const std::string &key, std::unique_ptr<const Importer> f, bool topPriority = true);
bool registerExporter(const TClass *key, std::unique_ptr<const Exporter> f, bool topPriority = true);
int removeImporters(const std::string &needle);
int removeExporters(const std::string &needle);
void printImporters();
void printExporters();

void loadFactoryExpressions(const std::string &fname);
void clearFactoryExpressions();
void printFactoryExpressions();
void loadExportKeys(const std::string &fname);
void clearExportKeys();
void printExportKeys();

} // namespace JSONIO

} // namespace RooFit

#endif
