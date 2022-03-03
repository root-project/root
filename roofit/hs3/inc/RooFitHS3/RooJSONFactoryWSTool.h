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

#ifndef RooFitHS3_RooJSONFactoryWSTool_h
#define RooFitHS3_RooJSONFactoryWSTool_h

#include <RooArgSet.h>
#include <RooGlobalFunc.h>

#include <map>
#include <memory>
#include <string>

class RooAbsArg;
class RooAbsReal;
class RooAbsPdf;
class RooDataHist;
class RooDataSet;
class RooRealVar;
class RooRealVar;
class RooWorkspace;

class TH1;

namespace RooFit {
namespace Experimental {
class JSONNode;
}
} // namespace RooFit

class RooJSONFactoryWSTool {
public:
   struct Config {
      static bool stripObservables;
   };

   class Importer {
   public:
      virtual bool importPdf(RooJSONFactoryWSTool *, const RooFit::Experimental::JSONNode &) const { return false; }
      virtual bool importFunction(RooJSONFactoryWSTool *, const RooFit::Experimental::JSONNode &) const
      {
         return false;
      }
      virtual ~Importer(){};
   };
   class Exporter {
   public:
      virtual std::string const &key() const = 0;
      virtual bool autoExportDependants() const { return true; }
      virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *, RooFit::Experimental::JSONNode &) const
      {
         return false;
      }
      virtual ~Exporter(){};
   };
   struct ExportKeys {
      std::string type;
      std::map<std::string, std::string> proxies;
   };
   struct ImportExpression {
      TClass const* tclass = nullptr;
      std::vector<std::string> arguments;
   };

   typedef std::map<const std::string, std::vector<std::unique_ptr<const Importer>>> ImportMap;
   typedef std::map<TClass const*, std::vector<std::unique_ptr<const Exporter>>> ExportMap;
   typedef std::map<TClass const*, ExportKeys> ExportKeysMap;
   typedef std::map<const std::string, ImportExpression> ImportExpressionMap;

   // The following maps to hold the importers and exporters for runtime lookup
   // could also be static variables directly, but to avoid the static
   // initialization order fiasco these are functions that return static
   // variables.
   static ImportMap &staticImporters();
   static ExportMap &staticExporters();
   static ImportExpressionMap &staticPdfImportExpressions();
   static ImportExpressionMap &staticFunctionImportExpressions();
   static ExportKeysMap &staticExportKeys();

   struct Var {
      int nbins;
      double min;
      double max;
      std::vector<double> bounds;

      Var(int n) : nbins(n), min(0), max(n) {}
      Var(const RooFit::Experimental::JSONNode &val);
   };

   std::ostream &log(RooFit::MsgLevel level) const;

protected:
   struct Scope {
      RooArgSet observables;
      std::map<std::string, RooAbsArg *> objects;
   };
   mutable Scope _scope;
   const RooFit::Experimental::JSONNode *_rootnode_input = nullptr;
   RooFit::Experimental::JSONNode *_rootnode_output = nullptr;

   RooFit::Experimental::JSONNode &orootnode();
   const RooFit::Experimental::JSONNode &irootnode() const;

   RooWorkspace *_workspace;

   std::map<std::string, std::unique_ptr<RooAbsData>> loadData(const RooFit::Experimental::JSONNode &n);
   std::unique_ptr<RooDataSet> unbinned(RooDataHist const &hist);
   RooRealVar *getWeightVar(const char *name);
   RooRealVar *createObservable(const std::string &name, const RooJSONFactoryWSTool::Var &var);

public:
   class MissingRootnodeError : public std::exception {
   public:
      virtual const char *what() const noexcept override { return "no rootnode set"; }
   };

   class DependencyMissingError : public std::exception {
      std::string _parent, _child, _class, _message;

   public:
      DependencyMissingError(const std::string &p, const std::string &c, const std::string &classname)
         : _parent(p), _child(c), _class(classname)
      {
         _message = "object '" + _parent + "' is missing dependency '" + _child + "' of type '" + _class + "'";
      };
      const std::string &parent() const { return _parent; }
      const std::string &child() const { return _child; }
      const std::string &classname() const { return _class; }
      virtual const char *what() const noexcept override { return _message.c_str(); }
   };
   friend DependencyMissingError;

   static std::string name(const RooFit::Experimental::JSONNode &n);

   template <class T>
   T *request(const std::string &objname, const std::string &requestAuthor);

   RooJSONFactoryWSTool(RooWorkspace &ws) : _workspace{&ws} {}
   RooWorkspace *workspace() { return this->_workspace; }

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

   static bool registerImporter(const std::string &key, std::unique_ptr<const RooJSONFactoryWSTool::Importer> f,
                                bool topPriority = true);
   static bool registerExporter(const TClass *key, std::unique_ptr<const RooJSONFactoryWSTool::Exporter> f,
                                bool topPriority = true);
   static int removeImporters(const std::string &needle);
   static int removeExporters(const std::string &needle);
   static void printImporters();
   static void printExporters();

   static ImportMap const &importers() { return staticImporters(); }
   static ExportMap const &exporters() { return staticExporters(); }
   static ImportExpressionMap const &pdfImportExpressions() { return staticPdfImportExpressions(); }
   static ImportExpressionMap const &functionImportExpressions() { return staticFunctionImportExpressions(); }
   static ExportKeysMap const &exportKeys() { return staticExportKeys(); }

   // error handling helpers
   static void error(const char *s) { throw std::runtime_error(s); }
   static void error(const std::string &s) { throw std::runtime_error(s); }
   template <class T>
   static std::string concat(const T *items, const std::string &sep = ",")
   {
      // Returns a string being the concatenation of strings in input list <items>
      // (names of objects obtained using GetName()) separated by string <sep>.
      bool first = true;
      std::string text;

      // iterate over strings in list
      for (auto it : *items) {
         if (!first) {
            // insert separator string
            text += sep;
         } else {
            first = false;
         }
         if (!it)
            text += "NULL";
         else
            text += it->GetName();
      }
      return text;
   }
   template <class T>
   static std::vector<std::string> names(const T *items)
   {
      // Returns a string being the concatenation of strings in input list <items>
      // (names of objects obtained using GetName()) separated by string <sep>.
      std::vector<std::string> names;
      // iterate over strings in list
      for (auto it : *items) {
         if (!it)
            names.push_back("NULL");
         else
            names.push_back(it->GetName());
      }
      return names;
   }

   static std::string genPrefix(const RooFit::Experimental::JSONNode &p, bool trailing_underscore);
   static void exportHistogram(const TH1 &h, RooFit::Experimental::JSONNode &n,
                               const std::vector<std::string> &obsnames, const TH1 *errH = 0,
                               bool writeObservables = true, bool writeErrors = true);
   void exportData(RooAbsData *data, RooFit::Experimental::JSONNode &n);
   static void
   writeObservables(const TH1 &h, RooFit::Experimental::JSONNode &n, const std::vector<std::string> &varnames);
   static std::vector<std::vector<int>> generateBinIndices(const RooArgList &vars);
   std::unique_ptr<RooDataHist>
   readBinnedData(const RooFit::Experimental::JSONNode &n, const std::string &namecomp, RooArgList observables);
   static std::map<std::string, RooJSONFactoryWSTool::Var>
   readObservables(const RooFit::Experimental::JSONNode &n, const std::string &obsnamecomp);
   RooArgSet getObservables(const RooFit::Experimental::JSONNode &n, const std::string &obsnamecomp);
   void setScopeObservables(const RooArgList &args);
   RooAbsArg *getScopeObject(const std::string &name);
   void setScopeObject(const std::string &key, RooAbsArg *obj);
   void clearScope();

   bool importJSON(std::string const &filename);
   bool importYML(std::string const &filename);
   bool importJSON(std::istream &os);
   bool importYML(std::istream &os);
   bool exportJSON(std::string const &fileName);
   bool exportYML(std::string const &fileName);
   bool exportJSON(std::ostream &os);
   bool exportYML(std::ostream &os);

   std::string exportJSONtoString();
   std::string exportYMLtoString();
   bool importJSONfromString(const std::string &s);
   bool importYMLfromString(const std::string &s);

   static void loadFactoryExpressions(const std::string &fname);
   static void clearFactoryExpressions();
   static void printFactoryExpressions();
   static void loadExportKeys(const std::string &fname);
   static void clearExportKeys();
   static void printExportKeys();

   void importAllNodes(const RooFit::Experimental::JSONNode &n);

   void importFunctions(const RooFit::Experimental::JSONNode &n);
   void importPdfs(const RooFit::Experimental::JSONNode &n);
   void importVariables(const RooFit::Experimental::JSONNode &n);
   void importFunction(const RooFit::Experimental::JSONNode &n, bool isPdf);
   void importVariable(const RooFit::Experimental::JSONNode &n);
   void configureVariable(const RooFit::Experimental::JSONNode &p, RooRealVar &v);
   void importDependants(const RooFit::Experimental::JSONNode &n);

   void configureToplevelPdf(const RooFit::Experimental::JSONNode &n, RooAbsPdf &pdf);

   bool find(const RooFit::Experimental::JSONNode &n, const std::string &elem);
   void append(RooFit::Experimental::JSONNode &n, const std::string &elem);

   void exportAttributes(const RooAbsArg *arg, RooFit::Experimental::JSONNode &n);
   void exportVariable(const RooAbsReal *v, RooFit::Experimental::JSONNode &n);
   void exportVariables(const RooArgSet &allElems, RooFit::Experimental::JSONNode &n);
   void exportObject(const RooAbsArg *func, RooFit::Experimental::JSONNode &n);
   void exportFunctions(const RooArgSet &allElems, RooFit::Experimental::JSONNode &n);

   void exportAllObjects(RooFit::Experimental::JSONNode &n);
   void exportDependants(const RooAbsArg *source, RooFit::Experimental::JSONNode &n);
   void exportDependants(const RooAbsArg *source, RooFit::Experimental::JSONNode *n);
};
#endif
