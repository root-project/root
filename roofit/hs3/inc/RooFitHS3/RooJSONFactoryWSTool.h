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

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class RooArgList;
class RooAbsData;
class RooArgSet;
class RooAbsArg;
class RooAbsPdf;
class RooDataHist;
class RooDataSet;
class RooRealVar;
class RooWorkspace;

namespace RooFit {
namespace JSONIO {
namespace Detail {
class Domains;
}
} // namespace JSONIO
namespace Detail {
class JSONNode;
class JSONTree;
} // namespace Detail
} // namespace RooFit
namespace RooStats {
class ModelConfig;
}

class TH1;
class TClass;

class RooJSONFactoryWSTool {
public:
   static std::ostream &log(int level);

   static std::string name(const RooFit::Detail::JSONNode &n);

   template <class T>
   T *request(const std::string &objname, const std::string &requestAuthor);

   RooJSONFactoryWSTool(RooWorkspace &ws);

   ~RooJSONFactoryWSTool();

   RooWorkspace *workspace() { return &_workspace; }

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
            text += "nullptr";
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
            names.push_back("nullptr");
         else
            names.push_back(it->GetName());
      }
      return names;
   }

   static std::string genPrefix(const RooFit::Detail::JSONNode &p, bool trailing_underscore);
   static void exportHistogram(const TH1 &h, RooFit::Detail::JSONNode &n, const std::vector<std::string> &obsnames,
                               const TH1 *errH = nullptr, bool writeObservables = true, bool writeErrors = true);
   static void writeObservables(const TH1 &h, RooFit::Detail::JSONNode &n, const std::vector<std::string> &varnames);

   static std::unique_ptr<RooDataHist> readBinnedData(const RooFit::Detail::JSONNode &n, const std::string &namecomp);
   static std::unique_ptr<RooDataHist>
   readBinnedData(const RooFit::Detail::JSONNode &n, const std::string &namecomp, RooArgList varlist);

   static void
   getObservables(RooWorkspace &ws, const RooFit::Detail::JSONNode &n, const std::string &obsnamecomp, RooArgSet &out);

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

   void importFunctions(const RooFit::Detail::JSONNode &n);
   void importFunction(const RooFit::Detail::JSONNode &n, bool isPdf);
   RooFit::Detail::JSONNode *exportObject(const RooAbsArg *func);

   static std::unique_ptr<RooFit::Detail::JSONTree> createNewJSONTree();

private:
   struct Config {
      static bool stripObservables;
   };

   struct Var {
      int nbins;
      double min;
      double max;
      std::vector<double> bounds;

      Var(int n) : nbins(n), min(0), max(n) {}
      Var(const RooFit::Detail::JSONNode &val);
   };

   RooFit::Detail::JSONNode &orootnode();
   const RooFit::Detail::JSONNode &irootnode() const;

   std::map<std::string, std::unique_ptr<RooAbsData>> loadData(const RooFit::Detail::JSONNode &n);
   std::unique_ptr<RooDataSet> unbinned(RooDataHist const &hist);
   static RooRealVar *createObservable(RooWorkspace &ws, const std::string &name, const RooJSONFactoryWSTool::Var &var);

   class MissingRootnodeError : public std::exception {
   public:
      const char *what() const noexcept override { return "no rootnode set"; }
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
      const char *what() const noexcept override { return _message.c_str(); }
   };

   // error handling helpers
   void exportData(RooAbsData *data, RooFit::Detail::JSONNode &n);
   static std::vector<std::vector<int>> generateBinIndices(const RooArgList &vars);
   static std::map<std::string, RooJSONFactoryWSTool::Var>
   readObservables(const RooFit::Detail::JSONNode &n, const std::string &obsnamecomp);

   void importAllNodes(const RooFit::Detail::JSONNode &n);

   void importPdfs(const RooFit::Detail::JSONNode &n);
   void importVariables(const RooFit::Detail::JSONNode &n);
   void importVariable(const RooFit::Detail::JSONNode &n);
   void importDependants(const RooFit::Detail::JSONNode &n);
   void importAnalysis(const RooFit::Detail::JSONNode &analysisNode, const RooFit::Detail::JSONNode &likelihoodsNode);

   bool find(const RooFit::Detail::JSONNode &n, const std::string &elem);
   void append(RooFit::Detail::JSONNode &n, const std::string &elem);

   void exportAttributes(const RooAbsArg *arg, RooFit::Detail::JSONNode &n);
   void exportVariable(const RooAbsArg *v, RooFit::Detail::JSONNode &n);
   void exportVariables(const RooArgSet &allElems, RooFit::Detail::JSONNode &n);

   void exportAllObjects(RooFit::Detail::JSONNode &n);
   void exportDependants(const RooAbsArg *source);
   void exportTopLevelPdf(RooFit::Detail::JSONNode &node, RooAbsPdf const &pdf, std::string const &modelConfigName);

   void tagVariables(RooFit::Detail::JSONNode &rootnode, RooArgSet const *args, const char *tag);

   void exportModelConfig(RooFit::Detail::JSONNode &n, RooStats::ModelConfig const &mc);

   // member variables
   const RooFit::Detail::JSONNode *_rootnode_input = nullptr;
   RooFit::Detail::JSONNode *_rootnode_output = nullptr;
   RooWorkspace &_workspace;

   // objects to represent intermediate information
   std::unique_ptr<RooFit::JSONIO::Detail::Domains> _domains;
};
#endif
