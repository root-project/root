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

#include <RooFit/Detail/JSONInterface.h>

#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooGlobalFunc.h>
#include <RooWorkspace.h>

#include <map>
#include <stdexcept>
#include <set>
#include <vector>

class RooJSONFactoryWSTool;

namespace RooFit {
namespace HS3 {
struct Distribution;
struct HistFactoryDist;
struct ParameterPoint;
struct ParameterValue;
struct ToolExportState;
struct ToolImportState;
template <class Tag>
struct Ref;
using DistributionRef = Ref<struct DistributionTag>;
void setExportedObject(RooJSONFactoryWSTool &tool, HistFactoryDist dist,
                       std::vector<DistributionRef> channelConstraints);
} // namespace HS3
} // namespace RooFit
namespace RooStats {
class ModelConfig;
}
class RooRealVar;

class RooJSONFactoryWSTool {
public:
   /// Named objects are always stored in lists, as the HS3 standard requires. Kept for backwards compatibility.
   static constexpr bool useListsInsteadOfDicts = true;

   struct Config {
      bool allowExportInvalidNames = true;
      bool allowSanitizeNames = true;
      /// Identify constant factors of HistFactory samples that are used like the inverse bin width of HistFactory
      /// before ROOT 6.26 by their structure, and don't write them, since `histfactory_dist` implies them. Turn this
      /// off if a workspace has such factors that are not meant as bin widths: they are then written as normfactors.
      bool identifyLegacyBinWidthFactors = true;
   };

   static Config &config();

   static RooWorkspace sanitizeWS(const RooWorkspace &ws);
   static RooWorkspace cleanWS(const RooWorkspace &ws, bool onlyModelConfig = false);

   struct CombinedData {
      std::string name;
      std::map<std::string, std::string> components;
   };

   RooJSONFactoryWSTool(RooWorkspace &ws);

   ~RooJSONFactoryWSTool();

   static std::string name(const RooFit::Detail::JSONNode &n);
   static bool isValidName(const std::string &str);
   static bool testValidName(const std::string &str, bool forcError);
   static std::string sanitizeName(const std::string str);
   static void rebuildModelConfigInWorkspace(RooStats::ModelConfig *mc, RooWorkspace &ws);

   static RooFit::Detail::JSONNode &appendNamedChild(RooFit::Detail::JSONNode &node, std::string const &name);
   static RooFit::Detail::JSONNode const *findNamedChild(RooFit::Detail::JSONNode const &node, std::string const &name);

   static void fillSeq(RooFit::Detail::JSONNode &node, RooAbsCollection const &coll, size_t nMax = -1);

   template <class T>
   T *request(const std::string &objname, const std::string &requestAuthor)
   {
      if (T *out = requestImpl<T>(objname)) {
         return out;
      }
      throw DependencyMissingError(requestAuthor, objname, T::Class()->GetName());
   }

   template <class T>
   T *requestArg(const RooFit::Detail::JSONNode &node, const std::string &key)
   {
      std::string requestAuthor(RooJSONFactoryWSTool::name(node));
      if (!node.has_child(key)) {
         RooJSONFactoryWSTool::error("no \"" + key + "\" given in \"" + requestAuthor + "\"");
      }
      return request<T>(node[key].val(), requestAuthor);
   }

   template <class T, class Coll_t>
   Coll_t requestCollection(const RooFit::Detail::JSONNode &node, const std::string &seqName)
   {
      std::string requestAuthor(RooJSONFactoryWSTool::name(node));
      if (!node.has_child(seqName)) {
         RooJSONFactoryWSTool::error("no \"" + seqName + "\" given in \"" + requestAuthor + "\"");
      }
      if (!node[seqName].is_seq()) {
         RooJSONFactoryWSTool::error("\"" + seqName + "\" in \"" + requestAuthor + "\" is not a sequence");
      }

      Coll_t out;
      for (const auto &elem : node[seqName].children()) {
         out.add(*request<T>(elem.val(), requestAuthor));
      }
      return out;
   }

   template <class T>
   RooArgSet requestArgSet(const RooFit::Detail::JSONNode &node, const std::string &seqName)
   {
      return requestCollection<T, RooArgSet>(node, seqName);
   }

   template <class T>
   RooArgList requestArgList(const RooFit::Detail::JSONNode &node, const std::string &seqName)
   {
      return requestCollection<T, RooArgList>(node, seqName);
   }

   RooWorkspace *workspace() { return &_workspace; }

   template <class Obj_t>
   Obj_t &wsImport(Obj_t const &obj)
   {
      _workspace.import(obj, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return *static_cast<Obj_t *>(_workspace.obj(obj.GetName()));
   }

   template <class Obj_t, typename... Args_t>
   Obj_t &wsEmplace(RooStringView name, Args_t &&...args)
   {
      return wsImport(Obj_t(name.c_str(), name.c_str(), std::forward<Args_t>(args)...));
   }

   [[noreturn]] static void error(const char *s);
   [[noreturn]] inline static void error(const std::string &s) { error(s.c_str()); }
   static std::ostream &warning(const std::string &s);

   static RooArgSet readAxes(const RooFit::Detail::JSONNode &node);
   static std::unique_ptr<RooDataHist>
   readBinnedData(const RooFit::Detail::JSONNode &n, const std::string &namecomp, RooArgSet const &vars);

   bool importJSON(std::string const &filename);
   bool importJSON(std::istream &os);
   bool exportJSON(std::string const &fileName);
   bool exportJSON(std::ostream &os);

   std::string exportJSONtoString();
   bool importJSONfromString(const std::string &s);
   void importJSONElement(const std::string &name, const std::string &jsonString);
   void importVariableElement(const RooFit::Detail::JSONNode &n);

   void importFunction(const RooFit::Detail::JSONNode &p, bool importAllDependants);
   void importFunction(const std::string &jsonString, bool importAllDependants);

   static std::unique_ptr<RooFit::Detail::JSONTree> createNewJSONTree();

   static RooFit::Detail::JSONNode &makeVariablesNode(RooFit::Detail::JSONNode &rootNode);

   // error handling helpers
   class DependencyMissingError : public std::runtime_error {
      std::string _parent, _child, _class;

   public:
      DependencyMissingError(const std::string &p, const std::string &c, const std::string &classname)
         : std::runtime_error("object '" + p + "' is missing dependency '" + c + "' of type '" + classname + "'"),
           _parent(p),
           _child(c),
           _class(classname)
      {
      }
      const std::string &parent() const { return _parent; }
      const std::string &child() const { return _child; }
      const std::string &classname() const { return _class; }
   };

   template <typename... Keys_t>
   static RooFit::Detail::JSONNode &getRooFitInternal(RooFit::Detail::JSONNode &node, Keys_t const &...keys)
   {
      return node.get("misc", "ROOT_internal", keys...);
   }

   static void exportAxis(RooFit::Detail::JSONNode &obsNode, RooRealVar const &var);

   static void
   exportHisto(RooArgSet const &vars, std::size_t n, double const *contents, RooFit::Detail::JSONNode &output);

   static void exportArray(std::size_t n, double const *contents, RooFit::Detail::JSONNode &output);

   void exportCategory(RooAbsCategory const &cat, RooFit::Detail::JSONNode &node);

   void queueExport(RooAbsArg const &arg) { _serversToExport.push_back(&arg); }
   void queueExportTemporary(RooAbsArg *arg)
   {
      _serversToExport.push_back(arg);
      _serversToDelete.push_back(arg);
   }

   std::string exportTransformed(const RooAbsReal *original, const std::string &suffix, const std::string &formula);

   void setAttribute(const std::string &obj, const std::string &attrib);
   bool hasAttribute(const std::string &obj, const std::string &attrib);
   std::string getStringAttribute(const std::string &obj, const std::string &attrib);
   void setStringAttribute(const std::string &obj, const std::string &attrib, const std::string &value);

   /// The "hs3_version" of the document that is currently being imported, or an empty string outside of an import.
   std::string const &hs3VersionInput() const { return _hs3VersionInput; }

private:
   template <class T>
   T *requestImpl(const std::string &objname);
   void exportObject(RooAbsArg const &func, std::set<std::string> &exportedObjectNames);

   // To export multiple objects sorted alphabetically
   template <class T>
   void exportObjects(T const &args, std::set<std::string> &exportedObjectNames)
   {
      RooArgSet argSet;
      for (RooAbsArg const *arg : args) {
         argSet.add(*arg);
      }
      argSet.sort();
      for (RooAbsArg *arg : argSet) {
         exportObject(*arg, exportedObjectNames);
      }
   }

   void exportData(RooAbsData const &data);
   RooJSONFactoryWSTool::CombinedData exportCombinedData(RooAbsData const &data);

   void importAllNodes(RooFit::Detail::JSONNode &n);
   void importDistribution(RooFit::HS3::Distribution const &distribution);

   void importVariable(RooFit::HS3::ParameterValue const &p);
   void warnIfFreeWithoutDomain(RooRealVar const &var);
   void importDependants(const RooFit::Detail::JSONNode &n);

   void exportVariable(const RooAbsArg *v, RooFit::HS3::ParameterPoint &point, bool storeConstant);
   void exportVariables(const RooArgSet &allElems, RooFit::HS3::ParameterPoint &point, bool storeConstant);

   void exportAllObjects(RooFit::Detail::JSONNode &n);

   void exportModelConfig(RooStats::ModelConfig const &mc,
                          const std::vector<RooJSONFactoryWSTool::CombinedData> &combined,
                          const std::vector<RooAbsData *> &single);

   void exportSingleModelConfig(RooStats::ModelConfig const &mc, std::string const &analysisName,
                                std::map<std::string, std::string> const *dataComponents);

   friend void RooFit::HS3::setExportedObject(RooJSONFactoryWSTool &tool, RooFit::HS3::HistFactoryDist dist,
                                              std::vector<RooFit::HS3::DistributionRef> channelConstraints);

   // member variables
   std::string _hs3VersionInput;
   RooWorkspace &_workspace;

   // objects to represent intermediate information
   std::unique_ptr<RooFit::HS3::ToolImportState> _import;
   std::unique_ptr<RooFit::HS3::ToolExportState> _export;
   std::vector<RooAbsArg const *> _serversToExport;
   std::vector<RooAbsArg const *> _serversToDelete;

   // The objects that are currently being imported, innermost last, to detect circular dependencies.
   std::vector<std::string> _importStack;
};
#endif
