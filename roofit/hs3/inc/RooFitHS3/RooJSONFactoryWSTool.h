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
struct Version;
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
class RooAbsCategory;
class RooSimultaneous;

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
      /// HS3 requires the components of a combined distribution to have distinct variates, e.g. "mass_SR" and
      /// "mass_CR" instead of one "mass". If enabled, the observables of every combined data set are renamed on
      /// export to "obs_<key>" (the labels of the key joined by "_"), or "obs_<key>_<observable>" if a component
      /// has several observables; the components of the matching RooSimultaneous are written with the renamed
      /// observables (functions depending on them get the suffix "_<key>"). This allows exporting workspaces whose
      /// components share an observable. Off by default: without it, such workspaces can't be exported.
      bool renameCombinedObservables = false;
      /// HS3 requires the terms of a likelihood and the components of a combined distribution to have distinct
      /// variates, e.g. "mass_SR" and "mass_CR" for the observables of two channels instead of one "mass". By
      /// default, the export fails otherwise. If true, the document is written anyway, with a warning for every
      /// shared variate. It is not valid HS3 and can't be read back before the variates are renamed, e.g. by hand in
      /// the JSON.
      bool allowExportSharedVariates = false;
      // Write array distributions (HS3 `array_dist`). If false, no `array_dist` is written: HistFactory constraints
      // are named individually, and a RooProdPdf tagged as array distribution is written as `product_dist`. The
      // RooProdPdf that RooFit created itself when it read an `array_dist` is only written if another object uses
      // it; otherwise its components are written on their own.
      bool exportArrayDist = true;
      // How the constraint terms of HistFactory models enter the likelihood: listed in "aux_distributions", with
      // the observed values as constant parameters (false), or paired with a data set holding the observed values
      // of their observables (true). An array of constraints of a model that is a RooSimultaneous is then a component
      // of the combined distribution, and its data set a component of the combined data set, so that the combined
      // distribution is the sampling distribution of all measurements. Other constraints are terms of the likelihood.
      bool exportConstraintData = false;
      /// EXPERIMENTAL: how auxiliary measurements that are components of a combined distribution (a constraint term
      /// paired with the data set of its observed values, as RooFit writes them with `exportConstraintData`) are
      /// read. If false, they become constraint terms of the model as in HistFactory, with global observables set
      /// to the observed values; RooFit recognizes the components that it wrote like this by a mark in the "misc"
      /// section of the document. If true, or without that mark, they stay components of the RooSimultaneous, with
      /// their data set of one entry in the combined data set, so that the RooSimultaneous is the sampling
      /// distribution of all measurements. This option and the form of the resulting workspace may still change.
      bool importConstraintsAsChannels = false;
   };

   static Config &config();

   static RooWorkspace sanitizeWS(const RooWorkspace &ws);
   static RooWorkspace cleanWS(const RooWorkspace &ws, bool onlyModelConfig = false);

   struct CombinedData {
      std::string name;
      std::vector<std::string> columns;
   };

   RooJSONFactoryWSTool(RooWorkspace &ws);

   ~RooJSONFactoryWSTool();

   static std::string name(const RooFit::Detail::JSONNode &n);
   static bool isValidName(const std::string &str);
   static bool testValidName(const std::string &name, bool forceError);
   static std::string sanitizeName(const std::string str);

   static RooFit::Detail::JSONNode &appendNamedChild(RooFit::Detail::JSONNode &node, std::string const &name);
   static RooFit::Detail::JSONNode const *findNamedChild(RooFit::Detail::JSONNode const &node, std::string const &name);

   static void fillSeq(RooFit::Detail::JSONNode &node, RooAbsCollection const &coll, size_t nMax = -1);

   /// The object `objname` from the workspace, imported from the document first if needed. Throws a
   /// DependencyMissingError if neither has it.
   template <class T>
   T *request(const std::string &objname, const std::string &requestAuthor)
   {
      if (T *out = tryRequest<T>(objname)) {
         return out;
      }
      throw DependencyMissingError(requestAuthor, objname, T::Class()->GetName());
   }

   /// Like request(), but returns nullptr if neither the workspace nor the document has the object.
   template <class T>
   T *tryRequest(const std::string &objname)
   {
      return requestImpl<T>(objname);
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
   bool importJSON(std::istream &is);
   bool exportJSON(std::string const &fileName);
   bool exportJSON(std::ostream &os);

   std::string exportJSONtoString();
   bool importJSONfromString(const std::string &s);
   void importJSONElement(const std::string &name, const std::string &jsonString);
   void importVariableElement(const RooFit::Detail::JSONNode &elementNode);

   void importFunction(const RooFit::Detail::JSONNode &p, bool importAllDependants);
   void importFunction(const std::string &jsonString, bool importAllDependants);

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

   static void
   exportHisto(RooArgSet const &vars, std::size_t n, double const *contents, RooFit::Detail::JSONNode &output);

   static void exportArray(std::size_t n, double const *contents, RooFit::Detail::JSONNode &output);

   void queueExport(RooAbsArg const &arg) { _serversToExport.push_back(&arg); }
   void queueExportTemporary(RooAbsArg *arg)
   {
      _serversToExport.push_back(arg);
      _serversToDelete.push_back(arg);
   }

   std::string exportTransformed(const RooAbsReal *original, const std::string &suffix, const std::string &formula);
   std::string exportAsArrayComponent(RooAbsPdf const &pdf, std::string const &arrayName);
   /// The name of the top-level object whose export is in progress (the root of the current export recursion).
   std::string const &exportRootName() const;
   void recordConstraintTerms(std::string const &pdfName, std::vector<std::string> const &terms);
   /// Record that `observable` is the observable of the constraint term `pdf` (the side that is not the constrained
   /// parameter). A Gaussian whose mean it is gets written with x and mean swapped, so that the observable is x.
   void setConstraintObservable(RooAbsPdf const &pdf, std::string const &observable);
   std::string const *constraintObservable(std::string const &pdfName) const;

   void setAttribute(const std::string &obj, const std::string &attrib);
   bool hasAttribute(const std::string &obj, const std::string &attrib);
   std::string getStringAttribute(const std::string &obj, const std::string &attrib);
   void setStringAttribute(const std::string &obj, const std::string &attrib, const std::string &value);

   /// The HS3 version of the document that is currently being imported. Outside of an import it is 0.0.0, like a
   /// version that can't be parsed.
   RooFit::HS3::Version hs3Version() const;

   /// Whether the component `component` of the combined distribution `combinedDist` of the document that is
   /// currently being imported is read as a constraint term instead (see Config::importConstraintsAsChannels).
   bool isAuxiliaryChannelReadAsConstraint(std::string const &combinedDist, std::string const &component) const;

   /// Whether the constraint term `pdf` contributes to the likelihoods of the document that is currently being
   /// imported as a component of a combined distribution with data, by itself or as a component of an array
   /// distribution. It is then a measurement of its own, and must not be multiplied into the distributions that
   /// name it as a constraint as well.
   bool isConstraintReadAsChannel(RooAbsPdf const &pdf) const;

private:
   template <class T>
   T *requestImpl(const std::string &objname);
   void exportObject(RooAbsArg const &func);

   // To export multiple objects sorted alphabetically
   template <class T>
   void exportObjects(T const &args)
   {
      RooArgSet argSet;
      for (RooAbsArg const *arg : args) {
         argSet.add(*arg);
      }
      argSet.sort();
      for (RooAbsArg *arg : argSet) {
         exportObject(*arg);
      }
   }
   void exportArrayDists();
   bool exportConstraintData(std::string const &term, RooArgSet const *globalObservables, std::string const &dataName);

   void exportSimultaneous(RooSimultaneous const &simPdf);
   void recordCategory(RooAbsCategory const &cat);
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
                          const std::vector<RooJSONFactoryWSTool::CombinedData> &combDataSets,
                          const std::vector<RooAbsData *> &single);

   void exportSingleModelConfig(RooStats::ModelConfig const &mc, std::string const &analysisName,
                                std::string const *dataName);

   friend void RooFit::HS3::setExportedObject(RooJSONFactoryWSTool &tool, RooFit::HS3::HistFactoryDist dist,
                                              std::vector<RooFit::HS3::DistributionRef> channelConstraints);

   // member variables
   /// The components of combined distributions that are read as constraint terms, by combined distribution.
   std::map<std::string, std::set<std::string>> _auxiliaryChannelsAsConstraints;
   /// The components of combined distributions that have data, and are read as such.
   std::set<std::string> _componentsWithData;
   RooWorkspace &_workspace;

   // objects to represent intermediate information
   std::unique_ptr<RooFit::HS3::ToolImportState> _import;
   std::unique_ptr<RooFit::HS3::ToolExportState> _export;
   std::vector<RooAbsArg const *> _serversToExport;
   std::vector<RooAbsArg const *> _serversToDelete;
   // Renaming of the observables of combined data sets, see Config::renameCombinedObservables: for each data set and
   // each selection label, the new name of each observable; for each RooSimultaneous and selection label, the
   // component with renamed observables to export instead of the original one. The objects are owned here.
   std::map<std::string, std::map<std::string, std::map<std::string, std::string>>> _combinedObservableRenames;
   std::map<std::string, std::map<std::string, RooAbsPdf const *>> _renamedComponents;
   std::vector<std::unique_ptr<RooAbsArg>> _renamedObjects;
   void prepareCombinedObservableRenaming();

   // The objects that are currently being imported, innermost last, to detect circular dependencies.
   std::vector<std::string> _importStack;
};
#endif
