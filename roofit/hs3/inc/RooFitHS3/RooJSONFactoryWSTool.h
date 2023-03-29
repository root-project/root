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

namespace RooFit {
namespace JSONIO {
namespace Detail {
class Domains;
}
} // namespace JSONIO
} // namespace RooFit
namespace RooStats {
class ModelConfig;
}

class TH1;
class TClass;

class RooJSONFactoryWSTool {
public:
   RooJSONFactoryWSTool(RooWorkspace &ws);

   ~RooJSONFactoryWSTool();

   static std::ostream &log(int level);

   static std::string name(const RooFit::Detail::JSONNode &n);

   static RooFit::Detail::JSONNode &appendNamedChild(RooFit::Detail::JSONNode &node, std::string const &name);
   static RooFit::Detail::JSONNode const *findNamedChild(RooFit::Detail::JSONNode const &node, std::string const &name);

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
      return wsImport(Obj_t(name, name, std::forward<Args_t>(args)...));
   }

   static void error(const char *s);
   inline static void error(const std::string &s) { error(s.c_str()); }

   static void exportHistogram(const TH1 &h, RooFit::Detail::JSONNode &n, const std::vector<std::string> &obsnames,
                               const TH1 *errH = nullptr, bool writeObservables = true, bool writeErrors = true);
   static void writeObservables(const TH1 &h, RooFit::Detail::JSONNode &n, const std::vector<std::string> &varnames);

   static std::unique_ptr<RooDataHist> readBinnedData(const RooFit::Detail::JSONNode &n, const std::string &namecomp);
   static std::unique_ptr<RooDataHist>
   readBinnedData(const RooFit::Detail::JSONNode &n, const std::string &namecomp, RooArgList const &varlist);

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

   void importFunction(const RooFit::Detail::JSONNode &n, bool isPdf);
   RooFit::Detail::JSONNode *exportObject(const RooAbsArg *func);

   static std::unique_ptr<RooFit::Detail::JSONTree> createNewJSONTree();

   static RooFit::Detail::JSONNode &makeVariablesNode(RooFit::Detail::JSONNode &rootNode);

   // error handling helpers
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

   static void
   writeCombinedDataName(RooFit::Detail::JSONNode &rootnode, std::string const &pdfName, std::string const &dataName);

   static void writeChannelNames(RooFit::Detail::JSONNode &rootnode, std::string const &simPdfName,
                                 std::vector<std::string> const &channelNames);

   static void exportDataHist(RooDataHist const &dataHist, RooFit::Detail::JSONNode &node);

private:
   struct Config {
      static bool stripObservables;
   };

   template <class T>
   T *requestImpl(const std::string &objname);

   void exportData(RooAbsData const &data);

   void importAllNodes(const RooFit::Detail::JSONNode &n);

   void importVariables(const RooFit::Detail::JSONNode &n);
   void importVariable(const RooFit::Detail::JSONNode &n);
   void importDependants(const RooFit::Detail::JSONNode &n);

   void exportVariable(const RooAbsArg *v, RooFit::Detail::JSONNode &n);
   void exportVariables(const RooArgSet &allElems, RooFit::Detail::JSONNode &n);

   void exportAllObjects(RooFit::Detail::JSONNode &n);
   void exportDependants(const RooAbsArg *source);

   void exportModelConfig(RooFit::Detail::JSONNode &n, RooStats::ModelConfig const &mc);

   // member variables
   const RooFit::Detail::JSONNode *_rootnodeInput = nullptr;
   const RooFit::Detail::JSONNode *_attributesNode = nullptr;
   RooFit::Detail::JSONNode *_rootnodeOutput = nullptr;
   RooWorkspace &_workspace;

   // objects to represent intermediate information
   std::unique_ptr<RooFit::JSONIO::Detail::Domains> _domains;
};
#endif
