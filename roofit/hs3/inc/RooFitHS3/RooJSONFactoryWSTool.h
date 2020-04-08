#ifndef RooFitHS3_RooJSONFactoryWSTool_h
#define RooFitHS3_RooJSONFactoryWSTool_h

#include <RooArgSet.h>

#include <map>
#include <string>

class RooAbsArg;
class RooAbsReal;
class RooDataHist;
class RooDataSet;
class RooRealVar;
class RooRealVar;
class RooWorkspace;

class TH1;

namespace RooFit {
namespace Detail {
class JSONNode;
}
} // namespace RooFit

class RooJSONFactoryWSTool {
public:
   class Importer {
   public:
      virtual bool importPdf(RooJSONFactoryWSTool *, const RooFit::Detail::JSONNode &) const { return false; }
      virtual bool importFunction(RooJSONFactoryWSTool *, const RooFit::Detail::JSONNode &) const { return false; }
      virtual ~Importer(){};
   };
   class Exporter {
   public:
      virtual bool autoExportDependants() const { return true; }
      virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *, RooFit::Detail::JSONNode &) const
      {
         return false;
      }
      virtual ~Exporter(){};
   };

   typedef std::map<const std::string, const Importer *> ImportMap;
   typedef std::map<const TClass *, const Exporter *> ExportMap;

   struct Var {
      int nbins;
      double min;
      double max;
      std::vector<double> bounds;

      Var(int n) : nbins(n), min(0), max(n) {}
      Var(const RooFit::Detail::JSONNode &val);
   };

protected:
   struct Scope {
      RooArgSet observables;
      std::map<std::string, RooAbsArg *> objects;
   };
   mutable Scope _scope;

   RooWorkspace *_workspace;
   static ImportMap _importers;
   static ExportMap _exporters;
   void prepare();
   std::map<std::string, RooAbsData *> loadData(const RooFit::Detail::JSONNode &n);
   RooDataSet *unbinned(RooDataHist *hist);
   RooRealVar *getWeightVar(const char *name);
   RooRealVar *createObservable(const std::string &name, const RooJSONFactoryWSTool::Var &var);

public:
   static std::string name(const RooFit::Detail::JSONNode &n);

   RooJSONFactoryWSTool(RooWorkspace &ws) : _workspace{&ws} {}
   RooWorkspace *workspace() { return this->_workspace; }

   static bool registerImporter(const std::string &key, const RooJSONFactoryWSTool::Importer *f);
   static bool registerExporter(const TClass *key, const RooJSONFactoryWSTool::Exporter *f);
   static void printImporters();
   static void printExporters();

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

   static std::string genPrefix(const RooFit::Detail::JSONNode &p, bool trailing_underscore);
   static void exportHistogram(const TH1 &h, RooFit::Detail::JSONNode &n, const std::vector<std::string> &obsnames,
                               const TH1 *errH = 0, bool writeObservables = true, bool writeErrors = true);
   void exportData(RooAbsData *data, RooFit::Detail::JSONNode &n);
   static void writeObservables(const TH1 &h, RooFit::Detail::JSONNode &n, const std::vector<std::string> &varnames);
   static std::vector<std::vector<int>> generateBinIndices(const RooArgList &vars);
   RooDataHist *
   readBinnedData(const RooFit::Detail::JSONNode &n, const std::string &namecomp, const RooArgList &observables);
   static std::map<std::string, RooJSONFactoryWSTool::Var>
   readObservables(const RooFit::Detail::JSONNode &n, const std::string &obsnamecomp);
   RooArgSet getObservables(const RooFit::Detail::JSONNode &n, const std::string &obsnamecomp);
   void setScopeObservables(const RooArgList &args);
   RooAbsArg *getScopeObject(const std::string &name);
   void setScopeObject(const std::string &key, RooAbsArg *obj);
   void clearScope();

   bool importJSON(std::string const& filename);
   bool importYML(std::string const& filename);
   bool importJSON(std::istream &os);
   bool importYML(std::istream &os);
   bool exportJSON(std::string const& fileName);
   bool exportYML(std::string const& fileName);
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

   void importFunctions(const RooFit::Detail::JSONNode &n);
   void importPdfs(const RooFit::Detail::JSONNode &n);
   void importVariables(const RooFit::Detail::JSONNode &n);
   void importDependants(const RooFit::Detail::JSONNode &n);

   bool find(const RooFit::Detail::JSONNode &n, const std::string &elem);
   void append(RooFit::Detail::JSONNode &n, const std::string &elem);

   void exportAttributes(const RooAbsArg *arg, RooFit::Detail::JSONNode &n);
   void exportVariable(const RooAbsReal *v, RooFit::Detail::JSONNode &n);
   void exportVariables(const RooArgSet &allElems, RooFit::Detail::JSONNode &n);
   void exportObject(const RooAbsArg *func, RooFit::Detail::JSONNode &n);
   void exportFunctions(const RooArgSet &allElems, RooFit::Detail::JSONNode &n);

   void exportAll(RooFit::Detail::JSONNode &n);
   void exportDependants(const RooAbsArg *source, RooFit::Detail::JSONNode &n);
};
#endif
