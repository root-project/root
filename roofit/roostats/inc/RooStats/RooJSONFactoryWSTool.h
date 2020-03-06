#ifndef ROOJSONFACTORYWSTOOL_H
#define ROOJSONFACTORYWSTOOL_H
#include "RooWorkspace.h"
#include "TH1.h"
#include "RooStats/JSONInterface.h"
#include <string>

class RooJSONFactoryWSTool : public TNamed, RooPrintable {
 public:
  class Importer {
  public:
    virtual bool importPdf(RooJSONFactoryWSTool*, const TJSONNode&) const {
      return false;
    }
    virtual bool importFunction(RooJSONFactoryWSTool*, const TJSONNode&) const {
      return false;
    }
  };
  class Exporter {
  public:
    virtual bool autoExportDependants() const { return true; }
    virtual bool exportObject(RooJSONFactoryWSTool*, const RooAbsArg*, TJSONNode&) const {
      return false;
    }
  };   

 protected:
  RooWorkspace* _workspace;
  static std::map<std::string,const Importer*> _importers;
  static std::map<const TClass*,const Exporter*> _exporters;    
  void prepare();
 public:
  static std::string name(const TJSONNode& n);  
  
  RooJSONFactoryWSTool(RooWorkspace& ws);
  RooWorkspace* workspace() { return this->_workspace; }

  static bool registerImporter(const std::string& key, const RooJSONFactoryWSTool::Importer* f){
    if(RooJSONFactoryWSTool::_importers.find(key) != RooJSONFactoryWSTool::_importers.end()) return false;
    RooJSONFactoryWSTool::_importers[key] = f;
    return true;
  }
  static bool registerExporter(const TClass* key, const RooJSONFactoryWSTool::Exporter* f){
    if(RooJSONFactoryWSTool::_exporters.find(key) != RooJSONFactoryWSTool::_exporters.end()) return false;  
    RooJSONFactoryWSTool::_exporters[key] = f;
    return true;
  }

  // error handling helpers
  static void error(const char* s){
    throw std::runtime_error(s);
  }
  static void error(const std::string& s){
    throw std::runtime_error(s);
  }
  template<class T> static std::string concat(const T* items, const std::string& sep=",") {
    // Returns a string being the concatenation of strings in input list <items>
    // (names of objects obtained using GetName()) separated by string <sep>.
    bool first = true;
    std::string text;
    
    // iterate over strings in list
    for(auto it:*items){
      if (!first) {
        // insert separator string
        text += sep;
      } else {
        first = false;
      }
      if(!it) text+="NULL";
      else text+=it->GetName();
    }
    return text;
  }
  template<class T> static std::vector<std::string> names(const T* items) {
    // Returns a string being the concatenation of strings in input list <items>
    // (names of objects obtained using GetName()) separated by string <sep>.
    std::vector<std::string> names;
    // iterate over strings in list
    for(auto it:*items){
      if(!it) names.push_back("NULL");
      else names.push_back(it->GetName());
    }
    return names;
  }
  
  static void exportHistogram(const TH1& h, TJSONNode& n, const std::vector<std::string>& obsnames);
  static std::vector<std::vector<int> > generateBinIndices(RooArgList& vars);
  
  Bool_t importJSON(const char* filename);
  Bool_t importYML(const char* filename);
  Bool_t importJSON(std::istream& os);
  Bool_t importYML(std::istream& os);  
  Bool_t exportJSON(const char* fileName) ;
  Bool_t exportYML(const char* fileName) ;
  Bool_t exportJSON(std::ostream& os);
  Bool_t exportYML(std::ostream& os);
  
  static void loadFactoryExpressions(const std::string& fname);
  static void clearFactoryExpressions();
  static void printFactoryExpressions();
  static void loadExportKeys(const std::string& fname);
  static void clearExportKeys();
  static void printExportKeys();  

  void importFunctions(const TJSONNode& n);
  void importPdfs(const TJSONNode& n);
  void importVariables(const TJSONNode& n);
  void importDependants(const TJSONNode& n);

  bool find(const TJSONNode& n, const std::string& elem);
  void append(TJSONNode& n, const std::string& elem);
  
  void exportAttributes(const RooAbsArg* arg, TJSONNode& n);
  void exportVariable(const RooAbsReal*v, TJSONNode& n);
  void exportVariables(const RooArgSet& allElems, TJSONNode& n);
  void exportObject(const RooAbsArg* func, TJSONNode& n);
  void exportFunctions(const RooArgSet& allElems, TJSONNode& n);  

  void exportAll(TJSONNode& n);  
  void exportDependants(const RooAbsArg* source, TJSONNode& n);
};
#endif
