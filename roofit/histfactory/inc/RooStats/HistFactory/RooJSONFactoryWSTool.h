#include "RooWorkspace.h"
#include <string>

class RooJSONFactoryWSTool : public TNamed, RooPrintable {
  RooWorkspace* _workspace;
  std::vector<std::string> _strcache;
 public:
  RooJSONFactoryWSTool(RooWorkspace& ws);
  
// interfaces for JSON and YAML reading/writing
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
protected:
  template<class T> void importFunctions(const T& n);
  template<class T> void importPdfs(const T& n);
  template<class T> void importVariables(const T& n);
  template<class T> void importDependants(const T& n);
  
  template<class T> void exportFunctions(T& n);
  template<class T> void exportPdfs(T& n);
  template<class T> void exportVariables(T& n);
  template<class T> void exportAll(T& n);  
  
};
