#ifndef ROOJSONFACTORYWSTOOL_H
#define ROOJSONFACTORYWSTOOL_H
#include "RooWorkspace.h"
#include "TH1.h"
#include <string>

class RooJSONFactoryWSTool : public TNamed, RooPrintable {
  RooWorkspace* _workspace;
  static std::vector<std::string> _strcache;
  void prepare();
 public:
  RooJSONFactoryWSTool(RooWorkspace& ws);
  static const char* incache(const std::string& str);
  static void clearcache();

  template<class T> static void exportHistogram(const TH1& h, T& n, const std::vector<std::string>& obsnames);
  
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

  template<class T> void exportAll(T& n);  
  template<class T> void exportDependants(RooAbsArg* source, T& n);
};
#endif
