
#ifndef HISTFACTORY_DATA_H
#define HISTFACTORY_DATA_H

#include <string>
#include <fstream>
#include <iostream>

//#include "RooStats/HistFactory/HistCollector.h"
#include "RooStats/HistFactory/Sample.h"

namespace RooStats{
namespace HistFactory {

class Data {
  
public:
  //friend class Channel;

  Data() {;}
  Data( std::string HistoName, std::string InputFile, std::string HistoPath="" );
  
  void SetInputFile(const std::string& InputFile) { fInputFile = InputFile; }
  std::string GetInputFile() { return fInputFile; }

  void SetHistoName(const std::string& HistoName) { fHistoName = HistoName; }
  std::string GetHistoName() { return fHistoName; }

  void SetHistoPath(const std::string& HistoPath) { fHistoPath = HistoPath; }
  std::string GetHistoPath() { return fHistoPath; }


  void Print(std::ostream& = std::cout);
  void writeToFile( std::string FileName, std::string DirName );

  TH1* GetHisto();
  void SetHisto(TH1* Hist) { fhData = Hist; fHistoName=Hist->GetName(); }
  
protected:
  
  std::string fInputFile;
  std::string fHistoName;
  std::string fHistoPath;


  // The Data Histogram
  TRef fhData;


};

}
}


#endif
