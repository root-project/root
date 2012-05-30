
#ifndef HISTFACTORY_MEASUREMENT_H
#define HISTFACTORY_MEASUREMENT_H

#include <string>
#include <map>
#include <fstream>
#include <iostream>

#include "TObject.h"
#include "TFile.h"


#include "PreprocessFunction.h"
#include "RooStats/HistFactory/Channel.h"


namespace RooStats{
namespace HistFactory {

class Measurement : public TNamed {


public:

  Measurement();
  //  Measurement( const Measurement& other ); // Copy
  Measurement(const char* Name, const char* Title="");

  //  std::string Name;


  void SetOutputFilePrefix( const std::string& prefix ) { fOutputFilePrefix = prefix; }
  std::string GetOutputFilePrefix() { return fOutputFilePrefix; }

  void SetPOI( const std::string& POI ) { fPOI = POI; }
  std::string GetPOI() { return fPOI; }


  void AddConstantParam( const std::string& param ) { fConstantParams.push_back( param ); }
  void ClearConstantParams() { fConstantParams.clear(); }
  std::vector< std::string >& GetConstantParams() { return fConstantParams; }

  void AddFunctionObject( const RooStats::HistFactory::PreprocessFunction function) { fFunctionObjects.push_back( function ); }
  void SetFunctionObjects( std::vector< RooStats::HistFactory::PreprocessFunction > objects ) { fFunctionObjects = objects; }
  std::vector< RooStats::HistFactory::PreprocessFunction >& GetFunctionObjects() { return fFunctionObjects; }

  void AddPreprocessFunction( const std::string& function ) { fPreprocessFunctions.push_back( function ); }
  void SetPreprocessFunctions( std::vector< std::string > functions ) { fPreprocessFunctions = functions;  }
  std::vector< std::string >& GetPreprocessFunctions()  { return fPreprocessFunctions; }
  void ClearPreprocessFunctions() { fPreprocessFunctions.clear(); }

  void SetLumi(double Lumi ) { fLumi = Lumi; }
  void SetLumiRelErr( double RelErr ) { fLumiRelErr = RelErr; }
  double GetLumi() { return fLumi; }
  double GetLumiRelErr() { return fLumiRelErr; }
  
  void SetBinLow( int BinLow ) { fBinLow = BinLow; }
  void SetBinHigh ( int BinHigh ) { fBinHigh = BinHigh; }
  int GetBinLow() { return fBinLow; }
  int GetBinHigh() { return fBinHigh; } 

  void SetExportOnly( bool ExportOnly ) { fExportOnly = ExportOnly; }
  bool GetExportOnly() { return fExportOnly; }


  void PrintTree( std::ostream& = std::cout ); // Print to a stream
  void PrintXML( std::string Directory="", std::string NewOutputPrefix="" );

  std::vector< RooStats::HistFactory::Channel >& GetChannels() { return fChannels; }
  RooStats::HistFactory::Channel& GetChannel( std::string );
  void AddChannel( RooStats::HistFactory::Channel chan ) { fChannels.push_back( chan ); }

  bool HasChannel( std::string );
  void writeToFile( TFile* file );

  void CollectHistograms();


  std::map< std::string, double >& GetGammaSyst() { return fGammaSyst; }
  std::map< std::string, double >& GetUniformSyst() { return fUniformSyst; }
  std::map< std::string, double >& GetLogNormSyst() { return fLogNormSyst; }
  std::map< std::string, double >& GetNoSyst() { return fNoSyst; }



private:

  std::string fOutputFilePrefix;
  std::string fPOI;

  std::vector< RooStats::HistFactory::Channel > fChannels;
  std::vector< std::string > fConstantParams;
  std::vector< RooStats::HistFactory::PreprocessFunction > fFunctionObjects;
  std::vector< std::string > fPreprocessFunctions;
  //std::vector< std::string > constraintTerms;

  std::map< std::string, double > fGammaSyst;
  std::map< std::string, double > fUniformSyst;
  std::map< std::string, double > fLogNormSyst;
  std::map< std::string, double > fNoSyst;

  double fLumi;
  double fLumiRelErr;

  int fBinLow;
  int fBinHigh;

  bool fExportOnly;
  // bool fSaveExtra;

  std::string fInterpolationScheme;
  
  std::string GetDirPath( TDirectory* dir );

  ClassDef(RooStats::HistFactory::Measurement, 1);

};
 
} // namespace HistFactory
} // namespace RooStats

#endif
