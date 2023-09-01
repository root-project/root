// @(#)root/roostats:$Id$
// Author: George Lewis, Kyle Cranmer
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

  Data() {}
  /// constructor from name, file and path. Name of the histogram should not include the path
  Data( std::string HistoName, std::string InputFile, std::string HistoPath="" );

  std::string const& GetName() const { return fName; }
  void SetName(const std::string& name) { fName=name; }

  void SetInputFile(const std::string& InputFile) { fInputFile = InputFile; }
  std::string const& GetInputFile() const { return fInputFile; }

  void SetHistoName(const std::string& HistoName) { fHistoName = HistoName; }
  std::string const& GetHistoName() const { return fHistoName; }

  void SetHistoPath(const std::string& HistoPath) { fHistoPath = HistoPath; }
  std::string const& GetHistoPath() const { return fHistoPath; }

  void Print(std::ostream& = std::cout);
  void PrintXML( std::ostream& ) const;
  void writeToFile( std::string FileName, std::string DirName );

  TH1* GetHisto();
  const TH1* GetHisto() const;
  void SetHisto(TH1* Hist) { fhData = Hist; fHistoName=Hist->GetName(); }

protected:

  std::string fName;

  std::string fInputFile;
  std::string fHistoName;
  std::string fHistoPath;


  // The Data Histogram
  HistRef fhData;


};

}
}


#endif
