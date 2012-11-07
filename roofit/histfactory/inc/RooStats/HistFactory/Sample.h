// @(#)root/roostats:$Id$
// Author: George Lewis, Kyle Cranmer
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef HISTFACTORY_SAMPLE_H
#define HISTFACTORY_SAMPLE_H

#include <string>
#include <fstream>
#include <vector>
#include <iostream>

#include "TRef.h"
#include "RooStats/HistFactory/Systematics.h"

namespace RooStats{
namespace HistFactory {

class Sample {


public:

  Sample();
  Sample(std::string Name);
  Sample(std::string Name, std::string HistoName, std::string InputFile, std::string HistoPath="");


  void Print(std::ostream& = std::cout);  
  void PrintXML( std::ofstream& xml );
  void writeToFile( std::string FileName, std::string DirName );

  TH1* GetHisto();
  void SetHisto( TH1* histo ) { fhNominal = histo; fHistoName=histo->GetName(); }
  void SetValue( Double_t Val );

  // Some helper functions

  void ActivateStatError();
  void ActivateStatError( std::string HistoName, std::string InputFile, std::string HistoPath="" );

  void AddOverallSys( std::string Name, Double_t Low, Double_t High );
  void AddOverallSys( const OverallSys& Sys );

  void AddNormFactor( std::string Name, Double_t Val, Double_t Low, Double_t High, bool Const=false );
  void AddNormFactor( const NormFactor& Factor );

  void AddHistoSys(    std::string Name, std::string HistoNameLow,  std::string HistoFileLow,  std::string HistoPathLow,
		                         std::string HistoNameHigh, std::string HistoFileHigh, std::string HistoPathHigh );
  void AddHistoSys( const HistoSys& Sys );

  void AddHistoFactor( std::string Name, std::string HistoNameLow,  std::string HistoFileLow,  std::string HistoPathLow,  
		       std::string HistoNameHigh, std::string HistoFileHigh, std::string HistoPathHigh );
  void AddHistoFactor( const HistoFactor& Factor );

  void AddShapeFactor( std::string Name );
  void AddShapeFactor( const ShapeFactor& Factor );

  void AddShapeSys(    std::string Name, Constraint::Type ConstraintType, std::string HistoName, std::string HistoFile, std::string HistoPath="" );
  void AddShapeSys( const ShapeSys& Sys );

  void SetNormalizeByTheory( bool norm ) { fNormalizeByTheory = norm; }
  bool GetNormalizeByTheory() { return fNormalizeByTheory; }



  std::string GetName() { return fName; }
  void SetName(const std::string& Name) { fName = Name; }
  

  std::string GetInputFile() { return fInputFile; }
  void SetInputFile(const std::string& InputFile) { fInputFile = InputFile; }

  std::string GetHistoName() { return fHistoName; }
  void SetHistoName(const std::string& HistoName) { fHistoName = HistoName; }

  std::string GetHistoPath() { return fHistoPath; }
  void SetHistoPath(const std::string& HistoPath) { fHistoPath = HistoPath; }

  std::string GetChannelName() { return fChannelName; }
  void SetChannelName(const std::string& ChannelName) { fChannelName = ChannelName; }



  std::vector< RooStats::HistFactory::OverallSys >& GetOverallSysList() { return fOverallSysList; }
  std::vector< RooStats::HistFactory::NormFactor >& GetNormFactorList() { return fNormFactorList; }

  std::vector< RooStats::HistFactory::HistoSys >&    GetHistoSysList() {    return fHistoSysList; }
  std::vector< RooStats::HistFactory::HistoFactor >& GetHistoFactorList() { return fHistoFactorList; }

  std::vector< RooStats::HistFactory::ShapeSys >&    GetShapeSysList() {    return fShapeSysList; }
  std::vector< RooStats::HistFactory::ShapeFactor >& GetShapeFactorList() { return fShapeFactorList; }

  RooStats::HistFactory::StatError& GetStatError() { return fStatError; }
  void SetStatError( RooStats::HistFactory::StatError Error ) { fStatError = Error; }


protected:

  std::string fName;
  std::string fInputFile;
  std::string fHistoName;
  std::string fHistoPath;

  // The Name of the parent channel
  std::string fChannelName;

  //
  // Systematics
  //

  std::vector< RooStats::HistFactory::OverallSys >  fOverallSysList;
  std::vector< RooStats::HistFactory::NormFactor >  fNormFactorList;

  std::vector< RooStats::HistFactory::HistoSys >    fHistoSysList;
  std::vector< RooStats::HistFactory::HistoFactor > fHistoFactorList;

  std::vector< RooStats::HistFactory::ShapeSys >    fShapeSysList;
  std::vector< RooStats::HistFactory::ShapeFactor > fShapeFactorList;


  // Properties
  RooStats::HistFactory::StatError fStatError;

  bool fNormalizeByTheory;
  bool fStatErrorActivate;


  // The Nominal Shape
  TRef fhNominal;

};


} // namespace HistFactory
} // namespace RooStats

#endif
