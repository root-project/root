// @(#)root/roostats:$Id$
// Author: George Lewis, Kyle Cranmer
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef HISTFACTORY_SYSTEMATICS_H
#define HISTFACTORY_SYSTEMATICS_H


#include <string>
#include <fstream>
#include <iostream>

#include "TH1.h"
#include "TRef.h"

//#include "RooStats/HistFactory/HistCollector.h"

namespace RooStats{
namespace HistFactory {


  namespace Constraint {
    enum Type{ Gaussian, Poisson };            
    std::string Name( Type type ); 
    Type GetType( std::string Name );
  }



  class OverallSys {

  public:
    //friend class Channel;

    void SetName( const std::string& Name ) { fName = Name; }
    std::string GetName() { return fName; }

    void SetLow( double Low )   { fLow  = Low; }
    void SetHigh( double High ) { fHigh = High; }
    double GetLow() { return fLow; }
    double GetHigh() { return fHigh; }

    void Print(std::ostream& = std::cout);  

  protected:
    std::string fName;
    double fLow;
    double fHigh;

  };


  class NormFactor {

  public:
    //friend class Channel;

    NormFactor();

    void SetName( const std::string& Name ) { fName = Name; }
    std::string GetName() { return fName; }

    void SetVal( double Val ) { fVal = Val; }
    double GetVal() { return fVal; }

    void SetConst( bool Const=true ) { fConst = Const; }
    bool GetConst() { return fConst; }

    void SetLow( double Low )   { fLow  = Low; }
    void SetHigh( double High ) { fHigh = High; }
    double GetLow() { return fLow; }
    double GetHigh() { return fHigh; }

    void Print(std::ostream& = std::cout);      

  protected:

    std::string fName;
    double fVal;
    double fLow;
    double fHigh;
    bool fConst;

  };


  class HistoSys {

  public:
    //friend class Channel;


    HistoSys() : fhLow(NULL), fhHigh(NULL) {;}
    HistoSys(const std::string& Name) : fName(Name), fhLow(NULL), fhHigh(NULL) {;}

    void Print(std::ostream& = std::cout);  
    void writeToFile( std::string FileName, std::string DirName );

    void SetHistoLow( TH1* Low ) { fhLow = Low; }
    void SetHistoHigh( TH1* High ) { fhHigh = High; }
    
    TH1* GetHistoLow();
    TH1* GetHistoHigh();

    
    void SetName( const std::string& Name ) { fName = Name; }
    std::string GetName() { return fName; }

    void SetInputFileLow( const std::string& InputFileLow ) { fInputFileLow = InputFileLow; }
    void SetInputFileHigh( const std::string& InputFileHigh ) { fInputFileHigh = InputFileHigh; }
    
    std::string GetInputFileLow() { return fInputFileLow; }
    std::string GetInputFileHigh() { return fInputFileHigh; }

    void SetHistoNameLow( const std::string& HistoNameLow ) { fHistoNameLow = HistoNameLow; }
    void SetHistoNameHigh( const std::string& HistoNameHigh ) { fHistoNameHigh = HistoNameHigh; }
    
    std::string GetHistoNameLow() { return fHistoNameLow; }
    std::string GetHistoNameHigh() { return fHistoNameHigh; }

    void SetHistoPathLow( const std::string& HistoPathLow ) { fHistoPathLow = HistoPathLow; }
    void SetHistoPathHigh( const std::string& HistoPathHigh ) { fHistoPathHigh = HistoPathHigh; }
    
    std::string GetHistoPathLow() { return fHistoPathLow; }
    std::string GetHistoPathHigh() { return fHistoPathHigh; }


  protected:

    std::string fName;

    std::string fInputFileLow;
    std::string fHistoNameLow;
    std::string fHistoPathLow;

    std::string fInputFileHigh;
    std::string fHistoNameHigh;
    std::string fHistoPathHigh;

    // The Low and High Histograms
    TRef fhLow;
    TRef fhHigh;

  };


  class HistoFactor {

  public:
    //friend class Channel;  



    void SetName( const std::string& Name ) { fName = Name; }
    std::string GetName() { return fName; }

    void SetInputFileLow( const std::string& InputFileLow ) { fInputFileLow = InputFileLow; }
    void SetInputFileHigh( const std::string& InputFileHigh ) { fInputFileHigh = InputFileHigh; }
    
    std::string GetInputFileLow() { return fInputFileLow; }
    std::string GetInputFileHigh() { return fInputFileHigh; }

    void SetHistoNameLow( const std::string& HistoNameLow ) { fHistoNameLow = HistoNameLow; }
    void SetHistoNameHigh( const std::string& HistoNameHigh ) { fHistoNameHigh = HistoNameHigh; }
    
    std::string GetHistoNameLow() { return fHistoNameLow; }
    std::string GetHistoNameHigh() { return fHistoNameHigh; }

    void SetHistoPathLow( const std::string& HistoPathLow ) { fHistoPathLow = HistoPathLow; }
    void SetHistoPathHigh( const std::string& HistoPathHigh ) { fHistoPathHigh = HistoPathHigh; }
    
    std::string GetHistoPathLow() { return fHistoPathLow; }
    std::string GetHistoPathHigh() { return fHistoPathHigh; }

    void Print(std::ostream& = std::cout);  
    void writeToFile( std::string FileName, std::string DirName );


    TH1* GetHistoLow();
    TH1* GetHistoHigh();
    void SetHistoLow( TH1* Low ) { fhLow = Low; }
    void SetHistoHigh( TH1* High ) { fhHigh = High; }



  protected:

    std::string fName;

    std::string fInputFileLow;
    std::string fHistoNameLow;
    std::string fHistoPathLow;

    std::string fInputFileHigh;
    std::string fHistoNameHigh;
    std::string fHistoPathHigh;

    // The Low and High Histograms
    TRef fhLow;
    TRef fhHigh;

  };


  class ShapeSys {

  public:
    //friend class Channel;


    void SetName( const std::string& Name ) { fName = Name; }
    std::string GetName() { return fName; }

    void SetInputFile( const std::string& InputFile ) { fInputFile = InputFile; }
    std::string GetInputFile() { return fInputFile; }

    void SetHistoName( const std::string& HistoName ) { fHistoName = HistoName; }
    std::string GetHistoName() { return fHistoName; }

    void SetHistoPath( const std::string& HistoPath ) { fHistoPath = HistoPath; }
    std::string GetHistoPath() { return fHistoPath; }


    void Print(std::ostream& = std::cout);  
    void writeToFile( std::string FileName, std::string DirName );

    TH1* GetErrorHist();
    void SetErrorHist(TH1* hError) { fhError = hError; }



    void SetConstraintType( Constraint::Type ConstrType ) { fConstraintType = ConstrType; }
    Constraint::Type GetConstraintType() { return fConstraintType; }


  protected:

    std::string fName;
    std::string fInputFile;
    std::string fHistoName;
    std::string fHistoPath;
    Constraint::Type fConstraintType; 

    // The histogram holding the error
    TRef fhError;

  };


  class ShapeFactor {

  public:
    //friend class Channel;  

    void SetName( const std::string& Name ) { fName = Name; }
    std::string GetName() { return fName; }


    void Print(std::ostream& = std::cout);  

  protected:
    std::string fName;

  };


  class StatError {

  public:
    //friend class Channel;

    StatError() : fActivate(false), fUseHisto(false), fhError(NULL) {;}

    void Print(std::ostream& = std::cout);  
    void writeToFile( std::string FileName, std::string DirName );

    void Activate( bool IsActive=true ) { fActivate = IsActive; }
    bool GetActivate() { return fActivate; }

    void SetUseHisto( bool UseHisto=true ) { fUseHisto = UseHisto; }
    bool GetUseHisto() { return fUseHisto; }

    void SetInputFile( const std::string& InputFile ) { fInputFile = InputFile; }
    std::string GetInputFile() { return fInputFile; }

    void SetHistoName( const std::string& HistoName ) { fHistoName = HistoName; }
    std::string GetHistoName() { return fHistoName; }

    void SetHistoPath( const std::string& HistoPath ) { fHistoPath = HistoPath; }
    std::string GetHistoPath() { return fHistoPath; }


    TH1* GetErrorHist();
    void SetErrorHist(TH1* Error) { fhError = Error; }


  protected:

    bool fActivate;
    bool fUseHisto; // Use an external histogram for the errors 
    std::string fInputFile;
    std::string fHistoName;
    std::string fHistoPath;

    // The histogram holding the error
    TRef fhError;

  };

  class StatErrorConfig {

  public:
    //friend class Channel;


    StatErrorConfig() : fRelErrorThreshold( .05 ), fConstraintType( Constraint::Gaussian ) {;}
    void Print(std::ostream& = std::cout);  

    void SetRelErrorThreshold( double Threshold ) { fRelErrorThreshold = Threshold; }
    double GetRelErrorThreshold() { return fRelErrorThreshold; }

    void SetConstraintType( Constraint::Type ConstrType ) { fConstraintType = ConstrType; }
    Constraint::Type GetConstraintType() { return fConstraintType; }


  protected:

    double fRelErrorThreshold;
    Constraint::Type fConstraintType; 

  };


}
}

#endif
