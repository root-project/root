// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, George Lewis
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////

/*
BEGIN_HTML
<p>
</p>
END_HTML
*/
//


#include "RooStats/HistFactory/Systematics.h"
#include "RooStats/HistFactory/HistFactoryException.h"


// Constraints
std::string RooStats::HistFactory::Constraint::Name( Constraint::Type type ) {

  if( type == Constraint::Gaussian ) return "Gaussian";
  if( type == Constraint::Poisson )  return "Poisson";
  return "";
}

RooStats::HistFactory::Constraint::Type RooStats::HistFactory::Constraint::GetType( const std::string& Name ) {

  if( Name == "" ) {
    std::cout << "Error: Given empty name for ConstraintType" << std::endl;
    throw hf_exc();
  }

  else if ( Name == "Gaussian" || Name == "Gauss" ) {
    return Constraint::Gaussian;
  }

  else if ( Name == "Poisson" || Name == "Pois" ) {
    return Constraint::Poisson;
  }

  else {
    std::cout << "Error: Unknown name given for Constraint Type: " << Name << std::endl;
    throw hf_exc();
  }

}

// Norm Factor
RooStats::HistFactory::NormFactor::NormFactor() : fName(""), fVal(1.0),
                    fLow(1.0), fHigh(1.0) {}

void RooStats::HistFactory::NormFactor::Print( std::ostream& stream ) const {
  stream << "\t \t Name: " << fName
    << "\t Val: " << fVal
    << "\t Low: " << fLow
    << "\t High: " << fHigh
    << std::endl;
}

void RooStats::HistFactory::NormFactor::PrintXML( std::ostream& xml ) const {
  xml << "      <NormFactor Name=\"" << GetName() << "\" "
      << " Val=\""   << GetVal()   << "\" "
      << " High=\""  << GetHigh()  << "\" "
      << " Low=\""   << GetLow()   << "\" "
      << "  /> " << std::endl;
}

// Overall Sys
void RooStats::HistFactory::OverallSys::Print( std::ostream& stream ) const {
  stream << "\t \t Name: " << fName
    << "\t Low: " << fLow
    << "\t High: " << fHigh
    << std::endl;
}

void RooStats::HistFactory::OverallSys::PrintXML( std::ostream& xml ) const {
  xml << "      <OverallSys Name=\"" << GetName() << "\" "
      << " High=\"" << GetHigh() << "\" "
      << " Low=\""  << GetLow()  << "\" "
      << "  /> " << std::endl;
}


void RooStats::HistFactory::HistogramUncertaintyBase::Print( std::ostream& stream ) const {
  stream << "\t \t Name: " << fName
    << "\t HistoFileLow: " << fInputFileLow
    << "\t HistoNameLow: " << fHistoNameLow
    << "\t HistoPathLow: " << fHistoPathLow
    << "\t HistoFileHigh: " << fInputFileHigh
    << "\t HistoNameHigh: " << fHistoNameHigh
    << "\t HistoPathHigh: " << fHistoPathHigh
    << std::endl;
}

void RooStats::HistFactory::HistogramUncertaintyBase::writeToFile( const std::string& FileName,
                     const std::string& DirName ) {

  // This saves the histograms to a file and
  // changes the name of the local file and histograms

  auto histLow = GetHistoLow();
  if( histLow==nullptr ) {
    std::cout << "Error: Cannot write " << GetName()
         << " to file: " << FileName
         << " HistoLow is nullptr"
         << std::endl;
    throw hf_exc();
  }
  histLow->Write();
  fInputFileLow = FileName;
  fHistoPathLow = DirName;
  fHistoNameLow = histLow->GetName();

  auto histHigh = GetHistoHigh();
  if( histHigh==nullptr ) {
    std::cout << "Error: Cannot write " << GetName()
         << " to file: " << FileName
         << " HistoHigh is nullptr"
         << std::endl;
    throw hf_exc();
  }
  histHigh->Write();
  fInputFileHigh = FileName;
  fHistoPathHigh = DirName;
  fHistoNameHigh = histHigh->GetName();

  return;

}


void RooStats::HistFactory::HistoSys::PrintXML( std::ostream& xml ) const {
  xml << "      <HistoSys Name=\"" << GetName() << "\" "
      << " HistoFileLow=\""  << GetInputFileLow()  << "\" "
      << " HistoNameLow=\""  << GetHistoNameLow()  << "\" "
      << " HistoPathLow=\""  << GetHistoPathLow()  << "\" "

      << " HistoFileHigh=\""  << GetInputFileHigh()  << "\" "
      << " HistoNameHigh=\""  << GetHistoNameHigh()  << "\" "
      << " HistoPathHigh=\""  << GetHistoPathHigh()  << "\" "
      << "  /> " << std::endl;
}

// Shape Sys

void RooStats::HistFactory::ShapeSys::Print( std::ostream& stream ) const {
  stream << "\t \t Name: " << fName
    << "\t InputFile: " << fInputFileHigh
    << "\t HistoName: " << fHistoNameHigh
    << "\t HistoPath: " << fHistoPathHigh
    << std::endl;
}


void RooStats::HistFactory::ShapeSys::PrintXML( std::ostream& xml ) const {
  xml << "      <ShapeSys Name=\"" << GetName() << "\" "
      << " InputFile=\""  << GetInputFile()  << "\" "
      << " HistoName=\""  << GetHistoName()  << "\" "
      << " HistoPath=\""  << GetHistoPath()  << "\" "
      << " ConstraintType=\"" << std::string(Constraint::Name(GetConstraintType())) << "\" "
      << "  /> " << std::endl;
}


void RooStats::HistFactory::ShapeSys::writeToFile( const std::string& FileName,
                     const std::string& DirName ) {

  auto histError = GetErrorHist();
  if( histError==nullptr ) {
    std::cout << "Error: Cannot write " << GetName()
         << " to file: " << FileName
         << " ErrorHist is nullptr"
         << std::endl;
    throw hf_exc();
  }
  histError->Write();
  fInputFileHigh = FileName;
  fHistoPathHigh = DirName;
  fHistoNameHigh = histError->GetName();

  return;

}




// HistoFactor

void RooStats::HistFactory::HistoFactor::PrintXML( std::ostream& xml ) const {
  xml << "      <HistoFactor Name=\"" << GetName() << "\" "

      << " InputFileLow=\""  << GetInputFileLow()  << "\" "
      << " HistoNameLow=\""  << GetHistoNameLow()  << "\" "
      << " HistoPathLow=\""  << GetHistoPathLow()  << "\" "

      << " InputFileHigh=\""  << GetInputFileHigh()  << "\" "
      << " HistoNameHigh=\""  << GetHistoNameHigh()  << "\" "
      << " HistoPathHigh=\""  << GetHistoPathHigh()  << "\" "
      << "  /> " << std::endl;
}


// Shape Factor
void RooStats::HistFactory::ShapeFactor::Print( std::ostream& stream ) const {

  stream << "\t \t Name: " << fName << std::endl;

  if( fHistoNameHigh != "" ) {
    stream << "\t \t "
      << " Shape Hist Name: " << fHistoNameHigh
      << " Shape Hist Path Name: " << fHistoPathHigh
      << " Shape Hist FileName: " << fInputFileHigh
      << std::endl;
  }

  if( fConstant ) { stream << "\t \t ( Constant ): " << std::endl; }

}


void RooStats::HistFactory::ShapeFactor::writeToFile( const std::string& FileName,
                        const std::string& DirName ) {

  if( HasInitialShape() ) {
    auto histInitialShape = GetInitialShape();
    if( histInitialShape==nullptr ) {
      std::cout << "Error: Cannot write " << GetName()
      << " to file: " << FileName
      << " InitialShape is nullptr"
      << std::endl;
      throw hf_exc();
    }
    histInitialShape->Write();
    fInputFileHigh = FileName;
    fHistoPathHigh = DirName;
    fHistoNameHigh = histInitialShape->GetName();
  }

  return;

}


void RooStats::HistFactory::ShapeFactor::PrintXML( std::ostream& xml ) const {
  xml << "      <ShapeFactor Name=\"" << GetName() << "\" ";
  if( fHasInitialShape ) {
    xml << " InputFile=\""  << GetInputFile()  << "\" "
   << " HistoName=\""  << GetHistoName()  << "\" "
   << " HistoPath=\""  << GetHistoPath()  << "\" ";
  }
  xml << "  /> " << std::endl;
}


// Stat Error Config
void RooStats::HistFactory::StatErrorConfig::Print( std::ostream& stream ) const {
  stream << "\t \t RelErrorThreshold: " << fRelErrorThreshold
    << "\t ConstraintType: " << Constraint::Name( fConstraintType )
    << std::endl;
}

void RooStats::HistFactory::StatErrorConfig::PrintXML( std::ostream& xml ) const {
  xml << "    <StatErrorConfig RelErrorThreshold=\"" << GetRelErrorThreshold()
      << "\" "
      << "ConstraintType=\"" << Constraint::Name( GetConstraintType() )
      << "\" "
      << "/> " << std::endl << std::endl;

}


// Stat Error
void RooStats::HistFactory::StatError::Print( std::ostream& stream ) const {
  stream << "\t \t Activate: " << fActivate
    << "\t InputFile: " << fInputFileHigh
    << "\t HistoName: " << fHistoNameHigh
    << "\t histoPath: " << fHistoPathHigh
    << std::endl;
}

void RooStats::HistFactory::StatError::PrintXML( std::ostream& xml ) const {

  if( GetActivate() ) {
    xml << "      <StatError Activate=\""
   << (GetActivate() ? std::string("True") : std::string("False"))
   << "\" "
   << " InputFile=\"" << GetInputFile() << "\" "
   << " HistoName=\"" << GetHistoName() << "\" "
   << " HistoPath=\"" << GetHistoPath() << "\" "
   << " /> " << std::endl;
  }

}


void RooStats::HistFactory::StatError::writeToFile( const std::string& OutputFileName,
                      const std::string& DirName ) {

  if( fUseHisto ) {

    std::string statErrorHistName = "statisticalErrors";

    auto hStatError = GetErrorHist();
    if( hStatError == nullptr ) {
      std::cout << "Error: Stat Error error hist is nullptr" << std::endl;
      throw hf_exc();
    }
    hStatError->Write(statErrorHistName.c_str());

    fInputFileHigh = OutputFileName;
    fHistoNameHigh = statErrorHistName;
    fHistoPathHigh = DirName;

  }

  return;

}
