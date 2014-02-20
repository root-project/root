// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, George Lewis 
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________
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
						  fLow(1.0), fHigh(1.0), 
						  fConst(true) {;}

void RooStats::HistFactory::NormFactor::Print( std::ostream& stream ) {
  stream << "\t \t Name: " << fName
	 << "\t Val: " << fVal
	 << "\t Low: " << fLow
	 << "\t High: " << fHigh
	 << "\t Const: " << fConst
	 << std::endl;
}

void RooStats::HistFactory::NormFactor::PrintXML( std::ostream& xml ) {
  xml << "      <NormFactor Name=\"" << GetName() << "\" "
      << " Val=\""   << GetVal()   << "\" "
      << " High=\""  << GetHigh()  << "\" "
      << " Low=\""   << GetLow()   << "\" "
      << " Const=\"" << (GetConst() ? std::string("True") : std::string("False")) << "\" "
      << "  /> " << std::endl;
}

// Overall Sys
void RooStats::HistFactory::OverallSys::Print( std::ostream& stream ) {
  stream << "\t \t Name: " << fName
	 << "\t Low: " << fLow
	 << "\t High: " << fHigh
	 << std::endl;
}

void RooStats::HistFactory::OverallSys::PrintXML( std::ostream& xml ) {
  xml << "      <OverallSys Name=\"" << GetName() << "\" "
      << " High=\"" << GetHigh() << "\" "
      << " Low=\""  << GetLow()  << "\" "
      << "  /> " << std::endl;
}

// HistoSys
TH1* RooStats::HistFactory::HistoSys::GetHistoLow() {
  TH1* histo_low = (TH1*) fhLow.GetObject();
  return histo_low;
}

TH1* RooStats::HistFactory::HistoSys::GetHistoHigh() {
  TH1* histo_high = (TH1*) fhHigh.GetObject();
  return histo_high;
}

void RooStats::HistFactory::HistoSys::Print( std::ostream& stream ) {
  stream << "\t \t Name: " << fName
	 << "\t HistoFileLow: " << fInputFileLow
	 << "\t HistoNameLow: " << fHistoNameLow
	 << "\t HistoPathLow: " << fHistoPathLow
	 << "\t HistoFileHigh: " << fInputFileHigh
	 << "\t HistoNameHigh: " << fHistoNameHigh
	 << "\t HistoPathHigh: " << fHistoPathHigh
	 << std::endl;
}

void RooStats::HistFactory::HistoSys::PrintXML( std::ostream& xml ) {
  xml << "      <HistoSys Name=\"" << GetName() << "\" "
      << " HistoFileLow=\""  << GetInputFileLow()  << "\" "
      << " HistoNameLow=\""  << GetHistoNameLow()  << "\" "
      << " HistoPathLow=\""  << GetHistoPathLow()  << "\" "

      << " HistoFileHigh=\""  << GetInputFileHigh()  << "\" "
      << " HistoNameHigh=\""  << GetHistoNameHigh()  << "\" "
      << " HistoPathHigh=\""  << GetHistoPathHigh()  << "\" "
      << "  /> " << std::endl;
}

void RooStats::HistFactory::HistoSys::writeToFile( const std::string& FileName, 
						   const std::string& DirName ) {

  // This saves the histograms to a file and 
  // changes the name of the local file and histograms
  
  TH1* histLow = GetHistoLow();
  if( histLow==NULL ) {
    std::cout << "Error: Cannot write " << GetName()
	      << " to file: " << FileName
	      << " HistoLow is NULL" 
	      << std::endl;
    throw hf_exc();
  }
  histLow->Write();
  fInputFileLow = FileName;
  fHistoPathLow = DirName;
  fHistoNameLow = histLow->GetName(); 

  TH1* histHigh = GetHistoHigh();
  if( histHigh==NULL ) {
    std::cout << "Error: Cannot write " << GetName()
	      << " to file: " << FileName
	      << " HistoHigh is NULL" 
	      << std::endl;
    throw hf_exc();
  }
  histHigh->Write();
  fInputFileHigh = FileName;
  fHistoPathHigh = DirName;
  fHistoNameHigh = histHigh->GetName();

  return;

}


// Shape Sys

TH1* RooStats::HistFactory::ShapeSys::GetErrorHist() {
  TH1* error_hist = (TH1*) fhError.GetObject();
  return error_hist;
}


void RooStats::HistFactory::ShapeSys::Print( std::ostream& stream ) {
  stream << "\t \t Name: " << fName
	 << "\t InputFile: " << fInputFile
	 << "\t HistoName: " << fHistoName
	 << "\t HistoPath: " << fHistoPath
	 << std::endl;
}


void RooStats::HistFactory::ShapeSys::PrintXML( std::ostream& xml ) {
  xml << "      <ShapeSys Name=\"" << GetName() << "\" "
      << " InputFile=\""  << GetInputFile()  << "\" "
      << " HistoName=\""  << GetHistoName()  << "\" "
      << " HistoPath=\""  << GetHistoPath()  << "\" "
      << " ConstraintType=\"" << std::string(Constraint::Name(GetConstraintType())) << "\" "
      << "  /> " << std::endl;
}


void RooStats::HistFactory::ShapeSys::writeToFile( const std::string& FileName, 
						   const std::string& DirName ) {

  TH1* histError = GetErrorHist();
  if( histError==NULL ) {
    std::cout << "Error: Cannot write " << GetName()
	      << " to file: " << FileName
	      << " ErrorHist is NULL" 
	      << std::endl;
    throw hf_exc();
  }
  histError->Write();
  fInputFile = FileName;
  fHistoPath = DirName;
  fHistoName = histError->GetName(); 

  return;

}




// HistoFactor

void RooStats::HistFactory::HistoFactor::Print( std::ostream& stream ) {
  stream << "\t \t Name: " << fName
	 << "\t InputFileLow: " << fInputFileLow
	 << "\t HistoNameLow: " << fHistoNameLow
	 << "\t HistoPathLow: " << fHistoPathLow
	 << "\t InputFileHigh: " << fInputFileHigh
	 << "\t HistoNameHigh: " << fHistoNameHigh
	 << "\t HistoPathHigh: " << fHistoPathHigh
	 << std::endl;
}


TH1* RooStats::HistFactory::HistoFactor::GetHistoLow() {
  TH1* histo_low = (TH1*) fhLow.GetObject();
  return histo_low;
}

TH1* RooStats::HistFactory::HistoFactor::GetHistoHigh() {
  TH1* histo_high = (TH1*) fhHigh.GetObject();
  return histo_high;
}


void RooStats::HistFactory::HistoFactor::writeToFile( const std::string& FileName, 
						      const std::string& DirName ) {


  // This saves the histograms to a file and 
  // changes the name of the local file and histograms
  
  TH1* histLow = GetHistoLow();
  if( histLow==NULL ) {
    std::cout << "Error: Cannot write " << GetName()
	      << " to file: " << FileName
	      << " HistoLow is NULL" 
	      << std::endl;
    throw hf_exc();
  }
  histLow->Write();
  fInputFileLow = FileName;
  fHistoPathLow = DirName;
  fHistoNameLow = histLow->GetName(); 

  TH1* histHigh = GetHistoHigh();
  if( histHigh==NULL ) {
    std::cout << "Error: Cannot write " << GetName()
	      << " to file: " << FileName
	      << " HistoHigh is NULL" 
	      << std::endl;
    throw hf_exc();
  }
  histHigh->Write();
  fInputFileHigh = FileName;
  fHistoPathHigh = DirName;
  fHistoNameHigh = histHigh->GetName();

  return;

}

void RooStats::HistFactory::HistoFactor::PrintXML( std::ostream& xml ) {
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
RooStats::HistFactory::ShapeFactor::ShapeFactor() : fConstant(false), 
						    fHasInitialShape(false),
						    fhInitialShape(NULL) {;}

void RooStats::HistFactory::ShapeFactor::Print( std::ostream& stream ) {

  stream << "\t \t Name: " << fName << std::endl;
  
  if( fHistoName != "" ) {
    stream << "\t \t "
	   << " Shape Hist Name: " << fHistoName
	   << " Shape Hist Path Name: " << fHistoPath
	   << " Shape Hist FileName: " << fInputFile
	   << std::endl;
  }

  if( fConstant ) { stream << "\t \t ( Constant ): " << std::endl; }

}


void RooStats::HistFactory::ShapeFactor::writeToFile( const std::string& FileName, 
						      const std::string& DirName ) {

  if( HasInitialShape() ) {
    TH1* histInitialShape = GetInitialShape();
    if( histInitialShape==NULL ) {
      std::cout << "Error: Cannot write " << GetName()
		<< " to file: " << FileName
		<< " InitialShape is NULL" 
		<< std::endl;
      throw hf_exc();
    }
    histInitialShape->Write();
    fInputFile = FileName;
    fHistoPath = DirName;
    fHistoName = histInitialShape->GetName(); 
  }

  return;

}


void RooStats::HistFactory::ShapeFactor::PrintXML( std::ostream& xml ) {
  xml << "      <ShapeFactor Name=\"" << GetName() << "\" ";
  if( fHasInitialShape ) {
    xml << " InputFile=\""  << GetInputFile()  << "\" "
	<< " HistoName=\""  << GetHistoName()  << "\" "
	<< " HistoPath=\""  << GetHistoPath()  << "\" ";
  }
  xml << "  /> " << std::endl;
}


// Stat Error Config
void RooStats::HistFactory::StatErrorConfig::Print( std::ostream& stream ) {
  stream << "\t \t RelErrorThreshold: " << fRelErrorThreshold
	 << "\t ConstraintType: " << Constraint::Name( fConstraintType )
	 << std::endl;
}  

void RooStats::HistFactory::StatErrorConfig::PrintXML( std::ostream& xml ) {
  xml << "    <StatErrorConfig RelErrorThreshold=\"" << GetRelErrorThreshold() 
      << "\" "
      << "ConstraintType=\"" << Constraint::Name( GetConstraintType() ) 
      << "\" "
      << "/> " << std::endl << std::endl;

}


// Stat Error
TH1* RooStats::HistFactory::StatError::GetErrorHist() {
  return (TH1*) fhError.GetObject();
}


void RooStats::HistFactory::StatError::Print( std::ostream& stream ) {
  stream << "\t \t Activate: " << fActivate
	 << "\t InputFile: " << fInputFile
	 << "\t HistoName: " << fHistoName
	 << "\t histoPath: " << fHistoPath
	 << std::endl;
}  

void RooStats::HistFactory::StatError::PrintXML( std::ostream& xml ) {

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
    
    TH1* hStatError = GetErrorHist();
    if( hStatError == NULL ) {
      std::cout << "Error: Stat Error error hist is NULL" << std::endl;
      throw hf_exc();
    }
    hStatError->Write(statErrorHistName.c_str());
    
    fInputFile = OutputFileName;
    fHistoName = statErrorHistName;
    fHistoPath = DirName;
    
  }

  return;

}
