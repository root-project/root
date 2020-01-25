// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, George Lewis 
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::HistFactory::Measurement
 * \ingroup HistFactory
The RooStats::HistFactory::Measurement class can be used to construct a model
by combining multiple RooStats::HistFactory::Channel objects. It also allows
to set some general properties like the integrated luminosity, its relative
uncertainty or the functional form of constraints on nuisance parameters.
*/


#include <ctime>
#include <iostream>
#include <algorithm>
#include <sys/stat.h>
#include "TSystem.h"
#include "TTimeStamp.h"

#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/HistFactoryException.h"

using namespace std;

ClassImp(RooStats::HistFactory::Measurement); ;

/// Standard constructor
RooStats::HistFactory::Measurement::Measurement() :
  fPOI(), fLumi( 1.0 ), fLumiRelErr( .10 ), 
  fBinLow( 0 ), fBinHigh( 1 ), fExportOnly( false )
{

}

/*
RooStats::HistFactory::Measurement::Measurement(const Measurement& other) :
  POI( other.POI ), Lumi( other.Lumi ), LumiRelErr( other.LumiRelErr ), 
  BinLow( other.BinLow ), BinHigh( other.BinHigh ), ExportOnly( other.ExportOnly ),
  channels( other.channels ), OutputFilePrefix( other.outputFilePrefix ),
  constantParams( other.constantParams ), { ; }
*/

/// Standard constructor specifying name and title of measurement
RooStats::HistFactory::Measurement::Measurement(const char* Name, const char* Title) :
  TNamed( Name, Title ),
  fPOI(), fLumi( 1.0 ), fLumiRelErr( .10 ), 
  fBinLow( 0 ), fBinHigh( 1 ), fExportOnly( false )
{

}


/// Set a parameter in the model to be constant.
/// the parameter does not have to exist yet, the information will be used when
/// the model is actually created.
///
/// Also checks if the parameter is already set constant.
/// We don't need to set it constant twice,
/// and we issue a warning in case this is a hint
/// of a possible bug
void RooStats::HistFactory::Measurement::AddConstantParam( const std::string& param )
{ 


  if( std::find(fConstantParams.begin(), fConstantParams.end(), param) != fConstantParams.end() ) {
    std::cout << "Warning: Setting parameter: " << param 
	      << " to constant, but it is already listed as constant.  "
	      << "You may ignore this warning."
	      << std::endl;
    return;
  }

  fConstantParams.push_back( param ); 

}


/// Set parameter of the model to given value
void RooStats::HistFactory::Measurement::SetParamValue( const std::string& param, double value )
{
  // Check if this parameter is already set to a value
  // If so, issue a warning
  // (Not sure if we want to throw an exception here, or
  // issue a warning and move along.  Thoughts?)
  if( fParamValues.find(param) != fParamValues.end() ) {
    std::cout << "Warning: Chainging parameter: " << param
	      << " value from: " << fParamValues[param]
	      << " to: " << value 
	      << std::endl;
  }

  // Store the parameter and its value
  std::cout << "Setting parameter: " << param
	    << " value to " << value
	    << std::endl;

  fParamValues[param] = value;

}


/// Add a preprocessed function by giving the function a name,
/// a functional expression, and a string with a bracketed list of dependencies (eg "SigXsecOverSM[0,3]")
void RooStats::HistFactory::Measurement::AddPreprocessFunction( std::string name, std::string expression, std::string dependencies )
{


  PreprocessFunction func(name, expression, dependencies);
  AddFunctionObject(func);

}

/// Returns a list of defined preprocess function expressions
std::vector<std::string> RooStats::HistFactory::Measurement::GetPreprocessFunctions()
{

  
  std::vector<std::string> PreprocessFunctionExpressions;
  for( unsigned int i = 0; i < fFunctionObjects.size(); ++i ) {
    std::string expression = fFunctionObjects.at(i).GetCommand();
    PreprocessFunctionExpressions.push_back( expression );
  }
  return PreprocessFunctionExpressions;
}

/// Set constraint term for given systematic to Gamma distribution
void RooStats::HistFactory::Measurement::AddGammaSyst(std::string syst, double uncert)
{
  fGammaSyst[syst] = uncert;
}

/// Set constraint term for given systematic to LogNormal distribution
void RooStats::HistFactory::Measurement::AddLogNormSyst(std::string syst, double uncert)
{
  fLogNormSyst[syst] = uncert;
}

/// Set constraint term for given systematic to uniform distribution
void RooStats::HistFactory::Measurement::AddUniformSyst(std::string syst)
{
  fUniformSyst[syst] = 1.0; // Is this parameter simply a dummy?
}

/// Define given systematics to have no external constraint
void RooStats::HistFactory::Measurement::AddNoSyst(std::string syst)
{
  fNoSyst[syst] = 1.0; // dummy value
}

/// Check if the given channel is part of this measurement
bool RooStats::HistFactory::Measurement::HasChannel( std::string ChanName )
{


  for( unsigned int i = 0; i < fChannels.size(); ++i ) {

    Channel& chan = fChannels.at(i);
    if( chan.GetName() == ChanName ) {
      return true;
    }

  }

  return false;

}


/// Get channel with given name from this measurement
/// throws an exception in case the channel is not found
RooStats::HistFactory::Channel& RooStats::HistFactory::Measurement::GetChannel( std::string ChanName )
{
  for( unsigned int i = 0; i < fChannels.size(); ++i ) {

    Channel& chan = fChannels.at(i);
    if( chan.GetName() == ChanName ) {
      return chan;
    }

  }
  
  // If we get here, we didn't find the channel

  std::cout << "Error: Did not find channel: " << ChanName
	    << " in measurement: " << GetName() << std::endl;
  throw hf_exc();

  // No Need to return after throwing exception
  // return RooStats::HistFactory::BadChannel;


}

/*
  void RooStats::HistFactory::Measurement::Print( Option_t* option ) const {
  RooStats::HistFactory::Measurement::Print( std::cout );
  return;
  }
*/

/// Print information about measurement object in tree-like structure to given stream
void RooStats::HistFactory::Measurement::PrintTree( std::ostream& stream )
{
  

  stream << "Measurement Name: " << GetName()
	 << "\t OutputFilePrefix: " << fOutputFilePrefix
	 << "\t POI: ";
  for(unsigned int i = 0; i < fPOI.size(); ++i) {
    stream << fPOI.at(i);
  }
  stream << "\t Lumi: " << fLumi
	 << "\t LumiRelErr: " << fLumiRelErr
	 << "\t BinLow: " << fBinLow
	 << "\t BinHigh: " << fBinHigh
	 << "\t ExportOnly: " << fExportOnly
	 << std::endl;


  if( fConstantParams.size() != 0 ) {
    stream << "Constant Params: ";
    for( unsigned int i = 0; i < fConstantParams.size(); ++i ) {
      stream << " " << fConstantParams.at(i);
    }
    stream << std::endl;
  }

  if( fFunctionObjects.size() != 0 ) {
    stream << "Preprocess Functions: ";
    for( unsigned int i = 0; i < fFunctionObjects.size(); ++i ) {
      stream << " " << fFunctionObjects.at(i).GetCommand();
    }
    stream << std::endl;
  }
  
  if( fChannels.size() != 0 ) {
    stream << "Channels:" << std::endl;
    for( unsigned int i = 0; i < fChannels.size(); ++i ) {
      fChannels.at(i).Print( stream );
    }
  }

  std::cout << "End Measurement: " << GetName() << std::endl;

}


#ifdef INCLUDE_RYML
#include <ryml.hpp>
#include <c4/yml/std/map.hpp>
#include <c4/yml/std/string.hpp>

namespace RooStats { namespace HistFactory {
    void read(c4::yml::NodeRef const& n, PreprocessFunction *v)
    {
      std::string name;
      std::string expression;
      std::string dependents;
      std::string command;   

      n["name"]       >> name;      
      n["expression"] >> expression;
      n["dependents"] >> dependents;
      n["command"]    >> command;   
      
      v->SetName      (name      );
      v->SetExpression(expression);
      v->SetDependents(dependents);
      v->SetCommand   (command   );
    }
    
    void write(c4::yml::NodeRef *n, PreprocessFunction const& v)
    {
      *n |= c4::yml::MAP;

      auto ch = n->append_child();
      ch["name"] << v.GetName();
      ch["expression"] << v.GetExpression();
      ch["dependents"] << v.GetDependents();
      ch["command"] << v.GetCommand();
    }
  }
}

namespace c4 { namespace yml {
  template<class T> void read(c4::yml::NodeRef const& n, vector<T> *v){
    for(size_t i=0; i<n.num_children(); ++i){
      T e;
      n[i]>>e;
      v->push_back(e);
    }
  }
  
  template<class T> void write(c4::yml::NodeRef *n, vector<T> const& v){
    *n |= c4::yml::SEQ;
    for(auto e:v){
      n->append_child() << e;
    }
  }
  }}  

template<> void RooStats::HistFactory::Measurement::Export(c4::yml::NodeRef& n) const {
  for(const auto& ch:this->fChannels){
    if(!ch.CheckHistograms()) throw std::runtime_error("unable to export histograms, please call CollectHistograms first");
  }
  
  auto meas = n[c4::to_csubstr(this->GetName())];
  meas |= c4::yml::MAP;

  // collect information
  std::map<std::string,RooStats::HistFactory::Constraint::Type> constraints;
  std::map<std::string,NormFactor > normfactors;
  for(const auto& ch:fChannels){
    for(const auto&s: ch.GetSamples()){
      for(const auto& sys:s.GetOverallSysList()){
        constraints[sys.GetName()] = RooStats::HistFactory::Constraint::Gaussian;
      }
      for(const auto& sys:s.GetHistoSysList()){
        constraints[sys.GetName()] = RooStats::HistFactory::Constraint::Gaussian;
      }
      for(const auto& sys:s.GetShapeSysList()){
        constraints[sys.GetName()] = sys.GetConstraintType();
      }
      for(const auto& norm:s.GetNormFactorList()){
        normfactors[norm.GetName()] = norm;
      }
    }
  }

  // parameters
  
  auto parlist = meas["createParameterList"];
  parlist |= c4::yml::MAP;
  
  auto lumi = parlist["Lumi"];
  lumi |= c4::yml::MAP;  
  lumi["value"] << fLumi;  
  lumi["relErr"] << fLumiRelErr;

  for(const auto& par:fPOI){
    auto node = parlist[c4::to_csubstr(par)];
    node |= c4::yml::MAP;    
    if(fParamValues.find(par) != fParamValues.end()){
      node["value"] << fParamValues.at(par);
    }    
  }
  for(const auto& par:fConstantParams){
    auto node = parlist[c4::to_csubstr(par)];
    node |= c4::yml::MAP;    
    if(fParamValues.find(par)!=fParamValues.end()){
      node["value"] << fParamValues.at(par);
    }
  }

  for(const auto& norm:normfactors){
    auto node = parlist[c4::to_csubstr(norm.second.GetName())];
    node |= c4::yml::MAP;
    node["value"] << norm.second.GetVal();        
    node["low"] << norm.second.GetLow();
    node["high"] << norm.second.GetHigh();
    node["const"] << norm.second.GetConst();        
  }
  
  // pdfs
  
  auto pdflist = meas["createPdfList"];
  pdflist |= c4::yml::MAP;
  
  auto sim = pdflist["simultaneous"];
  sim |= c4::yml::MAP;  
  auto simdict = sim["dict"];
  simdict |= c4::yml::MAP;  
  simdict["InterpolationScheme"] << fInterpolationScheme;  
  
  auto ch = sim["channels"];
  ch |= c4::yml::MAP;
  for(const auto& c:fChannels){
    c.Export(ch);
  }

  for(const auto& sys:constraints){
    auto node = pdflist[c4::to_csubstr(sys.first)];
    node |= c4::yml::MAP;
    node["type"] << RooStats::HistFactory::Constraint::Name(sys.second);
    if(sys.second == RooStats::HistFactory::Constraint::Gaussian){
      node["x"] << std::string("alpha_")+sys.first;
      node["mean"] << "0.";
      node["sigma"] << "1.";
    }
  }
  
  for(const auto& sys:fGammaSyst){
    auto node = pdflist[c4::to_csubstr(sys.first)];
    node |= c4::yml::MAP;
    node["value"] << sys.second;    
  }
  for(auto sys:fUniformSyst){
    auto node = pdflist[c4::to_csubstr(sys.first)];
    node |= c4::yml::MAP;
    node["value"] << sys.second;    
  }
  for(auto sys:fLogNormSyst){
    auto node = pdflist[c4::to_csubstr(sys.first)];    
    node |= c4::yml::MAP;
    node["value"] << sys.second;    
  }
  for(auto sys:fNoSyst){
    auto node = pdflist[c4::to_csubstr(sys.first)];    
    node |= c4::yml::MAP;    
    node["value"] << sys.second;
  }

  if(fFunctionObjects.size() > 0){
    meas["createFunctionList"] << fFunctionObjects;
  }
}
#endif

void RooStats::HistFactory::Measurement::PrintJSON( std::ostream& os ) {
#ifdef INCLUDE_RYML  
  ryml::Tree t;
  c4::yml::NodeRef n = t.rootref();
  n |= c4::yml::MAP;
  this->Export(n);
  os << c4::yml::as_json(t);
#else
  std::cerr << "JSON export only support with rapidyaml!" << std::endl;
#endif
}
void RooStats::HistFactory::Measurement::PrintJSON( std::string filename ) {
  std::ofstream out(filename);
  this->PrintJSON(out);
}

void RooStats::HistFactory::Measurement::PrintYAML( std::ostream& os ) {
#ifdef INCLUDE_RYML  
  ryml::Tree t;
  c4::yml::NodeRef n = t.rootref();
  n |= c4::yml::MAP;
  this->Export(n);
  os << t;
#else
  std::cerr << "YAML export only support with rapidyaml!" << std::endl;
#endif
}
void RooStats::HistFactory::Measurement::PrintYAML( std::string filename ) {
  std::ofstream out(filename);
  this->PrintYAML(out);
}


/// Create XML files for this measurement in the given directory.
/// XML files can be configured with a different output prefix
/// Create an XML file for this measurement
/// First, create the XML driver
/// Then, create xml files for each channel
void RooStats::HistFactory::Measurement::PrintXML( std::string directory, std::string newOutputPrefix )
{
  // First, check that the directory exists:
  auto testExists = [](const std::string& theDirectory) {
    void* dir = gSystem->OpenDirectory(theDirectory.c_str());
    bool exists = dir != nullptr;
    if (exists)
      gSystem->FreeDirectory(dir);

    return exists;
  };
  
  if ( !directory.empty() && !testExists(directory) ) {
    int success = gSystem->MakeDirectory(directory.c_str() );    
    if( success != 0 ) {
      std::cout << "Error: Failed to make directory: " << directory << std::endl;
      throw hf_exc();
    }
  }

  // If supplied new Prefix, use that one:

  std::cout << "Printing XML Files for measurement: " << GetName() << std::endl;

  std::string XMLName = std::string(GetName()) + ".xml";
  if( directory != "" ) XMLName = directory + "/" + XMLName;

  ofstream xml( XMLName.c_str() );

  if( ! xml.is_open() ) {
    std::cout << "Error opening xml file: " << XMLName << std::endl;
    throw hf_exc();
  }


  // Add the time
  xml << "<!--" << std::endl;
  xml << "This xml file created automatically on: " << std::endl;
/*
  time_t t = time(0);   // get time now
  struct tm * now = localtime( &t );
  xml << (now->tm_year + 1900) << '-'
      << (now->tm_mon + 1) << '-'
      << now->tm_mday
      << std::endl;
*/
  // LM: use TTimeStamp 
  TTimeStamp t; 
  UInt_t year = 0; 
  UInt_t month = 0; 
  UInt_t day = 0; 
  t.GetDate(true, 0, &year, &month, &day);
  xml << year << '-'
      << month << '-'
      << day
      << std::endl;

  xml << "-->" << std::endl;

  // Add the doctype
  xml << "<!DOCTYPE Combination  SYSTEM 'HistFactorySchema.dtd'>" << std::endl << std::endl;

  // Add the combination name
  if (newOutputPrefix.empty() ) newOutputPrefix = fOutputFilePrefix;  
  xml << "<Combination OutputFilePrefix=\"" << newOutputPrefix /*OutputFilePrefix*/ << "\" >" << std::endl << std::endl;

  // Add the Preprocessed Functions
  for( unsigned int i = 0; i < fFunctionObjects.size(); ++i ) {
    RooStats::HistFactory::PreprocessFunction func = fFunctionObjects.at(i);
    func.PrintXML(xml);
    /*
    xml << "<Function Name=\"" << func.GetName() << "\" "
	<< "Expression=\""     << func.GetExpression() << "\" "
	<< "Dependents=\""     << func.GetDependents() << "\" "
	<< "/>" << std::endl;
    */
  }
  
  xml << std::endl;

  // Add the list of channels
  for( unsigned int i = 0; i < fChannels.size(); ++i ) {
     xml << "  <Input>" << "./";
     if (!directory.empty() ) xml << directory << "/";
     xml << GetName() << "_" << fChannels.at(i).GetName() << ".xml" << "</Input>" << std::endl;
  }

  xml << std::endl;

  // Open the Measurement, Set Lumi
  xml << "  <Measurement Name=\"" << GetName() << "\" "
      << "Lumi=\""        << fLumi       << "\" " 
      << "LumiRelErr=\""  << fLumiRelErr << "\" "
    //<< "BinLow=\""      << fBinLow     << "\" "
    // << "BinHigh=\""     << fBinHigh    << "\" "
      << "ExportOnly=\""  << (fExportOnly ? std::string("True") : std::string("False")) << "\" "
      << " >" <<  std::endl;


  // Set the POI
  xml << "    <POI>" ;
  for(unsigned int i = 0; i < fPOI.size(); ++i) {
    if(i==0) xml << fPOI.at(i);
    else     xml << " " << fPOI.at(i);
  } 
  xml << "</POI>  " << std::endl;
  
  // Set the Constant Parameters
  if(fConstantParams.size()) {
    xml << "    <ParamSetting Const=\"True\">";
    for( unsigned int i = 0; i < fConstantParams.size(); ++i ) {
      if (i==0) xml << fConstantParams.at(i);
      else      xml << " " << fConstantParams.at(i);;
    }
    xml << "</ParamSetting>" << std::endl;
  }
  
  // Set the Parameters with new Constraint Terms
  std::map<std::string, double>::iterator ConstrItr;
  
  // Gamma
  for( ConstrItr = fGammaSyst.begin(); ConstrItr != fGammaSyst.end(); ++ConstrItr ) {
    xml << "<ConstraintTerm Type=\"Gamma\" RelativeUncertainty=\""
	<< ConstrItr->second << "\">" << ConstrItr->first
	<< "</ConstraintTerm>" << std::endl; 
  }
  // Uniform
  for( ConstrItr = fUniformSyst.begin(); ConstrItr != fUniformSyst.end(); ++ConstrItr ) {
    xml << "<ConstraintTerm Type=\"Uniform\" RelativeUncertainty=\""
	<< ConstrItr->second << "\">" << ConstrItr->first
	<< "</ConstraintTerm>" << std::endl; 
  }
  // LogNormal
  for( ConstrItr = fLogNormSyst.begin(); ConstrItr != fLogNormSyst.end(); ++ConstrItr ) {
    xml << "<ConstraintTerm Type=\"LogNormal\" RelativeUncertainty=\""
	<< ConstrItr->second << "\">" << ConstrItr->first
	<< "</ConstraintTerm>" << std::endl; 
  }
  // NoSyst
  for( ConstrItr = fNoSyst.begin(); ConstrItr != fNoSyst.end(); ++ConstrItr ) {
    xml << "<ConstraintTerm Type=\"NoSyst\" RelativeUncertainty=\""
	<< ConstrItr->second << "\">" << ConstrItr->first
	<< "</ConstraintTerm>" << std::endl; 
  }


  // Close the Measurement
  xml << "  </Measurement> " << std::endl << std::endl;

  // Close the combination
  xml << "</Combination>" << std::endl;

  xml.close();

  // Now, make the xml files 
  // for the individual channels:

  std::string prefix = std::string(GetName()) + "_";

  for( unsigned int i = 0; i < fChannels.size(); ++i ) {
    fChannels.at(i).PrintXML( directory, prefix );
  }


  std::cout << "Finished printing XML files" << std::endl;

}


/// A measurement, once fully configured, can be saved into a ROOT
/// file. This will persitify the Measurement object, along with any
/// channels and samples that have been added to it. It can then be
/// loaded, potentially modified, and used to create new models.
///
/// Write every histogram to the file.
/// Edit the measurement to point to this file
/// and to point to each histogram in this file
/// Then write the measurement itself.
void RooStats::HistFactory::Measurement::writeToFile( TFile* file )
{

  // Create a tempory measurement
  // (This is the one that is actually written)
  RooStats::HistFactory::Measurement outMeas( *this );

  std::string OutputFileName = file->GetName();

  // Collect all histograms from file:
  // HistCollector collector;


  for( unsigned int chanItr = 0; chanItr < outMeas.fChannels.size(); ++chanItr ) {
    
    // Go to the main directory 
    // in the file
    file->cd();
    file->Flush();

    // Get the name of the channel:
    RooStats::HistFactory::Channel& channel = outMeas.fChannels.at( chanItr );
    std::string chanName = channel.GetName();

    
    if( ! channel.CheckHistograms() ) {
      std::cout << "Measurement.writeToFile(): Channel: " << chanName
		<< " has uninitialized histogram pointers" << std::endl;
      throw hf_exc();
      return;
    }
    
    // Get and cache the histograms for this channel:
    //collector.CollectHistograms( channel );
    // Do I need this...?
    // channel.CollectHistograms();

    // Make a directory to store the histograms
    // for this channel

    TDirectory* chanDir = file->mkdir( (chanName + "_hists").c_str() );
    if( chanDir == NULL ) {
      std::cout << "Error: Cannot create channel " << (chanName + "_hists")
		<< std::endl;
      throw hf_exc();
    }
    chanDir->cd();

    // Save the data:
    TDirectory* dataDir = chanDir->mkdir( "data" );
    if( dataDir == NULL ) {
      std::cout << "Error: Cannot make directory " << chanDir << std::endl;
      throw hf_exc();
    }
    dataDir->cd();

    channel.fData.writeToFile( OutputFileName, GetDirPath(dataDir) );

    /*
    // Write the data file to this directory
    TH1* hData = channel.data.GetHisto();
    hData->Write();

    // Set the location of the data
    // in the output measurement

    channel.data.InputFile = OutputFileName;
    channel.data.HistoName = hData->GetName();
    channel.data.HistoPath = GetDirPath( dataDir );
    */

    // Loop over samples:

    for( unsigned int sampItr = 0; sampItr < channel.GetSamples().size(); ++sampItr ) {

      RooStats::HistFactory::Sample& sample = channel.GetSamples().at( sampItr );
      std::string sampName = sample.GetName();
      
      std::cout << "Writing sample: " << sampName << std::endl;

      file->cd();
      chanDir->cd();
      TDirectory* sampleDir = chanDir->mkdir( sampName.c_str() );
      if( sampleDir == NULL ) {
	std::cout << "Error: Directory " << sampName << " not created properly" << std::endl;
	throw hf_exc();
      }
      std::string sampleDirPath = GetDirPath( sampleDir );

      if( ! sampleDir ) {
	std::cout << "Error making directory: " << sampName 
		  << " in directory: " << chanName
		  << std::endl;
	throw hf_exc();
      }

      // Write the data file to this directory
      sampleDir->cd();      
      
      sample.writeToFile( OutputFileName, sampleDirPath );
      /*
      TH1* hSample = sample.GetHisto();
      if( ! hSample ) {
	std::cout << "Error getting histogram for sample: " 
		  << sampName << std::endl;
	throw -1;
      }
      sampleDir->cd();    
      hSample->Write();

      sample.InputFile = OutputFileName;
      sample.HistoName = hSample->GetName();
      sample.HistoPath = sampleDirPath;
      */

      // Write the histograms associated with
      // systematics

      /*  THIS IS WHAT I"M COMMENTING
      sample.GetStatError().writeToFile( OutputFileName, sampleDirPath );

      // Must write all systematics that contain internal histograms
      // (This is not all systematics)

      for( unsigned int i = 0; i < sample.GetHistoSysList().size(); ++i ) {
	sample.GetHistoSysList().at(i).writeToFile( OutputFileName, sampleDirPath );
      }
      for( unsigned int i = 0; i < sample.GetHistoFactorList().size(); ++i ) {
	sample.GetHistoFactorList().at(i).writeToFile( OutputFileName, sampleDirPath );
      }
      for( unsigned int i = 0; i < sample.GetShapeSysList().size(); ++i ) {
	sample.GetShapeSysList().at(i).writeToFile( OutputFileName, sampleDirPath );
      }
      END COMMENT  */
      /*
      sample.statError.writeToFile( OutputFileName, sampleDirPath );

      // Now, get the Stat config histograms
      if( sample.statError.HistoName != "" ) {
	TH1* hStatError = sample.statError.GetErrorHist();
	if( ! hStatError ) {
	  std::cout << "Error getting stat error histogram for sample: " 
		    << sampName << std::endl;
	  throw -1;
	}
	hStatError->Write();
      
	sample.statError.InputFile = OutputFileName;
	sample.statError.HistoName = hStatError->GetName();
	sample.statError.HistoPath = sampleDirPath;

      }
      */

    }

  }
  
  
  // Finally, write the measurement itself:

  std::cout << "Saved all histograms" << std::endl;
  
  file->cd();
  outMeas.Write();

  std::cout << "Saved Measurement" << std::endl;

}

/// Return the directory's path,
/// stripped of unnecessary prefixes
std::string RooStats::HistFactory::Measurement::GetDirPath( TDirectory* dir )
{


  std::string path = dir->GetPath();

  if( path.find(":") != std::string::npos ) {
    size_t index = path.find(":");
    path.replace( 0, index+1, "" );
  }                   

  path = path + "/";

  return path;

  /*
      if( path.find(":") != std::string::npos ) {
	size_t index = path.find(":");
	SampleName.replace( 0, index, "" );
      }                   

      // Remove the file:
      */

}


/// The most common way to add histograms to channels is to have them
/// stored in ROOT files and to give HistFactory the location of these
/// files. This means providing the path to the ROOT file and the path
/// and name of the histogram within that file. When providing these
/// in a script, HistFactory doesn't load the histogram from the file
/// right away. Instead, once all such histograms have been supplied,
/// one should run this method to open all ROOT files and to copy and
/// save all necessary histograms.
void RooStats::HistFactory::Measurement::CollectHistograms() {


  for( unsigned int chanItr = 0; chanItr < fChannels.size(); ++chanItr) {

    RooStats::HistFactory::Channel& chan = fChannels.at( chanItr );
    
    chan.CollectHistograms();

  }

}



