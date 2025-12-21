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

#include <RooStats/HistFactory/Measurement.h>

#include <RooStats/HistFactory/HistFactoryException.h>

#include <HFMsgService.h>

#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <TDirectory.h>
#include <TFile.h>
#include <TH1.h>
#include <TKey.h>
#include <TSystem.h>
#include <TTimeStamp.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

namespace RooStats::HistFactory {

/// Set a parameter in the model to be constant.
/// the parameter does not have to exist yet, the information will be used when
/// the model is actually created.
///
/// Also checks if the parameter is already set constant.
/// We don't need to set it constant twice,
/// and we issue a warning in case this is a hint
/// of a possible bug
void Measurement::AddConstantParam(const std::string &param)
{

   if (std::find(fConstantParams.begin(), fConstantParams.end(), param) != fConstantParams.end()) {
      cxcoutWHF << "Warning: Setting parameter: " << param << " to constant, but it is already listed as constant.  "
                << "You may ignore this warning." << std::endl;
      return;
   }

   fConstantParams.push_back(param);
}

/// Set parameter of the model to given value
void Measurement::SetParamValue(const std::string &param, double value)
{
   // Check if this parameter is already set to a value
   // If so, issue a warning
   // (Not sure if we want to throw an exception here, or
   // issue a warning and move along.  Thoughts?)
   if (fParamValues.find(param) != fParamValues.end()) {
      cxcoutWHF << "Warning: Chainging parameter: " << param << " value from: " << fParamValues[param]
                << " to: " << value << std::endl;
   }

   // Store the parameter and its value
   cxcoutIHF << "Setting parameter: " << param << " value to " << value << std::endl;

   fParamValues[param] = value;
}

/// Add a preprocessed function by giving the function a name,
/// a functional expression, and a string with a bracketed list of dependencies (eg "SigXsecOverSM[0,3]")
void Measurement::AddPreprocessFunction(std::string name, std::string expression, std::string dependencies)
{

   PreprocessFunction func(name, expression, dependencies);
   AddFunctionObject(func);
}

/// Returns a list of defined preprocess function expressions
std::vector<std::string> Measurement::GetPreprocessFunctions() const
{

   std::vector<std::string> PreprocessFunctionExpressions;
   for (unsigned int i = 0; i < fFunctionObjects.size(); ++i) {
      std::string expression = fFunctionObjects.at(i).GetCommand();
      PreprocessFunctionExpressions.push_back(expression);
   }
   return PreprocessFunctionExpressions;
}

/// Set constraint term for given systematic to Gamma distribution
void Measurement::AddGammaSyst(std::string syst, double uncert)
{
   fGammaSyst[syst] = uncert;
}

/// Set constraint term for given systematic to LogNormal distribution
void Measurement::AddLogNormSyst(std::string syst, double uncert)
{
   fLogNormSyst[syst] = uncert;
}

/// Set constraint term for given systematic to uniform distribution
void Measurement::AddUniformSyst(std::string syst)
{
   fUniformSyst[syst] = 1.0; // Is this parameter simply a dummy?
}

/// Define given systematics to have no external constraint
void Measurement::AddNoSyst(std::string syst)
{
   fNoSyst[syst] = 1.0; // dummy value
}

/// Check if the given channel is part of this measurement
bool Measurement::HasChannel(std::string ChanName)
{

   for (unsigned int i = 0; i < fChannels.size(); ++i) {

      Channel &chan = fChannels.at(i);
      if (chan.GetName() == ChanName) {
         return true;
      }
   }

   return false;
}

/// Get channel with given name from this measurement
/// throws an exception in case the channel is not found
Channel &Measurement::GetChannel(std::string ChanName)
{
   for (unsigned int i = 0; i < fChannels.size(); ++i) {

      Channel &chan = fChannels.at(i);
      if (chan.GetName() == ChanName) {
         return chan;
      }
   }

   // If we get here, we didn't find the channel

   cxcoutEHF << "Error: Did not find channel: " << ChanName << " in measurement: " << GetName() << std::endl;
   throw hf_exc();

   // No Need to return after throwing exception
   // return BadChannel;
}

/// Print information about measurement object in tree-like structure to given stream
void Measurement::PrintTree(std::ostream &stream)
{

   stream << "Measurement Name: " << GetName() << "\t OutputFilePrefix: " << fOutputFilePrefix << "\t POI: ";
   for (unsigned int i = 0; i < fPOI.size(); ++i) {
      stream << fPOI.at(i);
   }
   stream << "\t Lumi: " << fLumi << "\t LumiRelErr: " << fLumiRelErr << "\t BinLow: " << fBinLow
          << "\t BinHigh: " << fBinHigh << "\t ExportOnly: " << fExportOnly << std::endl;

   if (!fConstantParams.empty()) {
      stream << "Constant Params: ";
      for (unsigned int i = 0; i < fConstantParams.size(); ++i) {
         stream << " " << fConstantParams.at(i);
      }
      stream << std::endl;
   }

   if (!fFunctionObjects.empty()) {
      stream << "Preprocess Functions: ";
      for (unsigned int i = 0; i < fFunctionObjects.size(); ++i) {
         stream << " " << fFunctionObjects.at(i).GetCommand();
      }
      stream << std::endl;
   }

   if (!fChannels.empty()) {
      stream << "Channels:" << std::endl;
      for (unsigned int i = 0; i < fChannels.size(); ++i) {
         fChannels.at(i).Print(stream);
      }
   }

   cxcoutIHF << "End Measurement: " << GetName() << std::endl;
}

/// Create XML files for this measurement in the given directory.
/// XML files can be configured with a different output prefix
/// Create an XML file for this measurement
/// First, create the XML driver
/// Then, create xml files for each channel
void Measurement::PrintXML(std::string directory, std::string newOutputPrefix)
{
   // First, check that the directory exists:
   auto testExists = [](const std::string &theDirectory) {
      void *dir = gSystem->OpenDirectory(theDirectory.c_str());
      bool exists = dir != nullptr;
      if (exists)
         gSystem->FreeDirectory(dir);

      return exists;
   };

   if (!directory.empty() && !testExists(directory)) {
      int success = gSystem->MakeDirectory(directory.c_str());
      if (success != 0) {
         cxcoutEHF << "Error: Failed to make directory: " << directory << std::endl;
         throw hf_exc();
      }
   }

   // If supplied new Prefix, use that one:

   cxcoutPHF << "Printing XML Files for measurement: " << GetName() << std::endl;

   std::string XMLName = std::string(GetName()) + ".xml";
   if (!directory.empty())
      XMLName = directory + "/" + XMLName;

   std::ofstream xml(XMLName.c_str());

   if (!xml.is_open()) {
      cxcoutEHF << "Error opening xml file: " << XMLName << std::endl;
      throw hf_exc();
   }

   // Add the time
   xml << "<!--" << std::endl;
   xml << "This xml file created automatically on: " << std::endl;

   // LM: use TTimeStamp
   TTimeStamp t;
   UInt_t year = 0;
   UInt_t month = 0;
   UInt_t day = 0;
   t.GetDate(true, 0, &year, &month, &day);
   xml << year << '-' << month << '-' << day << std::endl;

   xml << "-->" << std::endl;

   // Add the doctype
   xml << "<!DOCTYPE Combination  SYSTEM 'HistFactorySchema.dtd'>" << std::endl << std::endl;

   // Add the combination name
   if (newOutputPrefix.empty())
      newOutputPrefix = fOutputFilePrefix;
   xml << "<Combination OutputFilePrefix=\"" << newOutputPrefix /*OutputFilePrefix*/ << "\" >" << std::endl
       << std::endl;

   // Add the Preprocessed Functions
   for (unsigned int i = 0; i < fFunctionObjects.size(); ++i) {
      PreprocessFunction func = fFunctionObjects.at(i);
      func.PrintXML(xml);
   }

   xml << std::endl;

   // Add the list of channels
   for (unsigned int i = 0; i < fChannels.size(); ++i) {
      xml << "  <Input>" << "./";
      if (!directory.empty())
         xml << directory << "/";
      xml << GetName() << "_" << fChannels.at(i).GetName() << ".xml" << "</Input>" << std::endl;
   }

   xml << std::endl;

   // Open the Measurement, Set Lumi
   xml << "  <Measurement Name=\"" << GetName() << "\" "
       << "Lumi=\"" << fLumi << "\" "
       << "LumiRelErr=\"" << fLumiRelErr
       << "\" "
       //<< "BinLow=\""      << fBinLow     << "\" "
       // << "BinHigh=\""     << fBinHigh    << "\" "
       << "ExportOnly=\"" << (fExportOnly ? std::string("True") : std::string("False")) << "\" "
       << " >" << std::endl;

   // Set the POI
   xml << "    <POI>";
   for (unsigned int i = 0; i < fPOI.size(); ++i) {
      if (i == 0)
         xml << fPOI.at(i);
      else
         xml << " " << fPOI.at(i);
   }
   xml << "</POI>  " << std::endl;

   // Set the Constant Parameters
   if (!fConstantParams.empty()) {
      xml << "    <ParamSetting Const=\"True\">";
      for (unsigned int i = 0; i < fConstantParams.size(); ++i) {
         if (i == 0)
            xml << fConstantParams.at(i);
         else
            xml << " " << fConstantParams.at(i);
      }
      xml << "</ParamSetting>" << std::endl;
   }

   // Set the Parameters with new Constraint Terms
   std::map<std::string, double>::iterator ConstrItr;

   // Gamma
   for (ConstrItr = fGammaSyst.begin(); ConstrItr != fGammaSyst.end(); ++ConstrItr) {
      xml << "<ConstraintTerm Type=\"Gamma\" RelativeUncertainty=\"" << ConstrItr->second << "\">" << ConstrItr->first
          << "</ConstraintTerm>" << std::endl;
   }
   // Uniform
   for (ConstrItr = fUniformSyst.begin(); ConstrItr != fUniformSyst.end(); ++ConstrItr) {
      xml << "<ConstraintTerm Type=\"Uniform\" RelativeUncertainty=\"" << ConstrItr->second << "\">" << ConstrItr->first
          << "</ConstraintTerm>" << std::endl;
   }
   // LogNormal
   for (ConstrItr = fLogNormSyst.begin(); ConstrItr != fLogNormSyst.end(); ++ConstrItr) {
      xml << "<ConstraintTerm Type=\"LogNormal\" RelativeUncertainty=\"" << ConstrItr->second << "\">"
          << ConstrItr->first << "</ConstraintTerm>" << std::endl;
   }
   // NoSyst
   for (ConstrItr = fNoSyst.begin(); ConstrItr != fNoSyst.end(); ++ConstrItr) {
      xml << "<ConstraintTerm Type=\"NoSyst\" RelativeUncertainty=\"" << ConstrItr->second << "\">" << ConstrItr->first
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

   for (unsigned int i = 0; i < fChannels.size(); ++i) {
      fChannels.at(i).PrintXML(directory, prefix);
   }

   cxcoutPHF << "Finished printing XML files" << std::endl;
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
void Measurement::writeToFile(TFile *file)
{

   // Create a temporary measurement
   // (This is the one that is actually written)
   Measurement outMeas(*this);

   std::string OutputFileName = file->GetName();

   // Collect all histograms from file:
   // HistCollector collector;

   for (Channel &channel : fChannels) {
      // Go to the main directory
      // in the file
      file->cd();
      file->Flush();

      // Get the name of the channel:
      std::string chanName = channel.GetName();

      if (!channel.CheckHistograms()) {
         cxcoutEHF << "Measurement.writeToFile(): Channel: " << chanName << " has uninitialized histogram pointers"
                   << std::endl;
         throw hf_exc();
         return;
      }

      // Get and cache the histograms for this channel:
      // collector.CollectHistograms( channel );
      // Do I need this...?
      // channel.CollectHistograms();

      // Make a directory to store the histograms
      // for this channel

      TDirectory *chanDir = file->mkdir((chanName + "_hists").c_str());
      if (chanDir == nullptr) {
         cxcoutEHF << "Error: Cannot create channel " << (chanName + "_hists") << std::endl;
         throw hf_exc();
      }
      chanDir->cd();

      // Save the data:
      TDirectory *dataDir = chanDir->mkdir("data");
      if (dataDir == nullptr) {
         cxcoutEHF << "Error: Cannot make directory " << chanDir << std::endl;
         throw hf_exc();
      }
      dataDir->cd();

      channel.fData.writeToFile(OutputFileName, GetDirPath(dataDir));

      // Loop over samples:
      for (Sample &sample : channel.GetSamples()) {
         cxcoutPHF << "Writing sample: " << sample.GetName() << std::endl;

         file->cd();
         chanDir->cd();
         TDirectory *sampleDir = chanDir->mkdir(sample.GetName().c_str());
         if (sampleDir == nullptr) {
            cxcoutEHF << "Error: Directory " << sample.GetName() << " not created properly" << std::endl;
            throw hf_exc();
         }
         std::string sampleDirPath = GetDirPath(sampleDir);

         if (!sampleDir) {
            cxcoutEHF << "Error making directory: " << sample.GetName() << " in directory: " << chanName << std::endl;
            throw hf_exc();
         }

         // Write the data file to this directory
         sampleDir->cd();

         sample.writeToFile(OutputFileName, sampleDirPath);
      }
   }

   // Finally, write the measurement itself:

   cxcoutPHF << "Saved all histograms" << std::endl;

   file->cd();
   outMeas.Write();

   cxcoutPHF << "Saved Measurement" << std::endl;
}

/// Return the directory's path,
/// stripped of unnecessary prefixes
std::string Measurement::GetDirPath(TDirectory *dir)
{
   std::string path = dir->GetPath();

   if (path.find(':') != std::string::npos) {
      size_t index = path.find(':');
      path.replace(0, index + 1, "");
   }

   return path + "/";
}

/// The most common way to add histograms to channels is to have them
/// stored in ROOT files and to give HistFactory the location of these
/// files. This means providing the path to the ROOT file and the path
/// and name of the histogram within that file. When providing these
/// in a script, HistFactory doesn't load the histogram from the file
/// right away. Instead, once all such histograms have been supplied,
/// one should run this method to open all ROOT files and to copy and
/// save all necessary histograms.
void Measurement::CollectHistograms()
{
   for (Channel &chan : fChannels) {
      chan.CollectHistograms();
   }
}

//////////////////////////////////////////////////////////////////////////////
/** \class RooStats::HistFactory::Sample
 * \ingroup HistFactory
 */

Sample::Sample() = default;

Sample::~Sample() = default;

// copy constructor (important for Python)
Sample::Sample(const Sample &other)
   : fName(other.fName),
     fInputFile(other.fInputFile),
     fHistoName(other.fHistoName),
     fHistoPath(other.fHistoPath),
     fChannelName(other.fChannelName),

     fOverallSysList(other.fOverallSysList),
     fNormFactorList(other.fNormFactorList),
     fHistoSysList(other.fHistoSysList),
     fHistoFactorList(other.fHistoFactorList),
     fShapeSysList(other.fShapeSysList),
     fShapeFactorList(other.fShapeFactorList),

     fStatError(other.fStatError),
     fNormalizeByTheory(other.fNormalizeByTheory),
     fStatErrorActivate(other.fStatErrorActivate),
     fhNominal(other.fhNominal)
{
   if (other.fhCountingHist) {
      SetValue(other.fhCountingHist->GetBinContent(1));
   } else {
      fhCountingHist.reset();
   }
}

Sample &Sample::operator=(const Sample &other)
{
   fName = other.fName;
   fInputFile = other.fInputFile;
   fHistoName = other.fHistoName;
   fHistoPath = other.fHistoPath;
   fChannelName = other.fChannelName;

   fOverallSysList = other.fOverallSysList;
   fNormFactorList = other.fNormFactorList;
   fHistoSysList = other.fHistoSysList;
   fHistoFactorList = other.fHistoFactorList;
   fShapeSysList = other.fShapeSysList;
   fShapeFactorList = other.fShapeFactorList;

   fStatError = other.fStatError;
   fNormalizeByTheory = other.fNormalizeByTheory;
   fStatErrorActivate = other.fStatErrorActivate;
   fhNominal = other.fhNominal;

   fhCountingHist.reset();

   if (other.fhCountingHist) {
      SetValue(other.fhCountingHist->GetBinContent(1));
   } else {
      fhCountingHist.reset();
   }

   return *this;
}

Sample::Sample(std::string SampName, std::string SampHistoName, std::string SampInputFile, std::string SampHistoPath)
   : fName(SampName),
     fInputFile(SampInputFile),
     fHistoName(SampHistoName),
     fHistoPath(SampHistoPath),
     fNormalizeByTheory(true),
     fStatErrorActivate(false)
{
}

Sample::Sample(std::string SampName) : fName(SampName), fNormalizeByTheory(true), fStatErrorActivate(false) {}

const TH1 *Sample::GetHisto() const
{
   TH1 *histo = (TH1 *)fhNominal.GetObject();
   return histo;
}

void Sample::writeToFile(std::string OutputFileName, std::string DirName)
{

   const TH1 *histNominal = GetHisto();
   histNominal->Write();

   // Set the location of the data
   // in the output measurement

   fInputFile = OutputFileName;
   fHistoName = histNominal->GetName();
   fHistoPath = DirName;

   // Write this sample's StatError
   GetStatError().writeToFile(OutputFileName, DirName);

   // Must write all systematics that contain internal histograms
   // (This is not all systematics)
   for (unsigned int i = 0; i < GetHistoSysList().size(); ++i) {
      GetHistoSysList().at(i).writeToFile(OutputFileName, DirName);
   }
   for (unsigned int i = 0; i < GetHistoFactorList().size(); ++i) {
      GetHistoFactorList().at(i).writeToFile(OutputFileName, DirName);
   }
   for (unsigned int i = 0; i < GetShapeSysList().size(); ++i) {
      GetShapeSysList().at(i).writeToFile(OutputFileName, DirName);
   }
   for (unsigned int i = 0; i < GetShapeFactorList().size(); ++i) {
      GetShapeFactorList().at(i).writeToFile(OutputFileName, DirName);
   }
}

void Sample::SetValue(double val)
{

   // For use in a number counting measurement
   // Create a 1-bin histogram,
   // fill it with this input value,
   // and set this Sample's histogram to that hist

   std::string SampleHistName = fName + "_hist";

   // Histogram has 1-bin (hard-coded)
   fhCountingHist.reset();

   fhCountingHist = std::make_unique<TH1F>(SampleHistName.c_str(), SampleHistName.c_str(), 1, 0, 1);
   fhCountingHist->SetBinContent(1, val);

   // Set the histogram of the internally held data
   // node of this channel to this newly created histogram
   SetHisto(fhCountingHist.get());
}

void Sample::Print(std::ostream &stream) const
{

   stream << "\t \t Name: " << fName << "\t \t Channel: " << fChannelName
          << "\t NormalizeByTheory: " << (fNormalizeByTheory ? "True" : "False")
          << "\t StatErrorActivate: " << (fStatErrorActivate ? "True" : "False") << std::endl;

   stream << "\t \t \t \t "
          << "\t InputFile: " << fInputFile << "\t HistName: " << fHistoName << "\t HistoPath: " << fHistoPath
          << "\t HistoAddress: "
          << GetHisto()
          // << "\t Type: " << GetHisto()->ClassName()
          << std::endl;

   if (fStatError.GetActivate()) {
      stream << "\t \t \t StatError Activate: " << fStatError.GetActivate() << "\t InputFile: " << fInputFile
             << "\t HistName: " << fStatError.GetHistoName() << "\t HistoPath: " << fStatError.GetHistoPath()
             << "\t HistoAddress: " << fStatError.GetErrorHist() << std::endl;
   }

   /*
   stream<< " NormalizeByTheory: ";
   if(NormalizeByTheory)  stream << "True";
   else                   stream << "False";

   stream<< " StatErrorActivate: ";
   if(StatErrorActivate)  stream << "True";
   else                   stream << "False";
   */
}

void Sample::PrintXML(std::ofstream &xml) const
{

   // Create the sample tag
   xml << "    <Sample Name=\"" << fName << "\" "
       << " HistoPath=\"" << fHistoPath << "\" "
       << " HistoName=\"" << fHistoName << "\" "
       << " InputFile=\"" << fInputFile << "\" "
       << " NormalizeByTheory=\"" << (fNormalizeByTheory ? std::string("True") : std::string("False")) << "\" "
       << ">" << std::endl;

   // Print Stat Error (if necessary)
   fStatError.PrintXML(xml);
   /*
   if( fStatError.GetActivate() ) {
     xml << "      <StatError Activate=\"" << (fStatError.GetActivate() ? std::string("True") : std::string("False")) <<
   "\" "
    << " InputFile=\"" << fStatError.GetInputFile() << "\" "
    << " HistoName=\"" << fStatError.GetHistoName() << "\" "
    << " HistoPath=\"" << fStatError.GetHistoPath() << "\" "
    << " /> " << std::endl;
   }
   */

   // Now, print the systematics:
   for (unsigned int i = 0; i < fOverallSysList.size(); ++i) {
      OverallSys sys = fOverallSysList.at(i);
      sys.PrintXML(xml);
      /*
      xml << "      <OverallSys Name=\"" << sys.GetName() << "\" "
     << " High=\"" << sys.GetHigh() << "\" "
     << " Low=\""  << sys.GetLow()  << "\" "
     << "  /> " << std::endl;
      */
   }
   for (unsigned int i = 0; i < fNormFactorList.size(); ++i) {
      NormFactor sys = fNormFactorList.at(i);
      sys.PrintXML(xml);
      /*
      xml << "      <NormFactor Name=\"" << sys.GetName() << "\" "
     << " Val=\""   << sys.GetVal()   << "\" "
     << " High=\""  << sys.GetHigh()  << "\" "
     << " Low=\""   << sys.GetLow()   << "\" "
     << "  /> " << std::endl;
      */
   }
   for (unsigned int i = 0; i < fHistoSysList.size(); ++i) {
      HistoSys sys = fHistoSysList.at(i);
      sys.PrintXML(xml);
      /*
      xml << "      <HistoSys Name=\"" << sys.GetName() << "\" "

     << " InputFileLow=\""  << sys.GetInputFileLow()  << "\" "
     << " HistoNameLow=\""  << sys.GetHistoNameLow()  << "\" "
     << " HistoPathLow=\""  << sys.GetHistoPathLow()  << "\" "

     << " InputFileHigh=\""  << sys.GetInputFileHigh()  << "\" "
     << " HistoNameHigh=\""  << sys.GetHistoNameHigh()  << "\" "
     << " HistoPathHigh=\""  << sys.GetHistoPathHigh()  << "\" "
     << "  /> " << std::endl;
      */
   }
   for (unsigned int i = 0; i < fHistoFactorList.size(); ++i) {
      HistoFactor sys = fHistoFactorList.at(i);
      sys.PrintXML(xml);
      /*
      xml << "      <HistoFactor Name=\"" << sys.GetName() << "\" "

     << " InputFileLow=\""  << sys.GetInputFileLow()  << "\" "
     << " HistoNameLow=\""  << sys.GetHistoNameLow()  << "\" "
     << " HistoPathLow=\""  << sys.GetHistoPathLow()  << "\" "

     << " InputFileHigh=\""  << sys.GetInputFileHigh()  << "\" "
     << " HistoNameHigh=\""  << sys.GetHistoNameHigh()  << "\" "
     << " HistoPathHigh=\""  << sys.GetHistoPathHigh()  << "\" "
     << "  /> " << std::endl;
      */
   }
   for (unsigned int i = 0; i < fShapeSysList.size(); ++i) {
      ShapeSys sys = fShapeSysList.at(i);
      sys.PrintXML(xml);
      /*
      xml << "      <ShapeSys Name=\"" << sys.GetName() << "\" "

     << " InputFile=\""  << sys.GetInputFile()  << "\" "
     << " HistoName=\""  << sys.GetHistoName()  << "\" "
     << " HistoPath=\""  << sys.GetHistoPath()  << "\" "
     << " ConstraintType=\"" << std::string(Constraint::Name(sys.GetConstraintType())) << "\" "
     << "  /> " << std::endl;
      */
   }
   for (unsigned int i = 0; i < fShapeFactorList.size(); ++i) {
      ShapeFactor sys = fShapeFactorList.at(i);
      sys.PrintXML(xml);
      /*
      xml << "      <ShapeFactor Name=\"" << sys.GetName() << "\" "
     << "  /> " << std::endl;
      */
   }

   // Finally, close the tag
   xml << "    </Sample>" << std::endl;
}

// Some helper functions
// (Not strictly necessary because
//  methods are publicly accessible)

void Sample::ActivateStatError()
{

   fStatError.Activate(true);
   fStatError.SetUseHisto(false);
}

void Sample::ActivateStatError(std::string StatHistoName, std::string StatInputFile, std::string StatHistoPath)
{

   fStatError.Activate(true);
   fStatError.SetUseHisto(true);

   fStatError.SetInputFile(StatInputFile);
   fStatError.SetHistoName(StatHistoName);
   fStatError.SetHistoPath(StatHistoPath);
}

void Sample::AddOverallSys(std::string SysName, double SysLow, double SysHigh)
{

   OverallSys sys;
   sys.SetName(SysName);
   sys.SetLow(SysLow);
   sys.SetHigh(SysHigh);

   fOverallSysList.push_back(sys);
}

void Sample::AddOverallSys(const OverallSys &Sys)
{
   fOverallSysList.push_back(Sys);
}

void Sample::AddNormFactor(std::string const &SysName, double SysVal, double SysLow, double SysHigh)
{

   NormFactor norm;

   norm.SetName(SysName);
   norm.SetVal(SysVal);
   norm.SetLow(SysLow);
   norm.SetHigh(SysHigh);

   fNormFactorList.push_back(norm);
}

void Sample::AddNormFactor(const NormFactor &Factor)
{
   fNormFactorList.push_back(Factor);
}

void Sample::AddHistoSys(std::string SysName, std::string SysHistoNameLow, std::string SysHistoFileLow,
                         std::string SysHistoPathLow, std::string SysHistoNameHigh, std::string SysHistoFileHigh,
                         std::string SysHistoPathHigh)
{

   HistoSys sys;
   sys.SetName(SysName);

   sys.SetHistoNameLow(SysHistoNameLow);
   sys.SetHistoPathLow(SysHistoPathLow);
   sys.SetInputFileLow(SysHistoFileLow);

   sys.SetHistoNameHigh(SysHistoNameHigh);
   sys.SetHistoPathHigh(SysHistoPathHigh);
   sys.SetInputFileHigh(SysHistoFileHigh);

   fHistoSysList.push_back(sys);
}

void Sample::AddHistoSys(const HistoSys &Sys)
{
   fHistoSysList.push_back(Sys);
}

void Sample::AddHistoFactor(std::string SysName, std::string SysHistoNameLow, std::string SysHistoFileLow,
                            std::string SysHistoPathLow, std::string SysHistoNameHigh, std::string SysHistoFileHigh,
                            std::string SysHistoPathHigh)
{

   HistoFactor factor;
   factor.SetName(SysName);

   factor.SetHistoNameLow(SysHistoNameLow);
   factor.SetHistoPathLow(SysHistoPathLow);
   factor.SetInputFileLow(SysHistoFileLow);

   factor.SetHistoNameHigh(SysHistoNameHigh);
   factor.SetHistoPathHigh(SysHistoPathHigh);
   factor.SetInputFileHigh(SysHistoFileHigh);

   fHistoFactorList.push_back(factor);
}

void Sample::AddHistoFactor(const HistoFactor &Factor)
{
   fHistoFactorList.push_back(Factor);
}

void Sample::AddShapeFactor(std::string SysName)
{

   ShapeFactor factor;
   factor.SetName(SysName);
   fShapeFactorList.push_back(factor);
}

void Sample::AddShapeFactor(const ShapeFactor &Factor)
{
   fShapeFactorList.push_back(Factor);
}

void Sample::AddShapeSys(std::string SysName, Constraint::Type SysConstraintType, std::string SysHistoName,
                         std::string SysHistoFile, std::string SysHistoPath)
{

   ShapeSys sys;
   sys.SetName(SysName);
   sys.SetConstraintType(SysConstraintType);

   sys.SetHistoName(SysHistoName);
   sys.SetHistoPath(SysHistoPath);
   sys.SetInputFile(SysHistoFile);

   fShapeSysList.push_back(sys);
}

void Sample::AddShapeSys(const ShapeSys &Sys)
{
   fShapeSysList.push_back(Sys);
}

////////////////////////////////////////////////////////////////////////////////
/** \class RooStats::HistFactory::Channel
 *  \ingroup HistFactory
  This class encapsulates all information for the statistical interpretation of one experiment.
  It can be combined with other channels (e.g. for the combination of multiple experiments, or
  to constrain nuisance parameters with information obtained in a control region).
  A channel contains one or more samples which describe the contribution from different processes
  to this measurement.
*/

Channel::Channel(std::string ChanName, std::string ChanInputFile) : fName(ChanName), fInputFile(ChanInputFile)
{
   // create channel with given name and input file
}

// BadChannel = Channel();
Channel BadChannel;
//    BadChannel.Name = "BadChannel"; // = Channel(); //.Name = "BadChannel";

void Channel::AddSample(Sample sample)
{
   // add fully configured sample to channel

   sample.SetChannelName(GetName());
   fSamples.push_back(sample);
}

void Channel::Print(std::ostream &stream)
{
   // print information of channel to given stream

   stream << "\t Channel Name: " << fName << "\t InputFile: " << fInputFile << std::endl;

   stream << "\t Data:" << std::endl;
   fData.Print(stream);

   stream << "\t statErrorConfig:" << std::endl;
   fStatErrorConfig.Print(stream);

   if (!fSamples.empty()) {

      stream << "\t Samples: " << std::endl;
      for (unsigned int i = 0; i < fSamples.size(); ++i) {
         fSamples.at(i).Print(stream);
      }
   }

   stream << "\t End of Channel " << fName << std::endl;
}

void Channel::PrintXML(std::string const &directory, std::string const &prefix) const
{

   // Create an XML file for this channel
   cxcoutPHF << "Printing XML Files for channel: " << GetName() << std::endl;

   std::string XMLName = prefix + fName + ".xml";
   if (!directory.empty())
      XMLName = directory + "/" + XMLName;

   std::ofstream xml(XMLName.c_str());

   // Add the time
   xml << "<!--" << std::endl;
   xml << "This xml file created automatically on: " << std::endl;
   // LM: use TTimeStamp since time_t does not work on Windows
   TTimeStamp t;
   UInt_t year = 0;
   UInt_t month = 0;
   UInt_t day = 0;
   t.GetDate(true, 0, &year, &month, &day);
   xml << year << '-' << month << '-' << day << std::endl;
   xml << "-->" << std::endl;

   // Add the DOCTYPE
   xml << "<!DOCTYPE Channel  SYSTEM 'HistFactorySchema.dtd'>  " << std::endl << std::endl;

   // Add the Channel
   xml << "  <Channel Name=\"" << fName << "\" InputFile=\"" << fInputFile << "\" >" << std::endl << std::endl;

   fData.PrintXML(xml);
   for (auto const &data : fAdditionalData) {
      data.PrintXML(xml);
   }

   fStatErrorConfig.PrintXML(xml);
   /*
   xml << "    <StatErrorConfig RelErrorThreshold=\"" << fStatErrorConfig.GetRelErrorThreshold() << "\" "
       << "ConstraintType=\"" << Constraint::Name( fStatErrorConfig.GetConstraintType() ) << "\" "
       << "/> " << std::endl << std::endl;
   */

   for (auto const &sample : fSamples) {
      sample.PrintXML(xml);
      xml << std::endl << std::endl;
   }

   xml << std::endl;
   xml << "  </Channel>  " << std::endl;
   xml.close();

   cxcoutPHF << "Finished printing XML files" << std::endl;
}

void Channel::SetData(std::string DataHistoName, std::string DataInputFile, std::string DataHistoPath)
{
   // set data for this channel by specifying the name of the histogram,
   // the external ROOT file and the path to the histogram inside the ROOT file

   fData.SetHistoName(DataHistoName);
   fData.SetInputFile(DataInputFile);
   fData.SetHistoPath(DataHistoPath);
}

void Channel::SetData(TH1 *hData)
{
   // set data directly to some histogram
   fData.SetHisto(hData);
}

void Channel::SetData(double val)
{

   // For a NumberCounting measurement only
   // Set the value of data in a particular channel
   //
   // Internally, this simply creates a 1-bin TH1F for you

   std::string DataHistName = fName + "_data";

   // Histogram has 1-bin (hard-coded)
   TH1F *hData = new TH1F(DataHistName.c_str(), DataHistName.c_str(), 1, 0, 1);
   hData->SetBinContent(1, val);

   // Set the histogram of the internally held data
   // node of this channel to this newly created histogram
   SetData(hData);
}

void Channel::SetStatErrorConfig(double StatRelErrorThreshold, Constraint::Type StatConstraintType)
{

   fStatErrorConfig.SetRelErrorThreshold(StatRelErrorThreshold);
   fStatErrorConfig.SetConstraintType(StatConstraintType);
}

void Channel::SetStatErrorConfig(double StatRelErrorThreshold, std::string StatConstraintType)
{

   fStatErrorConfig.SetRelErrorThreshold(StatRelErrorThreshold);
   fStatErrorConfig.SetConstraintType(Constraint::GetType(StatConstraintType));
}

void Channel::CollectHistograms()
{

   // Loop through all Samples and Systematics
   // and collect all necessary histograms

   // Handles to open files for collecting histograms
   std::map<std::string, std::unique_ptr<TFile>> fileHandles;

   // Get the Data Histogram:

   if (!fData.GetInputFile().empty()) {
      fData.SetHisto(GetHistogram(fData.GetInputFile(), fData.GetHistoPath(), fData.GetHistoName(), fileHandles));
   }

   // Collect any histograms for additional Datasets
   for (auto &data : fAdditionalData) {
      if (!data.GetInputFile().empty()) {
         data.SetHisto(GetHistogram(data.GetInputFile(), data.GetHistoPath(), data.GetHistoName(), fileHandles));
      }
   }

   // Get the histograms for the samples:
   for (Sample &sample : fSamples) {
      // Get the nominal histogram:
      cxcoutDHF << "Collecting Nominal Histogram" << std::endl;
      TH1 *Nominal = GetHistogram(sample.GetInputFile(), sample.GetHistoPath(), sample.GetHistoName(), fileHandles);

      sample.SetHisto(Nominal);

      // Get the StatError Histogram (if necessary)
      if (sample.GetStatError().GetUseHisto()) {
         sample.GetStatError().SetErrorHist(GetHistogram(sample.GetStatError().GetInputFile(),
                                                         sample.GetStatError().GetHistoPath(),
                                                         sample.GetStatError().GetHistoName(), fileHandles));
      }

      // Get the HistoSys Variations:
      for (HistoSys &histoSys : sample.GetHistoSysList()) {
         histoSys.SetHistoLow(GetHistogram(histoSys.GetInputFileLow(), histoSys.GetHistoPathLow(),
                                           histoSys.GetHistoNameLow(), fileHandles));

         histoSys.SetHistoHigh(GetHistogram(histoSys.GetInputFileHigh(), histoSys.GetHistoPathHigh(),
                                            histoSys.GetHistoNameHigh(), fileHandles));
      } // End Loop over HistoSys

      // Get the HistoFactor Variations:
      for (HistoFactor &histoFactor : sample.GetHistoFactorList()) {
         histoFactor.SetHistoLow(GetHistogram(histoFactor.GetInputFileLow(), histoFactor.GetHistoPathLow(),
                                              histoFactor.GetHistoNameLow(), fileHandles));

         histoFactor.SetHistoHigh(GetHistogram(histoFactor.GetInputFileHigh(), histoFactor.GetHistoPathHigh(),
                                               histoFactor.GetHistoNameHigh(), fileHandles));
      } // End Loop over HistoFactor

      // Get the ShapeSys Variations:
      for (ShapeSys &shapeSys : sample.GetShapeSysList()) {
         shapeSys.SetErrorHist(
            GetHistogram(shapeSys.GetInputFile(), shapeSys.GetHistoPath(), shapeSys.GetHistoName(), fileHandles));
      } // End Loop over ShapeSys

      // Get any initial shape for a ShapeFactor
      for (ShapeFactor &shapeFactor : sample.GetShapeFactorList()) {
         // Check if we need an InitialShape
         if (shapeFactor.HasInitialShape()) {
            TH1 *hist = GetHistogram(shapeFactor.GetInputFile(), shapeFactor.GetHistoPath(), shapeFactor.GetHistoName(),
                                     fileHandles);
            shapeFactor.SetInitialShape(hist);
         }

      } // End Loop over ShapeFactor

   } // End Loop over Samples
}

bool Channel::CheckHistograms() const
{

   // Check that all internal histogram pointers
   // are properly configured (ie that they're not nullptr)

   if (fData.GetHisto() == nullptr && !fData.GetInputFile().empty()) {
      cxcoutEHF << "Error: Data Histogram for channel " << GetName() << " is nullptr." << std::endl;
      return false;
   }

   // Get the histograms for the samples:
   for (Sample const &sample : fSamples) {
      // Get the nominal histogram:
      if (sample.GetHisto() == nullptr) {
         cxcoutEHF << "Error: Nominal Histogram for sample " << sample.GetName() << " is nullptr." << std::endl;
         return false;
      } else {

         // Check if any bins are negative
         std::vector<int> NegativeBinNumber;
         std::vector<double> NegativeBinContent;
         const TH1 *histNominal = sample.GetHisto();
         for (int ibin = 1; ibin <= histNominal->GetNbinsX(); ++ibin) {
            if (histNominal->GetBinContent(ibin) < 0) {
               NegativeBinNumber.push_back(ibin);
               NegativeBinContent.push_back(histNominal->GetBinContent(ibin));
            }
         }
         if (!NegativeBinNumber.empty()) {
            cxcoutWHF << "WARNING: Nominal Histogram " << histNominal->GetName() << " for Sample = " << sample.GetName()
                      << " in Channel = " << GetName() << " has negative entries in bin numbers = ";

            for (unsigned int ibin = 0; ibin < NegativeBinNumber.size(); ++ibin) {
               if (ibin > 0)
                  std::cout << " , ";
               std::cout << NegativeBinNumber[ibin] << " : " << NegativeBinContent[ibin];
            }
            std::cout << std::endl;
         }
      }

      // Get the StatError Histogram (if necessary)
      if (sample.GetStatError().GetUseHisto()) {
         if (sample.GetStatError().GetErrorHist() == nullptr) {
            cxcoutEHF << "Error: Statistical Error Histogram for sample " << sample.GetName() << " is nullptr."
                      << std::endl;
            return false;
         }
      }

      // Get the HistoSys Variations:
      for (const HistoSys &histoSys : sample.GetHistoSysList()) {

         if (histoSys.GetHistoLow() == nullptr) {
            cxcoutEHF << "Error: HistoSyst Low for Systematic " << histoSys.GetName() << " in sample "
                      << sample.GetName() << " is nullptr." << std::endl;
            return false;
         }
         if (histoSys.GetHistoHigh() == nullptr) {
            cxcoutEHF << "Error: HistoSyst High for Systematic " << histoSys.GetName() << " in sample "
                      << sample.GetName() << " is nullptr." << std::endl;
            return false;
         }

      } // End Loop over HistoSys

      // Get the HistoFactor Variations:
      for (const HistoFactor &histoFactor : sample.GetHistoFactorList()) {

         if (histoFactor.GetHistoLow() == nullptr) {
            cxcoutEHF << "Error: HistoSyst Low for Systematic " << histoFactor.GetName() << " in sample "
                      << sample.GetName() << " is nullptr." << std::endl;
            return false;
         }
         if (histoFactor.GetHistoHigh() == nullptr) {
            cxcoutEHF << "Error: HistoSyst High for Systematic " << histoFactor.GetName() << " in sample "
                      << sample.GetName() << " is nullptr." << std::endl;
            return false;
         }

      } // End Loop over HistoFactor

      // Get the ShapeSys Variations:
      for (const ShapeSys &shapeSys : sample.GetShapeSysList()) {
         if (shapeSys.GetErrorHist() == nullptr) {
            cxcoutEHF << "Error: HistoSyst High for Systematic " << shapeSys.GetName() << " in sample "
                      << sample.GetName() << " is nullptr." << std::endl;
            return false;
         }
      } // End Loop over ShapeSys

   } // End Loop over Samples

   return true;
}

/// Open a file and copy a histogram
/// \param InputFile File where the histogram resides.
/// \param HistoPath Path of the histogram in the file.
/// \param HistoName Name of the histogram to retrieve.
/// \param lsof List of open files. Helps to prevent opening and closing a file hundreds of times.
TH1 *Channel::GetHistogram(std::string InputFile, std::string HistoPath, std::string HistoName,
                           std::map<std::string, std::unique_ptr<TFile>> &lsof)
{

   cxcoutPHF << "Getting histogram " << InputFile << ":" << HistoPath << "/" << HistoName << std::endl;

   auto &inFile = lsof[InputFile];
   if (!inFile || !inFile->IsOpen()) {
      inFile.reset(TFile::Open(InputFile.c_str()));
      if (!inFile || !inFile->IsOpen()) {
         cxcoutEHF << "Error: Unable to open input file: " << InputFile << std::endl;
         throw hf_exc();
      }
      cxcoutIHF << "Opened input file: " << InputFile << ": " << std::endl;
   }

   TDirectory *dir = inFile->GetDirectory(HistoPath.c_str());
   if (dir == nullptr) {
      cxcoutEHF << "Histogram path '" << HistoPath << "' wasn't found in file '" << InputFile << "'." << std::endl;
      throw hf_exc();
   }

   // Have to read histograms via keys, to ensure that the latest-greatest
   // name cycle is read from file. Otherwise, they might come from memory.
   auto key = dir->GetKey(HistoName.c_str());
   if (key == nullptr) {
      cxcoutEHF << "Histogram '" << HistoName << "' wasn't found in file '" << InputFile << "' in directory '"
                << HistoPath << "'." << std::endl;
      throw hf_exc();
   }

   std::unique_ptr<TH1> hist(key->ReadObject<TH1>());
   if (!hist) {
      cxcoutEHF << "Histogram '" << HistoName << "' wasn't found in file '" << InputFile << "' in directory '"
                << HistoPath << "'." << std::endl;
      throw hf_exc();
   }

   TDirectory::TContext ctx{nullptr};
   TH1 *ptr = static_cast<TH1 *>(hist->Clone());

   if (!ptr) {
      std::cerr << "Not all necessary info are set to access the input file. Check your config" << std::endl;
      std::cerr << "filename: " << InputFile << "path: " << HistoPath << "obj: " << HistoName << std::endl;
      throw hf_exc();
   }

#ifdef DEBUG
   std::cout << "Found Histogram: " << HistoName " at address: " << ptr << " with integral " << ptr->Integral()
             << " and mean " << ptr->GetMean() << std::endl;
#endif

   // Done
   return ptr;
}

////////////////////////////////////////////////////////////////////////////////
/** \class RooStats::HistFactory::Data
 *  \ingroup HistFactory
 */

Data::Data(std::string HistoName, std::string InputFile, std::string HistoPath)
   : fInputFile(InputFile), fHistoName(HistoName), fHistoPath(HistoPath)
{
}

TH1 *Data::GetHisto()
{
   return (TH1 *)fhData.GetObject();
}

const TH1 *Data::GetHisto() const
{
   return (TH1 *)fhData.GetObject();
}

void Data::Print(std::ostream &stream)
{

   stream << "\t \t InputFile: " << fInputFile << "\t HistoName: " << fHistoName << "\t HistoPath: " << fHistoPath
          << "\t HistoAddress: " << GetHisto() << std::endl;
}

void Data::writeToFile(std::string OutputFileName, std::string DirName)
{

   TH1 *histData = GetHisto();

   if (histData != nullptr) {

      histData->Write();

      // Set the location of the data
      // in the output measurement

      fInputFile = OutputFileName;
      fHistoName = histData->GetName();
      fHistoPath = DirName;
   }
}

void Data::PrintXML(std::ostream &xml) const
{

   xml << "    <Data HistoName=\"" << GetHistoName() << "\" "
       << "InputFile=\"" << GetInputFile() << "\" "
       << "HistoPath=\"" << GetHistoPath() << "\" ";
   if (!GetName().empty()) {
      xml << "Name=\"" << GetName() << "\" ";
   }
   xml << " /> " << std::endl << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/**
 * \ingroup HistFactory
 */

namespace {

/// Replaces the XML special characters with their escape codes.
std::string escapeXML(const std::string &src)
{
   std::stringstream dst;
   for (char ch : src) {
      switch (ch) {
      case '&': dst << "&amp;"; break;
      case '\'': dst << "&apos;"; break;
      case '"': dst << "&quot;"; break;
      case '<': dst << "&lt;"; break;
      case '>': dst << "&gt;"; break;
      default: dst << ch; break;
      }
   }
   return dst.str();
}

} // namespace

PreprocessFunction::PreprocessFunction(std::string const &name, std::string const &expression,
                                       std::string const &dependents)
   : fName(name), fExpression(expression), fDependents(dependents)
{
}

std::string PreprocessFunction::GetCommand() const
{
   return "expr::" + fName + "('" + fExpression + "',{" + fDependents + "})";
}

void PreprocessFunction::Print(std::ostream &stream) const
{
   stream << "\t \t Name: " << fName << "\t \t Expression: " << fExpression << "\t \t Dependents: " << fDependents
          << std::endl;
}

void PreprocessFunction::PrintXML(std::ostream &xml) const
{
   xml << "<Function Name=\"" << fName << "\" "
       << "Expression=\"" << escapeXML(fExpression) << "\" "
       << "Dependents=\"" << fDependents << "\" "
       << "/>\n";
}

////////////////////////////////////////////////////////////////////////////////
/** \class RooStats::HistFactory::Asimov
 *  \ingroup HistFactory
 *  TODO Here, we are missing some documentation.
 */

void Asimov::ConfigureWorkspace(RooWorkspace *wspace)
{

   // Here is where we set the values, and constantness
   // of all parameters in the workspace before creating
   // an asimov dataset

   /*
   // Okay, y'all, first we're going to create a snapshot
   // of the current state of the variables in the workspace

   std::string ListOfVariableNames = "";
   for( std::map< std::string, double >::iterator itr = fParamValsToSet.begin();
        itr != fParamValsToSet.end(); ++itr) {
     // Extend the Variable Name list
     ListOfVariableNames += "," + itr->first;
   }
   for( std::map< std::string, bool >::iterator itr = fParamsToFix.begin();
        itr != fParamsToFix.end(); ++itr) {
     // Extend the Variable Name list
     ListOfVariableNames += "," + itr->first;
   }

   // Save a snapshot
   std::string SnapShotName = "NominalParamValues";
   wspace->saveSnapshot(SnapShotName.c_str(), ListOfVariableNames.c_str());
   */

   //
   // First we set all parameters to their given values
   //

   for (std::map<std::string, double>::iterator itr = fParamValsToSet.begin(); itr != fParamValsToSet.end(); ++itr) {

      std::string param = itr->first;
      double val = itr->second;

      // Try to get the variable in the workspace
      RooRealVar *var = wspace->var(param);
      if (!var) {
         std::cout << "Error: Trying to set variable: " << var
                   << " to a specific value in creation of asimov dataset: " << fName
                   << " but this variable doesn't appear to exist in the workspace" << std::endl;
         throw hf_exc();
      }

      // Check that the desired value is in the range of the variable
      double inRange = var->inRange(val, nullptr);
      if (!inRange) {
         std::cout << "Error: Attempting to set variable: " << var << " to value: " << val << ", however it appears"
                   << " that this is not withn the variable's range: "
                   << "[" << var->getMin() << ", " << var->getMax() << "]" << std::endl;
         throw hf_exc();
      }

      // Set its value
      std::cout << "Configuring Asimov Dataset: Setting " << param << " = " << val << std::endl;
      var->setVal(val);
   }

   //
   // Then, we set any variables to constant
   //

   for (auto &[param, isConstant] : fParamsToFix) {

      // Try to get the variable in the workspace
      RooRealVar *var = wspace->var(param);
      if (!var) {
         std::cout << "Error: Trying to set variable: " << var << " constant in creation of asimov dataset: " << fName
                   << " but this variable doesn't appear to exist in the workspace" << std::endl;
         throw hf_exc();
      }

      std::cout << "Configuring Asimov Dataset: Setting " << param << " to constant " << std::endl;
      var->setConstant(isConstant);
   }
}

/** \class RooStats::HistFactory::HistRef
 * \ingroup HistFactory
 * Internal class wrapping an histogram and managing its content.
 * convenient for dealing with histogram pointers in the
 * HistFactory class
 */

/// constructor - use gives away ownerhip of the given pointer
HistRef::HistRef(TH1 *h) : fHist(h) {}

HistRef::HistRef(const HistRef &other)
{
   if (other.fHist)
      fHist.reset(CopyObject(other.fHist.get()));
}

HistRef::HistRef(HistRef &&other) : fHist(std::move(other.fHist)) {}

HistRef::~HistRef() = default;

/// assignment operator (delete previous contained histogram)
HistRef &HistRef::operator=(const HistRef &other)
{
   if (this == &other)
      return *this;

   fHist.reset(CopyObject(other.fHist.get()));
   return *this;
}

HistRef &HistRef::operator=(HistRef &&other)
{
   fHist = std::move(other.fHist);
   return *this;
}

TH1 *HistRef::CopyObject(const TH1 *h)
{
   // implementation of method copying the contained pointer
   // (just use Clone)
   if (!h)
      return nullptr;

   TDirectory::TContext ctx{nullptr}; // Don't associate histogram with currently open file
   return static_cast<TH1 *>(h->Clone());
}

TH1 *HistRef::GetObject() const
{
   return fHist.get();
}

/// set the object - user gives away the ownerhisp
void HistRef::SetObject(TH1 *h)
{
   fHist.reset(h);
}

/// operator= passing an object pointer :  user gives away its ownerhisp
void HistRef::operator=(TH1 *h)
{
   SetObject(h);
}

/// Release ownership of object.
TH1 *HistRef::ReleaseObject()
{
   return fHist.release();
}

// Constraints
std::string Constraint::Name(Constraint::Type type)
{

   if (type == Constraint::Gaussian)
      return "Gaussian";
   if (type == Constraint::Poisson)
      return "Poisson";
   return "";
}

Constraint::Type Constraint::GetType(const std::string &Name)
{

   if (Name.empty()) {
      std::cout << "Error: Given empty name for ConstraintType" << std::endl;
      throw hf_exc();
   }

   else if (Name == "Gaussian" || Name == "Gauss") {
      return Constraint::Gaussian;
   }

   else if (Name == "Poisson" || Name == "Pois") {
      return Constraint::Poisson;
   }

   else {
      std::cout << "Error: Unknown name given for Constraint Type: " << Name << std::endl;
      throw hf_exc();
   }
}

void NormFactor::Print(std::ostream &stream) const
{
   stream << "\t \t Name: " << fName << "\t Val: " << fVal << "\t Low: " << fLow << "\t High: " << fHigh << std::endl;
}

void NormFactor::PrintXML(std::ostream &xml) const
{
   xml << "      <NormFactor Name=\"" << GetName() << "\" "
       << " Val=\"" << GetVal() << "\" "
       << " High=\"" << GetHigh() << "\" "
       << " Low=\"" << GetLow() << "\" "
       << "  /> " << std::endl;
}

void OverallSys::Print(std::ostream &stream) const
{
   stream << "\t \t Name: " << fName << "\t Low: " << fLow << "\t High: " << fHigh << std::endl;
}

void OverallSys::PrintXML(std::ostream &xml) const
{
   xml << "      <OverallSys Name=\"" << GetName() << "\" "
       << " High=\"" << GetHigh() << "\" "
       << " Low=\"" << GetLow() << "\" "
       << "  /> " << std::endl;
}

HistogramUncertaintyBase::HistogramUncertaintyBase() = default;

HistogramUncertaintyBase::HistogramUncertaintyBase(const std::string &Name)
   : fName(Name), fhLow(nullptr), fhHigh(nullptr)
{
}

HistogramUncertaintyBase::HistogramUncertaintyBase(HistogramUncertaintyBase &&) = default;

HistogramUncertaintyBase &HistogramUncertaintyBase::operator=(HistogramUncertaintyBase &&) = default;

HistogramUncertaintyBase::~HistogramUncertaintyBase() = default;

void HistogramUncertaintyBase::Print(std::ostream &stream) const
{
   stream << "\t \t Name: " << fName << "\t HistoFileLow: " << fInputFileLow << "\t HistoNameLow: " << fHistoNameLow
          << "\t HistoPathLow: " << fHistoPathLow << "\t HistoFileHigh: " << fInputFileHigh
          << "\t HistoNameHigh: " << fHistoNameHigh << "\t HistoPathHigh: " << fHistoPathHigh << std::endl;
}

void HistogramUncertaintyBase::writeToFile(const std::string &FileName, const std::string &DirName)
{

   // This saves the histograms to a file and
   // changes the name of the local file and histograms

   auto histLow = GetHistoLow();
   if (histLow == nullptr) {
      std::cout << "Error: Cannot write " << GetName() << " to file: " << FileName << " HistoLow is nullptr"
                << std::endl;
      throw hf_exc();
   }
   histLow->Write();
   fInputFileLow = FileName;
   fHistoPathLow = DirName;
   fHistoNameLow = histLow->GetName();

   auto histHigh = GetHistoHigh();
   if (histHigh == nullptr) {
      std::cout << "Error: Cannot write " << GetName() << " to file: " << FileName << " HistoHigh is nullptr"
                << std::endl;
      throw hf_exc();
   }
   histHigh->Write();
   fInputFileHigh = FileName;
   fHistoPathHigh = DirName;
   fHistoNameHigh = histHigh->GetName();
}

void HistoSys::PrintXML(std::ostream &xml) const
{
   xml << "      <HistoSys Name=\"" << GetName() << "\" "
       << " HistoFileLow=\"" << GetInputFileLow() << "\" "
       << " HistoNameLow=\"" << GetHistoNameLow() << "\" "
       << " HistoPathLow=\"" << GetHistoPathLow() << "\" "

       << " HistoFileHigh=\"" << GetInputFileHigh() << "\" "
       << " HistoNameHigh=\"" << GetHistoNameHigh() << "\" "
       << " HistoPathHigh=\"" << GetHistoPathHigh() << "\" "
       << "  /> " << std::endl;
}

void ShapeSys::SetErrorHist(TH1 *hError)
{
   fhHigh.reset(hError);
}

void ShapeSys::Print(std::ostream &stream) const
{
   stream << "\t \t Name: " << fName << "\t InputFile: " << fInputFileHigh << "\t HistoName: " << fHistoNameHigh
          << "\t HistoPath: " << fHistoPathHigh << std::endl;
}

void ShapeSys::PrintXML(std::ostream &xml) const
{
   xml << "      <ShapeSys Name=\"" << GetName() << "\" "
       << " InputFile=\"" << GetInputFile() << "\" "
       << " HistoName=\"" << GetHistoName() << "\" "
       << " HistoPath=\"" << GetHistoPath() << "\" "
       << " ConstraintType=\"" << std::string(Constraint::Name(GetConstraintType())) << "\" "
       << "  /> " << std::endl;
}

void ShapeSys::writeToFile(const std::string &FileName, const std::string &DirName)
{
   auto histError = GetErrorHist();
   if (histError == nullptr) {
      std::cout << "Error: Cannot write " << GetName() << " to file: " << FileName << " ErrorHist is nullptr"
                << std::endl;
      throw hf_exc();
   }
   histError->Write();
   fInputFileHigh = FileName;
   fHistoPathHigh = DirName;
   fHistoNameHigh = histError->GetName();
}

void HistoFactor::PrintXML(std::ostream &xml) const
{
   xml << "      <HistoFactor Name=\"" << GetName() << "\" "

       << " InputFileLow=\"" << GetInputFileLow() << "\" "
       << " HistoNameLow=\"" << GetHistoNameLow() << "\" "
       << " HistoPathLow=\"" << GetHistoPathLow() << "\" "

       << " InputFileHigh=\"" << GetInputFileHigh() << "\" "
       << " HistoNameHigh=\"" << GetHistoNameHigh() << "\" "
       << " HistoPathHigh=\"" << GetHistoPathHigh() << "\" "
       << "  /> " << std::endl;
}

void ShapeFactor::SetInitialShape(TH1 *shape)
{
   fhHigh.reset(shape);
}

void ShapeFactor::Print(std::ostream &stream) const
{

   stream << "\t \t Name: " << fName << std::endl;

   if (!fHistoNameHigh.empty()) {
      stream << "\t \t "
             << " Shape Hist Name: " << fHistoNameHigh << " Shape Hist Path Name: " << fHistoPathHigh
             << " Shape Hist FileName: " << fInputFileHigh << std::endl;
   }

   if (fConstant) {
      stream << "\t \t ( Constant ): " << std::endl;
   }
}

void ShapeFactor::writeToFile(const std::string &FileName, const std::string &DirName)
{

   if (HasInitialShape()) {
      auto histInitialShape = GetInitialShape();
      if (histInitialShape == nullptr) {
         std::cout << "Error: Cannot write " << GetName() << " to file: " << FileName << " InitialShape is nullptr"
                   << std::endl;
         throw hf_exc();
      }
      histInitialShape->Write();
      fInputFileHigh = FileName;
      fHistoPathHigh = DirName;
      fHistoNameHigh = histInitialShape->GetName();
   }
}

void ShapeFactor::PrintXML(std::ostream &xml) const
{
   xml << "      <ShapeFactor Name=\"" << GetName() << "\" ";
   if (fHasInitialShape) {
      xml << " InputFile=\"" << GetInputFile() << "\" "
          << " HistoName=\"" << GetHistoName() << "\" "
          << " HistoPath=\"" << GetHistoPath() << "\" ";
   }
   xml << "  /> " << std::endl;
}

void StatError::SetErrorHist(TH1 *Error)
{
   fhHigh.reset(Error);
}

void StatErrorConfig::Print(std::ostream &stream) const
{
   stream << "\t \t RelErrorThreshold: " << fRelErrorThreshold
          << "\t ConstraintType: " << Constraint::Name(fConstraintType) << std::endl;
}

void StatErrorConfig::PrintXML(std::ostream &xml) const
{
   xml << "    <StatErrorConfig RelErrorThreshold=\"" << GetRelErrorThreshold() << "\" "
       << "ConstraintType=\"" << Constraint::Name(GetConstraintType()) << "\" "
       << "/> " << std::endl
       << std::endl;
}

void StatError::Print(std::ostream &stream) const
{
   stream << "\t \t Activate: " << fActivate << "\t InputFile: " << fInputFileHigh << "\t HistoName: " << fHistoNameHigh
          << "\t histoPath: " << fHistoPathHigh << std::endl;
}

void StatError::PrintXML(std::ostream &xml) const
{

   if (GetActivate()) {
      xml << "      <StatError Activate=\"" << (GetActivate() ? std::string("True") : std::string("False")) << "\" "
          << " InputFile=\"" << GetInputFile() << "\" "
          << " HistoName=\"" << GetHistoName() << "\" "
          << " HistoPath=\"" << GetHistoPath() << "\" "
          << " /> " << std::endl;
   }
}

void StatError::writeToFile(const std::string &OutputFileName, const std::string &DirName)
{

   if (fUseHisto) {

      std::string statErrorHistName = "statisticalErrors";

      auto hStatError = GetErrorHist();
      if (hStatError == nullptr) {
         std::cout << "Error: Stat Error error hist is nullptr" << std::endl;
         throw hf_exc();
      }
      hStatError->Write(statErrorHistName.c_str());

      fInputFileHigh = OutputFileName;
      fHistoNameHigh = statErrorHistName;
      fHistoPathHigh = DirName;
   }
}

HistogramUncertaintyBase::HistogramUncertaintyBase(const HistogramUncertaintyBase &oth)
   : fName{oth.fName},
     fInputFileLow{oth.fInputFileLow},
     fHistoNameLow{oth.fHistoNameLow},
     fHistoPathLow{oth.fHistoPathLow},
     fInputFileHigh{oth.fInputFileHigh},
     fHistoNameHigh{oth.fHistoNameHigh},
     fHistoPathHigh{oth.fHistoPathHigh},
     fhLow{oth.fhLow ? static_cast<TH1 *>(oth.fhLow->Clone()) : nullptr},
     fhHigh{oth.fhHigh ? static_cast<TH1 *>(oth.fhHigh->Clone()) : nullptr}
{
   if (fhLow)
      fhLow->SetDirectory(nullptr);
   if (fhHigh)
      fhHigh->SetDirectory(nullptr);
}

HistogramUncertaintyBase &HistogramUncertaintyBase::operator=(const HistogramUncertaintyBase &oth)
{
   fName = oth.fName;
   fInputFileLow = oth.fInputFileLow;
   fHistoNameLow = oth.fHistoNameLow;
   fHistoPathLow = oth.fHistoPathLow;
   fInputFileHigh = oth.fInputFileHigh;
   fHistoNameHigh = oth.fHistoNameHigh;
   fHistoPathHigh = oth.fHistoPathHigh;

   TDirectory::TContext ctx{nullptr}; // Don't associate clones to directories
   fhLow.reset(oth.fhLow ? static_cast<TH1 *>(oth.fhLow->Clone()) : nullptr);
   fhHigh.reset(oth.fhHigh ? static_cast<TH1 *>(oth.fhHigh->Clone()) : nullptr);

   return *this;
}

void HistogramUncertaintyBase::SetHistoLow(TH1 *Low)
{
   Low->SetDirectory(nullptr);
   fhLow.reset(Low);
}

void HistogramUncertaintyBase::SetHistoHigh(TH1 *High)
{
   High->SetDirectory(nullptr);
   fhHigh.reset(High);
}

void Data::SetHisto(TH1 *Hist)
{
   fhData = Hist;
   fHistoName = Hist->GetName();
}

void Sample::SetHisto(TH1 *histo)
{
   fhNominal = histo;
   fHistoName = histo->GetName();
}

} // namespace RooStats::HistFactory
