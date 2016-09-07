// @(#)root/treeplayer:$Id$
// Author: Luca Giommi   22/08/16

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TSimpleAnalysis.h"

#include "TFile.h"
#include "TChain.h"
#include "TH1.h"
#include "TError.h"
#include "TKey.h"

#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <iostream>

/** \class TSimpleAnalysis

A TSimpleAnalysis object creates histograms from a TChain. These histograms
are stored to an output file. The histogrammed (TTreeFormula) expressions,
their cuts, the input and output files are configured through a simple config
file that allows comments starting with '#'.
Here an example of configuration file:
```
# This is an example of configuration file
file_output.root   #the output file in which histograms are stored

# The next line has the name of the tree of the input data. It is
# optional if there is exactly one tree in the first input file.
ntuple   #name of the input tree

# The lines of the next block correspond to .root input files that
# contain the tree
hsimple1.root   #first .root input file
hsimple2.root   #second .root input file

# The next block is composed by lines that allow to configure the
# histograms. They have the following syntax:
# NAME = EXPRESSION if CUT
# which corresponds to chain->Draw("EXPRESSION >> NAME", "CUT")
# i.e. it will create a histogram called NAME and store it in
# file_output.root.
# "if CUT" is optional
hpx=px if px<-3   #first histogram
hpxpy=px:py    #second histogram

# End of the configuration file
```
*/

////////////////////////////////////////////////////////////////////////////////
/// Delete comments, leading and trailing white spaces in a string.
///
/// param[in] line - line read from the input file

static void DeleteCommentsAndSpaces(std::string& line)
{
   // Delete comments
   std::size_t comment = line.find('#');
   line = line.substr(0, comment);
   // Delete leading spaces
   std::size_t firstNotSpace = line.find_first_not_of(" \t");
   if (firstNotSpace != std::string::npos)
      line = line.substr(firstNotSpace);
   else {
      line.clear();
      return;
   }
   // Delete trailing spaces
   std::size_t lastNotSpace = line.find_last_not_of(" \t");
   if (lastNotSpace != std::string::npos)
      line = line.substr(0, lastNotSpace + 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Handle the expression lines of the input file in order to pass the
/// elements to the members of the object.
///
/// param[in] line - TTreeFormula expression, either read form the configuration
///                  file or passed as expression to the constructor

std::string TSimpleAnalysis::HandleExpressionConfig(const std::string& line)
{
   static const std::string kCutIntr = " if ";

   std::size_t equal = line.find("=");
   if (equal == std::string::npos)
      return "Error: missing '='";

   // Set the histName value
   std::string histName = line.substr(0, equal);
   DeleteCommentsAndSpaces(histName);
   if (histName.empty())
      return "Error: no histName found";

   //Set the histExpression value
   std::size_t cutPos = line.find(kCutIntr, equal);
   std::string histExpression;
   if (cutPos == std::string::npos)
      histExpression = line.substr(equal + 1);
   else
      histExpression = line.substr(equal + 1, cutPos - equal - 1);
   DeleteCommentsAndSpaces(histExpression);
   if (histExpression.empty())
      return "Error: no expression found";

   // Set the histCut value
   std::string histCut;
   if (cutPos != std::string::npos) {
      histCut = line.substr(cutPos + kCutIntr.size());
      DeleteCommentsAndSpaces(histCut);
      if (histCut.empty())
         return "Error: missing cut expression after 'if'";
   }
   else
      histCut = "";

   // Set the map that contains the histName, histExpressions and histCut values
   auto check = fHists.insert(std::make_pair((const std::string&)histName,
                                             std::make_pair(histExpression, histCut)));

   // Check if there are histograms with the same name
   if (!check.second)
      return "Duplicate histogram name";
   return "";
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor for the case of command line parsing arguments. It sets the members
/// of the object.
///
/// \param[in] output - name of the output file
/// \param[in] inputFiles - name of the input .root files
/// \param[in] expressions - what is shown in the histograms
/// \param[in] treeName - name of the tree
/// \throws std::runtime_error in case of ill-formed expressions

TSimpleAnalysis::TSimpleAnalysis(const std::string& output,
                                 const std::vector<std::string>& inputFiles,
                                 const std::vector<std::string>& expressions,
                                 const std::string& treeName = ""):
   fInputFiles(inputFiles), fOutputFile(output), fTreeName(treeName)
{
   for (const std::string& expr: expressions) {
      std::string errMessage = HandleExpressionConfig(expr);
      if (!errMessage.empty())
         throw std::runtime_error(errMessage + " in " +  expr);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Extract the name of the tree from the first input file when the tree name
/// isn't in the configuration file. Returns the name of the tree.

static std::string ExtractTreeName(std::string& firstInputFile)
{
   std::string treeName = "";
   TFile inputFile (firstInputFile.c_str());

   // Loop over all the keys inside the first input file
   for (TObject* keyAsObj : *inputFile.GetListOfKeys()) {
      TKey* key = dynamic_cast<TKey*>(keyAsObj);
      TClass* clObj = TClass::GetClass(key->GetClassName());
      if (!clObj)
         continue;
      // If the key is releted to and object that inherits from TTree::Class we
      // set treeName with the name of this key if treeName is empty, otherwise
      // error occours
      if (clObj->InheritsFrom(TTree::Class())) {
         if (treeName.empty())
            treeName = key->GetName();
         else {
            ::Error("TSimpleAnalysis::Analyze", "Multiple trees inside %s", firstInputFile.c_str());
            return "";
         }
      }
   }
   // If treeName is yet empty, error occours
   if (treeName.empty()) {
      ::Error("TSimpleAnalysis::Analyze", "No tree inside %s", firstInputFile.c_str());
      return "";
   }
   return treeName;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute all the TChain::Draw() as configured and stores the output histograms.
/// Returns true if the analysis succeeds.

bool TSimpleAnalysis::Run()
{
   // Silence possible error message from TFile constructor if this is a tree name.
   int oldLevel = gErrorIgnoreLevel;
   gErrorIgnoreLevel = kFatal;
   // Disambiguate tree name from first input file:
   // just try to open it, if that works it's an input file.
   if (TFile::Open(fTreeName.c_str())) {
      fInputFiles.insert(fInputFiles.begin(), fTreeName);
      fTreeName.clear();
      fTreeName = ExtractTreeName(fInputFiles[0]);
      if (fTreeName.empty())
         return false;
   }
   gErrorIgnoreLevel = oldLevel;

   // Do the chain of the fInputFiles
   TChain chain(fTreeName.c_str());
   for (const std::string& inputfile: fInputFiles)
      chain.Add(inputfile.c_str());

   // Sanity check that we can open the first file
   int errValue = chain.LoadTree(0);
   if (errValue < 0) {
      ::Error("TSimpleAnalysis::Analyze",
              "The chain is not correctly set up, chain.LoadTree(0) returns %d", errValue);
      return false;
   }

   TFile ofile(fOutputFile.c_str(), "RECREATE");
   if (ofile.IsZombie()) {
      ::Error("TSimpleAnalysis::Analyze", "Impossible to create %s", fOutputFile.c_str());
      return false;
   }

   // Save the histograms into the output file
   for (const auto &histo : fHists) {
      const std::string& expr = histo.second.first;
      const std::string& histoName = histo.first;
      const std::string& cut = histo.second.second;
      chain.Draw((expr + ">>" + histoName).c_str(), cut.c_str(), "goff");
      TH1F *ptrHisto = (TH1F*)gDirectory->Get(histoName.c_str());
      if (!ptrHisto)
         return false;
      ptrHisto->Write();
   }
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns false if not a tree name, otherwise sets the name of the tree.
///
/// param[in] line - line read from the input file

bool TSimpleAnalysis::HandleInputFileNameConfig(const std::string& line)
{
   if (line.find("=") == std::string::npos) {
      fInputFiles.push_back(line);
      return true;
   }
   return false;  // It's an expression
}

////////////////////////////////////////////////////////////////////////////////
/// Skip subsequent empty lines read from fIn and returns the next not empty line.
///
/// param[in] numbLine number of the input file line

std::string TSimpleAnalysis::GetLine(int& numbLine)
{
   std::string notEmptyLine;

   do {
      getline(fIn, notEmptyLine);
      DeleteCommentsAndSpaces(notEmptyLine);
      numbLine++;
   } while (fIn && notEmptyLine.empty());

   return notEmptyLine;
}

////////////////////////////////////////////////////////////////////////////////
/// This function has the aim of setting the arguments read from the input file.

bool TSimpleAnalysis::Configure()
{
   int readingSection = kReadingOutput;
   std::string line;
   int numbLine = 0;

   // Error if the input file does not exist
   fIn.open(fInputName);
   if (!fIn) {
      ::Error("TSimpleAnalysis", "File %s not found", fInputName.c_str());
      return false;
   }

   while (!fIn.eof()) {
      line = GetLine(numbLine);
      if (line.empty())  // It can happen if fIn.eof()
         continue;
      std::string errMessage;

      switch (readingSection) {

         // Set the name of the output file
      case kReadingOutput:
         fOutputFile = line;
         readingSection++;
         break;

         // Set the name of the tree
      case kReadingTreeName:
         fTreeName = line;
         readingSection++;
         break;

         // Set the input files
      case kReadingInput:
         if (!HandleInputFileNameConfig(line)) {
            // Not an input file name; try to parse as an expression
            errMessage = HandleExpressionConfig(line);
            readingSection = kReadingExpressions;
         }
         break;

         // Set the expressions
      case kReadingExpressions:
         errMessage = HandleExpressionConfig(line);
         break;
      }

      // Report any errors if occour during the configuration proceedings
      if (!errMessage.empty()) {
         ::Error("TSimpleAnalysis::Configure", "%s in %s:%d", errMessage.c_str(),
                 fInputName.c_str(), numbLine);
         return false;
      }
   }  // while (!fIn.eof())
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Function that allows to create the TSimpleAnalysis object and execute its
/// Configure and Analyze functions.
///
/// param[in] configurationFile name of the input file used to create the TSimpleAnalysis object

bool RunSimpleAnalysis (const char* configurationFile) {
   TSimpleAnalysis obj(configurationFile);
   if (!obj.Configure())
      return false;
   if (!obj.Run())
      return false;
   return true;  // Return true only if Configure() and Run() functions were performed correctly
}
