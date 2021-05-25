// @(#)root/treeplayer:$Id$
// Author: Luca Giommi   22/08/16

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSimpleAnalysis
#define ROOT_TSimpleAnalysis

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSimpleAnalysis                                                      //
//                                                                      //
// A TSimpleAnalysis object creates histograms from a TChain. These     //
// histograms are stored to an output file. The histogrammed            //
// (TTreeFormula) expressions, their cuts, the input and output files   //
// are configured through a simple config file that allows comments     //
// starting with '#'.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include <string>
#include <fstream>
#include <vector>
#include <map>

class TSimpleAnalysis {

private:
   std::string              fConfigFile; ///< Name of the configuration file
   std::vector<std::string> fInputFiles; ///< .root input files
   std::string              fOutputFile; ///< Output file in which are stored the histograms
   std::string              fTreeName;   ///< Name of the input tree
   std::ifstream            fIn;         ///< Stream for the input file

   /// The map contains in the first part the names of the histograms written in the output file, in the
   /// second part the pair of what is shown in the histograms and the cut applied on the variables
   std::map<std::string, std::pair<std::string, std::string>> fHists;

   /// The elements of the enumeration refer to the different types of elements
   /// that are in the input file
   enum EReadingWhat {
      kReadingOutput,     ///< Reading the name of the output file
      kReadingTreeName,   ///< Reading the name of the tree
      kReadingInput,      ///< Reading the name of the .root input files
      kReadingExpressions ///< Reading the expressions
   };

   std::string HandleExpressionConfig(const std::string& line);
   std::string GetLine(int& numbLine);
   bool HandleInputFileNameConfig(const std::string& line);
   bool SetTreeName();


public:
   TSimpleAnalysis(const std::string& file): fConfigFile (file) {}
   TSimpleAnalysis(const std::string& output, const std::vector<std::string>& inputFiles,
                   const std::vector<std::string>& expressions, const std::string& treeName);
   bool Run();
   bool Configure();

};

bool RunSimpleAnalysis(const char* configurationFile);

#endif
