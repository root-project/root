// @(#)root/tree:$Id$
// Author: Luca Giommi   22/08/16

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSimpleAnalysis
#define ROOT_TSimpleAnalysis

/** \class TSympleAnalysis
A TSimpleAnalysis object permit to, given an input file or through
command line, create an .root file in which are saved histograms
that the user want to create.
*/

#ifndef ROOT_TFile
#include "TFile.h"
#endif

#ifndef ROOT_TCanvas
#include "TCanvas.h"
#endif

#ifndef ROOT_TChain
#include "TChain.h"
#endif

#ifndef ROOT_TH1
#include "TH1.h"
#endif

#ifndef ROOT_TError
#include "TError.h"
#endif

#include <string>
#include <iostream>
#include <fstream>
#include <vector>




class TSimpleAnalysis {
private:

   std::string fInputName;   ///< Name of the input file
   std::vector<std::string> fInputFiles;   ///< .root input files
   std::vector<std::string> fExpressions;   ///< What is showed by the histograms inside the output file
   std::vector<std::string> fHNames;   ///< Names of the histograms
   std::vector<std::string> fCut;   ///< Cuts added to the hisograms
   std::string fOutputFile;   ///< Output file in which are stored the histograms
   std::string fTreeName;   ///< Name of the input tree
   ifstream in;   ///< Stream for the input file

   //the elements of the enum refer to the different kinds of elements that are
   //in the input file

   enum EReadingWhat {kReadingOutput,kReadingInput,kReadingTreeName,
                      kReadingExpressions,kEndOfFile};
   static const std::string kCutIntr;   ///< The string that represents the starter point of the cut expresson
   Int_t fCounter=0;   ///< Counter usefull for the reading file

public:
   ////////////////////////////////////////////////////////////////////////////////
   ///Constructor for the case with the input file
   ///
   /// \param[in] kFile name of the input file that has to be read

   TSimpleAnalysis(const std::string& kFile) {
      fInputName = kFile;
      in.open(kFile);
      if(!in) {
         ::Error("TSimpleAnalysis","File %s not found",kFile.c_str());
         throw 1;
      }
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Constructor for the case of command line parsing arguments
   ///
   /// \param[in] output name of the output file
   /// \param[in] inputFiles name of the input .root files
   /// \param[in] name name of the tree
   /// \param[in] expressions what is showed in the histograms

   TSimpleAnalysis(const std::string& output, std::vector<std::string> inputFiles,
                 const std::string& name, std::vector<std::string> expressions) {
      fOutputFile=output;
      fTreeName=name;
      for (std::string input: inputFiles)
         fInputFiles.push_back(input);
      for (std::string expr: expressions) {
         std::size_t equal=expr.find("=");
         if (equal == std::string::npos) {
            ::Error("TSimpleAnalysis",
                    "Missing '=' in fExpressions in %s",expr.c_str());
            throw 1;
         }
         std::size_t cutPos=expr.find(kCutIntr, equal);
         std::string substring=expr.substr(0,equal);
         if (substring.empty()) {
            ::Error("TSimpleAnalysis",
                 "No hname found in %s",expr.c_str());
            throw 2;
         }
         fHNames.push_back(substring);
         substring=expr.substr(equal+1,cutPos-equal-1);
         if (substring.empty()) {
            ::Error("TSimpleAnalysis",
                    "No expression found in %s",expr.c_str());
            throw 3;
         }
         fExpressions.push_back(substring);
         if (cutPos == std::string::npos) {
            fCut.push_back("");
         } else {
            fCut.push_back(expr.substr(cutPos+kCutIntr.size()));
         }
      }
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Virtual default denstructor

   virtual ~TSimpleAnalysis(){}

   ////////////////////////////////////////////////////////////////////////////////
   /// Tell us if the name of some histograms that will be created are the same
   ///
   /// param[in] name name of the histograms

   void CheckHNames(std::vector<std::string> name) {
      Int_t err=0;
      for (unsigned i=0; i<name.size()-1; i++)
         for (unsigned j=i+1; j<name.size(); j++) {
            if (name[i] == name[j]) {
               ::Error("TSimpleAnalysis::check_name",
                       "Multiple %s in the names of histograms (position %d and %d)",
                       name[i].c_str(), i, j);
               err++;
            }
         }
      if (err != 0)
         throw 1;
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Return true if the input arguments create the output file correctly

   Bool_t Analysis() {
      TChain chain(fTreeName.c_str());
      for (const std::string& inputfile: fInputFiles)
         chain.Add(inputfile.c_str());
      TFile ofile(fOutputFile.c_str(),"RECREATE");

      CheckHNames(fHNames);

      for (unsigned i=0; i<fExpressions.size(); i++) {
         chain.Draw((fExpressions[i] + ">>" + fHNames[i]).c_str(),fCut[i].c_str(),"goff");
         TH1F *histo = (TH1F*)gDirectory->Get(fHNames[i].c_str());
         histo->Write();
      }

      return true;
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Return false if not a tree name
   ///
   /// param[in] line line read from the input file

   Bool_t HandleTreeNameConfig(const std::string& line) {
      if (line.find("=") != std::string::npos)
         return false;
      else return true;
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// It handles the expression lines of the input file in order to pass the
   /// elements to the members of the object
   ///
   /// param[in] line line read from the input file
   /// param[in] numbLine number of the input file line

   void HandlefExpressionConfig(std::string& line, int& numbLine) {
      std::size_t equal=line.find("=");
      if (equal == std::string::npos) {
         ::Error("TSimpleAnalysis::HandlefExpressionConfig",
                 "Missing '=' in fExpressions in %s:%d",fInputName.c_str(), numbLine);
         throw 1;
      }
      std::size_t cutPos=line.find(kCutIntr, equal);
      std::string substring=line.substr(0,equal);
      if (substring.empty()) {
         ::Error("TSimpleAnalysis::HandlefExpressionConfig",
                 "No hname found in %s:%d",fInputName.c_str(), numbLine);
         throw 2;
      }
      fHNames.push_back(substring);
      substring=line.substr(equal+1,cutPos-equal-1);
      if (substring.empty()) {
         ::Error("TSimpleAnalysis::HandlefExpressionConfig",
                 "No expression found in %s:%d",fInputName.c_str(), numbLine);
         throw 3;
      }
      fExpressions.push_back(substring);
      if (cutPos == std::string::npos) {
            fCut.push_back("");
      } else {
         fCut.push_back(line.substr(cutPos+kCutIntr.size()));
      }
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Delete white spaces in a string
   ///
   /// param[in] line line read from the input file

   Bool_t DeleteSpaces (std::string& line) {
      line.erase(std::remove(line.begin(),line.end(),' '),line.end());
      return 1;
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Skip subsequent empty lines in a string and returns the number of the
   /// current line of the input file
   ///
   /// param[in] numbLine number of the input file line

   std::string SkipSubsequentEmptyLines(Int_t& numbLine) {
      std::string notEmptyLine;

      while (getline(in,notEmptyLine) && DeleteSpaces(notEmptyLine)
             && (notEmptyLine.empty() || notEmptyLine.find("//") == 0))
         {numbLine++;}

      numbLine++;
      return notEmptyLine;
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// After the eventual skip of subsequent empty lines, returns true if the
   /// line is a comment
   ///
   /// param[in] line line read from the input file
   /// param[in] readingSection current section of the read file
   /// param[in] numbLine number of the input file line

   Bool_t HandleLines(std::string& line, Int_t& readingSection, Int_t& numbLine) {

      if (line.empty() || line.find_first_not_of(" ") == std::string::npos) {
         if (readingSection == 0 && fCounter == 0)
            return 1;
         readingSection++;
         line = SkipSubsequentEmptyLines(numbLine);
      }

      line.erase(std::remove(line.begin(),line.end(),' '),line.end());
      std::size_t comment = line.find("//");
      if (comment==0)
         return 1;
      if (((comment != 0) || (comment != std::string::npos)) && readingSection == 0)
            fCounter++;
         line = line.substr(0,comment);
      return 0;


   }

   ////////////////////////////////////////////////////////////////////////////////
   /// This function has the aim of setting the arguments read from the input file

   void Settings() {

      int readingSection = kReadingOutput;
      std::string line;
      int numbLine = 0;

      while(!in.eof()) {

         getline (in,line);
         numbLine++;
         if (HandleLines(line,readingSection,numbLine)==1)
            continue;
         line.erase(std::remove(line.begin(),line.end(),' '),line.end());

         switch (readingSection) {
         case kReadingOutput:
            fOutputFile = line;
            break;

         case kReadingInput:
            fInputFiles.push_back(line);
            break;

         case kReadingTreeName:

            if (HandleTreeNameConfig(line) == true) {
               fTreeName = line;
            }
            else {
               readingSection = kReadingExpressions;
               HandlefExpressionConfig(line,numbLine);
            }
            break;

         case kReadingExpressions:
            HandlefExpressionConfig(line,numbLine);
            break;

         case kEndOfFile: break;
         }
      }
   }


   ClassDef(TSimpleAnalysis,0)

};

const std::string TSimpleAnalysis::kCutIntr="if";

#endif
