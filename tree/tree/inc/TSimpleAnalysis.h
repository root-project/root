
#ifndef ROOT_TSimpleAnalysis
#define ROOT_TSimpleAnalysis


#include "TFile.h"
#include "TCanvas.h"
#include <vector>
#include "TChain.h"
#include "TH1.h"
#include <string>
#include <iostream>
#include <fstream>
#include "TError.h"


class TSimpleAnalysis {
public:

   std::string fInputName;
   std::vector<std::string> fInputFiles;
   std::vector<std::string> fExpressions;
   std::vector<std::string> fHNames;
   std::vector<std::string> fCut;
   std::string fOutputFile;
   std::string fTreeName;
   ifstream in;
   enum EReadingWhat {kReadingOutput,kReadingInput,kReadingTreeName,
                      kReadingExpressions,kEndOfFile};
   const std::string kCutIntr="if";
   Int_t fCounter=0;


   TSimpleAnalysis(const std::string& kFile) {
      fInputName = kFile;
      in.open(kFile);
      if(!in) {
         ::Error("TSimpleAnalysis","File %s not found",kFile.c_str());
         throw 1;
      }
   }

   TSimpleAnalysis(const std::string& output, std::vector<std::string> inputFiles,
                 const std::string& name, std::vector<std::string> expressions) {
      cout<<"sono entrato nel costruttore parametrico"<<endl;
      fOutputFile=output;
      fTreeName=name;
      for (std::string input: inputFiles)
         fInputFiles.push_back(input);
      for (std::string expr: expressions) {
         cout<<expr<<endl;
         std::size_t equal=expr.find("="); // what if no '='
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

         cout<<expr<<endl;
      }
   }

   virtual ~TSimpleAnalysis(){}

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


   Bool_t Analysis() {
      cout<<"entro nella funzione"<<endl;
      cout<<fTreeName.c_str()<<endl;
      TChain chain(fTreeName.c_str());
      for (const std::string& inputfile: fInputFiles) {
         chain.Add(inputfile.c_str());
         cout<<inputfile.c_str()<<endl;
      }
      cout<<fOutputFile.c_str()<<endl;
      TFile ofile(fOutputFile.c_str(),"RECREATE");

      CheckHNames(fHNames);

      for (unsigned i=0; i<fExpressions.size(); i++) {
         chain.Draw((fExpressions[i] + ">>" + fHNames[i]).c_str(),fCut[i].c_str(),"goff");
         TH1F *histo = (TH1F*)gDirectory->Get(fHNames[i].c_str());
         histo->Write();
      }

      return true;
   }


   // Return false if not a tree name
   Bool_t HandleTreeNameConfig(const std::string& line) {
      if (line.find("=") != std::string::npos)
         return false;
      else return true;
   }

   void HandlefExpressionConfig(std::string& line, int& numbLine) {
      std::size_t equal=line.find("="); // what if no '='
      if (equal == std::string::npos) {
         ::Error("TSimpleAnalysis::HandlefExpressionConfig",
                 "Missing '=' in fExpressions in %s:%d",fInputName.c_str(), numbLine);
         throw 1;
      }
      cout<<endl;
      cout<<line<<endl;
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

   Bool_t DeleteSpaces (std::string& line) {
      line.erase(std::remove(line.begin(),line.end(),' '),line.end());
      return 1;
   }

   std::string SkipSubsequentEmptyLines(Int_t& numbLine) {
      std::string notEmptyLine;

      while (getline(in,notEmptyLine) && DeleteSpaces(notEmptyLine)
             && (notEmptyLine.empty() || notEmptyLine.find("//") == 0))
         {numbLine++;}

      numbLine++;
      return notEmptyLine;
   }


   Bool_t HandleLines(std::string& line, Int_t& readingSection, Int_t& numbLine) {

      if (line.empty() || line.find_first_not_of(" ") == std::string::npos) {
         cout<<readingSection<<endl;
         if (readingSection == 0 && fCounter == 0)
            return 1;
         readingSection++; // > kReadingFfExpressions?
         cout<<readingSection<<endl;
         line = SkipSubsequentEmptyLines(numbLine);
      }

      line.erase(std::remove(line.begin(),line.end(),' '),line.end());
      std::size_t comment = line.find("//");
      if (comment==0)
         return 1;
      cout<<"comment "<<comment<<endl;
      if ((comment != 0) || (comment != std::string::npos)) {
         cout<<line<<endl;
         if (readingSection == 0)
            fCounter++;
      }
         line = line.substr(0,comment);
      cout<<line<<endl;
      return 0;


   }



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
         cout<<line<<endl;



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


      //      if (readingSection != kEndOfFile)
      // Houston-we-have-a-problem!
      //checkSettings(); // do we have a file name? do we have an expression?

   }


ClassDef(TSimpleAnalysis,0)

};

#endif
