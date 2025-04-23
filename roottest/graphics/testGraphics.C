// Author: Adrian Düsselberg CERN 07/2024

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TPaveLabel.h"
#include "TPaveText.h"
#include "TArrow.h"
#include "TLine.h"
#include "TString.h"
#include "TApplication.h"
#include "TWebCanvas.h"
#include "TLatex.h"
#include <TSystem.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <tuple>


//---------------------HELPER--------------------------------------------------------------
// Function to read file content into a string
std::string readFileToString(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filePath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::vector<std::string> splitString(const std::string &str, const char delimiter = '\n')
{
    auto lines = std::vector<std::string>{};
    lines.reserve(128);
    auto strAsStream = std::stringstream{str};

    for (std::string line; std::getline(strAsStream, line, delimiter);) {
        lines.emplace_back(line);
    }

    return lines;
}

std::string chopString(const std::string &str, size_t width = 256)
{
    return str.length() > width ? (str.substr(0, width) + " ...") : str;
}

void printSideBySide(const std::string &str1, const std::string &str2)
{
    auto lines1 = splitString(str1);
    auto lines2 = splitString(str2);
    for (auto [idx1, idx2, nDiff] = std::tuple<unsigned int, unsigned int, unsigned int>{0, 0, 0};
         idx1 < lines1.size() || idx2 < lines2.size();) {
        auto &line1 = lines1[idx1];
        auto &line2 = lines2[idx2];
        if (line1 != line2) {
           nDiff++;
           auto choppedLine1 = chopString(line1);
           auto choppedLine2 = chopString(line2);
           std::cout << idx1 << ", " << idx2 << ": " << choppedLine1 << "  <->  " << choppedLine2 << std::endl;
        }

        if (nDiff == 16) {
            std::cout << "*** Truncating " << std::endl;
            idx1 = lines1.size();
            idx2 = lines2.size();
        }

        if (idx1 < lines1.size())
           ++idx1;
        if (idx2 < lines2.size())
           ++idx2;
    }
}

//---------------------JSON----------------------------------------------------------------
std::string preprocessJSONContent(const std::string& jsonString) {
    std::string result = jsonString;
    // Remove fTsumwx key
    std::regex fTsumwxRegex(R"("fTsumwx" : \d+\.\d+,)");
    result = std::regex_replace(result, fTsumwxRegex, "");

    // Remove fTsumwx2 key
    std::regex fTsumwx2Regex(R"("fTsumwx2" : \d+\.\d+,)");
    result = std::regex_replace(result, fTsumwx2Regex, "");
    return result;
}

bool compare_json(const TString& jsonOutput, const std::string& ref_filename, const std::string& macroName) {
    // Read the reference JSON content from file
    std::ifstream refFile(ref_filename);
    if (!refFile.is_open()) {
        std::string jsonFilePath = "./json_ref/" + macroName + ".json";
        std::cerr << "Failed to open reference JSON file: " << ref_filename << "\nCreating new reference JSON: " << jsonFilePath << std::endl;

        std::ofstream jsonFile(jsonFilePath);
        if (!jsonFile.is_open()) {
            throw std::runtime_error("Error: Unable to open file for writing:" + jsonFilePath);
            return false;
        }
        jsonFile << jsonOutput.Data();
        return true;
    }

    std::stringstream refBuffer;
    refBuffer << refFile.rdbuf();

    // Remove specific lines from both JSON strings
    std::string produced_json = preprocessJSONContent(jsonOutput.Data());
    std::string reference_json = preprocessJSONContent(refBuffer.str());

    // Compare the created JSON to the reference JSON
    std::cerr << "Length of produced JSON after adjustments: " << jsonOutput.Length() << std::endl;
    std::cerr << "Length of reference JSON after adjustments: " << refBuffer.str().length() << std::endl;

    const auto areEqual = produced_json == reference_json;
    if (!areEqual) printSideBySide(produced_json, reference_json);

    return areEqual;
}

int test_json(TCanvas* c1, const std::string& macroName, const std::string& builddir) {
    // Create JSON output from canvas
    TString jsonOutput = TWebCanvas::CreateCanvasJSON(c1, 1, kFALSE);

    // Path to the reference and produced JSON file
    std::string ref_filename = "./json_ref/" + macroName + ".json";
    std::string jsonFilePath = builddir + "/json_pro/" + macroName + "_pro.json";

    // Save generated Json file
    std::ofstream jsonFile(jsonFilePath);
    if (jsonFile.is_open()) {
        jsonFile << jsonOutput.Data();
        jsonFile.close();
    } else {
        std::cerr << "Error: Unable to open file for writing" << std::endl;
        return 1;
    }

    // Compare the created JSON to the reference JSON
    if (compare_json(jsonOutput, ref_filename, macroName)) {
        std::cout << "JSON test passed for " << macroName << std::endl;
    } else {
        std::cerr << "JSON test failed for " << macroName << std::endl;
        return 1;
    }
    return 0;
}

//---------------------SVG-----------------------------------------------------------------
// Function to remove or normalize specific parts of the SVG content
std::string preprocessSVGContent(const std::string& svgContent) {
    std::string result = svgContent;

    // Remove <title>...</title> sections
    std::regex titleRegex(R"(<title>(.|\n)*?<\/title>)");
    result = std::regex_replace(result, titleRegex, "");

    // Remove <desc>...</desc> sections
    std::regex descRegex(R"(<desc>(.|\n)*?<\/desc>)");
    result = std::regex_replace(result, descRegex, "");

    return result;
}

// Function to compare two SVG files (old graphics)
bool compareSVGFiles(const std::string& filePath1, const std::string& filePath2) {
    try {
        std::string content1 = readFileToString(filePath1);
        std::string content2 = readFileToString(filePath2);

        content1 = preprocessSVGContent(content1);
        content2 = preprocessSVGContent(content2);

         // Compare the created SVG to the reference SVG
        std::cerr << "Length of produced SVG after adjustments: " << content2.length() << std::endl;
        std::cerr << "Length of reference SVG after adjustments: " << content1.length() << std::endl;

        const auto areEqual = content1 == content2;
        if (!areEqual) printSideBySide(content1, content2);

        return areEqual;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

int test_svg(TCanvas* c1, const std::string& macroName, const std::string& builddir) {
    // Paths to the reference and generated SVG files
    std::string refFilePath = "./old_svg_ref/" + macroName + ".svg";
    std::string genFilePath = builddir + "/old_svg_pro/" + macroName + "_pro.svg";

    // Check if the reference file exists
    FileStat_t fstat;
    if (1 == gSystem->GetPathInfo(refFilePath.c_str(), fstat)) {
        std::cout << "Reference file not found. Saving generated file as reference: " << macroName << std::endl;
        c1->SaveAs(refFilePath.c_str());
        return 1;
    } else {
        // Save the generated SVG file
        c1->SaveAs(genFilePath.c_str());
    }

    // Compare the generated SVG file with the reference SVG file
    if (compareSVGFiles(refFilePath, genFilePath)) {
        std::cout << "SVG test passed for " << macroName << std::endl;
    } else {
        std::cout << "SVG test failed for " << macroName << std::endl;
        return 1;
    }
    return 0;
}

//---------------------PDF-----------------------------------------------------------------
std::string preprocessPDFContent(const std::string& pdfString) {    
    std::string result = pdfString;
    // Regular expressions to match the metadata patterns
    std::regex creationDatePattern(R"(/CreationDate \(D:\d{14}.*?\))");
    std::regex modDatePattern(R"(/ModDate \(D:\d{14}[-+Z].*?\))");
    std::regex titlePattern(R"(/Title \(.*?\))");

    // Regular expressions to match the cross-reference table and trailer section
    std::regex xrefPattern(R"(xref[\s\S]*?EOF)");

    // Remove the matched patterns from the content
    result = std::regex_replace(result, creationDatePattern, "");
    result = std::regex_replace(result, modDatePattern, "");
    result = std::regex_replace(result, titlePattern, "");
    result = std::regex_replace(result, xrefPattern, "");
    return result;
}

// Function to compare two PDF files (old graphics)
bool comparePDFFiles(const std::string& filePath1, const std::string& filePath2) {
    try {
        std::string content1 = readFileToString(filePath1);
        std::string content2 = readFileToString(filePath2);

        content1 = preprocessPDFContent(content1);
        content2 = preprocessPDFContent(content2);

         // Compare the created JSON to the reference JSON
        std::cerr << "Length of produced PDF after adjustments: " << content2.length() << std::endl;
        std::cerr << "Length of reference PDF after adjustments: " << content1.length() << std::endl;

        const auto areEqual = content1 == content2;

        return areEqual;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

int test_pdf(TCanvas* c1, const std::string& macroName, const std::string& buildir) {
    // Paths to the reference and generated SVG files
    std::string refFilePath = "./pdf_ref/" + macroName + ".pdf";
    std::string genFilePath = buildir + "/pdf_pro/" + macroName + "_pro.pdf";

    // Check if the reference file exists
    FileStat_t fstat;
    if (1 == gSystem->GetPathInfo(refFilePath.c_str(), fstat)) {
        std::cout << "Reference file not found. Saving generated file as reference: " << macroName << std::endl;
        c1->SaveAs(refFilePath.c_str());
        return 1;
    } else {
        // Save the generated PDF file
        c1->SaveAs(genFilePath.c_str());
    }

    // Compare the generated PDF file with the reference PDF file
    if (comparePDFFiles(refFilePath, genFilePath)) {
        std::cout << "PDF test passed for " << macroName << std::endl;
    } else {
        std::cout << "PDF test failed for " << macroName << std::endl;
        return 1;
    }
    return 0;
}

// TEST ROOT MACRO -------------------------------------------------------------------------------
int testGraphics(const std::string& macroName, const std::string& test_type, const std::string& macro_folder, const std::string& builddir) {
    // Set paths
    std::string macroPath = macro_folder + "/" + macroName + ".C";
    gROOT->SetMacroPath("./macros");
    gROOT->SetStyle("ATLAS");

   // Set statistics box parameters explicitly
    gStyle->SetStatX(0.7);     // X position of stat box
    gStyle->SetStatY(0.8);     // Y position of stat box
    gStyle->SetStatW(0.2);     // Width of stat box
    gStyle->SetStatH(0.1);     // Height of stat box
    gStyle->SetStatFormat("6.4g"); // Number format in stat box
    gStyle->SetStatFont(42);            // Font type for stat box (Helvetica)
    gStyle->SetStatFontSize(0.025);     // Font size for stat box
    gStyle->SetStatColor(0);            // Background color of stat box
    gStyle->SetStatTextColor(1);        // Text color of stat box
    gStyle->SetStatBorderSize(1);       // Border size of stat box


    // Call the macro to generate the canvas
    std::string command = ".x " + macroPath;
    gROOT->ProcessLine(command.c_str());

    // Retrieve the current canvas from gPad
    auto c1 = dynamic_cast<TCanvas*>(gPad->GetCanvas());
    if (!c1) {
        std::cerr << "Error: No canvas found in gPad" << std::endl;
        return 1;
    }

    // Different tests for root graphics
    // a == invokes all tests
    // j === test the JSON creation
    // o === test the SVG creation in ROOT (old graphics with --web=off)
    // p == test the PDF creation in ROOT (old graphics with --web=off)

    if (test_type == "j") {
        return test_json(c1, macroName, builddir);
    }
    if (test_type == "o") {
        gROOT->SetWebDisplay("off");
        return test_svg(c1, macroName, builddir);
    }
    if (test_type == "p") {
        gROOT->SetWebDisplay("off");
        return test_pdf(c1, macroName, builddir);
    }
    std::cerr << "Unrecognised test type '" << test_type << "'" << std::endl;
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <macro_name> <test_type> <macro_folder> <builddir>" << std::endl;
        return 1;
    }

    std::string macroName = argv[1];
    std::string test_type = argv[2];
    std::string macro_folder = argv[3];
    std::string builddir = argv[4];
    return testGraphics(macroName, test_type, macro_folder, builddir);
}
