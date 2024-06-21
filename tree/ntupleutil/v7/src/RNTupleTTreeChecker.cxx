/// \file RNTupleTTreeChecker.cxx
/// \ingroup NTuple ROOT7
/// \author Ida Caspary <ida.friederike.caspary@cern.ch>
/// \date 2024-10-14
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RNTupleTTreeChecker.hxx"
#include "ROOT/RNTupleTTreeCheckerCLI.hxx"
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <iomanip>
#include <TTree.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TBranch.h>
#include <TKey.h>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

void RNTupleTTreeChecker::PrintStyled(const std::string &text, const std::initializer_list<std::string> &styles, bool firstLineBreak, bool secondLineBreak) {
    for (const auto &style: styles) {
        std::cout << style;
    }
    std::cout << text << RESET;
    if (firstLineBreak) {
        std::cout << std::endl;
    }
    if (secondLineBreak) {
        std::cout << std::endl;
    }
}

void RNTupleTTreeChecker::PrintStyled(const std::string &text, const std::initializer_list<std::string> &styles, int width, bool firstLineBreak, bool secondLineBreak) {
    for (const auto &style: styles) {
        std::cout << style;
    }
    std::cout << std::setw(width) << std::left << text << RESET;
    if (firstLineBreak) {
        std::cout << std::endl;
    }
    if (secondLineBreak) {
        std::cout << std::endl;
    }
}

bool RNTupleTTreeChecker::RNTupleExists(const std::string &filename, const std::string &rntupleName) {
    try {
        const auto file = TFile::Open(filename.c_str());
        if (!file || file->IsZombie()) {
            PrintStyled("Cannot open file: " + filename, {RED});
            return false;
        }

        auto keys = file->GetListOfKeys();
        for (int i = 0; i < keys->GetEntries(); ++i) {
            auto key = dynamic_cast<TKey *>(keys->At(i));
            if (std::string(key->GetClassName()) == "ROOT::Experimental::RNTuple" &&
                std::string(key->GetName()) == rntupleName) {
                return true;
            }
        }

        file->Close();
    } catch (const std::exception &e) {
        PrintStyled("Error checking RNTuple existence: " + std::string(e.what()), {RED});
    }
    return false;
}

void RNTupleTTreeChecker::CountEntries(const std::string &ttreeFile, const std::string &rntupleFile, const std::string &ttreeName, const std::string &rntupleName) {
    PrintStyled("\n*** Entry Count ***", {MEDIUM_BLUE});

    const auto tfile = std::unique_ptr<TFile>(TFile::Open(ttreeFile.c_str()));
    const auto ttree = dynamic_cast<TTree *>(tfile->Get(ttreeName.c_str()));
    if (!ttree) {
        PrintStyled("Cannot find TTree: " + ttreeName + " in file: " + ttreeFile, {RED});
        return;
    }

    if (!RNTupleExists(rntupleFile, rntupleName)) {
        PrintStyled("Cannot find RNTuple: " + rntupleName + " in file: " + rntupleFile, {RED});
        return;
    }

    try {
        auto rntpl = RNTupleReader::Open(rntupleName, rntupleFile);
        if (!rntpl) {
            throw std::runtime_error("can't find rntuple");
        }

        const bool compareCount = (static_cast<int>(ttree->GetEntries()) == static_cast<int>(rntpl->GetNEntries()));
        if (compareCount) {
            PrintStyled("Number of entries: ", {DEFAULT}, false);
            PrintStyled(std::to_string(ttree->GetEntries()), {GREEN});
        } else {
            PrintStyled("Number of entries in TTree: ", {DEFAULT}, false);
            PrintStyled(std::to_string(ttree->GetEntries()), {RED});
            PrintStyled("Number of entries in RNTuple: ", {DEFAULT}, false);
            PrintStyled(std::to_string(rntpl->GetNEntries()), {RED});
        }

        std::ostringstream oss;
        oss << std::boolalpha << compareCount;
        PrintStyled("TTree and RNTuple have the same entry count: ", {DEFAULT}, false);
        if (compareCount) {
            PrintStyled(oss.str(), {BLACK, BG_GREEN}, true, true);
        } else {
            PrintStyled(oss.str(), {BLACK, BG_RED}, true, true);
        }
    } catch (const RException &e) {
        PrintStyled("Error opening RNTuple: " + std::string(e.what()), {RED});
    }
}

void RNTupleTTreeChecker::CountFields(const std::string &ttreeFile, const std::string &rntupleFile, const std::string &ttreeName, const std::string &rntupleName) {
    PrintStyled("*** Field Count ***", {MEDIUM_BLUE});

    const auto tfile = std::unique_ptr<TFile>(TFile::Open(ttreeFile.c_str()));
    const auto ttree = dynamic_cast<TTree *>(tfile->Get(ttreeName.c_str()));
    if (!ttree) {
        PrintStyled("Cannot find TTree: " + ttreeName + " in file: " + ttreeFile, {RED});
        return;
    }

    try {
        const auto rfile = std::unique_ptr<TFile>(TFile::Open(rntupleFile.c_str()));
        const auto rntpl = rfile->Get<RNTuple>(rntupleName.c_str());
        if (!rntpl) {
            throw std::runtime_error("can't find rntuple");
        }

        const auto reader = RNTupleReader::Open(rntpl);
        if (!reader) {
            PrintStyled("Failed to open RNTupleReader.", {RED});
            return;
        }

        const auto ttreeBranches = ttree->GetListOfBranches();
        const int ttreeFieldCount = ttreeBranches->GetEntries();

        auto fieldIterable = reader->GetDescriptor().GetFieldIterable(reader->GetDescriptor().GetFieldZeroId());
        const auto rntupleFieldCount = std::distance(fieldIterable.begin(), fieldIterable.end());

        const bool compareCount = ttreeFieldCount == rntupleFieldCount;

        if (compareCount) {
            PrintStyled("Number of fields:  ", {DEFAULT}, false);
            PrintStyled(std::to_string(ttreeFieldCount), {GREEN});
        } else {
            std::cout << "Number of fields in TTree: " << ttreeFieldCount << std::endl;
            std::cout << "Number of fields in RNTuple: " << rntupleFieldCount << "\n" << std::endl;
        }

        std::ostringstream oss;
        oss << std::boolalpha << compareCount;
        PrintStyled("TTree and RNTuple have the same field count: ", {DEFAULT}, false);
        if (compareCount) {
            PrintStyled(oss.str(), {BLACK, BG_GREEN}, true, true);
        } else {
            PrintStyled(oss.str(), {BLACK, BG_RED}, true, true);
        }
    } catch (const RException &e) {
        PrintStyled("Error opening RNTuple: " + std::string(e.what()), {RED});
    }
}

void RNTupleTTreeChecker::CompareFieldNames(const std::string &ttreeFile, const std::string &rntupleFile, const std::string &ttreeName, const std::string &rntupleName) {
    PrintStyled("*** Field Names ***", {MEDIUM_BLUE});

    const auto tfile = std::unique_ptr<TFile>(TFile::Open(ttreeFile.c_str()));
    const auto ttree = dynamic_cast<TTree *>(tfile->Get(ttreeName.c_str()));

    const auto rfile = std::unique_ptr<TFile>(TFile::Open(rntupleFile.c_str()));
    const auto ntpl = rfile->Get<RNTuple>(rntupleName.c_str());

    const auto reader = RNTupleReader::Open(ntpl);

    const auto ttreeBranches = ttree->GetListOfBranches();
    const int ttreeFieldCount = ttreeBranches->GetEntries();

    int width = 13;

    bool compareFields = true;
    PrintStyled("\nTTree Field  |  RNTuple Field\n-----------------------------", {DEFAULT});

    for (int i = 0; i < ttreeFieldCount; ++i) {
        const auto branch = dynamic_cast<TBranch *>(ttreeBranches->At(i));
        PrintStyled(std::string(branch->GetName()), {DEFAULT}, width, false);
        PrintStyled(std::string("|  "), {DEFAULT}, false);

        try {
            auto rntuplefield = reader->GetDescriptor().GetFieldDescriptor(i).GetFieldName();

            if (branch->GetName() == rntuplefield) {
                PrintStyled(rntuplefield, {DEFAULT}, width);
            } else {
                PrintStyled(rntuplefield, {RED}, width);
                compareFields = false;
            }
        } catch (const std::exception &e) {
            PrintStyled("No matching field", {RED});
            compareFields = false;
        }
    }

    std::ostringstream oss;
    oss << std::boolalpha << compareFields;
    PrintStyled("\nThe fields have the same names: ", {DEFAULT}, false);
    if (compareFields) {
        PrintStyled(oss.str(), {BLACK, BG_GREEN}, true, true);
    } else {
        PrintStyled(oss.str(), {BLACK, BG_RED}, true, true);
    }
}

void RNTupleTTreeChecker::CompareFieldTypes(const std::string &ttreeFile, const std::string &rntupleFile, const std::string &ttreeName, const std::string &rntupleName) {
    PrintStyled("*** Field Types ***", {MEDIUM_BLUE});

    const auto tfile = std::unique_ptr<TFile>(TFile::Open(ttreeFile.c_str()));
    const auto ttree = dynamic_cast<TTree *>(tfile->Get(ttreeName.c_str()));

    const auto rfile = std::unique_ptr<TFile>(TFile::Open(rntupleFile.c_str()));
    const auto ntpl = rfile->Get<RNTuple>(rntupleName.c_str());

    const auto reader = RNTupleReader::Open(ntpl);

    const auto ttreeBranches = ttree->GetListOfBranches();
    const int ttreeFieldCount = ttreeBranches->GetEntries();

    std::unordered_map<std::string, std::string> typeMap = {
        {"Int_t", "int"},
        {"std::int32_t", "int"},
        {"Float_t", "float"},
        {"float", "float"},
        {"Double_t", "double"},
        {"double", "double"},
        {"Bool_t", "bool"},
        {"bool", "bool"}
    };

    bool compareTypes = true;
    int width = 13;
    PrintStyled("\nType - TTree |  Type - Field    Field No\n-----------------------------", {DEFAULT});
    for (int i = 0; i < ttreeFieldCount; ++i) {
        const auto branch = dynamic_cast<TBranch *>(ttreeBranches->At(i));
        auto ttreeFieldType = typeMap[std::string(branch->GetLeaf(branch->GetName())->GetTypeName())];
        PrintStyled(ttreeFieldType, {DEFAULT}, width, false);
        PrintStyled(std::string("|  "), {DEFAULT}, false);

        try {
            auto rntupleFieldType = typeMap[std::string(reader->GetDescriptor().GetFieldDescriptor(i).GetTypeName())];
            PrintStyled(rntupleFieldType, {DEFAULT}, width, false);

            PrintStyled("   " + std::to_string(i), {DEFAULT}, width, false);

            if (ttreeFieldType != rntupleFieldType) {
                compareTypes = false;
                PrintStyled("   type mismatch   ", {WHITE, BG_RED}, false);
            }
            std::cout << std::endl;
        } catch (const std::exception &e) {
            PrintStyled("No matching field", {RED});
            compareTypes = false;
        }
    }

    std::ostringstream oss;
    oss << std::boolalpha << compareTypes;
    PrintStyled("\nThe fields have the same types: ", {DEFAULT}, false);
    if (compareTypes) {
        PrintStyled(oss.str(), {BLACK, BG_GREEN}, true, true);
    } else {
        PrintStyled(oss.str(), {BLACK, BG_RED}, true, true);
    }
}

void RNTupleTTreeChecker::Compare(const RNTupleTTreeCheckerCLI::CheckerConfig &config) {
    PrintStyled("\nComparing TTree '" + config.fTTreeName + "' <> RNTuple '" + config.fRNTupleName + "'", {DARKER_BLUE, BG_WHITE});

    CountEntries(config.fTTreeFile, config.fRNTupleFile, config.fTTreeName, config.fRNTupleName);
    CountFields(config.fTTreeFile, config.fRNTupleFile, config.fTTreeName, config.fRNTupleName);
    CompareFieldNames(config.fTTreeFile, config.fRNTupleFile, config.fTTreeName, config.fRNTupleName);
    CompareFieldTypes(config.fTTreeFile, config.fRNTupleFile, config.fTTreeName, config.fRNTupleName);
}

} // namespace Experimental
} // namespace ROOT
