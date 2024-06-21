/// \file RNTupleTTreeCheckerCLI.cxx
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

#include "ROOT/RNTupleTTreeCheckerCLI.hxx"
#include <iostream>

namespace {
const auto usageText = "Usage:\n"
                       " rntuplettreechecker (--ttree|-t) <input_ttree_file>\n"
                       "                     (--rntuple|-r) <input_rntuple_file>\n"
                       "                     (--treename|-tn) <ttree_name>\n"
                       "                     (--rntuplename|-rn) <rntuple_name>\n"
                       " rntuplettreechecker [--help|-h]\n\n";
}

namespace ROOT {
namespace Experimental {
namespace RNTupleTTreeCheckerCLI {

CheckerConfig ParseArgs(const std::vector<std::string> &args) {

    const auto argsProvided = args.size() >= 2;
    const auto helpUsed = argsProvided && (args[1] == "--help" || args[1] == "-h");

    if (!argsProvided || helpUsed) {
        std::cout << usageText;
        if (!argsProvided)
            std::cout << " Use --help or -h for usage help.";
        std::cout << std::endl;
        return {};
    }

    CheckerConfig config;

    for (size_t i = 1; i < args.size(); ++i) {
        const auto &arg = args[i];

        if (arg == "--ttree" || arg == "-t") {
            if (++i < args.size()) config.fTTreeFile = args[i];
        } else if (arg == "--rntuple" || arg == "-r") {
            if (++i < args.size()) config.fRNTupleFile = args[i];
        } else if (arg == "--treename" || arg == "-tn") {
            if (++i < args.size()) config.fTTreeName = args[i];
        } else if (arg == "--rntuplename" || arg == "-rn") {
            if (++i < args.size()) config.fRNTupleName = args[i];
        } else {
            std::cerr << "Unknown argument '" << arg << "'\n" << usageText << "\n";
            return {};
        }
    }

    if (config.fTTreeFile.empty()) {
        std::cerr << "Please provide the name of the TTree file to compare.\n\n" << usageText << "\n";
        return {};
    } else if (config.fRNTupleFile.empty()) {
        std::cerr << "Please provide the name of the RNTuple file to compare.\n\n" << usageText << "\n";
        return {};
    } else if (config.fTTreeName.empty()) {
        std::cerr << "Please provide the name of the TTree to compare.\n\n" << usageText << "\n";
        return {};
    } else if (config.fRNTupleName.empty()) {
        std::cerr << "Please provide the name of the RNTuple to compare.\n\n" << usageText << "\n";
        return {};
    }

    config.fShouldRun = true;
    return config;
}

CheckerConfig ParseArgs(int argc, char **argv) {
    std::vector<std::string> args;
    args.reserve(argc);

    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    return ParseArgs(args);
}

void RunChecker(const CheckerConfig &config) {
    RNTupleTTreeChecker checker;
    checker.Compare(config);
}

} // namespace RNTupleTTreeCheckerCLI
} // namespace Experimental
} // namespace ROOT

int main(int argc, char **argv) {
    auto config = ROOT::Experimental::RNTupleTTreeCheckerCLI::ParseArgs(argc, argv);

    if (!config.fShouldRun) {
        return 1;
    }

    ROOT::Experimental::RNTupleTTreeCheckerCLI::RunChecker(config);

    return 0;
}
