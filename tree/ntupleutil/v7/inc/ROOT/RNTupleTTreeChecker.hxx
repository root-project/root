/// \file RNTupleTTreeChecker.hxx
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

#ifndef ROOT_RNTupleTTreeChecker
#define ROOT_RNTupleTTreeChecker

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <TTree.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TBranch.h>
#include <TKey.h>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ROOT {
namespace Experimental {
namespace RNTupleTTreeCheckerCLI {
    struct CheckerConfig;
}

class RNTupleTTreeChecker {
public:
    void Compare(const RNTupleTTreeCheckerCLI::CheckerConfig &config);
private:
    bool RNTupleExists(const std::string &filename, const std::string &rntupleName);
    void CountEntries(const std::string &ttreeFile, const std::string &rntupleFile, const std::string &ttreeName, const std::string &rntupleName);
    void CountFields(const std::string &ttreeFile, const std::string &rntupleFile, const std::string &ttreeName, const std::string &rntupleName);
    void CompareFieldNames(const std::string &ttreeFile, const std::string &rntupleFile, const std::string &ttreeName, const std::string &rntupleName);
    void CompareFieldTypes(const std::string &ttreeFile, const std::string &rntupleFile, const std::string &ttreeName, const std::string &rntupleName);
    void PrintStyled(const std::string &text, const std::initializer_list<std::string> &styles, bool firstLineBreak = true, bool secondLineBreak = false);
    void PrintStyled(const std::string &text, const std::initializer_list<std::string> &styles, int width, bool firstLineBreak = true, bool secondLineBreak = false);

    static constexpr const char* GREEN = "\033[0;32m";
    static constexpr const char* RED = "\033[0;31m";
    static constexpr const char* YELLOW = "\033[0;33m";
    static constexpr const char* BLUE = "\033[0;34m";
    static constexpr const char* WHITE = "\033[0;37m";
    static constexpr const char* BLACK = "\033[0;30m";

    static constexpr const char* MEDIUM_BLUE = "\033[38;5;75m";
    static constexpr const char* DARKER_BLUE = "\033[38;5;18m";

    static constexpr const char* BG_WHITE = "\033[47m";
    static constexpr const char* BG_RED = "\033[41m";
    static constexpr const char* BG_GREEN = "\033[42m";

    static constexpr const char* RESET = "\033[0m";
    static constexpr const char* DEFAULT = "\033[39m";
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleTTreeChecker
