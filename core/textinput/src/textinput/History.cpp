//===--- History.cpp - Previously Entered Lines -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for setting and retrieving previously
//  entered input, with a persistent backend (i.e. a history file).
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#include "textinput/History.h"
#include <iostream>

namespace textinput {
  History::History(const char* filename):
    fHistFileLines(0), fMaxDepth((size_t) -1), fPruneLength(0) {
    // Create a history object, initialize from filename if the file
    // exists. Append new lines to filename taking into account the
    // maximal number of lines allowed by SetMaxDepth().
    if (filename) {
      ReadFile(filename);

      // Open output.
      fHistFile.open(filename, std::ios_base::app);
      if (!fHistFile) {
        std::cerr << "textinput::History(): cannot open file \"" << filename
          << "\" for writing!\n";
      }
      return;
    }

  }
  
  History::~History() {}

  void
  History::AddLine(const std::string& line) {
    // Add a line to entries and file.
    fEntries.push_back(line);
    AppendToFile();
  }

  void
  History::ReadFile(const char* FileName) {
    // Inject all lines of FileName.
    // Intentionally ignore fMaxDepth
    std::ifstream InHistFile(FileName);
    if (!InHistFile) return;
    std::string line;
    while (std::getline(InHistFile, line)) {
      while (!line.empty()) {
        size_t len = line.length();
        char c = line[len - 1];
        if (c != '\n' && c != '\r') break;
        line.erase(len - 1);
      }
      fEntries.push_back(line);
      ++fHistFileLines;
    }
  }

  void
  History::AppendToFile() {
    // Write last entry to hist file.
    // Prune if needed.
    if (!fHistFile || ! fMaxDepth) return;
    if (fHistFileLines >= fMaxDepth) {
      // Prune!
      fHistFile.seekp(0);
      size_t nPrune = fPruneLength;
      if (nPrune == (size_t)kPruneLengthDefault) {
        nPrune = fMaxDepth * 0.8;
      }
      for (size_t Idx = fEntries.size() - nPrune,
        E = fEntries.size(); Idx < E; ++Idx) {
        fHistFile << fEntries[Idx] << '\n';
      }
    }
    fHistFile << fEntries.back() << '\n';
  }
}
