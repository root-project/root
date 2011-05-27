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
    fHistFileName(filename ? filename : ""), fMaxDepth((size_t) -1),
    fPruneLength(0) {
    // Create a history object, initialize from filename if the file
    // exists. Append new lines to filename taking into account the
    // maximal number of lines allowed by SetMaxDepth().
    if (filename) ReadFile(filename);
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
    }
  }

  void
  History::AppendToFile() {
    // Write last entry to hist file.
    // Prune if needed.
    if (fHistFileName.empty() || !fMaxDepth) return;

    // Count lines:
    size_t numLines = 0;
    std::string line;
    std::ifstream in(fHistFileName.c_str());
    while (std::getline(in, line))
      ++numLines;
    if (numLines >= fMaxDepth) {
      // Prune!
      size_t nPrune = fPruneLength;
      if (nPrune == (size_t)kPruneLengthDefault) {
         nPrune = (size_t)(fMaxDepth * 0.8);
      }

      // Don't write our lines - other processes might have
      // added their own.
      in.clear();
      in.seekg(0);
      std::string pruneFileName = fHistFileName + "_prune";
      std::ofstream out(pruneFileName.c_str());
      if (out) {
        if (in) {
          while (numLines >= nPrune && std::getline(in, line)) {
            // skip
            --numLines;
          }
          while (std::getline(in, line)) {
            out << line << '\n';
          }
        }
        out << fEntries.back() << '\n';
        in.close();
        out.close();
        ::unlink(fHistFileName.c_str());
        ::rename(pruneFileName.c_str(), fHistFileName.c_str());
      }
    } else {
      std::ofstream out(fHistFileName.c_str(), std::ios_base::app);
      out << fEntries.back() << '\n';
    }
  }
}
