/// \file ROOT/StringUtils.hxx
/// \ingroup Base StdExt
/// \author Jonas Rembser <jonas.rembser@cern.ch>
/// \date 2021-08-09

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <fstream>
#include "ROOT/StringUtils.hxx"

namespace ROOT {

/// Splits a string at each character in delims.
/// The behavior mimics `str.split` from Python,
/// \param[in] str String to tokenise.
/// \param[in] delims One or more delimiters used to split the string.
/// \param[in] skipEmpty Strip empty strings from the output.
std::vector<std::string> Split(std::string_view str, std::string_view delims, bool skipEmpty /* =false */)
{
   std::vector<std::string> out;

   std::size_t beg = 0;
   std::size_t end = 0;
   while ((end = str.find_first_of(delims, beg)) != std::string::npos) {
      if (!skipEmpty || end > beg)
         out.emplace_back(str.substr(beg, end - beg));
      beg = end + 1;
   }
   if (!skipEmpty || str.size() > beg)
      out.emplace_back(str.substr(beg, str.size() - beg));

   return out;
}

/// Read and split into white-space separated components a (txt) file
/// A component string starting with '#' will be ignored
/// \param[in] str File to read
/// \param[out] std::vector<std::string> A vector of strings
std::vector<std::string> ReadFilelist (std::string filelist) {
    std::vector<std::string> input_file_list;
    std::fstream infile;
    infile.open(filelist, std::ios::in);
    
    if (infile.is_open()) {
        infile.seekg(0);
        std::string line;
        while( std::getline(infile, line) ) {
            for (const auto& p: ROOT::Split(line.c_str(), " ", true)) { if (p[0] != '#') { input_file_list.push_back(p);} }
        }
    }
    return input_file_list;
}



} // namespace ROOT
