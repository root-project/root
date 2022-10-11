// Author: Ivan Kabadzhov CERN  10/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RDFFromJSON.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "ROOT/RDF/RMetaData.hxx"

#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept> // std::logic_error

namespace ROOT {
namespace RDF {
namespace Experimental {

ROOT::RDataFrame MakeDataFrameFromJSON(const std::string &jsonFile)
{
   const nlohmann::json fullData = nlohmann::json::parse(std::ifstream(jsonFile));
   RSpecBuilder specBuilder;

   for (const auto &groups : fullData["groups"]) {
      std::string tag = groups["tag"];
      std::vector<std::string> trees = groups["trees"];
      std::vector<std::string> files = groups["files"];
      if (files.size() != trees.size() && trees.size() > 1)
         throw std::runtime_error("Mismatch between trees and files.");
      RMetaData m;
      // m.AddMetaData(groups["metadata"]);
      for (const auto &metadata : groups["metadata"].items())
         m.AddJSONMetaData(metadata.key(), metadata.value());
      specBuilder.AddGroup(tag, trees, files, m);
   }
   if (fullData.contains("friends")) {
      for (const auto &friends : fullData["friends"].items()) {
         std::string alias = friends.key();
         std::vector<std::string> trees = friends.value()["trees"];
         std::vector<std::string> files = friends.value()["files"];
         if (files.size() != trees.size() && trees.size() > 1)
            throw std::runtime_error("Mismatch between trees and files in a friend.");
         specBuilder.WithFriends(trees, files, alias);
      }
   }

   if (fullData.contains("range")) {
      std::vector<int> range = fullData["range"];

      if (range.size() == 1)
         specBuilder.WithRange({range[0]});
      else if (range.size() == 2)
         specBuilder.WithRange({range[0], range[1]});
   }
   // specBuilder.Build();
   return ROOT::RDataFrame(specBuilder.Build());
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT
