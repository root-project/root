// Author: Ivan Kabadzhov CERN  10/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RMETADATA
#define ROOT_RDF_RMETADATA

#include <nlohmann/json.hpp>

#include <string>

namespace ROOT {
namespace RDF {
namespace Experimental {

class RMetaData {
   nlohmann::json fJson;

public:
   void AddMetaData(const std::string &cat, int val);
   void AddMetaData(const std::string &cat, double val);
   void AddMetaData(const std::string &cat, const std::string &val);
   void AddJSONMetaData(const std::string &cat, const nlohmann::json &val);

   // getter always returning the string representation
   std::string Dump(const std::string &cat) const;
   // type-safe! getters
   int GetI(const std::string &cat) const;
   double GetD(const std::string &cat) const;
   std::string GetS(const std::string &cat) const;
   int GetI(const std::string &cat, int defaultVal) const;
   double GetD(const std::string &cat, double defaultVal) const;
   std::string GetS(const std::string &cat, std::string defaultVal) const;
};

struct RGroupMetaData {
   std::string fGroupName;
   unsigned int fGroupId;
   ROOT::RDF::Experimental::RMetaData fMetaData;
   RGroupMetaData(const std::string &groupName = "", unsigned int groupId = 0u,
                  const RMetaData &metaData = {});
   };

} // namespace Experimental
} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RMETADATA
