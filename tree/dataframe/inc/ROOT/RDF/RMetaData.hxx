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

/**
\class ROOT::RDF::Experimental::RMetaData
\ingroup dataframe
\brief Class behaving as a heterogenuous dictionary to store dataset metadata

 This class should be passed to an RSample object which represents a single dataset sample.
 Once a dataframe is built with RMetaData object, it could be accessed via DefinePerSample.
*/
class RMetaData {
   nlohmann::json fJson;

public:
   void Add(const std::string &key, int val);
   void Add(const std::string &key, double val);
   void Add(const std::string &key, const std::string &val);

   std::string Dump(const std::string &key) const; // always returns a string
   int GetI(const std::string &key) const;
   double GetD(const std::string &key) const;
   std::string GetS(const std::string &key) const;
   int GetI(const std::string &key, int defaultVal) const;
   double GetD(const std::string &key, double defaultVal) const;
   const std::string GetS(const std::string &key, const std::string &defaultVal) const;
};

} // namespace Experimental
} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RMETADATA
