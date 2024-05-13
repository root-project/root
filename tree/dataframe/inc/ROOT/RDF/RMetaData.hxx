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

#include <string>
#include <memory>

namespace ROOT {

namespace Internal {
namespace RDF {
// To avoid unnecessary dependence on nlohman json in the interface. Note that
// we should not forward declare nlohmann::json directly, since its declaration
// might change (it is currently a typedef). With this wrapper type, we are
// completely decoupled on nlohmann::json in the RMetaData interface.
struct RMetaDataJson;
}
}

namespace RDF {
namespace Experimental {

/**
\class ROOT::RDF::Experimental::RMetaData
\ingroup dataframe
\brief Class behaving as a heterogenuous dictionary to store the metadata of a dataset.

 The supported types of the metadata are: std::string, int and double. An example of creating the RMetaData object:
 ~~~{.cpp}
 ROOT::RDF::Experimental::RMetaData meta;
 meta.Add("sample_name", "name"");
 meta.Add("luminosity", 10064);
 meta.Add("xsecs", 1.0);
 ~~~

 The RMetaData object is passed to an RSample object which represents a single dataset sample.

 A dataframe built with the RMetaData object can be accessed with the \ref ROOT::RDF::RInterface< Proxied,
DS_t>::DefinePerSample "DefinePerSample()" method.
**/
class RMetaData {
public:

   RMetaData();
   // Note: each RMetaData instance should own its own fJson object, just as if
   // the underlying nlohmann::json object would be owned by value.
   RMetaData(RMetaData const&);
   RMetaData(RMetaData &&);
   RMetaData & operator=(RMetaData const&);
   RMetaData & operator=(RMetaData &&);
   ~RMetaData();

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

private:
   std::unique_ptr<Internal::RDF::RMetaDataJson> fJson;
};

} // namespace Experimental
} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RMETADATA
