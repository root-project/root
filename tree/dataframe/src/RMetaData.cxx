// Author: Ivan Kabadzhov CERN  10/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RMetaData.hxx"
#include <nlohmann/json.hpp>
#include <stdexcept> // std::logic_error

struct ROOT::Internal::RDF::RMetaDataJson {
   nlohmann::json payload;
};

namespace ROOT {
namespace RDF {
namespace Experimental {

RMetaData::RMetaData() : fJson{std::make_unique<Internal::RDF::RMetaDataJson>()} {}

RMetaData::RMetaData(RMetaData const &other) : fJson{std::make_unique<Internal::RDF::RMetaDataJson>(*other.fJson)} {}

RMetaData::RMetaData(RMetaData &&) = default;

RMetaData &RMetaData::operator=(RMetaData const &other)
{
   fJson = std::make_unique<Internal::RDF::RMetaDataJson>(*other.fJson);
   return *this;
}

RMetaData &RMetaData::operator=(RMetaData &&) = default;

RMetaData::~RMetaData() = default;

/// @brief Add an RMetaData class instance.
/// @param[in] key input key for a given RMetaData instance.
/// @param[in] val value for a given RMetaData instance of type int.
void RMetaData::Add(const std::string &key, int val)
{
   fJson->payload[key] = val;
}

/// @brief Add an RMetaData class instance.
/// @param[in] key input key for a given RMetaData instance.
/// @param[in] val metadata value for a given RMetaData instance of type double.
void RMetaData::Add(const std::string &key, double val)
{
   fJson->payload[key] = val;
}

/// @brief Add an RMetaData class instance
/// @param[in] key input key for a given RMetaData instance.
/// @param[in] val metadata value for of type string.
void RMetaData::Add(const std::string &key, const std::string &val)
{
   fJson->payload[key] = val;
}

/// @brief Dump the value of the metadata value given the key.
/// @param[in] key input key for a given RMetaData instance.
/// @return metadata value (as a string) associated with the input key, irrelevant of the actual type of the metadata
/// value.
std::string RMetaData::Dump(const std::string &key) const
{
   return fJson->payload[key].dump();
}
/// @brief Return the metadata value of type int given the key, or an error if the metadata value is of a non-int type.
/// @param[in] key input key for a given RMetaData instance.
int RMetaData::GetI(const std::string &key) const
{
   if (!fJson->payload.contains(key))
      throw std::logic_error("No key with name " + key + " in the metadata object.");
   if (!fJson->payload[key].is_number_integer())
      throw std::logic_error("Metadata value found at key '" + key + "' is not of type int.");
   return fJson->payload[key].get<int>();
}
/// @brief Return the metadata value of type double given the key, or an error if the metadata value is of a non-double
/// type.
/// @param[in] key input key for a given RMetaData instance.
double RMetaData::GetD(const std::string &key) const
{
   if (!fJson->payload.contains(key))
      throw std::logic_error("No key with name " + key + " in the metadata object.");
   if (!fJson->payload[key].is_number_float())
      throw std::logic_error("Metadata value found at key '" + key + "' is not of type double.");
   return fJson->payload[key].get<double>();
}

/// @brief Return the metadata value of type string given the key, or an error if the metadata value is of a non-string
/// type.
/// @param[in] key input key for a given RMetaData instance.
std::string RMetaData::GetS(const std::string &key) const
{
   if (!fJson->payload.contains(key))
      throw std::logic_error("No key with name " + key + " in the metadata object.");
   if (!fJson->payload[key].is_string())
      throw std::logic_error("Metadata value found at key '" + key + "' is not of type string.");
   return fJson->payload[key].get<std::string>();
}
/// @brief Return the metadata value of type int given the key, a default int metadata value if the key is not found, or
/// an error if the metadata value is of a non-int type.
/// @param[in] key input key for a given RMetaData instance.
/// @param[in] defaultVal metadata value of type int which is read as default while a given key cannot be found in the
/// dataset.
int RMetaData::GetI(const std::string &key, int defaultVal) const
{
   if (!fJson->payload.contains(key))
      return defaultVal;
   if (!fJson->payload[key].is_number_integer())
      throw std::logic_error("Metadata value found at key '" + key + "' is not of type int.");
   return fJson->payload[key].get<int>();
}
/// @brief Return the metadata value of type double given the key, a default double metadata value if the key is not
/// found, or an error if the metadata value is of a non-double type.
/// @param[in] key input key for a given RMetaData instance.
/// @param[in] defaultVal metadata value of type double which is read as default while a given key cannot be found in
/// the dataset.
double RMetaData::GetD(const std::string &key, double defaultVal) const
{
   if (!fJson->payload.contains(key))
      return defaultVal;
   if (!fJson->payload[key].is_number_float())
      throw std::logic_error("Metadata value found at key '" + key + "' is not of type double.");
   return fJson->payload[key].get<double>();
}

/// @brief Return the metadata value of type int given the key, a default int metadata value if the key is not found, or
/// an error if the metadata value is of a non-string type.
/// @param[in] key input key for a given RMetaData instance.
/// @param[in] defaultVal metadata value of type string which is read as default while a given key cannot be found in
/// the dataset.
const std::string RMetaData::GetS(const std::string &key, const std::string &defaultVal) const
{
   if (!fJson->payload.contains(key))
      return defaultVal;
   if (!fJson->payload[key].is_string())
      throw std::logic_error("Metadata value found at key '" + key + "' is not of type string.");
   return fJson->payload[key].get<std::string>();
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT

void ROOT::Internal::RDF::ImportJSON(ROOT::RDF::Experimental::RMetaData &metadata, const std::string &jsonString)
{
   metadata.fJson->payload = nlohmann::json::parse(jsonString);
}

std::string ROOT::Internal::RDF::ExportJSON(ROOT::RDF::Experimental::RMetaData &metadata)
{
   return metadata.fJson->payload.dump();
}
