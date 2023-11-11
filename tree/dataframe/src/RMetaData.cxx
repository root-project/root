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

void RMetaData::Add(const std::string &key, int val)
{
   fJson->payload[key] = val;
}

void RMetaData::Add(const std::string &key, double val)
{
   fJson->payload[key] = val;
}

void RMetaData::Add(const std::string &key, const std::string &val)
{
   fJson->payload[key] = val;
}

std::string RMetaData::Dump(const std::string &key) const
{
   return fJson->payload[key].dump();
}

int RMetaData::GetI(const std::string &key) const
{
   if (!fJson->payload.contains(key))
      throw std::logic_error("No key with name " + key + " in the metadata object.");
   if (!fJson->payload[key].is_number_integer())
      throw std::logic_error("Key " + key + " is not of type int.");
   return fJson->payload[key].get<int>();
}

double RMetaData::GetD(const std::string &key) const
{
   if (!fJson->payload.contains(key))
      throw std::logic_error("No key with name " + key + " in the metadata object.");
   if (!fJson->payload[key].is_number_float())
      throw std::logic_error("Key " + key + " is not of type double.");
   return fJson->payload[key].get<double>();
}

std::string RMetaData::GetS(const std::string &key) const
{
   if (!fJson->payload.contains(key))
      throw std::logic_error("No key with name " + key + " in the metadata object.");
   if (!fJson->payload[key].is_string())
      throw std::logic_error("Key " + key + " is not of type string.");
   return fJson->payload[key].get<std::string>();
}

int RMetaData::GetI(const std::string &key, int defaultVal) const
{
   if (!fJson->payload.contains(key))
      return defaultVal;
   if (!fJson->payload[key].is_number_integer())
      throw std::logic_error("Key " + key + " is not of type int.");
   return fJson->payload[key].get<int>();
}

double RMetaData::GetD(const std::string &key, double defaultVal) const
{
   if (!fJson->payload.contains(key))
      return defaultVal;
   if (!fJson->payload[key].is_number_float())
      throw std::logic_error("Key " + key + " is not of type double.");
   return fJson->payload[key].get<double>();
}

const std::string RMetaData::GetS(const std::string &key, const std::string &defaultVal) const
{
   if (!fJson->payload.contains(key))
      return defaultVal;
   if (!fJson->payload[key].is_string())
      throw std::logic_error("Key " + key + " is not of type string.");
   return fJson->payload[key].get<std::string>();
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT
