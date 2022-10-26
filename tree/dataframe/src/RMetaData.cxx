// Author: Ivan Kabadzhov CERN  10/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RMetaData.hxx"
#include <stdexcept> // std::logic_error

#include <iostream>

namespace ROOT {
namespace RDF {
namespace Experimental {

void RMetaData::AddMetaData(const std::string &cat, int val)
{
   fJson[cat] = val;
}

void RMetaData::AddMetaData(const std::string &cat, double val)
{
   fJson[cat] = val;
}

void RMetaData::AddMetaData(const std::string &cat, const std::string &val)
{
   fJson[cat] = val;
}

void RMetaData::AddJSONMetaData(const std::string &cat, const nlohmann::json &val)
{
   fJson[cat] = val;
}

std::string RMetaData::Dump(const std::string &cat) const
{
   return fJson[cat].dump();
}

int RMetaData::GetI(const std::string &cat) const
{
   if (!fJson.contains(cat))
      throw std::logic_error("No key with name " + cat + " in the metadata object.");
   if (!fJson[cat].is_number_integer())
      throw std::logic_error("Key " + cat + " is not of type int.");
   return fJson[cat].get<int>();
}

double RMetaData::GetD(const std::string &cat) const
{
   if (!fJson.contains(cat))
      throw std::logic_error("No key with name " + cat + " in the metadata object.");
   if (!fJson[cat].is_number_float())
      throw std::logic_error("Key " + cat + " is not of type double.");
   return fJson[cat].get<double>();
}

std::string RMetaData::GetS(const std::string &cat) const
{
   if (!fJson.contains(cat))
      throw std::logic_error("No key with name " + cat + " in the metadata object.");
   if (!fJson[cat].is_string())
      throw std::logic_error("Key " + cat + " is not of type string.");
   return fJson[cat].get<std::string>();
}

int RMetaData::GetI(const std::string &cat, int defaultVal) const
{
   if (!fJson.contains(cat))
      return defaultVal;
   if (!fJson[cat].is_number_integer())
      throw std::logic_error("Key " + cat + " is not of type int.");
   return fJson[cat].get<int>();
}

double RMetaData::GetD(const std::string &cat, double defaultVal) const
{
   if (!fJson.contains(cat))
      return defaultVal;
   if (!fJson[cat].is_number_float())
      throw std::logic_error("Key " + cat + " is not of type double.");
   return fJson[cat].get<double>();
}

std::string RMetaData::GetS(const std::string &cat, std::string defaultVal) const
{
   if (!fJson.contains(cat))
      return defaultVal;
   if (!fJson[cat].is_string())
      throw std::logic_error("Key " + cat + " is not of type string.");
   return fJson[cat].get<std::string>();
}

RGroupMetaData::RGroupMetaData(const std::string &groupName, unsigned int groupId,
                               const RMetaData &metaData)
   : fGroupName(groupName), fGroupId(groupId), fMetaData(metaData)
{
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT
