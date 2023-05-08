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

namespace ROOT {
namespace RDF {
namespace Experimental {

void RMetaData::Add(const std::string &key, int val)
{
   fJson[key] = val;
}

void RMetaData::Add(const std::string &key, double val)
{
   fJson[key] = val;
}

void RMetaData::Add(const std::string &key, const std::string &val)
{
   fJson[key] = val;
}

std::string RMetaData::Dump(const std::string &key) const
{
   return fJson[key].dump();
}

int RMetaData::GetI(const std::string &key) const
{
   if (!fJson.contains(key))
      throw std::logic_error("No key with name " + key + " in the metadata object.");
   if (!fJson[key].is_number_integer())
      throw std::logic_error("Key " + key + " is not of type int.");
   return fJson[key].get<int>();
}

double RMetaData::GetD(const std::string &key) const
{
   if (!fJson.contains(key))
      throw std::logic_error("No key with name " + key + " in the metadata object.");
   if (!fJson[key].is_number_float())
      throw std::logic_error("Key " + key + " is not of type double.");
   return fJson[key].get<double>();
}

std::string RMetaData::GetS(const std::string &key) const
{
   if (!fJson.contains(key))
      throw std::logic_error("No key with name " + key + " in the metadata object.");
   if (!fJson[key].is_string())
      throw std::logic_error("Key " + key + " is not of type string.");
   return fJson[key].get<std::string>();
}

int RMetaData::GetI(const std::string &key, int defaultVal) const
{
   if (!fJson.contains(key))
      return defaultVal;
   if (!fJson[key].is_number_integer())
      throw std::logic_error("Key " + key + " is not of type int.");
   return fJson[key].get<int>();
}

double RMetaData::GetD(const std::string &key, double defaultVal) const
{
   if (!fJson.contains(key))
      return defaultVal;
   if (!fJson[key].is_number_float())
      throw std::logic_error("Key " + key + " is not of type double.");
   return fJson[key].get<double>();
}

const std::string RMetaData::GetS(const std::string &key, const std::string &defaultVal) const
{
   if (!fJson.contains(key))
      return defaultVal;
   if (!fJson[key].is_string())
      throw std::logic_error("Key " + key + " is not of type string.");
   return fJson[key].get<std::string>();
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT
