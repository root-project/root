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

const std::string RMetaData::Dump(const std::string &key) const
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

const std::string RMetaData::GetS(const std::string &key) const
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

const std::string RMetaData::GetS(const std::string &key, std::string defaultVal) const
{
   if (!fJson.contains(key))
      return defaultVal;
   if (!fJson[key].is_string())
      throw std::logic_error("Key " + key + " is not of type string.");
   return fJson[key].get<std::string>();
}

RDatasetGroup::RDatasetGroup(const std::string &groupName, const std::string &treeName, const std::string &fileNameGlob,
                             const RMetaData &metaData)
: fGroupName(groupName), fTreeNames({treeName}), fFileNameGlobs({fileNameGlob}), fMetaData(metaData)
{
}

RDatasetGroup::RDatasetGroup(const std::string &groupName, const std::string &treeName,
                             const std::vector<std::string> &fileNameGlobs, const RMetaData &metaData)
: fGroupName(groupName), fTreeNames(std::vector<std::string>(fileNameGlobs.size(), treeName)), fFileNameGlobs(fileNameGlobs), fMetaData(metaData)
{
}

RDatasetGroup::RDatasetGroup(const std::string &groupName,
                             const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                             const RMetaData &metaData)
: fGroupName(groupName), fMetaData(metaData)
{
   fTreeNames.reserve(treeAndFileNameGlobs.size());
   fFileNameGlobs.reserve(treeAndFileNameGlobs.size());
   for (auto &p : treeAndFileNameGlobs) {
      fTreeNames.emplace_back(p.first);
      fFileNameGlobs.emplace_back(p.second);
   }
}

RDatasetGroup::RDatasetGroup(const std::string &groupName, const std::vector<std::string> &treeNames,
                             const std::vector<std::string> &fileNameGlobs, const RMetaData &metaData)
: fGroupName(groupName), fTreeNames(treeNames), fFileNameGlobs(fileNameGlobs), fMetaData(metaData)
{
   if (treeNames.size() != 1 && treeNames.size() != fileNameGlobs.size())
      throw std::logic_error("Mismatch between number of trees and file globs.");
}

const std::string &RDatasetGroup::GetGroupName() const
{
   return fGroupName;
}

const std::vector<std::string> &RDatasetGroup::GetTreeNames() const
{
   return fTreeNames;
}

const std::vector<std::string> &RDatasetGroup::GetFileNameGlobs() const
{
   return fFileNameGlobs;
}

const RMetaData &RDatasetGroup::GetMetaData() const
{
   return fMetaData;
}

unsigned int RDatasetGroup::GetGroupId() const
{
   return fGroupId;
}

void RDatasetGroup::SetGroupId(unsigned int id)
{
   fGroupId = id;
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT
