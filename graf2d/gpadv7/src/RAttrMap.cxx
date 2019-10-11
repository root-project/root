/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RAttrMap.hxx"

#include "ROOT/RAttrBase.hxx"
#include "ROOT/RLogger.hxx"

template<> bool ROOT::Experimental::RAttrMap::Value_t::Get<bool>() const { return GetBool(); }
template<> int ROOT::Experimental::RAttrMap::Value_t::Get<int>() const { return GetInt(); }
template<> double ROOT::Experimental::RAttrMap::Value_t::Get<double>() const { return GetDouble(); }
template<> std::string ROOT::Experimental::RAttrMap::Value_t::Get<std::string>() const { return GetString(); }

template<> bool ROOT::Experimental::RAttrMap::Value_t::GetValue<bool,void>(const Value_t *rec) { return rec ? rec->GetBool() : false; }
template<> int ROOT::Experimental::RAttrMap::Value_t::GetValue<int,void>(const Value_t *rec) { return rec ? rec->GetInt() : 0; }
template<> double ROOT::Experimental::RAttrMap::Value_t::GetValue<double,void>(const Value_t *rec) { return rec ? rec->GetDouble() : 0.; }
template<> std::string ROOT::Experimental::RAttrMap::Value_t::GetValue<std::string,void>(const Value_t *rec) { return rec ? rec->GetString() : ""; }

template<> const ROOT::Experimental::RAttrMap::Value_t *ROOT::Experimental::RAttrMap::Value_t::GetValue<const ROOT::Experimental::RAttrMap::Value_t *,void>(const Value_t *rec) { return rec; }
template<> const ROOT::Experimental::RAttrMap::Value_t *ROOT::Experimental::RAttrMap::Value_t::GetValue<const ROOT::Experimental::RAttrMap::Value_t *,bool>(const Value_t *rec) { return rec && rec->Kind() == RAttrMap::kBool ? rec : nullptr; }
template<> const ROOT::Experimental::RAttrMap::Value_t *ROOT::Experimental::RAttrMap::Value_t::GetValue<const ROOT::Experimental::RAttrMap::Value_t *,int>(const Value_t *rec) { return rec && rec->Kind() == RAttrMap::kInt ? rec : nullptr; }
template<> const ROOT::Experimental::RAttrMap::Value_t *ROOT::Experimental::RAttrMap::Value_t::GetValue<const ROOT::Experimental::RAttrMap::Value_t *,double>(const Value_t *rec) { return rec && rec->Kind() == RAttrMap::kDouble ? rec : nullptr; }
template<> const ROOT::Experimental::RAttrMap::Value_t *ROOT::Experimental::RAttrMap::Value_t::GetValue<const ROOT::Experimental::RAttrMap::Value_t *,std::string>(const Value_t *rec) { return rec && rec->Kind() == RAttrMap::kString ? rec : nullptr;  }


using namespace std::string_literals;

ROOT::Experimental::RAttrMap &ROOT::Experimental::RAttrMap::AddDefaults(const RAttrBase &vis)
{
   auto prefix = vis.GetPrefix();

   for (const auto &entry : vis.GetDefaults())
      m[prefix+entry.first] = entry.second->Copy();

   return *this;
}
