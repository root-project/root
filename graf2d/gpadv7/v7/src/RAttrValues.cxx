/// \file RAttrValues.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2017-09-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RAttrValues.hxx"
#include "ROOT/RAttrBase.hxx"

#include "ROOT/RLogger.hxx"

#include <algorithm>
#include <iterator>

template<> bool ROOT::Experimental::RAttrValues::Value_t::get<bool>() const { return GetBool(); }
template<> int ROOT::Experimental::RAttrValues::Value_t::get<int>() const { return GetInt(); }
template<> double ROOT::Experimental::RAttrValues::Value_t::get<double>() const { return GetDouble(); }
template<> std::string ROOT::Experimental::RAttrValues::Value_t::get<std::string>() const { return GetString(); }

template<> bool ROOT::Experimental::RAttrValues::Value_t::get_value<bool,void>(const Value_t *rec) { return rec ? rec->GetBool() : false; }
template<> int ROOT::Experimental::RAttrValues::Value_t::get_value<int,void>(const Value_t *rec) { return rec ? rec->GetInt() : 0; }
template<> double ROOT::Experimental::RAttrValues::Value_t::get_value<double,void>(const Value_t *rec) { return rec ? rec->GetDouble() : 0.; }
template<> std::string ROOT::Experimental::RAttrValues::Value_t::get_value<std::string,void>(const Value_t *rec) { return rec ? rec->GetString() : ""; }

template<> const ROOT::Experimental::RAttrValues::Value_t *ROOT::Experimental::RAttrValues::Value_t::get_value<const ROOT::Experimental::RAttrValues::Value_t *,void>(const Value_t *rec) { return rec; }
template<> const ROOT::Experimental::RAttrValues::Value_t *ROOT::Experimental::RAttrValues::Value_t::get_value<const ROOT::Experimental::RAttrValues::Value_t *,bool>(const Value_t *rec) { return rec && rec->Kind() == RAttrValues::kBool ? rec : nullptr; }
template<> const ROOT::Experimental::RAttrValues::Value_t *ROOT::Experimental::RAttrValues::Value_t::get_value<const ROOT::Experimental::RAttrValues::Value_t *,int>(const Value_t *rec) { return rec && rec->Kind() == RAttrValues::kInt ? rec : nullptr; }
template<> const ROOT::Experimental::RAttrValues::Value_t *ROOT::Experimental::RAttrValues::Value_t::get_value<const ROOT::Experimental::RAttrValues::Value_t *,double>(const Value_t *rec) { return rec && rec->Kind() == RAttrValues::kDouble ? rec : nullptr; }
template<> const ROOT::Experimental::RAttrValues::Value_t *ROOT::Experimental::RAttrValues::Value_t::get_value<const ROOT::Experimental::RAttrValues::Value_t *,std::string>(const Value_t *rec) { return rec && rec->Kind() == RAttrValues::kString ? rec : nullptr;  }


using namespace std::string_literals;

ROOT::Experimental::RAttrValues::Map_t &ROOT::Experimental::RAttrValues::Map_t::AddDefaults(const RAttrBase &vis)
{
   auto prefix = vis.GetPrefixToParent();

   for (const auto &entry : vis.GetDefaults())
      m[prefix+entry.first] = std::unique_ptr<Value_t>(entry.second->Copy());

   return *this;
}
