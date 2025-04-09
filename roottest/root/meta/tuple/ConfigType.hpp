// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
#ifndef SELF_REFERENCE_CONFIGTYPE_HPP
#define SELF_REFERENCE_CONFIGTYPE_HPP

// System include(s).
#include <map>
#include <string>
#include <tuple>

/// Simple configuration type
struct ConfigType {
   std::map<std::string, std::tuple<ConfigType, std::string> > m_children;
   void createOne() {
      m_children["one"] = std::tuple<ConfigType, std::string>{};
   }
};

#endif // SELF_REFERENCE_CONFIGTYPE_HPP
