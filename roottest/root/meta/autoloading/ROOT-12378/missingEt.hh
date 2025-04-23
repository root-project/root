#pragma once

#include "DataVector.hh"
#include "Element.hh"

namespace TestMissingETBase {
namespace  Types {
using jetlink_t = TestElementLink<TestDataVector<TestxAOD::Jet_v1> >;
//typedef ElementLink<xAOD::JetContainer> jetlink_t;
//typedef TestElementLink<TestDataVector<TestxAOD::Jet_v1> > jetlink_t;
};
};


