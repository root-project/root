/// \file RPadUserAxis.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-08-14
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#include <ROOT/RPadUserAxis.hxx>

// pin vtable
ROOT::Experimental::RPadUserAxisBase::~RPadUserAxisBase()
{
}

ROOT::Experimental::RPadLength::Normal
ROOT::Experimental::RPadCartesianUserAxis::ToNormal(const RPadLength::User &usercoord) const
{
   return (usercoord.fVal - GetBegin()) / GetSensibleDenominator();
}
