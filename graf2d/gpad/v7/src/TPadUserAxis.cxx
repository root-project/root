/// \file TPadUserAxis.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-08-14
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#include <ROOT/TPadUserAxis.hxx>

// pin vtable
ROOT::Experimental::Detail::TPadUserAxisBase::~TPadUserAxisBase()
{
}

ROOT::Experimental::TPadLength::Normal
ROOT::Experimental::TPadCartesianUserAxis::ToNormal(const TPadLength::User &usercoord) const
{
   return usercoord.fVal * (fEnd - fBegin) + fBegin;
}
