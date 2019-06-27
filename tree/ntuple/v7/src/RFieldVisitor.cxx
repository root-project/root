/// \file ROOT/RFieldVisitor.hxx
/// \ingroup NTuple ROOT7
/// \author Simon Leisibach <simon.satoshi.rene.leisibach@cern.ch>
/// \date 2019-06-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RFieldVisitor.hxx"
#include "ROOT/RField.hxx"
#include "ROOT/RNTuple.hxx"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>


void ROOT::Experimental::RPrintVisitor::visitField(const ROOT::Experimental::Detail::RFieldBase& fField) {
    std::string fNameAndType{fField.GetName() +" (" + fField.GetType()+")"};
    fOutput << " Field " << std::setw(FieldDistance(maxNoFields)+5-NumDigits(fField.GetOrder())) << fField.GetOrder() << " : " << CutIfNecessary(fNameAndType, fWidth-FieldDistance(maxNoFields)-16) << std::setw(fWidth-17-FieldDistance(maxNoFields)-fField.GetName().size()-fField.GetType().size());
}

int ROOT::Experimental::NumDigits(int x)
{
    x = abs(x);
    return (x < 10 ? 1 :
            (x < 100 ? 2 :
             (x < 1000 ? 3 :
              (x < 10000 ? 4 :
               (x < 100000 ? 5 :
                (x < 1000000 ? 6 :
                 (x < 10000000 ? 7 :
                  (x < 100000000 ? 8 :
                   (x < 1000000000 ? 9 :
                    10)))))))));
}

int ROOT::Experimental::FieldDistance(unsigned int x) {
    return (x < 10000 ? 0 :
             (x < 100000 ? 1 :
              (x < 1000000 ? 2 :
               (x < 10000000 ? 3 :
                (x < 100000000 ? 4 :
                 (x < 1000000000 ? 5 :
                  10))))));
}

std::string ROOT::Experimental::CutIfNecessary(const std::string &toCut, unsigned int maxAvailableSpace) {
    if(toCut.size() > maxAvailableSpace) {
        return std::string(toCut, 0, maxAvailableSpace-3) + "...";
    }
   return toCut;
}
