// Authors: Bertrand Bellenot <bertrand.bellenot@cern.ch> Sergey Linev <S.Linev@gsi.de>
// Date: 2019-02-28

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBrowserReply
#define ROOT7_RBrowserReply

#include <string>
#include <vector>
#include <ROOT/Browsable/RItem.hxx>

namespace ROOT {

/** \class ROOT::RBrowserReply
\ingroup rbrowser
\brief Reply on browser request
*/

class RBrowserReply {
public:
   std::vector<std::string> path;     ///< reply path
   int nchilds{0};                    ///< total number of childs in the node
   int first{0};                      ///< first node in returned list
   std::vector<const Browsable::RItem *> nodes; ///< list of pointers, no ownership!
};

} // namespace ROOT

#endif


