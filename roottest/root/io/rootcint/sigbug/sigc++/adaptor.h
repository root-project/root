// -*- c++ -*-
/*  
  Copyright 2000, Karl Einar Nelson

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
*/
#ifndef   SIGC_ADAPTOR_SLOT
#define   SIGC_ADAPTOR_SLOT
#include <sigc++/slot.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

// (internal) 
struct LIBSIGC_API AdaptorSlotNode : public SlotNode
  {
    Node slot_;
    Link link_;

    virtual Link* link();
    virtual void notify(bool from_child);

    AdaptorSlotNode(FuncPtr proxy,const Node& s);

    virtual ~AdaptorSlotNode();
  };

#ifdef SIGC_CXX_NAMESPACES
}
#endif


#endif // SIGC_ADAPTOR_SLOT
