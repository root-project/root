/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *   IP, Inti Pelupessy,  NL eScience Center, i.pelupessy@esciencecenter.nl  *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include <MultiProcess/messages.h>
#include <MultiProcess/BidirMMapPipe.h>

namespace RooFit {
  namespace MultiProcess {

    // for debugging
#define PROCESS_VAL(p) case(p): s = #p; break;

    std::ostream& operator<<(std::ostream& out, const M2Q value){
      const char* s = 0;
      switch(value){
        PROCESS_VAL(M2Q::terminate);
        PROCESS_VAL(M2Q::enqueue);
        PROCESS_VAL(M2Q::retrieve);
        PROCESS_VAL(M2Q::update_real);
        PROCESS_VAL(M2Q::switch_work_mode);
        PROCESS_VAL(M2Q::call_double_const_method);
      }
      return out << s;
    }

    std::ostream& operator<<(std::ostream& out, const Q2M value){
      const char* s = 0;
      switch(value){
        PROCESS_VAL(Q2M::retrieve_rejected);
        PROCESS_VAL(Q2M::retrieve_accepted);
      }
      return out << s;
    }

    std::ostream& operator<<(std::ostream& out, const W2Q value){
      const char* s = 0;
      switch(value){
        PROCESS_VAL(W2Q::dequeue);
        PROCESS_VAL(W2Q::send_result);
      }
      return out << s;
    }

    std::ostream& operator<<(std::ostream& out, const Q2W value){
      const char* s = 0;
      switch(value){
        PROCESS_VAL(Q2W::terminate);
        PROCESS_VAL(Q2W::dequeue_rejected);
        PROCESS_VAL(Q2W::dequeue_accepted);
        PROCESS_VAL(Q2W::update_real);
        PROCESS_VAL(Q2W::switch_work_mode);
        PROCESS_VAL(Q2W::result_received);
        PROCESS_VAL(Q2W::call_double_const_method);
      }
      return out << s;
    }

#undef PROCESS_VAL

  } // namespace MultiProcess

  // stream operators for message enum classes
  BidirMMapPipe &BidirMMapPipe::operator<<(const MultiProcess::M2Q& sent) {
    *this << static_cast<int>(sent);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator>>(MultiProcess::M2Q& received) {
    int receptor;
    *this >> receptor;
    received = static_cast<MultiProcess::M2Q>(receptor);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator<<(const MultiProcess::Q2M& sent) {
    *this << static_cast<int>(sent);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator>>(MultiProcess::Q2M& received) {
    int receptor;
    *this >> receptor;
    received = static_cast<MultiProcess::Q2M>(receptor);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator<<(const MultiProcess::W2Q& sent) {
    *this << static_cast<int>(sent);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator>>(MultiProcess::W2Q& received) {
    int receptor;
    *this >> receptor;
    received = static_cast<MultiProcess::W2Q>(receptor);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator<<(const MultiProcess::Q2W& sent) {
    *this << static_cast<int>(sent);
    return *this;
  }

  BidirMMapPipe &BidirMMapPipe::operator>>(MultiProcess::Q2W& received) {
    int receptor;
    *this >> receptor;
    received = static_cast<MultiProcess::Q2W>(receptor);
    return *this;
  }

} // namespace RooFit
