// $Id: HvsTagLock.h,v 1.3 2009-12-16 17:41:24 avalassi Exp $
#ifndef COOLKERNEL_HVSTAGLOCK_H
#define COOLKERNEL_HVSTAGLOCK_H

namespace cool {

  /** @file HvsTagLock.h
   *
   * Enum definition for the lock status of an HVS tag.
   *
   * @author Andrea Valassi
   * @date 2007-03-20
   */

  // HVS tag lock status.
  namespace HvsTagLock
  {
    enum Status { UNLOCKED=0, LOCKED=1, PARTIALLYLOCKED=2 };
  }

}

#endif
