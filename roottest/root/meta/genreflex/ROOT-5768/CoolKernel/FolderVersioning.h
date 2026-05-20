// $Id: FolderVersioning.h,v 1.2 2009-12-16 17:41:24 avalassi Exp $
#ifndef COOLKERNEL_FOLDERVERSIONING_H
#define COOLKERNEL_FOLDERVERSIONING_H

namespace cool {

  /** @file FolderVersioning.h
   *
   * Enum definition for the versioning mode of a COOL folder.
   *
   * IOV versioning in a folder is enabled only in the "MultiVersion" mode.
   * In the "SingleVersion" mode, a new object can be inserted only if its
   * IOV begins after the end of the IOV of the last object inserted.
   * The NONE value is only used internally for folder sets.
   *
   * @author Sven A. Schmidt and Andrea Valassi
   * @date 2004-11-05
   */

  // Folder versioning mode.
  namespace FolderVersioning
  {
    enum Mode { NONE=-1, SINGLE_VERSION=0, MULTI_VERSION };
  }

}

#endif
