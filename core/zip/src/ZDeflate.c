/* This file just puts all the ZIP files together -- to make the compression
 * run faster and to fix a bug with the 'level' variable declared as static.
 * Michal Kapalka (kapalka@icslab.agh.edu.pl), CERN, openlab, August 2003
 */

#include "ZIP.h"
#include "ZDeflate.h"
#include "Bits.h"
#include "ZTrees.h"
