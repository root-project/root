// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

#include "Math/GenVector/BitReproducible.h"

#include <sstream>
#include <iomanip>
#include <exception>

namespace ROOT { 
namespace Math { 
namespace GenVector_detail {

bool BitReproducible::fgByte_order_known = false;
int  BitReproducible::fgByte_order[8];

void BitReproducible::Fill_byte_order () {
   // Fill_byte_order
   double x = 1.0;
   int t30 = 1 << 30;
   int t22 = 1 << 22;
   x *= t30;
   x *= t22;
   double y = 1;
   double z = 1;
   x *= z;
   for (int k=0; k<6; k++) {
      x += y*z;
      y += 1;
      z *= 256;
   }
   // x, in IEEE format, would now be 0x4330060504030201
   union DB8 {
      unsigned char fB[8];
      double fD;
   };
   DB8 xb;
   xb.fD = x;
   int n;
   static const int kUNSET = -1;
   for (n=0; n<8; n++) {
      fgByte_order[n] = kUNSET;
   }
   int order;
   for (n=0; n<8; n++) {
      switch ( xb.fB[n] ) {
         case 0x43:
            order = 0;
            break;
         case 0x30:
            order = 1;
            break;
         case 0x06:
            order = 2;
            break;
         case 0x05:
            order = 3;
            break;
         case 0x04:
            order = 4;
            break;
         case 0x03:
            order = 5;
            break;
         case 0x02:
            order = 6;
            break;
         case 0x01:
            order = 7;
            break;
         default:
            throw BitReproducibleException(
                                           "Cannot determine byte-ordering of doubles on this system");
      }
      if (fgByte_order[n] != kUNSET) {
         throw BitReproducibleException(
                                        "Confusion in byte-ordering of doubles on this system");
      }
      fgByte_order[n] = order;
      fgByte_order_known = true;
   }
   return;
}

std::string BitReproducible::D2x(double d) {
   // hex conversion
   if ( !fgByte_order_known ) Fill_byte_order ();
   DB8 db;
   db.fD = d;
   std::ostringstream ss;
   for (int i=0; i<8; ++i) {
      int k = fgByte_order[i];
      ss << std::hex << std::setw(2) << std::setfill('0') << (int)db.fB[k];
   }
   return ss.str();
}

void BitReproducible::Dto2longs(double d, unsigned int& i, unsigned int& j) {
   // conversion to 2 longs
   if ( !fgByte_order_known ) Fill_byte_order ();
   DB8 db;
   db.fD = d;
   i    =   ((static_cast<unsigned int>(db.fB[fgByte_order[0]])) << 24)
      | ((static_cast<unsigned int>(db.fB[fgByte_order[1]])) << 16)
      | ((static_cast<unsigned int>(db.fB[fgByte_order[2]])) <<  8)
      | ((static_cast<unsigned int>(db.fB[fgByte_order[3]]))      );
   j    =   ((static_cast<unsigned int>(db.fB[fgByte_order[4]])) << 24)
      | ((static_cast<unsigned int>(db.fB[fgByte_order[5]])) << 16)
      | ((static_cast<unsigned int>(db.fB[fgByte_order[6]])) <<  8)
      | ((static_cast<unsigned int>(db.fB[fgByte_order[7]]))      );
}

double BitReproducible::Longs2double (unsigned int i, unsigned int j) {
   // conversion longs to double
   DB8 db;
   unsigned char bytes[8];
   if ( !fgByte_order_known ) Fill_byte_order ();
   bytes[0] = static_cast<unsigned char>((i >> 24) & 0xFF);
   bytes[1] = static_cast<unsigned char>((i >> 16) & 0xFF);
   bytes[2] = static_cast<unsigned char>((i >>  8) & 0xFF);
   bytes[3] = static_cast<unsigned char>((i      ) & 0xFF);
   bytes[4] = static_cast<unsigned char>((j >> 24) & 0xFF);
   bytes[5] = static_cast<unsigned char>((j >> 16) & 0xFF);
   bytes[6] = static_cast<unsigned char>((j >>  8) & 0xFF);
   bytes[7] = static_cast<unsigned char>((j      ) & 0xFF);
   for (int k=0; k<8; ++k) {
      db.fB[fgByte_order[k]] =  bytes[k];
   }
   return db.fD;
}

}  // namespace _GenVector_detail
}  // namespace Math
}  // namespace ROOT
