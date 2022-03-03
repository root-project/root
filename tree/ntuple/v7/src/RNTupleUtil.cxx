/// \file RNTupleUtil.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch> & Max Orok <maxwellorok@gmail.com>
/// \date 2020-07-14
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RNTupleUtil.hxx"

#include "ROOT/RLogger.hxx"
#include "ROOT/RMiniFile.hxx"

#include <cstring>
#include <iostream>

ROOT::Experimental::RLogChannel &ROOT::Experimental::NTupleLog() {
   static RLogChannel sLog("ROOT.NTuple");
   return sLog;
}


namespace ROOT {
namespace Experimental {
namespace Internal {

/// \brief Machine-independent serialization functions for fundamental types.
namespace RNTupleSerialization {

std::uint32_t SerializeInt64(std::int64_t val, void *buffer)
{
   if (buffer != nullptr) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);
      bytes[0] = (val & 0x00000000000000FF);
      bytes[1] = (val & 0x000000000000FF00) >> 8;
      bytes[2] = (val & 0x0000000000FF0000) >> 16;
      bytes[3] = (val & 0x00000000FF000000) >> 24;
      bytes[4] = (val & 0x000000FF00000000) >> 32;
      bytes[5] = (val & 0x0000FF0000000000) >> 40;
      bytes[6] = (val & 0x00FF000000000000) >> 48;
      bytes[7] = (val & 0xFF00000000000000) >> 56;
   }
   return 8;
}

std::uint32_t SerializeUInt64(std::uint64_t val, void *buffer)
{
   return SerializeInt64(val, buffer);
}

std::uint32_t DeserializeInt64(const void *buffer, std::int64_t *val)
{
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   *val = std::int64_t(bytes[0]) + (std::int64_t(bytes[1]) << 8) +
          (std::int64_t(bytes[2]) << 16) + (std::int64_t(bytes[3]) << 24) +
          (std::int64_t(bytes[4]) << 32) + (std::int64_t(bytes[5]) << 40) +
          (std::int64_t(bytes[6]) << 48) + (std::int64_t(bytes[7]) << 56);
   return 8;
}

std::uint32_t DeserializeUInt64(const void *buffer, std::uint64_t *val)
{
   return DeserializeInt64(buffer, reinterpret_cast<std::int64_t *>(val));
}

std::uint32_t SerializeInt32(std::int32_t val, void *buffer)
{
   if (buffer != nullptr) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);
      bytes[0] = (val & 0x000000FF);
      bytes[1] = (val & 0x0000FF00) >> 8;
      bytes[2] = (val & 0x00FF0000) >> 16;
      bytes[3] = (val & 0xFF000000) >> 24;
   }
   return 4;
}

std::uint32_t SerializeUInt32(std::uint32_t val, void *buffer)
{
   return SerializeInt32(val, buffer);
}

std::uint32_t DeserializeInt32(const void *buffer, std::int32_t *val)
{
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   *val = std::int32_t(bytes[0]) + (std::int32_t(bytes[1]) << 8) +
          (std::int32_t(bytes[2]) << 16) + (std::int32_t(bytes[3]) << 24);
   return 4;
}

std::uint32_t DeserializeUInt32(const void *buffer, std::uint32_t *val)
{
   return DeserializeInt32(buffer, reinterpret_cast<std::int32_t *>(val));
}

std::uint32_t SerializeInt16(std::int16_t val, void *buffer)
{
   if (buffer != nullptr) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);
      bytes[0] = (val & 0x00FF);
      bytes[1] = (val & 0xFF00) >> 8;
   }
   return 2;
}

std::uint32_t SerializeUInt16(std::uint16_t val, void *buffer)
{
   return SerializeInt16(val, buffer);
}

std::uint32_t DeserializeInt16(const void *buffer, std::int16_t *val)
{
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   *val = std::int16_t(bytes[0]) + (std::int16_t(bytes[1]) << 8);
   return 2;
}

std::uint32_t DeserializeUInt16(const void *buffer, std::uint16_t *val)
{
   return DeserializeInt16(buffer, reinterpret_cast<std::int16_t *>(val));
}

std::uint32_t SerializeString(const std::string &val, void *buffer)
{
   if (buffer != nullptr) {
      auto pos = reinterpret_cast<unsigned char *>(buffer);
      pos += SerializeUInt32(val.length(), pos);
      memcpy(pos, val.data(), val.length());
   }
   return SerializeUInt32(val.length(), nullptr) + val.length();
}

std::uint32_t DeserializeString(const void *buffer, std::string *val)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint32_t length;
   bytes += DeserializeUInt32(buffer, &length);
   val->resize(length);
   memcpy(&(*val)[0], bytes, length);
   return bytes + length - base;
}

} // namespace RNTupleSerialization

void PrintRNTuple(const RNTuple& ntuple, std::ostream& output) {
   output << "RNTuple {\n";
   output << "    fVersion: " << ntuple.fVersion << ",\n";
   output << "    fSize: " << ntuple.fSize << ",\n";
   output << "    fSeekHeader: " << ntuple.fSeekHeader << ",\n";
   output << "    fNBytesHeader: " << ntuple.fNBytesHeader << ",\n";
   output << "    fLenHeader: " << ntuple.fLenHeader << ",\n";
   output << "    fSeekFooter: " << ntuple.fSeekFooter << ",\n";
   output << "    fNBytesFooter: " << ntuple.fNBytesFooter << ",\n";
   output << "    fLenFooter: " << ntuple.fLenFooter << ",\n";
   output << "    fReserved: " << ntuple.fReserved << ",\n";
   output << "}";
}

} // namespace Internal
} // namespace Experimental
} // namespace ROOT
