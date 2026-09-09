/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */
#ifndef ROOT_ROOFIT_MultiProcess_Message
#define ROOT_ROOFIT_MultiProcess_Message

#include <cstddef>
#include <cstring>
#include <iterator>
#include <ostream>
#include <vector>

namespace RooFit {
namespace MultiProcess {

/// A contiguous byte buffer used as the unit of interprocess communication.
///
/// This is the plain replacement for zmq::message_t: Job implementations
/// build a Message on the sending side (e.g. from a struct or an array of
/// doubles) and read it out via the typed data<T>() accessors on the
/// receiving side.
class Message {
public:
   Message() = default;
   explicit Message(std::size_t size) : buf_(size) {}

   /// Create a message by copying the bytes of the elements in the range
   /// [first, last), like the equivalent zmq::message_t constructor.
   template <typename ForwardIt>
   Message(ForwardIt first, ForwardIt last)
   {
      using value_t = typename std::iterator_traits<ForwardIt>::value_type;
      buf_.resize(sizeof(value_t) * std::distance(first, last));
      char *out = buf_.data();
      for (ForwardIt it = first; it != last; ++it) {
         value_t const &item = *it;
         std::memcpy(out, &item, sizeof(value_t));
         out += sizeof(value_t);
      }
   }

   void *data() { return buf_.data(); }
   const void *data() const { return buf_.data(); }

   template <typename T>
   T *data()
   {
      return reinterpret_cast<T *>(buf_.data());
   }
   template <typename T>
   const T *data() const
   {
      return reinterpret_cast<const T *>(buf_.data());
   }

   /// Size of the message in bytes.
   std::size_t size() const { return buf_.size(); }

private:
   std::vector<char> buf_;
};

// for debug printing in the Messenger
inline std::ostream &operator<<(std::ostream &out, const Message &msg)
{
   return out << "Message(" << msg.size() << " bytes)";
}

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_Message
