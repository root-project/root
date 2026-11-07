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
#ifndef ROOT_ROOFIT_MultiProcess_Poller
#define ROOT_ROOFIT_MultiProcess_Poller

#include "RooFit/MultiProcess/Channel.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace RooFit {
namespace MultiProcess {

/// \class Poller
/// \brief Waits for input on a set of registered Channels
///
/// Replacement for the ZeroMQPoller: channels get a stable index in
/// registration order, poll() returns the indices of the channels that have
/// input available, and channels can be unregistered without changing the
/// indices of the others.
///
/// The Poller stores plain pointers, so registered Channel objects must stay
/// at their memory location while the Poller is in use.
class Poller {
public:
   /// Register a channel for input polling; returns its stable index.
   std::size_t register_channel(const Channel &channel)
   {
      entries_.emplace_back(next_index_++, &channel);
      return entries_.back().first;
   }

   void unregister_channel(const Channel &channel)
   {
      for (auto it = entries_.begin(); it != entries_.end(); ++it) {
         if (it->second == &channel) {
            entries_.erase(it);
            return;
         }
      }
      throw std::runtime_error("Poller::unregister_channel: channel not registered");
   }

   std::size_t size() const { return entries_.size(); }

   /// Wait for input; returns the registration indices of readable channels.
   /// Throws ppoll_error_t with num() == EINTR when a SIGTERM was received.
   std::vector<std::size_t> poll(int timeout_ms = -1) const
   {
      if (entries_.empty() && timeout_ms < 0) {
         throw std::logic_error("Poller::poll: waiting without timeout on a poller with no registered channels");
      }
      std::vector<const Channel *> channels;
      channels.reserve(entries_.size());
      for (auto &entry : entries_) {
         channels.push_back(entry.second);
      }
      std::vector<std::size_t> result;
      for (std::size_t pos : Channel::wait(channels, timeout_ms)) {
         result.push_back(entries_[pos].first);
      }
      return result;
   }

private:
   std::vector<std::pair<std::size_t, const Channel *>> entries_;
   std::size_t next_index_ = 0;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_Poller
