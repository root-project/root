// $Id: ChannelSelection.h,v 1.24 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_CHANNELSELECTION_H
#define COOLKERNEL_CHANNELSELECTION_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include <limits>
#include <vector>
#include "CoolKernel/ChannelId.h"

namespace cool 
{

  /** @class ChannelSelection ChannelSelection.h
   *
   *  Helper class to specify a selection of channels and their ordering
   *  for multi-channel bulk retrieval of IOVs.
   *
   *  So far, only selection of IOVs _within a given tag_ is supported (the
   *  choice of the selected tag is made outside the ChannelSelection class).
   *  Within each channel, IOVs can be browsed ordered by 'iovSince',
   *  and there is only one IOV valid at any given validity time.
   *
   *  @author Sven A. Schmidt, Andrea Valassi and Marco Clemencic
   *  @date   2005-08-08
   */

  class ChannelSelection
  {

    friend class ChannelSelectionTest;

  public:

    /// Internal helper class for channel ranges. Ideally this class should
    /// be private but PyCool dictionary generation does not like that!

    class ChannelRange
    {

    public:

      // Required by PyCool
#ifdef COOL290CO
      ChannelRange() 
        : m_firstChannel( 0 ) // Fix Coverity UNINIT_CTOR (bug #95363)
        , m_lastChannel( 0 ) {}
#else
      ChannelRange() {}
#endif

      ChannelRange( const ChannelId& firstChannel,
                    const ChannelId& lastChannel );

      ChannelId firstChannel() const;
      ChannelId lastChannel() const;
      bool inRange( const ChannelId& channel ) const;

    private:

      ChannelId m_firstChannel;
      ChannelId m_lastChannel;

    };

    /// There are two possible orders to browse IOVs (within a tag) across
    /// many channels: 'order by channel, since' and 'order by since, channel'.
    /// The second set of the ordering scheme lists the IOVs in reverse order.
    enum Order {
      channelBeforeSince, sinceBeforeChannel,
      channelBeforeSinceDesc, sinceDescBeforeChannel
    };

    /// Constructor to (implicitly) select IOVs from *all* channels
    /// with the given order (default is 'order by channel, since').
    explicit ChannelSelection( const Order& order = channelBeforeSince );

    /// Constructor to select IOVs for a given channel. This constructor is
    /// intended to be used to autoconvert ChannelId to a ChannelSelection.
    ChannelSelection( const ChannelId& channel );

    /// Constructor to select IOVs for channels within a given range
    /// with the given order (default is 'order by channel, since').
    ChannelSelection( const ChannelId& firstChannel,
                      const ChannelId& lastChannel,
                      const Order& order = channelBeforeSince );

    /// Constructor to select IOVs with a given channel name.
    ChannelSelection( const std::string& channelName,
                      const Order& order = channelBeforeSince );

    /// Returns true if selecting all channels.
    bool allChannels() const;

    /// Returns the first selected channel
    /// [std::numeric_limits<ChannelId>::min() if selecting all channels].
    ChannelId firstChannel() const;

    /// Returns the last selected channel
    /// [std::numeric_limits<ChannelId>::max() if selecting all channels].
    ChannelId lastChannel() const;

    /// Returns the selection order.
    Order order() const;

    /// Construct a selection to select *all* channels with the given order.
    static const
    ChannelSelection all( const Order& order = channelBeforeSince );

    /// Returns true if the given channel is in the selection
    bool inSelection( const ChannelId& channel ) const;

    /// Returns true if the given channelName is in the selection
    bool inSelection( const std::string& channelName ) const;

    /// Returns true if the selection is contiguous
    /// This is the case if every channel between the very first and the
    /// last of the selection ranges is in the selection.
    /// This method does not make any assumption about the granularity. The
    /// only requirement is that operator++ at the end of an internal channel
    /// range will step to the first channel of the next range and not 'land'
    /// on a ChannelId outside the selection.
    bool isContiguous() const;

    /// Adds a channel range to the selection
    /// For sake of simplicity, it is required that the range is added
    /// to the front or the back of the existing selection without overlap.
    void addRange( const ChannelId& firstChannel,
                   const ChannelId& lastChannel );

    /// Adds a channel to the selection
    void addChannel( const ChannelId& channel );

    /// Returns true is the ChannelSelection is numeric.
    bool isNumeric() const;

    /// Returns the channel name list
    const std::string& channelName() const;

    /// Returns the beginning of a const range iterator
    std::vector<ChannelRange>::const_iterator begin() const;

    /// Returns the end of a const range iterator
    std::vector<ChannelRange>::const_iterator end() const;

    /// Returns the range count
    unsigned int rangeCount() const;

  private:

    bool m_isNumeric;
    bool m_allChannels;
    std::vector<ChannelRange> m_ranges;
    std::string m_channelName;
    Order m_order;

  };

}
#endif // COOLKERNEL_CHANNELSELECTION_H
