#******************************************************************************
#   $Id: XrdOlbNotify.pm,v 1.4 2008/10/14 09:24:40 furano Exp $
#*                                                                            *
#*                       X r d O l b N o t i f y . p m                        *
#*                                                                            *
# (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University   *
#                          All Rights Reserved                                *
# Produced by Andrew Hanushevsky for Stanford University under contract       *
#            DE-AC03-76-SFO0515 with the Department of Energy                 *
#******************************************************************************
  
#!/usr/bin/env perl

package  XrdOlbNotify;
require  Exporter;
@ISA   = qw(Exporter);
@EXPORT= qw(setDebug FileGone FileHere SendMsg);
use Socket;
{1}

#******************************************************************************
#*                              F i l e G o n e                               *
#******************************************************************************
 
#Call:   FileGone([iname,]paths);

#Input:   iname   - is the instance name. If the instance name starts with a
#                   slash, it is treated as a path and the olb is unnamed.
#        @paths   - an array of paths that are no longer on the server.
#
#Processing:
#        The message is sent to the local OLB so that the manager OLB's are
#        informed that each file is no longer available on this server.
#
#Output: None.
  
sub FileGone {my(@paths) = @_;
    my($file, $iname);

# Get the instance name
#
$iname = shift(@paths) if substr($paths[0],0,1) ne '/';

# Send a message for each path in the path list
#
foreach $file (@paths) {&SendMsg($iname, "gone $file");}
}

#******************************************************************************
#*                              F i l e H e r e                               *
#******************************************************************************
 
#Call:   FileHere([iname,]paths);

#Input:   iname   - is the instance name. If the instance name starts with a
#                   slash, it is treated as a path and the olb is unnamed.
#        @paths   - an array of paths that are available on the server.
#
#Processing:
#        The message is sent to the local OLB so that the manager OLB's are
#        informed that each file is now available on this server.
#
#Output: None.
  
sub FileHere {my(@paths) = @_;
    my($file, $iname);

# Get the instance name
#
$iname = shift(@paths) if substr($paths[0],0,1) ne '/';

# Send a message for each path in the path list
#
foreach $file (@paths) {&SendMsg($iname, "have $file");}
}

#******************************************************************************
#*                               S e n d M s g                                *
#******************************************************************************
 
#Input:  $iname   - instance name, if any
#        $msg     - message to be sent
#
#Processing:
#        The message is sent to the olb udp path indicated in the olbd.pid file
#
#Output: 0 - Message sent
#        1 - Message not sent, could not find the pid file or olb not running
#
#Notes:  1. If an absolute path is given, we check whether the <pid> in the
#           file is still alive. If it is not, then no messages are sent.
  
sub SendMsg {my($iname,$msg) = @_;

# Allocate a socket if we do not have one
#
  return 1 if !fileno(OLBSOCK) && !getSock($iname);

# Get the target if we don't have it
#
  if (!kill(0, $OLBPID))
     {close(OLBSOCK);
      return 1 if !getSock($iname);
     }

# Send the message
#
  print STDERR "OlbNotify: Sending message '$msg'\n" if $DEBUG;
  chomp($msg);
  return 0 if send(OLBSOCK, "$msg\n", 0, $OLBADDR);
  print STDERR "OlbNotify: Unable to send to olb $OLBPID; $!\n" if $DEBUG;
  return 1;
}

#******************************************************************************
#*                              s e t D e b u g                               *
#******************************************************************************
 
#Input:  $dbg     - True if debug is to be turned off; false otherwise.
#
#Processing:
#        The global debug flag is set.
#
#Output: Previous debug setting.

sub setDebug {my($dbg) = @_; 
    my($olddbg) = $DEBUG; 
    $DEBUG = $dbg; 
    return $olddbg;
}

#******************************************************************************
#*                     P r i v a t e   F u n c t i o n s                      *
#******************************************************************************
#******************************************************************************
#*                               g e t S o c k                                *
#******************************************************************************
  
sub getSock {my($iname) = @_;
  my($path);

# Get the path we are to use
#
  return 0 if !($path = getConfig($iname));

# Create the path we are to use
#
  $path = "$path/olbd.notes";
  $path =~ tr:/:/:s;
  $OLBADDR = sockaddr_un($path);

# Create a socket
#
  if (!socket(OLBSOCK, PF_UNIX, SOCK_DGRAM, 0))
     {print STDERR  "OlbNotify: Unable to create socket; $!\n";
      return 0;
     }
  return 1;
}

#******************************************************************************
#*                             g e t C o n f i g                              *
#******************************************************************************
 
sub getConfig {my($iname) = @_;
  my($fn, @phval, $path, $line, $pp1, $pp2);

# Construct possible pid paths
#
  if ($iname eq '')
     {$pp1 = '/tmp/olbd.pid'; $pp2 = '/var/run/olbd/olbd.pid';}
     else
     {$pp1 = "/tmp/$iname/olbd.pid"; $pp2 = "/var/run/olbd/$iname/olbd.pid";}

# We will look for the pid file in one of two locations
#
  if (-r $pp1) {$fn = $pp1;}
     elsif (-r $pp2) {$fn = $pp2;}
        else {print STDERR "OlbNotify: Unable to find olbd pid file\n" if $DEBUG;
              return '';
             }

    if (!open(INFD, $fn))
       {print STDERR "OlbNotify: Unable to open $fn; $!\n" if $DEBUG; return '';}

    @phval = <INFD>;
    close(INFD);
    chomp(@phval);
    $OLBPID = shift(@phval);
    undef($path);
    if (kill(0, $OLBPID))
       {foreach $line (@phval) 
           {($path) = $line =~ m/^&ap=(.*)$/; last if $path;}
        if (!$path)
              {print STDERR "OlbNotify: Can't find olb admin path\n" if $DEBUG;}
       } else {print STDERR "OlbNotify: olbd process $OLBPID dead\n" if $DEBUG;}
    return $path;
}
