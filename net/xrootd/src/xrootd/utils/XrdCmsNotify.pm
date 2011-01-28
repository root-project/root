#******************************************************************************
#   $Id: XrdCmsNotify.pm,v 1.3 2010/05/01 23:51:16 abh Exp $
#*                                                                            *
#*                       X r d C m s N o t i f y . p m                        *
#*                                                                            *
# (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University   *
#                          All Rights Reserved                                *
# Produced by Andrew Hanushevsky for Stanford University under contract       *
#            DE-AC03-76-SFO0515 with the Department of Energy                 *
#******************************************************************************
  
#!/usr/bin/env perl

package  XrdCmsNotify;
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
#                   slash, it is treated as a path and the cmsd is unnamed.
#        @paths   - an array of paths that are no longer on the server.
#
#Processing:
#        The message is sent to the local cmsd so that the manager cmsd's are
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
#                   slash, it is treated as a path and the cmsd is unnamed.
#        @paths   - an array of paths that are available on the server.
#
#Processing:
#        The message is sent to the local Ocmsd so that the manager cmsd's are
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
#        The message is sent to the cmsd udp path indicated in the cmsd.pid file
#
#Output: 0 - Message sent
#        1 - Message not sent, could not find the pid file or cmsd not running
#
#Notes:  1. If an absolute path is given, we check whether the <pid> in the
#           file is still alive. If it is not, then no messages are sent.
  
sub SendMsg {my($iname,$msg) = @_;

# Allocate a socket if we do not have one
#
  return 1 if !fileno(CMSSOCK) && !getSock($iname);

# Get the target if we don't have it
#
  if (!kill(0, $CMSPID))
     {close(CMSSOCK);
      return 1 if !getSock($iname);
     }

# Send the message
#
  print STDERR "CmsNotify: Sending message '$msg'\n" if $DEBUG;
  chomp($msg);
  return 0 if send(CMSSOCK, "$msg\n", 0, $CMSADDR);
  print STDERR "CmsNotify: Unable to send to cmsd $CMSPID; $!\n" if $DEBUG;
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
  $CMSADDR = sockaddr_un($path);

# Create a socket
#
  if (!socket(CMSSOCK, PF_UNIX, SOCK_DGRAM, 0))
     {print STDERR  "CmsNotify: Unable to create socket; $!\n";
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
     {$pp1 = '/tmp/cmsd.pid'; $pp2 = '/var/run/cmsd/cmsd.pid';}
     else
     {$pp1 = "/tmp/$iname/cmsd.pid"; $pp2 = "/var/run/cmsd/$iname/cmsd.pid";}

# We will look for the pid file in one of two locations
#
  if (-r $pp1) {$fn = $pp1;}
     elsif (-r $pp2) {$fn = $pp2;}
        else {print STDERR "CmsNotify: Unable to find cmsd pid file\n" if $DEBUG;
              return '';
             }

    if (!open(INFD, $fn))
       {print STDERR "CmsNotify: Unable to open $fn; $!\n" if $DEBUG; return '';}

    @phval = <INFD>;
    close(INFD);
    chomp(@phval);
    $CMSPID = shift(@phval);
    undef($path);
    if (kill(0, $CMSPID))
       {foreach $line (@phval) 
           {($path) = $line =~ m/^&ap=(.*)$/; last if $path;}
        if (!$path)
              {print STDERR "CmsNotify: Can't find cmsd admin path\n" if $DEBUG;}
       } else {print STDERR "CmsNotify: cmsd process $CMSPID dead\n" if $DEBUG;}
    return $path;
}
