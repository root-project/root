#!/usr/bin/env perl

#******************************************************************************
#*                                                                            *
#*                          o o s s _ M o n P . p m                           *
#                                                                             *
# (c) 2002 by the Board of Trustees of the Leland Stanford, Jr., University   *
#                          All Rights Reserved                                *
# Produced by Andrew Hanushevsky for Stanford University under contract       *
#            DE-AC03-76-SFO0515 with the Department of Energy                 *
#******************************************************************************

# $Id: ooss_MonP.pm,v 1.4 2008/10/14 09:25:04 furano Exp $

# Monitor the parent process. If it stalls, optionally restart it.


# Establish our own name space
#
package  ooss_MonP;
require  Exporter;
@ISA   = qw(Exporter);
@EXPORT= qw(Monitor Poke);

#******************************************************************************
#*                      G l o b a l   V a r i a b l e s                       *
#******************************************************************************

use Fcntl;

$ooss_MonPValue = 0;
$ooss_MonPFname = '';
$ooss_MonProg   = '';

# OS-Dependent commands
#
$CMDlogger = '/usr/bin/logger';

{1}
  
#******************************************************************************
#*                               M o n i t o r                                *
#******************************************************************************
 
# Monitor takes two arguments:
#         $Seconds    - The maximum number of seconds that can go by without
#                       the parent calling Poke().
#         $RestartCMD - The command to use to restart the parent. If it is not
#                       specified, a stalled process is killed but not restarted
#
# Upon success, Monitor() returns a 1 (true). Otherwise, an error is indicated.
#      The caller must call Poke no more than every $Seconds; otherwise the
#      calling process will be restarted.
#
sub Monitor {my($Seconds, @RestartCMD) = @_;
    my($pid, $ppid, $pval, $nval, $deadline);

# Make sure we are called only once
#
  return Log('Monitor() called more than once.') if $ooss_MonPFname;
  $ooss_MonProg = ($RestartCMD[0] ? join(' ', @RestartCMD) : "process $$");

# Create a file that we will use as a communications channel
#
  $ooss_MonPFname = "/tmp/ooss_MonP.$$";
  if (!sysopen(OOSSFD, $ooss_MonPFname, O_CREAT|O_RDWR|O_TRUNC))
     {Log("Unable to create $ooss_MonPFname; $!");
      $ooss_MonPFname = '';
      return 0;
     }
  unlink($ooss_MonPFname);

# Do the Initial Poke
#
  if (!Poke())
     {$ooss_MonPFname = '';
      return 0;
     }

# Now fork the monitor process
#
  return 1 if $pid = fork();
  if (!defined($pid))
     {Log("Unable to fork monitor for $MonProg; $!");
      Cleanup(-1);
      return 0;
     }
  $ppid = getppid();

# Simply sleep between check for pokes.
#
  $sleeptime = (15 < $Seconds ? $Seconds : 15);
  $pval = pack('L', $ooss_MonPValue);
  while(1)
       {$deadline = time() + $Seconds;
        do {sleep($sleeptime);
            Cleanup(0) if !kill(0, $ppid);
           } until (time() >= $deadline);

        # Read the current value in the poke file
        #
        if ((!sysseek(OOSSFD,0,0) 
        ||   !sysread(OOSSFD, $nval, 4, 0)) && $!)
           {Log("Unable to read $ooss_MonPFname; $!");
            Cleanup(8);
           }

        # Make sure we got poked
        #
        if ($nval eq $pval)
           {Log("Process $ppid appears to be inactive");
            kill(9, $ppid);
            Cleanup(-1);

            # Restart the command if we have one to restart
            #
              if ($RestartCMD[0] ne '')
                 {Log("Restarting $ooss_MonProg");
                  exec(@RestartCMD);
                  Log("Unable to exec $ooss_MonProg; $!");
                  Cleanup(16);
                 }
            exit(1);
           }
        $pval = $nval;
       }
}

#******************************************************************************
#*                                  P o k e                                   *
#******************************************************************************
 
sub Poke {

# Make sure Monitor was called
#
  return Log("Monitor() not called prior to Poke()") if !$ooss_MonPFname;

# Simply write a new value into the poke file
#
  my($pval) = pack('L', ++$ooss_MonPValue);
  if (!sysseek(OOSSFD, 0, 0) || !syswrite(OOSSFD, $pval, length($pval), 0))
     {Log("Unable to write into $ooss_MonPFname; $!");
      Cleanup(-1);
      return 0;
     }

# All done
#
  return 1;
}

#******************************************************************************
#*                               C l e a n u p                                *
#******************************************************************************
 
sub Cleanup {my($rc) = @_;

close(OOSSFD);
$ooss_MonPValue = 0;
$ooss_MonPFname = '';
return 0 if $rc < 0;
exit($rc);
}

#******************************************************************************
#*                                   L o g                                    *
#******************************************************************************
 
sub Log {my($msg) = @_;

    print STDERR "ooss_MonP: $msg\n";
    system('$CMDlogger', '-p', 'daemon.notice', "ooss_MonP: $msg");
    return;
}
