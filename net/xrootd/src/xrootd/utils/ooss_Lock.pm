#!/usr/bin/env perl
#******************************************************************************
#                                                                             *
#                           o o s s _ L o c k . p m                           *
#                                                                             *
# (c) 1999 by the Board of Trustees of the Leland Stanford, Jr., University   *
#                          All Rights Reserved                                *
# Produced by Andrew Hanushevsky for Stanford University under contract       *
#            DE-AC03-76-SFO0515 with the Department of Energy                 *
#******************************************************************************

#  $Id: ooss_Lock.pm,v 1.6 2008/10/14 09:25:01 furano Exp $

# Perform cache lock management. This package of routines is usable by
# ooss_Stage as well as ooss_CacheMgr.

#******************************************************************************
#*               C o n f i g u r a t i o n   V a r i a b l e s                *
#******************************************************************************

# The following %Config variables may be modified by the caller *after* this
# module has been included via a use statement. This module aliases %Config in
# in the caller's name space with %Config in this name space (also $ and @).
# Here we establish default configuration values in caller's Config hash
#
    *ooss_Lock::Config = *Config;     # We use caller's Config hash
    $Config{debuglk}   = 0;           # Debug locking code
    $Config{lockfn}    = 'DIR_LOCK';  # The name of the directory lock file
    $Config{locktries} = 60;          # Maximum number of tries we will do;
    $Config{lockwait}  = 60;          # Number of seconds to wait between tries

# Establish our own name space
#
package  ooss_Lock;
require  Exporter;
@ISA   = qw(Exporter);
@EXPORT= qw(Lock LockWr Lock_Stats Lock_Time UnLock);

use Fcntl;

#******************************************************************************
#*                      G l o b a l   V a r i a b l e s                       *
#******************************************************************************
 
$LKFHREV = 0;                # Unique filehandle generator
$LK_SFX  = '.lock';          # The suffix for base filename locks

#LOCK_SH = 1;                # Shared lock
$LOCK_EX = 2;                # Exclusive (only when open for write)
$LOCK_NB = 4;                # Non-blocking
$LOCK_UN = 8;                # Unlock request

$LOCK_tm = 0;                # Time  to obtain a lock (see Lock_Stats())
$LOCK_ct = 0;                # Tries to obtain a lock (see Lock_Stats())

$EMSG    = '';               # Message to return upon error

{1}
  
#******************************************************************************
#*                                  L o c k                                   *
#******************************************************************************
 
sub Lock {my($Tdir, $fn) = @_;
    my($lktod, $c, $lkfname, $lkdname);
    my($Dir_LK)    = '';             # File handle for the directory lock
    my($Base_LK)   = '';             # File handle for the data file lock
    $EMSG          = '';             # No errors

# Construct target filename
#
  if ( ($c = chop($Tdir)) ne '/') {$Tdir .= "$c/"}
     else {$Tdir .= '/'}
  if ($fn) {$trg_fn = $Tdir.$fn}
     else  {$trg_fn = ''}

# Construct lock file names
#
  $lkdname = $Tdir.&mush($Config{lockfn});
  $lkfname =       &mush($trg_fn).$LK_SFX if $trg_fn;

# Now start the process to stage in the file
#
  $LOCK_ct = $Config{locktries}; $lktod = time();
  while($LOCK_ct--) {

      # Lock the target directory. This may be all that is wanted.
      #
        if (!($Dir_LK = &LockFile($lkdname, $LOCK_EX, 0))
        ||  !$trg_fn)
           {$Base_LK = $Dir_LK; $Dir_LK = ''; last;}
      
      # At this point obtain an exclusive lock on the base lock file,
      #
        last if ($Base_LK = &LockFile($lkfname, $LOCK_EX, $LOCK_NB));

      # We could not get the lock. Unlock the directory lock, wait for a lock
      # on the base file and then try the protocol sequence again
      #
        $Dir_LK = &UnLock($Dir_LK); $Dir_LK = '';
        if (!($Base_LK = &LockFile($lkfname, $LOCK_EX)))
           {sleep($Config{lockwait});}
           else {&UnLock($Base_LK); $Base_LK = '';}
      }

# Locking done (successful or not). Cleanup and return to the caller.
#
  $Dir_LK = &UnLock($Dir_LK) if $Dir_LK;
  $LOCK_tm = time() - $lktod;
  return ($Base_LK, $EMSG);
}

#******************************************************************************
#*                            L o c k _ S t a t s                             *
#******************************************************************************
 
sub Lock_Stats {return ($LOCK_tm, $Config{locktries}-$LOCK_ct)}

#******************************************************************************
#*                             L o c k _ T i m e                              *
#******************************************************************************
 
sub Lock_Time {my($path, $now) = @_; utime($now, $now, &mush($path).$LK_SFX);}

#******************************************************************************
#*                                U n L o c k                                 *
#******************************************************************************

sub UnLock {my($fh) = @_;
    Flock($fh, $LOCK_UN);
    close($fh);
    &Say("Unlock $fh.") if $Config{debuglk};
    return '';
    }
 
#******************************************************************************
#*                      l o c k i n g   r o u t i n e s                       *
#******************************************************************************
sub LockWr {
    my($fn, $lktype, $noblock) = @_;
    $EMSG = '';             # No errors
    my($fh) = &LockFile($fn,$lktype,$noblock);
    return (*$fh,$EMSG);
}
  
sub LockFile {my($fn, $lktype, $noblock) = @_;
    my($fh, $omode, $lkmode);

    # Generate a unique file handle
    #
    $LKFHREV++;
    $fh = 'LKFH'.$$.$LKFHREV;

    # Determine how we will open the file. For exclusive locks, open in append
    # mode since that will not change a/m/ctimes. Otherwise, open as r/o.
    #
    ($omode, $lkmode) = ($lktype != $LOCK_EX ? ('<', 'shr') : ('>>', 'xcl'));

    # Open the target file
    #
    if (!open($fh, $omode.$fn))
       {$EMSG = "Open '$fn' ".($omode eq '<' ? 'r/o' : 'r/w')." failed; $!.";
        return '';
       }

    # Lock the file appropriately
    #
    if (!Flock($fh, $lktype+$noblock))
       {$EMSG  = "Lock $lkmode failed for '$fn'; $!." if !$noblock;
        close($fh);
        $LKFHREV--;
        return '';
       }
    &Say("Locked $fh mode $lkmode file $fn.") if $Config{debuglk};
    seek $fh,0,2 if $lktype == $LOCK_EX;
    return $fh;
    }

#******************************************************************************

sub Flock {my($fh, $flags) = @_;
my($lk_type, $lk_mode, $lk_parms);

# Determine lock mode.
#
   if ($flags & 4) {$lk_mode = F_SETLK}
      else         {$lk_mode = F_SETLKW}

# Determine lock type.
#
   if ($flags & 8) {$lk_type = F_UNLCK}
elsif ($flags & 1) {$lk_type = F_RDLCK}
else               {$lk_type = F_WRLCK}

# Construct the parameter list and perform lock function.
#
$lk_parms = pack('sslllll', $lk_type, 0, 0, 0, 0, 0, 0);
return fcntl($fh, $lk_mode, $lk_parms);
}

#******************************************************************************
#*                             u t i l i t i e s                              *
#******************************************************************************
 
sub Say {my(@msg) = @_; print(STDERR 'ooss_Lock: ', join(' ', @msg), "\n");}

sub mush {my($fn) = @_; my($path, $fname);
          return $fn if !$Config{hidden};
          return '.' if $fn eq '';
          ($path, $fname) = $fn =~ m:^(.*/)(.*):g;
          return '.'.$fn if $path eq '';
          return $path.'.'.$fname;
}
