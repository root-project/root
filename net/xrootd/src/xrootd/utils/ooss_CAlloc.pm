#!/usr/bin/env perl
#******************************************************************************
#                                                                             *
#                         o o s s _ C A l l o c . p m                         *
#                                                                             *
# (c) 1999 by the Board of Trustees of the Leland Stanford, Jr., University   *
#                          All Rights Reserved                                *
# Produced by Andrew Hanushevsky for Stanford University under contract       *
#            DE-AC03-76-SFO0515 with the Department of Energy                 *
#******************************************************************************

#  $Id: ooss_CAlloc.pm,v 1.6 2008/10/14 09:24:59 furano Exp $

# Perform cache allocation management. This package of routines is usable by
# ooss_Stage as well as ooss_CacheMgr.

#******************************************************************************
#*               C o n f i g u r a t i o n   V a r i a b l e s                *
#******************************************************************************

# The following %Config variables may be modified by the caller *after* this
# module has been included via a use statement. This module aliases %Config in
# in the caller's name space with %Config in this name space (also $ and @).
#
#$Config{cfsrefresh} The number of seconds between fstats
#$Config{cpfile}     The check point file name
#$Config{cpfuzz}     The comparison fuzz factor to use when finding free space
#$Config{minsize}    The minimum allocation size
#$Config{msscmd}     The command to talk to the remote system for metadata ops
#$Config{mssstat}    The arguments to use to find the size of a file
#$Config{purgecmd}   The command to issue to do a one time purge
#$Config{purgeargs}  The arguments minimally with %sz and %fs as substitutions
 
# Establish default configuration values in caller's Config hash
#
    *ooss_CAlloc::Debug = *Debug;     # Use a common debug flag
    *ooss_CAlloc::Config= *Config;    # We use caller's Config hash
    $Config{cachefs}    = '';         # List of usable caches
    $Config{cfsrefresh} = 15*60;      # Refresh every 15 minutes
    $Config{cfsstretch} =  5*60;      # Stretch by 5 minutes if we purged it
    $Config{cfsmaxskip} =  5   ;      # But don't stretch more than 5 times
    $Config{cpfile}     = '/tmp/ooss_CAlloc.cp';
    $Config{cpfuzz}     = 0;          # Fuzz for maximal comaparisons (0 - 100)
    $Config{cpgroup}    = 'public';   # Default cache group to use
    $Config{minsize}    = 0;          # Minmum allocation size
    $Config{msscmd}     = ($Config{isMPS} ? '/opt/xrootd/utils/rxmss'
                                          : '/usr/etc/ooss/rxhpss');
    $Config{mssstat}    = 'statx %fn';
    $Config{noreloc}    = '';         # Paths not to be placed in the cache
    $Config{purgecmd}   = ($Config{isMPS} ? '/opt/xrootd/utils/mps_MigrPurg'
                                          : '/usr/etc/ooss/ooss_MigrPurg');
    $Config{purgeargs}  = '-O -P -e %fs,%sz';

# Establish our own name space
#
package  ooss_CAlloc;
require  Exporter;
@ISA   = qw(Exporter);
@EXPORT= qw(CA_Adjust CA_Config CA_GetFSize CA_Purge CA_SelectFS CA_Start);

# OS-dependent commands
#
$CMDls       = '/bin/ls';

#******************************************************************************
#*                C a c h e   C o n t r o l   V a r i a b l e                 *
#******************************************************************************

 $CacheEnt   =0; # Number of cache entries same as scalar(@CacheRoot);
 $CacheFSxeq =0; # Time at which a cache free space refresh must occur
#@CacheFSTime      Last time we did an fstat for the cache entry
#@CacheFSTskip     Number of times we can stretch an fstat for the entry
#@CacheGroup       Holds the allocation group for each cache entry
#@CacheRoot        Holds the mount point for each cache entry
#@CacheFree        Number of bytes that are free in each corresponding entry
#@CacheLRU         The array index table used for LRU selection
#%CacheIndex       The reverse translation of path to entry index number.
 $CPupdate   =0; # True if cahe pointer file is to be updated.
#@NoReloc          List of paths not to be relocated into the cache

{1}
 
#******************************************************************************
#*                     c a c h e   m a i n t e n a n c e                      *
#******************************************************************************

#******************************************************************************
#*                             C A _ C o n f i g                              *
#******************************************************************************
 
sub CA_Config {my($var, $val) = @_;

# Based on variable do the right thing. Ignore unwanted variables
#
     if ($var eq 'cachefs' || $var eq 'cache')
     {my($c, $grp, $path, $x, @resp);
      if ($var eq 'cache') {($grp, $path, $x) = split(' ', $val, 3)}
         else {($path, $x) = split(' ', $val, 2); $grp = $Config{cpgroup}; }
      if (($c = chop($path)) ne '*') {$resp[0] = $path.$c;}
         else {@resp = `$CMDls -d1 $path*`; chomp(@resp);}
      foreach $path (@resp)
         {&Trim('/', $path);
          return "Directory '$path' not found." if !-d $path;
          return "Duplicate cache path, '$path'." if $CacheIndex{$path};
          push(@CacheGroup, $grp);
          push(@CacheRoot, $path);
          $CacheIndex{$path} = $#CacheRoot;
          $CacheEnt++;
         }
     }
   elsif ($var eq 'noreloc')
     {my($path, $x);
      ($path, $x) = split(' ', $val, 2);
      &Trim('/', $path);
      push(@NoReloc, $path);
     }
  return '';
}

#******************************************************************************
#*                                                                            *
#*                              C A _ S t a r t                               *
#*                                                                            *
#******************************************************************************
 
sub CA_Start {
    my($i, $cp, $om); my($emsg) = '';

# If there are no external filesystems, then we are all done here
#
  return '' if !$CacheEnt;

# Initialize the free space statistics
#
  &InitFStat();

# Attempt to read back the LRU value that we saved
#
  if ($Config{cpfile})
     {if (-e $Config{cpfile}) {$om = '+<'}
         else {$om = '+>'}
      if (!open(CACHEPFD, $om.$Config{cpfile})
      || (!defined(read(CACHEPFD, $buff, 256) ) ) )
         {$emsg ="Unable to process file $Config{cpfile}; $!.";
         } else {($cp) = $buff =~ m/&cap=(\d*)/; $CPupdate = 1;
                 my($fd) = select(CACHEPFD); $|=1; select($fd);
                }
     }

# Fix up the cache pointer
#
  if (!defined($cp) || $cp eq '' || $cp >= $CacheEnt) {$cp = 0}
#    else {$cp = ($cp+1)%$CacheEnt}
  if ($Debug)
     {&Say($CacheEnt, 'caches; last=', $cp, 'cache list:');
      for ($i = 0; $i < $CacheEnt; $i++)
          {&Say($i, $CacheRoot[$i], $CacheFSpace[$i]);}
     }

# Construct the LRU table based on the root value
#
  for ($i = 0; $i < $CacheEnt; $i++)
      {$CacheLRU[$i] = (++$cp)%$CacheEnt;}
  &Say('LRU order =', @CacheLRU) if $Debug;
  return $emsg;
}

#******************************************************************************
#*                             C A _ A d j u s t                              *
#******************************************************************************
 
sub CA_Adjust {my($path, $sz) = @_;
    my($cp);

# Adjust the free space if path actually exists
#
  $cp = $CacheIndex{$path};
  if (defined($cp)) {$CacheFSpace[$cp] += $sz;}
     else {return "Can't adjust by $sz; unknown path - $path"}
  return '';
}

#******************************************************************************
#*                           C A _ G e t F S i z e                            *
#******************************************************************************

sub CA_GetFSize {my($fn) = @_;
    my($cmd, $resp, $rc, $x);

# Construct the command to send to the remote host to get back the filesize
#
  $cmd = $Config{mssstat};
  $cmd =~ s/%fn/$fn/;
  $cmd = $Config{msscmd}.' '.$cmd;

# Issue the command and ret the response in the from of <rc>\n<statline>
#
  &Say("Executing: $cmd\n") if $Debug;
  $resp = `$cmd`;
  &Say("Response:  $resp")   if $Debug;
  ($rc, $resp, $x) = split("\n", $resp);
  if ($rc != 0)
     {&Say($resp) if defined($resp) && $resp ne '';
      return undef;
     }

# Split apart the response and return the size
#
  my($type, $mode, $lnk, $uid, $gid, $at, $mt, $ct, $sz, $blk, $blks) =
    split(' ', $resp);
  if (&IsNum($sz)) {return $sz}
  &Say("Invalid response from $cmd - $resp");
  return 2*1024*1024*1024;
}
 
#******************************************************************************
#*                              C A _ P u r g e                               *
#******************************************************************************

sub CA_Purge {my($path, $sz, $cp) = @_;
    my($cmd, $resp, $fst);

# Construct a command
#
  $cmd = $Config{purgeargs};
  $cmd =~ s/%sz/$sz/;
  $cmd =~ s/%fs/$path/;
  $cmd = $Config{purgecmd}.' '.$cmd;

# Now issue the command
#
  $resp = `$cmd 2>&1`;
  if ($?) {return "Unable to obtain $sz free space in $path; $resp"}
     elsif ($resp) {&Say($resp)}

# Add amount purged into the free space for this filesystem if called with cp
# And make sure we wait about 5 minutes before refreshing free space but not
# beyond the actual stretch quantity
#
  if (defined($cp))
     {$CacheFSpace[$cp] += $sz;
      $CacheFSTime[$cp] += (time()+ $Config{cfsstretch})
            if ($CacheFSTskip-- > 0);
     }
  return '';
}
 
#******************************************************************************
#*                           C A _ S e l e c t F S                            *
#******************************************************************************
 
sub CA_SelectFS {my($base, $sz, $cgrp) = @_;
    my($fsz);

# If we have multiple caches, perform multiple file system selection. However,
# skip any paths that are not to be relocated into the cache area.
#
  return &SelectCache($base, $sz) if $CacheEnt && (!scalar(@NoReloc)
                                     || !&InList($base, \@NoReloc));


# If a size was specified, then we must see if we can actually fit the file
#
  $sz = $Config{minsize} if !$sz;
  if ($sz)
     {if (&Free_Space($base) <= $sz)
         {return (8, "Cannot obtain $sz free space in $base")
                 if &CA_Purge($CacheRoot[$cp], $sz, $cp);
         }
     }

# All done, return the obvious to the caller
#
  return ('', $base, 0);
}

#******************************************************************************
#*                           S e l e c t C a c h e                            *
#******************************************************************************
 
sub SelectCache {my($base, $sz, $cgrp) = @_;
    my($cp, $stype);

# Check if we should get new free space statistics
#
    &InitFStat() if (time() > $CacheFSxeq);

# Select correct cache group
#
  $cgrp = $Config{cpgroup} if !$cgrp;

# Determine selection algorithm to use
#
  $stype = 'first';
     if ($CacheEnt == 1) {$cp = 0; $stype = 'one';}
  elsif ($sz == 0 || ($cp = Select_First($sz)) < 0)
        {if ($Config{cpfuzz}) {$cp = Select_Maximal(); $stype = 'maximal';}
            else              {$cp = Select_Maximum(); $stype = 'maximum';}
        }

# Update the cursor
#
  &Say("sel $stype cache", $cp, $CacheRoot[$cp], 'LRU=', @CacheLRU) if $Debug;
  &Update_Cursor($cp);

# Verify that this file will fit
#
  $sz = $Config{minsize} if !$sz;
  if ($sz && $CacheFSpace[$cp] <= $sz)
     {return (8, "Unable to obtain $sz free cache space in $CacheRoot[$cp]")
             if &CA_Purge($CacheRoot[$cp], $sz);
     }

  return ('', $CacheRoot[$cp], 1);
}

#******************************************************************************
#*                  S e l e c t i o n   A l g o r i t h m s                   *
#******************************************************************************

# Select the first cache that will accomodate stated size. If none, return -1
#
sub Select_First {my($sz) = @_;
    my($i,$j);

# Find the first fit partition
#
  for ($i = 0; $i < $CacheEnt; $i++)
      {$j = $CacheLRU[$i]; last if ($CacheFSpace[$j] > $sz);}

# Check if we actually found something
#
  return -1 if $i >= $CacheEnt;
  push(@CacheLRU, splice(@CacheLRU, $i, 1));
  return $j;
}

# Select the entry that has a maximal amount of space
#
sub Select_Maximal {
    my($cpent, $cpmax, $maxsz, $cpsz, $dfsz, $fzsz, $i, $cp, $fuzz);

# Assume that the first entry is the maximum entry
#
  $cpent = 0;
  $cpmax = $CacheLRU[$cpent];
  $maxsz = $CacheFSpace[$cpmax];
  $fuzz  = $Config{cpfuzz}/100;

# Now find the right entry
#
  for ($i = 1; $i < $CacheEnt; $i++)
      {$cp   = $CacheLRU[$i];
       if ($cpsz = $CacheFSpace[$cp])
          {$dfsz  = abs(($cpsz-$maxsz)/($cpsz+$maxsz));
           print "CAlloc: Lookat $cpmax vs $cp diff=$dfsz\n" if ($Debug);
           if ($dfsz > $fuzz)
              {$cpent = $i; $cpmax = $cp; $maxsz = $cpsz;}
          }
      }
  print "CAlloc: Select $cpmax\n" if ($Debug);

# Fixup the LRU table and return the result
#
  push(@CacheLRU, splice(@CacheLRU, $cpent, 1));
  return $cpmax;
}

# Select the entry that has a maximum amount of space
#
sub Select_Maximum {
    my($cpent, $cpmax, $maxsz, $cpsz, $i, $cp);

# Assume that the first entry is the maximum entry
#
  $cpent = 0;
  $cpmax = $CacheLRU[$cpent];
  $maxsz = $CacheFSpace[$cpmax];

# Now find the right entry
#
  for ($i = 1; $i < $CacheEnt; $i++)
      {$cp   = $CacheLRU[$i];
       $cpsz = $CacheFSpace[$cp];
       if ($cpsz > $maxsz)
          {$cpent = $i; $cpmax = $cp; $maxsz = $cpsz;}
      }

# Fixup the LRU table and return the result
#
  push(@CacheLRU, splice(@CacheLRU, $cpent, 1));
  return $cpmax;
}
 
#******************************************************************************
#*                F r e e   S p a c e   M a i n t e n a n c e                 *
#******************************************************************************

sub InitFStat {
    my($i);
    my($nowtime) = time();
    my($refwait) = $Config{cfsrefresh};
    $CacheFSxeq  = $nowtime + $refwait;

    for ($i = 0; $i <  $CacheEnt; $i++)
        {if ( $nowtime > $CacheFSTime[$i] )
            {$CacheFSpace[$i] = &Free_Space($CacheRoot[$i]);
             $nowtime = time();
             $CacheFSTime[$i] = $nowtime + $refwait;;
             $CacheFSTskip[$i]= $Config{cfsmaxskip};
            }
            elsif ($CacheFSTime[$i] < $CacheFSxeq) 
                         {$CacheFSxeq = $CacheFSTime[$i]}
        }
}
 
sub Free_Space {my($path) = @_;
    my($resp,$x);
    $resp = `$Config{_fs_stat} $path`;
    if ($?)
       {&Say("$Config{_fs_stat} returned error $?.");
        return 2<<30;
       }
    ($resp,$x) = split(' ',$resp, 2);
    return $resp;
    }

#******************************************************************************
#*                             U t i l i t i e s                              *
#******************************************************************************

sub IsNum  {my($v) = @_; return ($v =~ m/^[0-9]+$/);}

sub Say {my(@msg) = @_; print STDERR 'ooss_CAlloc: ', join(' ', @msg), "\n";}

sub Trim {my($cc, $str) = @_;
    my($x);
    do {$x = chop($str);} until $x ne $cc;
    $str .= $x;
}

sub Update_Cursor {my($cval) = @_;
    my($buff);

# Lock the cursor file (if it's open) and update the value
#
  if ($CPupdate)
     {flock(CACHEPFD, 2);
      $buff =  "&cap=$cval";
      seek(CACHEPFD, 0, 0);
      print CACHEPFD $buff;
      seek(CACHEPFD, length($buff), 0);
      flock(CACHEPFD, 8);
     }
}

sub InList {local($tdir, *paths) = @_;
    my($ld);
    foreach $ld (@paths) {return 1 if ( ($tdir =~ m/^$ld/) )}
    return 0;
}
