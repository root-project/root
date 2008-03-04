#!/usr/local/bin/perl -w

# $Id$

if ( @ARGV<1 ) {
    print "Expected argument: rt log directory\n";
    exit;
}

$rtDir = $ARGV[0];

if ( ! -e $rtDir ) {
    print "Invalid arg - not a directory\n";
    exit;
}

$o1 = `grep '^o' $rtDir/rtLog_ver002.txt        | tail -1 | awk '{ print \$2 }'`;
$o2 = `grep '^o' $rtDir/rtLog_ver002.txt.backup | tail -1 | awk '{ print \$2 }'`;
$u1 = `grep '^u' $rtDir/rtLog_ver002.txt        | tail -1 | awk '{ print \$2 }'`;
$u2 = `grep '^u' $rtDir/rtLog_ver002.txt.backup | tail -1 | awk '{ print \$2 }'`;

chop($o1);chop($o2);chop($u1);chop($u2);

$oMax = 1;
$uMax = 1;

if ( $o1 =~ /[0-9]+/ ) { if ( $o1 > $oMax ) { $oMax = $o1; } }
if ( $o2 =~ /[0-9]+/ ) { if ( $o2 > $oMax ) { $oMax = $o2; } }
if ( $u1 =~ /[0-9]+/ ) { if ( $u1 > $uMax ) { $uMax = $u1; } }
if ( $u2 =~ /[0-9]+/ ) { if ( $u2 > $uMax ) { $uMax = $u2; } }

open  jnlF, "> $rtDir/rtMax.jnl";
print jnlF "o $oMax", "\n", "u $uMax";
close jnlF;

unlink "$rtDir/rtRunning.flag";
