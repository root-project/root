#!/usr/local/bin/perl -w

use Cwd;
# take care of arguments

if ( @ARGV == 2 ) {
    $configFile = $ARGV[0];
    $action = $ARGV[1];
} else {
    &printUsage('start', 'stop');
}

if ($action eq 'start' ) {
    &readConfigFile($configFile, 'collector', 1);
    &cleanup();
    chdir "${baseDir}/${thisSite}" or die ("Can't change dir to ${baseDir}/${thisSite}");
    mkdir ("${baseDir}/${thisSite}/logs", 0755);
    mkdir ("${baseDir}/${thisSite}/logs/out", 0755);
    $ts = &timestamp();
    $date = join '-', split(/ /,$ts);
    $logFile = "${baseDir}/${thisSite}/logs/out/xrdmonCollector.$date";
    system("xrdmonCollector -port $ctrPort > $logFile 2>&1 &");
} elsif ( $action eq 'stop' ) {
    &readConfigFile($configFile, 'collector', 0);
    $psLine = `ps -ef | grep xrdmonCollector  | grep $ctrPort |grep -v grep`;
    if ( ! $psLine ) {
             die "xrdmonCollector -p $ctrPort not running \n";
        }
    `touch $baseDir/$thisSite/logs/rt/rtLog.txt.stop`;
    print "This might take up to 120 s. \n";
    for ( $i = 0; $i < 120; $i++ ) {
        sleep 1;
        $psLine = `ps -ef | grep xrdmonCollector  | grep $ctrPort |grep -v grep`;
        chomp $psLine;
#        print "psLine: $psLine \n";
        if ( ! $psLine ) {
             die "xrdmonCollector -p $ctrPort is stopped \n";
        }
    }

    (@parts) = split(/ /,$psLine);
    $PID = $parts[4];
    print "PID: $PID \n";
   `kill -9 $PID`; 
    &cleanup();
} else {
    &printUsage('start', 'stop');
}

sub cleanup () {
    $rtDir = "$baseDir/$thisSite/logs/rt";

    if ( ! -e $rtDir ) {
       return;
    }
 
    $oMax = 1;
    $uMax = 1;

    if (-e "$rtDir/rtLog.txt") {
        $o1 = `grep '^o' $rtDir/rtLog.txt        | tail -1 | awk '{ print \$2 }'`;
        chop($o1);
        if ( $o1 =~ /[0-9]+/ ) { if ( $o1 > $oMax ) { $oMax = $o1; } }

        $u1 = `grep '^u' $rtDir/rtLog.txt        | tail -1 | awk '{ print \$2 }'`;
        chop($u1);
       if ( $u1 =~ /[0-9]+/ ) { if ( $u1 > $uMax ) { $uMax = $u1; } }
    }
    if ( -e "$rtDir/rtLog.txt.backup") {
        $o2 = `grep '^o' $rtDir/rtLog.txt.backup | tail -1 | awk '{ print \$2 }'`;
        chop($o2);
        if ( $o2 =~ /[0-9]+/ ) { if ( $o2 > $oMax ) { $oMax = $o2; } }

        $u2 = `grep '^u' $rtDir/rtLog.txt.backup | tail -1 | awk '{ print \$2 }'`;
        chop($u2);
        if ( $u2 =~ /[0-9]+/ ) { if ( $u2 > $uMax ) { $uMax = $u2; } }
    }

    open  jnlF, "> $rtDir/rtMax.jnl";
    print jnlF "o $oMax", "\n", "u $uMax";
    close jnlF;

    unlink "$rtDir/rtRunning.flag";
}
sub printUsage() {
    $opts = join('|', @_);
    die "Usage: $0 <configFile> $opts \n";
}     
sub readConfigFile() {
    my ($confFile, $caller, $print) = @_;
    unless ( open INFILE, "< $confFile" ) {
        print "Can't open file $confFile\n";
        exit;
    }

    $dbName = "";
    $mySQLUser = "";
    $webUser = "";
    $baseDir = "";
    $thisSite = "";
    $ctrPort = 9930;
    $backupIntDef = "1 DAY";
    $fileCloseWaitTime = "10 MINUNTE";
    $maxJobIdleTime = "15 MINUNTE";
    $maxSessionIdleTime = "12 HOUR";
    $maxConnectTime = "70 DAY";
    $closeFileInt = "15 MINUTE";
    $closeIdleSessionInt = "1 HOUR";
    $closeLongSessionInt = "1 DAY";
    $mysqlSocket = '/tmp/mysql.sock';
    $nTopPerfRows = 20;
    $yearlyStats = 0;
    $allYearsStats = 0;
    @flags = ('OFF', 'ON');
    @sites = ();


    while ( <INFILE> ) {
        chomp();
        my ($token, $v1, $v2, $v3, $v4) = split;
        if ( $token eq "dbName:" ) {
            $dbName = $v1;
        } elsif ( $token eq "MySQLUser:" ) {
            $mySQLUser = $v1;
        } elsif ( $token eq "webUser:" ) {
            $webUser = $v1;
        } elsif ( $token eq "MySQLSocket:" ) {
            $mysqlSocket = $v1;
        } elsif ( $token eq "baseDir:" ) {
            $baseDir = $v1;
        } elsif ( $token eq "ctrPort:" ) {
            $ctrPort = $v1;
        } elsif ( $token eq "thisSite:" ) {
            $thisSite = $v1;
        } elsif ( $token eq "site:" ) {
            push @sites, $v1;
            $timezones{'$v1'} = $v2;
            $firstDates{'$v1'} = "$v3 $v4";
        } elsif ( $token eq "backupIntDef:" ) {
            $backupIntDef = "$v1 $v2";
        } elsif ( $token eq "backupInt:" ) {
            $backupInts{$v1} = "$v2 $v3";
        } elsif ( $token eq "fileType:" ) {
            push @fileTypes, $v1;
            $maxRowsTypes{$v1} = $v2;
        } elsif ( $token eq "fileCloseWaitTime:" ) {
            $fileCloseWaitTime = "$v1 $v2";
        } elsif ( $token eq "maxJobIdleTime:" ) {
            $maxJobIdleTime = "$v1 $v2";
        } elsif ( $token eq "maxSessionIdleTime:" ) {
            $maxSessionIdleTime = "$v1 $v2";
        } elsif ( $token eq "maxConnectTime:" ) {
            $maxConnectTime = "$v1 $v2";
        } elsif ( $token eq "closeFileInt:" ) {
            $closeFileInt = "$v1 $v2";
        } elsif ( $token eq "closeIdleSessionInt:" ) {
            $closeIdleSessionInt = "$v1 $v2";
        } elsif ( $token eq "closeLongSessionInt:" ) {
            $closeLongSessionInt = "$v1 $v2";
        } elsif ( $token eq "nTopPerfRows:" ) {
            $nTopPerfRows = $v1;
        } elsif ( $token eq "yearlyStats:" ) {
            if ( lc($v1) eq "on" ) {$yearlyStats = 1;}
        } elsif ( $token eq "allYearsStats:" ) {
            if ( lc($v1) eq "on" ) {$allYearsStats = 1;}
        } else {
            print "Invalid entry: $_ \n";
            close INFILE;
            exit;
        }
    }
    close INFILE;
    # check missing tokens
    @missing = ();
    # $baseDir required for all callers except create
    if ( $caller ne "create" and ! $baseDir ) {
         push @missing, "baseDir";
    }

    if ( $caller eq "collector" ) {
         if ( ! $thisSite) {
            push @missing, "thisSite";    
         }
    } else { 
         if ( ! $dbName ) {push @missing, "dbName";}
         if ( ! $mySQLUser ) {push @missing, "mySQLUser";}
    }

    if ( @missing > 0 ) {
       print "Follwing tokens are missing from $confFile \n";
       foreach $token ( @missing ) {
           print "    $token \n";
       }
       exit
    }
    if ( $print ) {
        if ( $caller eq "collector" ) {
             print "  baseDir: $baseDir \n";
             print "  ctrPort: $ctrPort \n";
             print "  thisSite: $thisSite \n";
             return;
        }
        print "  dbName: $dbName  \n";
        print "  mySQLUser: $mySQLUser \n";
        print "  mysqlSocket: $mysqlSocket \n";
        print "  nTopPerfRows: $nTopPerfRows \n";
        if ( $caller eq "create" ) {
             print "backupIntDef: $backupIntDef \n";
             foreach $site ( @sites ) {
                 print "site: $site \n";
                 print "     timeZone: $timezones{$site}  \n";
                 print "     firstDate: $firstDates{$site} \n";
                 if ( $backupInts{$site} ) {
                     print "backupInt: $backupInts{$site} \n";
                 }
             }
             foreach $fileType ( @fileTypes ) {
                 print "fileType: $fileType $maxRowsTypes{$fileType} \n";
             }
         } else {
             print "  baseDir: $baseDir \n";
             print "  fileCloseWaitTime: $fileCloseWaitTime \n";
             print "  maxJobIdleTime: $maxJobIdleTime \n";
             print "  maxSessionIdleTime: $maxSessionIdleTime \n";
             print "  maxConnectTime: $maxConnectTime \n";
             print "  closeFileInt: $closeFileInt \n";
             print "  closeIdleSessionInt: $closeIdleSessionInt \n";
             print "  closeLongSessionInt: $closeLongSessionInt \n";
             print "  yearlyStats: $flags[$yearlyStats] \n";
             print "  allYearsStats: $flags[$allYearsStats] \n";
         }  
         if ( $caller eq "load" ) {
             foreach $site ( keys %backupInts ) {
                 print "  backupInts: $site $backupInts{$site} \n";
             }
         }
    }        
} 
    
sub timestamp() {
    my @localt = localtime(time());
    my $sec    = $localt[0];
    my $min    = $localt[1];
    my $hour   = $localt[2];
    my $day    = $localt[3];
    my $month  = $localt[4] + 1;
    my $year   = $localt[5] + 1900;

    return sprintf("%04d-%02d-%02d %02d:%02d:%02d",
                   $year, $month, $day, $hour, $min, $sec);
}

