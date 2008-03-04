#!/usr/local/bin/perl -w

use DBI;

###############################################################################
#                                                                             #
#                             prepareMySQLStats.pl                            #
#                                                                             #
#  (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  #
#                             All Rights Reserved                             #
#        Produced by Jacek Becla for Stanford University under contract       #
#               DE-AC02-76SF00515 with the Department of Energy               #
###############################################################################

# $Id$


use vars qw( @gmt $sec $min $hour $wday $sec2Sleep $loadTime);
## take care of arguments
if ( @ARGV!=1 ) {
    print "Expected argument <configFile>\n";
    exit;
}
$confFile = $ARGV[0];
unless ( open INFILE, "< $confFile" ) {
    print "Can't open file $confFile\n";
    exit;
}

my $stopFName = "$confFile.stop";
$inhibitFSize =  "$confFile.inhibitFSize";
$workDir = "/u1/xrdmon/xrdmon_kan_perl/workdir";

$mysqlSocket = '/tmp/mysql.sock';
while ( $_ = <INFILE> ) {
    chomp();
    my ($token, $v1) = split(/ /, $_);
    if ( $token =~ "dbName:" ) {
        $dbName = $v1;
    } elsif ( $token =~ "MySQLUser:" ) {
        $mySQLUser = $v1;
    } elsif ( $token =~ "MySQLSocket:" ) {
        $mysqlSocket = $v1;
    } else {
        print "Invalid entry: \"$_\"\n";
        close INFILE;
        exit;
    }
}
close INFILE;


# connect to the database
unless ( $dbh = DBI->connect("dbi:mysql:$dbName;mysql_socket=$mysqlSocket",$mySQLUser) ) {
    print "Error while connecting to database. $DBI::errstr\n";
    exit;
}

&doInitialization();

#start an infinite loop
while ( 1 ) {
    # wake up every minute at HH:MM:30
    @gmt   = gmtime(time());
    $sec   = $gmt[0];
    $sec2Sleep = 90 - $sec;
    if ( $sec <= 30 ) {
        $sec2Sleep -= 60;
    }
    if ( $sec2Sleep < 60 ) {
         print "sleeping $sec2Sleep sec... \n";
         sleep $sec2Sleep;
    }
 
    $loadTime = &timestamp();
    print "$loadTime\n";

    $loadTime = &gmtimestamp();
    @gmt   = gmtime(time());
    $sec   = $gmt[0];
    $min   = $gmt[1];
    $hour  = $gmt[2];
#   $day   = $gmt[3];
    $wday  = $gmt[6];
#   $yday  = $gmt[7];

    foreach $siteName (@siteNames) {
        $dbUpdates{$siteName} = &runQueryWithRet("SELECT dbUpdate 
                                                    FROM sites 
                                                   WHERE name = '$siteName' ");
	&prepareStats4OneSite($siteName, 
			      $loadTime,
			      $min,
			      $hour,
 			      $wday);
    }

    if ( -e $stopFName ) {
	unlink $stopFName;
	exit;
    }
    # make sure the loop takes at least 2 s
    if ( $loadTime eq &gmtimestamp() ) {
        sleep 2;
    }
}


###############################################################################
###############################################################################
###############################################################################

sub doInitialization() {
    my $theTime = &timestamp();
    print "Initialization started  $theTime \n";
    use vars qw($lastTime  $siteId $period);
    @primes = (101, 127, 157, 181, 199, 223, 239, 251, 271, 307);
    @periods = ( 'Hour', 'Day', 'Week', 'Month', 'Year');
    %sourcePeriods = ('Week' => 'Day' , 'Month' => 'Week' , 'Year' => 'Month' , 'AllYears' => 'Year' );
    %intervals  = ('Hour' => 1        , 'Day' => 15       , 'Week' => 1      , 'Month' => 6      , 'Year' => 1     );
    %timeUnits  = ('Hour' => 'MINUTE' , 'Day' => 'MINUTE' , 'Week' => 'HOUR' , 'Month' => 'HOUR' , 'Year' => 'DAY' );
    %seconds    = ('MINUTE' => 60 , 'HOUR' => 3600 , 'DAY' => 86400 );
    %firstDates = ( 'SLAC' => '2005-06-13 00:00:00',
                     'RAL' => '2005-08-24 20:00:00' );

    # create temporary tables for topPerformers
    &runQuery("CREATE TEMPORARY TABLE jj  (theId INT, n INT, INDEX (theId))");
    &runQuery("CREATE TEMPORARY TABLE ff  (theId INT, n INT, s INT, INDEX (theId))");
    &runQuery("CREATE TEMPORARY TABLE uu  (theId INT, n INT, INDEX (theId))");
    &runQuery("CREATE TEMPORARY TABLE vv  (theId INT, n INT, INDEX (theId))");
    &runQuery("CREATE TEMPORARY TABLE xx  (theId INT UNIQUE KEY, INDEX (theId))");
    @topPerfTables = ("jj", "ff", "uu", "vv", "xx");

    # create temporary tables for stats
    &runQuery("CREATE TEMPORARY TABLE times (begT DATETIME NOT NULL, endT DATETIME NOT NULL,
                                            INDEX (begT), INDEX(endT) )");
    &runQuery("CREATE TEMPORARY TABLE id_times ( theId MEDIUMINT UNSIGNED NOT NULL,
                                            begT DATETIME NOT NULL, endT DATETIME NOT NULL,
                                            INDEX(theId), INDEX (begT), INDEX(endT) )");


    # cleanup old entries
    $GMTnow = &gmtimestamp();
    &runQuery("DELETE FROM statsLastHour  WHERE date <= DATE_SUB('$GMTnow', INTERVAL 1 HOUR)");
    &runQuery("DELETE FROM statsLastDay   WHERE date <= DATE_SUB('$GMTnow', INTERVAL 1 DAY)");
    &runQuery("DELETE FROM statsLastWeek  WHERE date <= DATE_SUB('$GMTnow', INTERVAL 1 WEEK)");
    &runQuery("DELETE FROM statsLastMonth WHERE date <= DATE_SUB('$GMTnow', INTERVAL 1 MONTH)");
    &runQuery("DELETE FROM statsLastYear  WHERE date <= DATE_SUB('$GMTnow', INTERVAL 1 YEAR)");


    # find all sites
    @siteNames = &runQueryRetArray("SELECT name FROM sites");
    # fill in missing points in statsLast tables
    foreach $siteName (@siteNames) {
	$siteId = &runQueryWithRet("SELECT id 
                                      FROM sites 
                                     WHERE name = '$siteName'");
        $siteIds{$siteName} = $siteId;
        $dbUpdates{$siteName} = &runQueryWithRet("SELECT dbUpdate 
                                                    FROM sites 
                                                   WHERE name = '$siteName' ");

        # fill out missing bins in stats tables
        foreach $period ( @periods ) {
            next if ( $period eq "Hour" );
            &fillStatsMissingBins($siteName, $period, $GMTnow);
        }
        &fillStatsAllYearsMissingBins($siteName, $GMTnow);
      
        # initialize lastNo values
        $lastTime = &runQueryWithRet("SELECT MAX(date) FROM statsLastHour
                                       WHERE siteId = $siteId ");
        if ( $lastTime ) {
	     ($lastNoJobs[$siteId],    $lastNoUsers[$siteId], 
              $lastNoUniqueF[$siteId], $lastNoNonUniqueF[$siteId]) 
	      = &runQueryWithRet("SELECT noJobs,noUsers,noUniqueF,noNonUniqueF
                              FROM   statsLastHour 
                              WHERE  date   = '$lastTime'  AND
                                     siteId = $siteId           ");
        } else { 
              $lastNoJobs[$siteId]=$lastNoUsers[$siteId]=0;
              $lastNoUniqueF[$siteId]=$lastNoNonUniqueF[$siteId]=0;
        }
    }

    # some other stuff
    $bbkListSize = 250;
    $minSizeLoadTime = 5;
    $theTime = &timestamp();
    print "Initialization ended  $theTime \n";
}
sub prepareStats4OneSite() {
    my ($siteName, $loadTime, $min, $hour, $wday) = @_;

    print "--> $siteName <--\n";

    # every min at HH:MM:00
    &runQuery("DELETE FROM ${siteName}_closedSessions_LastHour  
                     WHERE disconnectT < DATE_SUB('$loadTime', INTERVAL  1 HOUR)");
    &runQuery("DELETE FROM ${siteName}_closedFiles_LastHour     
                     WHERE      closeT < DATE_SUB('$loadTime', INTERVAL  1 HOUR)");
    &loadStatsLastHour($siteName, $loadTime, $min % 60);
    &loadTopPerfPast("Hour", 20, $siteName, $loadTime);

    if ( $min == 5 || $min == 20 || $min == 35 || $min == 50 ) {
        # with 5 minutes delay:
	# every 15 min at HH:00:00, HH:15:00, HH:00:00, HH:45:00
        &closeOpenedFiles($siteName, $loadTime, 1);
        &runQuery("DELETE FROM ${siteName}_closedSessions_LastDay  
                         WHERE disconnectT < DATE_SUB('$loadTime', INTERVAL  1 DAY)");
        &runQuery("DELETE FROM ${siteName}_closedFiles_LastDay     
                         WHERE      closeT < DATE_SUB('$loadTime', INTERVAL  1 DAY)");
	&loadStatsLastPeriod($siteName, $loadTime, "Day", "Hour");
        #                                          period cfPeriod
	&loadTopPerfPast("Day", 20, $siteName, $loadTime);
    }

    if ( $min == 10) {
        # with 10 minutes delay:
	# every hour at HH:00:00
        &runQuery("DELETE FROM ${siteName}_closedSessions_LastWeek  
                         WHERE disconnectT < DATE_SUB('$loadTime', INTERVAL  7 DAY)");
        &runQuery("DELETE FROM ${siteName}_closedFiles_LastWeek     
                         WHERE      closeT < DATE_SUB('$loadTime', INTERVAL  7 DAY)");
        &closeIdleSessions($siteName, $loadTime, 1);
	&loadStatsLastPeriod($siteName, $loadTime, "Week", "Day");
        #                                          period cfPeriod
	&loadTopPerfPast("Week", 20, $siteName, $loadTime);
    }

    if ( $min == 15 ) {
        # with 15 minutes delay:
	if ( $hour == 0 || $hour == 6 || $hour == 12 || $hour == 18 ) {
	    # every 6 hours at 00:00:00, 06:00:00, 12:20:00, 18:00:00 GMT
            &runQuery("DELETE FROM ${siteName}_closedSessions_LastMonth  
                             WHERE disconnectT < DATE_SUB('$loadTime', INTERVAL 1 MONTH)");
            &runQuery("DELETE FROM ${siteName}_closedFiles_LastMonth     
                             WHERE      closeT < DATE_SUB('$loadTime', INTERVAL 1 MONTH)");
	    &loadStatsLastPeriod($siteName, $loadTime, "Month", "Day");
        #                                               period cfPeriod
	    &loadTopPerfPast("Month", 20, $siteName, $loadTime);
            &fillStatsMissingBins($siteName,"Day",$loadTime);

	}
    }
    if ( $min == 20 ) {
	if ( $hour == 0 ) {
            # with 20 minutes delay:
	    # every 24 hours at 00:00:00 GMT
            &runQuery("DELETE FROM ${siteName}_closedSessions_LastYear  
                             WHERE disconnectT < DATE_SUB('$loadTime', INTERVAL 1 YEAR)");
            &runQuery("DELETE FROM ${siteName}_closedFiles_LastYear     
                             WHERE      closeT < DATE_SUB('$loadTime', INTERVAL 1 YEAR)");
	    &loadStatsLastPeriod($siteName, $loadTime, "Year", "Week");
        #                                               period cfPeriod
	    &loadTopPerfPast("Year", 20, $siteName, $loadTime);
            &fillStatsMissingBins($siteName,"Week",$loadTime);
            &fillStatsMissingBins($siteName,"Month",$loadTime);
	}
        if ( $hour == 5 ) {
            # every 24 hours at 05:20:00 GMT
	    &closeLongSessions($siteName, $loadTime, 1);
	}
        if ( $hour == 0 && $wday == 0 ) {
            # with 20 minutes delay:
            # every Sunday at 00:00:00 GMT
            &loadStatsAllYears($siteName, $loadTime);
            &fillStatsMissingBins($siteName,"Year",$loadTime);
	}
    }

    # top Perf - "now"
    &loadTopPerfNow(20, $siteName);

    # truncate temporary tables
    foreach $table (@topPerfTables) { &runQuery("DELETE FROM $table"); }

    # load file sizes
    if ( ! -e $inhibitFSize ) {
        if ( $min == 37 ) {
             if ( $hour == 6 ) {
                  # every 24 hours at 06:37:00 GMT
	         &loadFileSizes( $siteName, -3 );
	     }
	     if ( $hour == 18 || $hour == 6 ) {
                 # every 12 hours at 18:37:00 and 06:37:00 GMT
	         &loadFileSizes( $siteName, -2 );
	     }
             # every hour at HH:37:00
             &loadFileSizes( $siteName, -1 );
         }
         # every min at HH:MM:00
         &loadFileSizes( $siteName, 0 );
    }
}
sub runQueryWithRet() {
    my $sql = shift @_;
#    print "$sql;\n";
    my $sth = $dbh->prepare($sql) 
        or die "Can't prepare statement $DBI::errstr\n";
    $sth->execute or die "Failed to exec \"$sql\", $DBI::errstr";
    return $sth->fetchrow_array;
}
sub runQueryRetArray() {
    use vars qw(@theArray);
    my $sql = shift @_;
    @theArray = ();   
#    print "$sql;\n";
    my $sth = $dbh->prepare($sql) 
        or die "Can't prepare statement $DBI::errstr\n";
    $sth->execute or die "Failed to exec \"$sql\", $DBI::errstr";

    while ( @x = $sth->fetchrow_array ) {
	push @theArray, @x;
    };
    return @theArray;
}

sub runQuery() {
    my ($sql) = @_;
#    print "$sql;\n";
    my $sth = $dbh->prepare($sql) 
        or die "Can't prepare statement $DBI::errstr\n";
    $sth->execute or die "Failed to exec \"$sql\", $DBI::errstr";
}
sub runQueryRetNum() {
    my $sql = shift @_;
#    print "$sql;\n";
    my $num = $dbh-> do ($sql) or die "Failed to exec \"$sql\", $DBI::errstr";
    return $num;
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

sub gmtimestamp() {
    my @gmt   = gmtime(time());
    my $sec   = $gmt[0];
    my $min   = $gmt[1];
    my $hour  = $gmt[2];
    my $day   = $gmt[3];
    my $month = $gmt[4] + 1;
    my $year  = $gmt[5] + 1900;

    return sprintf("%04d-%02d-%02d %02d:%02d:%02d", 
                   $year, $month, $day, $hour, $min, $sec);
}
sub loadFileSizes() {
    my ($siteName, $sizeIndex) = @_;
    print "Loading file sizes... \n";
    use vars qw($sizeIndex $fromId $toId $path $size @files @inBbk);
    &runQuery("CREATE TEMPORARY TABLE zerosize  
               (name VARCHAR(255), id MEDIUMINT UNSIGNED, hash MEDIUMINT)");
    &runQuery("INSERT INTO zerosize(name, id, hash)
                      SELECT name, id, hash FROM paths 
                      WHERE  size BETWEEN $sizeIndex AND 0 
                      ORDER BY id 
                      LIMIT 7500");
     
    my $bbkInput = "$workDir/bbkInput";
    $skip = 0;
    while () {
	my $t0 = time();
	my $sec = (localtime)[0];
	my $timeLeft = 90-$sec;
	if ( $sec < 30 ) {
	    $timeLeft -= 60;
	}
       last if ( $timeLeft < $minSizeLoadTime);
       @files = &runQueryRetArray("SELECT name FROM zerosize LIMIT $skip, $bbkListSize "); 
       #print scalar @files, "\n";
       last if ( ! @files );

       open ( BBKINPUT, ">$bbkInput" ) or die "Can't open bbkInput file: $!"; 
       my $index = 0;
       while ( defined $files[$index] ) {
           print BBKINPUT "$files[$index]\n";
           $index++;
       }
       @bbkOut   = `BbkUser --lfn-file=$bbkInput --quiet                 lfn bytes`;
       @bbkOut18 = `BbkUser --lfn-file=$bbkInput --quiet --dbname=bbkr18 lfn bytes`;
       @bbkOut   = (@bbkOut, @bbkOut18);
       @inBbk = ();
       while ( @bbkOut ) {
           $line = shift @bbkOut;
           chomp $line;
           ($path, $size) = split (' ', $line);
           @inBbk = (@inBbk, $path);
           my $hashValue = &returnHash("$path");
           my $id = &runQueryWithRet("SELECT id FROM zerosize 
                                      WHERE hash = $hashValue AND name = '$path'");
           if ( $id ) {
	       &runQuery("UPDATE paths SET size = $size WHERE id = $id ");
	   }
       }
       # decrement size by 1 for files that failed bbk.
       foreach $path ( @files ) {
           if ( ! grep { $_ eq $path } @inBbk ) {
               my $hashValue = &returnHash("$path");
               my $id = &runQueryWithRet("SELECT id FROM zerosize 
                                          WHERE hash = $hashValue AND name = '$path'");
               if ( $id ) {
		   &runQuery("UPDATE paths SET size = size - 1 WHERE id = $id ");
	       }
           }
       }
       print "Done ", scalar @files, " files updated. Update time = ", time() - $t0, " s \n";
       last if ( @files < $bbkListSize );
       $skip += $bbkListSize;
    }
    &runQuery("DROP TABLE IF EXISTS zerosize");
}
sub loadStatsLastHour() {
    my ($siteName, $loadTime, $seqNo) = @_;

    use vars qw($noJobs $noUsers $noUniqueF $noNonUniqueF $deltaJobs $jobs_p 
                $deltaUsers $users_p $deltaUniqueF $uniqueF_p $deltaNonUniqueF $nonUniqueF_p);

    if ( &getLastInsertTime($loadTime, "Hour") gt $dbUpdates{$siteName} ) {return;}
    my $siteId = $siteIds{$siteName};
    
    &runQuery("DELETE FROM statsLastHour WHERE date < DATE_SUB('$loadTime', INTERVAL  1 HOUR) AND
                                               siteId = $siteId     ");
    ($noJobs, $noUsers) = &runQueryWithRet("SELECT COUNT(DISTINCT jobId), COUNT(DISTINCT userId) 
                                              FROM ${siteName}_openedSessions");

    ($noUniqueF, $noNonUniqueF) = &runQueryWithRet("SELECT COUNT(DISTINCT pathId), COUNT(*) 
                                                      FROM ${siteName}_openedFiles");
    &runQuery("REPLACE INTO statsLastHour 
                      (seqNo, siteId, date, noJobs, noUsers, noUniqueF, noNonUniqueF) 
               VALUES ($seqNo, $siteId, '$loadTime', $noJobs, $noUsers, $noUniqueF, $noNonUniqueF)");

    $deltaJobs = $noJobs - $lastNoJobs[$siteId]; 
    $jobs_p = $lastNoJobs[$siteId] > 0 ? &roundoff( 100 * $deltaJobs / $lastNoJobs[$siteId] ) : -1;
    $deltaUsers = $noUsers - $lastNoUsers[$siteId];
    $users_p = $lastNoUsers[$siteId] > 0 ? &roundoff( 100 * $deltaUsers / $lastNoUsers[$siteId] ) : -1;
    $deltaUniqueF = $noUniqueF - $lastNoUniqueF[$siteId];
    $uniqueF_p = $lastNoUniqueF[$siteId] > 0 ? &roundoff( 100 * $deltaUniqueF / $lastNoUniqueF[$siteId] ) : -1;
    $deltaNonUniqueF = $noNonUniqueF - $lastNoNonUniqueF[$siteId];
    $nonUniqueF_p = $lastNoNonUniqueF[$siteId] > 0 ? &roundoff( 100 * $deltaNonUniqueF / $lastNoNonUniqueF[$siteId] ) : -1;
    &runQuery("REPLACE INTO rtChanges 
                           (siteId, jobs, jobs_p, users, users_p, uniqueF, uniqueF_p, 
                            nonUniqueF, nonUniqueF_p, lastUpdate)
                    VALUES ($siteId, $deltaJobs, $jobs_p, $deltaUsers, $users_p, $deltaUniqueF,
                            $uniqueF_p, $deltaNonUniqueF, $nonUniqueF_p, '$loadTime')");
    $lastNoJobs[$siteId]       = $noJobs;
    $lastNoUsers[$siteId]      = $noUsers;
    $lastNoUniqueF[$siteId]    = $noUniqueF;
    $lastNoNonUniqueF[$siteId] = $noNonUniqueF;
}

sub loadStatsAllYears() {
    my ($siteName, $loadTime) = @_;
    if ( &getLastInsertTime($loadTime, "AllYears") gt $dbUpdates{$siteName} ) {return;}

    my $siteId = $siteIds{$siteName};
    my ($noJobs, $noUsers, $noUniqueF, $noNonUniqueF) 
      = &runQueryWithRet("SELECT AVG(noJobs), AVG(noUsers), AVG(noUniqueF), AVG(noNonUniqueF) 
                          FROM   statsLastWeek
                          WHERE  siteId = $siteId"); 
    if ( $noJobs ) {
        &runQuery("INSERT IGNORE INTO statsAllYears
                              (siteId, date, noJobs, noUsers, noUniqueF, noNonUniqueF) 
                        VALUES ($siteId, '$loadTime', $noJobs, $noUsers,
                                $noUniqueF, $noNonUniqueF)");
    }
}
sub loadStatsLastPeriod() {
    use vars qw($noJobs   $noUsers   $noUniqueF   $noNonUniqueF $nSeqs);
    my ($siteName, $loadTime, $period, $cfPeriod) = @_;
    my $siteId = $siteIds{$siteName};
    my $interval = $intervals{$period};
    my $timeUnit = $timeUnits{$period};
    my $intervalSec = $interval * $seconds{$timeUnit};

    my $t2 = &getLastInsertTime($loadTime, $period);
    if ( $t2 gt $dbUpdates{$siteName} ) {return;}
    my $t1 = &runQueryWithRet("SELECT DATE_SUB('$t2', INTERVAL $interval $timeUnit)");
    if ( $period eq "Day" ) { 
        # use long method for Day       
        $avNumJobs = &avNumJobsInTimeInterval($t1, $t2, $intervalSec, $siteName);
        $avNumUsers = &avNumUsersInTimeInterval($t1, $t2, $intervalSec, $siteName);
        $avNumUniqueFiles = &avNumUniqueFilesInTimeInterval($t1, $t2, $intervalSec, $siteName, $cfPeriod);
        $avNumNonUniqueFiles = &avNumNonUniqueFilesInTimeInterval($t1, $t2, $intervalSec, $siteName, $cfPeriod);
    } else {
        # use average from statsLastDay but first make sure all needed points exist.
        $nSeqs = &runQueryWithRet("SELECT FLOOR(TIMESTAMPDIFF(MINUTE,'$t1','$t2')/15)");
        if ( $nSeqs > &runQueryWithRet("SELECT COUNT(*)
                                                FROM statsLastDay
                                               WHERE siteId = $siteId  AND
                                                       date >  '$t1'   AND
                                                       date <= '$t2'       ") ) {
             &fillStatsMissingBins($siteName,"Day",$loadTime);
        }
        ($avNumJobs, $avNumUsers, $avNumUniqueFiles, $avNumNonUniqueFiles) =
              &runQueryWithRet("SELECT AVG(noJobs), AVG(noUsers), AVG(noUniqueF), AVG(noNonUniqueF)
                                  FROM statsLastDay
                                 WHERE siteId = $siteId  AND
                                       date >  '$t1'      AND
                                       date <= '$t2'         ");
    }
    if ( $avNumJobs ) {
        $seqNo = &timeToSeqNo( $t2, $period );
        &runQuery("REPLACE INTO statsLast$period
                    VALUES ($seqNo, $siteId, '$t2', $avNumJobs, $avNumUsers, $avNumUniqueFiles, $avNumNonUniqueFiles)");
    }
}
sub roundoff() {
   my $a = shift;
   $d = 0;
   if ( $a < 10 ) {$d = $a < 1 ? 2 : 1;}
   return sprintf("%.${d}f", $a);
}

# closes opened files corresponding to closed sessions
sub closeOpenedFiles() {
    my ($siteName, $GMTnow, $loadLastTables) = @_;

    &printNow("Closing open files... ");
    &runQuery("CREATE TEMPORARY TABLE xcf like SLAC_closedFiles");

    # give it an extra hour: sometimes closeFile
    # info may arrive after the closeSession info
    # timeout on xrootd server is ~ 1min, so 10 min should suffice
    &runQuery("INSERT IGNORE INTO xcf
                 SELECT of.id,
                        of.sessionId,
                        of.pathId,
                        of.openT,
                        cs.disconnectT,
                        -1,
                        -1
                 FROM   ${siteName}_openedFiles of,
                        ${siteName}_closedSessions cs
                 WHERE  sessionId = cs.id    AND
                        disconnectT < DATE_SUB('$GMTnow', INTERVAL 10 MINUTE)");

    &runQuery("INSERT IGNORE INTO ${siteName}_closedFiles
                 SELECT * FROM xcf");

    if ( $loadLastTables ) {
        foreach $period ( @periods ) {
            next if ( $period eq "Hour" );
            &runQuery("INSERT IGNORE INTO ${siteName}_closedFiles_Last$period
                       SELECT * 
                         FROM xcf 
                        WHERE closeT > DATE_SUB('$GMTnow', INTERVAL 1 $period)");

        }
    }
 
    my $noDone =&runQueryRetNum("DELETE FROM  ${siteName}_openedFiles
                                  USING ${siteName}_openedFiles, xcf
                                  WHERE ${siteName}_openedFiles.id = xcf.id");
    &runQuery("DROP TABLE xcf");
    print " $noDone closed\n";
}

# closes opened sessions with no open files.
# Assignments:
# duration = MAX(closeT, connectT) - MIN(openT, connectT)
# disconnectT = MAX(closeT, connectT)
# status = I

sub closeIdleSessions() {
    my ($siteName, $GMTnow, $loadLastTables ) = @_;

    # be careful when changing this number, depending on the value:
    # 1) different input closedSessions table might be needed in
    #    "insert into x" query
    # 2) you might need to add the sesssions to more table, e.g.
    #    into closedSessions_LastMonth
    my $cutOff = 12; # [hour]

    &printNow("Closing idle sessions... ");

    my $cutOffDate = &runQueryWithRet("SELECT DATE_SUB('$GMTnow', INTERVAL $cutOff HOUR)");

    # make temporary table of open sessions with no open files.
    &runQuery("CREATE TEMPORARY TABLE os_no_of LIKE ${siteName}_openedSessions");
    my $noDone = &runQueryRetNum("INSERT INTO os_no_of
                                        SELECT os.*
                                          FROM        ${siteName}_openedSessions os
                                               LEFT JOIN
                                                      ${siteName}_openedFiles of
                                            ON     os.id = of.sessionId
                                         WHERE of.id IS NULL");
    if ( $noDone == 0 ) {
        &runQuery("DROP TABLE IF EXISTS os_no_of");
        return
    }

    &runQuery("CREATE TEMPORARY TABLE cs_no_of LIKE ${siteName}_closedSessions");
    # close sessions with closed files
    my $n_cs_cf = 
        &runQueryRetNum("INSERT 
                           INTO cs_no_of
                         SELECT os.id, jobId, userId, pId, clientHId, serverHId,
                                TIMESTAMPDIFF(SECOND, MIN(cf.openT), MAX(cf.closeT)),
                                MAX(cf.closeT) AS maxT,
                                'I'
                           FROM os_no_of os, ${siteName}_closedFiles cf
                          WHERE os.id = cf.sessionId
                       GROUP BY os.id
                         HAVING maxT < '$cutOffDate' ");

    # close sessions with no files
    my $n_cs_no_f =
        &runQueryRetNum("INSERT 
                           INTO cs_no_of
                         SELECT os.id, jobId, userId, pId, clientHId, serverHId, 0, connectT, 'I'
                           FROM     os_no_of os
                                LEFT JOIN
                                    ${siteName}_closedFiles cf
                                ON  os.id = cf.sessionId
                          WHERE cf.id IS NULL   AND
                                os.connectT < '$cutOffDate'     ");

    #insert into closedSessions tables and delete from openSession table
                     
    &runQuery("CREATE TEMPORARY TABLE IF NOT EXISTS ns ( jobId INT NOT NULL,
                                                           nos SMALLINT NOT NULL)");

    &runQuery("INSERT IGNORE INTO ${siteName}_closedSessions
                           SELECT *
                             FROM cs_no_of ");
    &runQuery("INSERT INTO ns 
                    SELECT jobId, count(jobId)
                      FROM cs_no_of
                  GROUP BY jobId ");
    &runQuery("UPDATE ${siteName}_jobs j, ns
                  SET noOpenSessions = noOpenSessions - nos
                WHERE j.jobId = ns.jobId  ");
    &runQuery("DELETE FROM ${siteName}_openedSessions os
                     USING ${siteName}_openedSessions os, cs_no_of cs
                     WHERE os.id = cs.id ");
    if ( $loadLastTables ) {
        foreach $period ( @periods ) {
            next if ( $period eq "Hour" );
            &runQuery("INSERT IGNORE INTO ${siteName}_closedSessions_Last$period
                            SELECT *
                              FROM cs_no_of
                             WHERE disconnectT > DATE_SUB('$GMTnow', INTERVAL 1 $period) ");
        }
    }

    &runQuery("DROP TABLE IF EXISTS ns");
    &runQuery("DROP TABLE IF EXISTS os_no_of");
    &runQuery("DROP TABLE IF EXISTS cs_no_of");
    print " closed $n_cs_cf sessions with closed files, \n"; 
    print " closed $n_cs_no_f sessions with no files\n";
}
# closes opened sessions with associated open files which were
# opened for longer than x days.
# Assignments:
# duration = MAX(openT) - MIN(openT)   
# disconnectT = MAX(openT, closeT)                     
# status = L

sub closeLongSessions() {
    my ($siteName, $GMTnow, $loadLastTables) = @_;

    # be careful when changing this number, depending on the value:
    # 1) different input closedSessions table might be needed in
    #    "insert into x" query
    # 2) you might need to add the sesssions to more table, e.g.
    #    into closedSessions_LastMonth
    $cutOff = 70; # [days]

    &printNow("Closing long sessions... ");

    my $cutOffDate = &runQueryWithRet("SELECT DATE_SUB('$GMTnow', INTERVAL $cutOff DAY)");
    &runQuery("CREATE TEMPORARY TABLE IF NOT EXISTS cs LIKE ${siteName}_closedSessions");
    my $noDone =&runQueryRetNum("
                 INSERT IGNORE INTO cs
                 SELECT os.id, jobId, userId, pId, clientHId, serverHId,
                        TIMESTAMPDIFF(SECOND,
                                      LEAST(IFNULL(MIN(cf.openT),'3'),
                                            MIN(of.openT)           ),
                                      GREATEST(IFNULL(MAX(cf.closeT),'2'),
                                               MAX(of.openT)            )  ),
                        GREATEST(IFNULL(MAX(cf.closeT),'2'),
                                 MAX(of.openT) ),
                        'L'
                   FROM         (${siteName}_openedSessions os,
                                 ${siteName}_openedFiles of)
                        LEFT JOIN
                                 ${siteName}_closedFiles cf
                        ON       os.id = cf.sessionId
                  WHERE 	os.id       = of.sessionId        AND
                        os.connectT < '$cutOffDate'
               GROUP BY os.id");

    if ( $noDone > 0 ) {
        &runQuery("INSERT IGNORE INTO ${siteName}_closedSessions          
                               SELECT * 
                                 FROM cs");
        if ( $loadLastTables ) {
            &runQuery("INSERT IGNORE INTO ${siteName}_closedSessions_LastYear 
                                   SELECT * 
                                     FROM cs");
        }        
        &runQuery("CREATE TEMPORARY TABLE IF NOT EXISTS ns ( jobId INT NOT NULL,
                                                               nos SMALLINT NOT NULL)");
        &runQuery("INSERT INTO ns 
                        SELECT jobId, count(jobId)
                          FROM cs
                      GROUP BY jobId ");
        &runQuery("UPDATE ${siteName}_jobs j, ns
                      SET noOpenSessions = noOpenSessions - nos
                    WHERE j.jobId = ns.jobId  ");

        &runQuery("DELETE FROM ${siteName}_openedSessions 
                         USING ${siteName}_openedSessions, cs
                         WHERE ${siteName}_openedSessions.id = cs.id      ");
        &runQuery("DROP TABLE IF EXISTS ns");
    }
    &runQuery("DROP TABLE IF EXISTS cs");
    print " $noDone closed\n";
}


sub loadTopPerfPast() {
    my ($theKeyword, $theLimit, $siteName, $loadTime) = @_;

    &runTopUsersQueriesPast($theKeyword, $theLimit, $siteName, $loadTime);
    foreach $table (@topPerfTables) { &runQuery("DELETE FROM $table"); }
    &runTopSkimsQueriesPast($theKeyword, $theLimit, "SKIMS", $siteName);
    foreach $table (@topPerfTables) { &runQuery("DELETE FROM $table"); }
    &runTopSkimsQueriesPast($theKeyword, $theLimit, "TYPES", $siteName);
    foreach $table (@topPerfTables) { &runQuery("DELETE FROM $table"); }
    &runTopFilesQueriesPast($theKeyword, $theLimit, $siteName);
    foreach $table (@topPerfTables) { &runQuery("DELETE FROM $table"); }
}
sub loadTopPerfNow() {
    my ($theLimit, $siteName) = @_;

    &runTopUsersQueriesNow($theLimit, $siteName);
    foreach $table (@topPerfTables) { &runQuery("DELETE FROM $table"); }
    &runTopSkimsQueriesNow($theLimit, "SKIMS", $siteName);
    foreach $table (@topPerfTables) { &runQuery("DELETE FROM $table"); }
    &runTopSkimsQueriesNow($theLimit, "TYPES", $siteName);
    foreach $table (@topPerfTables) { &runQuery("DELETE FROM $table"); }
    &runTopFilesQueriesNow($theLimit, $siteName);
    foreach $table (@topPerfTables) { &runQuery("DELETE FROM $table"); }
}
sub runTopUsersQueriesPast() {
    my ($theKeyword, $theLimit, $siteName, $loadTime) = @_;

    my $oneChar = substr($theKeyword, 0, 1);
    &printNow("$theKeyword: U ");

    $destinationTable = "${siteName}_topPerfUsersPast";

    # past jobs
    &runQuery("INSERT INTO jj
         SELECT userId, 
                COUNT(jobId) AS n
           FROM ${siteName}_jobs
          WHERE noOpenSessions = 0    AND
                endT > DATE_SUB('$loadTime', INTERVAL 1 $theKeyword)
       GROUP BY userId");

    # past files - through opened & closed sessions
    &runQuery("INSERT INTO ff           
        SELECT tmp.userId, 
               COUNT(tmp.pathId),
               SUM(tmp.size)/(1024*1024)
          FROM ( SELECT DISTINCT oc.userId, oc.pathId, oc.size
                   FROM ( SELECT userId, pathId, size 
                           FROM  ${siteName}_openedSessions os,
                                 ${siteName}_closedFiles_Last$theKeyword cf,
                                 paths p
                          WHERE  os.id = cf.sessionId     AND
                                 cf.pathId = p.id 
                      UNION ALL
                         SELECT  userId, pathId, size 
                           FROM  ${siteName}_closedSessions_Last$theKeyword cs,
                                 ${siteName}_closedFiles_Last$theKeyword cf,
                                 paths p
                          WHERE  cs.id = cf.sessionId    AND 
                                 cf.pathId = p.id 
                         )   AS  oc
                )   AS tmp
       GROUP BY tmp.userId");

    # past volume - through opened & closed sessions
    &runQuery("INSERT INTO vv
        SELECT oc.userId, 
               SUM(oc.bytesR)/(1024*1024)
          FROM ( SELECT  userId, bytesR
                   FROM  ${siteName}_openedSessions os,
                         ${siteName}_closedFiles_Last$theKeyword cf
                  WHERE  os.id = cf.sessionId
              UNION ALL
                 SELECT  userId, bytesR
                   FROM  ${siteName}_closedSessions_Last$theKeyword cs,
                         ${siteName}_closedFiles_Last$theKeyword cf
                  WHERE  cs.id = cf.sessionId
               )     AS  oc
      GROUP BY oc.userId");

    ##### now find all names for top X for each sorting 
    &runQuery("REPLACE INTO xx SELECT theId FROM jj ORDER BY n DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM ff ORDER BY n DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM ff ORDER BY s DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM vv ORDER BY n DESC LIMIT $theLimit");

    ## delete old data
    &runQuery("DELETE FROM $destinationTable WHERE timePeriod LIKE '$theKeyword'");

    ## and finally insert the new data
    &runQuery("INSERT INTO $destinationTable
               SELECT xx.theId, 
                      IFNULL(jj.n, 0) AS jobs, 
                      IFNULL(ff.n, 0) AS files, 
                      IFNULL(ff.s, 0) AS fSize, 
                      IFNULL(vv.n, 0) AS vol, 
                      '$theKeyword'
                 FROM xx 
                      LEFT OUTER JOIN jj ON xx.theId = jj.theId
                      LEFT OUTER JOIN ff ON xx.theId = ff.theId
                      LEFT OUTER JOIN vv ON xx.theId = vv.theId");
}

sub runTopFilesQueriesPast() {
    my ($theKeyword, $theLimit, $siteName) = @_;

    my $oneChar = substr($theKeyword, 0, 1);
    &printNow("F \n");

    $destinationTable = "${siteName}_topPerfFilesPast";

    # past jobs
    &runQuery("INSERT INTO jj
            SELECT DISTINCT oc.pathId, COUNT(DISTINCT oc.jobId) AS n
              FROM (SELECT pathId, jobId
                      FROM ${siteName}_closedSessions_Last$theKeyword cs,
                           ${siteName}_closedFiles_Last$theKeyword cf
                     WHERE cs.id = cf.sessionId     
                 UNION ALL
                    SELECT pathId, jobId
                      FROM ${siteName}_openedSessions os,
                           ${siteName}_closedFiles_Last$theKeyword cf
                     WHERE os.id = cf.sessionId
                   )    AS oc
                   
          GROUP BY oc.pathId");

    # past volume 
    &runQuery("INSERT INTO vv
        SELECT pathId, 
               SUM(bytesR)/(1024*1024)
          FROM ${siteName}_closedFiles_Last$theKeyword
      GROUP BY pathId");

    ##### now find all names for top X for each sorting 
    &runQuery("REPLACE INTO xx SELECT theId FROM jj ORDER BY n DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM vv ORDER BY n DESC LIMIT $theLimit");

    ## delete old data
    &runQuery("DELETE FROM $destinationTable WHERE timePeriod LIKE '$theKeyword'");

    ## and finally insert the new data
        &runQuery("INSERT INTO $destinationTable
            SELECT xx.theId, 
                   IFNULL(jj.n, 0) AS jobs,
                   IFNULL(vv.n, 0) AS vol, 
                   '$theKeyword'
            FROM   xx 
                   LEFT OUTER JOIN jj ON xx.theId = jj.theId
                   LEFT OUTER JOIN vv ON xx.theId = vv.theId");
}

sub runTopUsersQueriesNow() {
    my ($theLimit, $siteName) = @_;

    &printNow("Now: U ");

    $destinationTable = "${siteName}_topPerfUsersNow";
    $pastTable        = "${siteName}_topPerfUsersPast";

    # now jobs
    &runQuery("INSERT INTO jj
          SELECT  userId, COUNT(DISTINCT jobId ) AS n
            FROM  ${siteName}_openedSessions
        GROUP BY  userId");

    # now files
    &runQuery ("INSERT INTO ff 
          SELECT  tmp.userId, 
                  COUNT(tmp.pathId) AS n,
                  SUM(tmp.size)/(1024*1024) AS s
            FROM  (SELECT DISTINCT userId, pathId, size
                     FROM  ${siteName}_openedSessions os,
                           ${siteName}_openedFiles of,
                           paths p
                    WHERE  os.id = of.sessionId     AND
                           of.pathId = p.id
                  )    AS  tmp
        GROUP BY  tmp.userId");

    ##### now find all names for top X for each sorting 
    &runQuery("REPLACE INTO xx SELECT theId FROM $pastTable");
    &runQuery("REPLACE INTO xx SELECT theId FROM jj ORDER BY n DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM ff ORDER BY n DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM ff ORDER BY s DESC LIMIT $theLimit");

    &runQuery("DELETE FROM $destinationTable");

    ## and finally insert the new data
    &runQuery("INSERT INTO $destinationTable
          SELECT  DISTINCT xx.theId,
                  IFNULL(jj.n, 0) AS jobs,
                  IFNULL(ff.n, 0) AS files, 
                  IFNULL(ff.s, 0) AS fSize
            FROM  xx 
                  LEFT OUTER JOIN jj ON xx.theId = jj.theId
                  LEFT OUTER JOIN ff ON xx.theId = ff.theId");
}
sub runTopFilesQueriesNow() {
    my ($theLimit, $siteName) = @_;

    &printNow("F\n");

    $destinationTable = "${siteName}_topPerfFilesNow";
    $pastTable        = "${siteName}_topPerfFilesPast";

    # now jobs
    &runQuery("INSERT INTO jj
          SELECT pathId, COUNT(jobId ) AS n
           FROM  ${siteName}_openedSessions os,
                 ${siteName}_openedFiles of
          WHERE  os.id = of.sessionId
       GROUP BY  pathId");


    ##### now find all names for top X for each sorting 
    &runQuery("REPLACE INTO xx SELECT theId FROM $pastTable");
    &runQuery("REPLACE INTO xx SELECT theId FROM jj ORDER BY n DESC LIMIT $theLimit");

    &runQuery("DELETE FROM $destinationTable");

    ## and finally insert the new data
    &runQuery("INSERT INTO $destinationTable
          SELECT  DISTINCT xx.theId,
                  IFNULL(jj.n, 0) AS jobs
            FROM  xx 
                  LEFT OUTER JOIN jj ON xx.theId = jj.theId");
}
sub runTopSkimsQueriesPast() {
    my ($theKeyword, $theLimit, $what, $siteName) = @_;

    my $oneCharW = substr($what, 0, 1);
    my $oneCharT = substr($theKeyword, 0, 1);
    &printNow("$oneCharW ");

    my $idInPathTable    = "INVALID";
    my $destinationTable = "INVALID";

    if ( $what eq "SKIMS" ) {
        $idInPathTable    = "skimId";
        $destinationTable = "${siteName}_topPerfSkimsPast";
    } elsif ( $what eq "TYPES" ) {
        $idInPathTable    = "typeId";
        $destinationTable = "${siteName}_topPerfTypesPast";
    } else {
        die "Invalid arg, expected SKIMS or TYPES\n";
    }

    # past jobs
    &runQuery("REPLACE INTO jj
        SELECT oc.theId,  
               COUNT(DISTINCT oc.jobId ) AS n
        FROM   (SELECT $idInPathTable AS theId, jobId
                  FROM ${siteName}_closedSessions_Last$theKeyword cs,
                       ${siteName}_closedFiles_Last$theKeyword cf,
                       paths p
                 WHERE cs.id = cf.sessionId   AND
                       cf.pathId = p.id 
             UNION ALL
                SELECT $idInPathTable AS theId, jobId
                  FROM ${siteName}_openedSessions os,
                      ${siteName}_closedFiles_Last$theKeyword cf,
                       paths p
                 WHERE os.id = cf.sessionId   AND
                       cf.pathId = p.id
               ) AS oc
     GROUP BY oc.theId");

    # past files
    &runQuery("INSERT INTO ff 
        SELECT tmp.$idInPathTable,
               COUNT(tmp.pathId),
               SUM(tmp.size)/(1024*1024)
          FROM ( SELECT DISTINCT $idInPathTable, pathId, size
                   FROM ${siteName}_closedFiles_Last$theKeyword cf,
                        paths p
                  WHERE cf.pathId = p.id
               )     AS tmp
      GROUP BY tmp.$idInPathTable");


    # past users
    &runQuery("REPLACE INTO uu
       SELECT  $idInPathTable, 
               COUNT(DISTINCT userId) AS n
         FROM  ${siteName}_closedSessions_Last$theKeyword cs,
               ${siteName}_closedFiles_Last$theKeyword cf,
               paths p
        WHERE  cs.id = cf.sessionId   AND
               cf.pathId = p.id
     GROUP BY  $idInPathTable");

    # past volume - through opened & closed sessions
    &runQuery("INSERT INTO vv
         SELECT $idInPathTable, 
                SUM(bytesR/(1024*1024))
           FROM ${siteName}_closedFiles_Last$theKeyword cf,
                paths p
          WHERE cf.pathId = p.id
       GROUP BY $idInPathTable");

    ##### now find all names for top X for each sorting 
    &runQuery("REPLACE INTO xx SELECT theId FROM jj ORDER BY n DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM ff ORDER BY n DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM ff ORDER BY s DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM uu ORDER BY n DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM vv ORDER BY n DESC LIMIT $theLimit");

    ## delete old data
    &runQuery("DELETE FROM $destinationTable WHERE timePeriod LIKE '$theKeyword'");

    ## and finally insert the new data
    &runQuery("INSERT INTO $destinationTable
        SELECT xx.theId,
               IFNULL(jj.n, 0) AS jobs,
               IFNULL(ff.n, 0) AS files,
               IFNULL(ff.s, 0) AS fSize,
               IFNULL(uu.n, 0) AS users, 
               IFNULL(vv.n, 0) AS vol, 
               '$theKeyword'
        FROM   xx 
               LEFT OUTER JOIN jj ON xx.theId = jj.theId
               LEFT OUTER JOIN ff ON xx.theId = ff.theId
               LEFT OUTER JOIN uu ON xx.theId = uu.theId
               LEFT OUTER JOIN vv ON xx.theId = vv.theId");
}
sub runTopSkimsQueriesNow() {
    my ($theLimit, $what, $siteName) = @_;

    my $oneCharW = substr($what, 0, 1);
    &printNow("$oneCharW ");

    my $idInPathTable    = "INVALID";
    my $destinationTable = "INVALID";

    if ( $what eq "SKIMS" ) {
        $idInPathTable    = "skimId";
        $destinationTable = "${siteName}_topPerfSkimsNow";
        $pastTable        = "${siteName}_topPerfSkimsPast";
    } elsif ( $what eq "TYPES" ) {
        $idInPathTable    = "typeId";
        $destinationTable = "${siteName}_topPerfTypesNow";
        $pastTable        = "${siteName}_topPerfTypesPast";
    } else {
        die "Invalid arg, expected SKIMS or TYPES\n";
    }

    # now jobs
    &runQuery("INSERT INTO jj
        SELECT $idInPathTable,
               COUNT(DISTINCT jobId ) AS n
        FROM   ${siteName}_openedSessions os,
               ${siteName}_openedFiles of,
               paths p
        WHERE  os.id = of.sessionId     AND
               of.pathId = p.id
      GROUP BY $idInPathTable");

    # now files
    &runQuery("REPLACE INTO ff 
        SELECT tmp.$idInPathTable,
               COUNT(tmp.pathId) AS n,
               SUM(tmp.size)/(1024*1024)  AS s
          FROM ( SELECT  DISTINCT $idInPathTable, pathId, size
                   FROM  ${siteName}_openedFiles of,
                         paths p
                  WHERE  of.pathId = p.id
               )     AS  tmp
      GROUP BY tmp.$idInPathTable");

    # now users
    &runQuery("REPLACE INTO uu 
        SELECT $idInPathTable,
               COUNT(DISTINCT userId) AS n
          FROM ${siteName}_openedSessions os,
               ${siteName}_openedFiles of,
               paths p
         WHERE os.id = of.sessionId
           AND of.pathId = p.id
      GROUP BY $idInPathTable");

    ##### now find all names for top X for each sorting 
    &runQuery("REPLACE INTO xx SELECT theId FROM $pastTable");
    &runQuery("REPLACE INTO xx SELECT theId FROM jj ORDER BY n DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM ff ORDER BY n DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM ff ORDER BY s DESC LIMIT $theLimit");
    &runQuery("REPLACE INTO xx SELECT theId FROM uu ORDER BY n DESC LIMIT $theLimit");

    &runQuery("DELETE FROM $destinationTable");

    ## and finally insert the new data
    &runQuery("INSERT INTO $destinationTable
        SELECT DISTINCT xx.theId,
               IFNULL(jj.n, 0) AS jobs,
               IFNULL(ff.n, 0) AS files, 
               IFNULL(ff.s, 0) AS fSize,
               IFNULL(uu.n, 0) AS users
        FROM   xx 
               LEFT OUTER JOIN jj ON xx.theId = jj.theId
               LEFT OUTER JOIN ff ON xx.theId = ff.theId
               LEFT OUTER JOIN uu ON xx.theId = uu.theId");
}
sub returnHash() {
    ($_) = @_;
    my $i = 1;
    tr/0-9a-zA-Z/0-90-90-90-90-90-90-1/;
    tr/0-9//cd;
    my $hashValue = 0;
    foreach $char ( split / */ ) {
	$i++;
      # $primes initialized in doInit()
	$hashValue += $i * $primes[$char];
    }
    return $hashValue;
}
sub printNow() {
    my ($x) = @_;
    my $prev = $|;
    $| = 1;
    print $x;
    $| = $prev;
}

sub timeToSeqNo() {
    my ($t2, $period) = @_;

    if ( $period eq 'Hour' ) {
        return (&runQueryWithRet("SELECT MINUTE('$t2') % 60"));
    } elsif ( $period eq 'Day' ) {
        return (&runQueryWithRet("SELECT HOUR('$t2')*4+FLOOR(MINUTE('$t2')/15)"));
    } elsif ( $period eq 'Week' ) {
        return (&runQueryWithRet("SELECT ((WEEKDAY('$t2')+1)%7)*24+HOUR('$t2')"));
    } elsif ( $period eq 'Month' ) {
        return (&runQueryWithRet("SELECT (DAY('$t2')-1)*4+FLOOR(HOUR('$t2')/6)"));
    } elsif ( $period eq 'Year' ) {
        return (&runQueryWithRet("SELECT DAYOFYEAR('$t2')"));
    }
}
sub getLastInsertTime() {
    # calculates the nearest allowed insert time to reference (snapshot or load) time.
    use vars qw($sec $min $hour $day $lastInsertTime);
    my ($referenceTime, $period) = @_;

    # zero the second for all periods
    $sec = &runQueryWithRet("SELECT SECOND('$referenceTime')");
    $lastInsertTime = &runQueryWithRet("SELECT DATE_SUB('$referenceTime', INTERVAL $sec SECOND)");

    if ( $period eq "Hour" ) {return($lastInsertTime);}

    # zero the minute for everything but "Day" where it is every 15 minutes
    $min = &runQueryWithRet("SELECT MINUTE('$lastInsertTime')");

    if ( $period eq "Day" ) { 
         $min = $min % 15;
         return( &runQueryWithRet("SELECT DATE_SUB('$lastInsertTime', INTERVAL $min MINUTE)"));
    }

    $lastInsertTime = &runQueryWithRet("SELECT DATE_SUB('$lastInsertTime', INTERVAL $min MINUTE)");
    if ( $period eq "Week" ) {return($lastInsertTime);}

    # zero the hour for "Year" and "AllYears" and every 6 hours for "Month"
    if ( $period eq "Month" ) {
         $hour = &runQueryWithRet("SELECT HOUR('$lastInsertTime') % 6");
         return( &runQueryWithRet("SELECT DATE_SUB('$lastInsertTime', INTERVAL $hour HOUR)"));
    } 

    $hour = &runQueryWithRet("SELECT HOUR('$lastInsertTime')");
    if ( $period eq "Year" ) {
    return( &runQueryWithRet("SELECT DATE_SUB('$lastInsertTime', INTERVAL $hour HOUR)"));
    }
    
    
    # zero the day for "AllYears"
    if ( $period eq "AllYears" ) {
         $lastInsertTime = &runQueryWithRet("SELECT DATE_SUB('$lastInsertTime', INTERVAL $hour HOUR)");
         $day = &runQueryWithRet("SELECT WEEKDAY('$lastInsertTime')+1 ");
         return( &runQueryWithRet("SELECT DATE_SUB('$lastInsertTime', INTERVAL $day DAY)"));
    }
}
sub fillStatsAllYearsMissingBins() {
    use vars qw( $sourceTabTmin $firstInsertTime $avNumJobs $avNumUsers $avNumUniqueFiles $avNumNonUniqueFiles $seqNo);
    my ($siteName, $loadTime) = @_; 
    my $siteId = $siteIds{$siteName};

    # find the number of seq for this period and return if all are there
    my $lastInsertTime = &getLastInsertTime($loadTime, "AllYears");
    my $firstInsertTime = &getLastInsertTime($firstDates{$siteName}, "AllYears" );
    my $nSeqs = &runQueryWithRet("SELECT FLOOR(TIMESTAMPDIFF(WEEK,'$firstInsertTime','$lastInsertTime'))");
    if ( &runQueryWithRet("SELECT COUNT(*)
                             FROM statsAllYears
                            WHERE siteId = $siteId")
         == $nSeqs ) { return;}

    my $intervalSec = 7 * 86400;

    $sourceTabTmin = &runQueryWithRet("SELECT MIN(date) FROM statsLastYear WHERE siteId = $siteId");
                                          
    my $t2 = $firstInsertTime;
    my $t1 = &runQueryWithRet("SELECT DATE_SUB('$t2', INTERVAL 1 WEEK)");
    
    # use full tables to get older statistics
    while ( $t2 le $dbUpdates{$siteName} ) {
        if ( ! &runQueryWithRet("SELECT date 
                                   FROM statsAllYears 
                                  WHERE siteId = $siteId AND
                                          date = '$t2' ") ) {
        
            if ( $t1 lt $sourceTabTmin ) {
                # use full tables to get older statistics
                $avNumJobs = &avNumJobsInTimeInterval($t1, $t2, $intervalSec, $siteName);
                $avNumUsers = &avNumUsersInTimeInterval($t1, $t2, $intervalSec, $siteName);
                $avNumUniqueFiles = &avNumUniqueFilesInTimeInterval($t1, $t2, $intervalSec, $siteName, "All");
                $avNumNonUniqueFiles = &avNumNonUniqueFilesInTimeInterval($t1, $t2, $intervalSec, $siteName, "All");
            } else {
                # use stats table from previous period
                ($avNumJobs, $avNumUsers, $avNumUniqueFiles, $avNumNonUniqueFiles) =
                  &runQueryWithRet("SELECT AVG(noJobs), AVG(noUsers), AVG(noUniqueF), AVG(noNonUniqueF)
                                      FROM statsLastYear
                                     WHERE siteId = $siteId  AND
                                           date >  '$t1'      AND
                                           date <= '$t2'         ");
            }
            if ( $avNumJobs ) {
                &runQuery("INSERT IGNORE INTO statsAllYears
                                VALUES ($siteId, '$t2', $avNumJobs, $avNumUsers, $avNumUniqueFiles, $avNumNonUniqueFiles)");
            }
        }
        $t1 = $t2;
        $t2 = &runQueryWithRet("SELECT DATE_ADD('$t2', INTERVAL 1 WEEK)");
    }

}
sub fillStatsMissingBins() {
    use vars qw($sourceTable $firstInsertTime $avNumJobs $avNumUsers $avNumUniqueFiles $avNumNonUniqueFiles $seqNo);
    my ($siteName, $period, $loadTime) = @_; 
    my $siteId = $siteIds{$siteName};

    my $interval = $intervals{$period};
    my $timeUnit = $timeUnits{$period};
    
    # find the number of seq for this period and return if all are there
    my $lastInsertTime = &getLastInsertTime($loadTime, $period);
    my $firstInsertTime = &runQueryWithRet("SELECT DATE_SUB('$lastInsertTime', INTERVAL 1 $period)");
    my $nSeqs = &runQueryWithRet("SELECT FLOOR(TIMESTAMPDIFF($timeUnit,'$firstInsertTime','$lastInsertTime')/$interval)");
    if ( &runQueryWithRet("SELECT COUNT(*)
                             FROM statsLast$period
                            WHERE siteId = $siteId")
         == $nSeqs ) { return;}

    my $intervalSec = $interval * $seconds{$timeUnit};

    if ( $period eq "Day" ) {
        $sourceTabTmin = "";
    } else {
        $sourceTable = "statsLast$sourcePeriods{$period}";
        $sourceTabTmin = &runQueryWithRet("SELECT MIN(date) FROM $sourceTable WHERE siteId = $siteId");
    }
                                          

    my $t2 = $firstInsertTime;
    my $t1 = &runQueryWithRet("SELECT DATE_SUB('$t2', INTERVAL $interval $timeUnit)");
    
    # use full tables to get older statistics
    while ( $t2 le $dbUpdates{$siteName} ) {
        $seqNo = &timeToSeqNo( $t2, $period );
        if ( ! &runQueryWithRet("SELECT date FROM statsLast$period WHERE siteId = $siteId AND seqNo = $seqNo") ) {

            if ( $period eq "Day" or $t1 lt $sourceTabTmin ) {
                # use full tables to get older statistics
                $avNumJobs = &avNumJobsInTimeInterval($t1, $t2, $intervalSec, $siteName);
                $avNumUsers = &avNumUsersInTimeInterval($t1, $t2, $intervalSec, $siteName);
                $avNumUniqueFiles = &avNumUniqueFilesInTimeInterval($t1, $t2, $intervalSec, $siteName, $period);
                $avNumNonUniqueFiles = &avNumNonUniqueFilesInTimeInterval($t1, $t2, $intervalSec, $siteName, $period);
            } else {
                # use stats table from previous period
                ($avNumJobs, $avNumUsers, $avNumUniqueFiles, $avNumNonUniqueFiles) =
                  &runQueryWithRet("SELECT AVG(noJobs), AVG(noUsers), AVG(noUniqueF), AVG(noNonUniqueF)
                                      FROM $sourceTable
                                     WHERE siteId = $siteId  AND
                                           date >  '$t1'      AND
                                           date <= '$t2'         ");
            }
            if ( $avNumJobs ) {
                &runQuery("REPLACE INTO statsLast$period
                                 VALUES ($seqNo, $siteId, '$t2', $avNumJobs, $avNumUsers, $avNumUniqueFiles, $avNumNonUniqueFiles)");
            }
        }
        $t1 = $t2;
        $t2 = &runQueryWithRet("SELECT DATE_ADD('$t2', INTERVAL $interval $timeUnit)");
    }
}
 
sub avNumJobsInTimeInterval() {
    my ($t1, $t2, $interval, $siteName) = @_;

    if ( $t2 lt $firstDates{$siteName} ) {return(0);}
    print "sub avNumJobsInTimeInterval: $t1, $t2, $interval, $siteName \n";

    &runQuery("INSERT INTO times
                     SELECT GREATEST('$t1', beginT), LEAST('$t2', endT)
                       FROM ${siteName}_jobs
                      WHERE beginT < '$t2'         AND
                            endT   > '$t1'         AND
                            noOpenSessions = 0             ");
    print "Query 1 finished \n";
    &runQuery("INSERT INTO times
                     SELECT GREATEST('$t1', beginT), '$t2'
                       FROM ${siteName}_jobs
                      WHERE beginT < '$t2'         AND
                            noOpenSessions > 0             ");

    print "Query 2 finished \n";
    my $sumT = $interval *
               &runQueryWithRet("SELECT IFNULL(COUNT(*),0)
                                   FROM times
                                  WHERE begT = '$t1'  AND
                                        endT = '$t2'     ");

    $sumT += &runQueryWithRet("SELECT IFNULL(SUM(TIMESTAMPDIFF(SECOND,begT,endT)),0)
                                 FROM times
                                WHERE begT > '$t1'         OR
                                      endT <  '$t2'              ");
    &runQuery("TRUNCATE TABLE times");

    return (int ($sumT / $interval + .5 ));
}

sub avNumUsersInTimeInterval() {
    my ($t1, $t2, $interval, $siteName) = @_;

    if ( $t2 lt $firstDates{$siteName} ) {return(0);}
    print "sub avNumUsersInTimeInterval:$t1, $t2, $interval, $siteName \n";

    &runQuery("INSERT INTO id_times
                     SELECT userId, GREATEST('$t1', beginT), LEAST('$t2', endT)
                       FROM ${siteName}_jobs
                      WHERE beginT < '$t2'         AND
                            endT   > '$t1'         AND
                            noOpenSessions = 0             ");
    print "Query 1 finished \n";

    &runQuery("INSERT INTO id_times
                     SELECT userId, GREATEST('$t1', beginT), '$t2'
                       FROM ${siteName}_jobs
                      WHERE beginT < '$t2'         AND
                             noOpenSessions > 0             ");

    print "Query 2 finished \n";

    # find the user list in the time interval
    @userIds = &runQueryRetArray("SELECT DISTINCT theId
                                    FROM id_times           ");


    foreach $userId ( @userIds ) {
        my @beginTs = &runQueryRetArray("SELECT begT
                                           FROM id_times
                                          WHERE theId = $userId
                                       ORDER BY begT           ");

        my @endTs = &runQueryRetArray("SELECT endT
                                         FROM id_times
                                        WHERE theId = $userId
                                     ORDER BY begT           ");

        my $nJobs = @beginTs;
        my $beginT = $beginTs[0];
        my $endT   = $endTs[0];
        if ( $nJobs == 1 or $endTs[0] eq $t2 ) {
             &runQuery("INSERT INTO times
                             VALUES ('$beginT', '$endT') ");
             next;
        }

        my $n = 1;
        while ( $n < $nJobs ) {
            if ( $beginTs[$n] le $endT ) {
                 $endT = $endTs[$n];
            } else {
                 &runQuery("INSERT INTO times
                                 VALUES ('$beginT', '$endT') ");
                 $beginT = $beginTs[$n];
                 $endT   = $endTs[$n];
            }
            if ( $endT eq $t2 or $n = $nJobs - 1) {
                 &runQuery("INSERT INTO times
                                 VALUES ('$beginT', '$endT') ");
                 last;
            }
            $n++;
        }
    }
    print "Query 3 finished \n";

    $sumT = &runQueryWithRet("SELECT IFNULL(SUM(TIMESTAMPDIFF(SECOND,begT,endT)), 0)
                                FROM times" );
    &runQuery("TRUNCATE TABLE id_times");
    &runQuery("TRUNCATE TABLE times");

    return (int ($sumT / $interval + .5 ));
}

sub avNumNonUniqueFilesInTimeInterval() {
    use vars qw( $closedFiles );
    my ($t1, $t2, $interval, $siteName, $cfPeriod) = @_;
    if ( $cfPeriod eq "All" ) { 
         $closedFiles = "${siteName}_closedFiles";
    } else {
         $closedFiles = "${siteName}_closedFiles_Last$cfPeriod";
    }
    if ( $t2 lt $firstDates{$siteName} ) {return(0);}
    print "sub avNumNonUniqueFilesInTimeInterval: $t1, $t2, $interval, $siteName \n";

    &runQuery("INSERT INTO times
                     SELECT GREATEST('$t1', openT), LEAST('$t2', closeT)
                       FROM $closedFiles
                      WHERE openT  < '$t2'         AND
                            closeT > '$t1'             ");
    print "Query 1 finished \n";

    &runQuery("INSERT INTO times
                     SELECT GREATEST('$t1', openT), '$t2'
                       FROM ${siteName}_openedFiles
                      WHERE openT < '$t2'         ");
    print "Query 2 finished \n";

    my $sumT = $interval *
               &runQueryWithRet("SELECT IFNULL(COUNT(*),0)
                                   FROM times
                                  WHERE begT = '$t1'  AND
                                        endT = '$t2'     ");

    $sumT += &runQueryWithRet("SELECT IFNULL(SUM(TIMESTAMPDIFF(SECOND,begT,endT)),0)
                                 FROM times
                                WHERE begT > '$t1'         OR
                                      endT <  '$t2'              ");
    &runQuery("TRUNCATE TABLE times");

    return (int ($sumT / $interval + .5 ));
}

sub avNumUniqueFilesInTimeInterval() {
    use vars qw( $closedFiles );
    my ($t1, $t2, $interval, $siteName, $cfPeriod) = @_;
    if ( $cfPeriod eq "All" ) { 
         $closedFiles = "${siteName}_closedFiles";
    } else {
         $closedFiles = "${siteName}_closedFiles_Last$cfPeriod";
    }

    if ( $t2 lt $firstDates{$siteName} ) {return(0);}
    print "sub avNumUniqueFilesInTimeInterval: $t1, $t2, $interval, $siteName \n";

    &runQuery("INSERT INTO id_times
                     SELECT pathId, GREATEST('$t1', openT), LEAST('$t2', closeT)
                       FROM $closedFiles
                      WHERE openT  < '$t2'         AND
                            closeT > '$t1'             ");
    print "Query 1 finished \n";

    &runQuery("INSERT INTO id_times
                     SELECT pathId, GREATEST('$t1', openT), '$t2'
                       FROM ${siteName}_openedFiles
                      WHERE openT < '$t2'              ");
    print "Query 2 finished \n";


    # find the file list in the time interval
    @pathIds = &runQueryRetArray("SELECT DISTINCT theId
                                    FROM id_times          ");



    foreach $pathId ( @pathIds ) {
        my @openTs =  &runQueryRetArray("SELECT begT
                                           FROM id_times
                                          WHERE theId = $pathId
                                       ORDER BY begT           ");

        my @closeTs = &runQueryRetArray("SELECT endT
                                           FROM id_times
                                          WHERE theId = $pathId
                                       ORDER BY begT           ");

        my $nFiles = @openTs;
        my $openT  = $openTs[0];
        my $closeT = $closeTs[0];
        if ( $nFiles == 1 or $closeTs[0] eq $t2 ) {
             &runQuery("INSERT INTO times
                             VALUES ('$openT', '$closeT') ");
             next;
        }

        my $n = 1;
        while ( $n < $nFiles ) {
            if ( $openTs[$n] le $closeT ) {
                 $closeT = $closeTs[$n];
            } else {
                 &runQuery("INSERT INTO times
                                 VALUES ('$openT', '$closeT') ");
                 $openT  = $openTs[$n];
                 $closeT = $closeTs[$n];
            }
            if ( $closeT eq $t2 or $n = $nFiles - 1) {
                 &runQuery("INSERT INTO times
                                 VALUES ('$openT', '$closeT') ");
                 last;
            }
            $n++;
        }
    }
    print "Query 3 finished \n";

    $sumT = &runQueryWithRet("SELECT IFNULL(SUM(TIMESTAMPDIFF(SECOND,begT,endT)), 0)
                                FROM times" );
    &runQuery("TRUNCATE TABLE id_times");
    &runQuery("TRUNCATE TABLE times");

    return (int ($sumT / $interval + .5 ));
}

