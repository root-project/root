#!/usr/local/bin/perl -w

use DBI;
use Fcntl;
 
# take care of arguments
if ( @ARGV != 1 ) {
    print "Expected argument <configFile>\n";
    exit;
}

&readConfigFile($ARGV[0], 'create', 1);

@periods = ( 'Hour', 'Day', 'Week', 'Month', 'Year');
%maxStatsRows = ('Hour' => 255        , 'Day' => 255       , 'Week' => 255      , 'Month' => 255      ,   'Year' => 512 , 'AllYears' =>  1024    );  
%maxLastRows  = ('Hour' => 16777215   , 'Day' => 16777215  , 'Week' => 16777215 , 'Month' => 4294967296 , 'Year' => 1099511627776     );

$maxRowsTopPerfNow = 5 * $nTopPerfRows;
$maxRowsTopPerfPast = 25 * $nTopPerfRows;

# collector version
$collector = `which xrdmonCollector`;
chomp $collector;
if ( $collector =~ 'xrdmonCollector$') {
     $line = `xrdmonCollector -ver`;
     @line = split('\t', $line);
     $version = int($line[1]);
} else { 
     print "xrdmonCollector not in your PATH. No database created. \n";
     exit;
}

@dbs = `mysql -S $mysqlSocket -u $mySQLUser -B -e "SHOW DATABASES"`;
foreach $db ( @dbs ) {
    chomp $db;
    if ( $db eq $dbName ) {
         die "================ Exiting... $db already exists\n";
    }
}
`mysql -S $mysqlSocket -u $mySQLUser -e "CREATE DATABASE $dbName"`;

unless ( $dbh = DBI->connect("dbi:mysql:$dbName;mysql_socket=$mysqlSocket",$mySQLUser) ) {
    print "Error while connecting to database. $DBI::errstr\n";
    exit;
}

&runQuery("CREATE TABLE IF NOT EXISTS sites (
           id            	TINYINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
           name          	VARCHAR(32),
           timezone      	VARCHAR(64),
           backupInt     	VARCHAR(19) DEFAULT '$backupIntDef',
           backupTime    	DATETIME,
           dbUpdate      	DATETIME,
           version       	TINYINT NOT NULL,
           firstDate     	DATETIME NOT NULL,
           closeFileT    	DATETIME NOT NULL,
           closeIdleSessionT	DATETIME NOT NULL,
           closeLongSessionT	DATETIME NOT NULL
           ) MAX_ROWS=255");

# reflects changes since last entry, and last update
&runQuery("CREATE TABLE IF NOT EXISTS rtChanges (
           siteId        TINYINT UNSIGNED NOT NULL PRIMARY KEY,
           jobs          MEDIUMINT UNSIGNED,
           jobs_p        FLOAT,
           users         MEDIUMINT UNSIGNED,
           users_p       FLOAT,
           uniqueF       MEDIUMINT UNSIGNED,
           uniqueF_p     FLOAT,
           nonUniqueF    MEDIUMINT UNSIGNED,
           nonUniqueF_p  FLOAT,
           lastUpdate    DATETIME
           ) MAX_ROWS=255");

&runQuery("CREATE TABLE IF NOT EXISTS xrdRestarts (
           hostId        SMALLINT UNSIGNED NOT NULL,
           siteId        TINYINT UNSIGNED NOT NULL,
           startT        DATETIME,
           PRIMARY KEY (siteId, hostId, startT)
           ) MAX_ROWS=65535");

&runQuery("CREATE TABLE IF NOT EXISTS paths (
           id            MEDIUMINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
           name          VARCHAR(255) NOT NULL,
           hash          MEDIUMINT NOT NULL DEFAULT 0,
           INDEX (hash)
           ) MAX_ROWS=4294967296");

&runQuery("CREATE TABLE IF NOT EXISTS fileInfo (
           id            MEDIUMINT UNSIGNED NOT NULL PRIMARY KEY,
           size          BIGINT  NOT NULL DEFAULT 0
           ) MAX_ROWS=4294967296");

&runQuery("CREATE TABLE IF NOT EXISTS fileTypes (
           name         VARCHAR(32),
           tId          TINYINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY
           ) MAX_ROWS=255");


foreach $fileType (@fileTypes) {

     &runQuery("INSERT INTO fileTypes (name) VALUES ('$fileType')");
     $tId = &runQueryWithRet("SELECT LAST_INSERT_ID()");
    
     &runQuery("CREATE TABLE IF NOT EXISTS fileType_$tId (
                name         VARCHAR(32),
                id           SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY
                ) MAX_ROWS=$maxRowsTypes{$fileType}");

     &runQuery("ALTER TABLE fileInfo
                ADD COLUMN (typeId_$tId SMALLINT UNSIGNED NOT NULL)");
     &runQuery("ALTER TABLE fileInfo
                ADD INDEX (typeId_$tId) ");
}

# make sure $thisSite gets id = 1 in sites
&runQuery("INSERT INTO sites (name, version) 
                VALUES ('$thisSite', $version)");

foreach $site (@sites) {
    $firstDate = $firstDates{$site};
    $timezone = $timezones{$site};
    if ( $site ne $thisSite ) {
       &runQuery("INSERT INTO sites (name) 
                    VALUES       ('$site')");
    }
    &runQuery("UPDATE sites
                  SET timezone   = '$timezone',
                      firstDate  = '$firstDate', 
                      backupTime = '$firstDate',
                      dbUpdate   = '$firstDate',
                      closeFileT = '$firstDate',
                      closeIdleSessionT = '$firstDate',
                      closeLongSessionT = '$firstDate'
                WHERE name = '$site' ");
    
    if ( $backupInts{$site} ) {
         &runQuery("UPDATE sites
                       SET backupInt = '$backupInts{$site}'
                     WHERE name = '$site' ");
    }

    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_jobs (
               jobId          INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
               userId         SMALLINT  UNSIGNED NOT NULL,
               pId            SMALLINT  UNSIGNED NOT NULL,
               clientHId      SMALLINT  UNSIGNED NOT NULL,
               noOpenSessions SMALLINT  UNSIGNED NOT NULL,
               beginT         DATETIME NOT NULL,
               endT           DATETIME NOT NULL,
               INDEX ( userId, pId,  clientHId )
               ) MAX_ROWS=16777215");
                                                                           

    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_openedSessions (
              id            INT UNSIGNED NOT NULL PRIMARY KEY,
              jobId         INT       UNSIGNED NOT NULL,
              userId        SMALLINT  UNSIGNED NOT NULL,
              pId           SMALLINT  UNSIGNED NOT NULL,
              clientHId     SMALLINT  UNSIGNED NOT NULL,
              serverHId     SMALLINT  UNSIGNED NOT NULL,
              connectT      DATETIME  NOT NULL,
              INDEX (userId),
              INDEX (pId),
              INDEX (clientHId),
              INDEX (serverHId)
              ) MAX_ROWS=16777215");
    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_closedSessions (
             id            INT       UNSIGNED NOT NULL PRIMARY KEY,
             jobId         INT       UNSIGNED NOT NULL,
             userId        SMALLINT  UNSIGNED NOT NULL,
             pId           SMALLINT  UNSIGNED NOT NULL,
             clientHId     SMALLINT  UNSIGNED NOT NULL,
             serverHId     SMALLINT  UNSIGNED NOT NULL,
             duration      MEDIUMINT NOT NULL,
             disconnectT   DATETIME  NOT NULL,
             status        CHAR(1)   NOT NULL DEFAULT 'N',
             INDEX (userId),
             INDEX (pId),
             INDEX (clientHId),
             INDEX (serverHId),
             INDEX (disconnectT)
             ) MAX_ROWS=1099511627776");

    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_openedFiles (
               id            INT       UNSIGNED NOT NULL PRIMARY KEY,
               sessionId     INT       UNSIGNED NOT NULL,
               pathId        MEDIUMINT UNSIGNED NOT NULL,
               openT         DATETIME  NOT NULL,
               INDEX (sessionId),
               INDEX (pathId),
               INDEX (openT)
               ) MAX_ROWS=16777215");

    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_closedFiles (
               id            INT       UNSIGNED NOT NULL PRIMARY KEY,
               sessionId     INT       UNSIGNED NOT NULL,
               pathId	MEDIUMINT UNSIGNED NOT NULL,
               openT         DATETIME  NOT NULL,
               closeT        DATETIME  NOT NULL,
               bytesR        BIGINT    NOT NULL,
               bytesW        BIGINT    NOT NULL,
               INDEX (sessionId),
               INDEX (pathId),
               INDEX (closeT)
               ) MAX_ROWS=1099511627776");

    foreach $period ( @periods ) {
        &runQuery("CREATE TABLE IF NOT EXISTS ${site}_closedSessions_Last$period 
                                         LIKE ${site}_closedSessions");

        &runQuery("ALTER TABLE ${site}_closedSessions_Last$period   
                      MAX_ROWS=$maxLastRows{$period}");

        &runQuery("CREATE TABLE IF NOT EXISTS ${site}_closedFiles_Last$period 
                                         LIKE ${site}_closedFiles");

        &runQuery("ALTER TABLE ${site}_closedFiles_Last$period   
                      MAX_ROWS=$maxLastRows{$period}");
    }

    # compressed info for top performers (top users)
    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_topPerfUsersNow (
               theId      INT NOT NULL,    # user Id
               jobs       INT NOT NULL,
               files      INT NOT NULL,
               fSize      INT NOT NULL     # [MB]
               ) MAX_ROWS=$maxRowsTopPerfNow");
    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_topPerfUsersPast (
               theId      INT NOT NULL,    # user Id
               jobs       INT NOT NULL,
               files      INT NOT NULL,
               fSize      INT NOT NULL,    # [MB]
               volume     INT NOT NULL,
               timePeriod CHAR(6)          # 'hour', 'day', 'week', 'month', 'year'
               ) MAX_ROWS=$maxRowsTopPerfPast");

    # compressed info for top performers (top files)
    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_topPerfFilesNow (
               theId      INT NOT NULL,    # path Id
               jobs       INT NOT NULL
               ) MAX_ROWS=$maxRowsTopPerfNow");
    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_topPerfFilesPast (
               theId      INT NOT NULL,    # path Id
               jobs       INT NOT NULL,
               volume     INT NOT NULL,
               timePeriod CHAR(6)          # 'hour', 'day', 'week', 'month', 'year'
               ) MAX_ROWS=$maxRowsTopPerfPast");

    # compressed info for top performers (top file types)
    foreach $fileType (@fileTypes) {
        $no = &runQueryWithRet("SELECT tId 
                                  FROM fileTypes
                                 WHERE name = '$fileType' ");

        &runQuery("CREATE TABLE IF NOT EXISTS ${site}_topPerfType_${no}now (
                   theId      INT NOT NULL,    # type Id
                   jobs       INT NOT NULL,
                   files      INT NOT NULL,
                   fSize      INT NOT NULL,    # [MB]
                   users      INT NOT NULL
                   ) MAX_ROWS=$maxRowsTopPerfNow");
        &runQuery("CREATE TABLE IF NOT EXISTS ${site}_topPerfType_${no}past (
                   theId      INT NOT NULL,    # type Id
                   jobs       INT NOT NULL,
                   files      INT NOT NULL,
                   fSize      INT NOT NULL,    # [MB]
                   users      INT NOT NULL,
                   volume     INT NOT NULL,
                   timePeriod CHAR(6)          # 'hour', 'day', 'week', 'month', 'year'
                   ) MAX_ROWS=$maxRowsTopPerfPast");
    }

    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_users (
               id            SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
               name          VARCHAR(24) NOT NULL
               ) MAX_ROWS=65535");

    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_hosts (
               id            SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
               hostName      VARCHAR(64) NOT NULL,
               hostType      TINYINT NOT NULL
               ) MAX_ROWS=65535");

    # one row per minute, keeps last 60 minutes
    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_statsLastHour (
               seqNo         SMALLINT UNSIGNED NOT NULL PRIMARY KEY,
               date          DATETIME,
               noJobs        MEDIUMINT UNSIGNED,
               noUsers       MEDIUMINT UNSIGNED,
               noUniqueF     INT,
               noNonUniqueF  INT,
               INDEX (date)
               ) MAX_ROWS=$maxStatsRows{'Hour'}");

    foreach $period ( @periods ) {
        # one row every 15 minutes, LastDay
        # one row per hour, LastWeek
        # one row every 6 hours, LastMonth
        # one row per day, LastYear
        &runQuery("CREATE TABLE IF NOT EXISTS ${site}_statsLast$period LIKE ${site}_statsLastHour");
        &runQuery("ALTER TABLE ${site}_statsLast$period MAX_ROWS=$maxStatsRows{$period}");
    }

    # one row per week, growing indefinitely
    &runQuery("CREATE TABLE IF NOT EXISTS ${site}_statsAllYears  LIKE ${site}_statsLastHour");
    &runQuery("ALTER TABLE ${site}_statsAllYears MAX_ROWS=$maxStatsRows{'Year'}");
    &runQuery("ALTER TABLE ${site}_statsAllYears DROP PRIMARY KEY");
    &runQuery("ALTER TABLE ${site}_statsAllYears DROP COLUMN seqNo");

} ### end of site-specific tables ###

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
            $timezones{$v1} = $v2;
            $firstDates{$v1} = "$v3 $v4";
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

    if ( $caller eq "collector" or $caller eq "create") {
         if ( ! $thisSite) {
            push @missing, "thisSite";    
         }
    } 
    if ( $caller ne  "collector") {
         if ( ! $dbName ) {push @missing, "dbName";}
         if ( ! $mySQLUser ) {push @missing, "MySQLUser";}
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
        print "  MySQLUser: $mySQLUser \n";
        print "  MySQLSocket: $mysqlSocket \n";
        print "  nTopPerfRows: $nTopPerfRows \n";
        if ( $caller eq "create" ) {
             print "  backupIntDef: $backupIntDef \n";
             print "  thisSite: $thisSite \n";
             foreach $site ( @sites ) {
                 print "  site: $site \n";
                 print "     timeZone: $timezones{$site}  \n";
                 print "     firstDate: $firstDates{$site} \n";
                 if ( $backupInts{$site} ) {
                     print "  backupInt: $backupInts{$site} \n";
                 }
             }
             foreach $fileType ( @fileTypes ) {
                 print "  fileType: $fileType $maxRowsTypes{$fileType} \n";
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
    
sub runQuery() {
    my ($sql) = @_;
#    print "$sql;\n";
    my $sth = $dbh->prepare($sql) 
        or die "Can't prepare statement $DBI::errstr\n";
    $sth->execute or die "Failed to exec \"$sql\", $DBI::errstr";
}
sub runQueryWithRet() {
    my $sql = shift @_;
#    print "$sql;\n";
    my $sth = $dbh->prepare($sql) 
        or die "Can't prepare statement $DBI::errstr\n";
    $sth->execute or die "Failed to exec \"$sql\", $DBI::errstr";
    return $sth->fetchrow_array;
}
