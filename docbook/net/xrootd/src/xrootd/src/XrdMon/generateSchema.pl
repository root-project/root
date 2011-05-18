#!/usr/bin/perl

###############################################################################
#                                                                             #
#                              generateSchema.pl                              #
#                                                                             #
#  (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  #
#                             All Rights Reserved                             #
#        Produced by Jacek Becla for Stanford University under contract       #
#               DE-AC02-76SF00515 with the Department of Energy               #
###############################################################################

# $Id$


if ( @ARGV ne 1 ) {
    print "Expected arg: <db name>\n";
    exit;
}
 
my $dbName = $ARGV[0];

@sites     = ("SLAC"  , "RAL"   );
@timezones = ("PST8PDT", "WET");


print "

# DROP DATABASE IF EXISTS $dbName;
# CREATE DATABASE IF NOT EXISTS $dbName;

USE $dbName;


CREATE TABLE IF NOT EXISTS sites (
  id            TINYINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  name          VARCHAR(32),
  timezone      VARCHAR(64),
  backupInt     VARCHAR(19) DEFAULT '1 DAY',
  backupTime    DATETIME,
  dbUpdate      DATETIME,
  version       TINYINT NOT NULL
) MAX_ROWS=255;

# one row per minute, keeps last 60 minutes
CREATE TABLE IF NOT EXISTS statsLastHour (
  seqNo         SMALLINT UNSIGNED NOT NULL,
  siteId        TINYINT UNSIGNED NOT NULL,
  date          DATETIME,
  noJobs        MEDIUMINT UNSIGNED,
  noUsers       MEDIUMINT UNSIGNED,
  noUniqueF     INT,
  noNonUniqueF  INT,
  PRIMARY KEY (seqNo, siteId),
  INDEX (date)
) MAX_ROWS=65535;

# one row every 15 minutes, keeps last 24 hours
CREATE TABLE IF NOT EXISTS statsLastDay LIKE statsLastHour;

# one row per hour, keeps last 7 days
CREATE TABLE IF NOT EXISTS statsLastWeek LIKE statsLastHour;

# one row every 6 hours, keeps last 30 days
CREATE TABLE IF NOT EXISTS statsLastMonth LIKE statsLastHour;

# one row per day, LAST 365 dsys
CREATE TABLE IF NOT EXISTS statsLastYear  LIKE statsLastHour;

# one row per week, growing indefinitely
CREATE TABLE IF NOT EXISTS statsAllYears  LIKE statsLastHour;
ALTER TABLE statsAllYears DROP PRIMARY KEY;
ALTER TABLE statsAllYears DROP COLUMN seqNo;

# reflects changes since last entry, and last update
CREATE TABLE IF NOT EXISTS rtChanges (
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
) MAX_ROWS=255;

CREATE TABLE IF NOT EXISTS paths (
  id            MEDIUMINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  typeId        TINYINT NOT NULL,   # BaBar specific
  skimId        TINYINT NOT NULL,   # BaBar specific
  size          BIGINT  NOT NULL DEFAULT 0,
  hash          MEDIUMINT NOT NULL DEFAULT 0,
  name          VARCHAR(255) NOT NULL,
  INDEX (typeId),
  INDEX (skimId),
  INDEX (hash)
) MAX_ROWS=4294967296;

CREATE TABLE IF NOT EXISTS xrdRestarts (
  hostId        SMALLINT UNSIGNED NOT NULL,
  siteId        TINYINT UNSIGNED NOT NULL,
  startT        DATETIME,
  PRIMARY KEY (siteId, hostId, startT)
) MAX_ROWS=65535;

# BaBar specific!
# e.g.: SP, PR, SPskims, PRskims
CREATE TABLE IF NOT EXISTS fileTypes (
  name         VARCHAR(16),
  id           TINYINT NOT NULL AUTO_INCREMENT PRIMARY KEY
) MAX_ROWS=255;

# BaBar specific!
CREATE TABLE IF NOT EXISTS skimNames (
  name         VARCHAR(32),
  id           SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY
) MAX_ROWS=65535;

";


foreach $site (@sites) {
    print "


################ ${site} ################
CREATE TABLE IF NOT EXISTS ${site}_jobs (
  jobId          INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  userId         SMALLINT  UNSIGNED NOT NULL,
  pId            SMALLINT  UNSIGNED NOT NULL,
  clientHId      SMALLINT  UNSIGNED NOT NULL,
  noOpenSessions SMALLINT  UNSIGNED NOT NULL,
  beginT         DATETIME NOT NULL,
  endT           DATETIME NOT NULL,
  INDEX ( userId, pId,  clientHId )
) MAX_ROWS=16777215;
                                                                           

CREATE TABLE IF NOT EXISTS ${site}_openedSessions (
  id            INT UNSIGNED NOT NULL PRIMARY KEY,
  jobId         INT       UNSIGNED NOT NULL,
  userId        SMALLINT  UNSIGNED NOT NULL,
  pId           SMALLINT  UNSIGNED NOT NULL,
  clientHId     SMALLINT  UNSIGNED NOT NULL,
  serverHId     SMALLINT  UNSIGNED NOT NULL,
  INDEX (userId),
  INDEX (pId),
  INDEX (clientHId),
  INDEX (serverHId)
) MAX_ROWS=16777215;

CREATE TABLE IF NOT EXISTS ${site}_closedSessions (
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
) MAX_ROWS=1099511627776;

CREATE TABLE IF NOT EXISTS ${site}_closedSessions_LastHour  LIKE ${site}_closedSessions;
CREATE TABLE IF NOT EXISTS ${site}_closedSessions_LastDay   LIKE ${site}_closedSessions;
CREATE TABLE IF NOT EXISTS ${site}_closedSessions_LastWeek  LIKE ${site}_closedSessions;
CREATE TABLE IF NOT EXISTS ${site}_closedSessions_LastMonth LIKE ${site}_closedSessions;
CREATE TABLE IF NOT EXISTS ${site}_closedSessions_LastYear  LIKE ${site}_closedSessions;

ALTER TABLE ${site}_closedSessions_LastHour   MAX_ROWS=16777215;
ALTER TABLE ${site}_closedSessions_LastDay    MAX_ROWS=16777215;
ALTER TABLE ${site}_closedSessions_LastWeek   MAX_ROWS=16777215;
ALTER TABLE ${site}_closedSessions_LastMonth  MAX_ROWS=4294967296;

CREATE TABLE IF NOT EXISTS ${site}_openedFiles (
  id            INT       UNSIGNED NOT NULL PRIMARY KEY,
  sessionId     INT       UNSIGNED NOT NULL,
  pathId        MEDIUMINT UNSIGNED NOT NULL,
  openT         DATETIME  NOT NULL,
  INDEX (sessionId),
  INDEX (pathId),
  INDEX (openT)
) MAX_ROWS=16777215;

CREATE TABLE IF NOT EXISTS ${site}_closedFiles (
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
) MAX_ROWS=1099511627776;

CREATE TABLE IF NOT EXISTS ${site}_closedFiles_LastHour  LIKE ${site}_closedFiles;
CREATE TABLE IF NOT EXISTS ${site}_closedFiles_LastDay   LIKE ${site}_closedFiles;
CREATE TABLE IF NOT EXISTS ${site}_closedFiles_LastWeek  LIKE ${site}_closedFiles;
CREATE TABLE IF NOT EXISTS ${site}_closedFiles_LastMonth LIKE ${site}_closedFiles;
CREATE TABLE IF NOT EXISTS ${site}_closedFiles_LastYear  LIKE ${site}_closedFiles;

ALTER TABLE  ${site}_closedFiles_LastHour  MAX_ROWS=16777215;
ALTER TABLE  ${site}_closedFiles_LastDay   MAX_ROWS=16777215;
ALTER TABLE  ${site}_closedFiles_LastWeek  MAX_ROWS=16777215;
ALTER TABLE  ${site}_closedFiles_LastMonth MAX_ROWS=4294967296;

# compressed info for top performers (top users)
CREATE TABLE IF NOT EXISTS ${site}_topPerfUsersNow (
  theId      INT NOT NULL,    # user Id
  jobs       INT NOT NULL,
  files      INT NOT NULL,
  fSize      INT NOT NULL     # [MB]
) MAX_ROWS=65535;
CREATE TABLE IF NOT EXISTS ${site}_topPerfUsersPast (
  theId      INT NOT NULL,    # user Id
  jobs       INT NOT NULL,
  files      INT NOT NULL,
  fSize      INT NOT NULL,    # [MB]
  volume     INT NOT NULL,
  timePeriod CHAR(6)          # \"hour\", \"week\", \"month\", \"year\"
) MAX_ROWS=65535;

# compressed info for top performers (top skims)
CREATE TABLE IF NOT EXISTS ${site}_topPerfSkimsNow (
  theId      INT NOT NULL,    # skim Id
  jobs       INT NOT NULL,
  files      INT NOT NULL,
  fSize      INT NOT NULL,    # [MB]
  users      INT NOT NULL
) MAX_ROWS=65535;
CREATE TABLE IF NOT EXISTS ${site}_topPerfSkimsPast (
  theId      INT NOT NULL,    # skim Id
  jobs       INT NOT NULL,
  files      INT NOT NULL,
  fSize      INT NOT NULL,    # [MB]
  users      INT NOT NULL,
  volume     INT NOT NULL,
  timePeriod CHAR(6)          # \"hour\", \"week\", \"month\", \"year\"
) MAX_ROWS=65535;

# compressed info for top performers (top files)
CREATE TABLE IF NOT EXISTS ${site}_topPerfFilesNow (
  theId      INT NOT NULL,    # path Id
  jobs       INT NOT NULL
) MAX_ROWS=65535;
CREATE TABLE IF NOT EXISTS ${site}_topPerfFilesPast (
  theId      INT NOT NULL,    # path Id
  jobs       INT NOT NULL,
  volume     INT NOT NULL,
  timePeriod CHAR(6)          # \"hour\", \"week\", \"month\", \"year\"
) MAX_ROWS=65535;

# compressed info for top performers (top skims)
CREATE TABLE IF NOT EXISTS ${site}_topPerfTypesNow (
  theId      INT NOT NULL,    # type Id
  jobs       INT NOT NULL,
  files      INT NOT NULL,
  fSize      INT NOT NULL,    # [MB]
  users      INT NOT NULL
) MAX_ROWS=65535;
CREATE TABLE IF NOT EXISTS ${site}_topPerfTypesPast (
  theId      INT NOT NULL,    # type Id
  jobs       INT NOT NULL,
  files      INT NOT NULL,
  fSize      INT NOT NULL,    # [MB]
  users      INT NOT NULL,
  volume     INT NOT NULL,
  timePeriod CHAR(6)          # \"hour\", \"week\", \"month\", \"year\"
) MAX_ROWS=65535;

CREATE TABLE IF NOT EXISTS ${site}_users (
  id            SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  name          VARCHAR(24) NOT NULL
) MAX_ROWS=65535;

CREATE TABLE IF NOT EXISTS ${site}_hosts (
  id            SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  hostName      VARCHAR(64) NOT NULL
) MAX_ROWS=65535;

";
} ### end of site-specific tables ###

