#!/bin/tcsh -f
# 2013.06.05 Use /bin/tcsh instead of /bin/csh for better sqlplus return code!
# You should really get rid of this script or move it to bash...

if ( "$1" != "-html" ) then
  set theHtml = OFF
  set theArg1 = "$1"
  set theArg2 = "$2"
  set theArg3 = "$3"
else
  set theHtml = ON
  set theArg1 = "$2"
  set theArg2 = "$3"
  set theArg3 = "$4"
endif

if ( "$theArg2" == "-e" || "$theArg2" == "-eb" ) then
  if ( "$theArg2" == "-e" ) then 
    set theCmd = ON
  else
    set theCmd = BATCH
  endif
  set theConnStr = "$theArg1"
  set theSqlFile = "$theArg3"
else
  set theCmd = OFF
  set theConnStr = "$theArg1"
  set theSqlFile = "$theArg2"
endif

#echo "theHtml    = $theHtml"
#echo "theCmd     = $theCmd"
#echo "theConnStr = $theConnStr"
#echo "theSqlFile = $theSqlFile"

if ( "$theConnStr" == "" || "$theSqlFile" == "" ) then
  echo "Usage: $0 '[-html]' dbId { file.sql | -e 'command' | -eb 'command' }"
  echo Example: $0 '"oracle://SERVER;schema=SCHEMA;dbname=DB"' file.sql
  echo Example: $0 -html '"oracle://SERVER;schema=SCHEMA;dbname=DB"' file.sql
  echo Example: $0 '"mysql://SERVER;schema=SCHEMA;dbname=DB"' -e 'cmd'
  echo Example: $0 '"mysql://SERVER;schema=SCHEMA;dbname=DB"' -eb 'cmd'
  echo Example: $0 -html '"mysql://SERVER;schema=SCHEMA;dbname=DB"' -eb 'cmd'
  exit 1
endif

set theAuth = `coolAuthentication "$theConnStr" | & grep '==>'`
set theAuth = `echo "$theAuth ==>"`
#echo "theAuth   = $theAuth"

set theUrl = `echo "$theAuth" | awk '{str=$0; sep="==> urlHidePswd = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; str=substr(str,1,ind); l=length(str); cr=substr(str,l); if (cr=="\r") print substr(str,1,l-1); else print str;}'`
#echo "theUrl    = $theUrl"

if ( "$theUrl" == "" ) then
  echo "ERROR! Could not execute SQL script $theSqlFile against COOL database "\""$theConnStr"\"
  echo "ERROR! Invalid COOL databaseId or missing authentication credentials"
  exit 1
endif

if ( "${theHtml}" != ON && "${theCmd}" != BATCH ) then
  echo Execute SQL script \""$theSqlFile"\" against COOL database \""$theUrl"\"
endif

set theTech = `echo "$theAuth" | awk '{str=$0; sep="==> technology = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; str=substr(str,1,ind); l=length(str); cr=substr(str,l); if (cr=="\r") print substr(str,1,l-1); else print str;}'`
set theHost = `echo "$theAuth" | awk '{str=$0; sep="==> server = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; str=substr(str,1,ind); l=length(str); cr=substr(str,l); if (cr=="\r") print substr(str,1,l-1); else print str;}'`
set theSchema = `echo "$theAuth" | awk '{str=$0; sep="==> schema = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; str=substr(str,1,ind); l=length(str); cr=substr(str,l); if (cr=="\r") print substr(str,1,l-1); else print str;}'`
set theUser = `echo "$theAuth" | awk '{str=$0; sep="==> user = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; str=substr(str,1,ind); l=length(str); cr=substr(str,l); if (cr=="\r") print substr(str,1,l-1); else print str;}'`
set thePswd = `echo "$theAuth" | awk '{str=$0; sep="==> password = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; str=substr(str,1,ind); l=length(str); cr=substr(str,l); if (cr=="\r") print substr(str,1,l-1); else print str;}'`
set theDbName = `echo "$theAuth" | awk '{str=$0; sep="==> dbName = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; str=substr(str,1,ind); l=length(str); cr=substr(str,l); if (cr=="\r") print substr(str,1,l-1); else print str;}'`
#echo "theTech   = $theTech"
#echo "theHost   = $theHost"
#echo "theSchema = $theSchema"
#echo "theUser   = $theUser"
#echo "thePswd   = $thePswd"
#echo "theDbName = $theDbName"
#exit 1

# 2013.06.05 - support for Kerberos proxy authentication
if ( "$thePswd" == "" ) then
  set theProxy="[${theSchema}]"
else
  set theProxy="${theUser}"
endif

# 2005.06.22 
# Here was a CHARMING bug (introduced by upgrading version of cygwin?).
# For theTech="oracle", on Windows this was actually equal to "oracle\r".
# All of the following returned (7,7) on Windows and (6,0) on SLC.
# echo a | awk -v s=$theTech '{print length(s), index(s,"\15")}'
# echo a | awk -v s=$theTech '{print length(s), index(s,"\015")}'
# echo a | awk -v s=$theTech '{print length(s), index(s,"\r")}'
# As a consequence ( "$theTech" == "oracle" ) was false on Windows.
# Now fixed by explicitly removing "\r" in awk above. 
# For Oracle both Windows and SLC return (6,0).

if ( "$theCmd" == "OFF" ) then
  if ( ! -e "$theSqlFile" ) then
    echo "ERROR! File not found: '$theSqlFile'"
    exit 1
  endif
  set theSqlScript = `basename $theSqlFile`
  set theSqlDir = `dirname $theSqlFile`
  #echo "theSqlScript = $theSqlScript"
  #echo "theSqlDir = $theSqlDir"
else
  set theSqlDir = .  
endif

# HACK on Windows (limit to the size of file names)
pushd $theSqlDir > /dev/null
set theSqlDir = .
#echo "theSqlDir = $theSqlDir"

# Return code
set theStatus=0

#--------
# Oracle
#--------
if ( "$theTech" == "oracle" ) then

  set theSilent="-S"
  ###set theSilent=""

  if ( "${theHtml}" == ON ) then

    if ( "$theCmd" == "OFF" ) then
      # This format requires "quit;" at the end of the script...
      #sqlplus ${theSilent} -L -M "HTML ON" "${theProxy}/${thePswd}@${theHost}" @${theSqlDir}/${theSqlScript}
      # This format does NOT require "quit;" at the end of the script!
      cat ${theSqlDir}/${theSqlScript} | sqlplus ${theSilent} -L -M "HTML ON" "${theProxy}/${thePswd}@${theHost}"
      set theStatus=${status}
    else
      # This format does NOT require "quit;" at the end of the script!
      # Enclose it in quotes to be able to 'select * from table'...!
      echo "whenever sqlerror exit 1\n${theSqlFile}" | sqlplus ${theSilent} -L -M "HTML ON" "${theProxy}/${thePswd}@${theHost}"
      set theStatus=${status}
    endif

  else

    if ( "$theCmd" == "OFF" ) then
      # This format requires "quit;" at the end of the script...
      #sqlplus ${theSilent} -L "${theProxy}/${thePswd}@${theHost}" @${theSqlDir}/${theSqlScript}
      # This format does NOT requires "quit;" at the end of the script!
      cat ${theSqlDir}/${theSqlScript} | sqlplus ${theSilent} -L "${theProxy}/${thePswd}@${theHost}"
      set theStatus=${status}
    else
      # This format does NOT require "quit;" at the end of the script!
      # Enclose it in quotes to be able to 'select * from table'...!
      echo "whenever sqlerror exit 1\n${theSqlFile}" | sqlplus ${theSilent} -L "${theProxy}/${thePswd}@${theHost}"
      set theStatus=${status}
    endif

  endif

#-------
# MySQL
#-------
else if ( "$theTech" == "mysql" ) then

  if ( "${theHtml}" == ON ) then
    set theHtml = "--html"
  else
    set theHtml = ""
  endif

  # A special treatment is needed for the host:port syntax: 
  # the port number must be parsed out and passed explicitly
  set thePort = `echo "$theHost" | awk '{str=$0; sep=":"; ind=index(str,sep); if (ind>0) print "-P" substr(str,ind+length(sep)); else print "";};'`
  set theHost = `echo "$theHost" | awk '{str=$0; sep=":"; ind=index(str,sep); if (ind>0) print substr(str,0,ind-length(sep)); else print str;};'`
  #echo "theHost   = $theHost"
  #echo "thePort   = $thePort"

  if ( "$theCmd" == "OFF" ) then
    mysql ${theHtml} -u${theUser} -p${thePswd} -h${theHost} ${thePort} ${theSchema} < ${theSqlDir}/${theSqlScript}
  else
    if ( "$theCmd" == "BATCH" ) then
      mysql ${theHtml} -u${theUser} -p${thePswd} -h${theHost} ${thePort} ${theSchema} -B -N -e "${theSqlFile}"
    else
      mysql ${theHtml} -u${theUser} -p${thePswd} -h${theHost} ${thePort} ${theSchema} -e "${theSqlFile}"
    endif
  endif

#--------
# SQLite
#--------
else if ( "$theTech" == "sqlite" ) then

  if ( "${theHtml}" == ON ) then
    set theHtml = "-html"
  else
    set theHtml = ""
  endif

  if ( "$theCmd" == "OFF" ) then
    sqlite3 ${theHtml} ${theSchema} ".read ${theSqlDir}/${theSqlScript}"
  else
    sqlite3 ${theHtml} ${theSchema} "${theSqlFile}"
  endif

endif

# HACK on Windows 
popd > /dev/null

exit ${theStatus}
