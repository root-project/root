#!/bin/csh -f

#---------------------------------------------------------------------------
# NB: Make sure $HOME/private/authentication.xml contains YOUR credentials!
#---------------------------------------------------------------------------

# The path ${HOME}/private/authentication.xml is set in seal.opts.error
if ( ! -f ${HOME}/private/authentication.xml ) then
  echo "ERROR! File ${HOME}/private/authentication.xml not found!"
  exit 1
endif

#-----------------------------------------------------------------------------

if ( "$1" == "" || "$2" != "" ) then
  echo Usage: $0 dbId
  echo Example: $0 '"oracle://devdb10;schema=lcg_cool;dbname=COOL_REF"';
  #echo Example: $0 '"mysql://SERVER;schema=SCHEMA;dbname=DB"';
  exit 1
endif

unsetenv ORACLE_HOME
unsetenv SCRAM_HOME
setenv PATH /afs/cern.ch/sw/lcg/app/spi/scram:"${PATH}"

eval `scram runtime -csh`
setenv TNS_ADMIN ${LOCALRT}/src/RelationalCool/tests
setenv SEAL_CONFIGURATION_FILE ${LOCALRT}/src/RelationalCool/tests/seal.opts.error

#-----------------------------------------------------------------------------
# 1. Create reference schema

#setenv COOLTESTDB "$1"
#unitTest_RelationalCool_SchemaDump 

#-----------------------------------------------------------------------------
# 2. Dump reference schema

set theAuth = `coolAuthentication "$1" | & grep '==>'`
set theAuth = `echo "$theAuth ==>"`
#echo "theAuth   = $theAuth"

set theUrl = `echo "$theAuth" | awk '{str=$0; sep="==> urlHidePswd = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; print substr(str,0,ind);};'`
#echo "theUrl    = $theUrl"

if ( "$theUrl" == "" ) then
  echo "ERROR! Could not dump schema for COOL database "\""$1"\"
  echo "ERROR! Invalid COOL databaseId or missing authentication credentials"
  exit 1
endif

echo Dump schema for COOL database \""$theUrl"\"

set theTech = `echo "$theAuth" | awk '{str=$0; sep="==> technology = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; print substr(str,0,ind);};'`
set theHost = `echo "$theAuth" | awk '{str=$0; sep="==> server = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; print substr(str,0,ind);};'`
set theSchema = `echo "$theAuth" | awk '{str=$0; sep="==> schema = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; print substr(str,0,ind);};'`
set theUser = `echo "$theAuth" | awk '{str=$0; sep="==> user = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; print substr(str,0,ind);};'`
set thePswd = `echo "$theAuth" | awk '{str=$0; sep="==> password = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; print substr(str,0,ind);};'`
set theDbName = `echo "$theAuth" | awk '{str=$0; sep="==> dbname = "; ind=index(str,sep)+length(sep); str=substr(str,ind); sep=" ==>"; ind=index(str,sep)-1; print substr(str,0,ind);};'`

#echo "theTech   = $theTech"
#echo "theHost   = $theHost"
#echo "theSchema = $theSchema"
#echo "theUser   = $theUser"
#echo "thePswd   = $thePswd"
#echo "theDbName = $theDbName"
#exit

#--------
# Oracle
#--------
if ( "$theTech" == "oracle" ) then

  set theScript = `which $0`
  set theSqlDir = `dirname ${theScript}`/sql
  #echo "theScript = $theScript"
  #echo "theSqlDir = $theSqlDir"

  # HACK on Windows (limit to the size of file names)
  pushd $theSqlDir > /dev/null
  set theSqlDir = .
  #echo "theSqlDir = $theSqlDir"

  # No special treatment is needed for the user/password@host:port/service 
  # syntax: this is supported out-of-the-box by Oracle EasyConnect

  set theSchema = `echo $theSchema | awk '{print toupper($1)}'`
  set theDbName = `echo $theDbName | awk '{print toupper($1)}'`
  ###echo Dump Oracle schema

  #set theHtml = "OFF"
  set theHtml = "ON"

  if ( "$theHtml" == "ON" ) then
    set theOutFile = oracleRefSchema.html
    set theOut = `cd ..; pwd`/$theOutFile
    \rm -f ${theOut}
    echo Results will be in ${theOut}
    echo "<center><h1>" > ${theOut}
    echo Schema for COOL database \""$theUrl"\" >> ${theOut}
    echo "</h1></center><br>" >> ${theOut}
  else
    set theOut = /dev/stdout
  endif
  sqlplus -S -L -M "HTML ${theHtml}" ${theUser}/${thePswd}@${theHost} @${theSqlDir}/oracleSchemaDump.sql ${theSchema} ${theDbName} | grep -v "rows selected" >> ${theOut}

  set theTables = `sqlplus -S -L ${theUser}/${thePswd}@${theHost} @${theSqlDir}/oracleShowTables.sql ${theSchema} ${theDbName} | grep ${theDbName}_`
  #echo "theTables = $theTables"
  foreach aTable ( $theTables )
    #echo Describe Oracle table ${aTable}
    sqlplus -S -L -M "HTML ${theHtml}" ${theUser}/${thePswd}@${theHost} @${theSqlDir}/oracleDescTable.sql ${aTable} "${theSchema}.${aTable}" >> ${theOut}
  end

  # Copy the results in a directory visible from the Web
  if ( "${USER}" == "avalassi" ) then
    \cp ${theOut} ~/myLCG/www/tmp/${theOutFile}
    echo "Results can be browsed on"
    echo "  http://lcgapp.cern.ch/project/CondDB/tmp/${theOutFile}"
  endif

  # HACK on Windows 
  popd > /dev/null

#-------
# MySQL
#-------
else if ( "$theTech" == "mysql" ) then

  # A special treatment is needed for the host:port syntax: 
  # the port number must be parsed out and passed explicitly
  set thePort = `echo "$theHost" | awk '{str=$0; sep=":"; ind=index(str,sep); if (ind>0) print "-P" substr(str,ind+length(sep)); else print "";};'`
  set theHost = `echo "$theHost" | awk '{str=$0; sep=":"; ind=index(str,sep); if (ind>0) print substr(str,0,ind-length(sep)); else print str;};'`
  #echo "theHost   = $theHost"
  #echo "thePort   = $thePort"

  ###echo Dump MySQL schema
  #mysql -u${theUser} -p${thePswd} -h${theHost} ${thePort} ${theSchema} -e "drop table ${aTable};"
  echo NOT YET SUPPORTED

#--------
# SQLite
#--------
else if ( "$theTech" == "sqlite" ) then

  ###echo Dump SQLite schema
  echo NOT YET SUPPORTED

endif


