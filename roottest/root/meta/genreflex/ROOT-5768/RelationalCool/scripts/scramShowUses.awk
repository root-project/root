#----------------------------------------------------------------------------
#
# Title: "scramShowUses.awk" 
# Author: Andrea Valassi (Andrea.Valassi@cern.ch)
# Date: 26-MAR-2004
#
# Purpose: This AWK script is a 'CMT show uses' emulator for SCRAM.
#
# Usage: 'scram -debug b echo_INCLUDE | awk -f scramShowUses.awk'
#
# This command should be issued from a directory where one and only package
# would be built, i.e. a single package BuildFile is parsed by SCRAM.
# Error messages are printed if no package would be built, or if scram 
# would process many packages recursively ('About to process:'...).
# The dependency analysis is based from a (almost) dry run of scram
# in debug mode, letting scram parse BuildFiles recursively and feeding
# the resulting output to AWK. Write access to the local release top is 
# needed as 'scram b' will rebuild .mk file fragments in tmp.
#
#----------------------------------------------------------------------------
BEGIN { 
  status = 0;
  # Local top level project and package
  thisProjName = "";
  thisLocalRTop = ""; 
  thisReleaseTop = ""; 
  thisPkg = ""; 
  # External tools on which this package depends
  nExtTools = 0;
  delete extToolVersMap; #vers[name] for all selected tools (even not ref'ed)
  delete extToolNameVec; #name[1..n]
  extToolRef = "ext"     #my alternatives: 'ref' or 'ext' (keep 'use' for pkgs)
  # External projects on which this package depends
  nExtProjs = 0;
  delete extProjPathVec; #path[1..n]
  delete extProjNameMap; #name[path]
  # External packages on which this package depends
  nExtPkgs = 0;
  delete extPkgNameVec;  #name[1..n]
  delete extPkgProjMap;  #proj[name]
  # Current nesting level of dependencies
  currLevel = 0;
  currPkg = "";
  delete bfHashVec;  #hash[level]  
  delete bfHashMap;  #level[hash]  
  bfHashVec[0] = "";
  ###print "BEGIN: LEVEL=" currLevel;
}
#----------------------------------------------------------------------------
{
  if ( $1 == "About" && $2 == "to" && $3 == "process" ) {
    print $0;
    print "****************************************************************";
    print "* ERROR!                                                       *";
    print "* 1. Do not feed 'scram -debug b' to this script:              *";
    print "* feed the output of 'scram -debug b echo_INCLUDE' instead!    *";
    print "* 2. You cannot process multiple packages using this script:   *";
    print "* change directory so that 'scram b' sees a single BuildFile!  *";
    print "****************************************************************";
    status=1; exit status;
  }
}
#----------------------------------------------------------------------------
{
  if ( $1 == "Parse" && $2 == "Error" ) {
    print $0;
    getline;
    print $0;
    status=1; exit status;
  }
}
#----------------------------------------------------------------------------
{
  # Analysis of nesting level of dependencies proceeds via buildfile hash
  # Initially I thought of parsing the BuildFiles but that is error prone!
  if (index($1,">BuildSystem::BuildFile(BuildSystem::BuildFile=HASH(")==1) { 
    if ( bfHashVec[currLevel] != $1 ) {
      hash = $1;
      # First time a hash value is found - level 0
      if ( currLevel == 0 && bfHashVec[0] == "" ) {
	bfHashVec[0] = hash;
	bfHashMap[hash] = 0;
      }
      # Hash value of a previous build file - decrease level accordingly
      else if ( hash in bfHashMap ) {
        # Decrease until currLevel = bfHashMap[hash]
	while ( currLevel != bfHashMap[hash] ) {
	  delete bfHashMap[bfHashVec[currLevel]];
	  delete bfHashVec[currLevel];
	  currLevel--;
	}
      }
      # Hash value of a new build file - increase level by 1
      else {
	currLevel++;
	bfHashVec[currLevel] = hash;
	bfHashMap[hash] = currLevel;
	# Print the package names as dependencies for levels>=1
	# Leave two extra spaces for all levels>1
	space = "";
	for ( i=1; i<currLevel; i++ ) { space = space ". "; }	
	currProjName = extProjNameMap[extPkgProjMap[currPkg]];
	print "# "space"use "currProjName"::"currPkg;
	###print "# "space"use "currProjName"::"currPkg" (LEVEL="currLevel")";
      }
      ###print "LEVEL=" currLevel;
    }  
  }
}
#----------------------------------------------------------------------------
{
  # Local and global release top for the project of the top level package
  if ( $1 == "->Found" && $2 == "top" ) {
    if ( $3 == "" ) { print "ERROR! Null `top`!"; status=1; exit status; }
    # Read the project name from its Environment
    projEnv = $3 "/.SCRAM/Environment";
    projName = "";
    while ( projName == "" ) {
      get = getline record < projEnv;
      if ( get == -1 ) {
	print "ERROR! Environment not found for project " $3
	  status = 1; exit status;
      }
      else if ( get == 0 ) {
	break; #EOF
      }
      key = "SCRAM_PROJECTNAME";
      i = index( record, key );
      if ( i != 0 ) { projName = substr( record, i+length(key) ); }
      while( index(projName," ") == 1 || index(projName,"=") ) { 
	projName = substr( projName, 2 ); 
      }
    }
    # Is this the local or remote release top?
    if ( thisLocalRTop == "" ) { 
      thisLocalRTop = $3;
      thisProjName = projName;
      print "###############################################################";
      print "# *** PROJECT_NAME:       " projName;
      print "# *** PROJECT_RTOP_LOCAL: " thisLocalRTop;
      extProjNameMap[thisLocalRTop] = thisProjName; 
    } 
    else if ( thisReleaseTop == "" ) { 
      if ( projName != thisProjName ) {
	print "ERROR! Project name mismatch!";
	status = 1; exit status;
      }
      thisReleaseTop = $3; 
      print "# *** PROJECT_RTOP:       " thisReleaseTop;
      extProjNameMap[thisReleaseTop] = thisProjName; 
    }
    else { 
      print "ERROR! Too many release tops!"; 
      status = 1; exit status; 
    }    
  }
}
#----------------------------------------------------------------------------
{
  # Top level package
  if ( $2 == "ParseBuildFile:" ) {
    pkg = substr( $4, 1, index($4,"/BuildFile")-1 );
    # Project and release area for this package 
    if ( index(pkg,thisLocalRTop) == 1 ) { 
      pkg = substr( pkg, length(thisLocalRTop)+1 ); 
    }
    else {
      print "PANIC! ParseBuildFile invoked outside local release top? " $4; 
      status = 1; exit status; 
    }
    # Package name
    while ( index(pkg,"/") == 1 ) { pkg = substr( pkg, 2 ); }
    while ( index(pkg,"./") == 1 ) { pkg = substr( pkg, 3 ); }
    while ( index(pkg,"/") == 1 ) { pkg = substr( pkg, 2 ); }
    while ( index(pkg,"./") == 1 ) { pkg = substr( pkg, 3 ); }
    if ( index(pkg,"src/") == 1 ) { pkg = substr( pkg, 5 ); }
    while ( index(pkg,"/") == 1 ) { pkg = substr( pkg, 2 ); }
    if ( pkg == "" ) { 
      print "ERROR! Null package! BuildFile: " $4; 
      status = 1; exit status; 
    }
    # Two BuildFile's are expected here: from config and from the top level pkg
    # (AV March 2007: $LOCALRT/config can now be in $LOCALRT/src/config/scram) 
    # NB The same HASH value defines the environment for both BuildFile's
    if ( thisPkg == "" ) {
      if ( pkg != "config" && pkg != "config/scram" ) { 
	print "PANIC! Unknown package (expecting config or config/scram): " $4;
	status = 1; exit status; 
      }
      thisPkg = pkg; 
      print "###############################################################";
      print "# *** PROJECT_CONFIG:     " thisPkg;
      #print "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
      currPkg = pkg;
    } 
    else if ( thisPkg == "config" ) {
      thisPkg = pkg; 
      print "###############################################################";
      print "# *** PACKAGE_NAME:       " thisProjName"::"thisPkg;
      #print "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
      currPkg = pkg;
    }
    else { 
      ###print "PANIC! Unknown package (expecting external ref): " $4; 
      ###print "PANIC! thisPkg = " thisPkg; 
      ###status = 1; exit status; 
      thisPkg = pkg; 
      print "###############################################################";
      print "# *** SUBPACKAGE_NAME:    " thisProjName"::"thisPkg;
      #print "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
      currPkg = pkg;
    }
  } 
}
#----------------------------------------------------------------------------
{
  # Dependency on an external tool (project-wide)
  if ( $3 == "being" && $4 == "Initialised" ) {
    toolName = substr($1,3);
    toolVers = $2;
    print "# select " toolName " " toolVers;
    if ( ! ( toolName in extToolVersMap ) ) {
      extToolVersMap[toolName] = toolVers;      
    }
    else {
      if ( extToolVersMap[toolName] != toolVers ) {
	print "ERROR! Tool " toolName " " \
	  extToolVersMap[toolName] " already selected!";
	status=1; exit status;
      }      
      print "WARNING! Tool " toolName " " toolVers " already selected!";
    }    
  }
}
#----------------------------------------------------------------------------
{
  # Dependency on an external tool (package-specific)
  # !!! This requires a modified version of scram !!!
  if ( $2 == "External_StartTag:" && $3 == "select" && $4 == "Tool" ) {
    toolName = $5;
    # Print the tool names as dependencies for levels>=1
    # Leave two extra spaces for all levels>1
    space = "";
    for ( i=1; i<currLevel; i++ ) { space = space ". "; }	
    # External tool is ALWAYS one level below the current BuildFile!
    if ( currLevel>=1 ) space = space ". ";
    print "# " space extToolRef " " toolName;
    ###print "# " space extToolRef " " toolName " (LEVEL=" currLevel+1 ")";
    # Update tool dependency vector
    found = 0;
    for ( i=1; i<=nExtTools; i++ ) {
      if ( toolName == extToolNameVec[i] ) found = 1;      
    }
    if ( found == 0 ) {
      nExtTools++;
      extToolNameVec[nExtTools] = toolName;
    }    
  }
}
#----------------------------------------------------------------------------
{
  # Dependency on an external SCRAM project
  if ( $2 == "_pushremoteproject:" ) {
    # Handle verbose output from my enhanced test version of SCRAM 
    # (where an external project can have both a localtop and a releasetop)
    # Expect a colon-separated list of paths from _pushremoteproject: this is 
    # 'backward'-compatible to SCRAM V0_20_0 as it can handle a single path too
    npaths = split ( $4, paths, ":" );
    delete projs;
    proj = "";
    projEnv = "";
    for ( i=1; i<=npaths; i++ ) {
      if ( paths[i] == "" ) {
	projs[i] = "";
      }
      else {
	aProj = paths[i];
	while ( substr(aProj,length(aProj)) == "/" ) { 
	  aProj = substr(aProj,1,length(aProj)-1);
	}
	if ( substr(aProj,length(aProj)-3) == "/src" ) { 
	  aProj = substr(aProj,1,length(aProj)-4);
	}
	while ( substr(aProj,length(aProj)) == "/" ) { 
	  aProj = substr(aProj,1,length(aProj)-1);
	}
	projs[i] = aProj;
	aProjEnv = aProj "/.SCRAM/Environment";
	if ( system("ls " aProjEnv " >& /dev/null") == 0 ) {
	  ###print "FOUND " aProjEnv;
	  if ( projEnv == "" ) {
	    projEnv = aProjEnv;
	    proj = aProj;
	  }
	} 
	else {
	  ###print "NOT FOUND " aProjEnv;
	  if ( projEnv == "" ) {
	    proj = aProj;
	  }
	};
      }
    }
    # Read the project name from its Environment
    projName = "";
    while ( projName == "" ) {
      if ( projEnv == "" ) {
	print "ERROR! Environment not found for project " proj;
	status = 1; exit status;
      }
      getStatus = getline record < projEnv;
      if ( getStatus == -1 ) {
	print "ERROR! Environment not found for project " proj;
	status = 1; exit status;
      }
      else if ( getStatus == 0 ) {
	break; #EOF
      }
      key = "SCRAM_PROJECTNAME";
      i = index( record, key );
      if ( i != 0 ) { projName = substr( record, i+length(key) ); }
      while( index(projName," ") == 1 || index(projName,"=") ) { 
	projName = substr( projName, 2 ); 
      }
    }
    # Is this a new project?
    for ( i=1; i<=npaths; i++ ) {
      aProj = projs[i];
      if ( aProj != "" && ( ! ( aProj in extProjNameMap ) ) ) { 
	if ( aProj == thisLocalRTop ) {
	  print "ERROR! Circular dependency on this project (localRT)?";
	  status = 1; exit status;
	}
	if ( aProj == thisReleaseTop ) {
	  print "ERROR! Circular dependency on this project (releaseTop)?";
	  status = 1; exit status;
	}    
        # Add the project to the list
	nExtProjs++;
	extProjPathVec[nExtProjs] = aProj;
	extProjNameMap[aProj] = projName; 
	###print "# *** EXTERNAL_PROJECT("nExtProjs"): "projName" (" aProj ")";
      }
    }
  }
}
#----------------------------------------------------------------------------
{
  # Dependency on an external SCRAM package
  if ( $2 == "ParseBuildFile_Export:" ) {
    pkg = substr( $4, 1, index($4,"/BuildFile")-1 );
    # Project and release area for this package 
    pkgProj = "";
    if ( index(pkg,thisLocalRTop) == 1 ) { 
      pkg = substr( pkg, length(thisLocalRTop)+1 ); 
      pkgProj = thisLocalRTop;
    }
    else if ( thisReleaseTop != "" && index(pkg,thisReleaseTop) == 1 ) { 
      pkg = substr( pkg, length(thisReleaseTop)+1 ); 
      pkgProj = thisReleaseTop;
    }
    else {
      for ( proj in extProjNameMap ) {
	if ( index(pkg,proj) == 1 ) { 
	  pkg = substr( pkg, length(proj)+1 ); 
	  pkgProj = proj;
	}
      }
    }
    if ( pkgProj == "" ) { 
      print "ERROR! No project associated to package: " $4; 
      status = 1; exit status; 
    }
    # Package name
    while ( index(pkg,"/") == 1 ) { pkg = substr( pkg, 2 ); }
    if ( index(pkg,"src/") == 1 ) { pkg = substr( pkg, 5 ); }
    while ( index(pkg,"/") == 1 ) { pkg = substr( pkg, 2 ); }
    if ( pkg == "" ) { 
      print "ERROR! Null package! BuildFile: " $4; 
      status = 1; exit status; 
    }
    ####### Analyse the dependencies of this package
    ######system( "cat " $4 "| grep -i use" ); print "";
    # Add the package to the list if this is a new package
    if ( ! ( pkg in extPkgProjMap ) ) {
      nExtPkgs ++;
      extPkgNameVec[nExtPkgs] = pkg;      
      extPkgProjMap[pkg] = pkgProj;
    }
    # Pass the package name to be printed out with its level of nesting
    currPkg = pkg;
    ###print "# " pkg; #THIS LINE IS USEFUL FOR DEBUGGING      
  } 
}
#----------------------------------------------------------------------------
END {
  if ( status == 0 ) {    
    if ( thisPkg != "" && thisPkg != "config" ) {
      #print "#";
      print "###############################################################";
      print "#";
      print "#--------------------";
      print "# TOOL DEPENDENCIES:";  
      print "#--------------------";
      if ( nExtTools > 0 ) {	
	for ( i=1; i<=nExtTools; i++ ) {
	  toolName = extToolNameVec[i];
	  toolVers = extToolVersMap[toolName];
	  print extToolRef " " toolName " " toolVers;
	}  
      }      
      else {
	print "*** NONE ***";
      }      
      print "#";
      print "#-----------------------";
      print "# PACKAGE DEPENDENCIES:";  
      print "#-----------------------";
      if ( nExtPkgs > 0 ) {	
	for ( i=1; i<=nExtPkgs; i++ ) {
	  pkg = extPkgNameVec[i];
	  pkgProj = extPkgProjMap[pkg];
	  pkgProjName = extProjNameMap[pkgProj];
          ###print "use " pkgProjName "::" pkg " (" pkgProj ")"; #CMT-like
	  print "use " pkgProjName "::" pkg;
	  print "    (" pkgProj ")";
	}  
      }  
      else {
	print "*** NONE ***";
      }      
      ###print "END: LEVEL=" currLevel;
    }
    else {
      print "****************************************************************";
      print "* ERROR!                                                       *";
      print "* 1. You cannot process multiple packages using this script:   *";
      print "* change directory so that 'scram b' sees a single BuildFile!  *";
      print "* 2. Or maybe, 'scram b' does not see ANY BuildFile from here? *";
      print "****************************************************************";
    }  
  }  
}
#----------------------------------------------------------------------------
{
  # Print out debug messages from my modifcations to SCRAM
  if ( $1 == "__scramShowUses" ) print $0;
} 
#----------------------------------------------------------------------------

