#!/bin/sh -e 
#
# $Id: makerpmfiles.sh,v 1.2 2002/01/20 14:23:52 rdm Exp $
#
# Make filelists files for all packages. 
#  
# We have to be a bit tricky here, since rpm will happily copy the
# same file into many sub-packages, so we have to do wildcard
# expansio, substract already installed files and so on. We employ a
# trick here that's similar to what is done in dh_movesfiles from the
# debhelper package on Debian GNU/Linux. First we pack the files into 
# tar archives, IN THE ORDER GIVEN by variable $pkgs from common file,
# taking care to remove the old files, then we list the contents of
# these tar files and make that our file list, and finally we unpack
# the tar archives again, so that rpm can find them. 
# 

### echo %%% save the build root in a variable 
blddir=$1  ; shift 
cmndir=$1  ; shift 
tgtdir=$1  ; shift
prefix=$1  ; shift 
etcdir=$1  ; shift 
docdir=$1  ; shift
pkgs=$* 

### echo %%% remove leading / from $prefix and $etcdir
prefix=`echo $prefix | sed 's,^/,,'`
etcdir=`echo $etcdir | sed 's,^/,,'`
docdir=`echo $docdir | sed 's,^/,,'`

### echo %%% Make the directory 
mkdir -p ${tgtdir} 

### echo %%% The types of files we looking at. 
types="files conffiles docs examples shlibs" 

### echo %%% loop over the defined packages in specific order
for p in $pkgs ; do 
    ### echo %%% skip the task package
    if test "x$p" = "xtask-root" ; then 
	continue
    fi 

    ### echo %%% make sure we have a fresh file 
    rm -f ${cmndir}/tmp 

    for j in $types ; do 
	### echo %%% if ${cmndir}/${p}.${j} exists, use it
	if [ -f ${cmndir}/${p}.${j} ] ; then 
	    cat ${cmndir}/${p}.${j} >> ${cmndir}/tmp
	fi 
    done
    if [ ! -s ${cmndir}/tmp ] ; then 
	continue
    fi

    ### echo %%% do a primitive wildcard expansion on temp file list
    entries=`cat ${cmndir}/tmp`

    ### echo %%% save the original directory 
    savdir=`pwd`

    ### echo %%% make sure we have a fresh file 
    rm -f ${blddir}/tmp.${j}

    ### echo %%% cd into the subdirectory first 
    cd ${blddir}

    ### echo %%% make sure we have a clean file 
    rm -f tmp.${p} 

    ### echo %%% let a for loop do wildcard expansion in `pwd` on $entries
    for j in ${entries} ; do 
	foo=`echo "$j" | sed -e "s,@prefix@,${prefix}," -e "s,@etcdir@,${etcdir}," -e "s,@docdir@,${docdir},"`
	echo $foo | tr ' ' '\n' >> tmp.${p}
    done

    ### echo %%% move the files to a temporary tar archive 
    echo -n "Creating tmp archive for $p ... "

    tar --create \
	--remove-files \
	--files-from=tmp.${p} \
	--file=${p}.tar 
    ### echo %%% clean up
    rm -f tmp.${p}

    ### echo %%% go back to top-level dir 
    cd ${savdir} 
    echo "done"

    ### echo %%% clean up
    rm -f ${cmndir}/tmp
done

### echo %%% Loop over the packages, and make the filelists
for p in $pkgs ; do 
    ### echo %%% skip the task package
    if test "x$p" = "xtask-root" ; then 
	continue
    fi 
    
    ### echo %%% if tar archive does not exit, continue to next 
    if test ! -f ${blddir}/${p}.tar ; then
	echo "No such file: ${blddir}/${p}.tar - strange"
	continue
    fi

    ### echo %%% create the file list 
    echo -n "Creating package list for $p ... "

    ### echo %%% first set the default attributes
    echo "%defattr(-,root,root)" > ${tgtdir}/${p}.files

    ### echo %%% and make @prefix@/share/doc/root a documentation dir
    echo "%docdir /$docdir" >> ${tgtdir}/${p}.files

    ### echo %%% List the contents of the tar file, ignoring pure 
    ### echo %%% directories and marking files in /etc as
    ### echo %%% confuguration files and files in @prefix@/share/doc as
    ### echo %%% documentation files
    tar --list --file=${blddir}/${p}.tar | \
	sed -e 's|^\(.*man1/.*\.1\)|/\1*|' \
	    -e 's|^\(.*\)|/\1|' \
	    -e "s|^/${etcdir}|%config /${etcdir}|" \
	    -e '\|.*/$|d' | \
	sort -u >> \
	${tgtdir}/${p}.files

    ### echo %%% unpack the tar archive so that rpm may find the files
    ### echo %%% again 	
    echo -n "extracting ... " 
    tar --extract --directory=${blddir} --file=${blddir}/${p}.tar 

    ### echo %%% clean up 
    rm -f ${blddir}/${p}.tar
    echo "done"
done 

#
# $Log: makerpmfiles.sh,v $
# Revision 1.2  2002/01/20 14:23:52  rdm
# Mega patch by Christian Holm concerning the configure, build and
# Debian and RedHat packaging scripts. The configure script has been
# rationalized (introduction of two shell functions to find package
# headers and libraries). Extensive update of the INSTALL writeup,
# including description of all new packages (SapDB, PgSql, etc.).
# More options to the root-config script. Man page for memprobe.
# Big overhaul of the Debian and RedHat packaging scripts, supporting
# the new libraries.
#
# Revision 1.1  2001/04/23 14:11:47  rdm
# part of the debian and redhat build system.
#
#

