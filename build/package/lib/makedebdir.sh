#!/bin/sh
#
# $Id: makedebdir.sh,v 1.5 2002/05/27 16:27:56 rdm Exp $
#
# Make the debian packaging directory 
#
### echo %%% Various needed variables 
base=root
tgtdir="debian"
cmndir=build/package/common
libdir=build/package/lib
debdir=build/package/debian
vrsfil=build/version_number

### echo %%% Installation directories 
prefix=usr
etcdir=etc/root
docdir=${prefix}/share/doc/root-doc 

### echo %%% Packages ordered by preference
pkgs="root-daemon root-ttf root-zebra root-gl root-mysql root-pgsql root-table root-shift root-cint root-bin libroot-dev libroot"
pkgs=`./configure linuxdeb --pkglist --enable-soversion --enable-table --enable-thread --enable-shared | sed -n 's,packages: ,,p'`
### echo %%% Package list is: $pkgs
lvls="preinst postinst prerm postrm"


### echo %%% ROOT version 
major=`sed 's|\(.*\)\..*/.*|\1|' < ${vrsfil}`
minor=`sed 's|.*\.\(.*\)/.*|\1|' < ${vrsfil}`
revis=`sed 's|.*\..*/\(.*\)|\1|' < ${vrsfil}`
versi="${major}.${minor}.${revis}"

### echo %%% Make the directory 
mkdir -p ${tgtdir} 

### echo %%% Copy the README file to the directory 
sed -e "s,@prefix@,/${prefix},g" \
    -e "s,@etcdir@,/${etcdir},g" \
    < ${cmndir}/README > ${tgtdir}/root-doc.README.Debian 

### echo %%% Copy root-bin menu file
sed -e "s,@prefix@,/${prefix},g" \
    -e "s,@etcdir@,/${etcdir},g" \
    < ${debdir}/root-bin.menu.in > ${tgtdir}/root-bin.menu

### echo %%% Copy watch file 
cp ${debdir}/watch ${tgtdir}

### echo %%% Copy mime file 
cp ${debdir}/root-bin.mime ${tgtdir}

### echo %%% make the changelog 
${libdir}/makedebchangelog.sh $tgtdir $debdir $versi

### echo %%% make the toplevel copyright file 
${libdir}/makedebcopyright.sh $tgtdir $debdir $cmndir 

### echo %%% Copy the header of the control file to debian/control 
if [ ! -f ${debdir}/head.control.in ] ; then 
    echo "$0: Couldn't find ${debdir}/head.control.in - very bad" 
    echo "Giving up. Something is very screwy"
    exit 10
fi

### echoo %%% But first we have to insert the build dependencies
bd=""
for i in $pkgs; do 
    case $i in 
    # Since we always have libxpm4-dev first, we can add a comma freely. 
    # That is, we don't have to worry if the entry is the first in the
    # list, because it never is. Thank god for that. 
    root-gl)     bd="${bd}, libgl-dev" ;; 
    root-gliv)   bd="${bd}, inventor-dev, libgl-dev, lesstif-dev" ;;
    root-mysql)  bd="${bd}, libmysqlclient-dev" ;;
    root-pgsql)  bd="${bd}, postgresql-dev" ;;
    root-pythia) bd="${bd}, libpythia-dev" ;; 
    root-ttf)    bd="${bd}, freetype2-dev" ;; 
    *) ;;
    esac
done

### echo %%% Now insert the line 
sed "s/@build-depends@/${bd},/" < ${debdir}/head.control.in \
    > ${tgtdir}/control
echo "" >> ${tgtdir}/control

### echo %%% Make the sub-package stuff
for i in ${pkgs} ; do 
    echo "Processing for package $i ... "
    ### echo %%% First append to the control file 
    ${libdir}/makedebcontrol.sh $tgtdir $debdir $cmndir $i

    ### echo %%% Append to the shlibs.local file 
    ${libdir}/makedebshlocal.sh $tgtdir $debdir $cmndir $versi $major $minor $i

    ### echo %%% Then make the file lists
    ${libdir}/makedebfiles.sh $tgtdir $cmndir $prefix $etcdir $docdir $i      
    ${libdir}/makedebconffiles.sh $tgtdir $cmndir $etcdir $i      
    ${libdir}/makedebdocs.sh $tgtdir $cmndir $prefix $etcdir $docdir $i      
    ${libdir}/makedebexamples.sh $tgtdir $cmndir $prefix $etcdir $docdir $i

    ### echo %%% Make copyright file 
    ${libdir}/makedebcopyright.sh $tgtdir $debdir $cmndir $i 

    ### echo %%% make the kinds of scripts 
    for j in $lvls ; do 
        ${libdir}/makedebscr.sh $tgtdir $debdir $cmndir \
            $prefix $etcdir $docdir $i $j 
    done 

    ### echo %%% Update the rules file 
    # if [ "x$i" != "xtask" ] ; then 
    # 	${libdir}/makedebrules.sh $i 
    # fi
done 

### echo %%% Copy the skeleton rules file to 
if [ ! -f ${debdir}/rules.in ] ; then 
    echo "$0: I cannot find the ESSENTIAL file ${debdir}/rules.in"
    echo "Giving up. Something is very screwy"
    exit 10
fi
### echo %%% Make the rules file 
sed -e "s,@prefix@,/${prefix},g" \
    -e "s,@etcdir@,/${etcdir},g" \
    -e "s,@docdir@,/${docdir},g" \
    -e "s,@pkgs@,${pkgs},g"      \
    < ${debdir}/rules.in > ${tgtdir}/rules
chmod 755 ${tgtdir}/rules

#
# $Log: makedebdir.sh,v $
# Revision 1.5  2002/05/27 16:27:56  rdm
# rename libStar to libTable.
#
# Revision 1.4  2002/05/14 15:45:28  rdm
# several Debian related packaging and build changes. By Christian Holm.
#
# Revision 1.3  2002/01/22 10:53:28  rdm
# port to Debian distribution of GNU/Hurd by Christian Holm.
#
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
