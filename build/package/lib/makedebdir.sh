#!/bin/sh -e 
#
# $Id$
#
# Make the debian packaging directory 
#
. build/package/lib/common.sh debian 

### echo %%% Make the directory 
mkdir -p ${tgtdir} 

### echo %%% Copy the README file to the directory 
cp ${cmndir}/README ${tgtdir}/README.Debian 

### echo %%% Copy task-root readme 
cp ${debdir}/task-root.README.Debian ${tgtdir}

### echo %%% Copy root-bin menu file
cp ${debdir}/root-bin.menu ${tgtdir}

### echo %%% Copy watch file 
cp ${debdir}/watch ${tgtdir}

### echo %%% make the changelog 
${libdir}/makedebchangelog.sh 

### echo %%% make the toplevel copyright file 
${libdir}/makedebcopyright.sh 

### echo %%% Copy the skeleton rules file to ${mndir}/rules.tmp 
if [ ! -f ${debdir}/rules.in ] ; then 
    echo "$0: I cannot find the ESSENTIAL file ${debdir}/rules.in"
    echo "Giving up. Something is very screwy"
    exit 10
fi
cp ${debdir}/rules.in ${cmndir}/rules.tmp 

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
    "gl")     bd="${bd}, libgl-dev" ;; 
    "mysql")  bd="${bd}, libmysqlclient6-dev (>= 3.22.30)" ;;
    "pythia") bd="${bd}, libpythia-dev" ;; 
    "ttf")    bd="${bd}, freetype2-dev" ;; 
    *) ;;
    esac
done

### echo %%% Now insert the line 
sed "s|@build-depends@|${bd}|" < ${debdir}/head.control.in \
    > ${tgtdir}/control
echo "" >> ${tgtdir}/control

### echo %%% Make the sub-package stuff
for i in ${pkgs} ; do 
    echo "Processing for package $i ... "
    ### echo %%% First append to the control file 
    ${libdir}/makedebcontrol.sh $i

    ### echo %%% Append to the shlibs.local file 
    ${libdir}/makedebshlocal.sh $i

    ### echo %%% Then make the file lists
    ${libdir}/makedebfiles.sh $i      
    ${libdir}/makedebconffiles.sh $i  
    ${libdir}/makedebdocs.sh $i       
    ${libdir}/makedebexamples.sh $i   

    ### echo %%% Make copyright file 
    ${libdir}/makedebcopyright.sh $i 

    ### echo %%% make the kinds of scripts 
    for j in $lvls ; do 
	${libdir}/makedebscr.sh $i $j 
    done 

    ### echo %%% Update the rules file 
    if [ "x$i" != "xtask" ] ; then 
	${libdir}/makedebrules.sh $i 
    fi
done 

### echo %%% Insert the configuration command 
### echo %%% first split file
csplit -q -f ${cmndir}/tmp. -k  ${cmndir}/rules.tmp "/@configure@/"

### echo %%% Cat the first part 
sed -e  '/@pkg@/d' \
    < ${cmndir}/tmp.00 > ${tgtdir}/rules

### echo %%% now the configuration command 
${libdir}/makeconfigure.sh debian >> ${tgtdir}/rules

### echo %%% and finally the last part 
sed -e '/@configure@/d' \
    < ${cmndir}/tmp.01 >> ${tgtdir}/rules

### echo %%% clean up 
rm -f ${cmndir}/tmp.00 ${cmndir}/tmp.01 ${cmndir}/rules.tmp

#
# $Log$
#
