#!/bin/sh
#
# $Id: makedebdir.sh,v 1.13 2006/02/28 16:38:23 rdm Exp $
#
# Make the debian packaging directory 
#

### echo %%% possibly update the changelog file
root_vers=`cat build/version_number | tr '/' '.'` 
last_vers=`head -n 1 build/package/debian/changelog | sed 's/root (\(.*\)).*/\1/'`
root_lvers=`echo $root_vers | awk 'BEGIN {FS="."} {printf "%d", (($1 * 1000) + $2) * 1000 + $3}'`
last_lvers=`echo $last_vers | awk 'BEGIN {FS="."} {printf "%d", (($1 * 1000) + $2) * 1000 + $3}'`
root_sovers=`cat build/version_number | sed 's,/.*,,'` 
if test $root_lvers -gt $last_lvers ; then 
    dch -v ${root_vers}-1 -c build/package/debian/changelog "New upstream version"
fi

### echo %%% Make the directory 
mkdir -p debian

### echo %%% Copy files to directory, making subsitutions if needed
for i in build/package/debian/* ; do 
    if test -d $i ; then 
	case $i in 
	    */CVS) ;;
	    *)     
		echo "Copying directory `basename $i` to debian/" 
		cp -a $i debian/ ;; 
	esac
	continue
    fi
    case $i in 
	*/lib*-dev*)
	    echo "Copying `basename $i` to debian/"
	    cp -a $i debian/
	    ;;
	*/lib*.overrides.in)
	    b=`basename $i .overrides.in `
	    echo "Copying ${b}.overrides to debian/${b}${root_sovers}.overrides"
	    sed "s/@libvers@/${root_sovers}/g" \
		< $i > debian/${b}${root_sovers}.overrides
	    ;;
	*/lib*.in)
	    e=`basename $i .in | sed 's/.*\.//'`
	    b=`basename $i .$e.in`
	    echo "Copying ${b}.${e}.in to debian/${b}${root_sovers}.${e}.in"
	    cp -a $i debian/${b}${root_sovers}.${e}.in
	    ;;
	*/lib*)
	    e=`basename $i | sed 's/.*\.//'`
	    b=`basename $i .$e`
	    echo "Copying ${b}.${e}.in to debian/${b}${root_sovers}.${e}.in"
	    cp -a $i debian/${b}${root_sovers}.${e} 
	    ;; 
	*)
	    echo "Copying `basename $i` to debian/"
	    cp -a $i debian/
	    ;;
    esac
done
# cp -a build/package/debian/* debian/
find debian -name "CVS" | xargs -r rm -frv 
rm -fr debian/root-bin.png
chmod a+x debian/rules 
chmod a+x build/package/lib/*


### echo %%% Make skeleton control file 
# cat build/package/debian/control.in build/package/common/*.control \
#     > debian/control.in 

#
# EOF
#
