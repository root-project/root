#!/bin/sh
#
# $Id: makedebdir.sh,v 1.17 2007/05/14 07:42:44 rdm Exp $
#
# Make the debian packaging directory 
#
purge=0
leave=0
clean=0
setup=1
upcl=1
root_sovers=`cat build/version_number | sed 's,/.*,,'` 


# ____________________________________________________________________
usage ()
{
    cat <<EOF
Usage: $0 [OPTIONS]
 
Options:
	-h,--help	This help 
	-p,--purge	Purge source directory
	-c,--clean	Clean the source directory 
	-n,--no-setup	Do not setup debian directory
EOF
    exit 0
}

# ____________________________________________________________________
message()
{
    opt=
    post=
    while test $# -gt 0 ; do 
	case $1 in 
	    -n) post=" ..." ; opt="$opt $1" ;;
	    -*) opt="$opt $1" ;; 
	    *)  break;;
	esac
	shift
    done
    echo $opt "[1m$@[0m${post}"
}

# ____________________________________________________________________
check_retval()
{
    retval=$?
    if test $# -gt 0 ; then 
	message -n $@ ":"
    fi
    if test $retval -ne 0 ; then 
	echo "[1;31m Failure: $@[0m"
	cd $savdir
	exit $retval
    else
	echo "[1;32m OK[0m"
    fi
}

# ____________________________________________________________________
extract_tarballs()
{
    dir=$1 ; shift
    ext=$1 ; shift
    if test "x$ext" = "x" ; then ext=.tar.gz ; fi
    case $ext in 
	.tar.gz|.tgz) 	dopt=z ;; 
	.tar.bz2|.tbz2) dopt=j ;;
	.tar.Z)		dopt=Z ;;
	*)              dopt=a ;;
    esac
    tars=`find $dir -name "*${ext}"  2>/dev/null` 
    for i in ${tars} ; do 
	case `basename $i ${ext}` in 
	    .bogus) continue ;; 
            *)
		sub=`tar -t${dopt}f $i | head -n 1 | xargs basename` 
		if test "x$sub" != "x" && \
		    test "x$sub" != "x." ; then 
		    message -n "Removing $dir/$sub" 
		    rm -rf $dir/$sub 
		    check_retval
		fi
		message -n "Extracting `basename $i`"
		tar -x${dopt}f $i -C $dir
		check_retval ""
		
		message -n "Removing $i"
		rm -f ${i}
		touch ${dir}/headers.d
		check_retval 
		;;
	esac
    done
}


# ____________________________________________________________________
purge ()
{
    test $purge -lt 1 && return 0 

    cat <<-EOF 
	[1;31m=============================================================
	Warning: Purging sources of unwanted stuff
	
	I will expand tar-balls, and remove them.  I will also remove 
	non-free True Type Fonts.  To restore these files, you should 
	do a CVS update. 
	=============================================================[0m
	EOF
    # Now, remove files we definitely don't want 
    # rm -f fonts/*.ttf 
    message -n "Removing unwanted files"
    rm -f \
        build/package/common/libroot-clarens.control		\
        build/package/common/libroot.control			\
        build/package/common/libroot-dev.install.in		\
        build/package/common/libroot.install.in			\
        build/package/common/libroot-mathmore.control		\
        build/package/common/libroot-minuit.control		\
        build/package/common/libroot-mlp.control		\
        build/package/common/libroot-python.control		\
        build/package/common/libroot-python-dev.install.in	\
        build/package/common/libroot-python.install.in		\
        build/package/common/libroot-quadp.control		\
        build/package/common/libroot-ruby.control		\
        build/package/common/libroot-unuran.control		\
	build/package/common/libroot-dev.control		\
	build/package/debian/libroot.postinst			\
	build/package/debian/libroot.postrm			\
        build/package/common/root-plugin-alien.control		\
        build/package/common/root-plugin-castor.control		\
        build/package/common/root-plugin-chirp.control		\
        build/package/common/root-plugin-dcache.control		\
        build/package/common/root-plugin-fftw3.control		\
        build/package/common/root-plugin-fumili.control		\
        build/package/common/root-plugin-gl.control		\
        build/package/common/root-plugin-globus.control		\
        build/package/common/root-plugin-hbook.control		\
        build/package/common/root-plugin-hbook.install.in	\
        build/package/common/root-plugin-krb5.control		\
        build/package/common/root-plugin-maxdb.control		\
        build/package/common/root-plugin-minuit2.control	\
        build/package/common/root-plugin-mysql.control		\
        build/package/common/root-plugin-netx.control		\
        build/package/common/root-plugin-odbc.control		\
        build/package/common/root-plugin-oracle.control		\
        build/package/common/root-plugin-peac.control		\
        build/package/common/root-plugin-pgsql.control		\
        build/package/common/root-plugin-pythia5.control	\
        build/package/common/root-plugin-pythia6.control	\
        build/package/common/root-plugin-qt.control		\
        build/package/common/root-plugin-sql.control		\
        build/package/common/root-plugin-srp.control		\
        build/package/common/root-plugin-venus.control		\
        build/package/common/root-plugin-xml.control		\
        build/package/common/root-plugin-xproof.control		\
	build/package/common/root-plugin-clarens.control	\
	build/package/common/root-plugin-ldap.control		\
	build/package/common/root-plugin-minuit.control		\
	build/package/common/root-plugin-mlp.control		\
	build/package/common/root-plugin-python.control		\
	build/package/common/root-plugin-python.install.in	\
	build/package/common/root-plugin-quadp.control		\
	build/package/common/root-plugin-roofit.control		\
	build/package/common/root-plugin-ruby.control		\
	build/package/common/root-plugin-sapdb.control		\
	build/package/debian/root-plugin-roofit.copyright	\
	build/package/debian/root-system-proofd.postinst.in	\
	build/package/debian/root-system-rootd.postinst.in	\
	build/package/common/root-cint.control			\
	build/package/common/root-cint.copyright		\
	build/package/common/root-cint.install.in		\
	build/package/debian/root-cint.copyright		\
	build/package/debian/root-cint.postinst.in		\
	build/package/debian/root-cint.postrm.in		\
	build/package/debian/root-cint.prerm.in			\
	build/package/common/root-rootd.install.in		\
	build/package/common/root-xrootd.install.old		\
	build/package/debian/pycompat				\
	build/package/common/ttf-root.control			\
	build/package/common/ttf-root.install.in		\
	build/package/debian/ttf-root.copyright			\
	build/package/debian/dirs				\
	build/package/lib/makerpmspecs.sh			\
	fonts/LICENSE						
    check_retval "unwanted files"


    # rm -rf asimage/src/libAfterImage				
    # rm -rf xrootd/src/xrootd 
    # rm -rf unuran/src/unuran-*-root
    message -n "Removing non-free fonts"
    for i in fonts/*.ttf ; do 
	if test ! -f ${i} ; then continue ; fi 
	case $i in 
	    */symbol.ttf) ;; 
	    *) rm $i ;;
	esac
    done
    check_retval 

    if test $leave -lt 1 ; then 
	message -n "Removing old packaging files"
        # Remove old package files 
	for i in build/package/*/root-{bin,doc,common,xrootd,rootd,proofd}* 
	  do 
	  if test ! -f $i ; then continue ; fi 
	  rm $i 
	done
	check_retval 
    fi

    # Extract tar-balls, and remove the tar-balls. 
    message "Extracting tar-balls"
    extract_tarballs math/unuran/src 
    if test $? -eq 0 ; then 
	rm -f math/unuran/src/unuran-*-root/config.status
	rm -f math/unuran/src/unuran-*-root/config.log
    fi
}

# ____________________________________________________________________
clean()
{
    if test $clean -lt 1 ; then return 0 ; fi 

    message -n "Cleaning" 
    make maintainer-clean \
		UNURANETAG= \
		UNURKEEP=yes > /dev/null 2>&1
    rm -rf debian
    rm -f fonts/s050000l.pfb
    rm -f fonts/s050000l.pe
    check_retval
}


# ____________________________________________________________________
vers2num()
{
    echo $1 | 		\
	tr '/' '.' | 	\
	awk 'BEGIN {FS="."}{printf "%d", (($1*1000)+$2)*1000+$3}'
}    

# ____________________________________________________________________
update_cl()
{
    test $upcl -lt 1 && return 0 
    
    cl=build/package/debian/changelog
    message -n "Update $cl"
    root_vers=`cat build/version_number | tr '/' '.'` 
    last_vers=`head -n 1 $cl | sed 's/root-system (\(.*\)).*/\1/'`
    root_lvers=`vers2num $root_vers`
    last_lvers=`vers2num $last_vers`
    if test $root_lvers -gt $last_lvers ; then 
	res=$root_lvers
	dch -v ${root_vers}-1 -c $cl "New upstream version"
    else 
	res="same version"
    fi
    check_retval $res
}

# ____________________________________________________________________
setup()
{
    test $setup -lt 1 && return 0 

    ### echo %%% Make the directory 
    message "Setting up debian directory ..."
    mkdir -p debian

    ### echo %%% Copy files to directory, making subsitutions if needed
    for i in build/package/debian/* ; do 
	if test -d $i ; then 
	    case $i in 
		*/CVS|.svn) continue ;;
	    esac
	fi

	case $i in 
	    */lib*-static.*.in)
		e=`basename $i .in | sed 's/.*\.//'`
		b=`basename $i .$e.in`
		t="${b}.${e}.in"
		echo "Copying ${b}.${e}.in to debian/${t}"
		cp -a $i debian/${t}
		;;
	    */lib*-dev*)
		echo "Copying `basename $i` to debian/"
		cp -a $i debian/
		;;
	    */lib*.overrides.in)
		b=`basename $i .overrides.in `
		t="${b}${root_sovers}.overrides"
		echo "Copying ${b}.overrides to debian/${t}"
		sed "s/@libvers@/${root_sovers}/g" < $i > debian/${t}
		;;
	    */lib*.in)
		e=`basename $i .in | sed 's/.*\.//'`
		b=`basename $i .$e.in`
		t="${b}${root_sovers}.${e}.in"
		echo "Copying ${b}.${e}.in to debian/${t}"
		cp -a $i debian/${t}
		;;
	    */lib*)
		e=`basename $i | sed 's/.*\.//'`
		b=`basename $i .$e`
		t="${b}${root_sovers}.${e}"
		echo "Copying ${b}.${e}.in to debian/${t}"
		cp -a $i debian/${t} 
		;; 
	    */s050000l.pfb|*/s050000l.pe)
                # Copying s050000l.pfb and s050000l.pe to font directory
		b=`basename $i` 
		echo "Copying $b to fonts/$b" 
		cp $i fonts/
		;;
	    */po)
		b=`basename $i`
		echo "Making directory debian/$b"
		mkdir -p debian/$b
		echo "Copying to directory debian/$b"
		cp -a $i/* debian/$b/
		;;
	    *)
		b=`basename ${i}`
		echo "Copying $b to debian/$b"
		cp -a $i debian/
		;;
	esac
    done

    # cp -a build/package/debian/* debian/
    find debian -name "CVS"  | xargs -r rm -frv 
    find debian -name ".svn" | xargs -r rm -frv 
    rm -fr debian/root-system-bin.png
    rm -fr debian/application-x-root.png
    chmod a+x debian/rules 
    chmod a+x build/package/lib/*

    # Make sure we rebuild debian/control
    touch debian/control.in
    check_retval "Setting up debian directory"
}

# ____________________________________________________________________
while test $# -gt 0 ; do 
    case $1 in 
	-h|--help) 	usage			;; 
	-p|--purge)    	purge=1 		;; 
	-c|--clean)   	clean=1 		;; 
	-n|--no-setup) 	setup=0 ; 	upcl=0 	;;
	-o|--leave-old) leave=1 		;;
	*) echo "Unknown option: $1, try $0 --help" > /dev/stderr ;;
    esac
    shift
done 

# ____________________________________________________________________
purge
clean
update_cl
setup

#
# EOF
#
