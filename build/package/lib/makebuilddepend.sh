#!/bin/sh 
#

if test "x$1" = "xrpm" ; then 
    shift
    for i in $* ; do 
	case $i in 
	    libroot-dev)						;;
	    libroot)							;;
	    root-bin)							;;
	    root-cint)							;;
	    root-doc)							;;
	    *alien)	echo -n ", AliEn-Client" 			;;
#	    *asimage)	echo -n ", AfterStep-devel"			;;
	    *asimage)							;;
	    *castor)	echo -n ", CASTOR-client"			;;
	    *chirp)							;;
	    *clarens)							;;
	    *dcache)							;;
	    *fumili)							;;
	    *gl)							;;
	    *globus)	echo -n ", libglobus-dev"			;;
	    *hbook)	echo -n ", CERNLIB, gcc-g77"			;;
	    *krb5)	echo -n ", krb5-devel"				;;
	    *ldap)	echo -n ", openldap-devel"			;;
	    *minuit)							;;
	    *mlp)							;;
	    *mysql)	echo -n ", mysql-devel"				;;
	    *netx)							;;
	    *oracle)    echo -n ", oracle-instantclient-devel"		;;
	    *peac)							;;
	    *pgsql)	echo -n ", postgresql-devel"			;;
	    *proof)							;;
	    *pythia5)	echo -n ", pythia5-devel"			;;
	    *pythia6)	echo -n ", pythia6-devel"			;;
	    *python)	echo -n ", python-devel >= 2.1"			;;
	    *qt)	echo -n ", qt3-devel"				;;
	    *quadp)							;;
	    *ruby)	echo -n ", ruby-devel >= 1.8"			;;
	    *sapdb)	echo -n ", sapdb-callif"			;;
	    *srp)							;;
	    *venus)							;;
	    *xml)	echo -n ", libxml2-devel"			;;
	    root-proofd)						;;
	    root-rootd)							;;
	    root-xrootd) echo -n ", krb5-devel"				;;
	    ttf-root*)							;;
	    *) 
		echo "*** Warning *** Unknown package $i - please update $0" \
		    > /dev/stderr 
		;;
	esac
    done
    exit 0
fi    

### echo %%% Making build dependencies
bd=
for i in $* ; do 
    case $i in 
	libroot-dev)							;;
	libroot)							;;
	root-bin)							;;
	root-cint)							;;
	root-doc)							;;
	*alien)		echo -n ", libalien-dev" 			;;
	*asimage)	echo -n ", libafterimage-dev"			;;
	*castor)	echo -n ", libshift-dev"			;;
	*chirp)		echo -n ", libchirp-dev"			;;
	*clarens)	echo -n ", libxmlrpc-c-dev"			;;
	*dcache)	echo -n ", libdcap-dev"				;;
	*fumili)							;;
	*gl)		echo -n ", xlibmesa-glu-dev |  libglu-dev"	;;
	*globus)	echo -n ", libglobus-dev"			;;
	*hbook)		echo -n ", libpacklib1-dev, fortran-compiler|g77" ;;
	*krb5)		echo -n ", libkrb5-dev|heimdal-dev"		;;
	*ldap)		echo -n ", libldap-dev"				;;
	*oracle)    	echo -n ", oracle-instantclient-devel"		;;
	*minuit)							;;
	*mlp)								;;
	*mysql)		echo -n ", libmysqlclient-dev | libmysqlclient12-dev | libmysqlclient14-dev" ;;
	*netx)								;;
	*peac)								;;
	*pgsql)		echo -n ", postgresql-dev"			;;
	*proof)								;;
	*pythia5)	echo -n ", pythia5-dev"				;;
	*pythia6)	echo -n ", pythia6-dev"				;;
	*python)	echo -n ", python-dev (>= 2.1)"			;;
	*qt)		echo -n ", libqt3-dev | libqt3-mt-dev"		;;
	*quadp)								;;
	*ruby)		echo -n ", ruby1.8-dev | ruby-dev (>= 1.8)"	;;
	*sapdb)		echo -n ", sapdb-callif"			;;
	*srp)		echo -n ", libsrputil-dev"			;;
	*venus)		echo -n ", libvenus-dev"			;;
	*xml)		echo -n ", libxml2-dev"				;;
	root-proofd)							;;
	root-rootd)							;;
	root-xrootd)	echo -n ", libkrb5-dev|heimdal-dev"		;;
	ttf-root*)							;;
	*) 
	    echo "*** Warning *** Unknown package $i - please update $0" \
		> /dev/stderr 
	    ;;
    esac
done

#
# EOF
#
