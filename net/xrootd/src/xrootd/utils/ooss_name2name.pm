#!/usr/bin/env perl
#******************************************************************************
# (c) 2002 by the Board of Trustees of the Leland Stanford, Jr., University   *
#                          All Rights Reserved                                *
# Produced by Andrew Hanushevsky for Stanford University under contract       *
#            DE-AC03-76-SFO0515 with the Department of Energy                 *
#******************************************************************************

#  $Id: ooss_name2name.pm,v 1.5 2008/10/22 14:13:42 furano Exp $

# Generate file name for local and mss systems

*ooss_name2name::basedir = *basedir;     # Use caller's basedir
*ooss_name2name::mssdir  = *mssdir;      # Use caller's mssdir

# Establish our own name space
#
package  ooss_name2name;
require  Exporter;
@ISA   = qw(Exporter);
@EXPORT= qw(local2mss mss2local mss2none none2mss);


#
# If mssdir is an URL, this mechanism works only if we make sure that
# mssdir ends with a double slash, if the url does not contain a path.
# e.g.
#  root://host//
#  root://host//path
#
sub fixURLmssdir {
    my($d) = @_;
    
    if ( $d =~ m/^(http\:|root\:|xroot\:)\/\// )  {
       chop($d) if (substr($d,length($d)-1,1) eq '/');
       chop($d) if (substr($d,length($d)-1,1) eq '/');
       chop($d) if (substr($d,length($d)-1,1) eq '/');
    
       # now $d should be without any trailing slash, we add two
       # in the case the path after the hostname is empty
       return $d if ( $d =~ m|//(.*)//(.*)| );
       
       return "$d//";
    }

    return $d;
}

# Assumption is that basedir and mssdir are directory paths which may
# or may not be specified with a trailing '/'. There are four combinations
# which we must handle:
#         basedir ends in /    mssdir ends in /
# case 1:        Y                   Y
# case 2:        Y                   N 
# case 3:        N                   Y
# case 4:        N                   N

sub local2mss {
    my($fn) = @_;
    my($mssdir) = fixURLmssdir($mssdir);

    if (substr($basedir,length($basedir)-1,1) eq '/') {
	if ($fn =~ m|^$basedir(.*)|) {
	    if (substr($mssdir,length($mssdir)-1,1) eq '/') {
		return $mssdir . $1;      # case 1
	    }
	    else {
		return $mssdir . "/$1";   # case 2
	    }
	}
    }
    else {
	if ($fn =~ m|^$basedir/(.*)|) {
	    if (substr($mssdir,length($mssdir)-1,1) eq '/') {
		return $mssdir . $1;      # case 3
	    }
	    else {
		return $mssdir . "/$1";   # case 4
	    }
	}
    }
    return $fn;  # no match with basedir
}
sub mss2local {
    my($fn) = @_;
    my($mssdir) = fixURLmssdir($mssdir);

    if (substr($mssdir,length($mssdir)-1,1) eq '/') {
	if ($fn =~ m|^$mssdir(.*)|) {
	    if (substr($basedir,length($basedir)-1,1) eq '/') {
		return $basedir . $1;     # case 1
	    }
	    else {
		return $basedir . "/$1";  # case 3
	    }
	}
    }
    else {
	if ($fn =~ m|^$mssdir/(.*)|) {
	    if (substr($basedir,length($basedir)-1,1) eq '/') {
		return $basedir . $1;     # case 2
	    }
	    else {
		return $basedir . "/$1";  # case 4
	    }
	}
    }
    return $fn;  # no match with mssdir
}
sub mss2none {
    my($fn) = @_;
    my($mssdir) = fixURLmssdir($mssdir);

    if (substr($mssdir,length($mssdir)-1,1) eq '/') {
	return "/$1" if ($fn =~ m|^$mssdir(.*)|);
    }
    else {
	return "/$1" if ($fn =~ m|^$mssdir/(.*)|);
    }
    return $fn;  # no match with mssdir
}
sub none2mss {
    my($fn) = @_;
    my($mssdir) = fixURLmssdir($mssdir);


    if (substr($mssdir,length($mssdir)-1,1) eq '/') {
        my($tfn) = $mssdir;
	chop($tfn);
	return $tfn . $fn;
    }
    else {
	return $mssdir . $fn;
    }
}
