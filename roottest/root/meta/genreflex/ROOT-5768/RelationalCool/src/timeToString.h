// $Id: timeToString.h,v 1.13 2012-06-29 14:04:15 avalassi Exp $
#ifndef RELATIONALCOOL_TIMETOSTRING_H
#define RELATIONALCOOL_TIMETOSTRING_H

// Include files
#include <cstdio> // For sprintf on gcc45
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include "CoolKernel/Exception.h"
#include "CoolKernel/Time.h"

namespace cool {

  /// Convert a string date "yyyy-mm-dd_hh:mm:ss.nnnnnnnnn into a Time.
  /// The format is the one we used to store dates for both Oracle and MySQL.
  /// The input string is expected to represent a GMT time and ends with "GMT"
  inline const Time stringToTime( const std::string& timeString )
  {
    int year, month, day, hour, min, sec;
    long nsec;
    if ( timeString.size() ==
         std::string( "yyyy-mm-dd_hh:mm:ss.nnnnnnnnn GMT" ).size()
         && sscanf( timeString.c_str(), "%4d-%2d-%2d_%2d:%2d:%2d.%9ld GMT",
                    &year, &month, &day, &hour, &min, &sec, &nsec) == 7
         && timeString.substr(4,1) == "-"
         && timeString.substr(7,1) == "-" // Month in [1-12]
         && timeString.substr(10,1) == "_"
         && timeString.substr(13,1) == ":"
         && timeString.substr(16,1) == ":"
         && timeString.substr(19,1) == "."
         && timeString.substr(29,4) == " GMT" ) {
      Time time( year, month, day, hour, min, sec, nsec );
      /*
      std::cout<< "__stringToTime ( '" << timeString << "' )" << std::endl;
      std::cout<<"Year: "<<year<<" -> "<<time.year()<<std::endl;
      std::cout<<"Month: "<<month<<" -> "<<time.month()<<std::endl;
      std::cout<<"Day: "<<day<<" -> "<<time.day()<<std::endl;
      std::cout<<"Hour: "<<hour<<" -> "<<time.hour()<<std::endl;
      std::cout<<"Min: "<<min<<" -> "<<time.minute()<<std::endl;
      std::cout<<"Sec: "<<sec<<" -> "<<time.second()<<std::endl;
      std::cout<<"nSec: "<<nsec<<" -> "<<time.nanosecond()<<std::endl;
      std::cout<< "__stringToTime = '" << time << "'" << std::endl;
      */
      return time;
    } else {
      std::stringstream msg;
      msg << "Error decoding string '" << timeString << "' into Time";
      throw cool::Exception( msg.str(), "cool::stringToTime" );
    }
  }

  /// Convert any ITime into a string in the format expected by stringToTime.
  /// Useful to print an ITime (there is no default operator<<).
  /// The output string represents a GMT time and ends with "GMT"
  inline const std::string timeToString( const ITime& time )
  {
    int year = time.year();
    int month = time.month(); // Months are in [1-12]
    int day = time.day();
    int hour = time.hour();
    int min = time.minute();
    int sec = time.second();
    long nsec = time.nanosecond();
    char timeChar[] = "yyyy-mm-dd_hh:mm:ss.nnnnnnnnn GMT";
    int nChar = std::string(timeChar).size();
    if ( snprintf( timeChar, strlen(timeChar)+1, // Fix Coverity SECURE_CODING
                   "%4.4d-%2.2d-%2.2d_%2.2d:%2.2d:%2.2d.%9.9ld GMT",
                   year, month, day, hour, min, sec, nsec) == nChar ) {
      std::string timeString = std::string( timeChar );
      /*
      std::cout<< "__timeToString ( '" << time << "' )" << std::endl;
      std::cout<<"Year: "<<year<<std::endl;
      std::cout<<"Month(1-12): "<<month<<std::endl;
      std::cout<<"Day: "<<day<<std::endl;
      std::cout<<"Hour: "<<hour<<std::endl;
      std::cout<<"Min: "<<min<<std::endl;
      std::cout<<"Sec: "<<sec<<std::endl;
      std::cout<<"nSec: "<<nsec<<std::endl;
      std::cout<< "__timeToString = '" << timeString << "'" << std::endl;
      */
      return timeString;
    } else {
      std::stringstream msg;
      msg << "PANIC! Error encoding ITime into string: "
          << year << "-" << month << "-" << day << "_"
          << hour << ":" << min << ":" << sec << "." << nsec;
      throw cool::Exception( msg.str(), "cool::timeToString" );
    }

  }

}
#endif // RELATIONALCOOL_TIMETOSTRING_H
