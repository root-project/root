// $Id: HvsPathHandler.h,v 1.9 2009-12-17 18:38:53 avalassi Exp $
#ifndef RELATIONALCOOL_HVSPATHHANDLER_H
#define RELATIONALCOOL_HVSPATHHANDLER_H 1

// Include files
#include <map>
#include <string>
#include <vector>

namespace cool
{

  /** @class HvsPathHandler HvsPathHandler.h
   *
   *  Handler of hierarchical path names for HVS.
   *
   *  This class is used to encode, decode and validate full path names
   *  of the form "/node1/node2/node3" in a UNIX-like node hierarchy.
   *
   *  The handler is also used in the COOL conditions database implementation:
   *  any features specific to COOL may be changed by virtual inheritance.
   *
   *  The documentation for the class is written assuming "/" is the
   *  separator character, but this is hardcoded in a single clas variable.
   *
   *  @author Andrea Valassi and Sven A. Schmidt
   *  @date   2004-12-09
   */

  class HvsPathHandler {

    friend class HvsPathHandlerTest;

  public:

    /// Standard constructor.
    HvsPathHandler();

    /// Destructor.
    virtual ~HvsPathHandler();

    /// Return the separator character '/'.
    char separator() const
    {
      return '/';
    }

    /// Return the unresolved root name "".
    const std::string rootUnresolvedName() const
    {
      return "";
    }

    /// Return the full root path "/".
    const std::string rootFullPath() const
    {
      return rootUnresolvedName() + std::string( 1, separator() );
    }

    /// Split a full HVS path into parent full path and child unresolved name,
    /// e.g. split "/a/b/c" into "/a/b" and "c".
    /// Special case: "/a" is split into "/" and "a".
    /// Throw an exception if the path has double or trailing separators.
    /// Throw an exception if the path does not start by "/".
    /// Throw an exception if the path is the root folder "/".
    const std::pair<std::string, std::string>
    splitFullPath( const std::string& fullPath );

    /// Decode a full HVS path into a list of hierarchy node unresolved names,
    /// e.g. split "/a/b/c" into "", "a", "b" and "c".
    /// Throw an exception if the path has double or trailing separators.
    /// Throw an exception if the path does not start by "/"
    /// (special case: "/" represents the root folder "").
    const std::vector<std::string>
    decodeFullPath( const std::string& fullPath );

    /// Encode a list of unresolved hierarchy node names into a full HVS path,
    /// e.g. encode "", "a", "b", "c" into "/a/b/c".
    /// Throw an exception if any node name contains a separator.
    /// Throw an exception if the first node is not the root node "".
    const std::string
    encodeFullPath( const std::vector<std::string>& nodeList );

  private:

    /// Remove trailing separators from a string,
    /// e.g. simplify "/a//b/c/" into "/a//b/c/".
    /// Special case: "/" is left as "/".
    const std::string
    removeTrailingSeparators( const std::string& aString );

    /// Remove double separators from a string,
    /// e.g. simplify "/a//b/c" into "/a/b/c".
    const std::string
    removeDoubleSeparators( const std::string& aString );

  };

}

#endif // RELATIONALCOOL_HVSPATHHANDLER_H
