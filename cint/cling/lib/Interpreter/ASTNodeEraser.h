//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_ASTREVERSION
#define CLING_ASTREVERSION

namespace clang {
  class Decl;
  class Sema;
}

namespace cling {

  class DeclReverter;

  /// \brief A simple eraser class that removes already created AST Nodes.
  class ASTNodeEraser {
  private:
    clang::Sema* m_Sema;
    DeclReverter* m_DeclReverter;
  public:
    ASTNodeEraser(clang::Sema* S);
    ~ASTNodeEraser();

    ///\brief Removes given declaration from the AST. Removing includes reseting
    /// various internal stuctures in the compiler to their previous state. For
    /// example it resets the lookup tables if the declaration has name and can
    /// be looked up. Reverts the redeclaration chain if the declaration was 
    /// redeclarable and so on.
    /// Note1 that the code generated for the declaration is not removed yet.
    /// Note2 does not do dependency analysis.
    ///
    /// @param[in] D - The declaration to be removed. 
    ///
    bool RevertDecl(clang::Decl *D);
  };
} // end namespace cling

#endif // CLING_ASTREVERSION
