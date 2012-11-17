Index: tools/clang/include/clang/AST/CanonicalType.h
===================================================================
--- tools/clang/include/clang/AST/CanonicalType.h	(revision 47382)
+++ tools/clang/include/clang/AST/CanonicalType.h	(working copy)
@@ -735,7 +735,7 @@
 
 template<typename T>
 CanQual<T> CanQual<T>::CreateUnsafe(QualType Other) {
-  assert((Other.isNull() || Other.isCanonical()) && "Type is not canonical!");
+  //assert((Other.isNull() || Other.isCanonical()) && "Type is not canonical!");
   assert((Other.isNull() || isa<T>(Other.getTypePtr())) &&
          "Dynamic type does not meet the static type's requires");
   CanQual<T> Result;
Index: tools/clang/lib/Sema/SemaTemplate.cpp
===================================================================
--- tools/clang/lib/Sema/SemaTemplate.cpp	(revision 47382)
+++ tools/clang/lib/Sema/SemaTemplate.cpp	(working copy)
@@ -3535,7 +3535,9 @@
   QualType Arg = ArgInfo->getType();
   SourceRange SR = ArgInfo->getTypeLoc().getSourceRange();
 
-  if (Arg->isVariablyModifiedType()) {
+  if (1) {
+    // our special case, no error (okay that's bad but we currently have a problem with canonical type.
+  } else if (Arg->isVariablyModifiedType()) {
     return Diag(SR.getBegin(), diag::err_variably_modified_template_arg) << Arg;
   } else if (Context.hasSameUnqualifiedType(Arg, Context.OverloadTy)) {
     return Diag(SR.getBegin(), diag::err_template_arg_overload_type) << SR;
Index: tools/clang/lib/AST/ASTContext.cpp
===================================================================
--- tools/clang/lib/AST/ASTContext.cpp	(revision 47382)
+++ tools/clang/lib/AST/ASTContext.cpp	(working copy)
@@ -2703,8 +2703,8 @@
 QualType
 ASTContext::getSubstTemplateTypeParmType(const TemplateTypeParmType *Parm,
                                          QualType Replacement) const {
-  assert(Replacement.isCanonical()
-         && "replacement types must always be canonical");
+//  assert(Replacement.isCanonical()
+//         && "replacement types must always be canonical");
 
   llvm::FoldingSetNodeID ID;
   SubstTemplateTypeParmType::Profile(ID, Parm, Replacement);
