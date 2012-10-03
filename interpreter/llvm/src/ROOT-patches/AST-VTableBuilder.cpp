Index: tools/clang/lib/AST/VTableBuilder.cpp
===================================================================
--- tools/clang/lib/AST/VTableBuilder.cpp	(revision 46280)
+++ tools/clang/lib/AST/VTableBuilder.cpp	(working copy)
@@ -1160,7 +1160,8 @@
       continue;
     }
 
-    if (MD->getParent() == MostDerivedClass)
+    if (MD->getParent()->getCanonicalDecl()
+        == MostDerivedClass->getCanonicalDecl())
       AddThunk(MD, Thunk);
   }
 }
@@ -1353,7 +1354,8 @@
   
   // If the overrider is the first base in the primary base chain, we know
   // that the overrider will be used.
-  if (Overrider->getParent() == FirstBaseInPrimaryBaseChain)
+  if (Overrider->getParent()->getCanonicalDecl()
+      == FirstBaseInPrimaryBaseChain->getCanonicalDecl())
     return true;
   
   VTableBuilder::PrimaryBasesSetVectorTy PrimaryBases;
@@ -1417,7 +1419,8 @@
       const CXXMethodDecl *OverriddenMD = *I;
       
       // We found our overridden method.
-      if (OverriddenMD->getParent() == PrimaryBase)
+      if (OverriddenMD->getParent()->getCanonicalDecl()
+          == PrimaryBase->getCanonicalDecl())
         return OverriddenMD;
     }
   }
@@ -1512,7 +1515,8 @@
                                   Overrider);
 
           if (ThisAdjustment.VCallOffsetOffset &&
-              Overrider.Method->getParent() == MostDerivedClass) {
+              Overrider.Method->getParent()->getCanonicalDecl()
+              == MostDerivedClass->getCanonicalDecl()) {
 
             // There's no return adjustment from OverriddenMD and MD,
             // but that doesn't mean there isn't one between MD and
