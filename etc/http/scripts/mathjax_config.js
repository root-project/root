MathJax.Hub.Config({ TeX: { extensions: ["color.js"] }, SVG: { mtextFontInherit: true, minScaleAdjust: 100, matchFontHeight: true } });
MathJax.Hub.Register.StartupHook("SVG Jax Ready",function () {
   var VARIANT = MathJax.OutputJax.SVG.FONTDATA.VARIANT;
   VARIANT["normal"].fonts.unshift("MathJax_SansSerif");
   VARIANT["bold"].fonts.unshift("MathJax_SansSerif-bold");
   VARIANT["italic"].fonts.unshift("MathJax_SansSerif");
   VARIANT["-tex-mathit"].fonts.unshift("MathJax_SansSerif");
});
MathJax.Ajax.loadComplete(JSROOT.source_dir + "scripts/mathjax_config.js");
