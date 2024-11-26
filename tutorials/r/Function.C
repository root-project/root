/// \file
/// \ingroup tutorial_r
/// \notebook -nodraw
///
/// \macro_code
///
/// \author

using namespace ROOT::R;
void Function()
{
   auto &r = ROOT::R::TRInterface::Instance();
   r.SetVerbose(1);

   // Defining functions to be used from R
   TRFunctionImport c("c");
   TRFunctionImport rlist("list");
   TRFunctionImport asformula("as.formula");
   TRFunctionImport nls("nls");
   TRFunctionImport confint("confint");
   TRFunctionImport summary("summary");
   TRFunctionImport print("print");
   TRFunctionImport plot("plot");
   TRFunctionImport lines("lines");
   TRFunctionImport devnew("dev.new");
   TRFunctionImport devoff("dev.off");
   TRFunctionImport devcur("dev.cur");
   TRFunctionImport rmin("min");
   TRFunctionImport rmax("max");
   TRFunctionImport seq("seq");
   TRFunctionImport predict("predict");

   r<<"options(device='pdf')";

   // doing the procedure
   TRObject xdata = c(-2,-1.64,-1.33,-0.7,0,0.45,1.2,1.64,2.32,2.9);
   TRObject ydata = c(0.699369,0.700462,0.695354,1.03905,1.97389,2.41143,1.91091,0.919576,-0.730975,-1.42001);

   TRDataFrame data;
   data["xdata"]=xdata;
   data["ydata"]=ydata;

   // fit = nls(ydata ~ p1*cos(p2*xdata) + p2*sin(p1*xdata), start=list(p1=1,p2=0.2))
   TRObject fit = nls(asformula("ydata ~ p1*cos(p2*xdata) + p2*sin(p1*xdata)"),Label["data"]=data, Label["start"]=rlist(Label["p1"]=1,Label["p2"]=0.2));
   print(summary(fit));

   print(confint(fit));

   if (!gROOT->IsBatch()) {
      devnew("Fitting Regression");
      plot(xdata,ydata);

      TRObject xgrid=seq(rmin(xdata),rmax(xdata),Label["len"]=10);
      lines(xgrid,predict(fit,xgrid),Label["col"] = "green");
      devoff(Label["which"] = devcur() );
   }
}
