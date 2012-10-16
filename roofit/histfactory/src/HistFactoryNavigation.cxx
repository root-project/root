
#include <iomanip>

#include "RooStats/HistFactory/HistFactoryNavigation.h"
#include "RooStats/HistFactory/HistFactoryException.h"


ClassImp(RooStats::HistFactory::HistFactoryNavigation);


namespace RooStats {
  namespace HistFactory {


    HistFactoryNavigation::HistFactoryNavigation(ModelConfig* mc) {

      // Save the model pointer
      fModel = mc->GetPdf();
      fObservables = (RooArgSet*) mc->GetObservables();
      
      // Initialize the rest of the members
      _GetNodes(fModel, fObservables);

    }


    void HistFactoryNavigation::PrintState(const std::string& channel) {

      int label_print_width = 20;
      int bin_print_width = 10;
      std::cout << std::endl << channel << ":" << std::endl;

      // Loop over the SampleFunctionMap and print the individual histograms
      // to get the total histogram for the channel
      int num_bins = 0;
      std::map< std::string, RooAbsReal*> SampleFunctionMap = GetSampleFunctionMap(channel);
      std::map< std::string, RooAbsReal*>::iterator itr = SampleFunctionMap.begin();
      for( ; itr != SampleFunctionMap.end(); ++itr) {
	std::string sample_name = itr->first;
	std::string tmp_name = sample_name + channel + "_pretty_tmp";
	TH1* sample_hist = GetSampleHist(channel, sample_name, tmp_name);
	num_bins = sample_hist->GetNbinsX();
	std::cout << std::setw(label_print_width) << sample_name;
	for(int i = 0; i < num_bins; ++i) {
	  std::cout << std::setw(bin_print_width) << sample_hist->GetBinContent(i+1);
	}
	std::cout << std::endl;
	delete sample_hist;
      }

      std::string line_break;
      for(int i = 0; i < label_print_width + num_bins*bin_print_width; ++i) {
	line_break += "=";
      }

      //std::cout << "=================================" << std::endl;
      std::cout << line_break << std::endl;

      std::string tmp_name = channel + "_pretty_tmp";
      TH1* channel_hist = GetChannelHist(channel, tmp_name);
      std::cout << std::setw(label_print_width) << "TOTAL:";
      for(int i = 0; i < channel_hist->GetNbinsX(); ++i) {
	std::cout << std::setw(bin_print_width) << channel_hist->GetBinContent(i+1);
      }
      std::cout << std::endl;
      delete channel_hist;

      return;

    }


    void HistFactoryNavigation::PrintState() {
      // Loop over channels and print their states, one after another
      for(unsigned int i = 0; i < fChannelNameVec.size(); ++i) {
	PrintState(fChannelNameVec.at(i));
      }
    }


    void HistFactoryNavigation::PrintDataSet(RooDataSet* data, const std::string& channel_to_print) {

      // Print the contents of a 'HistFactory' RooDataset
      // These are stored in a somewhat odd way that makes
      // them difficult to inspect for humans.
      // They have the following layout:
      // =====================================================
      // ChannelA      ChannelB     ChannelCat   Weight
      // -----------------------------------------------------
      // bin_1_center   0           ChannelA     bin_1_height
      // bin_2_center   0           ChannelA     bin_2_height
      //      0        bin_1_center ChannelB     bin_1_height
      //      0        bin_2_center ChannelB     bin_2_height
      //                        ...etc...
      // =====================================================

      // Create a map of channel names to the channel's bin values
      std::map< std::string, std::vector<double> > ChannelBinsMap;

      // Then loop and fill these vectors for each channel      
      for(int i = 0; i < data->numEntries(); ++i) {

	// Get the row
	const RooArgSet* row = data->get(i);

	// The current bin height is the weight
	// of this row.
	double bin_height = data->weight();

	// Let's figure out the channel
	// For now, the variable 'channelCat' is magic, but
	// we should change this to be a bit smarter...
	std::string channel = row->getCatLabel("channelCat");

	// Get the vector of bin heights (creating if necessary)
	// and append
	std::vector<double>& bins = ChannelBinsMap[channel];
	bins.push_back(bin_height);

      }

      // Now that we have the information, we loop over
      // our newly created object and pretty print the info
      std::map< std::string, std::vector<double> >::iterator itr = ChannelBinsMap.begin();
      for( ; itr != ChannelBinsMap.end(); ++itr) {

	std::string channel_name = itr->first;

	// If we pass a channel string, we only print that one channel
	if( channel_to_print != "" && channel_name != channel_to_print) continue;

	std::cout << std::setw(20) << channel_name + " (data)";
	std::vector<double>& bins = itr->second;
	for(unsigned int i = 0; i < bins.size(); ++i) {
	  std::cout << std::setw(10) << bins.at(i);
	}
	std::cout << std::endl;

      }
    }


    void HistFactoryNavigation::PrintModelAndData(RooDataSet* data) {
      // Loop over all channels and print model
      // (including all samples) and compare
      // it to the supplied dataset

      for( unsigned int i = 0; i < fChannelNameVec.size(); ++i) {
	std::string channel = fChannelNameVec.at(i);
	PrintState(channel);
	PrintDataSet(data, channel);
      }
      
      std::cout << std::endl;

    }


    void HistFactoryNavigation::PrintParameters(bool IncludeConstantParams) {

      // Get the list of parameters
      RooArgSet* params = fModel->getParameters(*fObservables);
      
      std::cout << std::endl;

      // Create the title row
      std::cout << std::setw(30) << "Parameter";
      std::cout << std::setw(15) << "Value"
		<< std::setw(15) << "Error Low" 
		<< std::setw(15) << "Error High"
		<< std::endl;
      
      // Loop over the parameters and print their values, etc
      TIterator* paramItr = params->createIterator();
      RooRealVar* param = NULL;
      while( (param=(RooRealVar*)paramItr->Next()) ) {

	if( !IncludeConstantParams && param->isConstant() ) continue;

	std::cout << std::setw(30) << param->GetName();
	std::cout << std::setw(15) << param->getVal();
	if( !param->isConstant() ) {
	  std::cout << std::setw(15) << param->getErrorLo() << std::setw(15) << param->getErrorHi();
	}
	std::cout<< std::endl;
      }
      
      std::cout << std::endl;

      return;
    }

    double HistFactoryNavigation::GetBinValue(int bin, const std::string& channel) {
      // Get the total bin height for the ith bin (ROOT indexing convention)
      // in channel 'channel'
      // (Could be optimized, it uses an intermediate histogram for now...)
      
      // Get the histogram, fetch the bin content, and return
      TH1* channel_hist_tmp = GetChannelHist(channel, (channel+"_tmp").c_str());
      double val = channel_hist_tmp->GetBinContent(bin);
      delete channel_hist_tmp;
      return val;
    }


    double HistFactoryNavigation::GetBinValue(int bin, const std::string& channel, const std::string& sample){  
      // Get the total bin height for the ith bin (ROOT indexing convention)
      // in channel 'channel'
      // (This will be slow if you plan on looping over it.
      //  Could be optimized, it uses an intermediate histogram for now...)

      // Get the histogram, fetch the bin content, and return
      TH1* sample_hist_tmp = GetSampleHist(channel, sample,  (channel+"_tmp").c_str());
      double val = sample_hist_tmp->GetBinContent(bin);
      delete sample_hist_tmp;
      return val;
    }


    std::map< std::string, RooAbsReal*> HistFactoryNavigation::GetSampleFunctionMap(const std::string& channel) {
      // Get a map of strings to function pointers, 
      // which each function cooresponds to a sample

      std::map< std::string, std::map< std::string, RooAbsReal*> >::iterator channel_itr;
      channel_itr = fChannelSampleFunctionMap.find(channel);
      if( channel_itr==fChannelSampleFunctionMap.end() ){
	std::cout << "Error: Channel: " << channel << " not found in Navigation" << std::endl;
	throw hf_exc();
      }

      return channel_itr->second;
    }


    RooAbsReal* HistFactoryNavigation::SampleFunction(const std::string& channel, const std::string& sample){
      // Return the function object pointer cooresponding
      // to a particular sample in a particular channel

      std::map< std::string, std::map< std::string, RooAbsReal*> >::iterator channel_itr;
      channel_itr = fChannelSampleFunctionMap.find(channel);
      if( channel_itr==fChannelSampleFunctionMap.end() ){
	std::cout << "Error: Channel: " << channel << " not found in Navigation" << std::endl;
	throw hf_exc();
      }

      std::map< std::string, RooAbsReal*>& SampleMap = channel_itr->second;
      std::map< std::string, RooAbsReal*>::iterator sample_itr;
      sample_itr = SampleMap.find(sample);
      if( sample_itr==SampleMap.end() ){
	std::cout << "Error: Sample: " << sample << " not found in Navigation" << std::endl;
	throw hf_exc();
      }
      
      return sample_itr->second;

    }


    RooArgSet* HistFactoryNavigation::GetObservableSet(const std::string& channel) {
      // Get the observables for a particular channel

      std::map< std::string, RooArgSet*>::iterator channel_itr;
      channel_itr = fChannelObservMap.find(channel);
      if( channel_itr==fChannelObservMap.end() ){
	std::cout << "Error: Channel: " << channel << " not found in Navigation" << std::endl;
	throw hf_exc();
      }
      
      return channel_itr->second;

    }


    TH1* HistFactoryNavigation::GetSampleHist(const std::string& channel, const std::string& sample,
					      const std::string& hist_name) {
      // Get a histogram of the expected values for
      // a particular sample in a particular channel
      // Give a name, or a default one will be used

      RooArgList observable_list( *GetObservableSet(channel) );
      
      std::string name = hist_name;
      if(hist_name=="") name = channel + "_" + sample + "_hist";

      RooAbsReal* sample_function = SampleFunction(channel, sample);

      return MakeHistFromRooFunction( sample_function, observable_list, name );
				     
    }


    TH1* HistFactoryNavigation::GetChannelHist(const std::string& channel, const std::string& hist_name) {
      // Get a histogram of the total expected value
      // per bin for this channel
      // Give a name, or a default one will be used

      RooArgList observable_list( *GetObservableSet(channel) );

      std::map< std::string, RooAbsReal*> SampleFunctionMap = GetSampleFunctionMap(channel);

      // Okay, 'loop' once 
      TH1* total_hist=NULL;
      std::map< std::string, RooAbsReal*>::iterator itr = SampleFunctionMap.begin();
      for( ; itr != SampleFunctionMap.end(); ++itr) {
	std::string sample_name = itr->first;
	std::string tmp_hist_name = sample_name + "_hist_tmp";
	RooAbsReal* sample_function = itr->second;
	TH1* sample_hist = MakeHistFromRooFunction(sample_function, observable_list, tmp_hist_name);
	total_hist = (TH1*) sample_hist->Clone("TotalHist");
	delete sample_hist;
	break;
      }
      total_hist->Reset();

      // Loop over the SampleFunctionMap and add up all the histograms
      // to get the total histogram for the channel
      itr = SampleFunctionMap.begin();
      for( ; itr != SampleFunctionMap.end(); ++itr) {
	std::string sample_name = itr->first;
	std::string tmp_hist_name = sample_name + "_hist_tmp";
	RooAbsReal* sample_function = itr->second;
	TH1* sample_hist = MakeHistFromRooFunction(sample_function, observable_list, tmp_hist_name);
	total_hist->Add(sample_hist);
	delete sample_hist;
      }

      if(hist_name=="") total_hist->SetName(hist_name.c_str());
      else total_hist->SetName( (channel + "_hist").c_str() ); 

      return total_hist;

    }


    void HistFactoryNavigation::_GetNodes(RooAbsPdf* modelPdf, const RooArgSet* observables) {

      // Get the pdf from the ModelConfig
      //RooAbsPdf* modelPdf = mc->GetPdf();
      //RooArgSet* observables = mc->GetObservables();

      // Create vectors to hold the channel pdf's
      // as well as the set of observables for each channel
      //std::map< std::string, RooAbsPdf* >  channelPdfMap;
      //std::map< std::string, RooArgSet* >  channelObservMap;
      
      // Check if it is a simultaneous pdf or not
      // (if it's an individual channel, it won't be, if it's
      // combined, it's simultaneous)
      // Fill the channel vectors based on the structure
      // (Obviously, if it's not simultaneous, there will be
      // only one entry in the vector for the single channel)
      if(strcmp(modelPdf->ClassName(),"RooSimultaneous")==0){

	// If so, get a list of the component pdf's:
	RooSimultaneous* simPdf = (RooSimultaneous*) modelPdf;
	RooCategory* channelCat = (RooCategory*) (&simPdf->indexCat());
	
	// Iterate over the categories and get the
	// pdf and observables for each category
	TIterator* iter = channelCat->typeIterator() ;
	RooCatType* tt = NULL;
        while((tt=(RooCatType*) iter->Next())) {
	  std::string ChannelName = tt->GetName();
	  fChannelNameVec.push_back( ChannelName );
	  RooAbsPdf* pdftmp = simPdf->getPdf(ChannelName.c_str()) ;
	  RooArgSet* obstmp = pdftmp->getObservables(*observables) ;
	  fChannelPdfMap[ChannelName] = pdftmp;
	  fChannelObservMap[ChannelName] =  obstmp;
	}

      } else { 
	RooArgSet* obstmp = modelPdf->getObservables(*observables) ;	
	std::string ChannelName = modelPdf->GetName();
	fChannelNameVec.push_back(ChannelName);
	fChannelPdfMap[ChannelName] = modelPdf;
	fChannelObservMap[ChannelName] = obstmp;

      }

      // Okay, now we have maps of the pdfs 
      // and the observable list per channel
      // We then loop over the channel pdfs:
      // and find their RooRealSumPdfs
      // std::map< std::string, RooRealSumPdf* > channelSumNodeMap;

      for( unsigned int i = 0; i < fChannelNameVec.size(); ++i ) {

	std::string ChannelName = fChannelNameVec.at(i);
	RooAbsPdf* pdf = fChannelPdfMap[ChannelName];
	//std::string Name = fChannelNameMap[ChannelName];

	// Loop over the pdf's components and find
	// the (one) that is a RooRealSumPdf
	// Based on the mode, we assume that node is 
	// the "unconstrained" pdf node for that channel
	RooArgSet* components = pdf->getComponents();
	TIterator* argItr = components->createIterator();
	RooAbsArg* arg = NULL;
	while( (arg=(RooAbsArg*)argItr->Next()) ) {
	  std::string ClassName = arg->ClassName();
	  if( ClassName == "RooRealSumPdf" ) {
	    fChannelSumNodeMap[ChannelName] = (RooRealSumPdf*) arg;
	    break;
	  }
	}
      }
      
      // Okay, now we have all necessary
      // nodes filled for each channel.
      for( unsigned int i = 0; i < fChannelNameVec.size(); ++i ) {

	std::string ChannelName = fChannelNameVec.at(i);
	RooRealSumPdf* sumPdf = (RooRealSumPdf*) fChannelSumNodeMap[ChannelName];
	
	// We now take the RooRealSumPdf and loop over
	// its component functions.  The RooRealSumPdf turns
	// a list of functions (expected events or bin heights
	// per sample) and turns it into a pdf.
	// Therefore, we loop over it to find the expected
	// height for the various samples

	// First, create a map to store the function nodes
	// for each sample in this channel
	std::map< std::string, RooAbsReal*> sampleFunctionMap;

	// Loop over the sample nodes in this
	// channel's RooRealSumPdf
	RooArgList nodes = sumPdf->funcList();
	TIterator* sampleItr = nodes.createIterator();
	RooAbsArg* sample;
	while( (sample=(RooAbsArg*)sampleItr->Next()) ) {

	  // Cast this node as a function
	  RooAbsReal* func = (RooAbsReal*) sample;

	  // Do a bit of work to get the name of each sample
	  std::string SampleName = sample->GetName();
	  if( SampleName.find("L_x_") != std::string::npos ) {
	    size_t index = SampleName.find("L_x_");
	    SampleName.replace( index, 4, "" );
	  }
	  if( SampleName.find(ChannelName.c_str()) != std::string::npos ) {
	    size_t index = SampleName.find(ChannelName.c_str());
	    SampleName = SampleName.substr(0, index-1);
	  }

	  // And simply save this node into our map
	  sampleFunctionMap[SampleName] = func;

	}

	fChannelSampleFunctionMap[ChannelName] = sampleFunctionMap;

	// Okay, now we have a list of histograms
	// representing the samples for this channel.

      }

    }


    TH1* HistFactoryNavigation::MakeHistFromRooFunction( RooAbsReal* func, RooArgList vars, std::string name ) {

      // Turn a RooAbsReal* into a TH1* based 
      // on a template histogram.  
      // The 'vars' arg list defines the x (and y and z variables)
      // Loop over the bins of the Template,
      // find the bin centers, 
      // Scan the input Var over those bin centers,
      // and use the value of the function
      // to make the new histogram

      // Make the new histogram
      // Cone and empty the template
      //      TH1* hist = (TH1*) histTemplate.Clone( name.c_str() );

      int dim = vars.getSize();

      TH1* hist=NULL;

      if( dim==1 ) {
	RooRealVar* varX = (RooRealVar*) vars.at(0);
	hist = func->createHistogram( name.c_str(),*varX, RooFit::Binning(varX->getBinning()), RooFit::Scaling(false) );
      }
      else if( dim==2 ) {
	RooRealVar* varX = (RooRealVar*) vars.at(0);
	RooRealVar* varY = (RooRealVar*) vars.at(1);
	hist = func->createHistogram( name.c_str(),*varX, RooFit::Binning(varX->getBinning()), RooFit::Scaling(false),
				      RooFit::YVar(*varY, RooFit::Binning(varY->getBinning())) );
      }
      else if( dim==3 ) {
	RooRealVar* varX = (RooRealVar*) vars.at(0);
	RooRealVar* varY = (RooRealVar*) vars.at(1);
	RooRealVar* varZ = (RooRealVar*) vars.at(2);
	hist = func->createHistogram( name.c_str(),*varX, RooFit::Binning(varX->getBinning()), RooFit::Scaling(false),
				      RooFit::YVar(*varY, RooFit::Binning(varY->getBinning())),
				      RooFit::YVar(*varZ, RooFit::Binning(varZ->getBinning())) );
      }
      else {
	std::cout << "Error: To Create Histogram from RooAbsReal function, Dimension must be 1, 2, or 3" << std::endl;
	throw hf_exc();
      }

      return hist;
    }

    // A simple wrapper to use a ModelConfig
    void HistFactoryNavigation::_GetNodes(ModelConfig* mc) {
      RooAbsPdf* modelPdf = mc->GetPdf();
      const RooArgSet* observables = mc->GetObservables();
      _GetNodes(modelPdf, observables);
    }

  } // namespace HistFactory
} // namespace RooStats




