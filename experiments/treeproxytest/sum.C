double sum() {


 ofstream ndfile;
 ndfile.open("neutrino_pc.txt", ios::app);
 ofstream adfile;
 adfile.open("antineu_pc.txt", ios::app);

  Int_t evt_run_no[20];
  Int_t evt_subrun_no[20];
  Int_t evt_snarl_no[20];

  Int_t evt_beg_plane[20];
  Int_t evt_nu_plane[20];
  Int_t evt_nv_plane[20];

  Float_t evt_vtx_x[20];
  Float_t evt_vtx_y[20];
  Float_t evt_vtx_z[20];
  Float_t evt_vtx_u[20];
  Float_t evt_vtx_v[20];

  Float_t evt_end_x[20];
  Float_t evt_end_y[20];
  Float_t evt_end_z[20];
  Float_t evt_end_u[20];
  Float_t evt_end_v[20];

  Float_t evt_trk_ds[20];
  Float_t evt_trk_fit_pass[20];
  Float_t evt_trk_red_chi2[20];
  Int_t evt_trk_plane_begu[20];
  Int_t evt_trk_plane_begv[20];

  Float_t evt_neu_energy[20];
  Float_t evt_shw_energy[20];
  Float_t evt_trk_range_mom[20];
  Float_t evt_trk_fit_mom[20];
  Float_t evt_trk_eqp_mom[20];
  Float_t evt_trk_qp_mom[20];

  Float_t evt_trk_time[20];
  Float_t evt_shw_time[20];

  Int_t evt_trk[20];
  Int_t evt_shw[20];
 
  Float_t evt_trk_dcosz[20];
  Float_t evt_y_reco[20];
  Float_t evt_x_reco[20];
  Float_t evt_q2_reco[20];

  Int_t two_trks[20];
  Float_t trk_momentum_fit;
  Float_t trk_reduced_chi2;

  Int_t evt_recomc[20];
  Float_t evt_mc_mu_energy[20];
  Float_t evt_mc_shw_energy[20];
  Float_t evt_mc_neu_energy[20];
  Float_t evt_mc_vtxx[20];
  Float_t evt_mc_vtxy[20];
  Float_t evt_mc_vtxz[20];

  Float_t evt_mc_a[20];
  Int_t evt_mc_inu[20];
  Int_t evt_mc_iaction[20];
  Int_t evt_mc_ires[20];
  Float_t evt_mc_x[20];
  Float_t evt_mc_y[20];
  Float_t evt_mc_q2[20];

  for(int jj=0;jj<20;jj++)
    {
      evt_run_no[jj]=0;
      evt_subrun_no[jj]=0;
      evt_snarl_no[jj]=0;

      evt_beg_plane[jj]=0;
      evt_nu_plane[jj]=0;
      evt_nv_plane[jj]=0;

      evt_vtx_x[jj]=0;
      evt_vtx_y[jj]=0;
      evt_vtx_z[jj]=0;
      evt_vtx_u[jj]=0;
      evt_vtx_v[jj]=0;

      evt_end_x[jj]=0;
      evt_end_y[jj]=0;
      evt_end_z[jj]=0;
      evt_end_u[jj]=0;
      evt_end_v[jj]=0;

      evt_trk_ds[jj]=0;
      evt_trk_fit_pass[jj]=0;
      evt_trk_red_chi2[jj]=0;
      evt_trk_plane_begu[jj]=0;
      evt_trk_plane_begv[jj]=0;

      evt_neu_energy[jj]=0;
      evt_shw_energy[jj]=0;
      evt_trk_range_mom[jj]=0;
      evt_trk_fit_mom[jj]=0;
      evt_trk_qp_mom[jj]=0;
      evt_trk_eqp_mom[jj]=0;

      evt_trk_time[jj]=0;
      evt_shw_time[jj]=0;

      evt_trk[jj]=0;
      evt_shw[jj]=0;
      
      evt_trk_dcosz[jj]=0;
      evt_y_reco[jj]=0;
      evt_x_reco[jj]=0;
      evt_q2_reco[jj]=0;
      two_trks[jj]=0;   

      evt_recomc[jj]=0;
      evt_mc_mu_energy[jj]=0;
      evt_mc_shw_energy[jj]=0;
      evt_mc_neu_energy[jj]=0; 
      evt_mc_vtxx[jj]=0;
      evt_mc_vtxy[jj]=0;
      evt_mc_vtxz[jj]=0;
      evt_mc_a[jj]=0;
      evt_mc_inu[jj]=0;
      evt_mc_iaction[jj]=0;
      evt_mc_ires[jj]=0;
      evt_mc_x[jj]=0;
      evt_mc_y[jj]=0;
      evt_mc_q2[jj]=0;

    }

int counter=0;
Float_t timediff = 200e-09;

for ( Int_t itrk = 0; itrk < NtpSR.evthdr.ntrack; itrk++ ) 

    {
        int match=0;

       if(NtpSR.trk.momentum.qp[itrk]==0) trk_momentum_fit=0;
       else {trk_momentum_fit = 1/NtpSR.trk.momentum.qp[itrk];} 

       if(NtpSR.trk.fit.ndof[itrk] ==0) trk_reduced_chi2=0;
       else {trk_reduced_chi2 = NtpSR.trk.time.chi2[itrk] / NtpSR.trk.fit.ndof[itrk] ;}

       // reconstruction cuts

	if(trk_reduced_chi2 > 20 )  continue;

	if(NtpSR.trk.fit.pass[itrk] == 0 ) continue;

	if(TMath::Abs(NtpSR.trk.plane.begu[itrk] - NtpSR.trk.plane.begv[itrk] )> 6 ) continue;


//cuts for ensuring that tracks begin in the instrumented region

	if(NtpSR.trk.vtx.z[itrk] < 0.6 || NtpSR.trk.vtx.z[itrk]> 3.56 ) continue;

	if(NtpSR.trk.vtx.u[itrk] < 0.3 || NtpSR.trk.vtx.u[itrk]> 1.8 ) continue;
                                                                                                                           
	if(NtpSR.trk.vtx.v[itrk] < -1.8 || NtpSR.trk.vtx.v[itrk]> -0.3 ) continue;

	if(NtpSR.trk.vtx.x[itrk] > 2.4) continue;

	if(TMath::Sqrt((NtpSR.trk.vtx.x[itrk]*NtpSR.trk.vtx.x[itrk])+(NtpSR.trk.vtx.y[itrk]*NtpSR.trk.vtx.y[itrk]))<0.8) continue;

        
        for ( Int_t ishw = 0; ishw < NtpSR.evthdr.nshower; ishw++ )
	    {
	      if(TMath::Abs(NtpSR.shw.plane.beg[ishw]-NtpSR.trk.plane.beg[itrk])<=5 && TMath::Abs(NtpSR.shw.vtx.x[ishw]-NtpSR.trk.vtx.x[itrk])<=0.15 && TMath::Abs(NtpSR.shw.vtx.y[ishw]-NtpSR.trk.vtx.y[itrk])<=0.15 && TMath::Abs(NtpSR.shw.vtx.t[ishw]-NtpSR.trk.vtx.t[itrk])<timediff)
		{
                  match=1;

                  evt_recomc[counter]=NtpTH.thtrk.neumc[itrk];
                  Int_t index = evt_recomc[counter];  
                  if(index<0) continue;
    
                  evt_mc_mu_energy[counter]=mc.p4mu1[index][3];            
                  evt_mc_shw_energy[counter]=mc.p4shw[index][3];            
                  evt_mc_neu_energy[counter]=mc.p4neu[index][3];            
                  evt_mc_vtxx[counter]=mc.vtxx[index];
                  evt_mc_vtxy[counter]=mc.vtxy[index];
                  evt_mc_vtxz[counter]=mc.vtxz[index];
                  evt_mc_a[counter]=mc.a[index];      
                  evt_mc_inu[counter]=mc.inu[index];                        
                  evt_mc_iaction[counter]=mc.iaction[index];      
                  evt_mc_ires[counter]=mc.iresonance[index];      
                  evt_mc_x[counter]=mc.x[index];      
                  evt_mc_y[counter]=mc.y[index];      
                  evt_mc_q2[counter]=mc.q2[index];      

                  evt_run_no[counter]= NtpSR.NtpSRRecord.fRun;
                  evt_subrun_no[counter]= NtpSR.NtpSRRecord.fSubRun;
                  evt_snarl_no[counter]= NtpSR.NtpSRRecord.fSnarl;          

		}//end of if statement

		}//end of loop for showers


    }      //end of loop for trks


return 0;

}
