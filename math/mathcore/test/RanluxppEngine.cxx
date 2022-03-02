// @(#)root/mathcore:$Id$
// Author: Jonas Hahnfeld 11/2020

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// This test uses EXPECT_EQ also for floating point numbers - the expected
// values are entered with enough digits to ensure binary equality.

#include "Math/RanluxppEngine.h"

#include "gtest/gtest.h"

using namespace ROOT::Math;

TEST(RanluxppEngine, random2048)
{
   RanluxppEngine2048 rng(314159265);

   // The following values were obtained without skipping.

   EXPECT_EQ(rng.IntRndm(), 39378223178113);
   EXPECT_EQ(rng.Rndm(), 0.57072241146576274673);

   // Skip ahead in block.
   rng.Skip(8);
   EXPECT_EQ(rng.IntRndm(), 52221857391813);
   EXPECT_EQ(rng.Rndm(), 0.16812543081078956675);

   // The next call needs to advance the state.
   EXPECT_EQ(rng.IntRndm(), 185005245121693);
   EXPECT_EQ(rng.Rndm(), 0.28403302782895423206);

   // Skip ahead to start of next block.
   rng.Skip(10);
   EXPECT_EQ(rng.IntRndm(), 89237874214503);
   EXPECT_EQ(rng.Rndm(), 0.79969842495805920635);

   // Skip ahead across blocks.
   rng.Skip(42);
   EXPECT_EQ(rng.IntRndm(), 49145148745150);
   EXPECT_EQ(rng.Rndm(), 0.74670661284082484599);
}

TEST(RanluxppCompatEngineJames, P3)
{
   RanluxppCompatEngineJamesP3 rng(314159265);

   // These values match: gsl_rng_ranlux, ranluxI_James, ranluxpp_James
   EXPECT_EQ(rng.Rndm(), 0.53981816768646240234);
   EXPECT_EQ(rng.Rndm(), 0.76155042648315429688);
   EXPECT_EQ(rng.Rndm(), 0.06029939651489257812);
   EXPECT_EQ(rng.Rndm(), 0.79600262641906738281);
   EXPECT_EQ(rng.Rndm(), 0.30631220340728759766);

   // Skip to the 24th number, right before the LCG is used to advance the state.
   rng.Skip(18);
   EXPECT_EQ(rng.Rndm(), 0.20569473505020141602);
   // The LCG advances for the next call:
   EXPECT_EQ(rng.Rndm(), 0.76727509498596191406);

   // Skip to the 101st number.
   rng.Skip(75);
   EXPECT_EQ(rng.Rndm(), 0.43156743049621582031);
   EXPECT_EQ(rng.Rndm(), 0.03774416446685791016);
   EXPECT_EQ(rng.Rndm(), 0.24897110462188720703);
   EXPECT_EQ(rng.Rndm(), 0.00147783756256103516);
   EXPECT_EQ(rng.Rndm(), 0.90274453163146972656);
}

TEST(RanluxppCompatEngineJames, P4)
{
   RanluxppCompatEngineJamesP4 rng(314159265);

   // These values match: gsl_rng_ranlux389
   EXPECT_EQ(rng.Rndm(), 0.53981816768646240234);
   EXPECT_EQ(rng.Rndm(), 0.76155042648315429688);
   EXPECT_EQ(rng.Rndm(), 0.06029939651489257812);
   EXPECT_EQ(rng.Rndm(), 0.79600262641906738281);
   EXPECT_EQ(rng.Rndm(), 0.30631220340728759766);

   // Skip to the 24th number, right before the LCG is used to advance the state.
   rng.Skip(18);
   EXPECT_EQ(rng.Rndm(), 0.20569473505020141602);
   // The LCG advances for the next call:
   EXPECT_EQ(rng.Rndm(), 0.84534603357315063477);

   // Skip to the 101st number.
   rng.Skip(75);
   EXPECT_EQ(rng.Rndm(), 0.67576026916503906250);
   EXPECT_EQ(rng.Rndm(), 0.90395343303680419922);
   EXPECT_EQ(rng.Rndm(), 0.31414842605590820312);
   EXPECT_EQ(rng.Rndm(), 0.98801732063293457031);
   EXPECT_EQ(rng.Rndm(), 0.93221199512481689453);
}

TEST(RanluxppCompatEngineGslRanlxs, ranlxs0)
{
   RanluxppCompatEngineGslRanlxs0 rng(314159265);

   // These values match: gsl_rng_ranlxs0
   EXPECT_EQ(rng.Rndm(), 0.95476531982421875000);
   EXPECT_EQ(rng.Rndm(), 0.10175001621246337891);
   EXPECT_EQ(rng.Rndm(), 0.03923547267913818359);
   EXPECT_EQ(rng.Rndm(), 0.23141473531723022461);
   EXPECT_EQ(rng.Rndm(), 0.56545680761337280273);

   // Skip to the 24th number, right before the LCG is used to advance the state.
   rng.Skip(18);
   EXPECT_EQ(rng.Rndm(), 0.66594201326370239258);
   // The LCG advances for the next call:
   EXPECT_EQ(rng.Rndm(), 0.08081126213073730469);

   // Skip to the 101st number.
   rng.Skip(75);
   EXPECT_EQ(rng.Rndm(), 0.74328583478927612305);
   EXPECT_EQ(rng.Rndm(), 0.79350239038467407227);
   EXPECT_EQ(rng.Rndm(), 0.09384918212890625000);
   EXPECT_EQ(rng.Rndm(), 0.00877797603607177734);
   EXPECT_EQ(rng.Rndm(), 0.81286895275115966797);
}

TEST(RanluxppCompatEngineGslRanlxs, ranlxs0_default)
{
   RanluxppCompatEngineGslRanlxs0 rng;
   EXPECT_EQ(rng.Rndm(), 0.32085895538330078125);

   rng.SetSeed(0);
   EXPECT_EQ(rng.Rndm(), 0.32085895538330078125);
}

TEST(RanluxppCompatEngineGslRanlxs, ranlxs1)
{
   RanluxppCompatEngineGslRanlxs1 rng(314159265);

   // These values match: gsl_rng_ranlxs1
   EXPECT_EQ(rng.Rndm(), 0.64368855953216552734);
   EXPECT_EQ(rng.Rndm(), 0.67669576406478881836);
   EXPECT_EQ(rng.Rndm(), 0.21001303195953369141);
   EXPECT_EQ(rng.Rndm(), 0.27618223428726196289);
   EXPECT_EQ(rng.Rndm(), 0.04699170589447021484);

   // Skip to the 24th number, right before the LCG is used to advance the state.
   rng.Skip(18);
   EXPECT_EQ(rng.Rndm(), 0.81624960899353027344);
   // The LCG advances for the next call:
   EXPECT_EQ(rng.Rndm(), 0.52857619524002075195);

   // Skip to the 101st number.
   rng.Skip(75);
   EXPECT_EQ(rng.Rndm(), 0.76676791906356811523);
   EXPECT_EQ(rng.Rndm(), 0.28538894653320312500);
   EXPECT_EQ(rng.Rndm(), 0.30258500576019287109);
   EXPECT_EQ(rng.Rndm(), 0.38395434617996215820);
   EXPECT_EQ(rng.Rndm(), 0.50305640697479248047);
}

TEST(RanluxppCompatEngineGslRanlxs, ranlxs1_default)
{
   RanluxppCompatEngineGslRanlxs1 rng;
   EXPECT_EQ(rng.Rndm(), 0.06963491439819335938);

   rng.SetSeed(0);
   EXPECT_EQ(rng.Rndm(), 0.06963491439819335938);
}

TEST(RanluxppCompatEngineGslRanlxs, ranlxs2)
{
   RanluxppCompatEngineGslRanlxs2 rng(314159265);

   // These values match: gsl_rng_ranlxs2
   EXPECT_EQ(rng.Rndm(), 0.84824979305267333984);
   EXPECT_EQ(rng.Rndm(), 0.75232243537902832031);
   EXPECT_EQ(rng.Rndm(), 0.12018418312072753906);
   EXPECT_EQ(rng.Rndm(), 0.05532860755920410156);
   EXPECT_EQ(rng.Rndm(), 0.05795300006866455078);

   // Skip to the 24th number, right before the LCG is used to advance the state.
   rng.Skip(18);
   EXPECT_EQ(rng.Rndm(), 0.13871461153030395508);
   // The LCG advances for the next call:
   EXPECT_EQ(rng.Rndm(), 0.97801673412322998047);

   // Skip to the 101st number.
   rng.Skip(75);
   EXPECT_EQ(rng.Rndm(), 0.92746645212173461914);
   EXPECT_EQ(rng.Rndm(), 0.82626664638519287109);
   EXPECT_EQ(rng.Rndm(), 0.77763950824737548828);
   EXPECT_EQ(rng.Rndm(), 0.49001514911651611328);
   EXPECT_EQ(rng.Rndm(), 0.88770687580108642578);
}

TEST(RanluxppCompatEngineGslRanlxs, ranlxs2_default)
{
   RanluxppCompatEngineGslRanlxs2 rng;
   EXPECT_EQ(rng.Rndm(), 0.53008824586868286133);

   rng.SetSeed(0);
   EXPECT_EQ(rng.Rndm(), 0.53008824586868286133);
}

TEST(RanluxppCompatEngineGslRanlxd, ranlxd1)
{
   RanluxppCompatEngineGslRanlxd1 rng(314159265);

   // These values match: gsl_rng_ranlxd1
   EXPECT_EQ(rng.Rndm(), 0.32330420393203596063);
   EXPECT_EQ(rng.Rndm(), 0.72381776078560733367);
   EXPECT_EQ(rng.Rndm(), 0.88817512439535306612);
   EXPECT_EQ(rng.Rndm(), 0.04598644245910676887);
   EXPECT_EQ(rng.Rndm(), 0.76110447268111158792);

   // Skip to the 12th number, right before the LCG is used to advance the state.
   rng.Skip(6);
   EXPECT_EQ(rng.Rndm(), 0.18375035143688478456);
   // The LCG advances for the next call:
   EXPECT_EQ(rng.Rndm(), 0.83312931792853106572);

   // Skip to the 101st number.
   rng.Skip(87);
   EXPECT_EQ(rng.Rndm(), 0.49329437415119770094);
   EXPECT_EQ(rng.Rndm(), 0.37550852748539753634);
   EXPECT_EQ(rng.Rndm(), 0.93543254436396239271);
   EXPECT_EQ(rng.Rndm(), 0.43517686324045001811);
   EXPECT_EQ(rng.Rndm(), 0.77751776122982363404);
}

TEST(RanluxppCompatEngineGslRanlxd, ranlxd1_default)
{
   RanluxppCompatEngineGslRanlxd1 rng;
   EXPECT_EQ(rng.Rndm(), 0.83451879245814453157);

   rng.SetSeed(0);
   EXPECT_EQ(rng.Rndm(), 0.83451879245814453157);
}

TEST(RanluxppCompatEngineGslRanlxd, ranlxd2)
{
   RanluxppCompatEngineGslRanlxd2 rng(314159265);

   // These values match: gsl_rng_ranlxd2
   EXPECT_EQ(rng.Rndm(), 0.28994560412829528673);
   EXPECT_EQ(rng.Rndm(), 0.06729010717305428102);
   EXPECT_EQ(rng.Rndm(), 0.69404482039860582177);
   EXPECT_EQ(rng.Rndm(), 0.56285566362747729841);
   EXPECT_EQ(rng.Rndm(), 0.34655505440137091000);

   // Skip to the 12th number, right before the LCG is used to advance the state.
   rng.Skip(6);
   EXPECT_EQ(rng.Rndm(), 0.12507187350916737500);
   // The LCG advances for the next call:
   EXPECT_EQ(rng.Rndm(), 0.97552452698574398937);

   // Skip to the 101st number.
   rng.Skip(87);
   EXPECT_EQ(rng.Rndm(), 0.85118391727546338643);
   EXPECT_EQ(rng.Rndm(), 0.94509863457767551154);
   EXPECT_EQ(rng.Rndm(), 0.29384155017066149185);
   EXPECT_EQ(rng.Rndm(), 0.48683495244473462549);
   EXPECT_EQ(rng.Rndm(), 0.70125558306756730076);
}

TEST(RanluxppCompatEngineGslRanlxd, ranlxd2_default)
{
   RanluxppCompatEngineGslRanlxd2 rng;
   EXPECT_EQ(rng.Rndm(), 0.07725383918716843823);

   rng.SetSeed(0);
   EXPECT_EQ(rng.Rndm(), 0.07725383918716843823);
}

TEST(RanluxppCompatEngineLuescherRanlxs, ranlxs0)
{
   RanluxppCompatEngineLuescherRanlxs0 rng(314159265);

   // These values match: ranlxs0
   EXPECT_EQ(rng.Rndm(), 0.65404480695724487305);
   EXPECT_EQ(rng.Rndm(), 0.08226150274276733398);
   EXPECT_EQ(rng.Rndm(), 0.29213166236877441406);
   EXPECT_EQ(rng.Rndm(), 0.50535500049591064453);
   EXPECT_EQ(rng.Rndm(), 0.95475935935974121094);

   // Skip to the 92nd number, right before the LCG is used to advance the state.
   rng.Skip(87);
   EXPECT_EQ(rng.Rndm(), 0.63037145137786865234);
   EXPECT_EQ(rng.Rndm(), 0.82800912857055664062);
   EXPECT_EQ(rng.Rndm(), 0.31651723384857177734);
   EXPECT_EQ(rng.Rndm(), 0.09057545661926269531);
   // The LCG advances for the next calls:
   EXPECT_EQ(rng.Rndm(), 0.06262010335922241211);
   EXPECT_EQ(rng.Rndm(), 0.39436388015747070312);
   EXPECT_EQ(rng.Rndm(), 0.61765056848526000977);
   EXPECT_EQ(rng.Rndm(), 0.14016568660736083984);

   // Skip to the 401st number.
   rng.Skip(300);
   EXPECT_EQ(rng.Rndm(), 0.21224951744079589844);
   EXPECT_EQ(rng.Rndm(), 0.28637069463729858398);
   EXPECT_EQ(rng.Rndm(), 0.01129972934722900391);
   EXPECT_EQ(rng.Rndm(), 0.62236660718917846680);
   EXPECT_EQ(rng.Rndm(), 0.11493819952011108398);
}

TEST(RanluxppCompatEngineLuescherRanlxs, ranlxs1)
{
   RanluxppCompatEngineLuescherRanlxs1 rng(314159265);

   // These values match: ranlxs1
   EXPECT_EQ(rng.Rndm(), 0.90099400281906127930);
   EXPECT_EQ(rng.Rndm(), 0.76765865087509155273);
   EXPECT_EQ(rng.Rndm(), 0.22530400753021240234);
   EXPECT_EQ(rng.Rndm(), 0.83992105722427368164);
   EXPECT_EQ(rng.Rndm(), 0.59816044569015502930);

   // Skip to the 92nd number, right before the LCG is used to advance the state.
   rng.Skip(87);
   EXPECT_EQ(rng.Rndm(), 0.64669114351272583008);
   EXPECT_EQ(rng.Rndm(), 0.46657902002334594727);
   EXPECT_EQ(rng.Rndm(), 0.12610912322998046875);
   EXPECT_EQ(rng.Rndm(), 0.10862994194030761719);
   // The LCG advances for the next calls:
   EXPECT_EQ(rng.Rndm(), 0.01416563987731933594);
   EXPECT_EQ(rng.Rndm(), 0.58082711696624755859);
   EXPECT_EQ(rng.Rndm(), 0.38216590881347656250);
   EXPECT_EQ(rng.Rndm(), 0.91653412580490112305);

   // Skip to the 401st number.
   rng.Skip(300);
   EXPECT_EQ(rng.Rndm(), 0.76539427042007446289);
   EXPECT_EQ(rng.Rndm(), 0.11502689123153686523);
   EXPECT_EQ(rng.Rndm(), 0.40491354465484619141);
   EXPECT_EQ(rng.Rndm(), 0.96093446016311645508);
   EXPECT_EQ(rng.Rndm(), 0.38819086551666259766);
}

TEST(RanluxppCompatEngineLuescherRanlxs, ranlxs2)
{
   RanluxppCompatEngineLuescherRanlxs2 rng(314159265);

   // These values match: ranlxs2
   EXPECT_EQ(rng.Rndm(), 0.60732543468475341797);
   EXPECT_EQ(rng.Rndm(), 0.74212568998336791992);
   EXPECT_EQ(rng.Rndm(), 0.76778668165206909180);
   EXPECT_EQ(rng.Rndm(), 0.56459045410156250000);
   EXPECT_EQ(rng.Rndm(), 0.51524215936660766602);

   // Skip to the 92nd number, right before the LCG is used to advance the state.
   rng.Skip(87);
   EXPECT_EQ(rng.Rndm(), 0.07774782180786132812);
   EXPECT_EQ(rng.Rndm(), 0.12600058317184448242);
   EXPECT_EQ(rng.Rndm(), 0.56134593486785888672);
   EXPECT_EQ(rng.Rndm(), 0.36321890354156494141);
   // The LCG advances for the next calls:
   EXPECT_EQ(rng.Rndm(), 0.23446798324584960938);
   EXPECT_EQ(rng.Rndm(), 0.42847990989685058594);
   EXPECT_EQ(rng.Rndm(), 0.21235740184783935547);
   EXPECT_EQ(rng.Rndm(), 0.30497443675994873047);

   // Skip to the 401st number.
   rng.Skip(300);
   EXPECT_EQ(rng.Rndm(), 0.17799735069274902344);
   EXPECT_EQ(rng.Rndm(), 0.23861807584762573242);
   EXPECT_EQ(rng.Rndm(), 0.65686619281768798828);
   EXPECT_EQ(rng.Rndm(), 0.39222949743270874023);
   EXPECT_EQ(rng.Rndm(), 0.45217937231063842773);
}

TEST(RanluxppCompatEngineLuescherRanlxd, ranlxd1)
{
   RanluxppCompatEngineLuescherRanlxd1 rng(314159265);

   // These values match: ranlxd1
   EXPECT_EQ(rng.Rndm(), 0.40183950697007020381);
   EXPECT_EQ(rng.Rndm(), 0.75729181403551848462);
   EXPECT_EQ(rng.Rndm(), 0.44293039330603889425);
   EXPECT_EQ(rng.Rndm(), 0.27412463825007193918);
   EXPECT_EQ(rng.Rndm(), 0.90543407471131232001);

   // Skip to the 44th number, right before the LCG is used to advance the state.
   rng.Skip(39);
   EXPECT_EQ(rng.Rndm(), 0.35330884918042571030);
   EXPECT_EQ(rng.Rndm(), 0.53342096246839787455);
   EXPECT_EQ(rng.Rndm(), 0.87389084649229076263);
   EXPECT_EQ(rng.Rndm(), 0.89137004735275837675);
   // The LCG advances for the next calls:
   EXPECT_EQ(rng.Rndm(), 0.92005646450699529737);
   EXPECT_EQ(rng.Rndm(), 0.83437278691344118897);
   EXPECT_EQ(rng.Rndm(), 0.96002403910338784954);
   EXPECT_EQ(rng.Rndm(), 0.19537819235692666098);

   // Skip to the 401st number.
   rng.Skip(348);
   EXPECT_EQ(rng.Rndm(), 0.74482704516052322674);
   EXPECT_EQ(rng.Rndm(), 0.36930523933338221809);
   EXPECT_EQ(rng.Rndm(), 0.36572707642276824913);
   EXPECT_EQ(rng.Rndm(), 0.91548098968585378543);
   EXPECT_EQ(rng.Rndm(), 0.55814623941527585771);
}

TEST(RanluxppCompatEngineLuescherRanlxd, ranlxd2)
{
   RanluxppCompatEngineLuescherRanlxd2 rng(314159265);

   // These values match: ranlxd2
   EXPECT_EQ(rng.Rndm(), 0.52702589450092673928);
   EXPECT_EQ(rng.Rndm(), 0.14545363991469173470);
   EXPECT_EQ(rng.Rndm(), 0.79180414135301191436);
   EXPECT_EQ(rng.Rndm(), 0.07733853533090595533);
   EXPECT_EQ(rng.Rndm(), 0.14180497711929263005);

   // Skip to the 44th number, right before the LCG is used to advance the state.
   rng.Skip(39);
   EXPECT_EQ(rng.Rndm(), 0.18603867668678475411);
   EXPECT_EQ(rng.Rndm(), 0.13778587858302770996);
   EXPECT_EQ(rng.Rndm(), 0.70244055389935411426);
   EXPECT_EQ(rng.Rndm(), 0.90056757149901045523);
   // The LCG advances for the next calls:
   EXPECT_EQ(rng.Rndm(), 0.93044152534523050235);
   EXPECT_EQ(rng.Rndm(), 0.99330367160287025285);
   EXPECT_EQ(rng.Rndm(), 0.63528838564139178402);
   EXPECT_EQ(rng.Rndm(), 0.61276861283659656010);

   // Skip to the 401st number.
   rng.Skip(348);
   EXPECT_EQ(rng.Rndm(), 0.37774257837078550892);
   EXPECT_EQ(rng.Rndm(), 0.20167640311612444748);
   EXPECT_EQ(rng.Rndm(), 0.34101141228811471251);
   EXPECT_EQ(rng.Rndm(), 0.35947589472738883387);
   EXPECT_EQ(rng.Rndm(), 0.51160837322844798791);
}

TEST(RanluxppCompatEngineStdRanlux24, compare)
{
   RanluxppCompatEngineStdRanlux24 rng(314159265);

   // These values match: std::ranlux24
   EXPECT_EQ(rng.IntRndm(), 6389521);
   EXPECT_EQ(rng.IntRndm(), 1245860);
   EXPECT_EQ(rng.IntRndm(), 9047089);
   EXPECT_EQ(rng.IntRndm(), 5613314);
   EXPECT_EQ(rng.IntRndm(), 15388463);

   // Skip to the 23rd number, right before the LCG is used to advance the state.
   rng.Skip(17);
   EXPECT_EQ(rng.IntRndm(), 14135596);
   // The LCG advances for the next call:
   EXPECT_EQ(rng.IntRndm(), 1842057);

   // Skip to the 101st number.
   rng.Skip(76);
   EXPECT_EQ(rng.IntRndm(), 7894321);
   EXPECT_EQ(rng.IntRndm(), 2634015);
   EXPECT_EQ(rng.IntRndm(), 12134196);
   EXPECT_EQ(rng.IntRndm(), 15231589);
   EXPECT_EQ(rng.IntRndm(), 11032869);
}

TEST(RanluxppCompatEngineStdRanlux24, default)
{
   RanluxppCompatEngineStdRanlux24 rng;

   // Skip to the 10000th number, which is specified by the standard.
   rng.Skip(9999);
   EXPECT_EQ(rng.IntRndm(), 9901578);
}

TEST(RanluxppCompatEngineStdRanlux48, compare)
{
   RanluxppCompatEngineStdRanlux48 rng(314159265);

   // These values match: std::ranlux48
   EXPECT_EQ(rng.IntRndm(), 2902733192977);
   EXPECT_EQ(rng.IntRndm(), 183625379875889);
   EXPECT_EQ(rng.IntRndm(), 164401649471280);
   EXPECT_EQ(rng.IntRndm(), 158572192479531);
   EXPECT_EQ(rng.IntRndm(), 227024878504710);

   // Skip to the 11th number, right before the LCG is used to advance the state.
   rng.Skip(5);
   EXPECT_EQ(rng.IntRndm(), 137631957902972);
   // The LCG advances for the next call:
   EXPECT_EQ(rng.IntRndm(), 122438241205867);

   // Skip to the 101st number.
   rng.Skip(88);
   EXPECT_EQ(rng.IntRndm(), 155713325118081);
   EXPECT_EQ(rng.IntRndm(), 75203262258561);
   EXPECT_EQ(rng.IntRndm(), 164155826303104);
   EXPECT_EQ(rng.IntRndm(), 58159697115827);
   EXPECT_EQ(rng.IntRndm(), 89006261856016);
}

TEST(RanluxppCompatEngineStdRanlux48, default)
{
   RanluxppCompatEngineStdRanlux48 rng;

   // Skip to the 10000th number, which is specified by the standard.
   rng.Skip(9999);
   EXPECT_EQ(rng.IntRndm(), 249142670248501);
}
