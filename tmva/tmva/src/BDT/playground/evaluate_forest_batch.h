// File automatically generated! 
/// Functions that defines the inference of a single tree


#pragma cling optimize(3)

void evaluate_forest_batch ( const std::vector<std::vector<float>>& events_vector, std::vector<bool> &preds){
std::vector<float> event;
float result;
for (size_t i=0; i<500000; i++){
     result = 0;
     event = events_vector[i];
if (event[1] < 1.434955){
if (event[3] < 3.665319){
if (event[4] < -2.844356){
// This is a leaf node
result += -0.014995;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000082;
}
}
else { // if condition is not respected
if (event[1] < -1.266439){
// This is a leaf node
result += 0.076923;
}
else { // if condition is not respected
// This is a leaf node
result += -0.100000;
}
}
}
else { // if condition is not respected
if (event[1] < 1.443307){
if (event[2] < -0.955072){
// This is a leaf node
result += 0.019355;
}
else { // if condition is not respected
// This is a leaf node
result += -0.031262;
}
}
else { // if condition is not respected
if (event[2] < -3.014824){
// This is a leaf node
result += -0.072727;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002941;
}
}
}
if (event[1] < 1.434955){
if (event[3] < 3.510437){
if (event[0] < 2.326553){
// This is a leaf node
result += 0.000111;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006633;
}
}
else { // if condition is not respected
if (event[1] < -1.435244){
// This is a leaf node
result += 0.046664;
}
else { // if condition is not respected
// This is a leaf node
result += -0.064791;
}
}
}
else { // if condition is not respected
if (event[1] < 1.443307){
if (event[0] < 1.422001){
// This is a leaf node
result += -0.026306;
}
else { // if condition is not respected
// This is a leaf node
result += 0.042294;
}
}
else { // if condition is not respected
if (event[4] < 3.129455){
// This is a leaf node
result += -0.002794;
}
else { // if condition is not respected
// This is a leaf node
result += 0.074546;
}
}
}
if (event[1] < 1.434955){
if (event[1] < 1.430497){
if (event[4] < -2.844356){
// This is a leaf node
result += -0.013533;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000045;
}
}
else { // if condition is not respected
if (event[0] < 1.787396){
// This is a leaf node
result += 0.030905;
}
else { // if condition is not respected
// This is a leaf node
result += -0.085537;
}
}
}
else { // if condition is not respected
if (event[2] < -2.618376){
if (event[3] < 0.967049){
// This is a leaf node
result += -0.053550;
}
else { // if condition is not respected
// This is a leaf node
result += 0.030089;
}
}
else { // if condition is not respected
if (event[2] < -2.572128){
// This is a leaf node
result += 0.088482;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002632;
}
}
}
if (event[3] < 3.510437){
if (event[1] < 1.434955){
if (event[3] < -0.202299){
// This is a leaf node
result += -0.000711;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000580;
}
}
else { // if condition is not respected
if (event[3] < 2.923465){
// This is a leaf node
result += -0.002521;
}
else { // if condition is not respected
// This is a leaf node
result += 0.060132;
}
}
}
else { // if condition is not respected
if (event[1] < -1.435244){
if (event[3] < 3.716276){
// This is a leaf node
result += -0.043602;
}
else { // if condition is not respected
// This is a leaf node
result += 0.112850;
}
}
else { // if condition is not respected
if (event[0] < -0.929110){
// This is a leaf node
result += -0.130393;
}
else { // if condition is not respected
// This is a leaf node
result += -0.042540;
}
}
}
if (event[2] < -2.622372){
if (event[3] < 0.665693){
if (event[3] < 0.320590){
// This is a leaf node
result += -0.010207;
}
else { // if condition is not respected
// This is a leaf node
result += -0.060055;
}
}
else { // if condition is not respected
if (event[1] < 0.982395){
// This is a leaf node
result += 0.023304;
}
else { // if condition is not respected
// This is a leaf node
result += -0.025311;
}
}
}
else { // if condition is not respected
if (event[2] < -2.548010){
if (event[0] < 0.514031){
// This is a leaf node
result += 0.038726;
}
else { // if condition is not respected
// This is a leaf node
result += -0.012033;
}
}
else { // if condition is not respected
if (event[2] < -2.531265){
// This is a leaf node
result += -0.052290;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000108;
}
}
}
if (event[3] < 3.665319){
if (event[2] < 2.406896){
if (event[2] < 2.247168){
// This is a leaf node
result += -0.000120;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012083;
}
}
else { // if condition is not respected
if (event[1] < -0.498264){
// This is a leaf node
result += 0.006267;
}
else { // if condition is not respected
// This is a leaf node
result += -0.012645;
}
}
}
else { // if condition is not respected
if (event[1] < -1.266439){
// This is a leaf node
result += 0.067049;
}
else { // if condition is not respected
if (event[3] < 3.797373){
// This is a leaf node
result += -0.125511;
}
else { // if condition is not respected
// This is a leaf node
result += -0.032218;
}
}
}
if (event[1] < 1.434816){
if (event[1] < 1.434327){
if (event[1] < 0.666113){
// This is a leaf node
result += -0.000245;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001274;
}
}
else { // if condition is not respected
if (event[3] < -0.846518){
// This is a leaf node
result += -0.020104;
}
else { // if condition is not respected
// This is a leaf node
result += 0.094080;
}
}
}
else { // if condition is not respected
if (event[4] < 3.129455){
if (event[1] < 1.443307){
// This is a leaf node
result += -0.018468;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001961;
}
}
else { // if condition is not respected
if (event[1] < 1.562093){
// This is a leaf node
result += -0.025916;
}
else { // if condition is not respected
// This is a leaf node
result += 0.087679;
}
}
}
if (event[2] < -2.622372){
if (event[1] < 0.376648){
if (event[1] < -0.348773){
// This is a leaf node
result += -0.013510;
}
else { // if condition is not respected
// This is a leaf node
result += 0.021359;
}
}
else { // if condition is not respected
if (event[3] < 1.023571){
// This is a leaf node
result += -0.035008;
}
else { // if condition is not respected
// This is a leaf node
result += 0.014462;
}
}
}
else { // if condition is not respected
if (event[2] < -2.549614){
if (event[4] < 1.932237){
// This is a leaf node
result += 0.022973;
}
else { // if condition is not respected
// This is a leaf node
result += -0.090717;
}
}
else { // if condition is not respected
if (event[2] < -2.531265){
// This is a leaf node
result += -0.040678;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000073;
}
}
}
if (event[3] < 3.665319){
if (event[3] < -0.202299){
if (event[3] < -0.202442){
// This is a leaf node
result += -0.000752;
}
else { // if condition is not respected
// This is a leaf node
result += -0.119821;
}
}
else { // if condition is not respected
if (event[3] < 0.323339){
// This is a leaf node
result += 0.002422;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000715;
}
}
}
else { // if condition is not respected
if (event[1] < -1.266439){
// This is a leaf node
result += 0.062717;
}
else { // if condition is not respected
if (event[3] < 3.797373){
// This is a leaf node
result += -0.116594;
}
else { // if condition is not respected
// This is a leaf node
result += -0.029497;
}
}
}
if (event[0] < 2.326553){
if (event[0] < 2.325281){
if (event[2] < 2.406896){
// This is a leaf node
result += 0.000021;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006681;
}
}
else { // if condition is not respected
if (event[4] < 0.328957){
// This is a leaf node
result += 0.150290;
}
else { // if condition is not respected
// This is a leaf node
result += 0.022273;
}
}
}
else { // if condition is not respected
if (event[0] < 2.368357){
if (event[4] < -0.612402){
// This is a leaf node
result += -0.065038;
}
else { // if condition is not respected
// This is a leaf node
result += -0.013619;
}
}
else { // if condition is not respected
if (event[2] < 1.659111){
// This is a leaf node
result += -0.005093;
}
else { // if condition is not respected
// This is a leaf node
result += 0.032190;
}
}
}
if (event[3] < -2.946279){
if (event[1] < -1.227662){
if (event[4] < 1.348020){
// This is a leaf node
result += -0.083054;
}
else { // if condition is not respected
// This is a leaf node
result += 0.018293;
}
}
else { // if condition is not respected
if (event[1] < -1.009945){
// This is a leaf node
result += 0.064290;
}
else { // if condition is not respected
// This is a leaf node
result += -0.012997;
}
}
}
else { // if condition is not respected
if (event[3] < -2.924001){
if (event[4] < 1.008340){
// This is a leaf node
result += 0.076617;
}
else { // if condition is not respected
// This is a leaf node
result += -0.015077;
}
}
else { // if condition is not respected
if (event[1] < -0.688016){
// This is a leaf node
result += 0.000910;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000378;
}
}
}
if (event[3] < 3.492048){
if (event[3] < 3.262835){
if (event[3] < 3.229567){
// This is a leaf node
result += -0.000066;
}
else { // if condition is not respected
// This is a leaf node
result += -0.075424;
}
}
else { // if condition is not respected
if (event[1] < -1.346547){
// This is a leaf node
result += -0.088927;
}
else { // if condition is not respected
// This is a leaf node
result += 0.051210;
}
}
}
else { // if condition is not respected
if (event[0] < -0.929110){
if (event[2] < 0.259773){
// This is a leaf node
result += -0.138565;
}
else { // if condition is not respected
// This is a leaf node
result += -0.020176;
}
}
else { // if condition is not respected
if (event[0] < -0.697585){
// This is a leaf node
result += 0.066722;
}
else { // if condition is not respected
// This is a leaf node
result += -0.032241;
}
}
}
if (event[1] < 2.086814){
if (event[1] < 2.076918){
if (event[1] < 2.068745){
// This is a leaf node
result += 0.000012;
}
else { // if condition is not respected
// This is a leaf node
result += -0.036315;
}
}
else { // if condition is not respected
if (event[2] < -1.574700){
// This is a leaf node
result += -0.106777;
}
else { // if condition is not respected
// This is a leaf node
result += 0.033288;
}
}
}
else { // if condition is not respected
if (event[2] < 1.454877){
if (event[2] < 1.349896){
// This is a leaf node
result += -0.002861;
}
else { // if condition is not respected
// This is a leaf node
result += 0.042294;
}
}
else { // if condition is not respected
if (event[0] < -1.010630){
// This is a leaf node
result += 0.017651;
}
else { // if condition is not respected
// This is a leaf node
result += -0.035005;
}
}
}
if (event[1] < -3.336838){
if (event[3] < 0.747019){
if (event[2] < -0.938920){
// This is a leaf node
result += 0.027093;
}
else { // if condition is not respected
// This is a leaf node
result += -0.068845;
}
}
else { // if condition is not respected
if (event[1] < -3.402231){
// This is a leaf node
result += 0.082863;
}
else { // if condition is not respected
// This is a leaf node
result += -0.099274;
}
}
}
else { // if condition is not respected
if (event[1] < -2.316840){
if (event[3] < -0.782308){
// This is a leaf node
result += -0.009890;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009377;
}
}
else { // if condition is not respected
if (event[2] < -2.622372){
// This is a leaf node
result += -0.008442;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000063;
}
}
}
if (event[0] < 2.326553){
if (event[0] < 2.325281){
if (event[0] < 2.141423){
// This is a leaf node
result += -0.000048;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007151;
}
}
else { // if condition is not respected
if (event[4] < 0.328957){
// This is a leaf node
result += 0.139594;
}
else { // if condition is not respected
// This is a leaf node
result += 0.021064;
}
}
}
else { // if condition is not respected
if (event[2] < -2.342321){
if (event[4] < -0.863538){
// This is a leaf node
result += 0.068221;
}
else { // if condition is not respected
// This is a leaf node
result += -0.101722;
}
}
else { // if condition is not respected
if (event[0] < 2.368357){
// This is a leaf node
result += -0.024858;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002110;
}
}
}
if (event[0] < -0.586003){
if (event[0] < -0.587047){
if (event[4] < 3.025721){
// This is a leaf node
result += -0.000772;
}
else { // if condition is not respected
// This is a leaf node
result += -0.038306;
}
}
else { // if condition is not respected
if (event[0] < -0.586787){
// This is a leaf node
result += -0.099608;
}
else { // if condition is not respected
// This is a leaf node
result += -0.019677;
}
}
}
else { // if condition is not respected
if (event[4] < -2.843313){
if (event[4] < -3.028109){
// This is a leaf node
result += -0.000812;
}
else { // if condition is not respected
// This is a leaf node
result += -0.032870;
}
}
else { // if condition is not respected
if (event[0] < -0.560185){
// This is a leaf node
result += 0.006801;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000229;
}
}
}
if (event[3] < -0.202299){
if (event[3] < -0.202404){
if (event[4] < 2.275642){
// This is a leaf node
result += -0.000521;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010005;
}
}
else { // if condition is not respected
if (event[1] < 1.019669){
// This is a leaf node
result += -0.148798;
}
else { // if condition is not respected
// This is a leaf node
result += -0.014715;
}
}
}
else { // if condition is not respected
if (event[3] < 0.323339){
if (event[4] < -2.842286){
// This is a leaf node
result += -0.033399;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002301;
}
}
else { // if condition is not respected
if (event[1] < -0.218616){
// This is a leaf node
result += 0.000734;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001574;
}
}
}
if (event[3] < 3.665319){
if (event[3] < -2.946279){
if (event[4] < 1.838505){
// This is a leaf node
result += -0.015689;
}
else { // if condition is not respected
// This is a leaf node
result += 0.082601;
}
}
else { // if condition is not respected
if (event[3] < -2.924001){
// This is a leaf node
result += 0.056969;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000019;
}
}
}
else { // if condition is not respected
if (event[1] < -1.266439){
// This is a leaf node
result += 0.060765;
}
else { // if condition is not respected
if (event[0] < -0.062568){
// This is a leaf node
result += -0.117477;
}
else { // if condition is not respected
// This is a leaf node
result += -0.028489;
}
}
}
if (event[2] < 2.478288){
if (event[2] < 2.476540){
if (event[2] < 2.247168){
// This is a leaf node
result += -0.000037;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006915;
}
}
else { // if condition is not respected
if (event[1] < -0.220568){
// This is a leaf node
result += -0.022128;
}
else { // if condition is not respected
// This is a leaf node
result += 0.154405;
}
}
}
else { // if condition is not respected
if (event[3] < 1.558936){
if (event[0] < 1.229889){
// This is a leaf node
result += -0.011106;
}
else { // if condition is not respected
// This is a leaf node
result += 0.019414;
}
}
else { // if condition is not respected
if (event[2] < 2.560512){
// This is a leaf node
result += -0.037595;
}
else { // if condition is not respected
// This is a leaf node
result += 0.043616;
}
}
}
if (event[0] < 2.244170){
if (event[0] < 2.141423){
if (event[0] < 2.140868){
// This is a leaf node
result += -0.000024;
}
else { // if condition is not respected
// This is a leaf node
result += -0.093302;
}
}
else { // if condition is not respected
if (event[4] < -1.560337){
// This is a leaf node
result += 0.060302;
}
else { // if condition is not respected
// This is a leaf node
result += 0.010169;
}
}
}
else { // if condition is not respected
if (event[2] < -2.342321){
if (event[4] < -0.863538){
// This is a leaf node
result += 0.063664;
}
else { // if condition is not respected
// This is a leaf node
result += -0.095115;
}
}
else { // if condition is not respected
if (event[2] < -2.017441){
// This is a leaf node
result += 0.047302;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004534;
}
}
}
if (event[1] < -3.336838){
if (event[3] < 0.747019){
if (event[0] < -0.542882){
// This is a leaf node
result += -0.107758;
}
else { // if condition is not respected
// This is a leaf node
result += -0.020219;
}
}
else { // if condition is not respected
if (event[1] < -3.402231){
// This is a leaf node
result += 0.075466;
}
else { // if condition is not respected
// This is a leaf node
result += -0.092879;
}
}
}
else { // if condition is not respected
if (event[1] < -3.324472){
if (event[4] < 0.177580){
// This is a leaf node
result += 0.021687;
}
else { // if condition is not respected
// This is a leaf node
result += 0.119743;
}
}
else { // if condition is not respected
if (event[1] < 1.434955){
// This is a leaf node
result += 0.000119;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001709;
}
}
}
if (event[4] < 1.671161){
if (event[4] < 1.578457){
if (event[2] < -2.622372){
// This is a leaf node
result += -0.008926;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000040;
}
}
else { // if condition is not respected
if (event[4] < 1.614472){
// This is a leaf node
result += 0.018957;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000798;
}
}
}
else { // if condition is not respected
if (event[4] < 1.682022){
if (event[0] < 2.423629){
// This is a leaf node
result += -0.030550;
}
else { // if condition is not respected
// This is a leaf node
result += 0.111757;
}
}
else { // if condition is not respected
if (event[1] < -1.034734){
// This is a leaf node
result += -0.010568;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000047;
}
}
}
if (event[2] < -3.749223){
if (event[2] < -3.845556){
if (event[1] < 0.030143){
// This is a leaf node
result += 0.035375;
}
else { // if condition is not respected
// This is a leaf node
result += -0.074023;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.136169;
}
}
else { // if condition is not respected
if (event[2] < -2.622372){
if (event[3] < 0.665693){
// This is a leaf node
result += -0.014619;
}
else { // if condition is not respected
// This is a leaf node
result += 0.014654;
}
}
else { // if condition is not respected
if (event[2] < -2.606295){
// This is a leaf node
result += 0.039158;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000003;
}
}
}
if (event[3] < -1.971181){
if (event[1] < -2.639684){
if (event[0] < 0.120414){
// This is a leaf node
result += -0.112329;
}
else { // if condition is not respected
// This is a leaf node
result += -0.021158;
}
}
else { // if condition is not respected
if (event[3] < -2.086916){
// This is a leaf node
result += 0.000497;
}
else { // if condition is not respected
// This is a leaf node
result += 0.011806;
}
}
}
else { // if condition is not respected
if (event[3] < -1.970558){
if (event[0] < -0.643196){
// This is a leaf node
result += -0.017912;
}
else { // if condition is not respected
// This is a leaf node
result += -0.149805;
}
}
else { // if condition is not respected
if (event[3] < -0.202848){
// This is a leaf node
result += -0.000755;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000367;
}
}
}
if (event[3] < 3.492048){
if (event[3] < 3.262835){
if (event[3] < 3.229567){
// This is a leaf node
result += -0.000016;
}
else { // if condition is not respected
// This is a leaf node
result += -0.068985;
}
}
else { // if condition is not respected
if (event[1] < -1.346547){
// This is a leaf node
result += -0.082337;
}
else { // if condition is not respected
// This is a leaf node
result += 0.046313;
}
}
}
else { // if condition is not respected
if (event[0] < -0.929110){
if (event[2] < 0.259773){
// This is a leaf node
result += -0.128277;
}
else { // if condition is not respected
// This is a leaf node
result += -0.016667;
}
}
else { // if condition is not respected
if (event[0] < 0.995751){
// This is a leaf node
result += 0.005616;
}
else { // if condition is not respected
// This is a leaf node
result += -0.079797;
}
}
}
if (event[0] < -0.608389){
if (event[0] < -0.775190){
if (event[0] < -0.778606){
// This is a leaf node
result += -0.000230;
}
else { // if condition is not respected
// This is a leaf node
result += 0.035800;
}
}
else { // if condition is not respected
if (event[4] < -1.775700){
// This is a leaf node
result += -0.023613;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002919;
}
}
}
else { // if condition is not respected
if (event[0] < -0.603108){
if (event[0] < -0.603373){
// This is a leaf node
result += 0.010452;
}
else { // if condition is not respected
// This is a leaf node
result += 0.074682;
}
}
else { // if condition is not respected
if (event[0] < -0.602978){
// This is a leaf node
result += -0.130283;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000239;
}
}
}
if (event[4] < 0.572174){
if (event[4] < 0.571559){
if (event[4] < -0.370595){
// This is a leaf node
result += -0.000545;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001040;
}
}
else { // if condition is not respected
if (event[1] < 1.750692){
// This is a leaf node
result += 0.074306;
}
else { // if condition is not respected
// This is a leaf node
result += -0.065695;
}
}
}
else { // if condition is not respected
if (event[2] < -1.873970){
if (event[3] < -0.750097){
// This is a leaf node
result += 0.026081;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000755;
}
}
else { // if condition is not respected
if (event[1] < -3.650717){
// This is a leaf node
result += -0.120983;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000960;
}
}
}
if (event[1] < -0.688016){
if (event[4] < 1.859916){
if (event[4] < 1.857151){
// This is a leaf node
result += 0.001066;
}
else { // if condition is not respected
// This is a leaf node
result += 0.103300;
}
}
else { // if condition is not respected
if (event[4] < 1.864946){
// This is a leaf node
result += -0.077907;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007676;
}
}
}
else { // if condition is not respected
if (event[2] < 2.387063){
if (event[2] < 2.305496){
// This is a leaf node
result += -0.000246;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020128;
}
}
else { // if condition is not respected
if (event[2] < 2.391339){
// This is a leaf node
result += -0.120690;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007306;
}
}
}
if (event[1] < 2.086769){
if (event[1] < 2.076918){
if (event[1] < 2.068745){
// This is a leaf node
result += 0.000050;
}
else { // if condition is not respected
// This is a leaf node
result += -0.032416;
}
}
else { // if condition is not respected
if (event[2] < -1.574700){
// This is a leaf node
result += -0.099553;
}
else { // if condition is not respected
// This is a leaf node
result += 0.030537;
}
}
}
else { // if condition is not respected
if (event[2] < 1.454877){
if (event[2] < 1.349896){
// This is a leaf node
result += -0.002306;
}
else { // if condition is not respected
// This is a leaf node
result += 0.038467;
}
}
else { // if condition is not respected
if (event[0] < -1.010630){
// This is a leaf node
result += 0.016551;
}
else { // if condition is not respected
// This is a leaf node
result += -0.031183;
}
}
}
if (event[3] < -2.946279){
if (event[1] < -1.227662){
if (event[3] < -3.129026){
// This is a leaf node
result += -0.033646;
}
else { // if condition is not respected
// This is a leaf node
result += -0.099778;
}
}
else { // if condition is not respected
if (event[1] < -1.009945){
// This is a leaf node
result += 0.059647;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010348;
}
}
}
else { // if condition is not respected
if (event[3] < -2.924001){
if (event[3] < -2.927862){
// This is a leaf node
result += 0.026005;
}
else { // if condition is not respected
// This is a leaf node
result += 0.106873;
}
}
else { // if condition is not respected
if (event[3] < -1.971181){
// This is a leaf node
result += 0.003326;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000076;
}
}
}
if (event[3] < 3.665319){
if (event[0] < 2.244170){
if (event[0] < 2.141204){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.011917;
}
}
else { // if condition is not respected
if (event[1] < 0.382041){
// This is a leaf node
result += 0.000249;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011405;
}
}
}
else { // if condition is not respected
if (event[1] < -1.266439){
// This is a leaf node
result += 0.057998;
}
else { // if condition is not respected
if (event[3] < 3.797373){
// This is a leaf node
result += -0.101587;
}
else { // if condition is not respected
// This is a leaf node
result += -0.017746;
}
}
}
if (event[2] < -0.196310){
if (event[2] < -0.202890){
if (event[2] < -0.206643){
// This is a leaf node
result += -0.000490;
}
else { // if condition is not respected
// This is a leaf node
result += 0.015244;
}
}
else { // if condition is not respected
if (event[3] < -1.840182){
// This is a leaf node
result += -0.072205;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011977;
}
}
}
else { // if condition is not respected
if (event[2] < -0.193327){
if (event[2] < -0.193925){
// This is a leaf node
result += 0.011829;
}
else { // if condition is not respected
// This is a leaf node
result += 0.060347;
}
}
else { // if condition is not respected
if (event[2] < -0.193299){
// This is a leaf node
result += -0.119899;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000326;
}
}
}
if (event[2] < -3.749223){
if (event[2] < -3.845556){
if (event[1] < 0.030143){
// This is a leaf node
result += 0.032533;
}
else { // if condition is not respected
// This is a leaf node
result += -0.068965;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.125748;
}
}
else { // if condition is not respected
if (event[2] < -2.304605){
if (event[3] < 0.664962){
// This is a leaf node
result += -0.008351;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007197;
}
}
else { // if condition is not respected
if (event[2] < -2.302579){
// This is a leaf node
result += 0.099752;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000029;
}
}
}
if (event[3] < 0.326655){
if (event[3] < -0.202848){
if (event[3] < -0.203357){
// This is a leaf node
result += -0.000469;
}
else { // if condition is not respected
// This is a leaf node
result += -0.058399;
}
}
else { // if condition is not respected
if (event[3] < 0.325666){
// This is a leaf node
result += 0.001915;
}
else { // if condition is not respected
// This is a leaf node
result += 0.042539;
}
}
}
else { // if condition is not respected
if (event[3] < 0.383059){
if (event[2] < 1.513044){
// This is a leaf node
result += -0.006939;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020268;
}
}
else { // if condition is not respected
if (event[1] < -0.218616){
// This is a leaf node
result += 0.001012;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001223;
}
}
}
if (event[0] < 3.515198){
if (event[0] < 3.430337){
if (event[0] < 3.332963){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.050318;
}
}
else { // if condition is not respected
if (event[0] < 3.473993){
// This is a leaf node
result += 0.156832;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001820;
}
}
}
else { // if condition is not respected
if (event[2] < 0.882167){
if (event[2] < -1.141997){
// This is a leaf node
result += 0.058895;
}
else { // if condition is not respected
// This is a leaf node
result += -0.081145;
}
}
else { // if condition is not respected
if (event[0] < 3.774580){
// This is a leaf node
result += 0.133974;
}
else { // if condition is not respected
// This is a leaf node
result += 0.023340;
}
}
}
if (event[1] < -3.244704){
if (event[1] < -3.256130){
if (event[0] < -0.912125){
// This is a leaf node
result += -0.064719;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001211;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.119706;
}
}
else { // if condition is not respected
if (event[1] < -3.217031){
if (event[4] < 0.073783){
// This is a leaf node
result += 0.102543;
}
else { // if condition is not respected
// This is a leaf node
result += -0.021721;
}
}
else { // if condition is not respected
if (event[1] < -3.204350){
// This is a leaf node
result += -0.094024;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000003;
}
}
}
if (event[4] < 2.624323){
if (event[4] < 2.623344){
if (event[3] < 3.665319){
// This is a leaf node
result += 0.000026;
}
else { // if condition is not respected
// This is a leaf node
result += -0.042514;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.120573;
}
}
else { // if condition is not respected
if (event[3] < -0.981270){
if (event[2] < -0.992802){
// This is a leaf node
result += 0.039296;
}
else { // if condition is not respected
// This is a leaf node
result += -0.049381;
}
}
else { // if condition is not respected
if (event[4] < 2.655724){
// This is a leaf node
result += -0.042759;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003010;
}
}
}
if (event[0] < -0.153146){
if (event[0] < -0.169503){
if (event[1] < 3.383786){
// This is a leaf node
result += -0.000313;
}
else { // if condition is not respected
// This is a leaf node
result += 0.050415;
}
}
else { // if condition is not respected
if (event[0] < -0.168293){
// This is a leaf node
result += -0.049490;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010462;
}
}
}
else { // if condition is not respected
if (event[0] < -0.139817){
if (event[4] < -1.741237){
// This is a leaf node
result += -0.046795;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012886;
}
}
else { // if condition is not respected
if (event[1] < 3.150256){
// This is a leaf node
result += 0.000307;
}
else { // if condition is not respected
// This is a leaf node
result += -0.031191;
}
}
}
if (event[2] < -3.319989){
if (event[1] < 0.103900){
if (event[0] < -0.823253){
// This is a leaf node
result += -0.019986;
}
else { // if condition is not respected
// This is a leaf node
result += 0.067420;
}
}
else { // if condition is not respected
if (event[2] < -3.488229){
// This is a leaf node
result += -0.059449;
}
else { // if condition is not respected
// This is a leaf node
result += 0.031693;
}
}
}
else { // if condition is not respected
if (event[2] < -3.020078){
if (event[1] < 1.043653){
// This is a leaf node
result += -0.004847;
}
else { // if condition is not respected
// This is a leaf node
result += -0.097729;
}
}
else { // if condition is not respected
if (event[2] < -2.995612){
// This is a leaf node
result += 0.057238;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000005;
}
}
}
if (event[2] < 2.406896){
if (event[2] < 2.402218){
if (event[2] < 2.247168){
// This is a leaf node
result += -0.000009;
}
else { // if condition is not respected
// This is a leaf node
result += 0.008164;
}
}
else { // if condition is not respected
if (event[3] < -0.656974){
// This is a leaf node
result += -0.009584;
}
else { // if condition is not respected
// This is a leaf node
result += 0.080983;
}
}
}
else { // if condition is not respected
if (event[1] < -1.258240){
if (event[0] < 0.967631){
// This is a leaf node
result += 0.007121;
}
else { // if condition is not respected
// This is a leaf node
result += 0.069111;
}
}
else { // if condition is not respected
if (event[2] < 2.423539){
// This is a leaf node
result += -0.040564;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005590;
}
}
}
if (event[1] < -3.336838){
if (event[3] < 0.747019){
if (event[2] < -0.938920){
// This is a leaf node
result += 0.029809;
}
else { // if condition is not respected
// This is a leaf node
result += -0.056762;
}
}
else { // if condition is not respected
if (event[1] < -3.402231){
// This is a leaf node
result += 0.070376;
}
else { // if condition is not respected
// This is a leaf node
result += -0.086234;
}
}
}
else { // if condition is not respected
if (event[1] < -3.324472){
if (event[0] < 0.473791){
// This is a leaf node
result += 0.117378;
}
else { // if condition is not respected
// This is a leaf node
result += 0.017232;
}
}
else { // if condition is not respected
if (event[1] < -2.316840){
// This is a leaf node
result += 0.004580;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000043;
}
}
}
if (event[3] < 2.319761){
if (event[3] < 2.309603){
if (event[3] < 2.152436){
// This is a leaf node
result += -0.000025;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009041;
}
}
else { // if condition is not respected
if (event[1] < 0.145666){
// This is a leaf node
result += 0.103685;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011625;
}
}
}
else { // if condition is not respected
if (event[1] < -0.908032){
if (event[4] < 0.380298){
// This is a leaf node
result += 0.031413;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005064;
}
}
else { // if condition is not respected
if (event[3] < 2.325979){
// This is a leaf node
result += -0.087280;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007484;
}
}
}
if (event[3] < 0.323457){
if (event[3] < -0.202848){
if (event[3] < -0.203357){
// This is a leaf node
result += -0.000411;
}
else { // if condition is not respected
// This is a leaf node
result += -0.052917;
}
}
else { // if condition is not respected
if (event[1] < -2.996860){
// This is a leaf node
result += -0.040181;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001875;
}
}
}
else { // if condition is not respected
if (event[3] < 0.325290){
if (event[2] < -0.565737){
// This is a leaf node
result += -0.062465;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010603;
}
}
else { // if condition is not respected
if (event[3] < 0.326655){
// This is a leaf node
result += 0.030278;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000521;
}
}
}
if (event[2] < 2.092766){
if (event[2] < 2.090705){
if (event[2] < 1.888311){
// This is a leaf node
result += -0.000031;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006448;
}
}
else { // if condition is not respected
if (event[1] < 0.184759){
// This is a leaf node
result += 0.135311;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001088;
}
}
}
else { // if condition is not respected
if (event[0] < -2.382587){
if (event[2] < 2.738124){
// This is a leaf node
result += 0.092003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.008037;
}
}
else { // if condition is not respected
if (event[2] < 2.102926){
// This is a leaf node
result += -0.037111;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002659;
}
}
}
if (event[0] < -0.585980){
if (event[0] < -0.587047){
if (event[1] < 3.035493){
// This is a leaf node
result += -0.000641;
}
else { // if condition is not respected
// This is a leaf node
result += 0.037128;
}
}
else { // if condition is not respected
if (event[0] < -0.586787){
// This is a leaf node
result += -0.090568;
}
else { // if condition is not respected
// This is a leaf node
result += -0.017176;
}
}
}
else { // if condition is not respected
if (event[0] < -0.585862){
if (event[1] < 0.657637){
// This is a leaf node
result += 0.114136;
}
else { // if condition is not respected
// This is a leaf node
result += 0.022041;
}
}
else { // if condition is not respected
if (event[4] < -2.843313){
// This is a leaf node
result += -0.013111;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000271;
}
}
}
if (event[1] < 1.986864){
if (event[1] < 1.971039){
if (event[1] < 1.963674){
// This is a leaf node
result += 0.000057;
}
else { // if condition is not respected
// This is a leaf node
result += -0.031308;
}
}
else { // if condition is not respected
if (event[2] < 1.435596){
// This is a leaf node
result += 0.026649;
}
else { // if condition is not respected
// This is a leaf node
result += -0.055354;
}
}
}
else { // if condition is not respected
if (event[2] < 1.458617){
if (event[3] < 1.794408){
// This is a leaf node
result += -0.002213;
}
else { // if condition is not respected
// This is a leaf node
result += 0.023592;
}
}
else { // if condition is not respected
if (event[2] < 1.473362){
// This is a leaf node
result += -0.100978;
}
else { // if condition is not respected
// This is a leaf node
result += -0.016083;
}
}
}
if (event[0] < 2.237703){
if (event[0] < 2.218218){
if (event[3] < -2.946279){
// This is a leaf node
result += -0.011287;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000041;
}
}
else { // if condition is not respected
if (event[1] < 0.456722){
// This is a leaf node
result += 0.046734;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000788;
}
}
}
else { // if condition is not respected
if (event[4] < 2.545454){
if (event[4] < -0.634073){
// This is a leaf node
result += -0.012816;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000452;
}
}
else { // if condition is not respected
if (event[1] < 0.352303){
// This is a leaf node
result += -0.135361;
}
else { // if condition is not respected
// This is a leaf node
result += 0.042561;
}
}
}
if (event[3] < -1.971181){
if (event[2] < -2.023206){
if (event[4] < -0.380040){
// This is a leaf node
result += -0.009553;
}
else { // if condition is not respected
// This is a leaf node
result += 0.055970;
}
}
else { // if condition is not respected
if (event[2] < -1.557590){
// This is a leaf node
result += -0.023801;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002982;
}
}
}
else { // if condition is not respected
if (event[3] < -1.970558){
if (event[0] < -0.643196){
// This is a leaf node
result += -0.016545;
}
else { // if condition is not respected
// This is a leaf node
result += -0.139128;
}
}
else { // if condition is not respected
if (event[3] < -1.929478){
// This is a leaf node
result += -0.009476;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000038;
}
}
}
if (event[4] < 0.572174){
if (event[4] < 0.571559){
if (event[0] < -1.243195){
// This is a leaf node
result += 0.002404;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000031;
}
}
else { // if condition is not respected
if (event[3] < 0.374560){
// This is a leaf node
result += 0.084126;
}
else { // if condition is not respected
// This is a leaf node
result += 0.008787;
}
}
}
else { // if condition is not respected
if (event[2] < -1.675402){
if (event[2] < -1.676901){
// This is a leaf node
result += 0.004215;
}
else { // if condition is not respected
// This is a leaf node
result += 0.148238;
}
}
else { // if condition is not respected
if (event[2] < -1.659738){
// This is a leaf node
result += -0.041176;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000835;
}
}
}
if (event[0] < -0.608184){
if (event[0] < -0.664270){
if (event[0] < -0.666318){
// This is a leaf node
result += -0.000392;
}
else { // if condition is not respected
// This is a leaf node
result += 0.032488;
}
}
else { // if condition is not respected
if (event[0] < -0.659027){
// This is a leaf node
result += -0.025469;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003420;
}
}
}
else { // if condition is not respected
if (event[0] < -0.603108){
if (event[0] < -0.603144){
// This is a leaf node
result += 0.011713;
}
else { // if condition is not respected
// This is a leaf node
result += 0.129298;
}
}
else { // if condition is not respected
if (event[0] < -0.602978){
// This is a leaf node
result += -0.119875;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000215;
}
}
}
if (event[2] < -0.196334){
if (event[2] < -0.433119){
if (event[2] < -0.436768){
// This is a leaf node
result += -0.000006;
}
else { // if condition is not respected
// This is a leaf node
result += 0.018018;
}
}
else { // if condition is not respected
if (event[2] < -0.365662){
// This is a leaf node
result += -0.007217;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000630;
}
}
}
else { // if condition is not respected
if (event[2] < -0.193327){
if (event[0] < -0.867893){
// This is a leaf node
result += 0.055136;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009450;
}
}
else { // if condition is not respected
if (event[4] < -2.218043){
// This is a leaf node
result += -0.006662;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000388;
}
}
}
if (event[2] < 4.023448){
if (event[2] < 3.986183){
if (event[2] < -2.304283){
// This is a leaf node
result += -0.003813;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000039;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.117559;
}
}
else { // if condition is not respected
if (event[3] < -0.613557){
// This is a leaf node
result += -0.020547;
}
else { // if condition is not respected
// This is a leaf node
result += 0.095123;
}
}
if (event[4] < 3.028272){
if (event[4] < 3.020253){
if (event[4] < 2.939700){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.026254;
}
}
else { // if condition is not respected
if (event[0] < -0.989813){
// This is a leaf node
result += 0.022911;
}
else { // if condition is not respected
// This is a leaf node
result += 0.144001;
}
}
}
else { // if condition is not respected
if (event[1] < 0.997171){
if (event[4] < 3.037205){
// This is a leaf node
result += -0.116411;
}
else { // if condition is not respected
// This is a leaf node
result += -0.019500;
}
}
else { // if condition is not respected
if (event[1] < 1.401275){
// This is a leaf node
result += 0.083248;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009308;
}
}
}
if (event[0] < 4.111604){
if (event[0] < 3.689817){
if (event[0] < 3.651816){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += 0.119304;
}
}
else { // if condition is not respected
if (event[4] < 0.450543){
// This is a leaf node
result += -0.110382;
}
else { // if condition is not respected
// This is a leaf node
result += 0.013253;
}
}
}
else { // if condition is not respected
if (event[0] < 4.301860){
// This is a leaf node
result += 0.111406;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005892;
}
}
if (event[2] < -3.749223){
if (event[2] < -3.845556){
if (event[0] < 0.420350){
// This is a leaf node
result += -0.047645;
}
else { // if condition is not respected
// This is a leaf node
result += 0.045746;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.116637;
}
}
else { // if condition is not respected
if (event[2] < -2.622372){
if (event[1] < 0.355268){
// This is a leaf node
result += 0.002496;
}
else { // if condition is not respected
// This is a leaf node
result += -0.020822;
}
}
else { // if condition is not respected
if (event[2] < -2.548010){
// This is a leaf node
result += 0.016989;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000004;
}
}
}
if (event[1] < 0.526281){
if (event[1] < 0.469962){
if (event[2] < -3.296677){
// This is a leaf node
result += 0.031814;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000115;
}
}
else { // if condition is not respected
if (event[1] < 0.477258){
// This is a leaf node
result += -0.024577;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002553;
}
}
}
else { // if condition is not respected
if (event[4] < -0.156816){
if (event[4] < -0.158315){
// This is a leaf node
result += -0.001171;
}
else { // if condition is not respected
// This is a leaf node
result += -0.068115;
}
}
else { // if condition is not respected
if (event[4] < -0.155783){
// This is a leaf node
result += 0.076695;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001973;
}
}
}
if (event[4] < 0.572174){
if (event[4] < 0.571559){
if (event[0] < 0.417635){
// This is a leaf node
result += -0.000278;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001221;
}
}
else { // if condition is not respected
if (event[1] < 1.750692){
// This is a leaf node
result += 0.061823;
}
else { // if condition is not respected
// This is a leaf node
result += -0.065086;
}
}
}
else { // if condition is not respected
if (event[0] < -2.761744){
if (event[4] < 1.053715){
// This is a leaf node
result += -0.051308;
}
else { // if condition is not respected
// This is a leaf node
result += 0.008069;
}
}
else { // if condition is not respected
if (event[0] < -2.649194){
// This is a leaf node
result += 0.053059;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000628;
}
}
}
if (event[1] < 1.434955){
if (event[1] < 0.666113){
if (event[1] < 0.646511){
// This is a leaf node
result += -0.000095;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010231;
}
}
else { // if condition is not respected
if (event[1] < 0.672041){
// This is a leaf node
result += 0.020926;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001135;
}
}
}
else { // if condition is not respected
if (event[4] < -2.090442){
if (event[3] < -0.582020){
// This is a leaf node
result += 0.054738;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000005;
}
}
else { // if condition is not respected
if (event[4] < -2.082158){
// This is a leaf node
result += -0.124906;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001614;
}
}
}
if (event[4] < 2.624323){
if (event[4] < 2.623344){
if (event[3] < 3.665319){
// This is a leaf node
result += 0.000027;
}
else { // if condition is not respected
// This is a leaf node
result += -0.038918;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.113523;
}
}
else { // if condition is not respected
if (event[2] < -0.925358){
if (event[2] < -1.252853){
// This is a leaf node
result += -0.009425;
}
else { // if condition is not respected
// This is a leaf node
result += 0.058241;
}
}
else { // if condition is not respected
if (event[3] < -0.981270){
// This is a leaf node
result += -0.042287;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004818;
}
}
}
if (event[3] < -3.993109){
if (event[2] < 0.081382){
// This is a leaf node
result += 0.042197;
}
else { // if condition is not respected
// This is a leaf node
result += -0.127938;
}
}
else { // if condition is not respected
if (event[3] < -3.681860){
if (event[0] < 0.110352){
// This is a leaf node
result += 0.013940;
}
else { // if condition is not respected
// This is a leaf node
result += 0.115616;
}
}
else { // if condition is not respected
if (event[3] < -2.946279){
// This is a leaf node
result += -0.012265;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000014;
}
}
}
if (event[3] < -1.971181){
if (event[1] < 2.878712){
if (event[1] < -2.639684){
// This is a leaf node
result += -0.060641;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002449;
}
}
else { // if condition is not respected
if (event[1] < 3.425874){
// This is a leaf node
result += 0.113168;
}
else { // if condition is not respected
// This is a leaf node
result += -0.023640;
}
}
}
else { // if condition is not respected
if (event[3] < -1.970558){
if (event[0] < -0.643196){
// This is a leaf node
result += -0.015459;
}
else { // if condition is not respected
// This is a leaf node
result += -0.130219;
}
}
else { // if condition is not respected
if (event[1] < -2.314285){
// This is a leaf node
result += 0.004349;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000101;
}
}
}
if (event[0] < 2.237703){
if (event[0] < 2.218218){
if (event[0] < 2.215728){
// This is a leaf node
result += 0.000026;
}
else { // if condition is not respected
// This is a leaf node
result += -0.042903;
}
}
else { // if condition is not respected
if (event[2] < -1.497804){
// This is a leaf node
result += -0.042221;
}
else { // if condition is not respected
// This is a leaf node
result += 0.033651;
}
}
}
else { // if condition is not respected
if (event[4] < 2.545454){
if (event[4] < 1.430267){
// This is a leaf node
result += -0.004472;
}
else { // if condition is not respected
// This is a leaf node
result += 0.018268;
}
}
else { // if condition is not respected
if (event[1] < 0.352303){
// This is a leaf node
result += -0.123610;
}
else { // if condition is not respected
// This is a leaf node
result += 0.038688;
}
}
}
if (event[0] < 4.111604){
if (event[0] < 3.689817){
if (event[0] < 3.651816){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += 0.112649;
}
}
else { // if condition is not respected
if (event[4] < 0.450543){
// This is a leaf node
result += -0.101289;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012358;
}
}
}
else { // if condition is not respected
if (event[0] < 4.301860){
// This is a leaf node
result += 0.105583;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005734;
}
}
if (event[2] < 0.710045){
if (event[2] < 0.705562){
if (event[0] < 3.515198){
// This is a leaf node
result += -0.000147;
}
else { // if condition is not respected
// This is a leaf node
result += -0.048156;
}
}
else { // if condition is not respected
if (event[3] < -0.953912){
// This is a leaf node
result += 0.013705;
}
else { // if condition is not respected
// This is a leaf node
result += -0.036576;
}
}
}
else { // if condition is not respected
if (event[3] < -2.821038){
if (event[4] < -0.743107){
// This is a leaf node
result += -0.085330;
}
else { // if condition is not respected
// This is a leaf node
result += -0.016217;
}
}
else { // if condition is not respected
if (event[2] < 0.710336){
// This is a leaf node
result += 0.075742;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000701;
}
}
}
if (event[0] < 3.125710){
if (event[0] < 2.997290){
if (event[0] < 2.987079){
// This is a leaf node
result += -0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += 0.080233;
}
}
else { // if condition is not respected
if (event[1] < 0.691081){
// This is a leaf node
result += -0.004371;
}
else { // if condition is not respected
// This is a leaf node
result += -0.071400;
}
}
}
else { // if condition is not respected
if (event[1] < -0.911887){
if (event[0] < 3.423038){
// This is a leaf node
result += 0.122310;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004389;
}
}
else { // if condition is not respected
if (event[1] < 0.825405){
// This is a leaf node
result += -0.022046;
}
else { // if condition is not respected
// This is a leaf node
result += 0.056250;
}
}
}
if (event[2] < 2.092766){
if (event[2] < 2.090705){
if (event[2] < 1.888311){
// This is a leaf node
result += -0.000025;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005752;
}
}
else { // if condition is not respected
if (event[1] < 0.184759){
// This is a leaf node
result += 0.124015;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001166;
}
}
}
else { // if condition is not respected
if (event[0] < -2.382587){
if (event[1] < -0.411801){
// This is a leaf node
result += 0.006923;
}
else { // if condition is not respected
// This is a leaf node
result += 0.095708;
}
}
else { // if condition is not respected
if (event[2] < 2.102926){
// This is a leaf node
result += -0.033491;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002443;
}
}
}
if (event[1] < 4.088089){
if (event[1] < 3.989292){
if (event[1] < 3.939937){
// This is a leaf node
result += 0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += -0.089357;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.098840;
}
}
else { // if condition is not respected
if (event[2] < 0.152884){
// This is a leaf node
result += -0.021856;
}
else { // if condition is not respected
// This is a leaf node
result += -0.080312;
}
}
if (event[2] < 4.023448){
if (event[2] < 3.986183){
if (event[3] < 2.319761){
// This is a leaf node
result += 0.000037;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003576;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.110828;
}
}
else { // if condition is not respected
if (event[4] < -0.480630){
// This is a leaf node
result += -0.022935;
}
else { // if condition is not respected
if (event[3] < -0.255596){
// This is a leaf node
result += 0.022902;
}
else { // if condition is not respected
// This is a leaf node
result += 0.115099;
}
}
}
if (event[0] < 4.111604){
if (event[0] < 3.689817){
if (event[0] < 3.651816){
// This is a leaf node
result += 0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += 0.108753;
}
}
else { // if condition is not respected
if (event[2] < 0.904339){
// This is a leaf node
result += -0.070734;
}
else { // if condition is not respected
// This is a leaf node
result += 0.067747;
}
}
}
else { // if condition is not respected
if (event[0] < 4.301860){
// This is a leaf node
result += 0.102465;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006765;
}
}
if (event[1] < -3.336838){
if (event[3] < 0.747019){
if (event[0] < -0.542882){
// This is a leaf node
result += -0.090857;
}
else { // if condition is not respected
// This is a leaf node
result += -0.013006;
}
}
else { // if condition is not respected
if (event[1] < -3.402231){
// This is a leaf node
result += 0.063880;
}
else { // if condition is not respected
// This is a leaf node
result += -0.081533;
}
}
}
else { // if condition is not respected
if (event[1] < -3.324472){
if (event[1] < -3.331943){
// This is a leaf node
result += 0.017768;
}
else { // if condition is not respected
// This is a leaf node
result += 0.110850;
}
}
else { // if condition is not respected
if (event[1] < -3.315744){
// This is a leaf node
result += -0.090398;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000007;
}
}
}
if (event[1] < -0.687984){
if (event[1] < -0.764778){
if (event[0] < 3.129130){
// This is a leaf node
result += 0.000026;
}
else { // if condition is not respected
// This is a leaf node
result += 0.066383;
}
}
else { // if condition is not respected
if (event[0] < 1.338370){
// This is a leaf node
result += 0.004452;
}
else { // if condition is not respected
// This is a leaf node
result += 0.019949;
}
}
}
else { // if condition is not respected
if (event[1] < -0.681708){
if (event[1] < -0.682050){
// This is a leaf node
result += -0.010683;
}
else { // if condition is not respected
// This is a leaf node
result += -0.097515;
}
}
else { // if condition is not respected
if (event[1] < -0.681589){
// This is a leaf node
result += 0.081302;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000173;
}
}
}
if (event[1] < 2.086658){
if (event[1] < 2.086332){
if (event[1] < 0.526281){
// This is a leaf node
result += -0.000227;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000726;
}
}
else { // if condition is not respected
if (event[2] < 0.223485){
// This is a leaf node
result += 0.019387;
}
else { // if condition is not respected
// This is a leaf node
result += 0.123390;
}
}
}
else { // if condition is not respected
if (event[0] < 1.509778){
if (event[0] < 1.408763){
// This is a leaf node
result += -0.001874;
}
else { // if condition is not respected
// This is a leaf node
result += 0.053907;
}
}
else { // if condition is not respected
if (event[1] < 2.167930){
// This is a leaf node
result += -0.059968;
}
else { // if condition is not respected
// This is a leaf node
result += -0.013842;
}
}
}
if (event[3] < 0.323457){
if (event[3] < -0.202848){
if (event[3] < -0.203357){
// This is a leaf node
result += -0.000377;
}
else { // if condition is not respected
// This is a leaf node
result += -0.048001;
}
}
else { // if condition is not respected
if (event[1] < 2.657802){
// This is a leaf node
result += 0.001543;
}
else { // if condition is not respected
// This is a leaf node
result += 0.026322;
}
}
}
else { // if condition is not respected
if (event[3] < 0.324632){
if (event[1] < -0.616714){
// This is a leaf node
result += -0.085338;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004007;
}
}
else { // if condition is not respected
if (event[3] < 0.324687){
// This is a leaf node
result += 0.102189;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000434;
}
}
}
if (event[4] < 0.550736){
if (event[4] < 0.496927){
if (event[4] < 0.496569){
// This is a leaf node
result += 0.000099;
}
else { // if condition is not respected
// This is a leaf node
result += -0.093336;
}
}
else { // if condition is not respected
if (event[4] < 0.497690){
// This is a leaf node
result += 0.064264;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004812;
}
}
}
else { // if condition is not respected
if (event[0] < -1.546158){
if (event[4] < 0.839070){
// This is a leaf node
result += -0.015151;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000493;
}
}
else { // if condition is not respected
if (event[0] < -1.541794){
// This is a leaf node
result += 0.054501;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000287;
}
}
}
if (event[0] < -1.292493){
if (event[0] < -1.326237){
if (event[4] < 0.351193){
// This is a leaf node
result += 0.002210;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002918;
}
}
else { // if condition is not respected
if (event[3] < -0.765295){
// This is a leaf node
result += -0.006228;
}
else { // if condition is not respected
// This is a leaf node
result += 0.018044;
}
}
}
else { // if condition is not respected
if (event[0] < -0.625203){
if (event[4] < -0.384120){
// This is a leaf node
result += -0.004053;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000133;
}
}
else { // if condition is not respected
if (event[0] < -0.625029){
// This is a leaf node
result += 0.067685;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000187;
}
}
}
if (event[3] < -1.971181){
if (event[4] < 1.266341){
if (event[4] < 0.587407){
// This is a leaf node
result += 0.004186;
}
else { // if condition is not respected
// This is a leaf node
result += -0.013313;
}
}
else { // if condition is not respected
if (event[4] < 2.270741){
// This is a leaf node
result += 0.021197;
}
else { // if condition is not respected
// This is a leaf node
result += -0.033802;
}
}
}
else { // if condition is not respected
if (event[3] < -1.970558){
if (event[0] < -0.643196){
// This is a leaf node
result += -0.014376;
}
else { // if condition is not respected
// This is a leaf node
result += -0.122631;
}
}
else { // if condition is not respected
if (event[2] < -2.304283){
// This is a leaf node
result += -0.004171;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000008;
}
}
}
if (event[4] < -1.884302){
if (event[0] < 1.750216){
if (event[4] < -1.884816){
// This is a leaf node
result += 0.000599;
}
else { // if condition is not respected
// This is a leaf node
result += 0.123663;
}
}
else { // if condition is not respected
if (event[0] < 2.029965){
// This is a leaf node
result += 0.054044;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012145;
}
}
}
else { // if condition is not respected
if (event[4] < -1.842529){
if (event[2] < 1.683675){
// This is a leaf node
result += -0.013452;
}
else { // if condition is not respected
// This is a leaf node
result += 0.037464;
}
}
else { // if condition is not respected
if (event[4] < -1.842215){
// This is a leaf node
result += 0.108634;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000033;
}
}
}
if (event[2] < 2.478055){
if (event[2] < 2.476540){
if (event[2] < 2.247168){
// This is a leaf node
result += -0.000009;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005835;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.125308;
}
}
else { // if condition is not respected
if (event[3] < 1.558936){
if (event[3] < 1.038480){
// This is a leaf node
result += -0.003491;
}
else { // if condition is not respected
// This is a leaf node
result += -0.034213;
}
}
else { // if condition is not respected
if (event[2] < 2.558546){
// This is a leaf node
result += -0.035448;
}
else { // if condition is not respected
// This is a leaf node
result += 0.040259;
}
}
}
if (event[1] < 1.434955){
if (event[1] < 1.430327){
if (event[1] < 1.430128){
// This is a leaf node
result += 0.000089;
}
else { // if condition is not respected
// This is a leaf node
result += -0.096095;
}
}
else { // if condition is not respected
if (event[4] < -1.010733){
// This is a leaf node
result += -0.028169;
}
else { // if condition is not respected
// This is a leaf node
result += 0.032102;
}
}
}
else { // if condition is not respected
if (event[0] < -1.803190){
if (event[0] < -2.009404){
// This is a leaf node
result += -0.001008;
}
else { // if condition is not respected
// This is a leaf node
result += 0.026662;
}
}
else { // if condition is not respected
if (event[3] < 2.549623){
// This is a leaf node
result += -0.001773;
}
else { // if condition is not respected
// This is a leaf node
result += 0.029381;
}
}
}
if (event[2] < -3.749223){
if (event[2] < -3.845556){
if (event[0] < -0.624683){
// This is a leaf node
result += -0.072956;
}
else { // if condition is not respected
// This is a leaf node
result += 0.018479;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.108413;
}
}
else { // if condition is not respected
if (event[4] < -2.212085){
if (event[0] < -2.835312){
// This is a leaf node
result += 0.133479;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003350;
}
}
else { // if condition is not respected
if (event[4] < -2.211330){
// This is a leaf node
result += 0.123283;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000032;
}
}
}
if (event[4] < -1.646589){
if (event[1] < 2.875306){
if (event[0] < -2.879530){
// This is a leaf node
result += 0.071180;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001175;
}
}
else { // if condition is not respected
if (event[4] < -2.449046){
// This is a leaf node
result += -0.000886;
}
else { // if condition is not respected
// This is a leaf node
result += 0.102524;
}
}
}
else { // if condition is not respected
if (event[4] < -1.645776){
if (event[3] < 0.511981){
// This is a leaf node
result += -0.148488;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000285;
}
}
else { // if condition is not respected
if (event[4] < -1.605698){
// This is a leaf node
result += -0.012596;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000014;
}
}
}
if (event[3] < -3.993109){
if (event[2] < 0.081382){
// This is a leaf node
result += 0.039588;
}
else { // if condition is not respected
// This is a leaf node
result += -0.119980;
}
}
else { // if condition is not respected
if (event[3] < -3.681860){
if (event[0] < -1.039718){
// This is a leaf node
result += -0.036571;
}
else { // if condition is not respected
// This is a leaf node
result += 0.082576;
}
}
else { // if condition is not respected
if (event[3] < -2.946279){
// This is a leaf node
result += -0.010690;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000013;
}
}
}
if (event[1] < 4.088089){
if (event[1] < 3.989292){
if (event[1] < 3.906952){
// This is a leaf node
result += 0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += -0.070379;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.091477;
}
}
else { // if condition is not respected
if (event[2] < 0.152884){
// This is a leaf node
result += -0.020312;
}
else { // if condition is not respected
// This is a leaf node
result += -0.075938;
}
}
if (event[2] < 4.023448){
if (event[2] < 3.986183){
if (event[3] < 3.492048){
// This is a leaf node
result += 0.000005;
}
else { // if condition is not respected
// This is a leaf node
result += -0.023973;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.104592;
}
}
else { // if condition is not respected
if (event[4] < -0.480630){
// This is a leaf node
result += -0.021431;
}
else { // if condition is not respected
if (event[3] < -0.255596){
// This is a leaf node
result += 0.021786;
}
else { // if condition is not respected
// This is a leaf node
result += 0.108521;
}
}
}
if (event[2] < 2.092766){
if (event[2] < 2.090705){
if (event[2] < 1.888311){
// This is a leaf node
result += -0.000020;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005154;
}
}
else { // if condition is not respected
if (event[1] < 0.184759){
// This is a leaf node
result += 0.114463;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001165;
}
}
}
else { // if condition is not respected
if (event[0] < -2.411729){
if (event[3] < -1.241215){
// This is a leaf node
result += -0.034828;
}
else { // if condition is not respected
// This is a leaf node
result += 0.080608;
}
}
else { // if condition is not respected
if (event[4] < -2.592435){
// This is a leaf node
result += -0.076260;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002656;
}
}
}
if (event[4] < 1.671135){
if (event[4] < 1.632809){
if (event[4] < 1.630600){
// This is a leaf node
result += 0.000044;
}
else { // if condition is not respected
// This is a leaf node
result += -0.054384;
}
}
else { // if condition is not respected
if (event[2] < 2.072350){
// This is a leaf node
result += 0.013687;
}
else { // if condition is not respected
// This is a leaf node
result += -0.066006;
}
}
}
else { // if condition is not respected
if (event[4] < 1.673165){
if (event[4] < 1.672220){
// This is a leaf node
result += -0.012399;
}
else { // if condition is not respected
// This is a leaf node
result += -0.107088;
}
}
else { // if condition is not respected
if (event[1] < -0.368850){
// This is a leaf node
result += -0.005737;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001187;
}
}
}
if (event[1] < -0.687141){
if (event[1] < -0.764778){
if (event[3] < 2.152789){
// This is a leaf node
result += -0.000111;
}
else { // if condition is not respected
// This is a leaf node
result += 0.013772;
}
}
else { // if condition is not respected
if (event[0] < 1.338370){
// This is a leaf node
result += 0.003997;
}
else { // if condition is not respected
// This is a leaf node
result += 0.017952;
}
}
}
else { // if condition is not respected
if (event[1] < -0.686985){
if (event[0] < -0.277321){
// This is a leaf node
result += 0.000356;
}
else { // if condition is not respected
// This is a leaf node
result += -0.128713;
}
}
else { // if condition is not respected
if (event[3] < 2.344643){
// This is a leaf node
result += -0.000130;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006585;
}
}
}
if (event[0] < 2.237703){
if (event[0] < 2.218218){
if (event[0] < 2.217745){
// This is a leaf node
result += 0.000023;
}
else { // if condition is not respected
// This is a leaf node
result += -0.075789;
}
}
else { // if condition is not respected
if (event[4] < 0.767422){
// This is a leaf node
result += 0.036483;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010313;
}
}
}
else { // if condition is not respected
if (event[4] < -0.634073){
if (event[3] < 1.336598){
// This is a leaf node
result += -0.016011;
}
else { // if condition is not respected
// This is a leaf node
result += 0.037779;
}
}
else { // if condition is not respected
if (event[2] < -2.356787){
// This is a leaf node
result += -0.073856;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000714;
}
}
}
if (event[2] < -3.319989){
if (event[2] < -3.397475){
if (event[1] < 0.051816){
// This is a leaf node
result += 0.027437;
}
else { // if condition is not respected
// This is a leaf node
result += -0.032945;
}
}
else { // if condition is not respected
if (event[1] < -1.263014){
// This is a leaf node
result += -0.040559;
}
else { // if condition is not respected
// This is a leaf node
result += 0.090045;
}
}
}
else { // if condition is not respected
if (event[2] < -3.020078){
if (event[4] < 1.293926){
// This is a leaf node
result += -0.025976;
}
else { // if condition is not respected
// This is a leaf node
result += 0.089252;
}
}
else { // if condition is not respected
if (event[2] < -2.995612){
// This is a leaf node
result += 0.053461;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000001;
}
}
}
if (event[1] < -2.316840){
if (event[4] < -2.014430){
if (event[2] < 1.559556){
// This is a leaf node
result += 0.059085;
}
else { // if condition is not respected
// This is a leaf node
result += -0.085467;
}
}
else { // if condition is not respected
if (event[3] < -2.412790){
// This is a leaf node
result += -0.067323;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002808;
}
}
}
else { // if condition is not respected
if (event[1] < -2.316143){
// This is a leaf node
result += -0.108185;
}
else { // if condition is not respected
if (event[1] < -2.184444){
// This is a leaf node
result += -0.007521;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000001;
}
}
}
if (event[1] < -3.244704){
if (event[1] < -3.256130){
if (event[0] < -0.912125){
// This is a leaf node
result += -0.056459;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000916;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.111593;
}
}
else { // if condition is not respected
if (event[1] < -3.217031){
if (event[4] < 0.073783){
// This is a leaf node
result += 0.093197;
}
else { // if condition is not respected
// This is a leaf node
result += -0.019427;
}
}
else { // if condition is not respected
if (event[1] < -3.204350){
// This is a leaf node
result += -0.088093;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000006;
}
}
}
if (event[4] < 3.258467){
if (event[4] < 3.251998){
if (event[4] < 3.246398){
// This is a leaf node
result += 0.000007;
}
else { // if condition is not respected
// This is a leaf node
result += -0.090543;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.099771;
}
}
else { // if condition is not respected
if (event[1] < 0.826593){
if (event[4] < 3.286601){
// This is a leaf node
result += -0.095009;
}
else { // if condition is not respected
// This is a leaf node
result += -0.022834;
}
}
else { // if condition is not respected
if (event[4] < 3.337547){
// This is a leaf node
result += -0.050201;
}
else { // if condition is not respected
// This is a leaf node
result += 0.079005;
}
}
}
if (event[0] < 4.111604){
if (event[0] < 3.515198){
if (event[0] < 3.430337){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.074172;
}
}
else { // if condition is not respected
if (event[2] < 0.882167){
// This is a leaf node
result += -0.061386;
}
else { // if condition is not respected
// This is a leaf node
result += 0.109498;
}
}
}
else { // if condition is not respected
if (event[0] < 4.301860){
// This is a leaf node
result += 0.096093;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005570;
}
}
if (event[4] < -0.370595){
if (event[4] < -0.371846){
if (event[2] < -2.193375){
// This is a leaf node
result += -0.010986;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000246;
}
}
else { // if condition is not respected
if (event[4] < -0.370966){
// This is a leaf node
result += -0.051587;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000286;
}
}
}
else { // if condition is not respected
if (event[4] < -0.364834){
if (event[1] < 1.184914){
// This is a leaf node
result += 0.009570;
}
else { // if condition is not respected
// This is a leaf node
result += 0.085259;
}
}
else { // if condition is not respected
if (event[4] < -0.362103){
// This is a leaf node
result += -0.019948;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000218;
}
}
}
if (event[4] < 0.911245){
if (event[4] < 0.911066){
if (event[2] < 2.972512){
// This is a leaf node
result += 0.000130;
}
else { // if condition is not respected
// This is a leaf node
result += 0.014795;
}
}
else { // if condition is not respected
if (event[3] < -0.694308){
// This is a leaf node
result += 0.022305;
}
else { // if condition is not respected
// This is a leaf node
result += 0.140097;
}
}
}
else { // if condition is not respected
if (event[4] < 0.911633){
if (event[2] < -0.920595){
// This is a leaf node
result += 0.018370;
}
else { // if condition is not respected
// This is a leaf node
result += -0.104005;
}
}
else { // if condition is not respected
if (event[2] < 2.320230){
// This is a leaf node
result += -0.000515;
}
else { // if condition is not respected
// This is a leaf node
result += -0.016634;
}
}
}
if (event[3] < -1.894090){
if (event[1] < 2.878712){
if (event[4] < -1.037759){
// This is a leaf node
result += -0.007755;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003355;
}
}
else { // if condition is not respected
if (event[1] < 3.156651){
// This is a leaf node
result += 0.130178;
}
else { // if condition is not respected
// This is a leaf node
result += 0.021605;
}
}
}
else { // if condition is not respected
if (event[3] < -1.859238){
if (event[0] < 1.108913){
// This is a leaf node
result += -0.005687;
}
else { // if condition is not respected
// This is a leaf node
result += -0.047422;
}
}
else { // if condition is not respected
if (event[1] < 2.818218){
// This is a leaf node
result += -0.000008;
}
else { // if condition is not respected
// This is a leaf node
result += -0.008997;
}
}
}
if (event[1] < 0.526281){
if (event[1] < 0.518191){
if (event[4] < 3.258467){
// This is a leaf node
result += -0.000143;
}
else { // if condition is not respected
// This is a leaf node
result += -0.026913;
}
}
else { // if condition is not respected
if (event[2] < -1.379676){
// This is a leaf node
result += 0.049824;
}
else { // if condition is not respected
// This is a leaf node
result += -0.017795;
}
}
}
else { // if condition is not respected
if (event[4] < -0.156816){
if (event[4] < -0.174787){
// This is a leaf node
result += -0.000875;
}
else { // if condition is not respected
// This is a leaf node
result += -0.018920;
}
}
else { // if condition is not respected
if (event[4] < -0.155783){
// This is a leaf node
result += 0.069607;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001739;
}
}
}
if (event[4] < -1.646589){
if (event[1] < 2.875306){
if (event[0] < 1.724642){
// This is a leaf node
result += 0.000612;
}
else { // if condition is not respected
// This is a leaf node
result += 0.016998;
}
}
else { // if condition is not respected
if (event[0] < 1.031109){
// This is a leaf node
result += 0.094436;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001832;
}
}
}
else { // if condition is not respected
if (event[4] < -1.638354){
if (event[0] < 1.873119){
// This is a leaf node
result += -0.037768;
}
else { // if condition is not respected
// This is a leaf node
result += 0.077525;
}
}
else { // if condition is not respected
if (event[4] < -1.605698){
// This is a leaf node
result += -0.008653;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000018;
}
}
}
if (event[1] < 1.434816){
if (event[1] < 1.434327){
if (event[1] < 1.434138){
// This is a leaf node
result += 0.000093;
}
else { // if condition is not respected
// This is a leaf node
result += -0.108921;
}
}
else { // if condition is not respected
if (event[3] < -0.846518){
// This is a leaf node
result += -0.020655;
}
else { // if condition is not respected
// This is a leaf node
result += 0.084040;
}
}
}
else { // if condition is not respected
if (event[4] < -2.090442){
if (event[3] < -0.725689){
// This is a leaf node
result += 0.053521;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001244;
}
}
else { // if condition is not respected
if (event[4] < -2.059795){
// This is a leaf node
result += -0.056899;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001351;
}
}
}
if (event[4] < -2.212085){
if (event[4] < -2.373916){
if (event[0] < 1.856176){
// This is a leaf node
result += 0.000298;
}
else { // if condition is not respected
// This is a leaf node
result += 0.056907;
}
}
else { // if condition is not respected
if (event[3] < -0.627826){
// This is a leaf node
result += -0.041492;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001536;
}
}
}
else { // if condition is not respected
if (event[4] < -2.153306){
if (event[0] < 1.281225){
// This is a leaf node
result += 0.009612;
}
else { // if condition is not respected
// This is a leaf node
result += 0.070633;
}
}
else { // if condition is not respected
if (event[4] < -2.149900){
// This is a leaf node
result += -0.045397;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000011;
}
}
}
if (event[4] < -2.843313){
if (event[3] < 1.235030){
if (event[3] < 1.012421){
// This is a leaf node
result += -0.007635;
}
else { // if condition is not respected
// This is a leaf node
result += -0.068117;
}
}
else { // if condition is not respected
if (event[3] < 2.056719){
// This is a leaf node
result += 0.049700;
}
else { // if condition is not respected
// This is a leaf node
result += -0.031171;
}
}
}
else { // if condition is not respected
if (event[4] < -2.817898){
if (event[1] < -0.156040){
// This is a leaf node
result += -0.009730;
}
else { // if condition is not respected
// This is a leaf node
result += 0.081611;
}
}
else { // if condition is not respected
if (event[4] < -2.794321){
// This is a leaf node
result += -0.039111;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000016;
}
}
}
if (event[2] < -0.196310){
if (event[2] < -0.202890){
if (event[1] < -1.721491){
// This is a leaf node
result += -0.004657;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000107;
}
}
else { // if condition is not respected
if (event[0] < -0.880857){
// This is a leaf node
result += -0.035729;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007525;
}
}
}
else { // if condition is not respected
if (event[2] < -0.186625){
if (event[2] < -0.189525){
// This is a leaf node
result += 0.001967;
}
else { // if condition is not respected
// This is a leaf node
result += 0.030889;
}
}
else { // if condition is not respected
if (event[2] < -0.186441){
// This is a leaf node
result += -0.099731;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000226;
}
}
}
if (event[1] < -0.687141){
if (event[0] < 3.162592){
if (event[2] < 1.513580){
// This is a leaf node
result += 0.000920;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004803;
}
}
else { // if condition is not respected
if (event[0] < 3.462018){
// This is a leaf node
result += 0.093154;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006400;
}
}
}
else { // if condition is not respected
if (event[1] < -0.681284){
if (event[0] < 1.917206){
// This is a leaf node
result += -0.011065;
}
else { // if condition is not respected
// This is a leaf node
result += -0.113681;
}
}
else { // if condition is not respected
if (event[2] < 1.220473){
// This is a leaf node
result += -0.000365;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001519;
}
}
}
if (event[0] < 2.237703){
if (event[0] < 2.141204){
if (event[0] < 2.140868){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.101616;
}
}
else { // if condition is not respected
if (event[3] < -1.549615){
// This is a leaf node
result += 0.063806;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006744;
}
}
}
else { // if condition is not respected
if (event[1] < 0.386825){
if (event[4] < 2.545454){
// This is a leaf node
result += 0.001513;
}
else { // if condition is not respected
// This is a leaf node
result += -0.113626;
}
}
else { // if condition is not respected
if (event[1] < 1.166071){
// This is a leaf node
result += -0.017748;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006905;
}
}
}
if (event[1] < -2.298697){
if (event[1] < -2.303315){
if (event[3] < 0.637218){
// This is a leaf node
result += -0.001323;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012312;
}
}
else { // if condition is not respected
if (event[3] < -0.829968){
// This is a leaf node
result += -0.024209;
}
else { // if condition is not respected
// This is a leaf node
result += 0.084468;
}
}
}
else { // if condition is not respected
if (event[1] < -2.268882){
if (event[1] < -2.276794){
// This is a leaf node
result += -0.003434;
}
else { // if condition is not respected
// This is a leaf node
result += -0.062706;
}
}
else { // if condition is not respected
if (event[1] < -2.245341){
// This is a leaf node
result += 0.025107;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000035;
}
}
}
if (event[3] < 0.323860){
if (event[3] < -0.202848){
if (event[3] < -0.203357){
// This is a leaf node
result += -0.000334;
}
else { // if condition is not respected
// This is a leaf node
result += -0.043568;
}
}
else { // if condition is not respected
if (event[3] < -0.202741){
// This is a leaf node
result += 0.080943;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001452;
}
}
}
else { // if condition is not respected
if (event[3] < 0.324561){
if (event[1] < 1.081719){
// This is a leaf node
result += -0.065325;
}
else { // if condition is not respected
// This is a leaf node
result += 0.054653;
}
}
else { // if condition is not respected
if (event[1] < 1.769154){
// This is a leaf node
result += -0.000213;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004827;
}
}
}
if (event[1] < 0.525445){
if (event[1] < 0.521333){
if (event[2] < -3.235507){
// This is a leaf node
result += 0.024876;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000191;
}
}
else { // if condition is not respected
if (event[2] < -1.371173){
// This is a leaf node
result += 0.049609;
}
else { // if condition is not respected
// This is a leaf node
result += -0.023167;
}
}
}
else { // if condition is not respected
if (event[1] < 0.551360){
if (event[0] < 2.444216){
// This is a leaf node
result += 0.007751;
}
else { // if condition is not respected
// This is a leaf node
result += 0.102890;
}
}
else { // if condition is not respected
if (event[1] < 0.555633){
// This is a leaf node
result += -0.024801;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000383;
}
}
}
if (event[4] < 0.911245){
if (event[4] < 0.911066){
if (event[4] < 0.905579){
// This is a leaf node
result += 0.000123;
}
else { // if condition is not respected
// This is a leaf node
result += 0.013058;
}
}
else { // if condition is not respected
if (event[3] < -0.694308){
// This is a leaf node
result += 0.020958;
}
else { // if condition is not respected
// This is a leaf node
result += 0.129253;
}
}
}
else { // if condition is not respected
if (event[4] < 0.911548){
if (event[3] < -0.426623){
// This is a leaf node
result += 0.006087;
}
else { // if condition is not respected
// This is a leaf node
result += -0.115478;
}
}
else { // if condition is not respected
if (event[2] < -0.779125){
// This is a leaf node
result += 0.002235;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001460;
}
}
}
if (event[1] < -3.336838){
if (event[3] < 0.747019){
if (event[2] < -1.649798){
// This is a leaf node
result += 0.092662;
}
else { // if condition is not respected
// This is a leaf node
result += -0.038228;
}
}
else { // if condition is not respected
if (event[1] < -3.402231){
// This is a leaf node
result += 0.058472;
}
else { // if condition is not respected
// This is a leaf node
result += -0.077047;
}
}
}
else { // if condition is not respected
if (event[1] < -3.324472){
if (event[4] < 0.177580){
// This is a leaf node
result += 0.014956;
}
else { // if condition is not respected
// This is a leaf node
result += 0.107776;
}
}
else { // if condition is not respected
if (event[1] < -3.315744){
// This is a leaf node
result += -0.084424;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000006;
}
}
}
if (event[0] < 1.275674){
if (event[0] < 1.179359){
if (event[0] < 1.179268){
// This is a leaf node
result += -0.000017;
}
else { // if condition is not respected
// This is a leaf node
result += -0.138516;
}
}
else { // if condition is not respected
if (event[2] < -1.618831){
// This is a leaf node
result += -0.018011;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007736;
}
}
}
else { // if condition is not respected
if (event[3] < -1.210643){
if (event[4] < 2.525112){
// This is a leaf node
result += -0.008218;
}
else { // if condition is not respected
// This is a leaf node
result += -0.092753;
}
}
else { // if condition is not respected
if (event[4] < -1.868134){
// This is a leaf node
result += 0.014670;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000432;
}
}
}
if (event[1] < 4.088089){
if (event[1] < 4.035286){
if (event[1] < 3.711486){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.035216;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.099197;
}
}
else { // if condition is not respected
if (event[4] < -0.391049){
// This is a leaf node
result += -0.071769;
}
else { // if condition is not respected
// This is a leaf node
result += -0.018004;
}
}
if (event[0] < -1.292493){
if (event[0] < -1.326237){
if (event[4] < 0.351193){
// This is a leaf node
result += 0.001969;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002646;
}
}
else { // if condition is not respected
if (event[3] < -0.657385){
// This is a leaf node
result += -0.004211;
}
else { // if condition is not respected
// This is a leaf node
result += 0.016716;
}
}
}
else { // if condition is not respected
if (event[0] < -1.291454){
if (event[4] < 0.661174){
// This is a leaf node
result += -0.012830;
}
else { // if condition is not respected
// This is a leaf node
result += -0.110530;
}
}
else { // if condition is not respected
if (event[0] < -1.291425){
// This is a leaf node
result += 0.127761;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000101;
}
}
}
if (event[3] < -1.971181){
if (event[2] < -2.023206){
if (event[4] < -0.380040){
// This is a leaf node
result += -0.008092;
}
else { // if condition is not respected
// This is a leaf node
result += 0.049982;
}
}
else { // if condition is not respected
if (event[2] < -1.835402){
// This is a leaf node
result += -0.041042;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001977;
}
}
}
else { // if condition is not respected
if (event[3] < -1.968372){
if (event[1] < -0.491265){
// This is a leaf node
result += -0.138525;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006659;
}
}
else { // if condition is not respected
if (event[1] < -2.314285){
// This is a leaf node
result += 0.003626;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000081;
}
}
}
if (event[3] < -3.993109){
if (event[2] < 0.081382){
// This is a leaf node
result += 0.036941;
}
else { // if condition is not respected
// This is a leaf node
result += -0.113278;
}
}
else { // if condition is not respected
if (event[3] < -3.681860){
if (event[0] < -1.039718){
// This is a leaf node
result += -0.034758;
}
else { // if condition is not respected
// This is a leaf node
result += 0.075236;
}
}
else { // if condition is not respected
if (event[3] < -2.946279){
// This is a leaf node
result += -0.009769;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000011;
}
}
}
if (event[0] < 4.010478){
if (event[0] < 3.689817){
if (event[0] < 3.651816){
// This is a leaf node
result += -0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += 0.102469;
}
}
else { // if condition is not respected
if (event[1] < -0.213191){
// This is a leaf node
result += 0.019019;
}
else { // if condition is not respected
// This is a leaf node
result += -0.096314;
}
}
}
else { // if condition is not respected
if (event[1] < 0.520747){
// This is a leaf node
result += -0.010411;
}
else { // if condition is not respected
// This is a leaf node
result += 0.097562;
}
}
if (event[2] < 4.023448){
if (event[2] < 3.986183){
if (event[4] < -2.590437){
// This is a leaf node
result += -0.004544;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000022;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.099008;
}
}
else { // if condition is not respected
if (event[4] < -0.480630){
// This is a leaf node
result += -0.020905;
}
else { // if condition is not respected
if (event[3] < -0.255596){
// This is a leaf node
result += 0.020906;
}
else { // if condition is not respected
// This is a leaf node
result += 0.102301;
}
}
}
if (event[3] < 2.319761){
if (event[3] < 2.309603){
if (event[3] < 2.308932){
// This is a leaf node
result += 0.000020;
}
else { // if condition is not respected
// This is a leaf node
result += -0.127634;
}
}
else { // if condition is not respected
if (event[1] < 0.145666){
// This is a leaf node
result += 0.093811;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010483;
}
}
}
else { // if condition is not respected
if (event[1] < -0.908032){
if (event[0] < -2.001135){
// This is a leaf node
result += 0.107001;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012257;
}
}
else { // if condition is not respected
if (event[0] < -0.366880){
// This is a leaf node
result += -0.018822;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000144;
}
}
}
if (event[1] < 3.383786){
if (event[1] < 3.371411){
if (event[1] < 3.154752){
// This is a leaf node
result += 0.000005;
}
else { // if condition is not respected
// This is a leaf node
result += -0.019441;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.124853;
}
}
else { // if condition is not respected
if (event[0] < 0.468883){
if (event[1] < 3.657566){
// This is a leaf node
result += 0.061272;
}
else { // if condition is not respected
// This is a leaf node
result += -0.008274;
}
}
else { // if condition is not respected
if (event[3] < -0.100981){
// This is a leaf node
result += -0.097340;
}
else { // if condition is not respected
// This is a leaf node
result += 0.013140;
}
}
}
if (event[1] < -3.123382){
if (event[1] < -3.127925){
if (event[2] < 1.581593){
// This is a leaf node
result += -0.012574;
}
else { // if condition is not respected
// This is a leaf node
result += 0.077676;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.139803;
}
}
else { // if condition is not respected
if (event[1] < -3.113010){
if (event[0] < 0.580553){
// This is a leaf node
result += 0.141741;
}
else { // if condition is not respected
// This is a leaf node
result += -0.039362;
}
}
else { // if condition is not respected
if (event[1] < -2.593957){
// This is a leaf node
result += 0.006312;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000018;
}
}
}
if (event[2] < -3.749223){
if (event[2] < -3.845556){
if (event[0] < 0.420350){
// This is a leaf node
result += -0.044638;
}
else { // if condition is not respected
// This is a leaf node
result += 0.037589;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.101866;
}
}
else { // if condition is not respected
if (event[2] < -3.453882){
if (event[1] < -0.484509){
// This is a leaf node
result += 0.036280;
}
else { // if condition is not respected
// This is a leaf node
result += -0.064774;
}
}
else { // if condition is not respected
if (event[2] < -3.319989){
// This is a leaf node
result += 0.043124;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000006;
}
}
}
if (event[3] < 3.589159){
if (event[3] < 3.116394){
if (event[3] < 3.086900){
// This is a leaf node
result += -0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.035402;
}
}
else { // if condition is not respected
if (event[3] < 3.157028){
// This is a leaf node
result += 0.098467;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001872;
}
}
}
else { // if condition is not respected
if (event[1] < -1.425893){
// This is a leaf node
result += 0.057875;
}
else { // if condition is not respected
if (event[0] < -0.168487){
// This is a leaf node
result += -0.095965;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004636;
}
}
}
if (event[2] < -0.196334){
if (event[2] < -0.433119){
if (event[2] < -0.436768){
// This is a leaf node
result += 0.000043;
}
else { // if condition is not respected
// This is a leaf node
result += 0.016371;
}
}
else { // if condition is not respected
if (event[2] < -0.428705){
// This is a leaf node
result += -0.022045;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001722;
}
}
}
else { // if condition is not respected
if (event[2] < -0.193327){
if (event[2] < -0.193925){
// This is a leaf node
result += 0.008472;
}
else { // if condition is not respected
// This is a leaf node
result += 0.052857;
}
}
else { // if condition is not respected
if (event[2] < -0.190753){
// This is a leaf node
result += -0.019407;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000257;
}
}
}
if (event[3] < -3.993109){
if (event[2] < 0.081382){
// This is a leaf node
result += 0.034876;
}
else { // if condition is not respected
// This is a leaf node
result += -0.107402;
}
}
else { // if condition is not respected
if (event[3] < -3.681860){
if (event[0] < 0.110352){
// This is a leaf node
result += 0.005823;
}
else { // if condition is not respected
// This is a leaf node
result += 0.093333;
}
}
else { // if condition is not respected
if (event[3] < -2.862381){
// This is a leaf node
result += -0.007728;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000012;
}
}
}
if (event[3] < -1.894090){
if (event[1] < 2.878712){
if (event[1] < 2.856992){
// This is a leaf node
result += 0.001682;
}
else { // if condition is not respected
// This is a leaf node
result += -0.135526;
}
}
else { // if condition is not respected
if (event[1] < 3.156651){
// This is a leaf node
result += 0.119156;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020624;
}
}
}
else { // if condition is not respected
if (event[3] < -1.893658){
if (event[4] < -0.259879){
// This is a leaf node
result += -0.020964;
}
else { // if condition is not respected
// This is a leaf node
result += -0.119360;
}
}
else { // if condition is not respected
if (event[3] < -1.859238){
// This is a leaf node
result += -0.008942;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000029;
}
}
}
if (event[2] < 4.023448){
if (event[2] < 3.986183){
if (event[1] < 0.526281){
// This is a leaf node
result += -0.000198;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000462;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.094352;
}
}
else { // if condition is not respected
if (event[3] < -0.613557){
// This is a leaf node
result += -0.020285;
}
else { // if condition is not respected
if (event[1] < -0.216012){
// This is a leaf node
result += 0.023100;
}
else { // if condition is not respected
// This is a leaf node
result += 0.101417;
}
}
}
if (event[1] < 2.818218){
if (event[1] < 2.779244){
if (event[1] < 2.762832){
// This is a leaf node
result += 0.000011;
}
else { // if condition is not respected
// This is a leaf node
result += -0.043671;
}
}
else { // if condition is not respected
if (event[1] < 2.782566){
// This is a leaf node
result += 0.138888;
}
else { // if condition is not respected
// This is a leaf node
result += 0.021177;
}
}
}
else { // if condition is not respected
if (event[3] < -1.319610){
if (event[1] < 2.878712){
// This is a leaf node
result += -0.035403;
}
else { // if condition is not respected
// This is a leaf node
result += 0.068305;
}
}
else { // if condition is not respected
if (event[3] < -0.398717){
// This is a leaf node
result += -0.039968;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000325;
}
}
}
if (event[0] < -0.586003){
if (event[0] < -0.587138){
if (event[1] < 0.044524){
// This is a leaf node
result += 0.000761;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001714;
}
}
else { // if condition is not respected
if (event[3] < 1.183520){
// This is a leaf node
result += -0.021598;
}
else { // if condition is not respected
// This is a leaf node
result += -0.125379;
}
}
}
else { // if condition is not respected
if (event[0] < -0.560185){
if (event[3] < 2.371461){
// This is a leaf node
result += 0.006808;
}
else { // if condition is not respected
// This is a leaf node
result += -0.090991;
}
}
else { // if condition is not respected
if (event[0] < -0.559860){
// This is a leaf node
result += -0.058219;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000125;
}
}
}
if (event[0] < -1.241425){
if (event[2] < 2.408053){
if (event[2] < 2.245631){
// This is a leaf node
result += 0.000932;
}
else { // if condition is not respected
// This is a leaf node
result += 0.035164;
}
}
else { // if condition is not respected
if (event[2] < 3.125425){
// This is a leaf node
result += -0.031694;
}
else { // if condition is not respected
// This is a leaf node
result += 0.043109;
}
}
}
else { // if condition is not respected
if (event[0] < -1.239173){
if (event[4] < 0.011571){
// This is a leaf node
result += -0.085919;
}
else { // if condition is not respected
// This is a leaf node
result += -0.012307;
}
}
else { // if condition is not respected
if (event[0] < -0.625203){
// This is a leaf node
result += -0.001245;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000163;
}
}
}
if (event[3] < 0.326655){
if (event[3] < 0.325698){
if (event[3] < -0.202299){
// This is a leaf node
result += -0.000310;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001252;
}
}
else { // if condition is not respected
if (event[2] < 0.776240){
// This is a leaf node
result += 0.055752;
}
else { // if condition is not respected
// This is a leaf node
result += -0.041477;
}
}
}
else { // if condition is not respected
if (event[3] < 0.383059){
if (event[2] < 1.513044){
// This is a leaf node
result += -0.006108;
}
else { // if condition is not respected
// This is a leaf node
result += 0.018112;
}
}
else { // if condition is not respected
if (event[3] < 0.383231){
// This is a leaf node
result += 0.074699;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000158;
}
}
}
if (event[2] < 2.092766){
if (event[2] < 2.090705){
if (event[2] < 1.888311){
// This is a leaf node
result += -0.000018;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004551;
}
}
else { // if condition is not respected
if (event[1] < 0.184759){
// This is a leaf node
result += 0.106294;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000883;
}
}
}
else { // if condition is not respected
if (event[0] < -2.382587){
if (event[1] < -0.411801){
// This is a leaf node
result += 0.001379;
}
else { // if condition is not respected
// This is a leaf node
result += 0.084144;
}
}
else { // if condition is not respected
if (event[3] < 2.815389){
// This is a leaf node
result += -0.002946;
}
else { // if condition is not respected
// This is a leaf node
result += 0.083932;
}
}
}
if (event[2] < 1.488528){
if (event[2] < 1.269283){
if (event[2] < 1.268569){
// This is a leaf node
result += -0.000051;
}
else { // if condition is not respected
// This is a leaf node
result += -0.050082;
}
}
else { // if condition is not respected
if (event[3] < -3.050962){
// This is a leaf node
result += -0.103089;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004013;
}
}
}
else { // if condition is not respected
if (event[1] < 2.176867){
if (event[2] < 1.488886){
// This is a leaf node
result += -0.100243;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000643;
}
}
else { // if condition is not respected
if (event[1] < 2.210618){
// This is a leaf node
result += -0.091972;
}
else { // if condition is not respected
// This is a leaf node
result += -0.021683;
}
}
}
if (event[1] < 0.526281){
if (event[1] < 0.469962){
if (event[1] < 0.469804){
// This is a leaf node
result += -0.000077;
}
else { // if condition is not respected
// This is a leaf node
result += 0.066434;
}
}
else { // if condition is not respected
if (event[1] < 0.477258){
// This is a leaf node
result += -0.021980;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001788;
}
}
}
else { // if condition is not respected
if (event[1] < 0.526348){
if (event[3] < -0.528093){
// This is a leaf node
result += 0.022277;
}
else { // if condition is not respected
// This is a leaf node
result += 0.145788;
}
}
else { // if condition is not respected
if (event[4] < -0.156816){
// This is a leaf node
result += -0.001088;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001630;
}
}
}
if (event[4] < 0.542113){
if (event[4] < 0.536056){
if (event[2] < -3.556891){
// This is a leaf node
result += 0.047985;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000139;
}
}
else { // if condition is not respected
if (event[0] < -1.421722){
// This is a leaf node
result += -0.028388;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020491;
}
}
}
else { // if condition is not respected
if (event[2] < -1.873970){
if (event[3] < -0.716241){
// This is a leaf node
result += 0.022108;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000378;
}
}
else { // if condition is not respected
if (event[1] < -3.650717){
// This is a leaf node
result += -0.107372;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000648;
}
}
}
if (event[2] < -2.304283){
if (event[1] < -0.353279){
if (event[2] < -2.329827){
// This is a leaf node
result += -0.007871;
}
else { // if condition is not respected
// This is a leaf node
result += -0.061074;
}
}
else { // if condition is not respected
if (event[1] < -0.117763){
// This is a leaf node
result += 0.032350;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003409;
}
}
}
else { // if condition is not respected
if (event[2] < -2.302579){
if (event[1] < -1.043266){
// This is a leaf node
result += 0.026734;
}
else { // if condition is not respected
// This is a leaf node
result += 0.125409;
}
}
else { // if condition is not respected
if (event[2] < -2.301145){
// This is a leaf node
result += -0.083345;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000031;
}
}
}
if (event[2] < -2.622372){
if (event[3] < 0.665693){
if (event[3] < 0.278093){
// This is a leaf node
result += -0.003095;
}
else { // if condition is not respected
// This is a leaf node
result += -0.044028;
}
}
else { // if condition is not respected
if (event[0] < -1.511454){
// This is a leaf node
result += 0.081613;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009762;
}
}
}
else { // if condition is not respected
if (event[2] < -2.548010){
if (event[3] < -1.371227){
// This is a leaf node
result += -0.055790;
}
else { // if condition is not respected
// This is a leaf node
result += 0.023114;
}
}
else { // if condition is not respected
if (event[2] < -2.537097){
// This is a leaf node
result += -0.054352;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000012;
}
}
}
if (event[4] < -1.884302){
if (event[0] < 1.016262){
if (event[4] < -1.884816){
// This is a leaf node
result += -0.000553;
}
else { // if condition is not respected
// This is a leaf node
result += 0.110322;
}
}
else { // if condition is not respected
if (event[4] < -2.220736){
// This is a leaf node
result += -0.000360;
}
else { // if condition is not respected
// This is a leaf node
result += 0.023272;
}
}
}
else { // if condition is not respected
if (event[4] < -1.883370){
if (event[1] < -0.619273){
// This is a leaf node
result += 0.022407;
}
else { // if condition is not respected
// This is a leaf node
result += -0.103548;
}
}
else { // if condition is not respected
if (event[4] < -1.882930){
// This is a leaf node
result += 0.095205;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000052;
}
}
}
if (event[0] < 1.266936){
if (event[0] < 1.262777){
if (event[0] < 1.179359){
// This is a leaf node
result += -0.000012;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004940;
}
}
else { // if condition is not respected
if (event[1] < -1.241299){
// This is a leaf node
result += 0.101117;
}
else { // if condition is not respected
// This is a leaf node
result += 0.023661;
}
}
}
else { // if condition is not respected
if (event[3] < -1.472573){
if (event[0] < 1.310618){
// This is a leaf node
result += -0.039974;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007612;
}
}
else { // if condition is not respected
if (event[4] < -1.919399){
// This is a leaf node
result += 0.012124;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000570;
}
}
}
if (event[3] < -1.894090){
if (event[3] < -1.894742){
if (event[4] < 1.266467){
// This is a leaf node
result += 0.000307;
}
else { // if condition is not respected
// This is a leaf node
result += 0.013337;
}
}
else { // if condition is not respected
if (event[0] < 0.501181){
// This is a leaf node
result += 0.125490;
}
else { // if condition is not respected
// This is a leaf node
result += 0.029772;
}
}
}
else { // if condition is not respected
if (event[3] < -1.893310){
if (event[1] < -0.202569){
// This is a leaf node
result += -0.106010;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003483;
}
}
else { // if condition is not respected
if (event[3] < -1.893069){
// This is a leaf node
result += 0.093328;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000050;
}
}
}
if (event[0] < 3.073634){
if (event[0] < 3.058667){
if (event[0] < 3.049398){
// This is a leaf node
result += -0.000007;
}
else { // if condition is not respected
// This is a leaf node
result += 0.086208;
}
}
else { // if condition is not respected
if (event[2] < 0.558834){
// This is a leaf node
result += -0.146696;
}
else { // if condition is not respected
// This is a leaf node
result += 0.028402;
}
}
}
else { // if condition is not respected
if (event[1] < -0.558785){
if (event[4] < -0.140707){
// This is a leaf node
result += 0.008915;
}
else { // if condition is not respected
// This is a leaf node
result += 0.076450;
}
}
else { // if condition is not respected
if (event[1] < 0.467791){
// This is a leaf node
result += -0.029038;
}
else { // if condition is not respected
// This is a leaf node
result += 0.032083;
}
}
}
if (event[0] < 2.326553){
if (event[0] < 2.325281){
if (event[0] < 2.291232){
// This is a leaf node
result += 0.000009;
}
else { // if condition is not respected
// This is a leaf node
result += 0.018023;
}
}
else { // if condition is not respected
if (event[4] < 0.328957){
// This is a leaf node
result += 0.130626;
}
else { // if condition is not respected
// This is a leaf node
result += 0.019646;
}
}
}
else { // if condition is not respected
if (event[0] < 2.364230){
if (event[4] < -0.931952){
// This is a leaf node
result += -0.073789;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010842;
}
}
else { // if condition is not respected
if (event[2] < 1.659111){
// This is a leaf node
result += -0.002544;
}
else { // if condition is not respected
// This is a leaf node
result += 0.029171;
}
}
}
if (event[4] < -2.212085){
if (event[0] < -2.835312){
// This is a leaf node
result += 0.119555;
}
else { // if condition is not respected
if (event[4] < -2.373916){
// This is a leaf node
result += 0.001737;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011672;
}
}
}
else { // if condition is not respected
if (event[4] < -2.211330){
// This is a leaf node
result += 0.111939;
}
else { // if condition is not respected
if (event[4] < -1.884302){
// This is a leaf node
result += 0.004437;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000044;
}
}
}
if (event[4] < -2.590437){
if (event[2] < 1.734092){
if (event[0] < 1.807037){
// This is a leaf node
result += -0.004043;
}
else { // if condition is not respected
// This is a leaf node
result += 0.051124;
}
}
else { // if condition is not respected
if (event[3] < 1.568288){
// This is a leaf node
result += -0.071846;
}
else { // if condition is not respected
// This is a leaf node
result += 0.111843;
}
}
}
else { // if condition is not respected
if (event[4] < -2.556193){
if (event[0] < 0.370612){
// This is a leaf node
result += 0.055846;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000736;
}
}
else { // if condition is not respected
if (event[4] < -2.553907){
// This is a leaf node
result += -0.091032;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000010;
}
}
}
if (event[4] < 3.594640){
if (event[4] < 3.471214){
if (event[4] < 3.403118){
// This is a leaf node
result += 0.000004;
}
else { // if condition is not respected
// This is a leaf node
result += -0.047844;
}
}
else { // if condition is not respected
if (event[2] < 0.940680){
// This is a leaf node
result += 0.071071;
}
else { // if condition is not respected
// This is a leaf node
result += -0.063741;
}
}
}
else { // if condition is not respected
if (event[4] < 3.647938){
if (event[1] < 0.597298){
// This is a leaf node
result += -0.136473;
}
else { // if condition is not respected
// This is a leaf node
result += -0.027943;
}
}
else { // if condition is not respected
if (event[1] < 0.768242){
// This is a leaf node
result += -0.021094;
}
else { // if condition is not respected
// This is a leaf node
result += 0.063240;
}
}
}
if (event[3] < -3.395477){
if (event[0] < -1.039718){
if (event[0] < -1.739598){
// This is a leaf node
result += 0.028826;
}
else { // if condition is not respected
// This is a leaf node
result += -0.108544;
}
}
else { // if condition is not respected
if (event[1] < -0.445583){
// This is a leaf node
result += -0.005102;
}
else { // if condition is not respected
// This is a leaf node
result += 0.052994;
}
}
}
else { // if condition is not respected
if (event[3] < -3.335923){
if (event[2] < 0.181666){
// This is a leaf node
result += 0.023118;
}
else { // if condition is not respected
// This is a leaf node
result += -0.107013;
}
}
else { // if condition is not respected
if (event[3] < -3.304952){
// This is a leaf node
result += 0.066114;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000005;
}
}
}
if (event[3] < -3.993109){
if (event[2] < 0.081382){
// This is a leaf node
result += 0.030424;
}
else { // if condition is not respected
// This is a leaf node
result += -0.102623;
}
}
else { // if condition is not respected
if (event[3] < -3.681860){
if (event[3] < -3.789101){
// This is a leaf node
result += 0.003523;
}
else { // if condition is not respected
// This is a leaf node
result += 0.085466;
}
}
else { // if condition is not respected
if (event[3] < -3.658852){
// This is a leaf node
result += -0.077604;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000001;
}
}
}
if (event[1] < 4.088089){
if (event[1] < 3.989292){
if (event[1] < 3.939937){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.076550;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.082368;
}
}
else { // if condition is not respected
if (event[2] < 0.152884){
// This is a leaf node
result += -0.013384;
}
else { // if condition is not respected
// This is a leaf node
result += -0.068023;
}
}
if (event[0] < 2.237703){
if (event[0] < 2.216883){
if (event[0] < 2.215728){
// This is a leaf node
result += 0.000022;
}
else { // if condition is not respected
// This is a leaf node
result += -0.083096;
}
}
else { // if condition is not respected
if (event[1] < 0.456722){
// This is a leaf node
result += 0.038173;
}
else { // if condition is not respected
// This is a leaf node
result += -0.012772;
}
}
}
else { // if condition is not respected
if (event[0] < 2.291232){
if (event[3] < 0.943209){
// This is a leaf node
result += -0.023437;
}
else { // if condition is not respected
// This is a leaf node
result += 0.023015;
}
}
else { // if condition is not respected
if (event[0] < 2.310485){
// This is a leaf node
result += 0.038691;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002732;
}
}
}
if (event[0] < 4.010478){
if (event[0] < 3.689817){
if (event[0] < 3.651816){
// This is a leaf node
result += 0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += 0.096622;
}
}
else { // if condition is not respected
if (event[1] < -0.213191){
// This is a leaf node
result += 0.016286;
}
else { // if condition is not respected
// This is a leaf node
result += -0.089458;
}
}
}
else { // if condition is not respected
if (event[1] < 0.520747){
// This is a leaf node
result += -0.009688;
}
else { // if condition is not respected
// This is a leaf node
result += 0.089328;
}
}
if (event[1] < 3.460927){
if (event[1] < 3.426914){
if (event[1] < 3.383786){
// This is a leaf node
result += -0.000005;
}
else { // if condition is not respected
// This is a leaf node
result += 0.055566;
}
}
else { // if condition is not respected
if (event[0] < -0.919180){
// This is a leaf node
result += 0.032436;
}
else { // if condition is not respected
// This is a leaf node
result += -0.115377;
}
}
}
else { // if condition is not respected
if (event[3] < 0.199900){
if (event[3] < -0.014721){
// This is a leaf node
result += 0.018902;
}
else { // if condition is not respected
// This is a leaf node
result += 0.131398;
}
}
else { // if condition is not respected
if (event[3] < 0.448409){
// This is a leaf node
result += -0.100233;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004531;
}
}
}
if (event[1] < 1.398992){
if (event[1] < 1.393134){
if (event[1] < 1.392972){
// This is a leaf node
result += 0.000064;
}
else { // if condition is not respected
// This is a leaf node
result += -0.091297;
}
}
else { // if condition is not respected
if (event[2] < -1.136734){
// This is a leaf node
result += -0.043311;
}
else { // if condition is not respected
// This is a leaf node
result += 0.037228;
}
}
}
else { // if condition is not respected
if (event[1] < 1.399748){
if (event[0] < -0.419555){
// This is a leaf node
result += -0.147321;
}
else { // if condition is not respected
// This is a leaf node
result += -0.033522;
}
}
else { // if condition is not respected
if (event[0] < -1.789608){
// This is a leaf node
result += 0.008348;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001201;
}
}
}
if (event[2] < 2.972512){
if (event[2] < 2.952371){
if (event[2] < 2.936414){
// This is a leaf node
result += -0.000010;
}
else { // if condition is not respected
// This is a leaf node
result += 0.051944;
}
}
else { // if condition is not respected
if (event[3] < 1.457287){
// This is a leaf node
result += -0.075351;
}
else { // if condition is not respected
// This is a leaf node
result += 0.065926;
}
}
}
else { // if condition is not respected
if (event[2] < 3.042542){
if (event[2] < 3.038306){
// This is a leaf node
result += 0.033764;
}
else { // if condition is not respected
// This is a leaf node
result += 0.149032;
}
}
else { // if condition is not respected
if (event[2] < 3.084193){
// This is a leaf node
result += -0.067469;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005487;
}
}
}
if (event[2] < 1.386067){
if (event[2] < 1.269283){
if (event[2] < 1.269186){
// This is a leaf node
result += -0.000049;
}
else { // if condition is not respected
// This is a leaf node
result += -0.110191;
}
}
else { // if condition is not respected
if (event[3] < -2.730171){
// This is a leaf node
result += -0.095623;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006717;
}
}
}
else { // if condition is not respected
if (event[2] < 1.389171){
if (event[0] < 1.199127){
// This is a leaf node
result += -0.046233;
}
else { // if condition is not respected
// This is a leaf node
result += 0.027248;
}
}
else { // if condition is not respected
if (event[3] < 2.832452){
// This is a leaf node
result += -0.000859;
}
else { // if condition is not respected
// This is a leaf node
result += 0.045983;
}
}
}
if (event[3] < 2.319761){
if (event[3] < 2.309603){
if (event[3] < 2.152436){
// This is a leaf node
result += -0.000025;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007898;
}
}
else { // if condition is not respected
if (event[1] < -0.373130){
// This is a leaf node
result += 0.108582;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012180;
}
}
}
else { // if condition is not respected
if (event[3] < 2.322264){
if (event[1] < -1.214269){
// This is a leaf node
result += 0.064081;
}
else { // if condition is not respected
// This is a leaf node
result += -0.112894;
}
}
else { // if condition is not respected
if (event[1] < -0.908032){
// This is a leaf node
result += 0.013863;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005513;
}
}
}
if (event[3] < 3.665319){
if (event[3] < 3.262835){
if (event[3] < 3.229567){
// This is a leaf node
result += -0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += -0.062877;
}
}
else { // if condition is not respected
if (event[1] < -1.346547){
// This is a leaf node
result += -0.080245;
}
else { // if condition is not respected
// This is a leaf node
result += 0.033770;
}
}
}
else { // if condition is not respected
if (event[3] < 3.740767){
// This is a leaf node
result += -0.094052;
}
else { // if condition is not respected
if (event[3] < 3.762374){
// This is a leaf node
result += 0.094956;
}
else { // if condition is not respected
// This is a leaf node
result += -0.020581;
}
}
}
if (event[0] < 3.073634){
if (event[0] < 3.058667){
if (event[0] < 3.049398){
// This is a leaf node
result += -0.000007;
}
else { // if condition is not respected
// This is a leaf node
result += 0.079474;
}
}
else { // if condition is not respected
if (event[2] < 0.558834){
// This is a leaf node
result += -0.135323;
}
else { // if condition is not respected
// This is a leaf node
result += 0.026524;
}
}
}
else { // if condition is not respected
if (event[0] < 3.224831){
if (event[2] < -1.177767){
// This is a leaf node
result += -0.096392;
}
else { // if condition is not respected
// This is a leaf node
result += 0.043241;
}
}
else { // if condition is not respected
if (event[1] < -0.926566){
// This is a leaf node
result += 0.067971;
}
else { // if condition is not respected
// This is a leaf node
result += -0.019936;
}
}
}
if (event[4] < -0.370595){
if (event[4] < -0.371846){
if (event[2] < -2.193375){
// This is a leaf node
result += -0.009691;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000205;
}
}
else { // if condition is not respected
if (event[2] < 0.858487){
// This is a leaf node
result += -0.022078;
}
else { // if condition is not respected
// This is a leaf node
result += -0.082157;
}
}
}
else { // if condition is not respected
if (event[4] < -0.363996){
if (event[1] < 1.184914){
// This is a leaf node
result += 0.009558;
}
else { // if condition is not respected
// This is a leaf node
result += 0.059379;
}
}
else { // if condition is not respected
if (event[4] < -0.362103){
// This is a leaf node
result += -0.030413;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000185;
}
}
}
if (event[3] < 0.606984){
if (event[3] < 0.606380){
if (event[3] < 0.605813){
// This is a leaf node
result += 0.000169;
}
else { // if condition is not respected
// This is a leaf node
result += -0.045666;
}
}
else { // if condition is not respected
if (event[2] < -0.521648){
// This is a leaf node
result += 0.001177;
}
else { // if condition is not respected
// This is a leaf node
result += 0.090887;
}
}
}
else { // if condition is not respected
if (event[3] < 0.611101){
if (event[2] < -1.908811){
// This is a leaf node
result += -0.121623;
}
else { // if condition is not respected
// This is a leaf node
result += -0.018085;
}
}
else { // if condition is not respected
if (event[2] < -2.539169){
// This is a leaf node
result += 0.016894;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000454;
}
}
}
if (event[4] < 0.799553){
if (event[4] < 0.778987){
if (event[4] < 0.766351){
// This is a leaf node
result += 0.000127;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011328;
}
}
else { // if condition is not respected
if (event[4] < 0.779060){
// This is a leaf node
result += 0.150238;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009273;
}
}
}
else { // if condition is not respected
if (event[4] < 0.813669){
if (event[1] < -0.119768){
// This is a leaf node
result += -0.004339;
}
else { // if condition is not respected
// This is a leaf node
result += -0.031187;
}
}
else { // if condition is not respected
if (event[2] < -0.852295){
// This is a leaf node
result += 0.002717;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000888;
}
}
}
if (event[1] < 0.526281){
if (event[1] < 0.526070){
if (event[1] < 0.501719){
// This is a leaf node
result += -0.000108;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005894;
}
}
else { // if condition is not respected
if (event[2] < -0.661807){
// This is a leaf node
result += 0.015769;
}
else { // if condition is not respected
// This is a leaf node
result += -0.098522;
}
}
}
else { // if condition is not respected
if (event[1] < 0.551360){
if (event[2] < -2.499369){
// This is a leaf node
result += -0.097101;
}
else { // if condition is not respected
// This is a leaf node
result += 0.008683;
}
}
else { // if condition is not respected
if (event[1] < 0.551674){
// This is a leaf node
result += -0.086325;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000249;
}
}
}
if (event[0] < 4.010478){
if (event[0] < 3.515198){
if (event[0] < 3.430337){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.069158;
}
}
else { // if condition is not respected
if (event[2] < 0.882167){
// This is a leaf node
result += -0.056952;
}
else { // if condition is not respected
// This is a leaf node
result += 0.098059;
}
}
}
else { // if condition is not respected
if (event[1] < 0.520747){
// This is a leaf node
result += -0.007612;
}
else { // if condition is not respected
// This is a leaf node
result += 0.085073;
}
}
if (event[2] < 4.023448){
if (event[2] < 3.986183){
if (event[2] < 3.831835){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.052000;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.090052;
}
}
else { // if condition is not respected
if (event[4] < -0.005265){
// This is a leaf node
result += -0.009083;
}
else { // if condition is not respected
// This is a leaf node
result += 0.084085;
}
}
if (event[1] < 4.088089){
if (event[1] < 4.035286){
if (event[1] < 3.711486){
// This is a leaf node
result += 0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += -0.030780;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.090074;
}
}
else { // if condition is not respected
if (event[0] < -0.139709){
// This is a leaf node
result += -0.012922;
}
else { // if condition is not respected
// This is a leaf node
result += -0.065111;
}
}
if (event[3] < -3.993109){
if (event[2] < 0.081382){
// This is a leaf node
result += 0.028623;
}
else { // if condition is not respected
// This is a leaf node
result += -0.097909;
}
}
else { // if condition is not respected
if (event[3] < -3.681860){
if (event[0] < 0.110352){
// This is a leaf node
result += 0.000970;
}
else { // if condition is not respected
// This is a leaf node
result += 0.083101;
}
}
else { // if condition is not respected
if (event[3] < -3.658852){
// This is a leaf node
result += -0.073169;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000001;
}
}
}
if (event[2] < -2.622372){
if (event[4] < 1.809290){
if (event[2] < -2.724762){
// This is a leaf node
result += 0.000560;
}
else { // if condition is not respected
// This is a leaf node
result += -0.023026;
}
}
else { // if condition is not respected
if (event[0] < -1.062392){
// This is a leaf node
result += -0.026809;
}
else { // if condition is not respected
// This is a leaf node
result += 0.067222;
}
}
}
else { // if condition is not respected
if (event[2] < -2.617360){
if (event[0] < 0.382310){
// This is a leaf node
result += 0.111244;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006479;
}
}
else { // if condition is not respected
if (event[2] < -2.613911){
// This is a leaf node
result += -0.060701;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000017;
}
}
}
if (event[4] < 1.671161){
if (event[4] < 1.578457){
if (event[4] < 1.500528){
// This is a leaf node
result += 0.000045;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005208;
}
}
else { // if condition is not respected
if (event[4] < 1.611857){
// This is a leaf node
result += 0.017862;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000669;
}
}
}
else { // if condition is not respected
if (event[4] < 1.673165){
if (event[4] < 1.672220){
// This is a leaf node
result += -0.012193;
}
else { // if condition is not respected
// This is a leaf node
result += -0.097627;
}
}
else { // if condition is not respected
if (event[0] < 1.727109){
// This is a leaf node
result += -0.001705;
}
else { // if condition is not respected
// This is a leaf node
result += 0.014331;
}
}
}
if (event[0] < 1.275674){
if (event[0] < 1.182472){
if (event[0] < 1.182188){
// This is a leaf node
result += -0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.059076;
}
}
else { // if condition is not respected
if (event[0] < 1.188806){
// This is a leaf node
result += 0.028568;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003483;
}
}
}
else { // if condition is not respected
if (event[3] < -1.210643){
if (event[3] < -1.224594){
// This is a leaf node
result += -0.006428;
}
else { // if condition is not respected
// This is a leaf node
result += -0.049386;
}
}
else { // if condition is not respected
if (event[0] < 1.414653){
// This is a leaf node
result += -0.004745;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001349;
}
}
}
if (event[0] < 2.244000){
if (event[0] < 2.242428){
if (event[0] < 2.141204){
// This is a leaf node
result += -0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007162;
}
}
else { // if condition is not respected
if (event[4] < 0.613621){
// This is a leaf node
result += 0.104084;
}
else { // if condition is not respected
// This is a leaf node
result += -0.022422;
}
}
}
else { // if condition is not respected
if (event[0] < 2.248870){
if (event[2] < 0.454419){
// This is a leaf node
result += -0.082690;
}
else { // if condition is not respected
// This is a leaf node
result += 0.052120;
}
}
else { // if condition is not respected
if (event[2] < -2.342321){
// This is a leaf node
result += -0.053525;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001474;
}
}
}
if (event[3] < -1.971181){
if (event[3] < -2.086916){
if (event[4] < 2.533240){
// This is a leaf node
result += -0.000157;
}
else { // if condition is not respected
// This is a leaf node
result += -0.068827;
}
}
else { // if condition is not respected
if (event[0] < 0.944726){
// This is a leaf node
result += 0.012691;
}
else { // if condition is not respected
// This is a leaf node
result += -0.009975;
}
}
}
else { // if condition is not respected
if (event[3] < -1.970558){
if (event[0] < -0.643196){
// This is a leaf node
result += -0.012371;
}
else { // if condition is not respected
// This is a leaf node
result += -0.112191;
}
}
else { // if condition is not respected
if (event[3] < -1.929478){
// This is a leaf node
result += -0.008607;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000019;
}
}
}
if (event[1] < -2.997432){
if (event[2] < 1.589043){
if (event[3] < 1.145397){
// This is a leaf node
result += -0.018098;
}
else { // if condition is not respected
// This is a leaf node
result += 0.032997;
}
}
else { // if condition is not respected
if (event[4] < 0.273185){
// This is a leaf node
result += 0.008113;
}
else { // if condition is not respected
// This is a leaf node
result += 0.116068;
}
}
}
else { // if condition is not respected
if (event[1] < -2.845962){
if (event[4] < 0.781750){
// This is a leaf node
result += 0.034171;
}
else { // if condition is not respected
// This is a leaf node
result += -0.032712;
}
}
else { // if condition is not respected
if (event[1] < -2.806640){
// This is a leaf node
result += -0.039359;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000005;
}
}
}
if (event[1] < 3.460927){
if (event[1] < 3.154752){
if (event[1] < 3.134330){
// This is a leaf node
result += -0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += 0.088628;
}
}
else { // if condition is not respected
if (event[2] < -0.601957){
// This is a leaf node
result += -0.071339;
}
else { // if condition is not respected
// This is a leaf node
result += 0.008956;
}
}
}
else { // if condition is not respected
if (event[2] < -0.075205){
if (event[2] < -0.468379){
// This is a leaf node
result += 0.024094;
}
else { // if condition is not respected
// This is a leaf node
result += -0.105391;
}
}
else { // if condition is not respected
if (event[1] < 3.658014){
// This is a leaf node
result += 0.081808;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005045;
}
}
}
if (event[3] < 3.492048){
if (event[3] < 3.262835){
if (event[3] < 3.229567){
// This is a leaf node
result += -0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += -0.057554;
}
}
else { // if condition is not respected
if (event[2] < 0.152876){
// This is a leaf node
result += 0.064041;
}
else { // if condition is not respected
// This is a leaf node
result += -0.016007;
}
}
}
else { // if condition is not respected
if (event[0] < -0.929110){
if (event[2] < 0.259773){
// This is a leaf node
result += -0.115924;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007062;
}
}
else { // if condition is not respected
if (event[4] < -1.602765){
// This is a leaf node
result += -0.088578;
}
else { // if condition is not respected
// This is a leaf node
result += 0.013137;
}
}
}
if (event[3] < -2.278308){
if (event[0] < -1.516241){
if (event[3] < -2.298016){
// This is a leaf node
result += -0.031706;
}
else { // if condition is not respected
// This is a leaf node
result += 0.103539;
}
}
else { // if condition is not respected
if (event[3] < -2.328769){
// This is a leaf node
result += 0.001981;
}
else { // if condition is not respected
// This is a leaf node
result += 0.022806;
}
}
}
else { // if condition is not respected
if (event[3] < -2.277082){
if (event[1] < 0.267636){
// This is a leaf node
result += -0.125833;
}
else { // if condition is not respected
// This is a leaf node
result += 0.014885;
}
}
else { // if condition is not respected
if (event[3] < -2.128897){
// This is a leaf node
result += -0.006803;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000011;
}
}
}
if (event[0] < -1.241425){
if (event[0] < -1.241803){
if (event[3] < 2.354836){
// This is a leaf node
result += 0.000955;
}
else { // if condition is not respected
// This is a leaf node
result += -0.018674;
}
}
else { // if condition is not respected
if (event[4] < -0.877818){
// This is a leaf node
result += -0.000152;
}
else { // if condition is not respected
// This is a leaf node
result += 0.100650;
}
}
}
else { // if condition is not respected
if (event[0] < -1.239173){
if (event[1] < -0.536993){
// This is a leaf node
result += 0.012592;
}
else { // if condition is not respected
// This is a leaf node
result += -0.062818;
}
}
else { // if condition is not respected
if (event[0] < -1.237443){
// This is a leaf node
result += 0.026184;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000087;
}
}
}
if (event[4] < -2.843313){
if (event[2] < -0.321356){
if (event[3] < 0.395150){
// This is a leaf node
result += -0.010965;
}
else { // if condition is not respected
// This is a leaf node
result += 0.050696;
}
}
else { // if condition is not respected
if (event[3] < -0.462046){
// This is a leaf node
result += 0.007184;
}
else { // if condition is not respected
// This is a leaf node
result += -0.024345;
}
}
}
else { // if condition is not respected
if (event[4] < -2.817898){
if (event[2] < -1.422541){
// This is a leaf node
result += -0.081772;
}
else { // if condition is not respected
// This is a leaf node
result += 0.055688;
}
}
else { // if condition is not respected
if (event[4] < -2.794321){
// This is a leaf node
result += -0.034901;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000013;
}
}
}
if (event[4] < -3.274415){
if (event[4] < -3.368190){
if (event[1] < 0.038694){
// This is a leaf node
result += -0.035412;
}
else { // if condition is not respected
// This is a leaf node
result += 0.018231;
}
}
else { // if condition is not respected
if (event[0] < 0.823460){
// This is a leaf node
result += 0.083632;
}
else { // if condition is not respected
// This is a leaf node
result += -0.039633;
}
}
}
else { // if condition is not respected
if (event[4] < -3.226538){
if (event[1] < 1.106959){
// This is a leaf node
result += -0.065926;
}
else { // if condition is not respected
// This is a leaf node
result += 0.042589;
}
}
else { // if condition is not respected
if (event[4] < -2.843313){
// This is a leaf node
result += -0.008761;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000011;
}
}
}
if (event[2] < -0.258216){
if (event[2] < -0.262831){
if (event[3] < 1.645020){
// This is a leaf node
result += -0.000485;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003791;
}
}
else { // if condition is not respected
if (event[1] < 1.881961){
// This is a leaf node
result += -0.012825;
}
else { // if condition is not respected
// This is a leaf node
result += -0.089278;
}
}
}
else { // if condition is not respected
if (event[2] < -0.251471){
if (event[4] < 0.802055){
// This is a leaf node
result += 0.022944;
}
else { // if condition is not respected
// This is a leaf node
result += -0.020764;
}
}
else { // if condition is not respected
if (event[2] < -0.250950){
// This is a leaf node
result += -0.049402;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000183;
}
}
}
if (event[3] < 1.568337){
if (event[3] < 1.427665){
if (event[3] < 1.427438){
// This is a leaf node
result += -0.000023;
}
else { // if condition is not respected
// This is a leaf node
result += -0.110718;
}
}
else { // if condition is not respected
if (event[3] < 1.440740){
// This is a leaf node
result += 0.023652;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002512;
}
}
}
else { // if condition is not respected
if (event[3] < 1.648609){
if (event[2] < -2.690231){
// This is a leaf node
result += 0.103155;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010157;
}
}
else { // if condition is not respected
if (event[2] < -0.791153){
// This is a leaf node
result += 0.006918;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001390;
}
}
}
if (event[3] < 0.355272){
if (event[3] < 0.347621){
if (event[3] < 0.347234){
// This is a leaf node
result += 0.000178;
}
else { // if condition is not respected
// This is a leaf node
result += -0.065354;
}
}
else { // if condition is not respected
if (event[2] < 1.429214){
// This is a leaf node
result += 0.006135;
}
else { // if condition is not respected
// This is a leaf node
result += 0.058883;
}
}
}
else { // if condition is not respected
if (event[3] < 0.360086){
if (event[4] < 0.892743){
// This is a leaf node
result += -0.013427;
}
else { // if condition is not respected
// This is a leaf node
result += -0.069809;
}
}
else { // if condition is not respected
if (event[3] < 0.360139){
// This is a leaf node
result += 0.100471;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000263;
}
}
}
if (event[3] < 1.849256){
if (event[3] < 1.796716){
if (event[3] < 1.795864){
// This is a leaf node
result += -0.000004;
}
else { // if condition is not respected
// This is a leaf node
result += -0.083950;
}
}
else { // if condition is not respected
if (event[2] < 1.427331){
// This is a leaf node
result += 0.019331;
}
else { // if condition is not respected
// This is a leaf node
result += -0.041254;
}
}
}
else { // if condition is not respected
if (event[1] < -0.915823){
if (event[1] < -1.227747){
// This is a leaf node
result += -0.003192;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020761;
}
}
else { // if condition is not respected
if (event[1] < -0.775954){
// This is a leaf node
result += -0.030391;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001737;
}
}
}
if (event[1] < 2.366266){
if (event[1] < 2.335522){
if (event[1] < 2.334934){
// This is a leaf node
result += -0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += 0.095454;
}
}
else { // if condition is not respected
if (event[2] < -1.544207){
// This is a leaf node
result += 0.039235;
}
else { // if condition is not respected
// This is a leaf node
result += -0.041861;
}
}
}
else { // if condition is not respected
if (event[0] < 1.011447){
if (event[0] < 0.922028){
// This is a leaf node
result += 0.004222;
}
else { // if condition is not respected
// This is a leaf node
result += 0.051695;
}
}
else { // if condition is not respected
if (event[4] < 1.929524){
// This is a leaf node
result += -0.008858;
}
else { // if condition is not respected
// This is a leaf node
result += -0.104694;
}
}
}
if (event[1] < 2.818218){
if (event[1] < 2.779244){
if (event[1] < 2.762832){
// This is a leaf node
result += 0.000010;
}
else { // if condition is not respected
// This is a leaf node
result += -0.039574;
}
}
else { // if condition is not respected
if (event[1] < 2.782566){
// This is a leaf node
result += 0.128296;
}
else { // if condition is not respected
// This is a leaf node
result += 0.019149;
}
}
}
else { // if condition is not respected
if (event[3] < -1.319610){
if (event[0] < 0.693506){
// This is a leaf node
result += 0.058024;
}
else { // if condition is not respected
// This is a leaf node
result += -0.045781;
}
}
else { // if condition is not respected
if (event[3] < -0.398717){
// This is a leaf node
result += -0.035950;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000210;
}
}
}
if (event[0] < 3.125710){
if (event[0] < 3.107625){
if (event[0] < 3.100054){
// This is a leaf node
result += -0.000008;
}
else { // if condition is not respected
// This is a leaf node
result += 0.068389;
}
}
else { // if condition is not respected
if (event[4] < 0.368731){
// This is a leaf node
result += -0.022154;
}
else { // if condition is not respected
// This is a leaf node
result += -0.126075;
}
}
}
else { // if condition is not respected
if (event[0] < 3.154023){
if (event[0] < 3.147211){
// This is a leaf node
result += 0.026830;
}
else { // if condition is not respected
// This is a leaf node
result += 0.139012;
}
}
else { // if condition is not respected
if (event[1] < -0.558785){
// This is a leaf node
result += 0.046405;
}
else { // if condition is not respected
// This is a leaf node
result += -0.013671;
}
}
}
if (event[4] < 3.594640){
if (event[4] < 3.471214){
if (event[4] < 3.403118){
// This is a leaf node
result += 0.000004;
}
else { // if condition is not respected
// This is a leaf node
result += -0.043623;
}
}
else { // if condition is not respected
if (event[2] < 0.940680){
// This is a leaf node
result += 0.064813;
}
else { // if condition is not respected
// This is a leaf node
result += -0.060293;
}
}
}
else { // if condition is not respected
if (event[4] < 3.647938){
if (event[1] < 0.597298){
// This is a leaf node
result += -0.127819;
}
else { // if condition is not respected
// This is a leaf node
result += -0.026693;
}
}
else { // if condition is not respected
if (event[1] < 0.768242){
// This is a leaf node
result += -0.019022;
}
else { // if condition is not respected
// This is a leaf node
result += 0.058204;
}
}
}
if (event[3] < -3.894619){
if (event[2] < -0.431760){
// This is a leaf node
result += 0.050059;
}
else { // if condition is not respected
if (event[0] < -0.002408){
// This is a leaf node
result += -0.000151;
}
else { // if condition is not respected
// This is a leaf node
result += -0.099716;
}
}
}
else { // if condition is not respected
if (event[3] < -3.681860){
if (event[4] < -0.907245){
// This is a leaf node
result += -0.033590;
}
else { // if condition is not respected
// This is a leaf node
result += 0.087780;
}
}
else { // if condition is not respected
if (event[3] < -3.658852){
// This is a leaf node
result += -0.068828;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000001;
}
}
}
if (event[4] < -0.370966){
if (event[4] < -0.371560){
if (event[0] < 0.461815){
// This is a leaf node
result += -0.001045;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001199;
}
}
else { // if condition is not respected
if (event[1] < 1.175406){
// This is a leaf node
result += -0.067140;
}
else { // if condition is not respected
// This is a leaf node
result += 0.047451;
}
}
}
else { // if condition is not respected
if (event[4] < -0.363685){
if (event[3] < -1.738977){
// This is a leaf node
result += 0.074704;
}
else { // if condition is not respected
// This is a leaf node
result += 0.010136;
}
}
else { // if condition is not respected
if (event[4] < -0.362103){
// This is a leaf node
result += -0.034525;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000176;
}
}
}
if (event[0] < 0.281172){
if (event[0] < 0.279190){
if (event[0] < 0.004480){
// This is a leaf node
result += -0.000226;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002168;
}
}
else { // if condition is not respected
if (event[2] < 2.060161){
// This is a leaf node
result += 0.031076;
}
else { // if condition is not respected
// This is a leaf node
result += -0.109601;
}
}
}
else { // if condition is not respected
if (event[0] < 0.293038){
if (event[1] < 2.381957){
// This is a leaf node
result += -0.013572;
}
else { // if condition is not respected
// This is a leaf node
result += -0.116672;
}
}
else { // if condition is not respected
if (event[0] < 0.395085){
// This is a leaf node
result += -0.003567;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000172;
}
}
}
if (event[0] < 1.266979){
if (event[0] < 1.262777){
if (event[0] < 1.260728){
// This is a leaf node
result += 0.000077;
}
else { // if condition is not respected
// This is a leaf node
result += -0.033405;
}
}
else { // if condition is not respected
if (event[2] < 0.222241){
// This is a leaf node
result += 0.046624;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002811;
}
}
}
else { // if condition is not respected
if (event[3] < -1.472573){
if (event[3] < -2.255479){
// This is a leaf node
result += 0.008344;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011701;
}
}
else { // if condition is not respected
if (event[3] < -1.445981){
// This is a leaf node
result += 0.029056;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000363;
}
}
}
if (event[3] < -1.894090){
if (event[2] < 2.762845){
if (event[3] < -1.904364){
// This is a leaf node
result += 0.001228;
}
else { // if condition is not respected
// This is a leaf node
result += 0.026226;
}
}
else { // if condition is not respected
if (event[4] < -0.128799){
// This is a leaf node
result += 0.002932;
}
else { // if condition is not respected
// This is a leaf node
result += -0.110798;
}
}
}
else { // if condition is not respected
if (event[3] < -1.893658){
if (event[0] < -0.436030){
// This is a leaf node
result += -0.012564;
}
else { // if condition is not respected
// This is a leaf node
result += -0.110286;
}
}
else { // if condition is not respected
if (event[3] < -1.859238){
// This is a leaf node
result += -0.007778;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000026;
}
}
}
if (event[2] < 2.972512){
if (event[2] < 2.952371){
if (event[2] < 2.936414){
// This is a leaf node
result += -0.000010;
}
else { // if condition is not respected
// This is a leaf node
result += 0.047514;
}
}
else { // if condition is not respected
if (event[3] < 1.457287){
// This is a leaf node
result += -0.068253;
}
else { // if condition is not respected
// This is a leaf node
result += 0.062803;
}
}
}
else { // if condition is not respected
if (event[2] < 3.042542){
if (event[2] < 3.038306){
// This is a leaf node
result += 0.030787;
}
else { // if condition is not respected
// This is a leaf node
result += 0.138453;
}
}
else { // if condition is not respected
if (event[1] < -0.114551){
// This is a leaf node
result += 0.020124;
}
else { // if condition is not respected
// This is a leaf node
result += -0.022754;
}
}
}
if (event[4] < -1.646589){
if (event[1] < 2.875306){
if (event[0] < -2.879530){
// This is a leaf node
result += 0.062322;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000888;
}
}
else { // if condition is not respected
if (event[4] < -2.449046){
// This is a leaf node
result += -0.003104;
}
else { // if condition is not respected
// This is a leaf node
result += 0.086513;
}
}
}
else { // if condition is not respected
if (event[4] < -1.645776){
if (event[3] < 0.511981){
// This is a leaf node
result += -0.133549;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003227;
}
}
else { // if condition is not respected
if (event[4] < -1.605698){
// This is a leaf node
result += -0.009932;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000010;
}
}
}
if (event[1] < -0.618571){
if (event[4] < 1.859754){
if (event[4] < 1.857151){
// This is a leaf node
result += 0.000642;
}
else { // if condition is not respected
// This is a leaf node
result += 0.103728;
}
}
else { // if condition is not respected
if (event[4] < 2.344918){
// This is a leaf node
result += -0.011993;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005950;
}
}
}
else { // if condition is not respected
if (event[1] < -0.618062){
if (event[0] < 1.228240){
// This is a leaf node
result += -0.054351;
}
else { // if condition is not respected
// This is a leaf node
result += 0.068701;
}
}
else { // if condition is not respected
if (event[1] < -0.617935){
// This is a leaf node
result += 0.094639;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000156;
}
}
}
if (event[1] < 0.525445){
if (event[1] < 0.524623){
if (event[0] < -3.168697){
// This is a leaf node
result += -0.020416;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000145;
}
}
else { // if condition is not respected
if (event[2] < 0.383502){
// This is a leaf node
result += -0.066702;
}
else { // if condition is not respected
// This is a leaf node
result += 0.027079;
}
}
}
else { // if condition is not respected
if (event[1] < 0.525524){
if (event[0] < 0.255128){
// This is a leaf node
result += 0.143703;
}
else { // if condition is not respected
// This is a leaf node
result += 0.022676;
}
}
else { // if condition is not respected
if (event[4] < -0.156816){
// This is a leaf node
result += -0.000999;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001481;
}
}
}
if (event[4] < 0.542113){
if (event[4] < 0.496927){
if (event[4] < 0.496569){
// This is a leaf node
result += 0.000067;
}
else { // if condition is not respected
// This is a leaf node
result += -0.085198;
}
}
else { // if condition is not respected
if (event[1] < -1.450832){
// This is a leaf node
result += 0.030083;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003712;
}
}
}
else { // if condition is not respected
if (event[3] < 2.992590){
if (event[3] < 2.731960){
// This is a leaf node
result += -0.000393;
}
else { // if condition is not respected
// This is a leaf node
result += -0.038427;
}
}
else { // if condition is not respected
if (event[2] < -0.983874){
// This is a leaf node
result += -0.036475;
}
else { // if condition is not respected
// This is a leaf node
result += 0.041601;
}
}
}
if (event[0] < -3.528087){
if (event[3] < 0.561117){
if (event[4] < -0.275777){
// This is a leaf node
result += 0.094423;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002001;
}
}
else { // if condition is not respected
if (event[3] < 1.388737){
// This is a leaf node
result += -0.102835;
}
else { // if condition is not respected
// This is a leaf node
result += 0.055949;
}
}
}
else { // if condition is not respected
if (event[0] < -3.476108){
if (event[2] < 0.498262){
// This is a leaf node
result += -0.118646;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000850;
}
}
else { // if condition is not respected
if (event[1] < 1.398814){
// This is a leaf node
result += 0.000081;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000928;
}
}
}
if (event[3] < 3.367037){
if (event[3] < 3.262835){
if (event[3] < 3.229567){
// This is a leaf node
result += 0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += -0.053521;
}
}
else { // if condition is not respected
if (event[1] < 0.271883){
// This is a leaf node
result += 0.005487;
}
else { // if condition is not respected
// This is a leaf node
result += 0.107075;
}
}
}
else { // if condition is not respected
if (event[0] < 0.397784){
if (event[2] < 0.309631){
// This is a leaf node
result += 0.048058;
}
else { // if condition is not respected
// This is a leaf node
result += -0.056666;
}
}
else { // if condition is not respected
if (event[0] < 1.635993){
// This is a leaf node
result += -0.066510;
}
else { // if condition is not respected
// This is a leaf node
result += 0.031423;
}
}
}
if (event[0] < 4.111604){
if (event[0] < 3.515198){
if (event[0] < 3.430337){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.063967;
}
}
else { // if condition is not respected
if (event[2] < 0.882167){
// This is a leaf node
result += -0.049491;
}
else { // if condition is not respected
// This is a leaf node
result += 0.098060;
}
}
}
else { // if condition is not respected
if (event[0] < 4.301860){
// This is a leaf node
result += 0.087247;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003552;
}
}
if (event[0] < -1.241486){
if (event[0] < -1.241803){
if (event[2] < 2.408053){
// This is a leaf node
result += 0.000892;
}
else { // if condition is not respected
// This is a leaf node
result += -0.019682;
}
}
else { // if condition is not respected
if (event[2] < 0.452665){
// This is a leaf node
result += 0.106985;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012901;
}
}
}
else { // if condition is not respected
if (event[0] < -1.239173){
if (event[4] < -0.469205){
// This is a leaf node
result += -0.098353;
}
else { // if condition is not respected
// This is a leaf node
result += -0.016636;
}
}
else { // if condition is not respected
if (event[0] < -1.127390){
// This is a leaf node
result += -0.003098;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000003;
}
}
}
if (event[2] < 4.023448){
if (event[2] < 3.986183){
if (event[2] < 3.831835){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.049659;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.085990;
}
}
else { // if condition is not respected
if (event[4] < -0.480630){
// This is a leaf node
result += -0.021046;
}
else { // if condition is not respected
if (event[3] < -0.255596){
// This is a leaf node
result += 0.015714;
}
else { // if condition is not respected
// This is a leaf node
result += 0.091878;
}
}
}
if (event[2] < 0.710054){
if (event[2] < 0.705562){
if (event[2] < 0.705247){
// This is a leaf node
result += -0.000112;
}
else { // if condition is not respected
// This is a leaf node
result += 0.051452;
}
}
else { // if condition is not respected
if (event[2] < 0.705711){
// This is a leaf node
result += -0.116977;
}
else { // if condition is not respected
// This is a leaf node
result += -0.020573;
}
}
}
else { // if condition is not respected
if (event[2] < 0.710336){
if (event[1] < 0.038296){
// This is a leaf node
result += 0.105117;
}
else { // if condition is not respected
// This is a leaf node
result += 0.014104;
}
}
else { // if condition is not respected
if (event[1] < 1.440378){
// This is a leaf node
result += 0.000806;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003967;
}
}
}
if (event[2] < 1.368580){
if (event[2] < 1.272424){
if (event[2] < 1.272307){
// This is a leaf node
result += -0.000037;
}
else { // if condition is not respected
// This is a leaf node
result += -0.079249;
}
}
else { // if condition is not respected
if (event[4] < -0.975827){
// This is a leaf node
result += -0.007181;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009746;
}
}
}
else { // if condition is not respected
if (event[4] < -0.718483){
if (event[1] < 0.648544){
// This is a leaf node
result += 0.006222;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004634;
}
}
else { // if condition is not respected
if (event[4] < -0.714493){
// This is a leaf node
result += -0.076583;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002050;
}
}
}
if (event[1] < 0.525445){
if (event[1] < 0.524623){
if (event[0] < -2.193289){
// This is a leaf node
result += -0.004820;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000101;
}
}
else { // if condition is not respected
if (event[2] < 0.383502){
// This is a leaf node
result += -0.060444;
}
else { // if condition is not respected
// This is a leaf node
result += 0.024567;
}
}
}
else { // if condition is not respected
if (event[1] < 0.525524){
if (event[0] < 0.255128){
// This is a leaf node
result += 0.133868;
}
else { // if condition is not respected
// This is a leaf node
result += 0.021422;
}
}
else { // if condition is not respected
if (event[4] < -0.156816){
// This is a leaf node
result += -0.000843;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001386;
}
}
}
if (event[0] < -1.943462){
if (event[0] < -1.945927){
if (event[2] < -1.676771){
// This is a leaf node
result += 0.019741;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000235;
}
}
else { // if condition is not respected
if (event[1] < -0.505130){
// This is a leaf node
result += -0.012086;
}
else { // if condition is not respected
// This is a leaf node
result += 0.107628;
}
}
}
else { // if condition is not respected
if (event[0] < -1.941848){
if (event[1] < 0.673899){
// This is a leaf node
result += -0.074101;
}
else { // if condition is not respected
// This is a leaf node
result += 0.035143;
}
}
else { // if condition is not respected
if (event[0] < -1.722601){
// This is a leaf node
result += -0.003909;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000028;
}
}
}
if (event[1] < 2.366266){
if (event[1] < 2.337770){
if (event[1] < 2.086814){
// This is a leaf node
result += 0.000032;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003658;
}
}
else { // if condition is not respected
if (event[2] < 0.730138){
// This is a leaf node
result += -0.016442;
}
else { // if condition is not respected
// This is a leaf node
result += -0.084259;
}
}
}
else { // if condition is not respected
if (event[3] < 1.746789){
if (event[1] < 2.366663){
// This is a leaf node
result += 0.120968;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001274;
}
}
else { // if condition is not respected
if (event[1] < 2.746255){
// This is a leaf node
result += 0.065311;
}
else { // if condition is not respected
// This is a leaf node
result += -0.030495;
}
}
}
if (event[0] < 0.281160){
if (event[0] < 0.221412){
if (event[0] < 0.219587){
// This is a leaf node
result += 0.000063;
}
else { // if condition is not respected
// This is a leaf node
result += -0.026414;
}
}
else { // if condition is not respected
if (event[2] < 3.099449){
// This is a leaf node
result += 0.005048;
}
else { // if condition is not respected
// This is a leaf node
result += -0.145964;
}
}
}
else { // if condition is not respected
if (event[0] < 0.343922){
if (event[3] < -2.108124){
// This is a leaf node
result += 0.034999;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006604;
}
}
else { // if condition is not respected
if (event[0] < 0.345214){
// This is a leaf node
result += 0.037712;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000020;
}
}
}
if (event[4] < 2.624323){
if (event[4] < 2.618027){
if (event[4] < 2.107707){
// This is a leaf node
result += -0.000028;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003184;
}
}
else { // if condition is not respected
if (event[4] < 2.619708){
// This is a leaf node
result += 0.139549;
}
else { // if condition is not respected
// This is a leaf node
result += 0.010354;
}
}
}
else { // if condition is not respected
if (event[4] < 2.655832){
if (event[2] < -0.210578){
// This is a leaf node
result += -0.097302;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000271;
}
}
else { // if condition is not respected
if (event[2] < -0.925358){
// This is a leaf node
result += 0.025695;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006452;
}
}
}
if (event[4] < 0.911245){
if (event[4] < 0.911066){
if (event[4] < 0.821965){
// This is a leaf node
result += 0.000028;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003060;
}
}
else { // if condition is not respected
if (event[3] < -0.694308){
// This is a leaf node
result += 0.019609;
}
else { // if condition is not respected
// This is a leaf node
result += 0.120285;
}
}
}
else { // if condition is not respected
if (event[4] < 0.925565){
if (event[4] < 0.924505){
// This is a leaf node
result += -0.007990;
}
else { // if condition is not respected
// This is a leaf node
result += -0.066169;
}
}
else { // if condition is not respected
if (event[4] < 0.940455){
// This is a leaf node
result += 0.009406;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000531;
}
}
}
if (event[3] < -2.862381){
if (event[1] < -1.469514){
if (event[0] < 0.534804){
// This is a leaf node
result += -0.034159;
}
else { // if condition is not respected
// This is a leaf node
result += -0.121336;
}
}
else { // if condition is not respected
if (event[3] < -2.900259){
// This is a leaf node
result += 0.003249;
}
else { // if condition is not respected
// This is a leaf node
result += -0.051872;
}
}
}
else { // if condition is not respected
if (event[3] < -2.854355){
if (event[0] < -0.567992){
// This is a leaf node
result += -0.022703;
}
else { // if condition is not respected
// This is a leaf node
result += 0.106878;
}
}
else { // if condition is not respected
if (event[3] < -2.639947){
// This is a leaf node
result += 0.010441;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000013;
}
}
}
if (event[3] < 0.606942){
if (event[3] < 0.605068){
if (event[3] < 0.487732){
// This is a leaf node
result += -0.000025;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002874;
}
}
else { // if condition is not respected
if (event[2] < -0.079853){
// This is a leaf node
result += 0.003808;
}
else { // if condition is not respected
// This is a leaf node
result += 0.058574;
}
}
}
else { // if condition is not respected
if (event[3] < 0.614468){
if (event[2] < 0.706984){
// This is a leaf node
result += -0.025205;
}
else { // if condition is not respected
// This is a leaf node
result += 0.019051;
}
}
else { // if condition is not respected
if (event[3] < 0.614532){
// This is a leaf node
result += 0.125556;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000311;
}
}
}
if (event[3] < 1.045245){
if (event[3] < 1.044841){
if (event[3] < 0.980328){
// This is a leaf node
result += -0.000016;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004658;
}
}
else { // if condition is not respected
if (event[4] < -0.049362){
// This is a leaf node
result += -0.013282;
}
else { // if condition is not respected
// This is a leaf node
result += -0.138147;
}
}
}
else { // if condition is not respected
if (event[1] < -2.833028){
if (event[1] < -2.869112){
// This is a leaf node
result += 0.028590;
}
else { // if condition is not respected
// This is a leaf node
result += 0.104797;
}
}
else { // if condition is not respected
if (event[3] < 1.047289){
// This is a leaf node
result += 0.031039;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000438;
}
}
}
if (event[1] < -3.123382){
if (event[1] < -3.127925){
if (event[2] < 1.727619){
// This is a leaf node
result += -0.009921;
}
else { // if condition is not respected
// This is a leaf node
result += 0.078227;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.132001;
}
}
else { // if condition is not respected
if (event[1] < -3.113010){
if (event[0] < 0.580553){
// This is a leaf node
result += 0.132594;
}
else { // if condition is not respected
// This is a leaf node
result += -0.035545;
}
}
else { // if condition is not respected
if (event[1] < -2.593957){
// This is a leaf node
result += 0.005383;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000015;
}
}
}
if (event[4] < -3.274415){
if (event[4] < -3.368190){
if (event[0] < 0.359604){
// This is a leaf node
result += -0.025813;
}
else { // if condition is not respected
// This is a leaf node
result += 0.024991;
}
}
else { // if condition is not respected
if (event[0] < 0.823460){
// This is a leaf node
result += 0.075707;
}
else { // if condition is not respected
// This is a leaf node
result += -0.036257;
}
}
}
else { // if condition is not respected
if (event[4] < -3.264382){
if (event[1] < -0.380114){
// This is a leaf node
result += -0.012754;
}
else { // if condition is not respected
// This is a leaf node
result += -0.097444;
}
}
else { // if condition is not respected
if (event[4] < -3.256793){
// This is a leaf node
result += 0.094085;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000005;
}
}
}
if (event[2] < 2.972512){
if (event[2] < 2.952371){
if (event[2] < 2.936414){
// This is a leaf node
result += -0.000009;
}
else { // if condition is not respected
// This is a leaf node
result += 0.043718;
}
}
else { // if condition is not respected
if (event[4] < -1.331602){
// This is a leaf node
result += 0.043698;
}
else { // if condition is not respected
// This is a leaf node
result += -0.065921;
}
}
}
else { // if condition is not respected
if (event[2] < 3.018909){
if (event[3] < -1.281459){
// This is a leaf node
result += -0.032635;
}
else { // if condition is not respected
// This is a leaf node
result += 0.062858;
}
}
else { // if condition is not respected
if (event[1] < -0.114551){
// This is a leaf node
result += 0.021105;
}
else { // if condition is not respected
// This is a leaf node
result += -0.018252;
}
}
}
if (event[0] < -1.292493){
if (event[0] < -1.326237){
if (event[4] < 2.285236){
// This is a leaf node
result += -0.000009;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020371;
}
}
else { // if condition is not respected
if (event[1] < 2.329020){
// This is a leaf node
result += 0.010993;
}
else { // if condition is not respected
// This is a leaf node
result += -0.075092;
}
}
}
else { // if condition is not respected
if (event[0] < -1.292175){
if (event[3] < -0.530190){
// This is a leaf node
result += 0.025292;
}
else { // if condition is not respected
// This is a leaf node
result += -0.107074;
}
}
else { // if condition is not respected
if (event[0] < -1.292105){
// This is a leaf node
result += 0.112624;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000084;
}
}
}
if (event[4] < 0.542113){
if (event[4] < 0.536056){
if (event[4] < 0.536010){
// This is a leaf node
result += 0.000133;
}
else { // if condition is not respected
// This is a leaf node
result += -0.094350;
}
}
else { // if condition is not respected
if (event[2] < 1.380405){
// This is a leaf node
result += 0.017775;
}
else { // if condition is not respected
// This is a leaf node
result += -0.028102;
}
}
}
else { // if condition is not respected
if (event[2] < -1.873970){
if (event[3] < -0.692668){
// This is a leaf node
result += 0.019671;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000177;
}
}
else { // if condition is not respected
if (event[2] < -1.831940){
// This is a leaf node
result += -0.019782;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000511;
}
}
}
if (event[2] < -2.622372){
if (event[2] < -2.625874){
if (event[3] < 2.234444){
// This is a leaf node
result += -0.003837;
}
else { // if condition is not respected
// This is a leaf node
result += 0.078730;
}
}
else { // if condition is not respected
if (event[1] < 0.285565){
// This is a leaf node
result += -0.014421;
}
else { // if condition is not respected
// This is a leaf node
result += -0.145108;
}
}
}
else { // if condition is not respected
if (event[2] < -2.460530){
if (event[4] < 1.676832){
// This is a leaf node
result += 0.013213;
}
else { // if condition is not respected
// This is a leaf node
result += -0.062975;
}
}
else { // if condition is not respected
if (event[2] < -2.456612){
// This is a leaf node
result += -0.064927;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000000;
}
}
}
if (event[2] < -2.304283){
if (event[4] < -1.915505){
if (event[2] < -3.231444){
// This is a leaf node
result += 0.069107;
}
else { // if condition is not respected
// This is a leaf node
result += -0.041869;
}
}
else { // if condition is not respected
if (event[1] < -0.353279){
// This is a leaf node
result += -0.009933;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003236;
}
}
}
else { // if condition is not respected
if (event[2] < -2.302579){
if (event[4] < 0.684738){
// This is a leaf node
result += 0.109667;
}
else { // if condition is not respected
// This is a leaf node
result += 0.010375;
}
}
else { // if condition is not respected
if (event[2] < -2.301145){
// This is a leaf node
result += -0.076632;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000026;
}
}
}
if (event[2] < -3.749223){
if (event[2] < -3.845556){
if (event[0] < -0.624683){
// This is a leaf node
result += -0.070563;
}
else { // if condition is not respected
// This is a leaf node
result += 0.013935;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.093793;
}
}
else { // if condition is not respected
if (event[2] < -3.453882){
if (event[1] < -0.484509){
// This is a leaf node
result += 0.032239;
}
else { // if condition is not respected
// This is a leaf node
result += -0.060560;
}
}
else { // if condition is not respected
if (event[2] < -3.319989){
// This is a leaf node
result += 0.039216;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000005;
}
}
}
if (event[0] < 4.010478){
if (event[0] < 3.689817){
if (event[0] < 3.651816){
// This is a leaf node
result += -0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += 0.094024;
}
}
else { // if condition is not respected
if (event[3] < 1.025447){
// This is a leaf node
result += -0.070191;
}
else { // if condition is not respected
// This is a leaf node
result += 0.038477;
}
}
}
else { // if condition is not respected
if (event[1] < 0.520747){
// This is a leaf node
result += -0.008339;
}
else { // if condition is not respected
// This is a leaf node
result += 0.080057;
}
}
if (event[4] < 3.003102){
if (event[4] < 2.939700){
if (event[4] < 2.905666){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.041947;
}
}
else { // if condition is not respected
if (event[0] < -0.649538){
// This is a leaf node
result += 0.093319;
}
else { // if condition is not respected
// This is a leaf node
result += 0.024643;
}
}
}
else { // if condition is not respected
if (event[1] < 0.990869){
if (event[3] < -0.080699){
// This is a leaf node
result += -0.033285;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000614;
}
}
else { // if condition is not respected
if (event[4] < 3.188599){
// This is a leaf node
result += -0.005353;
}
else { // if condition is not respected
// This is a leaf node
result += 0.068358;
}
}
}
if (event[1] < 3.524422){
if (event[1] < 3.504731){
if (event[1] < 3.481369){
// This is a leaf node
result += -0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += 0.065585;
}
}
else { // if condition is not respected
if (event[2] < 0.211567){
// This is a leaf node
result += -0.109054;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006455;
}
}
}
else { // if condition is not respected
if (event[1] < 3.587595){
if (event[3] < 0.183371){
// This is a leaf node
result += 0.151312;
}
else { // if condition is not respected
// This is a leaf node
result += -0.065830;
}
}
else { // if condition is not respected
if (event[4] < -0.970480){
// This is a leaf node
result += -0.064636;
}
else { // if condition is not respected
// This is a leaf node
result += 0.008568;
}
}
}
if (event[3] < -2.946279){
if (event[4] < 1.838505){
if (event[2] < 0.672982){
// This is a leaf node
result += -0.000081;
}
else { // if condition is not respected
// This is a leaf node
result += -0.036815;
}
}
else { // if condition is not respected
if (event[4] < 2.427359){
// This is a leaf node
result += 0.113203;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010408;
}
}
}
else { // if condition is not respected
if (event[3] < -2.907827){
if (event[0] < 1.298431){
// This is a leaf node
result += 0.049676;
}
else { // if condition is not respected
// This is a leaf node
result += -0.069984;
}
}
else { // if condition is not respected
if (event[3] < -2.862381){
// This is a leaf node
result += -0.039795;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000011;
}
}
}
if (event[1] < -2.983040){
if (event[3] < 1.097380){
if (event[3] < 0.842587){
// This is a leaf node
result += -0.006809;
}
else { // if condition is not respected
// This is a leaf node
result += -0.090437;
}
}
else { // if condition is not respected
if (event[3] < 1.279186){
// This is a leaf node
result += 0.081058;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007547;
}
}
}
else { // if condition is not respected
if (event[1] < -2.973975){
// This is a leaf node
result += 0.122937;
}
else { // if condition is not respected
if (event[1] < -2.845962){
// This is a leaf node
result += 0.014810;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000004;
}
}
}
if (event[4] < 3.594640){
if (event[4] < 3.579999){
if (event[4] < 3.565615){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.070456;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.082361;
}
}
else { // if condition is not respected
if (event[4] < 3.647938){
if (event[1] < 0.597298){
// This is a leaf node
result += -0.120083;
}
else { // if condition is not respected
// This is a leaf node
result += -0.027949;
}
}
else { // if condition is not respected
if (event[0] < -0.565315){
// This is a leaf node
result += -0.045838;
}
else { // if condition is not respected
// This is a leaf node
result += 0.027440;
}
}
}
if (event[0] < 2.237703){
if (event[0] < 2.236785){
if (event[0] < 2.236049){
// This is a leaf node
result += 0.000028;
}
else { // if condition is not respected
// This is a leaf node
result += -0.092743;
}
}
else { // if condition is not respected
if (event[4] < 0.524864){
// This is a leaf node
result += 0.117863;
}
else { // if condition is not respected
// This is a leaf node
result += -0.023452;
}
}
}
else { // if condition is not respected
if (event[3] < -3.125730){
if (event[3] < -3.297996){
// This is a leaf node
result += 0.033556;
}
else { // if condition is not respected
// This is a leaf node
result += 0.119220;
}
}
else { // if condition is not respected
if (event[3] < -2.592641){
// This is a leaf node
result += -0.088218;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002171;
}
}
}
if (event[0] < 3.125710){
if (event[0] < 3.107625){
if (event[0] < 3.100054){
// This is a leaf node
result += -0.000007;
}
else { // if condition is not respected
// This is a leaf node
result += 0.062786;
}
}
else { // if condition is not respected
if (event[4] < 0.368731){
// This is a leaf node
result += -0.020197;
}
else { // if condition is not respected
// This is a leaf node
result += -0.118298;
}
}
}
else { // if condition is not respected
if (event[0] < 3.154023){
if (event[2] < -0.485501){
// This is a leaf node
result += -0.030854;
}
else { // if condition is not respected
// This is a leaf node
result += 0.099420;
}
}
else { // if condition is not respected
if (event[1] < -0.911887){
// This is a leaf node
result += 0.060926;
}
else { // if condition is not respected
// This is a leaf node
result += -0.008430;
}
}
}
if (event[3] < 0.323977){
if (event[3] < 0.323907){
if (event[3] < 0.323860){
// This is a leaf node
result += 0.000198;
}
else { // if condition is not respected
// This is a leaf node
result += -0.098802;
}
}
else { // if condition is not respected
if (event[0] < 1.085067){
// This is a leaf node
result += 0.132402;
}
else { // if condition is not respected
// This is a leaf node
result += 0.029529;
}
}
}
else { // if condition is not respected
if (event[3] < 0.324561){
if (event[1] < 0.272880){
// This is a leaf node
result += -0.089895;
}
else { // if condition is not respected
// This is a leaf node
result += 0.026106;
}
}
else { // if condition is not respected
if (event[4] < 2.939965){
// This is a leaf node
result += -0.000337;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020202;
}
}
}
if (event[3] < -0.202299){
if (event[3] < -0.202442){
if (event[4] < -1.054039){
// This is a leaf node
result += -0.002313;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000052;
}
}
else { // if condition is not respected
if (event[2] < -0.649355){
// This is a leaf node
result += 0.051271;
}
else { // if condition is not respected
// This is a leaf node
result += -0.150320;
}
}
}
else { // if condition is not respected
if (event[3] < -0.184463){
if (event[3] < -0.184559){
// This is a leaf node
result += 0.007565;
}
else { // if condition is not respected
// This is a leaf node
result += 0.136735;
}
}
else { // if condition is not respected
if (event[3] < -0.177506){
// This is a leaf node
result += -0.009779;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000164;
}
}
}
if (event[4] < -1.646685){
if (event[1] < 2.945819){
if (event[4] < -1.648610){
// This is a leaf node
result += 0.000880;
}
else { // if condition is not respected
// This is a leaf node
result += 0.041723;
}
}
else { // if condition is not respected
if (event[0] < 1.031109){
// This is a leaf node
result += 0.097482;
}
else { // if condition is not respected
// This is a leaf node
result += -0.018032;
}
}
}
else { // if condition is not respected
if (event[4] < -1.641505){
if (event[3] < 1.303070){
// This is a leaf node
result += -0.044786;
}
else { // if condition is not respected
// This is a leaf node
result += 0.052503;
}
}
else { // if condition is not respected
if (event[4] < -1.641255){
// This is a leaf node
result += 0.079609;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000048;
}
}
}
if (event[1] < 2.818218){
if (event[1] < 2.569115){
if (event[1] < 2.536491){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.031599;
}
}
else { // if condition is not respected
if (event[1] < 2.578853){
// This is a leaf node
result += 0.066944;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007163;
}
}
}
else { // if condition is not respected
if (event[3] < 1.892345){
if (event[3] < 1.653552){
// This is a leaf node
result += -0.005193;
}
else { // if condition is not respected
// This is a leaf node
result += 0.096494;
}
}
else { // if condition is not respected
if (event[4] < -0.349983){
// This is a leaf node
result += -0.135112;
}
else { // if condition is not respected
// This is a leaf node
result += -0.009129;
}
}
}
if (event[4] < 3.002024){
if (event[4] < 2.972779){
if (event[4] < 2.968638){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.068565;
}
}
else { // if condition is not respected
if (event[0] < -0.585117){
// This is a leaf node
result += 0.113426;
}
else { // if condition is not respected
// This is a leaf node
result += 0.023610;
}
}
}
else { // if condition is not respected
if (event[1] < 0.990869){
if (event[2] < 1.556756){
// This is a leaf node
result += -0.018336;
}
else { // if condition is not respected
// This is a leaf node
result += 0.044922;
}
}
else { // if condition is not respected
if (event[4] < 3.188599){
// This is a leaf node
result += -0.005683;
}
else { // if condition is not respected
// This is a leaf node
result += 0.061857;
}
}
}
if (event[0] < -1.842866){
if (event[0] < -1.850449){
if (event[3] < -0.677947){
// This is a leaf node
result += 0.006619;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001210;
}
}
else { // if condition is not respected
if (event[1] < 0.487659){
// This is a leaf node
result += 0.058976;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006119;
}
}
}
else { // if condition is not respected
if (event[0] < -1.816692){
if (event[1] < 0.984510){
// This is a leaf node
result += -0.026302;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020100;
}
}
else { // if condition is not respected
if (event[0] < -1.794475){
// This is a leaf node
result += 0.009683;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000027;
}
}
}
if (event[2] < -0.272899){
if (event[2] < -0.277702){
if (event[2] < -0.433119){
// This is a leaf node
result += 0.000114;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002220;
}
}
else { // if condition is not respected
if (event[4] < 0.207572){
// This is a leaf node
result += -0.003786;
}
else { // if condition is not respected
// This is a leaf node
result += -0.039039;
}
}
}
else { // if condition is not respected
if (event[2] < -0.271314){
if (event[0] < 1.277195){
// This is a leaf node
result += 0.032600;
}
else { // if condition is not respected
// This is a leaf node
result += -0.054326;
}
}
else { // if condition is not respected
if (event[2] < -0.271076){
// This is a leaf node
result += -0.048313;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000191;
}
}
}
if (event[3] < -4.097473){
// This is a leaf node
result += -0.058653;
}
else { // if condition is not respected
if (event[3] < -3.406969){
if (event[0] < -1.039718){
// This is a leaf node
result += -0.062422;
}
else { // if condition is not respected
// This is a leaf node
result += 0.034119;
}
}
else { // if condition is not respected
if (event[3] < -2.946279){
// This is a leaf node
result += -0.010594;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000010;
}
}
}
if (event[3] < -3.894619){
if (event[2] < -0.431760){
// This is a leaf node
result += 0.046345;
}
else { // if condition is not respected
if (event[0] < -0.002408){
// This is a leaf node
result += 0.000491;
}
else { // if condition is not respected
// This is a leaf node
result += -0.093680;
}
}
}
else { // if condition is not respected
if (event[3] < -3.681860){
if (event[4] < -0.907245){
// This is a leaf node
result += -0.032257;
}
else { // if condition is not respected
// This is a leaf node
result += 0.081125;
}
}
else { // if condition is not respected
if (event[3] < -3.658852){
// This is a leaf node
result += -0.065970;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000000;
}
}
}
if (event[2] < 1.491608){
if (event[2] < 1.489851){
if (event[2] < 1.488528){
// This is a leaf node
result += 0.000068;
}
else { // if condition is not respected
// This is a leaf node
result += -0.041424;
}
}
else { // if condition is not respected
if (event[2] < 1.490240){
// This is a leaf node
result += 0.132902;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007834;
}
}
}
else { // if condition is not respected
if (event[2] < 1.492989){
if (event[3] < 1.169113){
// This is a leaf node
result += -0.085443;
}
else { // if condition is not respected
// This is a leaf node
result += 0.052619;
}
}
else { // if condition is not respected
if (event[1] < 2.176867){
// This is a leaf node
result += -0.000440;
}
else { // if condition is not respected
// This is a leaf node
result += -0.022293;
}
}
}
if (event[1] < 0.531639){
if (event[1] < 0.531564){
if (event[0] < 1.725214){
// This is a leaf node
result += -0.000281;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002628;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.125704;
}
}
else { // if condition is not respected
if (event[1] < 0.545910){
if (event[2] < -2.533395){
// This is a leaf node
result += -0.121241;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012183;
}
}
else { // if condition is not respected
if (event[1] < 0.546005){
// This is a leaf node
result += -0.092506;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000231;
}
}
}
if (event[0] < 2.027467){
if (event[0] < 2.008068){
if (event[0] < 2.005329){
// This is a leaf node
result += 0.000024;
}
else { // if condition is not respected
// This is a leaf node
result += -0.082172;
}
}
else { // if condition is not respected
if (event[0] < 2.014476){
// This is a leaf node
result += 0.061268;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009600;
}
}
}
else { // if condition is not respected
if (event[0] < 2.028569){
if (event[4] < -1.484969){
// This is a leaf node
result += 0.062652;
}
else { // if condition is not respected
// This is a leaf node
result += -0.141355;
}
}
else { // if condition is not respected
if (event[4] < -2.466976){
// This is a leaf node
result += 0.042312;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001687;
}
}
}
if (event[0] < 1.236130){
if (event[0] < 1.209939){
if (event[0] < 1.209676){
// This is a leaf node
result += 0.000018;
}
else { // if condition is not respected
// This is a leaf node
result += -0.097902;
}
}
else { // if condition is not respected
if (event[0] < 1.214644){
// This is a leaf node
result += 0.039494;
}
else { // if condition is not respected
// This is a leaf node
result += 0.008515;
}
}
}
else { // if condition is not respected
if (event[0] < 1.247642){
if (event[1] < 1.442240){
// This is a leaf node
result += -0.024616;
}
else { // if condition is not respected
// This is a leaf node
result += 0.043747;
}
}
else { // if condition is not respected
if (event[0] < 1.248991){
// This is a leaf node
result += 0.045854;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000466;
}
}
}
if (event[2] < 4.023448){
if (event[2] < 3.986183){
if (event[2] < 3.831835){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.046893;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.082777;
}
}
else { // if condition is not respected
if (event[4] < -0.480630){
// This is a leaf node
result += -0.019813;
}
else { // if condition is not respected
if (event[3] < -0.255596){
// This is a leaf node
result += 0.015067;
}
else { // if condition is not respected
// This is a leaf node
result += 0.087670;
}
}
}
if (event[1] < -0.618571){
if (event[4] < -2.659788){
if (event[4] < -2.690200){
// This is a leaf node
result += 0.010090;
}
else { // if condition is not respected
// This is a leaf node
result += 0.103225;
}
}
else { // if condition is not respected
if (event[4] < -2.576034){
// This is a leaf node
result += -0.054168;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000397;
}
}
}
else { // if condition is not respected
if (event[1] < -0.616454){
if (event[2] < 1.799393){
// This is a leaf node
result += -0.025938;
}
else { // if condition is not respected
// This is a leaf node
result += 0.067022;
}
}
else { // if condition is not respected
if (event[2] < 2.387063){
// This is a leaf node
result += -0.000079;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006293;
}
}
}
if (event[2] < 2.972512){
if (event[2] < 2.806383){
if (event[2] < 2.784039){
// This is a leaf node
result += -0.000007;
}
else { // if condition is not respected
// This is a leaf node
result += 0.047469;
}
}
else { // if condition is not respected
if (event[0] < -0.307300){
// This is a leaf node
result += -0.041250;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004221;
}
}
}
else { // if condition is not respected
if (event[2] < 2.999801){
if (event[3] < 0.291591){
// This is a leaf node
result += 0.009486;
}
else { // if condition is not respected
// This is a leaf node
result += 0.121499;
}
}
else { // if condition is not respected
if (event[4] < -1.448262){
// This is a leaf node
result += -0.052507;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006417;
}
}
}
if (event[4] < -2.590437){
if (event[4] < -2.645596){
if (event[3] < -1.205197){
// This is a leaf node
result += 0.025051;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003380;
}
}
else { // if condition is not respected
if (event[1] < -1.147050){
// This is a leaf node
result += -0.088002;
}
else { // if condition is not respected
// This is a leaf node
result += -0.015024;
}
}
}
else { // if condition is not respected
if (event[4] < -2.556193){
if (event[0] < -1.792234){
// This is a leaf node
result += -0.072939;
}
else { // if condition is not respected
// This is a leaf node
result += 0.037975;
}
}
else { // if condition is not respected
if (event[4] < -2.553907){
// This is a leaf node
result += -0.083937;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000008;
}
}
}
if (event[4] < 2.624323){
if (event[4] < 2.623344){
if (event[4] < 2.254232){
// This is a leaf node
result += -0.000016;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003958;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.106537;
}
}
else { // if condition is not respected
if (event[2] < -0.925358){
if (event[2] < -1.461405){
// This is a leaf node
result += -0.022841;
}
else { // if condition is not respected
// This is a leaf node
result += 0.040016;
}
}
else { // if condition is not respected
if (event[3] < -0.981270){
// This is a leaf node
result += -0.034856;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002904;
}
}
}
if (event[1] < 2.366266){
if (event[1] < 2.337770){
if (event[1] < 2.086814){
// This is a leaf node
result += 0.000027;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003218;
}
}
else { // if condition is not respected
if (event[2] < 0.730138){
// This is a leaf node
result += -0.014824;
}
else { // if condition is not respected
// This is a leaf node
result += -0.076125;
}
}
}
else { // if condition is not respected
if (event[0] < 1.011447){
if (event[0] < 0.922028){
// This is a leaf node
result += 0.003800;
}
else { // if condition is not respected
// This is a leaf node
result += 0.046484;
}
}
else { // if condition is not respected
if (event[4] < 1.929524){
// This is a leaf node
result += -0.007536;
}
else { // if condition is not respected
// This is a leaf node
result += -0.096183;
}
}
}
if (event[4] < -2.843313){
if (event[0] < -0.803646){
if (event[2] < -0.296891){
// This is a leaf node
result += 0.064727;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007491;
}
}
else { // if condition is not respected
if (event[0] < -0.649417){
// This is a leaf node
result += -0.084770;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007551;
}
}
}
else { // if condition is not respected
if (event[4] < -2.817898){
if (event[1] < -0.156040){
// This is a leaf node
result += -0.012578;
}
else { // if condition is not respected
// This is a leaf node
result += 0.070973;
}
}
else { // if condition is not respected
if (event[4] < -2.814892){
// This is a leaf node
result += -0.087372;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000008;
}
}
}
if (event[3] < 3.465692){
if (event[3] < 3.445596){
if (event[3] < 3.377400){
// This is a leaf node
result += 0.000004;
}
else { // if condition is not respected
// This is a leaf node
result += -0.044660;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.104408;
}
}
else { // if condition is not respected
if (event[0] < -0.925448){
if (event[4] < -1.106827){
// This is a leaf node
result += 0.002827;
}
else { // if condition is not respected
// This is a leaf node
result += -0.109821;
}
}
else { // if condition is not respected
if (event[0] < 0.134199){
// This is a leaf node
result += 0.043695;
}
else { // if condition is not respected
// This is a leaf node
result += -0.032804;
}
}
}
if (event[4] < -1.646589){
if (event[1] < 2.875306){
if (event[0] < -1.943911){
// This is a leaf node
result += 0.017488;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000517;
}
}
else { // if condition is not respected
if (event[4] < -2.449046){
// This is a leaf node
result += -0.006592;
}
else { // if condition is not respected
// This is a leaf node
result += 0.075220;
}
}
}
else { // if condition is not respected
if (event[4] < -1.645776){
if (event[3] < 0.511981){
// This is a leaf node
result += -0.120664;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004226;
}
}
else { // if condition is not respected
if (event[4] < -1.605698){
// This is a leaf node
result += -0.008541;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000013;
}
}
}
if (event[1] < 2.818218){
if (event[1] < 2.807080){
if (event[1] < 2.569115){
// This is a leaf node
result += -0.000013;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007741;
}
}
else { // if condition is not respected
if (event[3] < -0.088465){
// This is a leaf node
result += -0.001386;
}
else { // if condition is not respected
// This is a leaf node
result += 0.102232;
}
}
}
else { // if condition is not respected
if (event[3] < -1.319610){
if (event[1] < 2.878712){
// This is a leaf node
result += -0.035015;
}
else { // if condition is not respected
// This is a leaf node
result += 0.056856;
}
}
else { // if condition is not respected
if (event[3] < -0.398717){
// This is a leaf node
result += -0.032825;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000241;
}
}
}
if (event[0] < 4.010478){
if (event[0] < 3.473993){
if (event[0] < 3.430337){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.133689;
}
}
else { // if condition is not respected
if (event[3] < 1.072391){
// This is a leaf node
result += -0.046593;
}
else { // if condition is not respected
// This is a leaf node
result += 0.053917;
}
}
}
else { // if condition is not respected
if (event[1] < 0.520747){
// This is a leaf node
result += -0.007130;
}
else { // if condition is not respected
// This is a leaf node
result += 0.076221;
}
}
if (event[3] < 1.568291){
if (event[3] < 1.427665){
if (event[3] < 1.427438){
// This is a leaf node
result += -0.000021;
}
else { // if condition is not respected
// This is a leaf node
result += -0.102412;
}
}
else { // if condition is not respected
if (event[0] < -1.508959){
// This is a leaf node
result += 0.026528;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002756;
}
}
}
else { // if condition is not respected
if (event[3] < 1.648609){
if (event[1] < 0.677682){
// This is a leaf node
result += -0.004220;
}
else { // if condition is not respected
// This is a leaf node
result += -0.021179;
}
}
else { // if condition is not respected
if (event[2] < -0.791153){
// This is a leaf node
result += 0.006197;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001243;
}
}
}
if (event[2] < -0.303284){
if (event[2] < -0.304455){
if (event[2] < -0.433119){
// This is a leaf node
result += 0.000073;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002735;
}
}
else { // if condition is not respected
if (event[4] < 1.091285){
// This is a leaf node
result += -0.019887;
}
else { // if condition is not respected
// This is a leaf node
result += -0.116558;
}
}
}
else { // if condition is not respected
if (event[2] < -0.302385){
if (event[2] < -0.302736){
// This is a leaf node
result += -0.000013;
}
else { // if condition is not respected
// This is a leaf node
result += 0.097804;
}
}
else { // if condition is not respected
if (event[2] < -0.301903){
// This is a leaf node
result += -0.043902;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000190;
}
}
}
if (event[3] < 1.848377){
if (event[3] < 1.796716){
if (event[3] < 1.795864){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += -0.076816;
}
}
else { // if condition is not respected
if (event[2] < 1.427331){
// This is a leaf node
result += 0.017512;
}
else { // if condition is not respected
// This is a leaf node
result += -0.036093;
}
}
}
else { // if condition is not respected
if (event[1] < 1.720694){
if (event[1] < -0.915643){
// This is a leaf node
result += 0.005675;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003908;
}
}
else { // if condition is not respected
if (event[4] < 0.660849){
// This is a leaf node
result += 0.001846;
}
else { // if condition is not respected
// This is a leaf node
result += 0.047854;
}
}
}
if (event[4] < -2.212085){
if (event[4] < -2.373916){
if (event[4] < -2.400029){
// This is a leaf node
result += -0.000018;
}
else { // if condition is not respected
// This is a leaf node
result += 0.031391;
}
}
else { // if condition is not respected
if (event[3] < -0.627826){
// This is a leaf node
result += -0.036272;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000749;
}
}
}
else { // if condition is not respected
if (event[4] < -2.211330){
// This is a leaf node
result += 0.104066;
}
else { // if condition is not respected
if (event[4] < -2.153306){
// This is a leaf node
result += 0.011421;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000000;
}
}
}
if (event[3] < -1.135062){
if (event[3] < -1.144823){
if (event[3] < -1.148011){
// This is a leaf node
result += 0.000508;
}
else { // if condition is not respected
// This is a leaf node
result += -0.022966;
}
}
else { // if condition is not respected
if (event[1] < -0.130347){
// This is a leaf node
result += 0.005496;
}
else { // if condition is not respected
// This is a leaf node
result += 0.029006;
}
}
}
else { // if condition is not respected
if (event[3] < -1.130239){
if (event[1] < 1.208377){
// This is a leaf node
result += -0.008862;
}
else { // if condition is not respected
// This is a leaf node
result += -0.087602;
}
}
else { // if condition is not respected
if (event[3] < -1.126782){
// This is a leaf node
result += 0.020307;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000092;
}
}
}
if (event[4] < 3.594640){
if (event[4] < 3.454712){
if (event[4] < 3.403118){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.046342;
}
}
else { // if condition is not respected
if (event[2] < 0.940680){
// This is a leaf node
result += 0.057748;
}
else { // if condition is not respected
// This is a leaf node
result += -0.044095;
}
}
}
else { // if condition is not respected
if (event[4] < 3.647938){
if (event[1] < 0.597298){
// This is a leaf node
result += -0.113109;
}
else { // if condition is not respected
// This is a leaf node
result += -0.029519;
}
}
else { // if condition is not respected
if (event[2] < 1.129163){
// This is a leaf node
result += 0.016888;
}
else { // if condition is not respected
// This is a leaf node
result += -0.074091;
}
}
}
if (event[2] < -2.304283){
if (event[1] < -2.659611){
if (event[1] < -3.044231){
// This is a leaf node
result += 0.001615;
}
else { // if condition is not respected
// This is a leaf node
result += 0.099510;
}
}
else { // if condition is not respected
if (event[1] < -2.473958){
// This is a leaf node
result += -0.122623;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002427;
}
}
}
else { // if condition is not respected
if (event[2] < -2.302579){
if (event[2] < -2.303342){
// This is a leaf node
result += 0.030765;
}
else { // if condition is not respected
// This is a leaf node
result += 0.123693;
}
}
else { // if condition is not respected
if (event[2] < -2.301145){
// This is a leaf node
result += -0.071058;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000024;
}
}
}
if (event[0] < -0.625203){
if (event[0] < -0.625615){
if (event[0] < -0.664277){
// This is a leaf node
result += -0.000099;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005958;
}
}
else { // if condition is not respected
if (event[2] < 0.854820){
// This is a leaf node
result += -0.083321;
}
else { // if condition is not respected
// This is a leaf node
result += 0.010527;
}
}
}
else { // if condition is not respected
if (event[0] < -0.625029){
if (event[3] < -0.406871){
// This is a leaf node
result += -0.025505;
}
else { // if condition is not respected
// This is a leaf node
result += 0.114692;
}
}
else { // if condition is not respected
if (event[0] < -0.332660){
// This is a leaf node
result += 0.001407;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000066;
}
}
}
if (event[0] < -0.153130){
if (event[0] < -0.169503){
if (event[2] < 3.084193){
// This is a leaf node
result += -0.000134;
}
else { // if condition is not respected
// This is a leaf node
result += 0.029442;
}
}
else { // if condition is not respected
if (event[4] < -1.127754){
// This is a leaf node
result += 0.014730;
}
else { // if condition is not respected
// This is a leaf node
result += -0.015927;
}
}
}
else { // if condition is not respected
if (event[0] < -0.139817){
if (event[4] < -1.741237){
// This is a leaf node
result += -0.041414;
}
else { // if condition is not respected
// This is a leaf node
result += 0.011690;
}
}
else { // if condition is not respected
if (event[0] < -0.139616){
// This is a leaf node
result += -0.065953;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000144;
}
}
}
if (event[0] < -4.139326){
// This is a leaf node
result += 0.052242;
}
else { // if condition is not respected
if (event[0] < -3.967169){
// This is a leaf node
result += -0.108047;
}
else { // if condition is not respected
if (event[0] < -3.841982){
// This is a leaf node
result += 0.066526;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000001;
}
}
}
if (event[2] < 3.173035){
if (event[2] < 3.106857){
if (event[2] < 3.042542){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.035926;
}
}
else { // if condition is not respected
if (event[3] < -0.198675){
// This is a leaf node
result += 0.122010;
}
else { // if condition is not respected
// This is a leaf node
result += 0.015117;
}
}
}
else { // if condition is not respected
if (event[2] < 3.184392){
if (event[4] < -0.306575){
// This is a leaf node
result += -0.049006;
}
else { // if condition is not respected
// This is a leaf node
result += -0.137338;
}
}
else { // if condition is not respected
if (event[0] < -0.067254){
// This is a leaf node
result += 0.023225;
}
else { // if condition is not respected
// This is a leaf node
result += -0.025768;
}
}
}
if (event[3] < -0.202826){
if (event[3] < -0.204509){
if (event[3] < -0.204574){
// This is a leaf node
result += -0.000263;
}
else { // if condition is not respected
// This is a leaf node
result += 0.106777;
}
}
else { // if condition is not respected
if (event[1] < 0.126296){
// This is a leaf node
result += -0.043726;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006862;
}
}
}
else { // if condition is not respected
if (event[3] < 0.090088){
if (event[1] < 2.447654){
// This is a leaf node
result += 0.001758;
}
else { // if condition is not respected
// This is a leaf node
result += 0.025584;
}
}
else { // if condition is not respected
if (event[1] < -2.364760){
// This is a leaf node
result += 0.009764;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000319;
}
}
}
if (event[1] < -2.983040){
if (event[3] < -0.236571){
if (event[2] < 1.040672){
// This is a leaf node
result += 0.018331;
}
else { // if condition is not respected
// This is a leaf node
result += -0.041371;
}
}
else { // if condition is not respected
if (event[3] < 1.097380){
// This is a leaf node
result += -0.029655;
}
else { // if condition is not respected
// This is a leaf node
result += 0.022544;
}
}
}
else { // if condition is not respected
if (event[1] < -2.973975){
// This is a leaf node
result += 0.115330;
}
else { // if condition is not respected
if (event[1] < -2.971200){
// This is a leaf node
result += -0.085650;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000009;
}
}
}
if (event[3] < 1.045245){
if (event[3] < 1.044841){
if (event[3] < 0.606942){
// This is a leaf node
result += 0.000132;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001432;
}
}
else { // if condition is not respected
if (event[4] < -0.049362){
// This is a leaf node
result += -0.012095;
}
else { // if condition is not respected
// This is a leaf node
result += -0.127068;
}
}
}
else { // if condition is not respected
if (event[3] < 1.047289){
if (event[3] < 1.047014){
// This is a leaf node
result += 0.012886;
}
else { // if condition is not respected
// This is a leaf node
result += 0.108155;
}
}
else { // if condition is not respected
if (event[1] < -2.833028){
// This is a leaf node
result += 0.034037;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000425;
}
}
}
if (event[1] < 2.817672){
if (event[1] < 2.807080){
if (event[1] < 2.569115){
// This is a leaf node
result += -0.000010;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006631;
}
}
else { // if condition is not respected
if (event[0] < -0.942173){
// This is a leaf node
result += 0.140635;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007693;
}
}
}
else { // if condition is not respected
if (event[3] < 1.892345){
if (event[3] < 1.653552){
// This is a leaf node
result += -0.004759;
}
else { // if condition is not respected
// This is a leaf node
result += 0.087547;
}
}
else { // if condition is not respected
if (event[4] < -0.349983){
// This is a leaf node
result += -0.126207;
}
else { // if condition is not respected
// This is a leaf node
result += -0.009916;
}
}
}
if (event[1] < -2.983040){
if (event[0] < 1.173963){
if (event[0] < 1.036806){
// This is a leaf node
result += -0.005515;
}
else { // if condition is not respected
// This is a leaf node
result += 0.094155;
}
}
else { // if condition is not respected
if (event[0] < 1.875845){
// This is a leaf node
result += -0.062501;
}
else { // if condition is not respected
// This is a leaf node
result += 0.032941;
}
}
}
else { // if condition is not respected
if (event[1] < -2.973975){
// This is a leaf node
result += 0.109090;
}
else { // if condition is not respected
if (event[1] < -2.971200){
// This is a leaf node
result += -0.080398;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000009;
}
}
}
if (event[1] < 3.524422){
if (event[1] < 3.504731){
if (event[1] < 3.481369){
// This is a leaf node
result += -0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += 0.059894;
}
}
else { // if condition is not respected
if (event[2] < 0.211567){
// This is a leaf node
result += -0.103544;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005952;
}
}
}
else { // if condition is not respected
if (event[3] < 0.240275){
if (event[1] < 3.602510){
// This is a leaf node
result += 0.145980;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001998;
}
}
else { // if condition is not respected
if (event[1] < 3.610763){
// This is a leaf node
result += -0.112747;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006922;
}
}
}
if (event[4] < 0.542113){
if (event[4] < 0.496927){
if (event[4] < 0.496569){
// This is a leaf node
result += 0.000063;
}
else { // if condition is not respected
// This is a leaf node
result += -0.077688;
}
}
else { // if condition is not respected
if (event[4] < 0.504147){
// This is a leaf node
result += 0.019888;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001855;
}
}
}
else { // if condition is not respected
if (event[0] < -1.534526){
if (event[4] < 0.839070){
// This is a leaf node
result += -0.012894;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000027;
}
}
else { // if condition is not respected
if (event[0] < -1.519372){
// This is a leaf node
result += 0.035302;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000197;
}
}
}
if (event[0] < -2.627497){
if (event[0] < -2.684691){
if (event[1] < -1.778956){
// This is a leaf node
result += 0.060898;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004079;
}
}
else { // if condition is not respected
if (event[0] < -2.665749){
// This is a leaf node
result += 0.087228;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009925;
}
}
}
else { // if condition is not respected
if (event[0] < -2.508932){
if (event[0] < -2.517264){
// This is a leaf node
result += -0.006767;
}
else { // if condition is not respected
// This is a leaf node
result += -0.066890;
}
}
else { // if condition is not respected
if (event[0] < -2.500730){
// This is a leaf node
result += 0.047050;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000001;
}
}
}
if (event[1] < 2.736795){
if (event[1] < 2.569115){
if (event[1] < 2.518968){
// This is a leaf node
result += 0.000010;
}
else { // if condition is not respected
// This is a leaf node
result += -0.024041;
}
}
else { // if condition is not respected
if (event[3] < 1.906981){
// This is a leaf node
result += 0.008294;
}
else { // if condition is not respected
// This is a leaf node
result += 0.086344;
}
}
}
else { // if condition is not respected
if (event[1] < 2.752325){
if (event[1] < 2.746836){
// This is a leaf node
result += -0.020642;
}
else { // if condition is not respected
// This is a leaf node
result += -0.115857;
}
}
else { // if condition is not respected
if (event[3] < 1.892345){
// This is a leaf node
result += 0.000529;
}
else { // if condition is not respected
// This is a leaf node
result += -0.059776;
}
}
}
if (event[1] < 0.526281){
if (event[1] < 0.526106){
if (event[1] < 0.470429){
// This is a leaf node
result += -0.000056;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003476;
}
}
else { // if condition is not respected
if (event[2] < -0.661807){
// This is a leaf node
result += 0.032959;
}
else { // if condition is not respected
// This is a leaf node
result += -0.094883;
}
}
}
else { // if condition is not respected
if (event[1] < 0.526348){
if (event[3] < -0.528093){
// This is a leaf node
result += 0.020503;
}
else { // if condition is not respected
// This is a leaf node
result += 0.135266;
}
}
else { // if condition is not respected
if (event[0] < -3.203832){
// This is a leaf node
result += 0.038368;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000327;
}
}
}
if (event[4] < 1.231738){
if (event[4] < 1.230623){
if (event[4] < 1.078840){
// This is a leaf node
result += 0.000053;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003528;
}
}
else { // if condition is not respected
if (event[0] < -0.756257){
// This is a leaf node
result += 0.047760;
}
else { // if condition is not respected
// This is a leaf node
result += -0.083907;
}
}
}
else { // if condition is not respected
if (event[4] < 1.279068){
if (event[1] < 1.247748){
// This is a leaf node
result += 0.008079;
}
else { // if condition is not respected
// This is a leaf node
result += 0.034702;
}
}
else { // if condition is not respected
if (event[4] < 1.311158){
// This is a leaf node
result += -0.013577;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000573;
}
}
}
if (event[4] < 1.671161){
if (event[4] < 1.668152){
if (event[4] < 1.567766){
// This is a leaf node
result += -0.000015;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005238;
}
}
else { // if condition is not respected
if (event[0] < -0.279362){
// This is a leaf node
result += -0.008232;
}
else { // if condition is not respected
// This is a leaf node
result += 0.073319;
}
}
}
else { // if condition is not respected
if (event[4] < 1.682022){
if (event[3] < -0.777274){
// This is a leaf node
result += 0.008468;
}
else { // if condition is not respected
// This is a leaf node
result += -0.031764;
}
}
else { // if condition is not respected
if (event[3] < -1.821813){
// This is a leaf node
result += 0.015902;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001155;
}
}
}
if (event[3] < -0.202826){
if (event[3] < -0.203357){
if (event[3] < -0.203674){
// This is a leaf node
result += -0.000279;
}
else { // if condition is not respected
// This is a leaf node
result += 0.056619;
}
}
else { // if condition is not respected
if (event[0] < -1.668381){
// This is a leaf node
result += 0.083700;
}
else { // if condition is not respected
// This is a leaf node
result += -0.049895;
}
}
}
else { // if condition is not respected
if (event[3] < -0.202741){
if (event[2] < 0.831153){
// This is a leaf node
result += 0.096492;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004946;
}
}
else { // if condition is not respected
if (event[3] < 0.090088){
// This is a leaf node
result += 0.001692;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000183;
}
}
}
if (event[1] < 1.434816){
if (event[1] < 1.434327){
if (event[1] < 1.434138){
// This is a leaf node
result += 0.000067;
}
else { // if condition is not respected
// This is a leaf node
result += -0.101250;
}
}
else { // if condition is not respected
if (event[3] < -0.846518){
// This is a leaf node
result += -0.018816;
}
else { // if condition is not respected
// This is a leaf node
result += 0.077019;
}
}
}
else { // if condition is not respected
if (event[1] < 1.443307){
if (event[0] < 1.422001){
// This is a leaf node
result += -0.021118;
}
else { // if condition is not respected
// This is a leaf node
result += 0.047951;
}
}
else { // if condition is not respected
if (event[1] < 1.444061){
// This is a leaf node
result += 0.052581;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000658;
}
}
}
if (event[4] < -3.586279){
if (event[4] < -3.704080){
if (event[0] < -1.340873){
// This is a leaf node
result += 0.093267;
}
else { // if condition is not respected
// This is a leaf node
result += -0.029986;
}
}
else { // if condition is not respected
if (event[2] < 0.700506){
// This is a leaf node
result += 0.089228;
}
else { // if condition is not respected
// This is a leaf node
result += -0.061123;
}
}
}
else { // if condition is not respected
if (event[4] < -3.519582){
if (event[0] < -0.005713){
// This is a leaf node
result += -0.119296;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004899;
}
}
else { // if condition is not respected
if (event[4] < -3.274415){
// This is a leaf node
result += 0.013317;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000005;
}
}
}
if (event[4] < -2.843313){
if (event[4] < -2.937447){
if (event[1] < 1.732139){
// This is a leaf node
result += -0.001407;
}
else { // if condition is not respected
// This is a leaf node
result += 0.072851;
}
}
else { // if condition is not respected
if (event[1] < -0.088142){
// This is a leaf node
result += -0.056265;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001733;
}
}
}
else { // if condition is not respected
if (event[4] < -2.817898){
if (event[2] < -1.422541){
// This is a leaf node
result += -0.079163;
}
else { // if condition is not respected
// This is a leaf node
result += 0.046829;
}
}
else { // if condition is not respected
if (event[4] < -2.794321){
// This is a leaf node
result += -0.031078;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000012;
}
}
}
if (event[2] < 4.023448){
if (event[2] < 3.986183){
if (event[2] < 3.831835){
// This is a leaf node
result += -0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += 0.042233;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.079521;
}
}
else { // if condition is not respected
if (event[4] < -0.005265){
// This is a leaf node
result += -0.009605;
}
else { // if condition is not respected
// This is a leaf node
result += 0.073944;
}
}
if (event[3] < -3.894619){
if (event[2] < -0.431760){
// This is a leaf node
result += 0.043071;
}
else { // if condition is not respected
if (event[0] < -0.002408){
// This is a leaf node
result += 0.000297;
}
else { // if condition is not respected
// This is a leaf node
result += -0.087445;
}
}
}
else { // if condition is not respected
if (event[3] < -3.672421){
if (event[2] < -0.848409){
// This is a leaf node
result += -0.035233;
}
else { // if condition is not respected
// This is a leaf node
result += 0.073360;
}
}
else { // if condition is not respected
if (event[3] < -2.946279){
// This is a leaf node
result += -0.006899;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000009;
}
}
}
if (event[2] < -3.749223){
if (event[2] < -3.845556){
if (event[0] < -0.624683){
// This is a leaf node
result += -0.067380;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012858;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.088576;
}
}
else { // if condition is not respected
if (event[2] < -3.453882){
if (event[1] < -0.484509){
// This is a leaf node
result += 0.029604;
}
else { // if condition is not respected
// This is a leaf node
result += -0.054953;
}
}
else { // if condition is not respected
if (event[2] < -3.319989){
// This is a leaf node
result += 0.035791;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000004;
}
}
}
if (event[3] < -1.054299){
if (event[3] < -1.055890){
if (event[0] < 1.282994){
// This is a leaf node
result += 0.001075;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005064;
}
}
else { // if condition is not respected
if (event[2] < -1.335228){
// This is a leaf node
result += -0.039472;
}
else { // if condition is not respected
// This is a leaf node
result += 0.049072;
}
}
}
else { // if condition is not respected
if (event[3] < -1.052490){
if (event[3] < -1.053031){
// This is a leaf node
result += -0.002002;
}
else { // if condition is not respected
// This is a leaf node
result += -0.093980;
}
}
else { // if condition is not respected
if (event[3] < -1.052137){
// This is a leaf node
result += 0.056157;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000091;
}
}
}
if (event[0] < 3.125710){
if (event[0] < 2.997290){
if (event[0] < 2.987079){
// This is a leaf node
result += 0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += 0.073759;
}
}
else { // if condition is not respected
if (event[1] < 1.003106){
// This is a leaf node
result += -0.007604;
}
else { // if condition is not respected
// This is a leaf node
result += -0.078001;
}
}
}
else { // if condition is not respected
if (event[1] < 1.182344){
if (event[1] < -0.386664){
// This is a leaf node
result += 0.037438;
}
else { // if condition is not respected
// This is a leaf node
result += -0.022339;
}
}
else { // if condition is not respected
if (event[1] < 1.889359){
// This is a leaf node
result += 0.099557;
}
else { // if condition is not respected
// This is a leaf node
result += -0.041301;
}
}
}
if (event[0] < 3.332963){
if (event[0] < 3.308728){
if (event[0] < 3.300047){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.080267;
}
}
else { // if condition is not respected
if (event[3] < 0.588757){
// This is a leaf node
result += 0.122545;
}
else { // if condition is not respected
// This is a leaf node
result += -0.017802;
}
}
}
else { // if condition is not respected
if (event[2] < 1.648108){
if (event[0] < 3.374739){
// This is a leaf node
result += -0.090461;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004730;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.115056;
}
}
if (event[3] < 1.045245){
if (event[3] < 1.044841){
if (event[3] < 0.980328){
// This is a leaf node
result += -0.000019;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003943;
}
}
else { // if condition is not respected
if (event[4] < -0.049362){
// This is a leaf node
result += -0.010990;
}
else { // if condition is not respected
// This is a leaf node
result += -0.117679;
}
}
}
else { // if condition is not respected
if (event[3] < 1.047289){
if (event[3] < 1.047014){
// This is a leaf node
result += 0.011581;
}
else { // if condition is not respected
// This is a leaf node
result += 0.098805;
}
}
else { // if condition is not respected
if (event[1] < -2.833028){
// This is a leaf node
result += 0.031336;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000413;
}
}
}
if (event[1] < -2.983040){
if (event[3] < -0.236571){
if (event[4] < 0.208663){
// This is a leaf node
result += -0.010270;
}
else { // if condition is not respected
// This is a leaf node
result += 0.031323;
}
}
else { // if condition is not respected
if (event[0] < 1.106422){
// This is a leaf node
result += -0.008850;
}
else { // if condition is not respected
// This is a leaf node
result += -0.059252;
}
}
}
else { // if condition is not respected
if (event[1] < -2.973975){
// This is a leaf node
result += 0.103550;
}
else { // if condition is not respected
if (event[1] < -2.971200){
// This is a leaf node
result += -0.075657;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000009;
}
}
}
if (event[0] < -2.904250){
if (event[4] < -2.028374){
if (event[1] < -0.792695){
// This is a leaf node
result += 0.012071;
}
else { // if condition is not respected
// This is a leaf node
result += 0.114720;
}
}
else { // if condition is not respected
if (event[1] < -1.778956){
// This is a leaf node
result += 0.056790;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011134;
}
}
}
else { // if condition is not respected
if (event[0] < -2.877849){
if (event[3] < -0.545673){
// This is a leaf node
result += -0.011436;
}
else { // if condition is not respected
// This is a leaf node
result += 0.095868;
}
}
else { // if condition is not respected
if (event[0] < -2.869082){
// This is a leaf node
result += -0.058227;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000003;
}
}
}
if (event[2] < -0.365662){
if (event[2] < -0.412785){
if (event[1] < 3.114747){
// This is a leaf node
result += 0.000034;
}
else { // if condition is not respected
// This is a leaf node
result += -0.030476;
}
}
else { // if condition is not respected
if (event[0] < 0.807342){
// This is a leaf node
result += -0.003777;
}
else { // if condition is not respected
// This is a leaf node
result += -0.017955;
}
}
}
else { // if condition is not respected
if (event[2] < -0.365463){
if (event[3] < -0.518570){
// This is a leaf node
result += 0.157061;
}
else { // if condition is not respected
// This is a leaf node
result += 0.018587;
}
}
else { // if condition is not respected
if (event[3] < -1.999859){
// This is a leaf node
result += 0.004033;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000077;
}
}
}
if (event[1] < 3.524422){
if (event[1] < 3.500641){
if (event[1] < 3.481369){
// This is a leaf node
result += -0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += 0.068828;
}
}
else { // if condition is not respected
if (event[3] < -0.039495){
// This is a leaf node
result += -0.100580;
}
else { // if condition is not respected
// This is a leaf node
result += 0.022284;
}
}
}
else { // if condition is not respected
if (event[3] < 0.240275){
if (event[1] < 3.602510){
// This is a leaf node
result += 0.137346;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002696;
}
}
else { // if condition is not respected
if (event[1] < 3.610763){
// This is a leaf node
result += -0.106186;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005047;
}
}
}
if (event[3] < 2.957072){
if (event[3] < 2.948339){
if (event[3] < 2.944436){
// This is a leaf node
result += -0.000007;
}
else { // if condition is not respected
// This is a leaf node
result += 0.133530;
}
}
else { // if condition is not respected
if (event[1] < -0.877170){
// This is a leaf node
result += 0.035459;
}
else { // if condition is not respected
// This is a leaf node
result += -0.109501;
}
}
}
else { // if condition is not respected
if (event[4] < -1.136159){
if (event[2] < 1.463101){
// This is a leaf node
result += -0.058925;
}
else { // if condition is not respected
// This is a leaf node
result += 0.099194;
}
}
else { // if condition is not respected
if (event[2] < 0.921987){
// This is a leaf node
result += 0.023113;
}
else { // if condition is not respected
// This is a leaf node
result += -0.034271;
}
}
}
if (event[3] < 2.389359){
if (event[3] < 2.152436){
if (event[3] < 2.139556){
// This is a leaf node
result += -0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.043320;
}
}
else { // if condition is not respected
if (event[1] < 0.089647){
// This is a leaf node
result += 0.016293;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005257;
}
}
}
else { // if condition is not respected
if (event[3] < 2.409128){
if (event[3] < 2.406490){
// This is a leaf node
result += -0.024076;
}
else { // if condition is not respected
// This is a leaf node
result += -0.106527;
}
}
else { // if condition is not respected
if (event[2] < -0.752653){
// This is a leaf node
result += 0.012552;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004055;
}
}
}
if (event[3] < 1.568337){
if (event[3] < 1.548263){
if (event[3] < 1.545021){
// This is a leaf node
result += 0.000044;
}
else { // if condition is not respected
// This is a leaf node
result += -0.027119;
}
}
else { // if condition is not respected
if (event[1] < 1.193377){
// This is a leaf node
result += 0.017323;
}
else { // if condition is not respected
// This is a leaf node
result += -0.035044;
}
}
}
else { // if condition is not respected
if (event[3] < 1.570177){
if (event[2] < -1.110332){
// This is a leaf node
result += 0.051741;
}
else { // if condition is not respected
// This is a leaf node
result += -0.071081;
}
}
else { // if condition is not respected
if (event[2] < 2.228405){
// This is a leaf node
result += -0.001124;
}
else { // if condition is not respected
// This is a leaf node
result += 0.026887;
}
}
}
if (event[2] < 2.478055){
if (event[2] < 2.476540){
if (event[2] < 2.247168){
// This is a leaf node
result += -0.000019;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006019;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.116651;
}
}
else { // if condition is not respected
if (event[0] < 1.229889){
if (event[0] < 0.868947){
// This is a leaf node
result += -0.002440;
}
else { // if condition is not respected
// This is a leaf node
result += -0.033258;
}
}
else { // if condition is not respected
if (event[3] < 1.927400){
// This is a leaf node
result += 0.024427;
}
else { // if condition is not respected
// This is a leaf node
result += -0.099999;
}
}
}
if (event[2] < 2.092766){
if (event[2] < 2.090705){
if (event[2] < 1.888311){
// This is a leaf node
result += -0.000023;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004341;
}
}
else { // if condition is not respected
if (event[1] < 0.184759){
// This is a leaf node
result += 0.099196;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002297;
}
}
}
else { // if condition is not respected
if (event[0] < -2.411729){
if (event[3] < -1.241215){
// This is a leaf node
result += -0.033910;
}
else { // if condition is not respected
// This is a leaf node
result += 0.070921;
}
}
else { // if condition is not respected
if (event[2] < 2.247168){
// This is a leaf node
result += -0.008859;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000943;
}
}
}
if (event[2] < 1.720806){
if (event[2] < 1.706046){
if (event[2] < 1.704433){
// This is a leaf node
result += 0.000032;
}
else { // if condition is not respected
// This is a leaf node
result += -0.044240;
}
}
else { // if condition is not respected
if (event[0] < -1.719481){
// This is a leaf node
result += 0.131220;
}
else { // if condition is not respected
// This is a leaf node
result += 0.015810;
}
}
}
else { // if condition is not respected
if (event[2] < 1.721015){
// This is a leaf node
result += -0.142068;
}
else { // if condition is not respected
if (event[4] < -2.546126){
// This is a leaf node
result += -0.044939;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000897;
}
}
}
if (event[2] < 1.368580){
if (event[2] < 1.269283){
if (event[2] < 1.268978){
// This is a leaf node
result += -0.000036;
}
else { // if condition is not respected
// This is a leaf node
result += -0.062126;
}
}
else { // if condition is not respected
if (event[3] < -2.821095){
// This is a leaf node
result += -0.097863;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006415;
}
}
}
else { // if condition is not respected
if (event[4] < -0.718483){
if (event[4] < -0.748446){
// This is a leaf node
result += 0.002088;
}
else { // if condition is not respected
// This is a leaf node
result += 0.025608;
}
}
else { // if condition is not respected
if (event[0] < -2.352196){
// This is a leaf node
result += 0.025944;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002215;
}
}
}
if (event[0] < -0.153168){
if (event[0] < -0.170399){
if (event[3] < -2.446199){
// This is a leaf node
result += -0.009610;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000033;
}
}
else { // if condition is not respected
if (event[1] < 1.286919){
// This is a leaf node
result += -0.007699;
}
else { // if condition is not respected
// This is a leaf node
result += -0.037323;
}
}
}
else { // if condition is not respected
if (event[0] < -0.121247){
if (event[3] < -2.410731){
// This is a leaf node
result += -0.076089;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006326;
}
}
else { // if condition is not respected
if (event[0] < -0.120640){
// This is a leaf node
result += -0.065625;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000119;
}
}
}
if (event[3] < -2.639947){
if (event[1] < 0.655445){
if (event[3] < -2.642425){
// This is a leaf node
result += -0.003150;
}
else { // if condition is not respected
// This is a leaf node
result += 0.109529;
}
}
else { // if condition is not respected
if (event[2] < 1.265351){
// This is a leaf node
result += 0.027646;
}
else { // if condition is not respected
// This is a leaf node
result += -0.043279;
}
}
}
else { // if condition is not respected
if (event[3] < -2.539846){
if (event[0] < 0.943438){
// This is a leaf node
result += -0.029342;
}
else { // if condition is not respected
// This is a leaf node
result += 0.033224;
}
}
else { // if condition is not respected
if (event[3] < -2.527129){
// This is a leaf node
result += 0.062025;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000000;
}
}
}
if (event[4] < 3.775747){
if (event[4] < 3.687956){
if (event[4] < 3.594640){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.071699;
}
}
else { // if condition is not respected
if (event[0] < -0.181956){
// This is a leaf node
result += -0.016031;
}
else { // if condition is not respected
// This is a leaf node
result += 0.106235;
}
}
}
else { // if condition is not respected
if (event[0] < -1.051262){
if (event[2] < 0.474477){
// This is a leaf node
result += -0.017363;
}
else { // if condition is not respected
// This is a leaf node
result += 0.075639;
}
}
else { // if condition is not respected
if (event[1] < 1.024293){
// This is a leaf node
result += -0.067726;
}
else { // if condition is not respected
// This is a leaf node
result += 0.056011;
}
}
}
if (event[3] < 3.465692){
if (event[3] < 3.445596){
if (event[3] < 3.377400){
// This is a leaf node
result += 0.000004;
}
else { // if condition is not respected
// This is a leaf node
result += -0.041667;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.097187;
}
}
else { // if condition is not respected
if (event[0] < -0.925448){
if (event[4] < -1.106827){
// This is a leaf node
result += 0.006040;
}
else { // if condition is not respected
// This is a leaf node
result += -0.105011;
}
}
else { // if condition is not respected
if (event[0] < -0.741249){
// This is a leaf node
result += 0.085809;
}
else { // if condition is not respected
// This is a leaf node
result += -0.012866;
}
}
}
if (event[1] < 0.666113){
if (event[1] < 0.646511){
if (event[1] < 0.645995){
// This is a leaf node
result += -0.000066;
}
else { // if condition is not respected
// This is a leaf node
result += 0.044373;
}
}
else { // if condition is not respected
if (event[0] < -2.046923){
// This is a leaf node
result += 0.056277;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011261;
}
}
}
else { // if condition is not respected
if (event[1] < 0.672041){
if (event[3] < -0.806995){
// This is a leaf node
result += 0.047324;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009327;
}
}
else { // if condition is not respected
if (event[2] < -2.919535){
// This is a leaf node
result += -0.031459;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000321;
}
}
}
if (event[3] < -0.202284){
if (event[3] < -0.202442){
if (event[4] < -1.054039){
// This is a leaf node
result += -0.002113;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000058;
}
}
else { // if condition is not respected
if (event[2] < -0.649355){
// This is a leaf node
result += 0.025080;
}
else { // if condition is not respected
// This is a leaf node
result += -0.116974;
}
}
}
else { // if condition is not respected
if (event[3] < -0.184463){
if (event[3] < -0.184559){
// This is a leaf node
result += 0.006455;
}
else { // if condition is not respected
// This is a leaf node
result += 0.125117;
}
}
else { // if condition is not respected
if (event[3] < -0.177506){
// This is a leaf node
result += -0.009144;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000150;
}
}
}
if (event[4] < -1.884302){
if (event[1] < -1.587891){
if (event[4] < -2.336343){
// This is a leaf node
result += -0.002482;
}
else { // if condition is not respected
// This is a leaf node
result += 0.030757;
}
}
else { // if condition is not respected
if (event[0] < 1.842145){
// This is a leaf node
result += -0.000497;
}
else { // if condition is not respected
// This is a leaf node
result += 0.024284;
}
}
}
else { // if condition is not respected
if (event[4] < -1.873338){
if (event[1] < 0.987996){
// This is a leaf node
result += -0.029476;
}
else { // if condition is not respected
// This is a leaf node
result += 0.040920;
}
}
else { // if condition is not respected
if (event[4] < -1.872716){
// This is a leaf node
result += 0.073312;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000031;
}
}
}
if (event[0] < 2.237703){
if (event[0] < 2.216883){
if (event[0] < 2.215728){
// This is a leaf node
result += 0.000017;
}
else { // if condition is not respected
// This is a leaf node
result += -0.077501;
}
}
else { // if condition is not respected
if (event[1] < 0.456722){
// This is a leaf node
result += 0.033330;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011811;
}
}
}
else { // if condition is not respected
if (event[4] < -0.634073){
if (event[3] < -1.464886){
// This is a leaf node
result += -0.059159;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006464;
}
}
else { // if condition is not respected
if (event[4] < -0.538672){
// This is a leaf node
result += 0.034302;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000751;
}
}
}
if (event[0] < 3.125710){
if (event[0] < 3.110970){
if (event[0] < 3.100054){
// This is a leaf node
result += -0.000007;
}
else { // if condition is not respected
// This is a leaf node
result += 0.045734;
}
}
else { // if condition is not respected
if (event[2] < 0.430419){
// This is a leaf node
result += -0.105304;
}
else { // if condition is not respected
// This is a leaf node
result += 0.024592;
}
}
}
else { // if condition is not respected
if (event[0] < 3.152915){
if (event[0] < 3.147211){
// This is a leaf node
result += 0.017325;
}
else { // if condition is not respected
// This is a leaf node
result += 0.136622;
}
}
else { // if condition is not respected
if (event[1] < -0.386664){
// This is a leaf node
result += 0.035962;
}
else { // if condition is not respected
// This is a leaf node
result += -0.013094;
}
}
}
if (event[0] < 4.010478){
if (event[0] < 3.689817){
if (event[0] < 3.651816){
// This is a leaf node
result += -0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += 0.090152;
}
}
else { // if condition is not respected
if (event[4] < 0.450543){
// This is a leaf node
result += -0.073909;
}
else { // if condition is not respected
// This is a leaf node
result += 0.021070;
}
}
}
else { // if condition is not respected
if (event[3] < 0.132275){
if (event[3] < -0.557640){
// This is a leaf node
result += 0.042687;
}
else { // if condition is not respected
// This is a leaf node
result += -0.052247;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.077441;
}
}
if (event[1] < -3.332503){
if (event[2] < -0.938920){
if (event[1] < -3.409357){
// This is a leaf node
result += 0.078031;
}
else { // if condition is not respected
// This is a leaf node
result += -0.054330;
}
}
else { // if condition is not respected
if (event[3] < 0.555853){
// This is a leaf node
result += -0.041858;
}
else { // if condition is not respected
// This is a leaf node
result += 0.022929;
}
}
}
else { // if condition is not respected
if (event[1] < -3.324472){
// This is a leaf node
result += 0.110836;
}
else { // if condition is not respected
if (event[1] < -3.315744){
// This is a leaf node
result += -0.079430;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000004;
}
}
}
if (event[4] < -1.646589){
if (event[2] < 0.074283){
if (event[2] < -0.097715){
// This is a leaf node
result += 0.000516;
}
else { // if condition is not respected
// This is a leaf node
result += -0.014837;
}
}
else { // if condition is not respected
if (event[4] < -1.730687){
// This is a leaf node
result += 0.000989;
}
else { // if condition is not respected
// This is a leaf node
result += 0.017603;
}
}
}
else { // if condition is not respected
if (event[4] < -1.645817){
if (event[3] < 0.511981){
// This is a leaf node
result += -0.120932;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003057;
}
}
else { // if condition is not respected
if (event[4] < -1.605698){
// This is a leaf node
result += -0.007715;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000012;
}
}
}
if (event[0] < -3.528087){
if (event[3] < 0.561117){
if (event[2] < -1.378975){
// This is a leaf node
result += -0.083417;
}
else { // if condition is not respected
// This is a leaf node
result += 0.053462;
}
}
else { // if condition is not respected
if (event[3] < 1.388737){
// This is a leaf node
result += -0.093997;
}
else { // if condition is not respected
// This is a leaf node
result += 0.050793;
}
}
}
else { // if condition is not respected
if (event[0] < -3.476108){
if (event[2] < 0.498262){
// This is a leaf node
result += -0.110950;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001005;
}
}
else { // if condition is not respected
if (event[0] < -3.342121){
// This is a leaf node
result += -0.020400;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000004;
}
}
}
if (event[3] < -4.097473){
// This is a leaf node
result += -0.053840;
}
else { // if condition is not respected
if (event[3] < -3.395477){
if (event[0] < -1.039718){
// This is a leaf node
result += -0.059002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.030242;
}
}
else { // if condition is not respected
if (event[3] < -3.370402){
// This is a leaf node
result += -0.062040;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000002;
}
}
}
if (event[4] < -1.884302){
if (event[4] < -1.954781){
if (event[4] < -1.955022){
// This is a leaf node
result += -0.000289;
}
else { // if condition is not respected
// This is a leaf node
result += -0.147371;
}
}
else { // if condition is not respected
if (event[4] < -1.933029){
// This is a leaf node
result += 0.028590;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003798;
}
}
}
else { // if condition is not respected
if (event[4] < -1.884016){
// This is a leaf node
result += -0.099860;
}
else { // if condition is not respected
if (event[4] < -1.788130){
// This is a leaf node
result += -0.005711;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000004;
}
}
}
if (event[0] < 0.281223){
if (event[0] < 0.279190){
if (event[0] < 0.279083){
// This is a leaf node
result += 0.000155;
}
else { // if condition is not respected
// This is a leaf node
result += -0.124445;
}
}
else { // if condition is not respected
if (event[2] < 2.060161){
// This is a leaf node
result += 0.027822;
}
else { // if condition is not respected
// This is a leaf node
result += -0.111457;
}
}
}
else { // if condition is not respected
if (event[0] < 0.293038){
if (event[3] < 1.544912){
// This is a leaf node
result += -0.010142;
}
else { // if condition is not respected
// This is a leaf node
result += -0.050351;
}
}
else { // if condition is not respected
if (event[0] < 0.395085){
// This is a leaf node
result += -0.002995;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000182;
}
}
}
if (event[0] < -0.026886){
if (event[0] < -0.031308){
if (event[0] < -0.031708){
// This is a leaf node
result += -0.000168;
}
else { // if condition is not respected
// This is a leaf node
result += 0.049482;
}
}
else { // if condition is not respected
if (event[2] < -1.271377){
// This is a leaf node
result += 0.038568;
}
else { // if condition is not respected
// This is a leaf node
result += -0.029441;
}
}
}
else { // if condition is not respected
if (event[0] < -0.026402){
if (event[2] < -0.499046){
// This is a leaf node
result += -0.011269;
}
else { // if condition is not respected
// This is a leaf node
result += 0.076800;
}
}
else { // if condition is not respected
if (event[0] < 0.137619){
// This is a leaf node
result += 0.002454;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000123;
}
}
}
if (event[1] < 4.088089){
if (event[1] < 4.035286){
if (event[1] < 3.906952){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.041364;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.085293;
}
}
else { // if condition is not respected
if (event[2] < 0.152884){
// This is a leaf node
result += -0.002141;
}
else { // if condition is not respected
// This is a leaf node
result += -0.060973;
}
}
if (event[1] < 3.383786){
if (event[1] < 3.371411){
if (event[1] < 3.154752){
// This is a leaf node
result += 0.000004;
}
else { // if condition is not respected
// This is a leaf node
result += -0.013895;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.114252;
}
}
else { // if condition is not respected
if (event[0] < -0.339577){
if (event[0] < -1.519240){
// This is a leaf node
result += -0.047913;
}
else { // if condition is not respected
// This is a leaf node
result += 0.068878;
}
}
else { // if condition is not respected
if (event[2] < 0.121239){
// This is a leaf node
result += -0.039314;
}
else { // if condition is not respected
// This is a leaf node
result += 0.016056;
}
}
}
if (event[3] < -2.278308){
if (event[0] < -1.516241){
if (event[3] < -2.298016){
// This is a leaf node
result += -0.028420;
}
else { // if condition is not respected
// This is a leaf node
result += 0.094892;
}
}
else { // if condition is not respected
if (event[4] < 2.269966){
// This is a leaf node
result += 0.004548;
}
else { // if condition is not respected
// This is a leaf node
result += -0.063590;
}
}
}
else { // if condition is not respected
if (event[3] < -2.128897){
if (event[1] < -0.882368){
// This is a leaf node
result += 0.017502;
}
else { // if condition is not respected
// This is a leaf node
result += -0.013165;
}
}
else { // if condition is not respected
if (event[3] < -2.128141){
// This is a leaf node
result += 0.103323;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000011;
}
}
}
if (event[0] < -1.228112){
if (event[0] < -1.230195){
if (event[3] < 2.354836){
// This is a leaf node
result += 0.000716;
}
else { // if condition is not respected
// This is a leaf node
result += -0.017602;
}
}
else { // if condition is not respected
if (event[3] < 1.739180){
// This is a leaf node
result += 0.046617;
}
else { // if condition is not respected
// This is a leaf node
result += -0.093534;
}
}
}
else { // if condition is not respected
if (event[0] < -1.227925){
if (event[1] < -0.727620){
// This is a leaf node
result += 0.001018;
}
else { // if condition is not respected
// This is a leaf node
result += -0.148178;
}
}
else { // if condition is not respected
if (event[0] < -1.186120){
// This is a leaf node
result += -0.006230;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000018;
}
}
}
if (event[3] < 2.957072){
if (event[3] < 2.948339){
if (event[3] < 2.944436){
// This is a leaf node
result += -0.000008;
}
else { // if condition is not respected
// This is a leaf node
result += 0.125292;
}
}
else { // if condition is not respected
if (event[4] < 0.443897){
// This is a leaf node
result += -0.013236;
}
else { // if condition is not respected
// This is a leaf node
result += -0.139027;
}
}
}
else { // if condition is not respected
if (event[4] < 1.367887){
if (event[4] < 1.276314){
// This is a leaf node
result += 0.003763;
}
else { // if condition is not respected
// This is a leaf node
result += -0.118294;
}
}
else { // if condition is not respected
if (event[2] < 0.166244){
// This is a leaf node
result += 0.098366;
}
else { // if condition is not respected
// This is a leaf node
result += 0.010752;
}
}
}
if (event[1] < 0.525228){
if (event[1] < 0.525023){
if (event[1] < 0.524966){
// This is a leaf node
result += -0.000145;
}
else { // if condition is not respected
// This is a leaf node
result += 0.084545;
}
}
else { // if condition is not respected
if (event[2] < -0.937363){
// This is a leaf node
result += 0.071747;
}
else { // if condition is not respected
// This is a leaf node
result += -0.112252;
}
}
}
else { // if condition is not respected
if (event[1] < 0.551360){
if (event[0] < 2.444216){
// This is a leaf node
result += 0.005550;
}
else { // if condition is not respected
// This is a leaf node
result += 0.091384;
}
}
else { // if condition is not respected
if (event[1] < 0.551674){
// This is a leaf node
result += -0.078647;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000198;
}
}
}
if (event[0] < 2.027467){
if (event[0] < 2.008068){
if (event[0] < 2.005329){
// This is a leaf node
result += 0.000019;
}
else { // if condition is not respected
// This is a leaf node
result += -0.074882;
}
}
else { // if condition is not respected
if (event[0] < 2.014476){
// This is a leaf node
result += 0.055438;
}
else { // if condition is not respected
// This is a leaf node
result += 0.008669;
}
}
}
else { // if condition is not respected
if (event[0] < 2.028569){
if (event[4] < -1.484969){
// This is a leaf node
result += 0.058057;
}
else { // if condition is not respected
// This is a leaf node
result += -0.129352;
}
}
else { // if condition is not respected
if (event[3] < -0.902463){
// This is a leaf node
result += 0.006754;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002912;
}
}
}
if (event[3] < 1.045245){
if (event[3] < 1.042279){
if (event[3] < 1.042100){
// This is a leaf node
result += -0.000084;
}
else { // if condition is not respected
// This is a leaf node
result += 0.125202;
}
}
else { // if condition is not respected
if (event[4] < -1.444025){
// This is a leaf node
result += 0.049821;
}
else { // if condition is not respected
// This is a leaf node
result += -0.029432;
}
}
}
else { // if condition is not respected
if (event[0] < -2.911052){
if (event[0] < -2.961510){
// This is a leaf node
result += -0.013823;
}
else { // if condition is not respected
// This is a leaf node
result += -0.098637;
}
}
else { // if condition is not respected
if (event[0] < -2.333061){
// This is a leaf node
result += 0.018638;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000455;
}
}
}
if (event[3] < -3.894619){
if (event[2] < -0.431760){
// This is a leaf node
result += 0.041060;
}
else { // if condition is not respected
if (event[0] < -0.002408){
// This is a leaf node
result += 0.000388;
}
else { // if condition is not respected
// This is a leaf node
result += -0.083573;
}
}
}
else { // if condition is not respected
if (event[3] < -3.672421){
if (event[2] < -0.848409){
// This is a leaf node
result += -0.034400;
}
else { // if condition is not respected
// This is a leaf node
result += 0.067784;
}
}
else { // if condition is not respected
if (event[3] < -2.946279){
// This is a leaf node
result += -0.006251;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000008;
}
}
}
if (event[2] < 2.355999){
if (event[2] < 2.305389){
if (event[2] < 2.304456){
// This is a leaf node
result += -0.000004;
}
else { // if condition is not respected
// This is a leaf node
result += -0.102248;
}
}
else { // if condition is not respected
if (event[3] < -1.706016){
// This is a leaf node
result += -0.051908;
}
else { // if condition is not respected
// This is a leaf node
result += 0.026394;
}
}
}
else { // if condition is not respected
if (event[2] < 2.362816){
if (event[3] < 1.647140){
// This is a leaf node
result += -0.077380;
}
else { // if condition is not respected
// This is a leaf node
result += 0.070814;
}
}
else { // if condition is not respected
if (event[4] < 1.935911){
// This is a leaf node
result += 0.000216;
}
else { // if condition is not respected
// This is a leaf node
result += -0.045837;
}
}
}
if (event[2] < 2.903436){
if (event[2] < 2.898122){
if (event[2] < 2.806383){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.015734;
}
}
else { // if condition is not respected
if (event[0] < 0.767917){
// This is a leaf node
result += -0.129363;
}
else { // if condition is not respected
// This is a leaf node
result += -0.035877;
}
}
}
else { // if condition is not respected
if (event[3] < -1.862135){
if (event[4] < -0.128799){
// This is a leaf node
result += 0.036459;
}
else { // if condition is not respected
// This is a leaf node
result += -0.109147;
}
}
else { // if condition is not respected
if (event[3] < -1.515114){
// This is a leaf node
result += 0.076640;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005777;
}
}
}
if (event[2] < 3.173035){
if (event[2] < 3.106857){
if (event[2] < 3.042542){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.032260;
}
}
else { // if condition is not respected
if (event[3] < -0.667544){
// This is a leaf node
result += 0.132093;
}
else { // if condition is not respected
// This is a leaf node
result += 0.024439;
}
}
}
else { // if condition is not respected
if (event[2] < 3.184392){
if (event[0] < 0.051685){
// This is a leaf node
result += -0.128147;
}
else { // if condition is not respected
// This is a leaf node
result += -0.045453;
}
}
else { // if condition is not respected
if (event[3] < -0.580578){
// This is a leaf node
result += -0.039201;
}
else { // if condition is not respected
// This is a leaf node
result += 0.010962;
}
}
}
if (event[3] < -1.888217){
if (event[4] < 1.266467){
if (event[4] < 0.652970){
// This is a leaf node
result += 0.002527;
}
else { // if condition is not respected
// This is a leaf node
result += -0.012308;
}
}
else { // if condition is not respected
if (event[4] < 1.326463){
// This is a leaf node
result += 0.054408;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009038;
}
}
}
else { // if condition is not respected
if (event[3] < -1.886691){
if (event[0] < -0.439712){
// This is a leaf node
result += 0.030978;
}
else { // if condition is not respected
// This is a leaf node
result += -0.082104;
}
}
else { // if condition is not respected
if (event[3] < -1.886278){
// This is a leaf node
result += 0.069004;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000037;
}
}
}
if (event[2] < 3.244201){
if (event[2] < 3.084193){
if (event[2] < 3.064624){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.078595;
}
}
else { // if condition is not respected
if (event[4] < 1.590579){
// This is a leaf node
result += 0.034537;
}
else { // if condition is not respected
// This is a leaf node
result += -0.104915;
}
}
}
else { // if condition is not respected
if (event[0] < -1.312025){
if (event[1] < -0.664932){
// This is a leaf node
result += -0.025950;
}
else { // if condition is not respected
// This is a leaf node
result += 0.086616;
}
}
else { // if condition is not respected
if (event[2] < 3.458330){
// This is a leaf node
result += -0.037993;
}
else { // if condition is not respected
// This is a leaf node
result += 0.010380;
}
}
}
if (event[0] < -0.585980){
if (event[0] < -0.587047){
if (event[1] < 0.044524){
// This is a leaf node
result += 0.000763;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001477;
}
}
else { // if condition is not respected
if (event[2] < -0.783120){
// This is a leaf node
result += 0.025761;
}
else { // if condition is not respected
// This is a leaf node
result += -0.051311;
}
}
}
else { // if condition is not respected
if (event[0] < -0.585862){
if (event[1] < 0.657637){
// This is a leaf node
result += 0.106465;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020158;
}
}
else { // if condition is not respected
if (event[0] < -0.427879){
// This is a leaf node
result += 0.002032;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000023;
}
}
}
if (event[1] < 0.247833){
if (event[1] < 0.208027){
if (event[1] < 0.207294){
// This is a leaf node
result += -0.000039;
}
else { // if condition is not respected
// This is a leaf node
result += 0.036359;
}
}
else { // if condition is not respected
if (event[0] < -0.779792){
// This is a leaf node
result += -0.018997;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003204;
}
}
}
else { // if condition is not respected
if (event[1] < 0.247907){
// This is a leaf node
result += 0.134124;
}
else { // if condition is not respected
if (event[2] < -2.919535){
// This is a leaf node
result += -0.021785;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000318;
}
}
}
if (event[2] < -3.749223){
if (event[2] < -3.845556){
if (event[0] < -0.624683){
// This is a leaf node
result += -0.063210;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012706;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.084876;
}
}
else { // if condition is not respected
if (event[2] < -3.720107){
// This is a leaf node
result += -0.064968;
}
else { // if condition is not respected
if (event[1] < -1.813412){
// This is a leaf node
result += -0.001175;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000042;
}
}
}
if (event[1] < 3.020258){
if (event[1] < 3.004412){
if (event[1] < 2.922445){
// This is a leaf node
result += 0.000006;
}
else { // if condition is not respected
// This is a leaf node
result += -0.017947;
}
}
else { // if condition is not respected
if (event[2] < 0.416007){
// This is a leaf node
result += -0.109186;
}
else { // if condition is not respected
// This is a leaf node
result += 0.014083;
}
}
}
else { // if condition is not respected
if (event[2] < -0.604317){
if (event[1] < 3.067139){
// This is a leaf node
result += 0.057781;
}
else { // if condition is not respected
// This is a leaf node
result += -0.034012;
}
}
else { // if condition is not respected
if (event[2] < -0.540627){
// This is a leaf node
result += 0.142933;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012648;
}
}
}
if (event[4] < -3.839994){
if (event[2] < 0.248174){
if (event[0] < 0.325009){
// This is a leaf node
result += 0.103145;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004265;
}
}
else { // if condition is not respected
if (event[1] < -0.541267){
// This is a leaf node
result += 0.073228;
}
else { // if condition is not respected
// This is a leaf node
result += -0.074695;
}
}
}
else { // if condition is not respected
if (event[4] < -3.821068){
// This is a leaf node
result += -0.102367;
}
else { // if condition is not respected
if (event[4] < -3.801609){
// This is a leaf node
result += 0.069192;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000001;
}
}
}
if (event[2] < -3.319989){
if (event[2] < -3.331810){
if (event[0] < 1.768486){
// This is a leaf node
result += -0.002957;
}
else { // if condition is not respected
// This is a leaf node
result += 0.120035;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.124629;
}
}
else { // if condition is not respected
if (event[2] < -3.020078){
if (event[4] < 1.293926){
// This is a leaf node
result += -0.020788;
}
else { // if condition is not respected
// This is a leaf node
result += 0.080891;
}
}
else { // if condition is not respected
if (event[2] < -2.995612){
// This is a leaf node
result += 0.051111;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000001;
}
}
}
if (event[2] < -3.144128){
if (event[4] < 1.281482){
if (event[4] < 0.510205){
// This is a leaf node
result += 0.016896;
}
else { // if condition is not respected
// This is a leaf node
result += -0.063078;
}
}
else { // if condition is not respected
if (event[2] < -3.542583){
// This is a leaf node
result += -0.021647;
}
else { // if condition is not respected
// This is a leaf node
result += 0.098395;
}
}
}
else { // if condition is not respected
if (event[2] < -3.136446){
// This is a leaf node
result += -0.117386;
}
else { // if condition is not respected
if (event[2] < -3.125410){
// This is a leaf node
result += 0.074970;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000006;
}
}
}
if (event[0] < -1.242683){
if (event[2] < 2.408053){
if (event[2] < 2.245631){
// This is a leaf node
result += 0.000671;
}
else { // if condition is not respected
// This is a leaf node
result += 0.030545;
}
}
else { // if condition is not respected
if (event[2] < 2.427593){
// This is a leaf node
result += -0.122573;
}
else { // if condition is not respected
// This is a leaf node
result += -0.013145;
}
}
}
else { // if condition is not respected
if (event[0] < -1.242572){
// This is a leaf node
result += -0.126987;
}
else { // if condition is not respected
if (event[0] < -1.238326){
// This is a leaf node
result += -0.017957;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000058;
}
}
}
if (event[0] < 0.004480){
if (event[0] < 0.004404){
if (event[0] < 0.004297){
// This is a leaf node
result += -0.000222;
}
else { // if condition is not respected
// This is a leaf node
result += 0.070031;
}
}
else { // if condition is not respected
if (event[1] < 0.236219){
// This is a leaf node
result += -0.125013;
}
else { // if condition is not respected
// This is a leaf node
result += -0.015395;
}
}
}
else { // if condition is not respected
if (event[0] < 0.016715){
if (event[1] < 2.638434){
// This is a leaf node
result += 0.010406;
}
else { // if condition is not respected
// This is a leaf node
result += 0.137950;
}
}
else { // if condition is not respected
if (event[0] < 0.017696){
// This is a leaf node
result += -0.037996;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000148;
}
}
}
if (event[0] < 0.624537){
if (event[0] < 0.620841){
if (event[0] < 0.618669){
// This is a leaf node
result += 0.000113;
}
else { // if condition is not respected
// This is a leaf node
result += -0.020113;
}
}
else { // if condition is not respected
if (event[4] < 0.665485){
// This is a leaf node
result += 0.040505;
}
else { // if condition is not respected
// This is a leaf node
result += -0.015521;
}
}
}
else { // if condition is not respected
if (event[0] < 0.625062){
if (event[0] < 0.624806){
// This is a leaf node
result += -0.008871;
}
else { // if condition is not respected
// This is a leaf node
result += -0.078910;
}
}
else { // if condition is not respected
if (event[1] < 1.027388){
// This is a leaf node
result += 0.000123;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002892;
}
}
}
if (event[1] < 0.591687){
if (event[1] < 0.583061){
if (event[1] < 0.580580){
// This is a leaf node
result += -0.000142;
}
else { // if condition is not respected
// This is a leaf node
result += 0.042578;
}
}
else { // if condition is not respected
if (event[3] < 0.454173){
// This is a leaf node
result += 0.001387;
}
else { // if condition is not respected
// This is a leaf node
result += -0.038105;
}
}
}
else { // if condition is not respected
if (event[1] < 0.592120){
if (event[3] < 0.998637){
// This is a leaf node
result += 0.078382;
}
else { // if condition is not respected
// This is a leaf node
result += -0.055520;
}
}
else { // if condition is not respected
if (event[0] < 1.509514){
// This is a leaf node
result += 0.000625;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003767;
}
}
}
if (event[0] < 1.727004){
if (event[0] < 1.640039){
if (event[0] < 1.639606){
// This is a leaf node
result += 0.000014;
}
else { // if condition is not respected
// This is a leaf node
result += 0.129855;
}
}
else { // if condition is not respected
if (event[4] < 1.470741){
// This is a leaf node
result += -0.006099;
}
else { // if condition is not respected
// This is a leaf node
result += -0.035050;
}
}
}
else { // if condition is not respected
if (event[4] < 1.789107){
if (event[2] < -3.263287){
// This is a leaf node
result += 0.136518;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000340;
}
}
else { // if condition is not respected
if (event[0] < 1.838679){
// This is a leaf node
result += 0.065893;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006159;
}
}
}
if (event[4] < 1.614472){
if (event[4] < 1.578457){
if (event[4] < 1.577624){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += -0.042119;
}
}
else { // if condition is not respected
if (event[1] < -0.064562){
// This is a leaf node
result += 0.004040;
}
else { // if condition is not respected
// This is a leaf node
result += 0.025002;
}
}
}
else { // if condition is not respected
if (event[4] < 1.624825){
if (event[3] < -1.074820){
// This is a leaf node
result += 0.042089;
}
else { // if condition is not respected
// This is a leaf node
result += -0.040011;
}
}
else { // if condition is not respected
if (event[2] < -0.783193){
// This is a leaf node
result += 0.005673;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002008;
}
}
}
if (event[0] < 1.715416){
if (event[0] < 1.645012){
if (event[0] < 1.643492){
// This is a leaf node
result += 0.000010;
}
else { // if condition is not respected
// This is a leaf node
result += 0.048197;
}
}
else { // if condition is not respected
if (event[0] < 1.645618){
// This is a leaf node
result += -0.103509;
}
else { // if condition is not respected
// This is a leaf node
result += -0.008247;
}
}
}
else { // if condition is not respected
if (event[4] < 1.789107){
if (event[2] < -3.263287){
// This is a leaf node
result += 0.128068;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000319;
}
}
else { // if condition is not respected
if (event[3] < -0.506256){
// This is a leaf node
result += -0.013873;
}
else { // if condition is not respected
// This is a leaf node
result += 0.033172;
}
}
}
if (event[4] < 1.671135){
if (event[4] < 1.668587){
if (event[4] < 1.363607){
// This is a leaf node
result += -0.000058;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002377;
}
}
else { // if condition is not respected
if (event[3] < -0.672678){
// This is a leaf node
result += 0.098000;
}
else { // if condition is not respected
// This is a leaf node
result += 0.017546;
}
}
}
else { // if condition is not respected
if (event[4] < 1.673165){
if (event[3] < 1.228686){
// This is a leaf node
result += -0.063656;
}
else { // if condition is not respected
// This is a leaf node
result += 0.068242;
}
}
else { // if condition is not respected
if (event[1] < -1.034734){
// This is a leaf node
result += -0.007966;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000407;
}
}
}
if (event[4] < 0.911245){
if (event[4] < 0.911066){
if (event[4] < 0.821965){
// This is a leaf node
result += 0.000019;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002879;
}
}
else { // if condition is not respected
if (event[3] < -0.694308){
// This is a leaf node
result += 0.018355;
}
else { // if condition is not respected
// This is a leaf node
result += 0.112551;
}
}
}
else { // if condition is not respected
if (event[4] < 0.911548){
if (event[3] < -0.426623){
// This is a leaf node
result += 0.006079;
}
else { // if condition is not respected
// This is a leaf node
result += -0.105461;
}
}
else { // if condition is not respected
if (event[2] < 2.320230){
// This is a leaf node
result += -0.000330;
}
else { // if condition is not respected
// This is a leaf node
result += -0.013480;
}
}
}
if (event[2] < 2.903436){
if (event[2] < 2.898122){
if (event[2] < 2.895743){
// This is a leaf node
result += -0.000008;
}
else { // if condition is not respected
// This is a leaf node
result += 0.074065;
}
}
else { // if condition is not respected
if (event[1] < -0.512780){
// This is a leaf node
result += -0.031083;
}
else { // if condition is not respected
// This is a leaf node
result += -0.122870;
}
}
}
else { // if condition is not respected
if (event[3] < -1.862135){
if (event[4] < -0.128799){
// This is a leaf node
result += 0.034693;
}
else { // if condition is not respected
// This is a leaf node
result += -0.099241;
}
}
else { // if condition is not respected
if (event[1] < -0.026911){
// This is a leaf node
result += 0.019855;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003865;
}
}
}
if (event[4] < -0.888256){
if (event[4] < -0.888694){
if (event[3] < -1.711352){
// This is a leaf node
result += -0.006442;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000731;
}
}
else { // if condition is not respected
if (event[1] < -1.270059){
// This is a leaf node
result += -0.041275;
}
else { // if condition is not respected
// This is a leaf node
result += 0.083189;
}
}
}
else { // if condition is not respected
if (event[4] < -0.832112){
if (event[4] < -0.832569){
// This is a leaf node
result += -0.004322;
}
else { // if condition is not respected
// This is a leaf node
result += -0.066372;
}
}
else { // if condition is not respected
if (event[4] < -0.822230){
// This is a leaf node
result += 0.014827;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000064;
}
}
}
if (event[3] < -1.888217){
if (event[0] < -2.756541){
if (event[2] < -0.459121){
// This is a leaf node
result += 0.031782;
}
else { // if condition is not respected
// This is a leaf node
result += -0.105526;
}
}
else { // if condition is not respected
if (event[0] < -0.450949){
// This is a leaf node
result += 0.006546;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000986;
}
}
}
else { // if condition is not respected
if (event[3] < -1.886691){
if (event[0] < -0.439712){
// This is a leaf node
result += 0.028602;
}
else { // if condition is not respected
// This is a leaf node
result += -0.074738;
}
}
else { // if condition is not respected
if (event[3] < -1.886278){
// This is a leaf node
result += 0.063855;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000037;
}
}
}
if (event[0] < 0.004480){
if (event[0] < 0.004404){
if (event[0] < 0.004297){
// This is a leaf node
result += -0.000229;
}
else { // if condition is not respected
// This is a leaf node
result += 0.063995;
}
}
else { // if condition is not respected
if (event[1] < 0.236219){
// This is a leaf node
result += -0.116034;
}
else { // if condition is not respected
// This is a leaf node
result += -0.014373;
}
}
}
else { // if condition is not respected
if (event[0] < 0.004607){
if (event[2] < -0.534028){
// This is a leaf node
result += -0.001110;
}
else { // if condition is not respected
// This is a leaf node
result += 0.151138;
}
}
else { // if condition is not respected
if (event[0] < 0.060306){
// This is a leaf node
result += 0.004379;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000026;
}
}
}
if (event[2] < -0.365662){
if (event[2] < -0.433119){
if (event[2] < -0.436768){
// This is a leaf node
result += -0.000007;
}
else { // if condition is not respected
// This is a leaf node
result += 0.014898;
}
}
else { // if condition is not respected
if (event[0] < 0.952039){
// This is a leaf node
result += -0.003191;
}
else { // if condition is not respected
// This is a leaf node
result += -0.014487;
}
}
}
else { // if condition is not respected
if (event[2] < -0.365463){
if (event[3] < -0.518570){
// This is a leaf node
result += 0.145539;
}
else { // if condition is not respected
// This is a leaf node
result += 0.017062;
}
}
else { // if condition is not respected
if (event[3] < -1.999859){
// This is a leaf node
result += 0.003856;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000071;
}
}
}
if (event[2] < 3.173035){
if (event[2] < 3.084193){
if (event[2] < 3.064624){
// This is a leaf node
result += 0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += -0.072538;
}
}
else { // if condition is not respected
if (event[4] < 1.625332){
// This is a leaf node
result += 0.055874;
}
else { // if condition is not respected
// This is a leaf node
result += -0.086111;
}
}
}
else { // if condition is not respected
if (event[2] < 3.184392){
if (event[4] < -0.306575){
// This is a leaf node
result += -0.043718;
}
else { // if condition is not respected
// This is a leaf node
result += -0.123182;
}
}
else { // if condition is not respected
if (event[0] < -0.119905){
// This is a leaf node
result += 0.021627;
}
else { // if condition is not respected
// This is a leaf node
result += -0.022447;
}
}
}
if (event[3] < -4.097473){
// This is a leaf node
result += -0.052000;
}
else { // if condition is not respected
if (event[3] < -3.199995){
if (event[1] < 1.044176){
// This is a leaf node
result += 0.018644;
}
else { // if condition is not respected
// This is a leaf node
result += -0.044488;
}
}
else { // if condition is not respected
if (event[3] < -3.023203){
// This is a leaf node
result += -0.020411;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000007;
}
}
}
if (event[3] < 1.848377){
if (event[3] < 1.796716){
if (event[3] < 1.795864){
// This is a leaf node
result += -0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.070288;
}
}
else { // if condition is not respected
if (event[0] < 1.676267){
// This is a leaf node
result += 0.015004;
}
else { // if condition is not respected
// This is a leaf node
result += -0.049542;
}
}
}
else { // if condition is not respected
if (event[1] < 1.720694){
if (event[3] < 1.875726){
// This is a leaf node
result += -0.015374;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000953;
}
}
else { // if condition is not respected
if (event[4] < 0.932659){
// This is a leaf node
result += 0.004370;
}
else { // if condition is not respected
// This is a leaf node
result += 0.050378;
}
}
}
if (event[4] < 0.799553){
if (event[4] < 0.778987){
if (event[4] < 0.766351){
// This is a leaf node
result += 0.000095;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010074;
}
}
else { // if condition is not respected
if (event[4] < 0.779060){
// This is a leaf node
result += 0.139399;
}
else { // if condition is not respected
// This is a leaf node
result += 0.008433;
}
}
}
else { // if condition is not respected
if (event[4] < 0.813669){
if (event[2] < 2.447126){
// This is a leaf node
result += -0.018600;
}
else { // if condition is not respected
// This is a leaf node
result += 0.116352;
}
}
else { // if condition is not respected
if (event[2] < 2.806412){
// This is a leaf node
result += -0.000029;
}
else { // if condition is not respected
// This is a leaf node
result += -0.025721;
}
}
}
if (event[2] < 2.903436){
if (event[2] < 2.898122){
if (event[2] < 2.895743){
// This is a leaf node
result += -0.000008;
}
else { // if condition is not respected
// This is a leaf node
result += 0.069372;
}
}
else { // if condition is not respected
if (event[0] < 0.767917){
// This is a leaf node
result += -0.116073;
}
else { // if condition is not respected
// This is a leaf node
result += -0.027980;
}
}
}
else { // if condition is not respected
if (event[0] < -1.495049){
if (event[0] < -1.757706){
// This is a leaf node
result += -0.003754;
}
else { // if condition is not respected
// This is a leaf node
result += 0.102128;
}
}
else { // if condition is not respected
if (event[0] < -1.149179){
// This is a leaf node
result += -0.048457;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005663;
}
}
}
if (event[1] < 1.466723){
if (event[1] < 1.465180){
if (event[1] < 1.463403){
// This is a leaf node
result += 0.000059;
}
else { // if condition is not respected
// This is a leaf node
result += -0.027267;
}
}
else { // if condition is not respected
if (event[0] < -1.347372){
// This is a leaf node
result += -0.085348;
}
else { // if condition is not respected
// This is a leaf node
result += 0.052095;
}
}
}
else { // if condition is not respected
if (event[1] < 1.467420){
if (event[4] < 0.039426){
// This is a leaf node
result += -0.145795;
}
else { // if condition is not respected
// This is a leaf node
result += -0.036040;
}
}
else { // if condition is not respected
if (event[1] < 1.505394){
// This is a leaf node
result += -0.007515;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000135;
}
}
}
if (event[4] < -1.140589){
if (event[4] < -1.141200){
if (event[3] < 3.154894){
// This is a leaf node
result += -0.000464;
}
else { // if condition is not respected
// This is a leaf node
result += -0.062631;
}
}
else { // if condition is not respected
if (event[3] < 0.899428){
// This is a leaf node
result += -0.083324;
}
else { // if condition is not respected
// This is a leaf node
result += 0.047365;
}
}
}
else { // if condition is not respected
if (event[4] < -1.117706){
if (event[4] < -1.119072){
// This is a leaf node
result += 0.008154;
}
else { // if condition is not respected
// This is a leaf node
result += 0.076511;
}
}
else { // if condition is not respected
if (event[4] < -1.116429){
// This is a leaf node
result += -0.038464;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000030;
}
}
}
if (event[3] < 3.113537){
if (event[3] < 3.086900){
if (event[3] < 3.081855){
// This is a leaf node
result += -0.000004;
}
else { // if condition is not respected
// This is a leaf node
result += 0.066326;
}
}
else { // if condition is not respected
if (event[4] < -0.985447){
// This is a leaf node
result += 0.049611;
}
else { // if condition is not respected
// This is a leaf node
result += -0.059206;
}
}
}
else { // if condition is not respected
if (event[3] < 3.157028){
if (event[1] < 0.828446){
// This is a leaf node
result += 0.043727;
}
else { // if condition is not respected
// This is a leaf node
result += 0.144106;
}
}
else { // if condition is not respected
if (event[2] < 0.326275){
// This is a leaf node
result += 0.015672;
}
else { // if condition is not respected
// This is a leaf node
result += -0.035443;
}
}
}
if (event[3] < 2.999842){
if (event[3] < 2.996858){
if (event[3] < 2.727612){
// This is a leaf node
result += 0.000009;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007698;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.133841;
}
}
else { // if condition is not respected
if (event[4] < -1.136159){
if (event[0] < -0.819138){
// This is a leaf node
result += 0.043229;
}
else { // if condition is not respected
// This is a leaf node
result += -0.067637;
}
}
else { // if condition is not respected
if (event[4] < -0.960236){
// This is a leaf node
result += 0.094899;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009192;
}
}
}
if (event[2] < 4.023448){
if (event[2] < 3.986183){
if (event[4] < -3.274415){
// This is a leaf node
result += 0.009202;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000005;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.077760;
}
}
else { // if condition is not respected
if (event[4] < -0.005265){
// This is a leaf node
result += -0.010166;
}
else { // if condition is not respected
// This is a leaf node
result += 0.069927;
}
}
if (event[4] < -2.843313){
if (event[3] < 1.235030){
if (event[3] < 1.012421){
// This is a leaf node
result += -0.004846;
}
else { // if condition is not respected
// This is a leaf node
result += -0.061231;
}
}
else { // if condition is not respected
if (event[1] < -0.521716){
// This is a leaf node
result += 0.080141;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009285;
}
}
}
else { // if condition is not respected
if (event[4] < -2.817898){
if (event[1] < -0.156040){
// This is a leaf node
result += -0.014677;
}
else { // if condition is not respected
// This is a leaf node
result += 0.061769;
}
}
else { // if condition is not respected
if (event[4] < -2.814892){
// This is a leaf node
result += -0.079069;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000007;
}
}
}
if (event[0] < 3.125710){
if (event[0] < 2.997290){
if (event[0] < 2.987079){
// This is a leaf node
result += -0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += 0.067674;
}
}
else { // if condition is not respected
if (event[2] < -0.838161){
// This is a leaf node
result += -0.071063;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006539;
}
}
}
else { // if condition is not respected
if (event[1] < 1.114425){
if (event[1] < -0.911887){
// This is a leaf node
result += 0.054793;
}
else { // if condition is not respected
// This is a leaf node
result += -0.013974;
}
}
else { // if condition is not respected
if (event[1] < 1.889359){
// This is a leaf node
result += 0.084428;
}
else { // if condition is not respected
// This is a leaf node
result += -0.038329;
}
}
}
if (event[0] < 3.332963){
if (event[0] < 3.308728){
if (event[0] < 3.300047){
// This is a leaf node
result += 0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += -0.075282;
}
}
else { // if condition is not respected
if (event[3] < 0.588757){
// This is a leaf node
result += 0.114160;
}
else { // if condition is not respected
// This is a leaf node
result += -0.016369;
}
}
}
else { // if condition is not respected
if (event[2] < 1.648108){
if (event[2] < -0.064565){
// This is a leaf node
result += 0.013859;
}
else { // if condition is not respected
// This is a leaf node
result += -0.051495;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.109531;
}
}
if (event[0] < 3.073634){
if (event[0] < 3.058667){
if (event[0] < 3.049398){
// This is a leaf node
result += -0.000005;
}
else { // if condition is not respected
// This is a leaf node
result += 0.075814;
}
}
else { // if condition is not respected
if (event[2] < 0.558834){
// This is a leaf node
result += -0.123136;
}
else { // if condition is not respected
// This is a leaf node
result += 0.027771;
}
}
}
else { // if condition is not respected
if (event[0] < 3.224831){
if (event[2] < -1.547157){
// This is a leaf node
result += -0.121900;
}
else { // if condition is not respected
// This is a leaf node
result += 0.033263;
}
}
else { // if condition is not respected
if (event[2] < 1.648108){
// This is a leaf node
result += -0.012365;
}
else { // if condition is not respected
// This is a leaf node
result += 0.091278;
}
}
}
if (event[0] < 4.111604){
if (event[0] < 3.515198){
if (event[0] < 3.430337){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.056359;
}
}
else { // if condition is not respected
if (event[2] < 0.882167){
// This is a leaf node
result += -0.040444;
}
else { // if condition is not respected
// This is a leaf node
result += 0.087081;
}
}
}
else { // if condition is not respected
if (event[0] < 4.301860){
// This is a leaf node
result += 0.080713;
}
else { // if condition is not respected
// This is a leaf node
result += -0.009341;
}
}
if (event[3] < -3.863437){
if (event[2] < -0.379601){
// This is a leaf node
result += 0.045384;
}
else { // if condition is not respected
if (event[0] < -0.002408){
// This is a leaf node
result += -0.000765;
}
else { // if condition is not respected
// This is a leaf node
result += -0.081159;
}
}
}
else { // if condition is not respected
if (event[3] < -3.672421){
if (event[0] < -0.497434){
// This is a leaf node
result += -0.018176;
}
else { // if condition is not respected
// This is a leaf node
result += 0.082138;
}
}
else { // if condition is not respected
if (event[3] < -3.653632){
// This is a leaf node
result += -0.055692;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000000;
}
}
}
if (event[3] < -0.202826){
if (event[3] < -0.204509){
if (event[3] < -0.204837){
// This is a leaf node
result += -0.000241;
}
else { // if condition is not respected
// This is a leaf node
result += 0.048585;
}
}
else { // if condition is not respected
if (event[4] < 1.402225){
// This is a leaf node
result += -0.029244;
}
else { // if condition is not respected
// This is a leaf node
result += 0.047655;
}
}
}
else { // if condition is not respected
if (event[3] < -0.202761){
if (event[0] < 0.578909){
// This is a leaf node
result += 0.096661;
}
else { // if condition is not respected
// This is a leaf node
result += 0.011735;
}
}
else { // if condition is not respected
if (event[3] < 0.090088){
// This is a leaf node
result += 0.001520;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000160;
}
}
}
if (event[3] < 1.045245){
if (event[3] < 1.045181){
if (event[3] < 0.980328){
// This is a leaf node
result += -0.000020;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003661;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.113907;
}
}
else { // if condition is not respected
if (event[1] < -2.833028){
if (event[1] < -3.022195){
// This is a leaf node
result += 0.004585;
}
else { // if condition is not respected
// This is a leaf node
result += 0.057024;
}
}
else { // if condition is not respected
if (event[1] < -2.813704){
// This is a leaf node
result += -0.102260;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000479;
}
}
}
if (event[3] < 1.568277){
if (event[3] < 1.427665){
if (event[3] < 1.378493){
// This is a leaf node
result += 0.000038;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006881;
}
}
else { // if condition is not respected
if (event[3] < 1.440740){
// This is a leaf node
result += 0.020609;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001647;
}
}
}
else { // if condition is not respected
if (event[3] < 1.648609){
if (event[3] < 1.648139){
// This is a leaf node
result += -0.007037;
}
else { // if condition is not respected
// This is a leaf node
result += -0.095161;
}
}
else { // if condition is not respected
if (event[3] < 1.666581){
// This is a leaf node
result += 0.015807;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000294;
}
}
}
if (event[1] < -2.780941){
if (event[4] < -2.300618){
if (event[4] < -2.867363){
// This is a leaf node
result += 0.031119;
}
else { // if condition is not respected
// This is a leaf node
result += 0.128414;
}
}
else { // if condition is not respected
if (event[1] < -2.838322){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += -0.034760;
}
}
}
else { // if condition is not respected
if (event[1] < -2.662789){
if (event[4] < -0.444351){
// This is a leaf node
result += -0.013913;
}
else { // if condition is not respected
// This is a leaf node
result += 0.035191;
}
}
else { // if condition is not respected
if (event[1] < -2.641408){
// This is a leaf node
result += -0.046235;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000001;
}
}
}
if (event[1] < 4.088089){
if (event[1] < 4.035286){
if (event[1] < 3.906952){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.041842;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.079988;
}
}
else { // if condition is not respected
if (event[2] < 0.152884){
// This is a leaf node
result += -0.000979;
}
else { // if condition is not respected
// This is a leaf node
result += -0.059238;
}
}
if (event[4] < 4.442656){
if (event[4] < 4.082553){
if (event[4] < 3.937248){
// This is a leaf node
result += 0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += -0.068761;
}
}
else { // if condition is not respected
if (event[1] < -0.822735){
// This is a leaf node
result += -0.052797;
}
else { // if condition is not respected
// This is a leaf node
result += 0.116298;
}
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.051255;
}
if (event[0] < -0.097553){
if (event[0] < -0.100051){
if (event[0] < -0.100075){
// This is a leaf node
result += -0.000180;
}
else { // if condition is not respected
// This is a leaf node
result += 0.114075;
}
}
else { // if condition is not respected
if (event[1] < -1.822826){
// This is a leaf node
result += 0.070597;
}
else { // if condition is not respected
// This is a leaf node
result += -0.033442;
}
}
}
else { // if condition is not respected
if (event[0] < -0.094676){
if (event[1] < -0.334691){
// This is a leaf node
result += 0.006653;
}
else { // if condition is not respected
// This is a leaf node
result += 0.045826;
}
}
else { // if condition is not respected
if (event[1] < 3.150256){
// This is a leaf node
result += 0.000159;
}
else { // if condition is not respected
// This is a leaf node
result += -0.026365;
}
}
}
if (event[1] < 3.303042){
if (event[1] < 3.276201){
if (event[1] < 3.255574){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.065933;
}
}
else { // if condition is not respected
if (event[3] < 0.535346){
// This is a leaf node
result += -0.132798;
}
else { // if condition is not respected
// This is a leaf node
result += -0.008336;
}
}
}
else { // if condition is not respected
if (event[0] < 0.665794){
if (event[0] < 0.410594){
// This is a leaf node
result += 0.014484;
}
else { // if condition is not respected
// This is a leaf node
result += 0.089513;
}
}
else { // if condition is not respected
if (event[3] < -0.089537){
// This is a leaf node
result += -0.073938;
}
else { // if condition is not respected
// This is a leaf node
result += 0.015389;
}
}
}
if (event[1] < 3.020258){
if (event[1] < 3.004412){
if (event[1] < 2.922445){
// This is a leaf node
result += 0.000005;
}
else { // if condition is not respected
// This is a leaf node
result += -0.016430;
}
}
else { // if condition is not respected
if (event[0] < -0.451652){
// This is a leaf node
result += -0.003885;
}
else { // if condition is not respected
// This is a leaf node
result += -0.112556;
}
}
}
else { // if condition is not respected
if (event[0] < 1.669399){
if (event[2] < 0.167537){
// This is a leaf node
result += -0.004365;
}
else { // if condition is not respected
// This is a leaf node
result += 0.029298;
}
}
else { // if condition is not respected
if (event[1] < 3.074541){
// This is a leaf node
result += 0.043546;
}
else { // if condition is not respected
// This is a leaf node
result += -0.096708;
}
}
}
if (event[1] < -2.983040){
if (event[4] < -2.300618){
if (event[3] < -0.305254){
// This is a leaf node
result += 0.101880;
}
else { // if condition is not respected
// This is a leaf node
result += 0.021838;
}
}
else { // if condition is not respected
if (event[4] < -1.587281){
// This is a leaf node
result += -0.064835;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003515;
}
}
}
else { // if condition is not respected
if (event[1] < -2.924611){
if (event[2] < -0.136401){
// This is a leaf node
result += -0.014213;
}
else { // if condition is not respected
// This is a leaf node
result += 0.066667;
}
}
else { // if condition is not respected
if (event[1] < -2.900012){
// This is a leaf node
result += -0.045701;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000009;
}
}
}
if (event[1] < -0.608965){
if (event[1] < -0.609987){
if (event[4] < 1.859754){
// This is a leaf node
result += 0.000506;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005908;
}
}
else { // if condition is not respected
if (event[3] < 0.226704){
// This is a leaf node
result += 0.064348;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004934;
}
}
}
else { // if condition is not respected
if (event[1] < -0.607856){
if (event[4] < 1.629230){
// This is a leaf node
result += -0.047916;
}
else { // if condition is not respected
// This is a leaf node
result += 0.087385;
}
}
else { // if condition is not respected
if (event[1] < -0.607732){
// This is a leaf node
result += 0.081169;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000119;
}
}
}
if (event[4] < 2.092473){
if (event[4] < 2.083399){
if (event[4] < 1.861557){
// This is a leaf node
result += 0.000030;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003392;
}
}
else { // if condition is not respected
if (event[1] < -0.420807){
// This is a leaf node
result += -0.087412;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007090;
}
}
}
else { // if condition is not respected
if (event[4] < 2.093662){
if (event[1] < 0.161588){
// This is a leaf node
result += 0.026463;
}
else { // if condition is not respected
// This is a leaf node
result += 0.147265;
}
}
else { // if condition is not respected
if (event[3] < 1.991593){
// This is a leaf node
result += 0.002044;
}
else { // if condition is not respected
// This is a leaf node
result += -0.030487;
}
}
}
if (event[4] < 1.226623){
if (event[4] < 1.226062){
if (event[4] < 1.200546){
// This is a leaf node
result += -0.000020;
}
else { // if condition is not respected
// This is a leaf node
result += -0.008802;
}
}
else { // if condition is not respected
if (event[2] < 0.694371){
// This is a leaf node
result += -0.129164;
}
else { // if condition is not respected
// This is a leaf node
result += 0.026587;
}
}
}
else { // if condition is not respected
if (event[4] < 1.279068){
if (event[1] < 1.247748){
// This is a leaf node
result += 0.006503;
}
else { // if condition is not respected
// This is a leaf node
result += 0.030750;
}
}
else { // if condition is not respected
if (event[4] < 1.314252){
// This is a leaf node
result += -0.011675;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000559;
}
}
}
if (event[3] < 2.152134){
if (event[3] < 2.139556){
if (event[3] < 2.138307){
// This is a leaf node
result += -0.000011;
}
else { // if condition is not respected
// This is a leaf node
result += 0.059153;
}
}
else { // if condition is not respected
if (event[4] < 0.321886){
// This is a leaf node
result += -0.060784;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002147;
}
}
}
else { // if condition is not respected
if (event[1] < -0.908206){
if (event[4] < -0.569828){
// This is a leaf node
result += 0.040530;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004868;
}
}
else { // if condition is not respected
if (event[0] < 1.931198){
// This is a leaf node
result += -0.002311;
}
else { // if condition is not respected
// This is a leaf node
result += 0.036749;
}
}
}
if (event[2] < 1.368580){
if (event[2] < 1.341092){
if (event[2] < 1.339100){
// This is a leaf node
result += 0.000027;
}
else { // if condition is not respected
// This is a leaf node
result += -0.037437;
}
}
else { // if condition is not respected
if (event[1] < -0.017659){
// This is a leaf node
result += -0.000410;
}
else { // if condition is not respected
// This is a leaf node
result += 0.021009;
}
}
}
else { // if condition is not respected
if (event[3] < 2.832452){
if (event[3] < 2.725131){
// This is a leaf node
result += -0.000724;
}
else { // if condition is not respected
// This is a leaf node
result += -0.104022;
}
}
else { // if condition is not respected
if (event[0] < -0.701021){
// This is a leaf node
result += -0.035329;
}
else { // if condition is not respected
// This is a leaf node
result += 0.073896;
}
}
}
if (event[3] < 3.589159){
if (event[3] < 3.262835){
if (event[3] < 3.257366){
// This is a leaf node
result += -0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.093452;
}
}
else { // if condition is not respected
if (event[1] < -1.310636){
// This is a leaf node
result += -0.072822;
}
else { // if condition is not respected
// This is a leaf node
result += 0.028663;
}
}
}
else { // if condition is not respected
if (event[1] < -1.425893){
// This is a leaf node
result += 0.056053;
}
else { // if condition is not respected
if (event[0] < -0.055124){
// This is a leaf node
result += -0.080286;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002295;
}
}
}
if (event[3] < 3.367037){
if (event[3] < 3.262835){
if (event[3] < 3.205907){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.035566;
}
}
else { // if condition is not respected
if (event[1] < 0.271883){
// This is a leaf node
result += 0.001761;
}
else { // if condition is not respected
// This is a leaf node
result += 0.093842;
}
}
}
else { // if condition is not respected
if (event[0] < 0.397784){
if (event[2] < 0.277185){
// This is a leaf node
result += 0.043048;
}
else { // if condition is not respected
// This is a leaf node
result += -0.045801;
}
}
else { // if condition is not respected
if (event[0] < 1.443869){
// This is a leaf node
result += -0.063348;
}
else { // if condition is not respected
// This is a leaf node
result += 0.012787;
}
}
}
if (event[0] < 2.237703){
if (event[0] < 2.141204){
if (event[0] < 2.102566){
// This is a leaf node
result += 0.000017;
}
else { // if condition is not respected
// This is a leaf node
result += -0.012419;
}
}
else { // if condition is not respected
if (event[3] < -1.549615){
// This is a leaf node
result += 0.058467;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004768;
}
}
}
else { // if condition is not respected
if (event[3] < -3.125730){
if (event[3] < -3.297996){
// This is a leaf node
result += 0.029896;
}
else { // if condition is not respected
// This is a leaf node
result += 0.113739;
}
}
else { // if condition is not respected
if (event[3] < -2.282502){
// This is a leaf node
result += -0.054665;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001625;
}
}
}
if (event[4] < -0.370966){
if (event[4] < -0.371560){
if (event[0] < 0.461815){
// This is a leaf node
result += -0.000939;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001192;
}
}
else { // if condition is not respected
if (event[3] < -1.919694){
// This is a leaf node
result += 0.076377;
}
else { // if condition is not respected
// This is a leaf node
result += -0.058405;
}
}
}
else { // if condition is not respected
if (event[4] < -0.359192){
if (event[4] < -0.361489){
// This is a leaf node
result += 0.003762;
}
else { // if condition is not respected
// This is a leaf node
result += 0.035406;
}
}
else { // if condition is not respected
if (event[0] < 0.594992){
// This is a leaf node
result += 0.000486;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000949;
}
}
}
if (event[0] < 1.713806){
if (event[0] < 1.640020){
if (event[0] < 1.639606){
// This is a leaf node
result += 0.000008;
}
else { // if condition is not respected
// This is a leaf node
result += 0.127057;
}
}
else { // if condition is not respected
if (event[2] < -1.001728){
// This is a leaf node
result += -0.025445;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004637;
}
}
}
else { // if condition is not respected
if (event[4] < 1.789107){
if (event[4] < 0.686983){
// This is a leaf node
result += 0.002216;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006131;
}
}
else { // if condition is not respected
if (event[0] < 1.838679){
// This is a leaf node
result += 0.052278;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004296;
}
}
}
if (event[4] < 2.553708){
if (event[4] < 2.534851){
if (event[4] < 2.529881){
// This is a leaf node
result += 0.000012;
}
else { // if condition is not respected
// This is a leaf node
result += -0.069430;
}
}
else { // if condition is not respected
if (event[3] < -1.908523){
// This is a leaf node
result += -0.117034;
}
else { // if condition is not respected
// This is a leaf node
result += 0.038105;
}
}
}
else { // if condition is not respected
if (event[4] < 2.568027){
if (event[1] < -1.412363){
// This is a leaf node
result += 0.069055;
}
else { // if condition is not respected
// This is a leaf node
result += -0.053843;
}
}
else { // if condition is not respected
if (event[3] < -2.491692){
// This is a leaf node
result += -0.111431;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000678;
}
}
}
if (event[4] < -3.586279){
if (event[1] < 0.038694){
if (event[3] < 0.782887){
// This is a leaf node
result += 0.009438;
}
else { // if condition is not respected
// This is a leaf node
result += -0.112316;
}
}
else { // if condition is not respected
if (event[2] < 0.553979){
// This is a leaf node
result += 0.081142;
}
else { // if condition is not respected
// This is a leaf node
result += -0.037959;
}
}
}
else { // if condition is not respected
if (event[4] < -3.457016){
if (event[2] < 0.844353){
// This is a leaf node
result += -0.073258;
}
else { // if condition is not respected
// This is a leaf node
result += 0.077109;
}
}
else { // if condition is not respected
if (event[4] < -3.274415){
// This is a leaf node
result += 0.022215;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000005;
}
}
}
if (event[3] < -0.032061){
if (event[3] < -0.037370){
if (event[3] < -0.037469){
// This is a leaf node
result += -0.000153;
}
else { // if condition is not respected
// This is a leaf node
result += 0.086366;
}
}
else { // if condition is not respected
if (event[1] < 0.537285){
// This is a leaf node
result += -0.007378;
}
else { // if condition is not respected
// This is a leaf node
result += -0.038725;
}
}
}
else { // if condition is not respected
if (event[3] < 0.070226){
if (event[4] < 0.369382){
// This is a leaf node
result += 0.000965;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009209;
}
}
else { // if condition is not respected
if (event[3] < 0.070657){
// This is a leaf node
result += -0.056065;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000097;
}
}
}
if (event[3] < -1.135062){
if (event[3] < -1.143976){
if (event[3] < -1.144054){
// This is a leaf node
result += 0.000337;
}
else { // if condition is not respected
// This is a leaf node
result += -0.100805;
}
}
else { // if condition is not respected
if (event[3] < -1.143877){
// This is a leaf node
result += 0.110834;
}
else { // if condition is not respected
// This is a leaf node
result += 0.016439;
}
}
}
else { // if condition is not respected
if (event[3] < -1.130239){
if (event[1] < 1.208377){
// This is a leaf node
result += -0.007880;
}
else { // if condition is not respected
// This is a leaf node
result += -0.079610;
}
}
else { // if condition is not respected
if (event[3] < -1.126782){
// This is a leaf node
result += 0.018400;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000079;
}
}
}
if (event[1] < 0.593804){
if (event[1] < 0.583061){
if (event[1] < 0.580580){
// This is a leaf node
result += -0.000132;
}
else { // if condition is not respected
// This is a leaf node
result += 0.038441;
}
}
else { // if condition is not respected
if (event[3] < 1.077688){
// This is a leaf node
result += -0.002107;
}
else { // if condition is not respected
// This is a leaf node
result += -0.047582;
}
}
}
else { // if condition is not respected
if (event[1] < 0.594493){
if (event[0] < -0.333093){
// This is a leaf node
result += 0.110679;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009783;
}
}
else { // if condition is not respected
if (event[0] < 2.192036){
// This is a leaf node
result += 0.000429;
}
else { // if condition is not respected
// This is a leaf node
result += -0.008444;
}
}
}
if (event[0] < 2.659779){
if (event[0] < 2.641921){
if (event[0] < 2.524950){
// This is a leaf node
result += 0.000010;
}
else { // if condition is not respected
// This is a leaf node
result += -0.009605;
}
}
else { // if condition is not respected
if (event[2] < 0.105708){
// This is a leaf node
result += -0.084293;
}
else { // if condition is not respected
// This is a leaf node
result += 0.011940;
}
}
}
else { // if condition is not respected
if (event[0] < 2.700930){
if (event[3] < 0.921435){
// This is a leaf node
result += 0.052510;
}
else { // if condition is not respected
// This is a leaf node
result += -0.020466;
}
}
else { // if condition is not respected
if (event[1] < -1.111949){
// This is a leaf node
result += -0.027510;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002437;
}
}
}
if (event[3] < 1.045245){
if (event[3] < 1.044841){
if (event[3] < 1.044184){
// This is a leaf node
result += -0.000092;
}
else { // if condition is not respected
// This is a leaf node
result += 0.035521;
}
}
else { // if condition is not respected
if (event[4] < -0.049362){
// This is a leaf node
result += -0.007853;
}
else { // if condition is not respected
// This is a leaf node
result += -0.105454;
}
}
}
else { // if condition is not respected
if (event[3] < 1.047289){
if (event[4] < 0.512301){
// This is a leaf node
result += 0.041645;
}
else { // if condition is not respected
// This is a leaf node
result += -0.020363;
}
}
else { // if condition is not respected
if (event[3] < 1.047441){
// This is a leaf node
result += -0.098292;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000471;
}
}
}
if (event[0] < 1.713806){
if (event[0] < 1.676069){
if (event[0] < 1.671801){
// This is a leaf node
result += -0.000023;
}
else { // if condition is not respected
// This is a leaf node
result += 0.035388;
}
}
else { // if condition is not respected
if (event[4] < 1.817835){
// This is a leaf node
result += -0.008229;
}
else { // if condition is not respected
// This is a leaf node
result += -0.065457;
}
}
}
else { // if condition is not respected
if (event[2] < -3.263287){
// This is a leaf node
result += 0.121220;
}
else { // if condition is not respected
if (event[2] < -2.968040){
// This is a leaf node
result += -0.092933;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001054;
}
}
}
if (event[4] < -0.888006){
if (event[0] < 0.983173){
if (event[0] < 0.963797){
// This is a leaf node
result += -0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.026963;
}
}
else { // if condition is not respected
if (event[2] < -0.841747){
// This is a leaf node
result += -0.005666;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005804;
}
}
}
else { // if condition is not respected
if (event[4] < -0.886209){
if (event[4] < -0.886578){
// This is a leaf node
result += -0.003034;
}
else { // if condition is not respected
// This is a leaf node
result += -0.089920;
}
}
else { // if condition is not respected
if (event[4] < -0.651486){
// This is a leaf node
result += -0.001876;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000080;
}
}
}
if (event[0] < -0.983987){
if (event[0] < -0.985712){
if (event[4] < -0.994444){
// This is a leaf node
result += 0.003807;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000308;
}
}
else { // if condition is not respected
if (event[3] < -0.881657){
// This is a leaf node
result += 0.104739;
}
else { // if condition is not respected
// This is a leaf node
result += 0.027718;
}
}
}
else { // if condition is not respected
if (event[0] < -0.853287){
if (event[4] < -0.384105){
// This is a leaf node
result += -0.009778;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000477;
}
}
else { // if condition is not respected
if (event[0] < -0.847749){
// This is a leaf node
result += 0.026989;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000010;
}
}
}
if (event[4] < -1.142751){
if (event[4] < -1.159955){
if (event[4] < -1.160022){
// This is a leaf node
result += -0.000268;
}
else { // if condition is not respected
// This is a leaf node
result += 0.133778;
}
}
else { // if condition is not respected
if (event[3] < -1.735446){
// This is a leaf node
result += -0.103866;
}
else { // if condition is not respected
// This is a leaf node
result += -0.008657;
}
}
}
else { // if condition is not respected
if (event[4] < -1.117706){
if (event[4] < -1.119072){
// This is a leaf node
result += 0.006738;
}
else { // if condition is not respected
// This is a leaf node
result += 0.069110;
}
}
else { // if condition is not respected
if (event[4] < -1.116429){
// This is a leaf node
result += -0.034812;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000035;
}
}
}
if (event[0] < 4.010478){
if (event[0] < 3.332277){
if (event[0] < 3.311776){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += 0.091479;
}
}
else { // if condition is not respected
if (event[2] < 1.648108){
// This is a leaf node
result += -0.022784;
}
else { // if condition is not respected
// This is a leaf node
result += 0.101082;
}
}
}
else { // if condition is not respected
if (event[3] < 0.132275){
if (event[3] < -0.557640){
// This is a leaf node
result += 0.037581;
}
else { // if condition is not respected
// This is a leaf node
result += -0.048896;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.073877;
}
}
if (event[0] < 3.125710){
if (event[0] < 3.107625){
if (event[0] < 3.100054){
// This is a leaf node
result += -0.000006;
}
else { // if condition is not respected
// This is a leaf node
result += 0.055652;
}
}
else { // if condition is not respected
if (event[4] < 0.368731){
// This is a leaf node
result += -0.016207;
}
else { // if condition is not respected
// This is a leaf node
result += -0.106140;
}
}
}
else { // if condition is not respected
if (event[1] < 1.182344){
if (event[1] < -0.386664){
// This is a leaf node
result += 0.032264;
}
else { // if condition is not respected
// This is a leaf node
result += -0.018241;
}
}
else { // if condition is not respected
if (event[1] < 1.889359){
// This is a leaf node
result += 0.085747;
}
else { // if condition is not respected
// This is a leaf node
result += -0.036684;
}
}
}
if (event[4] < -2.219914){
if (event[4] < -2.373916){
if (event[4] < -2.400029){
// This is a leaf node
result += 0.000152;
}
else { // if condition is not respected
// This is a leaf node
result += 0.028352;
}
}
else { // if condition is not respected
if (event[3] < -0.644351){
// This is a leaf node
result += -0.036566;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000277;
}
}
}
else { // if condition is not respected
if (event[4] < -2.153306){
if (event[3] < 0.630653){
// This is a leaf node
result += 0.019641;
}
else { // if condition is not respected
// This is a leaf node
result += -0.017259;
}
}
else { // if condition is not respected
if (event[4] < -2.149900){
// This is a leaf node
result += -0.043184;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000005;
}
}
}
if (event[4] < -2.781338){
if (event[2] < -1.924009){
if (event[1] < 1.396391){
// This is a leaf node
result += -0.081404;
}
else { // if condition is not respected
// This is a leaf node
result += 0.031665;
}
}
else { // if condition is not respected
if (event[2] < -0.115107){
// This is a leaf node
result += 0.011846;
}
else { // if condition is not respected
// This is a leaf node
result += -0.012584;
}
}
}
else { // if condition is not respected
if (event[4] < -2.757866){
if (event[2] < -0.726398){
// This is a leaf node
result += 0.119064;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020130;
}
}
else { // if condition is not respected
if (event[4] < -2.748747){
// This is a leaf node
result += -0.061065;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000007;
}
}
}
if (event[0] < -4.139326){
// This is a leaf node
result += 0.044842;
}
else { // if condition is not respected
if (event[0] < -3.967169){
// This is a leaf node
result += -0.102074;
}
else { // if condition is not respected
if (event[0] < -3.841982){
// This is a leaf node
result += 0.058441;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000001;
}
}
}
if (event[4] < 0.895738){
if (event[4] < 0.895163){
if (event[4] < 0.894723){
// This is a leaf node
result += 0.000096;
}
else { // if condition is not respected
// This is a leaf node
result += -0.059100;
}
}
else { // if condition is not respected
if (event[1] < -1.008929){
// This is a leaf node
result += -0.027200;
}
else { // if condition is not respected
// This is a leaf node
result += 0.071337;
}
}
}
else { // if condition is not respected
if (event[4] < 0.898167){
if (event[1] < -0.090785){
// This is a leaf node
result += -0.057069;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009257;
}
}
else { // if condition is not respected
if (event[4] < 0.899856){
// This is a leaf node
result += 0.026253;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000426;
}
}
}
if (event[1] < 3.020258){
if (event[1] < 3.004412){
if (event[1] < 2.817672){
// This is a leaf node
result += 0.000008;
}
else { // if condition is not respected
// This is a leaf node
result += -0.009013;
}
}
else { // if condition is not respected
if (event[2] < 0.416007){
// This is a leaf node
result += -0.094888;
}
else { // if condition is not respected
// This is a leaf node
result += 0.016651;
}
}
}
else { // if condition is not respected
if (event[0] < -0.553341){
if (event[2] < 1.453141){
// This is a leaf node
result += 0.040545;
}
else { // if condition is not respected
// This is a leaf node
result += -0.063776;
}
}
else { // if condition is not respected
if (event[3] < -1.356803){
// This is a leaf node
result += 0.061188;
}
else { // if condition is not respected
// This is a leaf node
result += -0.009906;
}
}
}
if (event[1] < 2.359541){
if (event[1] < 2.337770){
if (event[1] < 2.337262){
// This is a leaf node
result += -0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += 0.072757;
}
}
else { // if condition is not respected
if (event[2] < -1.561372){
// This is a leaf node
result += 0.035614;
}
else { // if condition is not respected
// This is a leaf node
result += -0.042314;
}
}
}
else { // if condition is not respected
if (event[3] < 1.652078){
if (event[3] < 1.604525){
// This is a leaf node
result += 0.001435;
}
else { // if condition is not respected
// This is a leaf node
result += -0.071944;
}
}
else { // if condition is not respected
if (event[4] < 0.761447){
// This is a leaf node
result += 0.007914;
}
else { // if condition is not respected
// This is a leaf node
result += 0.090524;
}
}
}
if (event[2] < 0.710639){
if (event[2] < 0.705562){
if (event[2] < 0.705247){
// This is a leaf node
result += -0.000083;
}
else { // if condition is not respected
// This is a leaf node
result += 0.046262;
}
}
else { // if condition is not respected
if (event[2] < 0.705757){
// This is a leaf node
result += -0.102697;
}
else { // if condition is not respected
// This is a leaf node
result += -0.016266;
}
}
}
else { // if condition is not respected
if (event[2] < 0.712231){
if (event[2] < 0.711755){
// This is a leaf node
result += 0.008055;
}
else { // if condition is not respected
// This is a leaf node
result += 0.080992;
}
}
else { // if condition is not respected
if (event[2] < 0.712592){
// This is a leaf node
result += -0.069582;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000335;
}
}
}
if (event[2] < 1.368580){
if (event[2] < 1.272424){
if (event[2] < 1.272307){
// This is a leaf node
result += -0.000028;
}
else { // if condition is not respected
// This is a leaf node
result += -0.073477;
}
}
else { // if condition is not respected
if (event[4] < -0.975827){
// This is a leaf node
result += -0.007434;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007880;
}
}
}
else { // if condition is not respected
if (event[3] < 2.832452){
if (event[3] < 2.725131){
// This is a leaf node
result += -0.000699;
}
else { // if condition is not respected
// This is a leaf node
result += -0.095385;
}
}
else { // if condition is not respected
if (event[1] < 0.054232){
// This is a leaf node
result += 0.082902;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011705;
}
}
}
if (event[3] < 3.367037){
if (event[3] < 3.262835){
if (event[3] < 3.257366){
// This is a leaf node
result += -0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.087695;
}
}
else { // if condition is not respected
if (event[1] < -1.310636){
// This is a leaf node
result += -0.064609;
}
else { // if condition is not respected
// This is a leaf node
result += 0.052935;
}
}
}
else { // if condition is not respected
if (event[2] < -1.711236){
// This is a leaf node
result += -0.092704;
}
else { // if condition is not respected
if (event[2] < 0.218320){
// This is a leaf node
result += 0.016855;
}
else { // if condition is not respected
// This is a leaf node
result += -0.028866;
}
}
}
if (event[3] < 1.568337){
if (event[3] < 1.548263){
if (event[3] < 1.545021){
// This is a leaf node
result += 0.000038;
}
else { // if condition is not respected
// This is a leaf node
result += -0.024445;
}
}
else { // if condition is not respected
if (event[1] < 1.193377){
// This is a leaf node
result += 0.015353;
}
else { // if condition is not respected
// This is a leaf node
result += -0.031985;
}
}
}
else { // if condition is not respected
if (event[3] < 1.570177){
if (event[2] < -1.110332){
// This is a leaf node
result += 0.048721;
}
else { // if condition is not respected
// This is a leaf node
result += -0.063818;
}
}
else { // if condition is not respected
if (event[2] < 2.228405){
// This is a leaf node
result += -0.000991;
}
else { // if condition is not respected
// This is a leaf node
result += 0.024397;
}
}
}
if (event[3] < 1.058238){
if (event[3] < 1.054946){
if (event[3] < 1.045699){
// This is a leaf node
result += -0.000087;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007921;
}
}
else { // if condition is not respected
if (event[1] < 0.619096){
// This is a leaf node
result += -0.041413;
}
else { // if condition is not respected
// This is a leaf node
result += 0.024181;
}
}
}
else { // if condition is not respected
if (event[3] < 1.059328){
if (event[2] < 1.225814){
// This is a leaf node
result += 0.077422;
}
else { // if condition is not respected
// This is a leaf node
result += -0.054081;
}
}
else { // if condition is not respected
if (event[3] < 1.059491){
// This is a leaf node
result += -0.086117;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000427;
}
}
}
if (event[2] < 3.173035){
if (event[2] < 3.084193){
if (event[2] < 3.042542){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.047924;
}
}
else { // if condition is not respected
if (event[4] < 1.625332){
// This is a leaf node
result += 0.050167;
}
else { // if condition is not respected
// This is a leaf node
result += -0.079671;
}
}
}
else { // if condition is not respected
if (event[2] < 3.184392){
if (event[3] < 0.251957){
// This is a leaf node
result += -0.117359;
}
else { // if condition is not respected
// This is a leaf node
result += -0.039565;
}
}
else { // if condition is not respected
if (event[4] < -0.588552){
// This is a leaf node
result += -0.034601;
}
else { // if condition is not respected
// This is a leaf node
result += 0.011986;
}
}
}
if (event[1] < 4.088089){
if (event[1] < 4.035286){
if (event[1] < 3.906952){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.039465;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.075342;
}
}
else { // if condition is not respected
if (event[2] < 0.152884){
// This is a leaf node
result += 0.000365;
}
else { // if condition is not respected
// This is a leaf node
result += -0.057884;
}
}
if (event[2] < 4.023448){
if (event[2] < 3.986183){
if (event[2] < 3.831835){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.038576;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.074230;
}
}
else { // if condition is not respected
if (event[2] < 4.269265){
if (event[3] < -0.160747){
// This is a leaf node
result += 0.015042;
}
else { // if condition is not respected
// This is a leaf node
result += 0.080073;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.016708;
}
}
if (event[3] < -2.639947){
if (event[1] < 0.655445){
if (event[3] < -2.642425){
// This is a leaf node
result += -0.003136;
}
else { // if condition is not respected
// This is a leaf node
result += 0.101608;
}
}
else { // if condition is not respected
if (event[2] < 1.265351){
// This is a leaf node
result += 0.025028;
}
else { // if condition is not respected
// This is a leaf node
result += -0.039659;
}
}
}
else { // if condition is not respected
if (event[3] < -2.539846){
if (event[0] < 0.943438){
// This is a leaf node
result += -0.026930;
}
else { // if condition is not respected
// This is a leaf node
result += 0.029902;
}
}
else { // if condition is not respected
if (event[3] < -2.527129){
// This is a leaf node
result += 0.055530;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000001;
}
}
}
if (event[4] < -1.646589){
if (event[1] < 2.945819){
if (event[1] < -2.343797){
// This is a leaf node
result += 0.026024;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000550;
}
}
else { // if condition is not respected
if (event[0] < 1.031109){
// This is a leaf node
result += 0.084417;
}
else { // if condition is not respected
// This is a leaf node
result += -0.012745;
}
}
}
else { // if condition is not respected
if (event[4] < -1.645817){
if (event[3] < 0.511981){
// This is a leaf node
result += -0.113785;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002737;
}
}
else { // if condition is not respected
if (event[4] < -1.605698){
// This is a leaf node
result += -0.006902;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000012;
}
}
}
if (event[0] < -2.779831){
if (event[0] < -2.786980){
if (event[4] < -2.028374){
// This is a leaf node
result += 0.081490;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002977;
}
}
else { // if condition is not respected
if (event[2] < -0.226630){
// This is a leaf node
result += -0.011992;
}
else { // if condition is not respected
// This is a leaf node
result += -0.142232;
}
}
}
else { // if condition is not respected
if (event[0] < -2.708362){
if (event[0] < -2.711298){
// This is a leaf node
result += 0.018190;
}
else { // if condition is not respected
// This is a leaf node
result += 0.124904;
}
}
else { // if condition is not respected
if (event[0] < -2.703517){
// This is a leaf node
result += -0.098610;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000001;
}
}
}
if (event[3] < -4.097473){
// This is a leaf node
result += -0.049052;
}
else { // if condition is not respected
if (event[3] < -3.392746){
if (event[0] < -1.039718){
// This is a leaf node
result += -0.056386;
}
else { // if condition is not respected
// This is a leaf node
result += 0.025667;
}
}
else { // if condition is not respected
if (event[3] < -3.370402){
// This is a leaf node
result += -0.068458;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000001;
}
}
}
if (event[4] < -2.590437){
if (event[4] < -2.591750){
if (event[3] < -1.206326){
// This is a leaf node
result += 0.020363;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005423;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.112218;
}
}
else { // if condition is not respected
if (event[4] < -2.536155){
if (event[3] < -2.237476){
// This is a leaf node
result += -0.109171;
}
else { // if condition is not respected
// This is a leaf node
result += 0.026671;
}
}
else { // if condition is not respected
if (event[4] < -2.530629){
// This is a leaf node
result += -0.075521;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000006;
}
}
}
if (event[4] < 4.442656){
if (event[4] < 4.064100){
if (event[4] < 3.937248){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.077652;
}
}
else { // if condition is not respected
if (event[1] < -0.822735){
// This is a leaf node
result += -0.062315;
}
else { // if condition is not respected
// This is a leaf node
result += 0.120247;
}
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.048681;
}
if (event[3] < -3.863437){
if (event[2] < -0.379601){
// This is a leaf node
result += 0.043224;
}
else { // if condition is not respected
if (event[0] < -0.002408){
// This is a leaf node
result += -0.000514;
}
else { // if condition is not respected
// This is a leaf node
result += -0.077091;
}
}
}
else { // if condition is not respected
if (event[3] < -3.672421){
if (event[4] < 0.478157){
// This is a leaf node
result += 0.067829;
}
else { // if condition is not respected
// This is a leaf node
result += -0.038984;
}
}
else { // if condition is not respected
if (event[3] < -2.946279){
// This is a leaf node
result += -0.005847;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000007;
}
}
}
if (event[3] < 2.318792){
if (event[3] < 2.309603){
if (event[3] < 2.308932){
// This is a leaf node
result += 0.000012;
}
else { // if condition is not respected
// This is a leaf node
result += -0.121427;
}
}
else { // if condition is not respected
if (event[1] < 0.145666){
// This is a leaf node
result += 0.084131;
}
else { // if condition is not respected
// This is a leaf node
result += -0.006392;
}
}
}
else { // if condition is not respected
if (event[3] < 2.325979){
if (event[1] < -1.214269){
// This is a leaf node
result += 0.091875;
}
else { // if condition is not respected
// This is a leaf node
result += -0.072675;
}
}
else { // if condition is not respected
if (event[0] < -0.361989){
// This is a leaf node
result += -0.009300;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003747;
}
}
}
if (event[0] < 1.992759){
if (event[0] < 1.989172){
if (event[0] < 1.904449){
// This is a leaf node
result += -0.000019;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007365;
}
}
else { // if condition is not respected
if (event[3] < 0.839810){
// This is a leaf node
result += 0.034161;
}
else { // if condition is not respected
// This is a leaf node
result += 0.133128;
}
}
}
else { // if condition is not respected
if (event[0] < 2.008021){
if (event[4] < 0.124527){
// This is a leaf node
result += -0.010918;
}
else { // if condition is not respected
// This is a leaf node
result += -0.054105;
}
}
else { // if condition is not respected
if (event[0] < 2.014476){
// This is a leaf node
result += 0.049401;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001109;
}
}
}
if (event[2] < -2.304283){
if (event[1] < -0.353279){
if (event[2] < -2.329827){
// This is a leaf node
result += -0.006323;
}
else { // if condition is not respected
// This is a leaf node
result += -0.053764;
}
}
else { // if condition is not respected
if (event[1] < -0.117763){
// This is a leaf node
result += 0.029182;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002209;
}
}
}
else { // if condition is not respected
if (event[2] < -2.302579){
if (event[1] < -1.043266){
// This is a leaf node
result += 0.015437;
}
else { // if condition is not respected
// This is a leaf node
result += 0.105721;
}
}
else { // if condition is not respected
if (event[2] < -2.301145){
// This is a leaf node
result += -0.065284;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000020;
}
}
}
if (event[1] < -0.608965){
if (event[1] < -0.609987){
if (event[4] < -2.659788){
// This is a leaf node
result += 0.017781;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000225;
}
}
else { // if condition is not respected
if (event[2] < 0.354084){
// This is a leaf node
result += 0.010500;
}
else { // if condition is not respected
// This is a leaf node
result += 0.080998;
}
}
}
else { // if condition is not respected
if (event[1] < -0.607856){
if (event[1] < -0.607927){
// This is a leaf node
result += -0.023066;
}
else { // if condition is not respected
// This is a leaf node
result += -0.145444;
}
}
else { // if condition is not respected
if (event[1] < -0.607732){
// This is a leaf node
result += 0.074448;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000114;
}
}
}
if (event[4] < -2.590437){
if (event[2] < 1.734092){
if (event[4] < -2.591750){
// This is a leaf node
result += -0.000885;
}
else { // if condition is not respected
// This is a leaf node
result += -0.104576;
}
}
else { // if condition is not respected
if (event[3] < 1.568288){
// This is a leaf node
result += -0.059049;
}
else { // if condition is not respected
// This is a leaf node
result += 0.104794;
}
}
}
else { // if condition is not respected
if (event[4] < -2.536155){
if (event[3] < -2.237476){
// This is a leaf node
result += -0.101780;
}
else { // if condition is not respected
// This is a leaf node
result += 0.024056;
}
}
else { // if condition is not respected
if (event[4] < -2.530629){
// This is a leaf node
result += -0.068593;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000007;
}
}
}
if (event[4] < -2.212085){
if (event[4] < -2.213194){
if (event[0] < -2.835312){
// This is a leaf node
result += 0.098974;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001740;
}
}
else { // if condition is not respected
if (event[3] < -0.306340){
// This is a leaf node
result += -0.019449;
}
else { // if condition is not respected
// This is a leaf node
result += -0.120514;
}
}
}
else { // if condition is not respected
if (event[4] < -2.211330){
if (event[4] < -2.211732){
// This is a leaf node
result += 0.115918;
}
else { // if condition is not respected
// This is a leaf node
result += 0.041701;
}
}
else { // if condition is not respected
if (event[4] < -2.210537){
// This is a leaf node
result += -0.076782;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000023;
}
}
}
if (event[1] < 0.247780){
if (event[1] < 0.208027){
if (event[1] < 0.207849){
// This is a leaf node
result += -0.000023;
}
else { // if condition is not respected
// This is a leaf node
result += 0.060843;
}
}
else { // if condition is not respected
if (event[0] < 2.101555){
// This is a leaf node
result += -0.006734;
}
else { // if condition is not respected
// This is a leaf node
result += 0.045756;
}
}
}
else { // if condition is not respected
if (event[1] < 0.259408){
if (event[1] < 0.259179){
// This is a leaf node
result += 0.007320;
}
else { // if condition is not respected
// This is a leaf node
result += 0.106998;
}
}
else { // if condition is not respected
if (event[1] < 0.259607){
// This is a leaf node
result += -0.066669;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000162;
}
}
}
if (event[1] < 1.398814){
if (event[1] < 1.393134){
if (event[1] < 1.388847){
// This is a leaf node
result += 0.000054;
}
else { // if condition is not respected
// This is a leaf node
result += -0.021465;
}
}
else { // if condition is not respected
if (event[2] < -1.110939){
// This is a leaf node
result += -0.044324;
}
else { // if condition is not respected
// This is a leaf node
result += 0.035880;
}
}
}
else { // if condition is not respected
if (event[1] < 1.399748){
if (event[2] < 0.431943){
// This is a leaf node
result += -0.078436;
}
else { // if condition is not respected
// This is a leaf node
result += 0.008588;
}
}
else { // if condition is not respected
if (event[0] < -2.511831){
// This is a leaf node
result += -0.024101;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000444;
}
}
}
if (event[0] < -2.627497){
if (event[0] < -2.632281){
if (event[4] < 1.413152){
// This is a leaf node
result += -0.000960;
}
else { // if condition is not respected
// This is a leaf node
result += 0.033003;
}
}
else { // if condition is not respected
if (event[4] < 0.992681){
// This is a leaf node
result += 0.133289;
}
else { // if condition is not respected
// This is a leaf node
result += -0.041358;
}
}
}
else { // if condition is not respected
if (event[0] < -2.508932){
if (event[0] < -2.517264){
// This is a leaf node
result += -0.006625;
}
else { // if condition is not respected
// This is a leaf node
result += -0.061455;
}
}
else { // if condition is not respected
if (event[0] < -2.500730){
// This is a leaf node
result += 0.042127;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000002;
}
}
}
if (event[1] < -2.522265){
if (event[1] < -2.568782){
if (event[3] < -2.417477){
// This is a leaf node
result += -0.086868;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002929;
}
}
else { // if condition is not respected
if (event[0] < 0.890862){
// This is a leaf node
result += -0.044828;
}
else { // if condition is not respected
// This is a leaf node
result += 0.024581;
}
}
}
else { // if condition is not respected
if (event[1] < -2.517189){
if (event[4] < -0.520381){
// This is a leaf node
result += -0.038251;
}
else { // if condition is not respected
// This is a leaf node
result += 0.090569;
}
}
else { // if condition is not respected
if (event[1] < -2.511364){
// This is a leaf node
result += -0.042217;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000014;
}
}
}
if (event[1] < -2.983040){
if (event[3] < -0.236571){
if (event[2] < 1.040672){
// This is a leaf node
result += 0.016888;
}
else { // if condition is not respected
// This is a leaf node
result += -0.036765;
}
}
else { // if condition is not respected
if (event[2] < -0.348421){
// This is a leaf node
result += -0.039696;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001439;
}
}
}
else { // if condition is not respected
if (event[1] < -2.973975){
// This is a leaf node
result += 0.097368;
}
else { // if condition is not respected
if (event[1] < -2.971200){
// This is a leaf node
result += -0.072979;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000007;
}
}
}
if (event[0] < -1.842545){
if (event[0] < -1.850449){
if (event[1] < -0.525531){
// This is a leaf node
result += -0.004311;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002615;
}
}
else { // if condition is not respected
if (event[3] < 0.650335){
// This is a leaf node
result += 0.052546;
}
else { // if condition is not respected
// This is a leaf node
result += -0.022386;
}
}
}
else { // if condition is not respected
if (event[0] < -1.816692){
if (event[0] < -1.818077){
// This is a leaf node
result += -0.012290;
}
else { // if condition is not respected
// This is a leaf node
result += -0.083195;
}
}
else { // if condition is not respected
if (event[0] < -1.722601){
// This is a leaf node
result += -0.004717;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000034;
}
}
}
if (event[0] < -1.228112){
if (event[0] < -1.228436){
if (event[4] < 2.285427){
// This is a leaf node
result += 0.000378;
}
else { // if condition is not respected
// This is a leaf node
result += 0.016168;
}
}
else { // if condition is not respected
if (event[1] < -0.143620){
// This is a leaf node
result += -0.015357;
}
else { // if condition is not respected
// This is a leaf node
result += 0.152711;
}
}
}
else { // if condition is not respected
if (event[0] < -1.227979){
if (event[1] < -0.881966){
// This is a leaf node
result += 0.001882;
}
else { // if condition is not respected
// This is a leaf node
result += -0.148006;
}
}
else { // if condition is not respected
if (event[0] < -1.186120){
// This is a leaf node
result += -0.005607;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000019;
}
}
}
if (event[4] < 2.624323){
if (event[4] < 2.618027){
if (event[4] < 2.608845){
// This is a leaf node
result += 0.000013;
}
else { // if condition is not respected
// This is a leaf node
result += -0.025826;
}
}
else { // if condition is not respected
if (event[4] < 2.619708){
// This is a leaf node
result += 0.130835;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009574;
}
}
}
else { // if condition is not respected
if (event[4] < 2.655832){
if (event[2] < -0.210578){
// This is a leaf node
result += -0.088793;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001287;
}
}
else { // if condition is not respected
if (event[2] < -0.925358){
// This is a leaf node
result += 0.021283;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004936;
}
}
}
if (event[2] < 3.244201){
if (event[2] < 2.972512){
if (event[2] < 2.952371){
// This is a leaf node
result += -0.000004;
}
else { // if condition is not respected
// This is a leaf node
result += -0.040562;
}
}
else { // if condition is not respected
if (event[0] < -1.891978){
// This is a leaf node
result += -0.096964;
}
else { // if condition is not respected
// This is a leaf node
result += 0.019500;
}
}
}
else { // if condition is not respected
if (event[0] < -1.312025){
if (event[4] < 0.353465){
// This is a leaf node
result += 0.004787;
}
else { // if condition is not respected
// This is a leaf node
result += 0.113619;
}
}
else { // if condition is not respected
if (event[0] < -1.150069){
// This is a leaf node
result += -0.129202;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010183;
}
}
}
if (event[1] < -2.316840){
if (event[3] < -0.780421){
if (event[0] < -0.873094){
// This is a leaf node
result += -0.044584;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001559;
}
}
else { // if condition is not respected
if (event[3] < -0.729157){
// This is a leaf node
result += 0.060069;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003932;
}
}
}
else { // if condition is not respected
if (event[1] < -2.316143){
// This is a leaf node
result += -0.100153;
}
else { // if condition is not respected
if (event[1] < -2.184444){
// This is a leaf node
result += -0.006965;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000012;
}
}
}
if (event[3] < -1.054364){
if (event[3] < -1.055890){
if (event[0] < 1.282994){
// This is a leaf node
result += 0.000961;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004704;
}
}
else { // if condition is not respected
if (event[2] < -1.195540){
// This is a leaf node
result += -0.024729;
}
else { // if condition is not respected
// This is a leaf node
result += 0.048701;
}
}
}
else { // if condition is not respected
if (event[3] < -0.984261){
if (event[3] < -0.995772){
// This is a leaf node
result += -0.001124;
}
else { // if condition is not respected
// This is a leaf node
result += -0.018410;
}
}
else { // if condition is not respected
if (event[3] < -0.975094){
// This is a leaf node
result += 0.016884;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000048;
}
}
}
if (event[1] < -2.749831){
if (event[2] < -2.083972){
if (event[4] < -0.394332){
// This is a leaf node
result += -0.003959;
}
else { // if condition is not respected
// This is a leaf node
result += 0.091171;
}
}
else { // if condition is not respected
if (event[2] < -1.953131){
// This is a leaf node
result += -0.127680;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004245;
}
}
}
else { // if condition is not respected
if (event[1] < -2.701213){
if (event[2] < -1.278660){
// This is a leaf node
result += -0.031068;
}
else { // if condition is not respected
// This is a leaf node
result += 0.041609;
}
}
else { // if condition is not respected
if (event[1] < -2.698502){
// This is a leaf node
result += -0.143769;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000001;
}
}
}
if (event[3] < 2.152134){
if (event[3] < 2.139556){
if (event[3] < 2.138307){
// This is a leaf node
result += -0.000011;
}
else { // if condition is not respected
// This is a leaf node
result += 0.053848;
}
}
else { // if condition is not respected
if (event[4] < 0.321886){
// This is a leaf node
result += -0.055068;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002003;
}
}
}
else { // if condition is not respected
if (event[3] < 2.165845){
if (event[2] < 1.191007){
// This is a leaf node
result += 0.045888;
}
else { // if condition is not respected
// This is a leaf node
result += -0.057962;
}
}
else { // if condition is not respected
if (event[1] < -0.908206){
// This is a leaf node
result += 0.011463;
}
else { // if condition is not respected
// This is a leaf node
result += -0.002017;
}
}
}
if (event[4] < 3.002024){
if (event[4] < 2.939700){
if (event[4] < 2.905666){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.038203;
}
}
else { // if condition is not respected
if (event[0] < -0.649538){
// This is a leaf node
result += 0.085654;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020244;
}
}
}
else { // if condition is not respected
if (event[4] < 3.005722){
if (event[3] < 0.765738){
// This is a leaf node
result += -0.037529;
}
else { // if condition is not respected
// This is a leaf node
result += -0.121908;
}
}
else { // if condition is not respected
if (event[1] < 0.997171){
// This is a leaf node
result += -0.010088;
}
else { // if condition is not respected
// This is a leaf node
result += 0.028592;
}
}
}
if (event[0] < -2.903976){
if (event[1] < 2.130710){
if (event[1] < 0.406430){
// This is a leaf node
result += -0.011860;
}
else { // if condition is not respected
// This is a leaf node
result += 0.015833;
}
}
else { // if condition is not respected
if (event[0] < -3.215514){
// This is a leaf node
result += 0.014269;
}
else { // if condition is not respected
// This is a leaf node
result += -0.143117;
}
}
}
else { // if condition is not respected
if (event[0] < -2.877849){
if (event[3] < -0.545673){
// This is a leaf node
result += -0.011167;
}
else { // if condition is not respected
// This is a leaf node
result += 0.089926;
}
}
else { // if condition is not respected
if (event[0] < -2.869082){
// This is a leaf node
result += -0.054022;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000001;
}
}
}
if (event[2] < 4.023448){
if (event[2] < 3.709840){
if (event[2] < 3.457538){
// This is a leaf node
result += -0.000004;
}
else { // if condition is not respected
// This is a leaf node
result += 0.027504;
}
}
else { // if condition is not respected
if (event[4] < 1.110846){
// This is a leaf node
result += -0.059599;
}
else { // if condition is not respected
// This is a leaf node
result += 0.088541;
}
}
}
else { // if condition is not respected
if (event[4] < -0.005265){
// This is a leaf node
result += -0.008923;
}
else { // if condition is not respected
// This is a leaf node
result += 0.064839;
}
}
if (event[0] < 3.125710){
if (event[0] < 3.008761){
if (event[0] < 3.001923){
// This is a leaf node
result += -0.000000;
}
else { // if condition is not respected
// This is a leaf node
result += 0.075302;
}
}
else { // if condition is not respected
if (event[2] < -0.838161){
// This is a leaf node
result += -0.080153;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004018;
}
}
}
else { // if condition is not respected
if (event[0] < 3.224831){
if (event[2] < -1.577493){
// This is a leaf node
result += -0.097253;
}
else { // if condition is not respected
// This is a leaf node
result += 0.042853;
}
}
else { // if condition is not respected
if (event[1] < -0.926566){
// This is a leaf node
result += 0.048732;
}
else { // if condition is not respected
// This is a leaf node
result += -0.015154;
}
}
}
if (event[0] < 4.010478){
if (event[0] < 3.689817){
if (event[0] < 3.651816){
// This is a leaf node
result += -0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += 0.087625;
}
}
else { // if condition is not respected
if (event[4] < 0.450543){
// This is a leaf node
result += -0.064694;
}
else { // if condition is not respected
// This is a leaf node
result += 0.021050;
}
}
}
else { // if condition is not respected
if (event[3] < 0.132275){
if (event[3] < -0.557640){
// This is a leaf node
result += 0.035848;
}
else { // if condition is not respected
// This is a leaf node
result += -0.046605;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.071148;
}
}
if (event[3] < 2.615678){
if (event[3] < 2.613247){
if (event[3] < 2.544425){
// This is a leaf node
result += -0.000006;
}
else { // if condition is not respected
// This is a leaf node
result += 0.016087;
}
}
else { // if condition is not respected
if (event[0] < 0.502435){
// This is a leaf node
result += 0.038160;
}
else { // if condition is not respected
// This is a leaf node
result += 0.132149;
}
}
}
else { // if condition is not respected
if (event[3] < 2.695910){
if (event[0] < 0.599213){
// This is a leaf node
result += -0.010677;
}
else { // if condition is not respected
// This is a leaf node
result += -0.049877;
}
}
else { // if condition is not respected
if (event[3] < 2.727612){
// This is a leaf node
result += 0.048910;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003070;
}
}
}
if (event[3] < 2.355842){
if (event[3] < 2.352245){
if (event[3] < 2.351485){
// This is a leaf node
result += 0.000016;
}
else { // if condition is not respected
// This is a leaf node
result += -0.114647;
}
}
else { // if condition is not respected
if (event[1] < 0.477204){
// This is a leaf node
result += 0.097368;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004753;
}
}
}
else { // if condition is not respected
if (event[3] < 2.458675){
if (event[1] < 1.659469){
// This is a leaf node
result += -0.008845;
}
else { // if condition is not respected
// This is a leaf node
result += -0.079488;
}
}
else { // if condition is not respected
if (event[3] < 2.468025){
// This is a leaf node
result += 0.051145;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000201;
}
}
}
if (event[0] < -3.520390){
if (event[2] < 0.842266){
if (event[1] < -1.572505){
// This is a leaf node
result += -0.067806;
}
else { // if condition is not respected
// This is a leaf node
result += 0.050690;
}
}
else { // if condition is not respected
if (event[4] < -0.507941){
// This is a leaf node
result += 0.036933;
}
else { // if condition is not respected
// This is a leaf node
result += -0.104753;
}
}
}
else { // if condition is not respected
if (event[0] < -3.476108){
if (event[2] < 0.451203){
// This is a leaf node
result += -0.116760;
}
else { // if condition is not respected
// This is a leaf node
result += -0.019284;
}
}
else { // if condition is not respected
if (event[0] < -3.342121){
// This is a leaf node
result += -0.017691;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000004;
}
}
}
if (event[0] < -0.983987){
if (event[0] < -0.984593){
if (event[2] < -0.037897){
// This is a leaf node
result += -0.001091;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001780;
}
}
else { // if condition is not respected
if (event[1] < 0.281098){
// This is a leaf node
result += 0.036598;
}
else { // if condition is not respected
// This is a leaf node
result += 0.120486;
}
}
}
else { // if condition is not respected
if (event[0] < -0.853287){
if (event[4] < -0.384105){
// This is a leaf node
result += -0.008770;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000429;
}
}
else { // if condition is not respected
if (event[0] < -0.847749){
// This is a leaf node
result += 0.024368;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000004;
}
}
}
if (event[2] < -0.806206){
if (event[2] < -0.806425){
if (event[2] < -0.806557){
// This is a leaf node
result += 0.000391;
}
else { // if condition is not respected
// This is a leaf node
result += -0.087849;
}
}
else { // if condition is not respected
if (event[3] < 1.016702){
// This is a leaf node
result += 0.149663;
}
else { // if condition is not respected
// This is a leaf node
result += -0.022539;
}
}
}
else { // if condition is not respected
if (event[2] < -0.715613){
if (event[1] < -1.998067){
// This is a leaf node
result += -0.033582;
}
else { // if condition is not respected
// This is a leaf node
result += -0.003001;
}
}
else { // if condition is not respected
if (event[2] < -0.714386){
// This is a leaf node
result += 0.021461;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000010;
}
}
}
if (event[4] < -0.879296){
if (event[4] < -0.879747){
if (event[0] < 0.983173){
// This is a leaf node
result += -0.000193;
}
else { // if condition is not respected
// This is a leaf node
result += 0.003239;
}
}
else { // if condition is not respected
if (event[4] < -0.879587){
// This is a leaf node
result += 0.116559;
}
else { // if condition is not respected
// This is a leaf node
result += 0.043718;
}
}
}
else { // if condition is not respected
if (event[4] < -0.878830){
if (event[0] < 1.081478){
// This is a leaf node
result += -0.083563;
}
else { // if condition is not respected
// This is a leaf node
result += 0.029097;
}
}
else { // if condition is not respected
if (event[4] < -0.835456){
// This is a leaf node
result += -0.004327;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000022;
}
}
}
if (event[4] < 0.542113){
if (event[4] < 0.536049){
if (event[4] < 0.536010){
// This is a leaf node
result += 0.000098;
}
else { // if condition is not respected
// This is a leaf node
result += -0.102204;
}
}
else { // if condition is not respected
if (event[0] < -1.421722){
// This is a leaf node
result += -0.027388;
}
else { // if condition is not respected
// This is a leaf node
result += 0.016158;
}
}
}
else { // if condition is not respected
if (event[4] < 0.542791){
if (event[0] < 0.594086){
// This is a leaf node
result += -0.070398;
}
else { // if condition is not respected
// This is a leaf node
result += 0.043706;
}
}
else { // if condition is not respected
if (event[0] < -1.534526){
// This is a leaf node
result += -0.003803;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000054;
}
}
}
if (event[0] < -1.293801){
if (event[0] < -1.326237){
if (event[0] < -1.326787){
// This is a leaf node
result += 0.000170;
}
else { // if condition is not respected
// This is a leaf node
result += -0.060144;
}
}
else { // if condition is not respected
if (event[4] < 1.565366){
// This is a leaf node
result += 0.006796;
}
else { // if condition is not respected
// This is a leaf node
result += 0.043098;
}
}
}
else { // if condition is not respected
if (event[0] < -1.293642){
if (event[2] < 0.371141){
// This is a leaf node
result += -0.101864;
}
else { // if condition is not respected
// This is a leaf node
result += -0.021392;
}
}
else { // if condition is not respected
if (event[0] < -1.285103){
// This is a leaf node
result += -0.009974;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000048;
}
}
}
if (event[2] < 1.720242){
if (event[2] < 1.706046){
if (event[2] < 1.704433){
// This is a leaf node
result += 0.000022;
}
else { // if condition is not respected
// This is a leaf node
result += -0.039866;
}
}
else { // if condition is not respected
if (event[3] < -1.001229){
// This is a leaf node
result += 0.069436;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009577;
}
}
}
else { // if condition is not respected
if (event[2] < 1.723153){
if (event[0] < 0.623329){
// This is a leaf node
result += -0.064977;
}
else { // if condition is not respected
// This is a leaf node
result += 0.027025;
}
}
else { // if condition is not respected
if (event[0] < -2.296081){
// This is a leaf node
result += 0.027719;
}
else { // if condition is not respected
// This is a leaf node
result += -0.001022;
}
}
}
if (event[0] < 0.585868){
if (event[0] < 0.585360){
if (event[0] < 0.426014){
// This is a leaf node
result += -0.000101;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002584;
}
}
else { // if condition is not respected
if (event[4] < -1.577558){
// This is a leaf node
result += -0.078546;
}
else { // if condition is not respected
// This is a leaf node
result += 0.082578;
}
}
}
else { // if condition is not respected
if (event[0] < 0.586359){
if (event[4] < -0.547670){
// This is a leaf node
result += 0.012603;
}
else { // if condition is not respected
// This is a leaf node
result += -0.091046;
}
}
else { // if condition is not respected
if (event[0] < 0.586640){
// This is a leaf node
result += 0.063791;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000308;
}
}
}
if (event[0] < 0.281160){
if (event[0] < 0.221377){
if (event[0] < 0.219587){
// This is a leaf node
result += 0.000034;
}
else { // if condition is not respected
// This is a leaf node
result += -0.024591;
}
}
else { // if condition is not respected
if (event[2] < 3.099449){
// This is a leaf node
result += 0.004439;
}
else { // if condition is not respected
// This is a leaf node
result += -0.135068;
}
}
}
else { // if condition is not respected
if (event[0] < 0.293038){
if (event[0] < 0.292921){
// This is a leaf node
result += -0.010174;
}
else { // if condition is not respected
// This is a leaf node
result += -0.087460;
}
}
else { // if condition is not respected
if (event[1] < 1.738513){
// This is a leaf node
result += -0.000333;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004672;
}
}
}
if (event[1] < 1.986526){
if (event[1] < 1.971039){
if (event[1] < 1.970434){
// This is a leaf node
result += 0.000021;
}
else { // if condition is not respected
// This is a leaf node
result += -0.100323;
}
}
else { // if condition is not respected
if (event[2] < 1.435596){
// This is a leaf node
result += 0.025576;
}
else { // if condition is not respected
// This is a leaf node
result += -0.059065;
}
}
}
else { // if condition is not respected
if (event[1] < 1.991189){
if (event[2] < 0.306759){
// This is a leaf node
result += -0.066137;
}
else { // if condition is not respected
// This is a leaf node
result += 0.016264;
}
}
else { // if condition is not respected
if (event[3] < 1.794408){
// This is a leaf node
result += -0.001791;
}
else { // if condition is not respected
// This is a leaf node
result += 0.018691;
}
}
}
if (event[1] < 1.912272){
if (event[1] < 1.905800){
if (event[1] < 1.904039){
// This is a leaf node
result += 0.000027;
}
else { // if condition is not respected
// This is a leaf node
result += -0.039897;
}
}
else { // if condition is not respected
if (event[1] < 1.906795){
// This is a leaf node
result += 0.097164;
}
else { // if condition is not respected
// This is a leaf node
result += 0.016297;
}
}
}
else { // if condition is not respected
if (event[1] < 1.921377){
if (event[0] < 0.926359){
// This is a leaf node
result += -0.042752;
}
else { // if condition is not respected
// This is a leaf node
result += 0.051879;
}
}
else { // if condition is not respected
if (event[0] < -2.955364){
// This is a leaf node
result += -0.077969;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000445;
}
}
}
if (event[1] < -0.618571){
if (event[4] < -2.659788){
if (event[4] < -2.690200){
// This is a leaf node
result += 0.008789;
}
else { // if condition is not respected
// This is a leaf node
result += 0.092504;
}
}
else { // if condition is not respected
if (event[4] < -2.576034){
// This is a leaf node
result += -0.046269;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000314;
}
}
}
else { // if condition is not respected
if (event[1] < -0.617647){
if (event[0] < 1.032995){
// This is a leaf node
result += -0.044881;
}
else { // if condition is not respected
// This is a leaf node
result += 0.050392;
}
}
else { // if condition is not respected
if (event[1] < -0.617419){
// This is a leaf node
result += 0.050801;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000116;
}
}
}
if (event[4] < -2.781338){
if (event[2] < -1.924009){
if (event[4] < -3.328556){
// This is a leaf node
result += 0.023473;
}
else { // if condition is not respected
// This is a leaf node
result += -0.079177;
}
}
else { // if condition is not respected
if (event[2] < -0.115107){
// This is a leaf node
result += 0.010423;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011519;
}
}
}
else { // if condition is not respected
if (event[4] < -2.757866){
if (event[4] < -2.761207){
// This is a leaf node
result += 0.023274;
}
else { // if condition is not respected
// This is a leaf node
result += 0.127452;
}
}
else { // if condition is not respected
if (event[4] < -2.748747){
// This is a leaf node
result += -0.055090;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000007;
}
}
}
if (event[4] < 1.226623){
if (event[4] < 1.226062){
if (event[4] < 1.200546){
// This is a leaf node
result += -0.000019;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007897;
}
}
else { // if condition is not respected
if (event[2] < 0.694371){
// This is a leaf node
result += -0.118000;
}
else { // if condition is not respected
// This is a leaf node
result += 0.024249;
}
}
}
else { // if condition is not respected
if (event[4] < 1.275297){
if (event[4] < 1.275084){
// This is a leaf node
result += 0.007999;
}
else { // if condition is not respected
// This is a leaf node
result += 0.137599;
}
}
else { // if condition is not respected
if (event[4] < 1.314252){
// This is a leaf node
result += -0.009161;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000503;
}
}
}
if (event[3] < 1.848377){
if (event[3] < 1.847842){
if (event[3] < 1.785029){
// This is a leaf node
result += -0.000012;
}
else { // if condition is not respected
// This is a leaf node
result += 0.008782;
}
}
else { // if condition is not respected
if (event[4] < 0.104165){
// This is a leaf node
result += 0.133822;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001269;
}
}
}
else { // if condition is not respected
if (event[2] < 1.584886){
if (event[2] < 1.352938){
// This is a leaf node
result += -0.000989;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020062;
}
}
else { // if condition is not respected
if (event[2] < 2.256624){
// This is a leaf node
result += -0.024135;
}
else { // if condition is not respected
// This is a leaf node
result += 0.021826;
}
}
}
if (event[3] < 1.568277){
if (event[3] < 1.427665){
if (event[3] < 1.427438){
// This is a leaf node
result += -0.000010;
}
else { // if condition is not respected
// This is a leaf node
result += -0.094765;
}
}
else { // if condition is not respected
if (event[3] < 1.440740){
// This is a leaf node
result += 0.018574;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001392;
}
}
}
else { // if condition is not respected
if (event[2] < -1.622473){
if (event[1] < -1.066064){
// This is a leaf node
result += 0.042463;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005084;
}
}
else { // if condition is not respected
if (event[2] < -1.359300){
// This is a leaf node
result += -0.017757;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000822;
}
}
}
if (event[3] < 2.150650){
if (event[3] < 2.139556){
if (event[3] < 2.138307){
// This is a leaf node
result += -0.000013;
}
else { // if condition is not respected
// This is a leaf node
result += 0.049749;
}
}
else { // if condition is not respected
if (event[4] < -0.089325){
// This is a leaf node
result += -0.067924;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005637;
}
}
}
else { // if condition is not respected
if (event[3] < 2.165845){
if (event[2] < 1.139343){
// This is a leaf node
result += 0.039016;
}
else { // if condition is not respected
// This is a leaf node
result += -0.040079;
}
}
else { // if condition is not respected
if (event[3] < 2.167850){
// This is a leaf node
result += -0.064711;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000978;
}
}
}
if (event[0] < -3.769990){
if (event[1] < 0.089967){
if (event[3] < 0.341122){
// This is a leaf node
result += -0.119780;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007571;
}
}
else { // if condition is not respected
if (event[3] < -0.620902){
// This is a leaf node
result += 0.089885;
}
else { // if condition is not respected
// This is a leaf node
result += -0.023479;
}
}
}
else { // if condition is not respected
if (event[0] < -3.644391){
if (event[2] < 0.775102){
// This is a leaf node
result += 0.100059;
}
else { // if condition is not respected
// This is a leaf node
result += -0.054898;
}
}
else { // if condition is not respected
if (event[0] < -3.603663){
// This is a leaf node
result += -0.059772;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000000;
}
}
}
if (event[4] < 3.594640){
if (event[4] < 3.579999){
if (event[4] < 3.565615){
// This is a leaf node
result += 0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += -0.068286;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.075968;
}
}
else { // if condition is not respected
if (event[4] < 3.647938){
if (event[1] < 0.597298){
// This is a leaf node
result += -0.104016;
}
else { // if condition is not respected
// This is a leaf node
result += -0.026321;
}
}
else { // if condition is not respected
if (event[0] < -0.565315){
// This is a leaf node
result += -0.040375;
}
else { // if condition is not respected
// This is a leaf node
result += 0.026214;
}
}
}
if (event[4] < -1.142751){
if (event[4] < -1.159955){
if (event[4] < -1.160022){
// This is a leaf node
result += -0.000241;
}
else { // if condition is not respected
// This is a leaf node
result += 0.125185;
}
}
else { // if condition is not respected
if (event[3] < -1.735446){
// This is a leaf node
result += -0.094544;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007812;
}
}
}
else { // if condition is not respected
if (event[4] < -1.117700){
if (event[4] < -1.119072){
// This is a leaf node
result += 0.006059;
}
else { // if condition is not respected
// This is a leaf node
result += 0.061766;
}
}
else { // if condition is not respected
if (event[4] < -1.116429){
// This is a leaf node
result += -0.031914;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000032;
}
}
}
if (event[2] < -1.512943){
if (event[2] < -1.513505){
if (event[3] < 1.380988){
// This is a leaf node
result += -0.001293;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006664;
}
}
else { // if condition is not respected
if (event[3] < -0.186283){
// This is a leaf node
result += -0.138503;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007878;
}
}
}
else { // if condition is not respected
if (event[2] < -1.511129){
if (event[1] < -0.758913){
// This is a leaf node
result += -0.054702;
}
else { // if condition is not respected
// This is a leaf node
result += 0.083521;
}
}
else { // if condition is not respected
if (event[2] < -1.508584){
// This is a leaf node
result += -0.033133;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000047;
}
}
}
if (event[3] < 1.568277){
if (event[3] < 1.563270){
if (event[3] < 1.562757){
// This is a leaf node
result += 0.000043;
}
else { // if condition is not respected
// This is a leaf node
result += -0.074713;
}
}
else { // if condition is not respected
if (event[3] < 1.563513){
// This is a leaf node
result += 0.117484;
}
else { // if condition is not respected
// This is a leaf node
result += 0.010408;
}
}
}
else { // if condition is not respected
if (event[3] < 1.648609){
if (event[2] < -2.690231){
// This is a leaf node
result += 0.095163;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007180;
}
}
else { // if condition is not respected
if (event[4] < -1.116846){
// This is a leaf node
result += -0.007125;
}
else { // if condition is not respected
// This is a leaf node
result += 0.001338;
}
}
}
if (event[4] < -1.882806){
if (event[1] < -1.587891){
if (event[0] < 1.546712){
// This is a leaf node
result += 0.013088;
}
else { // if condition is not respected
// This is a leaf node
result += 0.079345;
}
}
else { // if condition is not respected
if (event[0] < 1.842145){
// This is a leaf node
result += -0.000475;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020987;
}
}
}
else { // if condition is not respected
if (event[4] < -1.881837){
if (event[1] < 0.002105){
// This is a leaf node
result += -0.013238;
}
else { // if condition is not respected
// This is a leaf node
result += -0.130644;
}
}
else { // if condition is not respected
if (event[4] < -1.788130){
// This is a leaf node
result += -0.004894;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000005;
}
}
}
if (event[4] < -1.646589){
if (event[4] < -1.648610){
if (event[4] < -1.649469){
// This is a leaf node
result += 0.000832;
}
else { // if condition is not respected
// This is a leaf node
result += -0.052620;
}
}
else { // if condition is not respected
if (event[0] < 0.424103){
// This is a leaf node
result += 0.065049;
}
else { // if condition is not respected
// This is a leaf node
result += -0.030613;
}
}
}
else { // if condition is not respected
if (event[4] < -1.645817){
if (event[3] < 0.511981){
// This is a leaf node
result += -0.107560;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002681;
}
}
else { // if condition is not respected
if (event[4] < -1.638354){
// This is a leaf node
result += -0.015064;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000030;
}
}
}
if (event[4] < -2.212085){
if (event[2] < 1.546558){
if (event[4] < -2.373860){
// This is a leaf node
result += 0.001188;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010999;
}
}
else { // if condition is not respected
if (event[2] < 1.624218){
// This is a leaf node
result += 0.077139;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005539;
}
}
}
else { // if condition is not respected
if (event[4] < -2.210980){
if (event[0] < -0.657381){
// This is a leaf node
result += -0.027588;
}
else { // if condition is not respected
// This is a leaf node
result += 0.114233;
}
}
else { // if condition is not respected
if (event[4] < -2.210537){
// This is a leaf node
result += -0.112273;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000022;
}
}
}
if (event[4] < -2.590437){
if (event[3] < 1.638091){
if (event[2] < 1.734092){
// This is a leaf node
result += -0.002997;
}
else { // if condition is not respected
// This is a leaf node
result += -0.053816;
}
}
else { // if condition is not respected
if (event[2] < -0.875781){
// This is a leaf node
result += -0.034988;
}
else { // if condition is not respected
// This is a leaf node
result += 0.045584;
}
}
}
else { // if condition is not respected
if (event[4] < -2.587945){
if (event[3] < 0.901125){
// This is a leaf node
result += 0.105001;
}
else { // if condition is not respected
// This is a leaf node
result += 0.016137;
}
}
else { // if condition is not respected
if (event[4] < -2.586124){
// This is a leaf node
result += -0.106165;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000013;
}
}
}
if (event[4] < -0.888006){
if (event[4] < -0.888746){
if (event[4] < -0.888967){
// This is a leaf node
result += 0.000409;
}
else { // if condition is not respected
// This is a leaf node
result += -0.083486;
}
}
else { // if condition is not respected
if (event[1] < -1.270059){
// This is a leaf node
result += -0.084677;
}
else { // if condition is not respected
// This is a leaf node
result += 0.060577;
}
}
}
else { // if condition is not respected
if (event[4] < -0.887942){
if (event[2] < 0.270091){
// This is a leaf node
result += -0.119241;
}
else { // if condition is not respected
// This is a leaf node
result += -0.020532;
}
}
else { // if condition is not respected
if (event[4] < -0.651486){
// This is a leaf node
result += -0.001698;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000056;
}
}
}
if (event[4] < -0.757004){
if (event[4] < -0.758911){
if (event[4] < -0.759871){
// This is a leaf node
result += 0.000349;
}
else { // if condition is not respected
// This is a leaf node
result += -0.046197;
}
}
else { // if condition is not respected
if (event[3] < -0.484201){
// This is a leaf node
result += -0.009811;
}
else { // if condition is not respected
// This is a leaf node
result += 0.059257;
}
}
}
else { // if condition is not respected
if (event[4] < -0.756236){
if (event[0] < 0.746779){
// This is a leaf node
result += -0.077151;
}
else { // if condition is not respected
// This is a leaf node
result += 0.034525;
}
}
else { // if condition is not respected
if (event[4] < -0.756152){
// This is a leaf node
result += 0.116985;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000097;
}
}
}
if (event[0] < 2.027467){
if (event[0] < 2.008068){
if (event[0] < 2.005329){
// This is a leaf node
result += 0.000018;
}
else { // if condition is not respected
// This is a leaf node
result += -0.064994;
}
}
else { // if condition is not respected
if (event[3] < 1.242420){
// This is a leaf node
result += 0.012459;
}
else { // if condition is not respected
// This is a leaf node
result += 0.079328;
}
}
}
else { // if condition is not respected
if (event[0] < 2.028569){
if (event[4] < -1.484969){
// This is a leaf node
result += 0.053729;
}
else { // if condition is not respected
// This is a leaf node
result += -0.119368;
}
}
else { // if condition is not respected
if (event[2] < 0.772819){
// This is a leaf node
result += 0.000892;
}
else { // if condition is not respected
// This is a leaf node
result += -0.007730;
}
}
}
if (event[1] < 4.088089){
if (event[1] < 4.035286){
if (event[1] < 3.906952){
// This is a leaf node
result += 0.000001;
}
else { // if condition is not respected
// This is a leaf node
result += -0.037552;
}
}
else { // if condition is not respected
// This is a leaf node
result += 0.072670;
}
}
else { // if condition is not respected
if (event[2] < 0.152884){
// This is a leaf node
result += 0.000130;
}
else { // if condition is not respected
// This is a leaf node
result += -0.055016;
}
}
if (event[1] < 3.524422){
if (event[1] < 3.500641){
if (event[1] < 3.486247){
// This is a leaf node
result += -0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += 0.071179;
}
}
else { // if condition is not respected
if (event[3] < -0.039495){
// This is a leaf node
result += -0.094541;
}
else { // if condition is not respected
// This is a leaf node
result += 0.020445;
}
}
}
else { // if condition is not respected
if (event[3] < 0.240275){
if (event[1] < 3.602510){
// This is a leaf node
result += 0.127006;
}
else { // if condition is not respected
// This is a leaf node
result += 0.002253;
}
}
else { // if condition is not respected
if (event[1] < 3.610763){
// This is a leaf node
result += -0.101069;
}
else { // if condition is not respected
// This is a leaf node
result += -0.005833;
}
}
}
if (event[1] < 1.466723){
if (event[1] < 1.466452){
if (event[1] < 1.451602){
// This is a leaf node
result += 0.000032;
}
else { // if condition is not respected
// This is a leaf node
result += 0.007711;
}
}
else { // if condition is not respected
if (event[2] < -0.866119){
// This is a leaf node
result += -0.024547;
}
else { // if condition is not respected
// This is a leaf node
result += 0.083707;
}
}
}
else { // if condition is not respected
if (event[1] < 1.467420){
if (event[3] < -0.214220){
// This is a leaf node
result += -0.007272;
}
else { // if condition is not respected
// This is a leaf node
result += -0.114435;
}
}
else { // if condition is not respected
if (event[3] < -2.969101){
// This is a leaf node
result += -0.051315;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000512;
}
}
}
if (event[3] < -2.638771){
if (event[1] < 0.655445){
if (event[1] < 0.639029){
// This is a leaf node
result += -0.000684;
}
else { // if condition is not respected
// This is a leaf node
result += -0.105939;
}
}
else { // if condition is not respected
if (event[2] < 1.265351){
// This is a leaf node
result += 0.023456;
}
else { // if condition is not respected
// This is a leaf node
result += -0.035365;
}
}
}
else { // if condition is not respected
if (event[3] < -2.538475){
if (event[0] < 0.943438){
// This is a leaf node
result += -0.023935;
}
else { // if condition is not respected
// This is a leaf node
result += 0.023978;
}
}
else { // if condition is not respected
if (event[3] < -2.536631){
// This is a leaf node
result += 0.119839;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000005;
}
}
}
if (event[3] < -2.277624){
if (event[1] < -1.997634){
if (event[2] < -0.823861){
// This is a leaf node
result += 0.043907;
}
else { // if condition is not respected
// This is a leaf node
result += -0.056588;
}
}
else { // if condition is not respected
if (event[0] < -1.516241){
// This is a leaf node
result += -0.018831;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004403;
}
}
}
else { // if condition is not respected
if (event[3] < -2.277082){
// This is a leaf node
result += -0.113754;
}
else { // if condition is not respected
if (event[3] < -2.128897){
// This is a leaf node
result += -0.006227;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000015;
}
}
}
if (event[3] < -1.887614){
if (event[4] < 1.266467){
if (event[4] < 0.652970){
// This is a leaf node
result += 0.002315;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011223;
}
}
else { // if condition is not respected
if (event[1] < -1.896554){
// This is a leaf node
result += 0.086519;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009515;
}
}
}
else { // if condition is not respected
if (event[3] < -1.886691){
if (event[0] < -0.891948){
// This is a leaf node
result += 0.044156;
}
else { // if condition is not respected
// This is a leaf node
result += -0.102825;
}
}
else { // if condition is not respected
if (event[3] < -1.886278){
// This is a leaf node
result += 0.058992;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000033;
}
}
}
if (event[1] < 2.359541){
if (event[1] < 2.337770){
if (event[1] < 2.337262){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.070073;
}
}
else { // if condition is not respected
if (event[3] < -0.868185){
// This is a leaf node
result += 0.016265;
}
else { // if condition is not respected
// This is a leaf node
result += -0.042484;
}
}
}
else { // if condition is not respected
if (event[1] < 2.360409){
// This is a leaf node
result += 0.108034;
}
else { // if condition is not respected
if (event[0] < -1.836658){
// This is a leaf node
result += 0.031247;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000762;
}
}
}
if (event[1] < 0.591687){
if (event[1] < 0.591256){
if (event[1] < 0.591172){
// This is a leaf node
result += -0.000115;
}
else { // if condition is not respected
// This is a leaf node
result += 0.077692;
}
}
else { // if condition is not respected
if (event[0] < -0.056964){
// This is a leaf node
result += -0.071937;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005051;
}
}
}
else { // if condition is not respected
if (event[1] < 0.592120){
if (event[3] < 0.998637){
// This is a leaf node
result += 0.070863;
}
else { // if condition is not respected
// This is a leaf node
result += -0.050364;
}
}
else { // if condition is not respected
if (event[3] < -1.294645){
// This is a leaf node
result += -0.002982;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000641;
}
}
}
if (event[3] < -1.136253){
if (event[3] < -1.143976){
if (event[1] < -0.921632){
// This is a leaf node
result += 0.003728;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000403;
}
}
else { // if condition is not respected
if (event[3] < -1.143877){
// This is a leaf node
result += 0.102471;
}
else { // if condition is not respected
// This is a leaf node
result += 0.017193;
}
}
}
else { // if condition is not respected
if (event[3] < -1.136198){
// This is a leaf node
result += -0.107362;
}
else { // if condition is not respected
if (event[3] < -0.935614){
// This is a leaf node
result += -0.001669;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000010;
}
}
}
if (event[3] < -1.054364){
if (event[3] < -1.058196){
if (event[3] < -1.058286){
// This is a leaf node
result += 0.000361;
}
else { // if condition is not respected
// This is a leaf node
result += -0.142088;
}
}
else { // if condition is not respected
if (event[1] < 1.346270){
// This is a leaf node
result += 0.016003;
}
else { // if condition is not respected
// This is a leaf node
result += 0.098475;
}
}
}
else { // if condition is not respected
if (event[3] < -1.052490){
if (event[3] < -1.053031){
// This is a leaf node
result += -0.001395;
}
else { // if condition is not respected
// This is a leaf node
result += -0.084852;
}
}
else { // if condition is not respected
if (event[3] < -1.052137){
// This is a leaf node
result += 0.051353;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000075;
}
}
}
if (event[4] < 1.671135){
if (event[4] < 1.632809){
if (event[4] < 1.630600){
// This is a leaf node
result += 0.000021;
}
else { // if condition is not respected
// This is a leaf node
result += -0.050447;
}
}
else { // if condition is not respected
if (event[2] < 2.072350){
// This is a leaf node
result += 0.010789;
}
else { // if condition is not respected
// This is a leaf node
result += -0.059840;
}
}
}
else { // if condition is not respected
if (event[4] < 1.682022){
if (event[3] < 0.576751){
// This is a leaf node
result += -0.030417;
}
else { // if condition is not respected
// This is a leaf node
result += 0.005724;
}
}
else { // if condition is not respected
if (event[4] < 1.683348){
// This is a leaf node
result += 0.055853;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000575;
}
}
}
if (event[3] < -2.445320){
if (event[3] < -2.456237){
if (event[0] < -0.068661){
// This is a leaf node
result += -0.007557;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006650;
}
}
else { // if condition is not respected
if (event[4] < 0.243010){
// This is a leaf node
result += -0.102381;
}
else { // if condition is not respected
// This is a leaf node
result += -0.018429;
}
}
}
else { // if condition is not respected
if (event[3] < -2.442633){
if (event[1] < 0.053462){
// This is a leaf node
result += 0.047872;
}
else { // if condition is not respected
// This is a leaf node
result += 0.125212;
}
}
else { // if condition is not respected
if (event[3] < -2.381733){
// This is a leaf node
result += 0.014174;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000006;
}
}
}
if (event[0] < -0.983987){
if (event[0] < -0.985712){
if (event[0] < -0.985828){
// This is a leaf node
result += 0.000362;
}
else { // if condition is not respected
// This is a leaf node
result += -0.099525;
}
}
else { // if condition is not respected
if (event[3] < 0.664293){
// This is a leaf node
result += 0.055244;
}
else { // if condition is not respected
// This is a leaf node
result += -0.009565;
}
}
}
else { // if condition is not respected
if (event[0] < -0.977486){
if (event[4] < -1.493913){
// This is a leaf node
result += 0.038528;
}
else { // if condition is not respected
// This is a leaf node
result += -0.018462;
}
}
else { // if condition is not respected
if (event[0] < -0.977380){
// This is a leaf node
result += 0.092728;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000065;
}
}
}
if (event[4] < -1.140589){
if (event[4] < -1.141200){
if (event[4] < -1.141653){
// This is a leaf node
result += -0.000501;
}
else { // if condition is not respected
// This is a leaf node
result += 0.058667;
}
}
else { // if condition is not respected
if (event[3] < 0.899428){
// This is a leaf node
result += -0.076971;
}
else { // if condition is not respected
// This is a leaf node
result += 0.043972;
}
}
}
else { // if condition is not respected
if (event[4] < -1.140235){
if (event[2] < -0.387140){
// This is a leaf node
result += -0.024133;
}
else { // if condition is not respected
// This is a leaf node
result += 0.127196;
}
}
else { // if condition is not respected
if (event[4] < -1.130740){
// This is a leaf node
result += 0.012469;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000040;
}
}
}
if (event[0] < 2.663655){
if (event[0] < 2.662780){
if (event[0] < 2.542813){
// This is a leaf node
result += 0.000008;
}
else { // if condition is not respected
// This is a leaf node
result += -0.011542;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.118167;
}
}
else { // if condition is not respected
if (event[0] < 2.700930){
if (event[3] < 1.421402){
// This is a leaf node
result += 0.047089;
}
else { // if condition is not respected
// This is a leaf node
result += -0.061660;
}
}
else { // if condition is not respected
if (event[1] < -2.402062){
// This is a leaf node
result += -0.097679;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000259;
}
}
}
if (event[2] < 2.903436){
if (event[2] < 2.898122){
if (event[2] < 2.806383){
// This is a leaf node
result += 0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += -0.013725;
}
}
else { // if condition is not respected
if (event[3] < 0.016241){
// This is a leaf node
result += -0.110805;
}
else { // if condition is not respected
// This is a leaf node
result += -0.024529;
}
}
}
else { // if condition is not respected
if (event[1] < -0.095296){
if (event[4] < -1.366729){
// This is a leaf node
result += 0.097434;
}
else { // if condition is not respected
// This is a leaf node
result += 0.006544;
}
}
else { // if condition is not respected
if (event[2] < 3.146538){
// This is a leaf node
result += 0.012483;
}
else { // if condition is not respected
// This is a leaf node
result += -0.027322;
}
}
}
if (event[2] < 3.457538){
if (event[2] < 3.414936){
if (event[2] < 3.324344){
// This is a leaf node
result += -0.000003;
}
else { // if condition is not respected
// This is a leaf node
result += 0.028505;
}
}
else { // if condition is not respected
if (event[0] < 0.068630){
// This is a leaf node
result += -0.010816;
}
else { // if condition is not respected
// This is a leaf node
result += -0.134319;
}
}
}
else { // if condition is not respected
if (event[4] < -0.224016){
if (event[1] < -0.361431){
// This is a leaf node
result += 0.035642;
}
else { // if condition is not respected
// This is a leaf node
result += -0.056236;
}
}
else { // if condition is not respected
if (event[2] < 3.591791){
// This is a leaf node
result += 0.081413;
}
else { // if condition is not respected
// This is a leaf node
result += 0.009859;
}
}
}
if (event[3] < -4.097473){
// This is a leaf node
result += -0.046806;
}
else { // if condition is not respected
if (event[3] < -3.199995){
if (event[4] < 1.757609){
// This is a leaf node
result += 0.003136;
}
else { // if condition is not respected
// This is a leaf node
result += 0.103238;
}
}
else { // if condition is not respected
if (event[3] < -3.023203){
// This is a leaf node
result += -0.018389;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000006;
}
}
}
if (event[3] < -3.863437){
if (event[2] < -0.379601){
// This is a leaf node
result += 0.038249;
}
else { // if condition is not respected
if (event[0] < -0.002408){
// This is a leaf node
result += -0.000139;
}
else { // if condition is not respected
// This is a leaf node
result += -0.071828;
}
}
}
else { // if condition is not respected
if (event[3] < -3.672421){
if (event[4] < 0.478157){
// This is a leaf node
result += 0.063027;
}
else { // if condition is not respected
// This is a leaf node
result += -0.036431;
}
}
else { // if condition is not respected
if (event[3] < -3.653632){
// This is a leaf node
result += -0.051393;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000000;
}
}
}
if (event[4] < -3.839994){
if (event[2] < 0.248174){
if (event[0] < 0.325009){
// This is a leaf node
result += 0.091424;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000824;
}
}
else { // if condition is not respected
if (event[2] < 0.375721){
// This is a leaf node
result += -0.100055;
}
else { // if condition is not respected
// This is a leaf node
result += 0.043405;
}
}
}
else { // if condition is not respected
if (event[4] < -3.821068){
// This is a leaf node
result += -0.096186;
}
else { // if condition is not respected
if (event[4] < -3.801609){
// This is a leaf node
result += 0.062849;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000001;
}
}
}
if (event[2] < 4.023448){
if (event[2] < 3.986183){
if (event[2] < 3.831835){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.043037;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.067834;
}
}
else { // if condition is not respected
if (event[2] < 4.269265){
if (event[3] < -0.160747){
// This is a leaf node
result += 0.013698;
}
else { // if condition is not respected
// This is a leaf node
result += 0.077234;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.016292;
}
}
if (event[2] < -0.788569){
if (event[2] < -0.791414){
if (event[1] < 1.388897){
// This is a leaf node
result += 0.000634;
}
else { // if condition is not respected
// This is a leaf node
result += -0.004073;
}
}
else { // if condition is not respected
if (event[3] < -0.780039){
// This is a leaf node
result += -0.007615;
}
else { // if condition is not respected
// This is a leaf node
result += 0.043612;
}
}
}
else { // if condition is not respected
if (event[2] < -0.773373){
if (event[3] < 1.211800){
// This is a leaf node
result += -0.007720;
}
else { // if condition is not respected
// This is a leaf node
result += -0.048513;
}
}
else { // if condition is not respected
if (event[2] < -0.773236){
// This is a leaf node
result += 0.086586;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000041;
}
}
}
if (event[1] < 0.666113){
if (event[1] < 0.646511){
if (event[1] < 0.645995){
// This is a leaf node
result += -0.000052;
}
else { // if condition is not respected
// This is a leaf node
result += 0.040153;
}
}
else { // if condition is not respected
if (event[0] < -2.046923){
// This is a leaf node
result += 0.050462;
}
else { // if condition is not respected
// This is a leaf node
result += -0.010215;
}
}
}
else { // if condition is not respected
if (event[1] < 0.672041){
if (event[1] < 0.671790){
// This is a leaf node
result += 0.013138;
}
else { // if condition is not respected
// This is a leaf node
result += 0.087812;
}
}
else { // if condition is not respected
if (event[1] < 0.674313){
// This is a leaf node
result += -0.022722;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000285;
}
}
}
if (event[0] < -2.194040){
if (event[0] < -2.240958){
if (event[1] < -1.584215){
// This is a leaf node
result += 0.021314;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000709;
}
}
else { // if condition is not respected
if (event[1] < 1.677791){
// This is a leaf node
result += -0.024276;
}
else { // if condition is not respected
// This is a leaf node
result += 0.077915;
}
}
}
else { // if condition is not respected
if (event[0] < -2.185743){
if (event[4] < -0.598036){
// This is a leaf node
result += 0.103245;
}
else { // if condition is not respected
// This is a leaf node
result += 0.011083;
}
}
else { // if condition is not respected
if (event[0] < -2.184178){
// This is a leaf node
result += -0.066248;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000018;
}
}
}
if (event[1] < -1.813412){
if (event[1] < -1.828730){
if (event[0] < 0.926309){
// This is a leaf node
result += -0.001989;
}
else { // if condition is not respected
// This is a leaf node
result += 0.008469;
}
}
else { // if condition is not respected
if (event[2] < -1.249018){
// This is a leaf node
result += 0.049352;
}
else { // if condition is not respected
// This is a leaf node
result += -0.033722;
}
}
}
else { // if condition is not respected
if (event[1] < -1.811917){
if (event[2] < -0.197370){
// This is a leaf node
result += 0.024025;
}
else { // if condition is not respected
// This is a leaf node
result += 0.106077;
}
}
else { // if condition is not respected
if (event[1] < -1.593941){
// This is a leaf node
result += 0.002745;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000029;
}
}
}
if (event[0] < 0.281223){
if (event[0] < 0.279190){
if (event[0] < 0.279083){
// This is a leaf node
result += 0.000134;
}
else { // if condition is not respected
// This is a leaf node
result += -0.115994;
}
}
else { // if condition is not respected
if (event[0] < 0.279369){
// This is a leaf node
result += 0.086855;
}
else { // if condition is not respected
// This is a leaf node
result += 0.015072;
}
}
}
else { // if condition is not respected
if (event[0] < 0.395085){
if (event[4] < 2.554636){
// This is a leaf node
result += -0.003202;
}
else { // if condition is not respected
// This is a leaf node
result += -0.059742;
}
}
else { // if condition is not respected
if (event[0] < 0.395221){
// This is a leaf node
result += 0.093624;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000144;
}
}
}
if (event[4] < 2.107707){
if (event[4] < 2.107086){
if (event[4] < 2.106895){
// This is a leaf node
result += -0.000022;
}
else { // if condition is not respected
// This is a leaf node
result += 0.111321;
}
}
else { // if condition is not respected
// This is a leaf node
result += -0.140895;
}
}
else { // if condition is not respected
if (event[4] < 2.124700){
if (event[3] < 0.194931){
// This is a leaf node
result += 0.015119;
}
else { // if condition is not respected
// This is a leaf node
result += 0.057780;
}
}
else { // if condition is not respected
if (event[3] < 1.991593){
// This is a leaf node
result += 0.000922;
}
else { // if condition is not respected
// This is a leaf node
result += -0.031441;
}
}
}
if (event[0] < 0.585868){
if (event[0] < 0.585360){
if (event[0] < 0.532056){
// This is a leaf node
result += -0.000002;
}
else { // if condition is not respected
// This is a leaf node
result += 0.004320;
}
}
else { // if condition is not respected
if (event[4] < -1.577558){
// This is a leaf node
result += -0.073805;
}
else { // if condition is not respected
// This is a leaf node
result += 0.074905;
}
}
}
else { // if condition is not respected
if (event[0] < 0.586359){
if (event[4] < 1.012785){
// This is a leaf node
result += -0.026841;
}
else { // if condition is not respected
// This is a leaf node
result += -0.138336;
}
}
else { // if condition is not respected
if (event[0] < 0.586640){
// This is a leaf node
result += 0.057876;
}
else { // if condition is not respected
// This is a leaf node
result += -0.000302;
}
}
}
if (event[0] < -2.779831){
if (event[0] < -2.786980){
if (event[3] < 1.628112){
// This is a leaf node
result += 0.002039;
}
else { // if condition is not respected
// This is a leaf node
result += -0.051915;
}
}
else { // if condition is not respected
if (event[2] < -0.892977){
// This is a leaf node
result += 0.023650;
}
else { // if condition is not respected
// This is a leaf node
result += -0.118511;
}
}
}
else { // if condition is not respected
if (event[0] < -2.753695){
if (event[1] < -0.341784){
// This is a leaf node
result += -0.009287;
}
else { // if condition is not respected
// This is a leaf node
result += 0.072874;
}
}
else { // if condition is not respected
if (event[0] < -2.745649){
// This is a leaf node
result += -0.056338;
}
else { // if condition is not respected
// This is a leaf node
result += 0.000006;
}
}
}
     result = 1. / (1. + (1. / std::exp(result)));
     preds.push_back((result > 0.5) ? 1 : 0);
     }
}

