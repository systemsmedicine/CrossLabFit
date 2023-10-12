#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <curand.h>

#include "ranNumbers.h"

#define THS_MAX 256

#define FLAG \
        fprintf(stderr, "Flag in %s:%d\n", __FILE__, __LINE__);\

#define A21 0.2
#define A31 0.075
#define A32 0.225
#define A41 (44.0/45.0)
#define A42 (-56.0/15.0)
#define A43 (32.0/9.0)
#define A51 (19372.0/6561.0)
#define A52 (-25360/2187.0)
#define A53 (64448.0/6561.0)
#define A54 (-212.0/729.0)
#define A61 (9017.0/3168.0)
#define A62 (-355.0/33.0)
#define A63 (46732.0/5247.0)
#define A64 (49.0/176.0)
#define A65 (-5103.0/18656.0)
#define A71 (35.0/384.0)
#define A73 (500.0/1113.0)
#define A74 (125.0/192.0)
#define A75 (-2187.0/6784.0)
#define A76 (11.0/84.0)
#define E1 (71.0/57600.0)
#define E3 (-71.0/16695.0)
#define E4 (71.0/1920.0)
#define E5 (-17253.0/339200.0)
#define E6 (22.0/525.0)
#define E7 -0.025

#define BETADP5 0.08
#define ALPHADP5 (0.2 - BETADP5*0.75)
#define SAFE 0.9
#define MINSCALE 0.2
#define MAXSCALE 10.0 

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- STRUCTURES =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/

typedef struct 
{
	float U1;
	float I1;
	float R1;
	float V1;

	float U2;
	float I2;
	float R2;
	float V2;
	float T2;

	float U3;
	float I3;
	float R3;
	float V3;
	float T3;
} 
comp;

typedef struct 
{
	float U10;
	float I10;
	float R10;
	float V10;

	float U20;
	float I20;
	float R20;
	float V20;
	float T20;

	float U30;
	float I30;
	float R30;
	float V30;
	float T30;

	float bet1;
	float xi1;
	float chi1;
	float del1;
	float rho1;
	float sig1;

	float bet2;
	float xi2;
	float chi2;
	float psi2;
	float del2;
	float rho2;
	float sig2;
	float eta2;
	float kap2;
	float alp2;
	float ups2;

	float bet3;
	float xi3;
	float chi3;
	float psi3;
	float del3;
	float rho3;
	float sig3;
	float eta3;
	float kap3;
	float alp3;
	float ups3;

	float gam12;
	float gam21;
	float gam23;
	float gam32;
	float zet23;
	float zet32;

	int m;

	float t0;
	float tN;
	float dt;
	int D;
	int Np;
	int nData;
	int nDataCD8;
	int flag8;
	int sizeSample;
} 
param;

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- FUNCTIONS =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/

// Encuentra la siguiente potencia de dos
long nextPow2(long x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

//-------------------------------------------------------------------------------

__device__ void derivs(int idx, param pars, float *pop, comp Y, comp *dotY)
{
	int ii = 1;
	float bet1 = pop[idx + ii];
	ii++;
	float xi1  = pop[idx + ii];
	ii++;
	float chi1 = pop[idx + ii];
	ii++;
	float del1 = pop[idx + ii];
	ii++;
	float rho1 = pop[idx + ii];
	ii++;
	float sig1 = pop[idx + ii];
	ii++;

	float bet2 = pop[idx + ii];
	ii++;
	float xi2  = pop[idx + ii];
	ii++;
	float chi2 = pop[idx + ii];
	ii++;
	float psi2 = pop[idx + ii];
	ii++;
	float del2 = pop[idx + ii];
	ii++;
	float rho2 = pop[idx + ii];
	ii++;
	float sig2 = pop[idx + ii];
	ii++;
	float eta2 = pop[idx + ii];
	ii++;
	float kap2 = pop[idx + ii];
	ii++;
	float alp2 = pop[idx + ii];
	ii++;
	float ups2 = pop[idx + ii];
	ii++;

	float bet3 = pop[idx + ii];
	ii++;
	float xi3  = pop[idx + ii];
	ii++;
	float chi3 = pop[idx + ii];
	ii++;
	float psi3 = pop[idx + ii];
	ii++;
	float del3 = pop[idx + ii];
	ii++;
	float rho3 = pop[idx + ii];
	ii++;
	float sig3 = pop[idx + ii];
	ii++;
	float eta3 = pop[idx + ii];
	ii++;
	float kap3 = pop[idx + ii];
	ii++;
	float alp3 = pop[idx + ii];
	ii++;
	float ups3 = pop[idx + ii];
	ii++;

	float gam12 = pop[idx + ii];
	ii++;
	float gam21 = pop[idx + ii];
	ii++;
	float gam23 = pop[idx + ii];
	ii++;
	float gam32 = pop[idx + ii];
	ii++;
	float zet23 = pop[idx + ii];
	ii++;
	float zet32 = pop[idx + ii];

	float m = (float) pars.m;
	float T20 = pars.T20;
	float T30 = pars.T30;

	dotY->U1 = -bet1*Y.U1*Y.V1 - xi1*Y.U1*Y.I1 + chi1*Y.R1;
	dotY->I1 = bet1*Y.U1*Y.V1 - del1*Y.I1;
	dotY->R1 = xi1*Y.U1*Y.I1 - chi1*Y.R1;
	dotY->V1 = rho1*Y.I1 - sig1*Y.V1 - gam12*Y.V1 + gam21*Y.V2;

	dotY->U2 = -bet2*Y.U2*Y.V2 - xi2*Y.U2*Y.I2 + chi2*Y.R2;
	dotY->I2 = bet2*Y.U2*Y.V2 - del2*Y.I2 - psi2*Y.I2*Y.T2;
	dotY->R2 = xi2*Y.U2*Y.I2 - chi2*Y.R2;
	dotY->V2 = rho2*Y.I2 - sig2*Y.V2 - gam21*Y.V2 + gam12*Y.V1 - gam23*Y.V2 + gam32*Y.V3;
	//dotY->T2 = eta2*Y.T2*(pow(Y.V2,m)/(pow(Y.V2,m) + pow(kap2,m)))
	//		- psi2*Y.I2*Y.T2 - ups2*Y.T2 + ups2*T20 - zet23*Y.T2 + zet32*Y.T3;
	dotY->T2 = eta2*Y.T2*(pow(Y.V2,m)/(pow(Y.V2,m) + pow(kap2,m))) - alp2*Y.T2/(1.0 + Y.V2*Y.V2)
                                - ups2*Y.T2 + ups2*T20 - zet23*Y.T2 + zet32*Y.T3;

	dotY->U3 = -bet3*Y.U3*Y.V3 - xi3*Y.U3*Y.I3 + chi3*Y.R3;
	dotY->I3 = bet3*Y.U3*Y.V3 - del3*Y.I3 - psi3*Y.I3*Y.T3;
	dotY->R3 = xi3*Y.U3*Y.I3 - chi3*Y.R3;
	dotY->V3 = rho3*Y.I3 - sig3*Y.V3 - gam32*Y.V3 + gam23*Y.V2;
	//dotY->T3 = eta3*Y.T3*(pow(Y.V3,m)/(pow(Y.V3,m) + pow(kap3,m)))
	//		- psi3*Y.I3*Y.T3 - ups3*Y.T3 + ups3*T30 - zet32*Y.T3 + zet23*Y.T2;
	dotY->T3 = eta3*Y.T3*(pow(Y.V3,m)/(pow(Y.V3,m) + pow(kap3,m))) - alp3*Y.T3/(1.0 + Y.V3*Y.V3)
				- ups3*Y.T3 + ups3*T30 - zet32*Y.T3 + zet23*Y.T2;

	return;
}

//-------------------------------------------------------------------------------
__global__ void costFunction(param pars, float *pop, float *timeData, float *dataN, float *dataT, float *dataL,
		float *timeCD8, float *cd8DataT, float *cd8DataL, float *valCostFn)
{
	int ind;

	ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind >= pars.Np) return;

	int idx;
	float t0, tN, tt;
	comp Y, dotY;

	idx = ind*pars.D;
	t0 = pars.t0;
	tN = pars.tN;

	// Initial values
	Y.U1 = pars.U10;
	Y.I1 = pars.I10;
	Y.R1 = pars.R10;
	Y.V1 = pow(10, pop[idx]); // V10

	Y.U2 = pars.U20;
	Y.I2 = pars.I20;
	Y.R2 = pars.R20;
	Y.V2 = pars.V20;
	Y.T2 = pars.T20;

	Y.U3 = pars.U30;
	Y.I3 = pars.I30;
	Y.R3 = pars.R30;
	Y.V3 = pars.V30;
	Y.T3 = pars.T30;

	derivs(idx, pars, pop, Y, &dotY);

	// ODE solver (5th-order Dormand-Prince)

	comp ytemp, k2, k3, k4, k5, k6, dotYnew, yOut;
	float aux;
	int nn, nn8;
	float h;

	float ttData, sum2, ttCD8; 
	float meanN, meanT, meanL, mean8;
	int nData, nDataCD8, sizeSample, ii, idxData;
	short nanFlag, flag, flag8;

	tt = t0;
	h = pars.dt;

	nn = 0;
	nn8 = 0;
	ttData = timeData[0];
	ttCD8 = timeCD8[0];
	sum2 = 0.0;
	nData = pars.nData;
	nDataCD8 = pars.nDataCD8;
	sizeSample = pars.sizeSample;
	flag = 0;
	flag8 = pars.flag8 == 0 ? 1 : 0; // If flag8 is off, then set up to 1 to skip it

	do
	{
		ytemp.U1 = Y.U1 + h*A21*dotY.U1;
		ytemp.I1 = Y.I1 + h*A21*dotY.I1;
		ytemp.R1 = Y.R1 + h*A21*dotY.R1;
		ytemp.V1 = Y.V1 + h*A21*dotY.V1;

		ytemp.U2 = Y.U2 + h*A21*dotY.U2;
		ytemp.I2 = Y.I2 + h*A21*dotY.I2;
		ytemp.R2 = Y.R2 + h*A21*dotY.R2;
		ytemp.V2 = Y.V2 + h*A21*dotY.V2;
		ytemp.T2 = Y.T2 + h*A21*dotY.T2;

		ytemp.U3 = Y.U3 + h*A21*dotY.U3;
		ytemp.I3 = Y.I3 + h*A21*dotY.I3;
		ytemp.R3 = Y.R3 + h*A21*dotY.R3;
		ytemp.V3 = Y.V3 + h*A21*dotY.V3;
		ytemp.T3 = Y.T3 + h*A21*dotY.T3;

		derivs(idx, pars, pop, ytemp, &k2);

		ytemp.U1 = Y.U1 + h*(A31*dotY.U1 + A32*k2.U1);
		ytemp.I1 = Y.I1 + h*(A31*dotY.I1 + A32*k2.I1);
		ytemp.R1 = Y.R1 + h*(A31*dotY.R1 + A32*k2.R1);
		ytemp.V1 = Y.V1 + h*(A31*dotY.V1 + A32*k2.V1);

		ytemp.U2 = Y.U2 + h*(A31*dotY.U2 + A32*k2.U2);
		ytemp.I2 = Y.I2 + h*(A31*dotY.I2 + A32*k2.I2);
		ytemp.R2 = Y.R2 + h*(A31*dotY.R2 + A32*k2.R2);
		ytemp.V2 = Y.V2 + h*(A31*dotY.V2 + A32*k2.V2);
		ytemp.T2 = Y.T2 + h*(A31*dotY.T2 + A32*k2.T2);

		ytemp.U3 = Y.U3 + h*(A31*dotY.U3 + A32*k2.U3);
		ytemp.I3 = Y.I3 + h*(A31*dotY.I3 + A32*k2.I3);
		ytemp.R3 = Y.R3 + h*(A31*dotY.R3 + A32*k2.R3);
		ytemp.V3 = Y.V3 + h*(A31*dotY.V3 + A32*k2.V3);
		ytemp.T3 = Y.T3 + h*(A31*dotY.T3 + A32*k2.T3);

		derivs(idx, pars, pop, ytemp, &k3);

		ytemp.U1 = Y.U1 + h*(A41*dotY.U1 + A42*k2.U1 + A43*k3.U1);
		ytemp.I1 = Y.I1 + h*(A41*dotY.I1 + A42*k2.I1 + A43*k3.I1);
		ytemp.R1 = Y.R1 + h*(A41*dotY.R1 + A42*k2.R1 + A43*k3.R1);
		ytemp.V1 = Y.V1 + h*(A41*dotY.V1 + A42*k2.V1 + A43*k3.V1);

		ytemp.U2 = Y.U2 + h*(A41*dotY.U2 + A42*k2.U2 + A43*k3.U2);
		ytemp.I2 = Y.I2 + h*(A41*dotY.I2 + A42*k2.I2 + A43*k3.I2);
		ytemp.R2 = Y.R2 + h*(A41*dotY.R2 + A42*k2.R2 + A43*k3.R2);
		ytemp.V2 = Y.V2 + h*(A41*dotY.V2 + A42*k2.V2 + A43*k3.V2);
		ytemp.T2 = Y.T2 + h*(A41*dotY.T2 + A42*k2.T2 + A43*k3.T2);

		ytemp.U3 = Y.U3 + h*(A41*dotY.U3 + A42*k2.U3 + A43*k3.U3);
		ytemp.I3 = Y.I3 + h*(A41*dotY.I3 + A42*k2.I3 + A43*k3.I3);
		ytemp.R3 = Y.R3 + h*(A41*dotY.R3 + A42*k2.R3 + A43*k3.R3);
		ytemp.V3 = Y.V3 + h*(A41*dotY.V3 + A42*k2.V3 + A43*k3.V3);
		ytemp.T3 = Y.T3 + h*(A41*dotY.T3 + A42*k2.T3 + A43*k3.T3);

		derivs(idx, pars, pop, ytemp, &k4);

		ytemp.U1 = Y.U1 + h*(A51*dotY.U1 + A52*k2.U1 + A53*k3.U1 + A54*k4.U1);
		ytemp.I1 = Y.I1 + h*(A51*dotY.I1 + A52*k2.I1 + A53*k3.I1 + A54*k4.I1);
		ytemp.R1 = Y.R1 + h*(A51*dotY.R1 + A52*k2.R1 + A53*k3.R1 + A54*k4.R1);
		ytemp.V1 = Y.V1 + h*(A51*dotY.V1 + A52*k2.V1 + A53*k3.V1 + A54*k4.V1);

		ytemp.U2 = Y.U2 + h*(A51*dotY.U2 + A52*k2.U2 + A53*k3.U2 + A54*k4.U2);
		ytemp.I2 = Y.I2 + h*(A51*dotY.I2 + A52*k2.I2 + A53*k3.I2 + A54*k4.I2);
		ytemp.R2 = Y.R2 + h*(A51*dotY.R2 + A52*k2.R2 + A53*k3.R2 + A54*k4.R2);
		ytemp.V2 = Y.V2 + h*(A51*dotY.V2 + A52*k2.V2 + A53*k3.V2 + A54*k4.V2);
		ytemp.T2 = Y.T2 + h*(A51*dotY.T2 + A52*k2.T2 + A53*k3.T2 + A54*k4.T2);

		ytemp.U3 = Y.U3 + h*(A51*dotY.U3 + A52*k2.U3 + A53*k3.U3 + A54*k4.U3);
		ytemp.I3 = Y.I3 + h*(A51*dotY.I3 + A52*k2.I3 + A53*k3.I3 + A54*k4.I3);
		ytemp.R3 = Y.R3 + h*(A51*dotY.R3 + A52*k2.R3 + A53*k3.R3 + A54*k4.R3);
		ytemp.V3 = Y.V3 + h*(A51*dotY.V3 + A52*k2.V3 + A53*k3.V3 + A54*k4.V3);
		ytemp.T3 = Y.T3 + h*(A51*dotY.T3 + A52*k2.T3 + A53*k3.T3 + A54*k4.T3);

		derivs(idx, pars, pop, ytemp, &k5);

		ytemp.U1 = Y.U1 + h*(A61*dotY.U1 + A62*k2.U1 + A63*k3.U1 + A64*k4.U1 + A65*k5.U1);
		ytemp.I1 = Y.I1 + h*(A61*dotY.I1 + A62*k2.I1 + A63*k3.I1 + A64*k4.I1 + A65*k5.I1);
		ytemp.R1 = Y.R1 + h*(A61*dotY.R1 + A62*k2.R1 + A63*k3.R1 + A64*k4.R1 + A65*k5.R1);
		ytemp.V1 = Y.V1 + h*(A61*dotY.V1 + A62*k2.V1 + A63*k3.V1 + A64*k4.V1 + A65*k5.V1);

		ytemp.U2 = Y.U2 + h*(A61*dotY.U2 + A62*k2.U2 + A63*k3.U2 + A64*k4.U2 + A65*k5.U2);
		ytemp.I2 = Y.I2 + h*(A61*dotY.I2 + A62*k2.I2 + A63*k3.I2 + A64*k4.I2 + A65*k5.I2);
		ytemp.R2 = Y.R2 + h*(A61*dotY.R2 + A62*k2.R2 + A63*k3.R2 + A64*k4.R2 + A65*k5.R2);
		ytemp.V2 = Y.V2 + h*(A61*dotY.V2 + A62*k2.V2 + A63*k3.V2 + A64*k4.V2 + A65*k5.V2);
		ytemp.T2 = Y.T2 + h*(A61*dotY.T2 + A62*k2.T2 + A63*k3.T2 + A64*k4.T2 + A65*k5.T2);

		ytemp.U3 = Y.U3 + h*(A61*dotY.U3 + A62*k2.U3 + A63*k3.U3 + A64*k4.U3 + A65*k5.U3);
		ytemp.I3 = Y.I3 + h*(A61*dotY.I3 + A62*k2.I3 + A63*k3.I3 + A64*k4.I3 + A65*k5.I3);
		ytemp.R3 = Y.R3 + h*(A61*dotY.R3 + A62*k2.R3 + A63*k3.R3 + A64*k4.R3 + A65*k5.R3);
		ytemp.V3 = Y.V3 + h*(A61*dotY.V3 + A62*k2.V3 + A63*k3.V3 + A64*k4.V3 + A65*k5.V3);
		ytemp.T3 = Y.T3 + h*(A61*dotY.T3 + A62*k2.T3 + A63*k3.T3 + A64*k4.T3 + A65*k5.T3);

		derivs(idx, pars, pop, ytemp, &k6);

		yOut.U1 = Y.U1 + h*(A71*dotY.U1 + A73*k3.U1 + A74*k4.U1 + A75*k5.U1 + A76*k6.U1);
		yOut.I1 = Y.I1 + h*(A71*dotY.I1 + A73*k3.I1 + A74*k4.I1 + A75*k5.I1 + A76*k6.I1);
		yOut.R1 = Y.R1 + h*(A71*dotY.R1 + A73*k3.R1 + A74*k4.R1 + A75*k5.R1 + A76*k6.R1);
		yOut.V1 = Y.V1 + h*(A71*dotY.V1 + A73*k3.V1 + A74*k4.V1 + A75*k5.V1 + A76*k6.V1);

		yOut.U2 = Y.U2 + h*(A71*dotY.U2 + A73*k3.U2 + A74*k4.U2 + A75*k5.U2 + A76*k6.U2);
		yOut.I2 = Y.I2 + h*(A71*dotY.I2 + A73*k3.I2 + A74*k4.I2 + A75*k5.I2 + A76*k6.I2);
		yOut.R2 = Y.R2 + h*(A71*dotY.R2 + A73*k3.R2 + A74*k4.R2 + A75*k5.R2 + A76*k6.R2);
		yOut.V2 = Y.V2 + h*(A71*dotY.V2 + A73*k3.V2 + A74*k4.V2 + A75*k5.V2 + A76*k6.V2);
		yOut.T2 = Y.T2 + h*(A71*dotY.T2 + A73*k3.T2 + A74*k4.T2 + A75*k5.T2 + A76*k6.T2);

		yOut.U3 = Y.U3 + h*(A71*dotY.U3 + A73*k3.U3 + A74*k4.U3 + A75*k5.U3 + A76*k6.U3);
		yOut.I3 = Y.I3 + h*(A71*dotY.I3 + A73*k3.I3 + A74*k4.I3 + A75*k5.I3 + A76*k6.I3);
		yOut.R3 = Y.R3 + h*(A71*dotY.R3 + A73*k3.R3 + A74*k4.R3 + A75*k5.R3 + A76*k6.R3);
		yOut.V3 = Y.V3 + h*(A71*dotY.V3 + A73*k3.V3 + A74*k4.V3 + A75*k5.V3 + A76*k6.V3);
		yOut.T3 = Y.T3 + h*(A71*dotY.T3 + A73*k3.T3 + A74*k4.T3 + A75*k5.T3 + A76*k6.T3);

		derivs(idx, pars, pop, yOut, &dotYnew);

		nanFlag = 0;
		if (isnan(yOut.U1)) nanFlag = 1;
		if (isnan(yOut.I1)) nanFlag = 1;
		if (isnan(yOut.R1)) nanFlag = 1;
		if (isnan(yOut.V1)) nanFlag = 1;

		if (isnan(yOut.U2)) nanFlag = 1;
		if (isnan(yOut.I2)) nanFlag = 1;
		if (isnan(yOut.R2)) nanFlag = 1;
		if (isnan(yOut.V2)) nanFlag = 1;
		if (isnan(yOut.T2)) nanFlag = 1;

		if (isnan(yOut.U3)) nanFlag = 1;
		if (isnan(yOut.I3)) nanFlag = 1;
		if (isnan(yOut.R3)) nanFlag = 1;
		if (isnan(yOut.V3)) nanFlag = 1;
		if (isnan(yOut.T3)) nanFlag = 1;
		if (nanFlag) break;

	        if (yOut.V1 < 0.0) yOut.V1 = 0.0;
	        if (yOut.V2 < 0.0) yOut.V2 = 0.0;
	        if (yOut.V3 < 0.0) yOut.V3 = 0.0;

		tt += h;

		// This part calculates the RMS
		if (tt > ttData && !flag)
		{
			meanN = yOut.V1 < 1.0 ? 0.0 : log10(yOut.V1);
			meanT = yOut.V2 < 1.0 ? 0.0 : log10(yOut.V2);
			meanL = yOut.V3 < 1.0 ? 0.0 : log10(yOut.V3);

			for (ii=0; ii<sizeSample; ii++)
			{
				idxData = ii + nn*sizeSample;
				aux = dataN[idxData] - meanN;
				sum2 += aux*aux;
				aux = dataT[idxData] - meanT;
				sum2 += aux*aux;
				aux = dataL[idxData] - meanL;
				sum2 += aux*aux;
			}

			nn++;
			if (nn >= nData) flag = 1;
			if (!flag) ttData = timeData[nn];
		}

		// This calculates the qualitative part
		if (tt > ttCD8 - 0.5 && !flag8)
		{
			mean8 = yOut.T2 < 1.0 ? 0.0 : log10(yOut.T2);
			aux = cd8DataT[nn8] - mean8;
			if (aux < 0.0) aux *= -1;
			if (aux > 0.25)
			{
				nanFlag = 1;
				break;
			}
			
			mean8 = yOut.T3 < 1.0 ? 0.0 : log10(yOut.T3);
			aux = cd8DataL[nn8] - mean8;
			if (aux < 0.0) aux *= -1;
			if (aux > 0.25)
			{
				nanFlag = 1;
				break;
			}
			
			if (tt >= ttCD8 + 0.5)
			{
				nn8++;
				if (nn8 >= nDataCD8) flag8 = 1;
				if (!flag8) ttCD8 = timeCD8[nn8];
			}
		}

		if (flag && flag8) break;

		dotY = dotYnew;
		Y = yOut;
	}
	while (tt <= tN);

	valCostFn[ind] = nanFlag ? 1e10 : sqrt(sum2/(nData*sizeSample));

	return;
}

//-------------------------------------------------------------------------------

__global__ void newPopulation(int Np, int D, float Cr, float Fm, float *randUni,
int3 *iiMut, float *lowerLim, float *upperLim, float *pop, float *newPop)
{
	int ind, jj, idx, flag = 0;
	int3 iiM, idxM;
	float trial, auxL, auxU;

	ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind >= Np) return;

	iiM = iiMut[ind];

	for (jj=0; jj<D; jj++)
	{
		idx = ind*D + jj;

		auxL = lowerLim[jj];
		auxU = upperLim[jj];
		if (auxL == auxU)
		{
			newPop[idx] = auxL;
			continue;
		}

		idxM.x = iiM.x*D + jj;
		idxM.y = iiM.y*D + jj;
		idxM.z = iiM.z*D + jj;

		if (randUni[idx] <= Cr)
		{
			//trial = pop[idxM.x] + Fm*(pop[idxM.y] - pop[idxM.z]);
			trial = pop[idx] + Fm*(pop[idxM.x] - pop[idx]) + Fm*(pop[idxM.y] - pop[idxM.z]);
			if (trial < auxL) trial = auxL;
			if (trial > auxU) trial = auxU;

			newPop[idx] = trial;
			flag = 1;
		}
		else newPop[idx] = pop[idx];
	}

	// Se asegura que exista al menos un elemento
	// del vector mutante en la nueva población
	if (!flag)
	{
		while (1)
		{
			jj = int(D*randUni[ind]);
			if (jj == D) jj--;
			auxL = lowerLim[jj];
			auxU = upperLim[jj];
			if (auxL == auxU) continue;
			break;
		}

		idx = ind*D + jj;
		idxM.x = iiM.x*D + jj;
		idxM.y = iiM.y*D + jj;
		idxM.z = iiM.z*D + jj;

		//trial = pop[idxM.x] + Fm*(pop[idxM.y] - pop[idxM.z]);
		trial = pop[idx] + Fm*(pop[idxM.x] - pop[idx]) + Fm*(pop[idxM.y] - pop[idxM.z]);
		if (trial < auxL) trial = auxL;
		if (trial > auxU) trial = auxU;

		newPop[idx] = trial;
	}

	return;
}

//-------------------------------------------------------------------------------

__global__ void selection(int Np, int D, float *pop, float *newPop,
float *valCostFn, float *newValCostFn)
{
	int ind, jj, idx;

	ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind >= Np) return;

	if  (newValCostFn[ind] > valCostFn[ind]) return;

	for (jj=0; jj<D; jj++)
	{
		idx = ind*D + jj;
		pop[idx] = newPop[idx];
	}
	valCostFn[ind] = newValCostFn[ind];

	return;
}

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- MAIN =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/

int main()
{
	/*+*+*+*+*+ START TO FETCH DATA	+*+*+*+*+*/
	int nData, nDataCD8, nn;
	float auxfloat;
	float *timeData, *meanN, *stdN, *meanT, *stdT, *meanL, *stdL;
	float *timeCD8, *cd8DataT, *cd8DataL;
	char renglon[200], dirData[500], *linea;
	FILE *fileRead;

	sprintf(dirData, "data/viral_load.csv");
	fileRead = fopen(dirData, "r");

	nData = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;
		nData++;
	}
	fclose(fileRead);

	if (nData == 0)
	{
		printf("Error: no hay datos\n");
		exit (1);
	}
	nData--;

	cudaMallocManaged(&timeData, nData*sizeof(float));
	meanN = (float *) malloc(nData*sizeof(float));
	stdN = (float *) malloc(nData*sizeof(float));
	meanT = (float *) malloc(nData*sizeof(float));
	stdT = (float *) malloc(nData*sizeof(float));
	meanL = (float *) malloc(nData*sizeof(float));
	stdL = (float *) malloc(nData*sizeof(float));

	fileRead = fopen(dirData, "r");
	if (fgets(renglon, sizeof(renglon), fileRead) == NULL) exit (1);

	nn = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;

		linea = strtok(renglon, ",");
		sscanf(linea, "%f", &auxfloat);
		timeData[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		meanN[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		stdN[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		meanT[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		stdT[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		meanL[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		stdL[nn] = auxfloat;

		nn++;
	}
	fclose(fileRead);

	sprintf(dirData, "data/cd8_h3n2.csv");
	fileRead = fopen(dirData, "r");

	nDataCD8 = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;
		nDataCD8++;
	}
	fclose(fileRead);

	if (nDataCD8 == 0)
	{
		printf("Error: no hay datos de CD8\n");
		exit (1);
	}
	nDataCD8--;

	cudaMallocManaged(&timeCD8, nDataCD8*sizeof(float));
	cudaMallocManaged(&cd8DataT, nDataCD8*sizeof(float));
	cudaMallocManaged(&cd8DataL, nDataCD8*sizeof(float));

	fileRead = fopen(dirData, "r");
	if (fgets(renglon, sizeof(renglon), fileRead) == NULL) exit (1);

	nn = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;

		linea = strtok(renglon, ",");
		sscanf(linea, "%f", &auxfloat);
		timeCD8[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		cd8DataT[nn] = log10(auxfloat);

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		cd8DataL[nn] = log10(auxfloat);

		nn++;
	}
	fclose(fileRead);

    	/*+*+*+*+*+ DIFERENTIAL EVOLUTION +*+*+*+*+*/
	int Np, itMax, seed, D, flag8;
	float Fm, Cr, t0, tN, dt;
	int err_flag = 0;

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	// Tamaño de la población
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &Np);

	// Iteraciones máximas
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &itMax);

	// Probabilidad de recombinación
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &Cr);

	// Factor de mutación
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &Fm);

	// Semilla para números aleatorios
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &seed);

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	// Tiempo inicial
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &t0);

	// Tiempo final
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &tN);

	// Step time
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &dt);

	// Numero de variables
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &D);

	// Include qualitative fit of CD8
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &flag8);

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	if (err_flag)
	{
		printf("Error en archivo de parámetros (.data)\n");
		exit (1);
	}

	param pars;

	pars.D = D;
	pars.m = 2;
	pars.t0 = t0;
	pars.tN = tN;
	pars.Np = Np;
	pars.dt = dt;
	pars.nData = nData;
	pars.nDataCD8 = nDataCD8;
	pars.flag8 = flag8;

	// Initial values
        pars.U10 = 5e8;
        pars.I10 = 0.0;
        pars.R10 = 0.0;

        pars.U20 = 5e8;
        pars.I20 = 0.0;
        pars.R20 = 0.0;
        pars.V20 = 0.0;
        pars.T20 = 2e2;

        pars.U30 = 5e8;
        pars.I30 = 0.0;
        pars.R30 = 0.0;
        pars.V30 = 0.0;
        pars.T30 = 2e5;


	float *lowerLim, *upperLim, *pop;
	int ii, jj, idx;

	cudaMallocManaged(&lowerLim, D*sizeof(float));
	cudaMallocManaged(&upperLim, D*sizeof(float));

	float aux;
	float auxL, auxU;

	for (jj=0; jj<D; jj++)
	{
		if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
		else sscanf(renglon, "[%f : %f]", &auxL, &auxU);
		lowerLim[jj] = auxL;
		upperLim[jj] = auxU;

		if (auxL > auxU)
		{
			printf("Error: Invalid range in parameter %d (param file)\n", jj);
			exit (1);
		}
	}

	cudaMallocManaged(&pop, Np*D*sizeof(float));

	// Inicializa números aleatorios
	if (seed < 0) seed *= -1;
	Ran ranUni(seed);
	Normaldev ranNorm(0.0, 1.0, seed); // Standard dev (Z)

	int sizeSample = 5;
	pars.sizeSample = sizeSample;

	// Generate random data in normal distribution
	float *dataN, *dataT, *dataL;
	cudaMallocManaged(&dataN, sizeSample*nData*sizeof(float));
	cudaMallocManaged(&dataT, sizeSample*nData*sizeof(float));
	cudaMallocManaged(&dataL, sizeSample*nData*sizeof(float));

	// Linear transformation from Z to normal dev X
	// Z = (X - meanX) / stdX -> X = Z*stdX + meanX
	for (ii=0; ii<nData; ii++)
		for (jj=0; jj<sizeSample; jj++)
		{
			idx = jj + ii*sizeSample;
			dataN[idx] = meanN[ii] + stdN[ii]*ranNorm.dev();
			dataT[idx] = meanT[ii] + stdT[ii]*ranNorm.dev();
			dataL[idx] = meanL[ii] + stdL[ii]*ranNorm.dev();
		}
	free(meanN);
	free(meanT);
	free(meanL);
	free(stdN);
	free(stdT);
	free(stdL);

	// Inicializa población
	for (jj=0; jj<D; jj++)
	{
		aux = upperLim[jj] - lowerLim[jj];
		for (ii=0; ii<Np; ii++)
		{
			idx = ii*D + jj;
			pop[idx] = lowerLim[jj] + aux*ranUni.doub();
		}
	}

	int ths, blks;
	float *valCostFn, *d_newValCostFn;

	cudaMallocManaged(&valCostFn, Np*sizeof(float));
	cudaMalloc(&d_newValCostFn, Np*sizeof(float));

	// Estimate the number of threads and blocks for the GPU
	ths = (Np < THS_MAX) ? nextPow2(Np) : THS_MAX;
	blks = 1 + (Np - 1)/ths;

	// Calcula el valor de la función objetivo
	costFunction<<<blks, ths>>>(pars, pop, timeData, dataN, dataT, dataL, timeCD8, cd8DataT, cd8DataL, valCostFn);
	cudaDeviceSynchronize();

    	/*+*+*+*+*+ START OPTIMIZATION +*+*+*+*+*/
	int it, xx, yy, zz, flag;
	int3 *iiMut;
	float *d_randUni, *d_newPop;
	float minVal;
	int iiMin;
	curandGenerator_t gen;

	cudaMallocManaged(&iiMut, Np*sizeof(int3));
	cudaMalloc(&d_newPop, Np*D*sizeof(float));

	// Initialize random numbers with a standard normal distribution
	// I use cuRand libraries 
	cudaMalloc(&d_randUni, Np*D*sizeof(float)); // Array only for GPU
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(gen, seed);

	// Empiezan las iteraciones
	for (it=0; it<itMax; it++)
	{
		flag = it%50;

		// Encuentra cual es el minimo de la pobalción
		minVal = valCostFn[0];
		iiMin = 0;
		if (!flag)
			for(ii=1; ii<Np; ii++) if (minVal > valCostFn[ii])
			{
				minVal = valCostFn[ii];
				iiMin = ii;
			}

		if (!flag)
		{
			printf("Iteration %d\n", it);
			printf("RMS_min = %f\n", minVal);
		}

		//xx = iiMin;
		for (ii=0; ii<Np; ii++)
		{
			do xx = Np*ranUni.doub(); while (xx == ii);
			do yy = Np*ranUni.doub(); while (yy == ii || yy == xx);
			do zz = Np*ranUni.doub(); while (zz == ii || zz == yy || zz == xx);

			iiMut[ii].x = xx; iiMut[ii].y = yy; iiMut[ii].z = zz;
		}

		// Generate random numbers and then update positions
		curandGenerateUniform(gen, d_randUni, Np*D);

		// Genera nueva población
		newPopulation<<<blks, ths>>>(Np, D, Cr, Fm, d_randUni, iiMut, lowerLim, upperLim, pop, d_newPop);

		// Calcula el valor de la función objetivo
		costFunction<<<blks, ths>>>(pars, d_newPop, timeData, dataN, dataT, dataL, timeCD8, cd8DataT, cd8DataL, d_newValCostFn);

		// Selecciona el mejor vector y lo guarda en la poblacion "pop"
		selection<<<blks, ths>>>(Np, D, pop, d_newPop, valCostFn, d_newValCostFn);

		cudaDeviceSynchronize();
	}

	// Encuentra cual es el minimo de la pobalción
	minVal = valCostFn[0];
	iiMin = 0;
	for (ii=1; ii<Np; ii++) if (minVal > valCostFn[ii])
	{
		minVal = valCostFn[ii];
		iiMin = ii;
	}

	// Imprime el mejor vector de parámetros

	FILE *fPar;
	fPar = fopen("bestPars.dat", "a");
	if (valCostFn[iiMin] < 10)
	{
		//fprintf(fPar, "#BestPar: RMS = %e\n", minVal);
		for (jj=0; jj<D-1; jj++) fprintf(fPar, "%.4e\t", pop[iiMin*D + jj]);
		fprintf(fPar, "%.4e\n", pop[iiMin*D + D-1]);
	}
	fclose(fPar);

	printf("FINISHED\n");

	cudaFree(timeData);
	cudaFree(timeCD8);
	cudaFree(lowerLim);
	cudaFree(upperLim);
	cudaFree(dataN);
	cudaFree(dataT);
	cudaFree(dataL);
	cudaFree(cd8DataT);
	cudaFree(cd8DataL);
	cudaFree(iiMut);
	cudaFree(pop);
	cudaFree(d_newPop);
	cudaFree(valCostFn);
	cudaFree(d_newValCostFn);
	cudaFree(d_randUni);
	curandDestroyGenerator(gen);

	exit (0);
}
