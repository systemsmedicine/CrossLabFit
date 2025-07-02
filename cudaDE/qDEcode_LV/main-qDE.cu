#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <curand.h>

#include "ranNumbers.h"

#define THS_MAX 256

#define FLAG \
        fprintf(stderr, "Flag in %s:%d\n", __FILE__, __LINE__);\

// Dormand-Prince coefficients
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

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- STRUCTURES =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/

typedef struct 
{
	float X1;
	float X2;
	float X3;
} 
comp;

typedef struct
{
        float min;
        float max;
}
window;

typedef struct 
{
	float X1_0;
	float X2_0;
	float X3_0;

	float a1;
	float a2;
	float a3;
	float a4;
	float a5;
	float a6;
	float a7;

	float t0;
	float tN;
	float dt;

	int D;
	int Np;
	int nData_X1;
	int nData_X3;
	int qnData_X3;
	int rssFlag_X3;
	int qFlag_X3;
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

__device__ void modelLV(int idx, param pars, float *pop, comp Y, comp *dotY)
{
	int ii = 0;
	float a0 = pop[idx + ii];
	ii++;
	float a1 = pop[idx + ii];
	ii++;
	float a2 = pop[idx + ii];
	ii++;
	float a3 = pop[idx + ii];
	ii++;
	float a4 = pop[idx + ii];
	ii++;
	float a5 = pop[idx + ii];
	ii++;
	float a6 = pop[idx + ii];
	ii++;
	float a7 = pop[idx + ii];
	ii++;
	float a8 = pop[idx + ii];
	ii++;
	float a9 = pop[idx + ii];

	// Three-species LV model
	dotY->X1 = a0*Y.X1 - a1*Y.X1 - a2*Y.X1*Y.X2 + a3*Y.X1*Y.X3;
	dotY->X2 = a4*Y.X1*Y.X2 - a5*Y.X2 - a6*Y.X2*Y.X3;
	dotY->X3 = -a7*Y.X1*Y.X3 + a8*Y.X2*Y.X3 - a9*Y.X3;

	return;
}

//-------------------------------------------------------------------------------

__device__ void deriv_step(int idx, param pars, float *pop, comp *Y)
{
        float h = pars.dt;
        comp Yold, Ytemp, k1, k2, k3, k4, k5, k6;

        // Old Y values
        Yold.X1 = Y->X1;
        Yold.X2 = Y->X2;
        Yold.X3 = Y->X3;

        modelLV(idx, pars, pop, Yold, &k1);

        Ytemp.X1 = Yold.X1 + h*A21*k1.X1;
        Ytemp.X2 = Yold.X2 + h*A21*k1.X2;
        Ytemp.X3 = Yold.X3 + h*A21*k1.X3;

        modelLV(idx, pars, pop, Ytemp, &k2);

        Ytemp.X1 = Yold.X1 + h*(A31*k1.X1 + A32*k2.X1);
        Ytemp.X2 = Yold.X2 + h*(A31*k1.X2 + A32*k2.X2);
        Ytemp.X3 = Yold.X3 + h*(A31*k1.X3 + A32*k2.X3);

        modelLV(idx, pars, pop, Ytemp, &k3);

        Ytemp.X1 = Yold.X1 + h*(A41*k1.X1 + A42*k2.X1 + A43*k3.X1);
        Ytemp.X2 = Yold.X2 + h*(A41*k1.X2 + A42*k2.X2 + A43*k3.X2);
        Ytemp.X3 = Yold.X3 + h*(A41*k1.X3 + A42*k2.X3 + A43*k3.X3);

        modelLV(idx, pars, pop, Ytemp, &k4);

        Ytemp.X1 = Yold.X1 + h*(A51*k1.X1 + A52*k2.X1 + A53*k3.X1 + A54*k4.X1);
        Ytemp.X2 = Yold.X2 + h*(A51*k1.X2 + A52*k2.X2 + A53*k3.X2 + A54*k4.X2);
        Ytemp.X3 = Yold.X3 + h*(A51*k1.X3 + A52*k2.X3 + A53*k3.X3 + A54*k4.X3);

        modelLV(idx, pars, pop, Ytemp, &k5);

        Ytemp.X1 = Yold.X1 + h*(A61*k1.X1 + A62*k2.X1 + A63*k3.X1 + A64*k4.X1 + A65*k5.X1);
        Ytemp.X2 = Yold.X2 + h*(A61*k1.X2 + A62*k2.X2 + A63*k3.X2 + A64*k4.X2 + A65*k5.X2);
        Ytemp.X3 = Yold.X3 + h*(A61*k1.X3 + A62*k2.X3 + A63*k3.X3 + A64*k4.X3 + A65*k5.X3);

        modelLV(idx, pars, pop, Ytemp, &k6);

        // New Y values
        Y->X1 = Yold.X1 + h*(A71*k1.X1 + A73*k3.X1 + A74*k4.X1 + A75*k5.X1 + A76*k6.X1);
        Y->X2 = Yold.X2 + h*(A71*k1.X2 + A73*k3.X2 + A74*k4.X2 + A75*k5.X2 + A76*k6.X2);
        Y->X3 = Yold.X3 + h*(A71*k1.X3 + A73*k3.X3 + A74*k4.X3 + A75*k5.X3 + A76*k6.X3);

        return;
}

//-------------------------------------------------------------------------------

__global__ void costFunction(param pars, float *pop, float *timeQt_X1, float *dataQt_X1,
		float *timeQt_X3, float *dataQt_X3, window *timeQl_X3, window *dataQl_X3,
		float *costFn)
{
	int ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind >= pars.Np) return;

	int penaltyFlag = 0;
	int rssFlag_X1 = 1;
	int rssFlag_X3 = pars.rssFlag_X3;
	int qFlag_X3 = pars.qFlag_X3;

	int nn_X1 = 0, nn_X3 = 0, qnn_X3 = 0;
        int nData_X1 = pars.nData_X1,
	    nData_X3 = pars.nData_X3,
	    qnData_X3 = pars.qnData_X3;
        float aux, sum2 = 0.0;
        float tQt_X1 = timeQt_X1[0];
        float tQt_X3 = timeQt_X3[0];
        window tQl_X3 = timeQl_X3[0];

	comp Y;
	int idx = ind*pars.D;
	float t = pars.t0;
	float h = pars.dt;

	// Initial values
	Y.X1 = pars.X1_0;
	Y.X2 = pars.X2_0;
	Y.X3 = pars.X3_0;

	while (t <= pars.tN)
	{
		// Dormand-Prince method to compute the next state
                deriv_step(idx, pars, pop, &Y);
                t += h;

		// Check for NaN and inf values
                if (isnan(Y.X1) || isnan(Y.X2) || isnan(Y.X3)
                        || isinf(Y.X1) || isinf(Y.X2) || isinf(Y.X3))
                {
                        penaltyFlag = 1;
                        break;
                }

		if (Y.X1 < 0.0) Y.X1 = 0.0;
                if (Y.X2 < 0.0) Y.X2 = 0.0;
                if (Y.X3 < 0.0) Y.X3 = 0.0;

		// This part calculates the quantitative RSS
		if (t >= tQt_X1 && rssFlag_X1)
		{
			while (1)
			{
				aux = dataQt_X1[nn_X1] - Y.X1;
				sum2 += aux*aux;
				nn_X1++;

				if (nn_X1 >= nData_X1)
				{
					rssFlag_X1 = 0;
					break;
				}

				if (timeQt_X1[nn_X1] != tQt_X1)
				{
					tQt_X1 = timeQt_X1[nn_X1];
					break;

				}
			}
		}

		if (t >= tQt_X3 && rssFlag_X3)
		{
			while (1)
			{
				aux = dataQt_X3[nn_X3] - Y.X3;
				sum2 += aux*aux;
				nn_X3++;

				if (nn_X3 >= nData_X3)
				{
					rssFlag_X3 = 0;
					break;
				}

				if (timeQt_X3[nn_X3] != tQt_X3)
				{
					tQt_X1 = timeQt_X1[nn_X3];
					break;

				}
			}
		}

		// Qualitative penalties
		if (t > tQl_X3.min && qFlag_X3)
		{
			if (Y.X3 > dataQl_X3[qnn_X3].min && Y.X3 < dataQl_X3[qnn_X3].max)
                        {
                                qnn_X3++;
                                if (qnn_X3 >= qnData_X3) qFlag_X3 = 0;
                                else tQl_X3 = timeQl_X3[qnn_X3];
                        }
                        else if (t > tQl_X3.max)
                        {
                                penaltyFlag = 1;
                                break;
                        }
		}

		if (!rssFlag_X1 && !rssFlag_X3 && !qFlag_X3) break;
	}

	if (isinf(sum2)) penaltyFlag = 1;
        costFn[ind] = penaltyFlag ? 1e38 : sum2;

	return;
}

//-------------------------------------------------------------------------------

__global__ void newPopulation(int Np, int D, float Cr, float Fm, float *randUni,
		int3 *iiMut, float *lowerLim, float *upperLim, float *pop, float *newPop)
{
	int ind, jj, idx, auxInt, flag = 0;
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
			// DE/rand/1 || DE/best/1
                        trial = pop[idxM.x] + Fm*(pop[idxM.y] - pop[idxM.z]);
                        // DE/current-to-best/1
                        //trial = pop[idx] + Fm*(pop[idxM.x] - pop[idx])
                        //              + Fm*(pop[idxM.y] - pop[idxM.z]);

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
		auxInt = ind*D;
		while (1)
		{
			jj = int(D*randUni[auxInt%(Np*D)]);
			if (jj == D) jj--;
			auxInt++;
			auxL = lowerLim[jj];
			auxU = upperLim[jj];
			if (auxL == auxU) continue;
			break;
		}

		idx = ind*D + jj;
		idxM.x = iiM.x*D + jj;
		idxM.y = iiM.y*D + jj;
		idxM.z = iiM.z*D + jj;

		// DE/rand/1 || DE/best/1
                trial = pop[idxM.x] + Fm*(pop[idxM.y] - pop[idxM.z]);
                // DE/current-to-best/1
                //trial = pop[idx] + Fm*(pop[idxM.x] - pop[idx])
                //              + Fm*(pop[idxM.y] - pop[idxM.z]);

		if (trial < auxL) trial = auxL;
		if (trial > auxU) trial = auxU;

		newPop[idx] = trial;
	}

	return;
}

//-------------------------------------------------------------------------------

__global__ void selection(int Np, int D, float *pop, float *newPop,
		float *costFn, float *newCostFn)
{
	int ind, jj, idx;

	ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind >= Np) return;

	if (newCostFn[ind] > costFn[ind]) return;

	for (jj=0; jj<D; jj++)
	{
		idx = ind*D + jj;
		pop[idx] = newPop[idx];
	}
	costFn[ind] = newCostFn[ind];

	return;
}

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- MAIN =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/

int main()
{
	/*+*+*+*+*+ FETCH DATA	+*+*+*+*+*/
	int nData_X1, nData_X3, nn;
	float auxFloat;
	float *timeQt_X1, *timeQt_X3;
	float *dataQt_X1_raw, *dataQt_X3_raw;
	char renglon[200], dirData[500], *linea;
	FILE *fileRead;

	// Raw data for X1 variable
	sprintf(dirData, "LVdata_noise.csv");
	fileRead = fopen(dirData, "r");

	nData_X1 = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;
		nData_X1++;
	}
	fclose(fileRead);

	if (nData_X1 == 0)
	{
		printf("Error: Empty file in %s\n", dirData);
		exit (1);
	}
	nData_X1--;

	cudaMallocManaged(&timeQt_X1, nData_X1*sizeof(float));
	dataQt_X1_raw = (float *) malloc(nData_X1*sizeof(float));

	fileRead = fopen(dirData, "r");
	if (fgets(renglon, sizeof(renglon), fileRead) == NULL) exit (1);

	nn = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;

		linea = strtok(renglon, ",");
		sscanf(linea, "%f", &auxFloat);
		timeQt_X1[nn] = auxFloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxFloat);
		dataQt_X1_raw[nn] = auxFloat;

		nn++;
	}
	fclose(fileRead);

	// Raw data for X3 variable
	sprintf(dirData, "LVdata_noise_X3.csv");
	fileRead = fopen(dirData, "r");

	nData_X3 = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;
		nData_X3++;
	}
	fclose(fileRead);

	if (nData_X3 == 0)
	{
		printf("Error: Empty file in %s\n", dirData);
		exit (1);
	}
	nData_X3--;

	cudaMallocManaged(&timeQt_X3, nData_X3*sizeof(float));
	dataQt_X3_raw = (float *) malloc(nData_X3*sizeof(float));

	fileRead = fopen(dirData, "r");
	if (fgets(renglon, sizeof(renglon), fileRead) == NULL) exit (1);

	nn = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;

		linea = strtok(renglon, ",");
		sscanf(linea, "%f", &auxFloat);
		timeQt_X3[nn] = auxFloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxFloat);
		dataQt_X3_raw[nn] = auxFloat;

		nn++;
	}
	fclose(fileRead);

	// Read qualitative data for X3 variable
	int qnData_X3;
	window *timeQl_X3, *dataQl_X3;

	sprintf(dirData, "qualX3cell.csv");
	fileRead = fopen(dirData, "r");

	qnData_X3 = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;
		qnData_X3++;
	}
	fclose(fileRead);

	if (qnData_X3 == 0)
	{
		printf("Error: Empty file in %s\n", dirData);
		exit (1);
	}
	qnData_X3--;

	cudaMallocManaged(&timeQl_X3, qnData_X3*sizeof(window));
	cudaMallocManaged(&dataQl_X3, qnData_X3*sizeof(window));

	fileRead = fopen(dirData, "r");
	if (fgets(renglon, sizeof(renglon), fileRead) == NULL) exit (1);

	nn = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;

		linea = strtok(renglon, ",");
		sscanf(linea, "%f", &auxFloat);
		timeQl_X3[nn].min = auxFloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxFloat);
		timeQl_X3[nn].max = auxFloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxFloat);
		dataQl_X3[nn].min = auxFloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxFloat);
		dataQl_X3[nn].max = auxFloat;

		nn++;
	}
	fclose(fileRead);

    	/*+*+*+*+*+ FETCH PARAMETERS +*+*+*+*+*/
	int Np, itMax, seed, D, bootFlag, rssFlag_X3, qFlag_X3;
	float Fm, Cr, t0, tN, dt;
	int err_flag = 0;

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	/* DE parameters */
	// Population of parameter vector
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &Np);

	// Maximum iterations
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &itMax);

	// Recombination probability
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &Cr);

	// Mutation factor
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &Fm);

	// Seed for random numbers
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &seed);

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	/* Initial conditions for ODE solve */
	// Initial time
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &t0);

	// Final time
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &tN);

	// Step time
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &dt);

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	/* Parameters to estimate */
	// Number of parameters to estimate
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &D);

	// Activate sampling for Bootstraping?
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &bootFlag);

	// Include qualitative fit?
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &qFlag_X3);

	// Include raw X3 data?
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &rssFlag_X3);

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	if (err_flag)
	{
		printf("Error: Something is wrong in the parameter file (.param)\n");
		exit (1);
	}

	param pars;

	pars.D = D;
	pars.t0 = t0;
	pars.tN = tN;
	pars.Np = Np;
	pars.dt = dt;
	pars.nData_X1 = nData_X1;
	pars.nData_X3 = nData_X3;
	pars.qnData_X3 = qnData_X3;
	pars.qFlag_X3 = qFlag_X3;
	if (!qFlag_X3) pars.rssFlag_X3 = rssFlag_X3;
        else pars.rssFlag_X3 = 0;

	// Initial values
        pars.X1_0 = 4.0;
        pars.X2_0 = 2.0;
        pars.X3_0 = 1.0;

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

	// Initialize random numbers
	if (seed < 0) seed *= -1;
	Ran ranUni(seed);

	// Initialize population
	for (jj=0; jj<D; jj++)
	{
		aux = upperLim[jj] - lowerLim[jj];
		for (ii=0; ii<Np; ii++)
		{
			idx = ii*D + jj;
			if (aux == 0.0) pop[idx] = lowerLim[jj];
			else pop[idx] = lowerLim[jj] + aux*ranUni.doub();
		}
	}

	// Sampling for bootstrap
	float *dataQt_X1;
	cudaMallocManaged(&dataQt_X1, nData_X1*sizeof(float));

	float *dataQt_X3;
	cudaMallocManaged(&dataQt_X3, nData_X3*sizeof(float));

	if (bootFlag)
	{
		int auxInt;
		int tt;
		float oldTime;

		tt = 0;
		oldTime = timeQt_X1[0];
		for (nn=1; nn<=nData_X1; nn++)
		{
			tt++;

			if (nn < nData_X1) if (oldTime == timeQt_X1[nn]) continue;

			for (ii=nn-tt; ii<nn; ii++)
			{
				auxInt = tt*ranUni.doub();
				// Using modulo for safety (auxInt!=tt)
				dataQt_X1[ii] = dataQt_X1_raw[nn-tt+(auxInt%tt)];
			}

			if (nn == nData_X1) break;
			
			tt = 0;
			oldTime = timeQt_X1[nn];
		}

		tt = 0;
		oldTime = timeQt_X3[0];
		for (nn=1; nn<=nData_X3; nn++)
		{
			tt++;

			if (nn < nData_X3) if (oldTime == timeQt_X3[nn]) continue;

			for (ii=nn-tt; ii<nn; ii++)
			{
				auxInt = tt*ranUni.doub();
				// Using modulo for safety (auxInt!=tt)
				dataQt_X3[ii] = dataQt_X3_raw[nn-tt+(auxInt%tt)];
			}

			if (nn == nData_X3) break;
			
			tt = 0;
			oldTime = timeQt_X3[nn];
		}
	}
	else
	{
		for (nn=0; nn<nData_X1; nn++) dataQt_X1[nn] = dataQt_X1_raw[nn];
		for (nn=0; nn<nData_X3; nn++) dataQt_X3[nn] = dataQt_X3_raw[nn];
	}
	free(dataQt_X1_raw);
	free(dataQt_X3_raw);

	int ths, blks;
	float *costFn, *d_newCostFn;

	cudaMallocManaged(&costFn, Np*sizeof(float));
	cudaMalloc(&d_newCostFn, Np*sizeof(float));

	// Estimate the number of threads and blocks for the GPU
	ths = (Np < THS_MAX) ? nextPow2(Np) : THS_MAX;
	blks = 1 + (Np - 1)/ths;

	// Calcula el valor de la función objetivo
	costFunction<<<blks, ths>>>(pars, pop, timeQt_X1, dataQt_X1, timeQt_X3, dataQt_X3,
			timeQl_X3, dataQl_X3, costFn);
	cudaDeviceSynchronize();

    	/*+*+*+*+*+ START OPTIMIZATION +*+*+*+*+*/
	int it, xx, yy, zz;
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

	// Start iterations
	for (it=0; it<itMax; it++)
	{
		//minVal = costFn[0];
		//iiMin = 0;
		//for(ii=1; ii<Np; ii++) if (minVal > costFn[ii])
		//{
		//	minVal = costFn[ii];
		//	iiMin = ii;
		//}

		//xx = iiMin; // best
		for (ii=0; ii<Np; ii++)
		{
			do xx = Np*ranUni.doub(); while (xx == ii || xx == Np);
			do yy = Np*ranUni.doub(); while (yy == ii || yy == xx || yy == Np);
			do zz = Np*ranUni.doub(); while (zz == ii || zz == yy || zz == xx || zz == Np);

			iiMut[ii].x = xx; iiMut[ii].y = yy; iiMut[ii].z = zz;
		}

		// Generate random numbers and then update positions
		curandGenerateUniform(gen, d_randUni, Np*D);

		// Generate new population
		newPopulation<<<blks, ths>>>(Np, D, Cr, Fm, d_randUni, iiMut,
				lowerLim, upperLim, pop, d_newPop);

		// Calculate cost function values
		costFunction<<<blks, ths>>>(pars, d_newPop, timeQt_X1, dataQt_X1, timeQt_X3, dataQt_X3,
				timeQl_X3, dataQl_X3, d_newCostFn);

		// Select the best vectors between new ones and old ones
		selection<<<blks, ths>>>(Np, D, pop, d_newPop, costFn, d_newCostFn);

		cudaDeviceSynchronize();
	}


	// Minimum of the population
	minVal = costFn[0];
	iiMin = 0;
	for (ii=1; ii<Np; ii++) if (minVal > costFn[ii])
	{
		minVal = costFn[ii];
		iiMin = ii;
	}

	FILE *fBestPars;
	fBestPars = fopen("bestPars.dat", "a");
	for (jj=0; jj<D; jj++) fprintf(fBestPars, "%.4e ", pop[iiMin*D + jj]);
	fprintf(fBestPars, "%.4e\n", minVal);
	fclose(fBestPars);

	printf("FINISHED\n");

	cudaFree(timeQt_X1);
	cudaFree(timeQt_X3);
	cudaFree(timeQl_X3);
	cudaFree(lowerLim);
	cudaFree(upperLim);
	cudaFree(dataQt_X1);
	cudaFree(dataQt_X3);
	cudaFree(dataQl_X3);
	cudaFree(iiMut);
	cudaFree(pop);
	cudaFree(d_newPop);
	cudaFree(costFn);
	cudaFree(d_newCostFn);
	cudaFree(d_randUni);
	curandDestroyGenerator(gen);

	exit (0);
}
