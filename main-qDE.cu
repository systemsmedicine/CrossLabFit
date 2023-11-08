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
	float X1;
	float X2;
	float X3;
} 
comp;

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
	float windowTime;
	float windowVal;
	int D;
	int Np;
	int nData;
	int qnData;
	int qflag;
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
	int ii = 0;
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

	dotY->X1 = a1*Y.X1 - a2*Y.X1*Y.X2;
	dotY->X2 = a3*Y.X1*Y.X2 - a4*Y.X2 - a5*Y.X2*Y.X3;
	dotY->X3 = a6*Y.X2*Y.X3 - a7*Y.X3;

	return;
}

//-------------------------------------------------------------------------------
__global__ void costFunction(param pars, float *pop, float *timeData, float *dataX1,
		float *qtime, float *qData, float *valCostFn)
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
	Y.X1 = pars.X1_0;
	Y.X2 = pars.X2_0;
	Y.X3 = pars.X3_0;

	derivs(idx, pars, pop, Y, &dotY);

	// ODE solver (5th-order Dormand-Prince)

	comp ytemp, k2, k3, k4, k5, k6, dotYnew, yOut;
	float aux;
	int nn, qnn;
	float h;

	float ttData, sum2, qtt; 
	float windowTime, windowVal; 
	int nData, qnData;
	short nanFlag, flag, qflag;

	tt = t0;
	h = pars.dt;

	nn = 0;
	qnn = 0;
	ttData = timeData[0];
	qtt = qtime[0];
	sum2 = 0.0;
	nData = pars.nData;
	qnData = pars.qnData;
	flag = 0;
	qflag = pars.qflag == 0 ? 1 : 0; // If qflag is off, then set up to 1 to skip it
	windowTime = pars.windowTime;
	windowVal = pars.windowVal;

	do
	{
		ytemp.X1 = Y.X1 + h*A21*dotY.X1;
		ytemp.X2 = Y.X2 + h*A21*dotY.X2;
		ytemp.X3 = Y.X3 + h*A21*dotY.X3;

		derivs(idx, pars, pop, ytemp, &k2);

		ytemp.X1 = Y.X1 + h*(A31*dotY.X1 + A32*k2.X1);
		ytemp.X2 = Y.X2 + h*(A31*dotY.X2 + A32*k2.X2);
		ytemp.X3 = Y.X3 + h*(A31*dotY.X3 + A32*k2.X3);

		derivs(idx, pars, pop, ytemp, &k3);

		ytemp.X1 = Y.X1 + h*(A41*dotY.X1 + A42*k2.X1 + A43*k3.X1);
		ytemp.X2 = Y.X2 + h*(A41*dotY.X2 + A42*k2.X2 + A43*k3.X2);
		ytemp.X3 = Y.X3 + h*(A41*dotY.X3 + A42*k2.X3 + A43*k3.X3);

		derivs(idx, pars, pop, ytemp, &k4);

		ytemp.X1 = Y.X1 + h*(A51*dotY.X1 + A52*k2.X1 + A53*k3.X1 + A54*k4.X1);
		ytemp.X2 = Y.X2 + h*(A51*dotY.X2 + A52*k2.X2 + A53*k3.X2 + A54*k4.X2);
		ytemp.X3 = Y.X3 + h*(A51*dotY.X3 + A52*k2.X3 + A53*k3.X3 + A54*k4.X3);

		derivs(idx, pars, pop, ytemp, &k5);

		ytemp.X1 = Y.X1 + h*(A61*dotY.X1 + A62*k2.X1 + A63*k3.X1 + A64*k4.X1 + A65*k5.X1);
		ytemp.X2 = Y.X2 + h*(A61*dotY.X2 + A62*k2.X2 + A63*k3.X2 + A64*k4.X2 + A65*k5.X2);
		ytemp.X3 = Y.X3 + h*(A61*dotY.X3 + A62*k2.X3 + A63*k3.X3 + A64*k4.X3 + A65*k5.X3);

		derivs(idx, pars, pop, ytemp, &k6);

		yOut.X1 = Y.X1 + h*(A71*dotY.X1 + A73*k3.X1 + A74*k4.X1 + A75*k5.X1 + A76*k6.X1);
		yOut.X2 = Y.X2 + h*(A71*dotY.X2 + A73*k3.X2 + A74*k4.X2 + A75*k5.X2 + A76*k6.X2);
		yOut.X3 = Y.X3 + h*(A71*dotY.X3 + A73*k3.X3 + A74*k4.X3 + A75*k5.X3 + A76*k6.X3);

		derivs(idx, pars, pop, yOut, &dotYnew);

		nanFlag = 0;
		if (isnan(yOut.X1)) nanFlag = 1;
		if (isnan(yOut.X2)) nanFlag = 1;
		if (isnan(yOut.X3)) nanFlag = 1;
		if (nanFlag) break;

	        //if (yOut.X1 < 0.0) yOut.X1 = 0.0;
	        //if (yOut.X2 < 0.0) yOut.X2 = 0.0;
	        //if (yOut.X3 < 0.0) yOut.X3 = 0.0;

		tt += h;

		// This part calculates the RMS
		if (tt > ttData && !flag)
		{
			while (1)
			{
				aux = dataX1[nn] - yOut.X1;
				sum2 += aux*aux;
				nn++;
				if (nn >= nData)
				{
					flag = 1;
					break;
				}
				if (!flag)
				{
					aux = timeData[nn];
					if (aux != ttData)
					{
						ttData = aux;
						break;
					}
				}
			}
		}

		// This calculates the qualitative part
		if (tt > qtt - windowTime/2.0 && !qflag)
		{
			aux = qData[qnn] - (yOut.X3-pars.X3_0)/(3.5-pars.X3_0);
			if (aux < 0.0) aux *= -1;
			if (aux < windowVal/2.0)
			{
				qnn++;
				if (qnn >= qnData) qflag = 1;
				else qtt = qtime[qnn];
			}
			else if (tt >= qtt + windowTime/2.0)
			{
				nanFlag = 1;
				break;
			}
		}

		if (flag && qflag) break;

		dotY = dotYnew;
		Y = yOut;
	}
	while (tt <= tN);

	valCostFn[ind] = nanFlag ? 1e10 : sqrt(sum2/nData);

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
	int nData, qnData, nn;
	float auxfloat;
	float *timeData, *dataX1;
	float *qtime, *qData;
	char renglon[200], dirData[500], *linea;
	FILE *fileRead;

	sprintf(dirData, "pyNotebooks/LVdata_clean.data");
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
	cudaMallocManaged(&dataX1, nData*sizeof(float));

	fileRead = fopen(dirData, "r");
	if (fgets(renglon, sizeof(renglon), fileRead) == NULL) exit (1);

	nn = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;

		linea = strtok(renglon, " ");
		sscanf(linea, "%f", &auxfloat);
		timeData[nn] = auxfloat;

		linea = strtok(NULL, " ");
		sscanf(linea, "%f", &auxfloat);
		dataX1[nn] = auxfloat;

		nn++;
	}
	fclose(fileRead);

	sprintf(dirData, "pyNotebooks/LVdata_qual.data");
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
		printf("Error in qualitative data\n");
		exit (1);
	}
	nData--;
	qnData = nData;

	cudaMallocManaged(&qtime, qnData*sizeof(float));
	cudaMallocManaged(&qData, qnData*sizeof(float));

	fileRead = fopen(dirData, "r");
	if (fgets(renglon, sizeof(renglon), fileRead) == NULL) exit (1);

	nn = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;

		linea = strtok(renglon, " ");
		sscanf(linea, "%f", &auxfloat);
		qtime[nn] = auxfloat;

		linea = strtok(NULL, " ");
		sscanf(linea, "%f", &auxfloat);
		qData[nn] = auxfloat;

		nn++;
	}
	fclose(fileRead);

    	/*+*+*+*+*+ DIFERENTIAL EVOLUTION +*+*+*+*+*/
	int Np, itMax, seed, D, qflag;
	float Fm, Cr, t0, tN, dt, windowTime, windowVal;
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

	// Include qualitative fit?
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &qflag);

	// Window of time for qual
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &windowTime);

	// Window of value fraction
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &windowVal);

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	if (err_flag)
	{
		printf("Error en archivo de parámetros (.data)\n");
		exit (1);
	}

	param pars;

	pars.D = D;
	pars.t0 = t0;
	pars.tN = tN;
	pars.Np = Np;
	pars.dt = dt;
	pars.nData = nData;
	pars.qnData = qnData;
	pars.qflag = qflag;
	pars.windowTime = windowTime;
	pars.windowVal = windowVal;

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

	// Inicializa números aleatorios
	if (seed < 0) seed *= -1;
	Ran ranUni(seed);

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
	costFunction<<<blks, ths>>>(pars, pop, timeData, dataX1, qtime, qData, valCostFn);
	cudaDeviceSynchronize();

    	/*+*+*+*+*+ START OPTIMIZATION +*+*+*+*+*/
	FILE *fPars;
	fPars = fopen("pars.dat", "w");
	fprintf(fPars, "a1 a2 a3 a4 a5 a6 a7 RMSE it\n");

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

	// Start iterations
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
		costFunction<<<blks, ths>>>(pars, d_newPop, timeData, dataX1, qtime, qData, d_newValCostFn);

		// Selecciona el mejor vector y lo guarda en la poblacion "pop"
		selection<<<blks, ths>>>(Np, D, pop, d_newPop, valCostFn, d_newValCostFn);

		cudaDeviceSynchronize();

		// Save population for analysis
		if (!flag) for (ii=0; ii<Np; ii++)
		{
			//if (valCostFn[ii] == 1e10) continue;
			for(jj=0; jj<D; jj++) fprintf(fPars, "%.3e ", pop[ii*D + jj]);
			fprintf(fPars, "%.3e %d\n", valCostFn[ii], it);
		}
	}

	fclose(fPars);

	// Encuentra cual es el minimo de la pobalción
	minVal = valCostFn[0];
	iiMin = 0;
	for (ii=1; ii<Np; ii++) if (minVal > valCostFn[ii])
	{
		minVal = valCostFn[ii];
		iiMin = ii;
	}

	// Imprime el mejor vector de parámetros

	FILE *fBestPars;
	fBestPars = fopen("bestPars.dat", "a");
	//fprintf(fBestPasr, "%e\n", minVal);
	for (jj=0; jj<D-1; jj++) fprintf(fBestPars, "%.4e ", pop[iiMin*D + jj]);
	fprintf(fBestPars, "%.4e\n", pop[iiMin*D + D-1]);
	fclose(fBestPars);

	printf("FINISHED\n");

	cudaFree(timeData);
	cudaFree(qtime);
	cudaFree(lowerLim);
	cudaFree(upperLim);
	cudaFree(dataX1);
	cudaFree(qData);
	cudaFree(iiMut);
	cudaFree(pop);
	cudaFree(d_newPop);
	cudaFree(valCostFn);
	cudaFree(d_newValCostFn);
	cudaFree(d_randUni);
	curandDestroyGenerator(gen);

	exit (0);
}
