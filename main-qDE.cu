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
	float X;
	float Y;
} 
comp;

typedef struct 
{
	float X0;
	float Y0;

	float alp;
	float bet;
	float del;
	float gam;

	float t0;
	float tN;
	float dt;
	int D;
	int Np;
	int nData;
	int qnData;
	int qflag;
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

__device__ void derivs(int idx, param pars, float *pop, comp Z, comp *dotZ)
{
	int ii = 0;
	float alp = pop[idx + ii];
	ii++;
	float bet = pop[idx + ii];
	ii++;
	float del = pop[idx + ii];
	ii++;
	float gam = pop[idx + ii];

	dotZ->X = alp*Z.X - bet*Z.X*Z.Y;
	dotZ->Y = del*Z.X*Z.Y - gam*Z.Y;

	return;
}

//-------------------------------------------------------------------------------
__global__ void costFunction(param pars, float *pop, float *timeData, float *dataPrey,
		float *qtime, float *qData, float *valCostFn)
{
	int ind;

	ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind >= pars.Np) return;

	int idx;
	float t0, tN, tt;
	comp Z, dotZ;

	idx = ind*pars.D;
	t0 = pars.t0;
	tN = pars.tN;

	// Initial values
	Z.X = pars.X0;
	Z.Y = pars.Y0;

	derivs(idx, pars, pop, Z, &dotZ);

	// ODE solver (5th-order Dormand-Prince)

	comp ztemp, k2, k3, k4, k5, k6, dotZnew, zOut;
	float aux;
	int nn, qnn;
	float h;

	float ttData, sum2, qtt; 
	int nData, qnData, sizeSample, ii, idxData;
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
	sizeSample = pars.sizeSample;
	flag = 0;
	qflag = pars.qflag == 0 ? 1 : 0; // If qflag is off, then set up to 1 to skip it

	do
	{
		ztemp.X = Z.X + h*A21*dotZ.X;
		ztemp.Y = Z.Y + h*A21*dotZ.Y;

		derivs(idx, pars, pop, ztemp, &k2);

		ztemp.X = Z.X + h*(A31*dotZ.X + A32*k2.X);
		ztemp.Y = Z.Y + h*(A31*dotZ.Y + A32*k2.Y);

		derivs(idx, pars, pop, ztemp, &k3);

		ztemp.X = Z.X + h*(A41*dotZ.X + A42*k2.X + A43*k3.X);
		ztemp.Y = Z.Y + h*(A41*dotZ.Y + A42*k2.Y + A43*k3.Y);

		derivs(idx, pars, pop, ztemp, &k4);

		ztemp.X = Z.X + h*(A51*dotZ.X + A52*k2.X + A53*k3.X + A54*k4.X);
		ztemp.Y = Z.Y + h*(A51*dotZ.Y + A52*k2.Y + A53*k3.Y + A54*k4.Y);

		derivs(idx, pars, pop, ztemp, &k5);

		ztemp.X = Z.X + h*(A61*dotZ.X + A62*k2.X + A63*k3.X + A64*k4.X + A65*k5.X);
		ztemp.Y = Z.Y + h*(A61*dotZ.Y + A62*k2.Y + A63*k3.Y + A64*k4.Y + A65*k5.Y);

		derivs(idx, pars, pop, ztemp, &k6);

		zOut.X = Z.X + h*(A71*dotZ.X + A73*k3.X + A74*k4.X + A75*k5.X + A76*k6.X);
		zOut.Y = Z.Y + h*(A71*dotZ.Y + A73*k3.Y + A74*k4.Y + A75*k5.Y + A76*k6.Y);

		derivs(idx, pars, pop, zOut, &dotZnew);

		nanFlag = 0;
		if (isnan(zOut.X)) nanFlag = 1;
		if (isnan(zOut.Y)) nanFlag = 1;
		if (nanFlag) break;

	        if (zOut.X < 0.0) zOut.X = 0.0;
	        if (zOut.Y < 0.0) zOut.Y = 0.0;

		tt += h;

		// This part calculates the RMS
		if (tt > ttData && !flag)
		{
			for (ii=0; ii<sizeSample; ii++)
			{
				idxData = ii + nn*sizeSample;
				aux = dataPrey[idxData] - zOut.X;
				sum2 += aux*aux;
			}

			nn++;
			if (nn >= nData) flag = 1;
			if (!flag) ttData = timeData[nn];
		}

		// This calculates the qualitative part
		if (tt > qtt - 2 && !qflag)
		{
			aux = qData[qnn] - zOut.Y;
			if (aux < 0.0) aux *= -1;
			if (aux > 4)
			{
				nanFlag = 1;
				break;
			}
			
			if (tt >= qtt + 2)
			{
				qnn++;
				if (qnn >= qnData) qflag = 1;
				if (!qflag) qtt = qtime[qnn];
			}
		}

		if (flag && qflag) break;

		dotZ = dotZnew;
		Z = zOut;
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
	int nData, qnData, nn;
	float auxfloat;
	float *timeData, *dataPrey;
	float *qtime, *qData;
	char renglon[200], dirData[500], *linea;
	FILE *fileRead;

	sprintf(dirData, "pyNotebooks/LVdata.data");
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
	qnData = nData;

	cudaMallocManaged(&timeData, nData*sizeof(float));
	cudaMallocManaged(&dataPrey, nData*sizeof(float));
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
		timeData[nn] = auxfloat;
		qtime[nn] = auxfloat;

		linea = strtok(NULL, " ");
		sscanf(linea, "%f", &auxfloat);
		dataPrey[nn] = auxfloat;

		linea = strtok(NULL, " ");
		sscanf(linea, "%f", &auxfloat);
		qData[nn] = auxfloat;

		nn++;
	}
	fclose(fileRead);


    	/*+*+*+*+*+ DIFERENTIAL EVOLUTION +*+*+*+*+*/
	int Np, itMax, seed, D, qflag;
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

	// Include qualitative fit?
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &qflag);

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

	// Initial values
        pars.X0 = 10.0;
        pars.Y0 = 10.0;

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
	//Normaldev ranNorm(0.0, 1.0, seed); // Standard dev (Z)

	int sizeSample = 1;
	pars.sizeSample = sizeSample;

	// Generate random data in normal distribution
	//float *dataN;
	//cudaMallocManaged(&dataN, sizeSample*nData*sizeof(float));

	// Linear transformation from Z to normal dev X
	// Z = (X - meanX) / stdX -> X = Z*stdX + meanX
	//for (ii=0; ii<nData; ii++)
	//	for (jj=0; jj<sizeSample; jj++)
	//	{
	//		idx = jj + ii*sizeSample;
	//		dataN[idx] = meanN[ii] + stdN[ii]*ranNorm.dev();
	//	}

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
	costFunction<<<blks, ths>>>(pars, pop, timeData, dataPrey, qtime, qData, valCostFn);
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
		costFunction<<<blks, ths>>>(pars, d_newPop, timeData, dataPrey, qtime, qData, d_newValCostFn);

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
	fPar = fopen("bestPars.dat", "w");
	if (valCostFn[iiMin] < 10)
	{
		//fprintf(fPar, "#BestPar: RMS = %e\n", minVal);
		for (jj=0; jj<D-1; jj++) fprintf(fPar, "%.4e\t", pop[iiMin*D + jj]);
		fprintf(fPar, "%.4e\n", pop[iiMin*D + D-1]);
	}
	fclose(fPar);

	printf("FINISHED\n");

	cudaFree(timeData);
	cudaFree(qtime);
	cudaFree(lowerLim);
	cudaFree(upperLim);
	cudaFree(dataPrey);
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
