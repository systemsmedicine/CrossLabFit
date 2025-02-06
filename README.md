# Integrative Parameter Estimation

This code uses the Diferential Evolution method GPU parallelized and a 5th Runge-Kutta to solve ODE, furthermore, utilize contraints from "external" data in a qualitative way.

To compile use `make`

To run using different random numbers:
`ran=$RANDOM; ./qDE < params.param | sed 's/SEED/'$ran'/'`

To run several times (parameter distributions):
`for ((i=0; i<100; i++)); do echo $i; ran=$RANDOM;  sed -e 's/SEED/'$ran'/' params.param > tmp.param; ../../qDE < tmp.param; done`
