# qualitativeDE

This code uses the Diferential Evolution method GPU parallelized and a 5th Runge-Kutta to solve ODE, furthermore, utilize contraints from another data in a qualitative way.

To run using different random numbers:
ran=$RANDOM; ./qDE < params.param | sed 's/SEED/'$ran'/'
