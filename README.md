# qualitativeDE

This code uses the Diferential Evolution method GPU parallelized and a 5th Runge-Kutta to solve ODE, furthermore, utilize contraints from another data in a qualitative way.

To run using different random numbers:
ran=$RANDOM; ./qDE < params.param | sed 's/SEED/'$ran'/'

To run several times (parameter distributions):
for ((i=0; i<100; i++)); do echo $i; ran=$RANDOM;  sed -e 's/SEED/'$ran'/' params.param > tmp.param; ../../qDE < tmp.param; done 

To get likelihood profiles:
for ((i=5e-2; i<=5e-1; i+=0.009)); do echo $i; ran=$RANDOM;  sed -e 's/SEED/'$ran'/' -e 's/VAL/'$i'/g' params.param > tmp.param; ./qDE < tmp.param; done
