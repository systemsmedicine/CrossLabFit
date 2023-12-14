# qualitativeDE

This code uses the Diferential Evolution method GPU parallelized and a 5th Runge-Kutta to solve ODE, furthermore, utilize contraints from another data in a qualitative way.

To run using different random numbers:
ran=$RANDOM; ./qDE < params.param | sed 's/SEED/'$ran'/'

To run several times (parameter distributions):
for ((i=0; i<100; i++)); do echo $i; ran=$RANDOM;  sed -e 's/SEED/'$ran'/' params.param > tmp.param; ../../qDE < tmp.param; done 

To get likelihood profiles:
for ((i=55; i<=305; i+=5)); do echo $(bc<<<"$i*0.001"); ran=$RANDOM;  sed -e 's/SEED/'$ran'/' -e 's/VAL/'$(bc<<<"$i*0.001")'/g' params.param > tmp.param; ../../qDE < tmp.param; done
for ((i=35; i<=285; i+=5)); do echo $(bc<<<"$i*0.001"); ran=$RANDOM;  sed -e 's/SEED/'$ran'/' -e 's/VAL/'$(bc<<<"$i*0.001")'/g' params.param > tmp.param; ../../qDE < tmp.param; done
for ((i=25; i<=275; i+=5)); do echo $(bc<<<"$i*0.001"); ran=$RANDOM;  sed -e 's/SEED/'$ran'/' -e 's/VAL/'$(bc<<<"$i*0.001")'/g' params.param > tmp.param; ../../qDE < tmp.param; done
for ((i=10; i<=210; i+=4)); do echo $(bc<<<"$i*0.001"); ran=$RANDOM;  sed -e 's/SEED/'$ran'/' -e 's/VAL/'$(bc<<<"$i*0.001")'/g' params.param > tmp.param; ../../qDE < tmp.param; done
for ((i=20; i<=220; i+=4)); do echo $(bc<<<"$i*0.001"); ran=$RANDOM;  sed -e 's/SEED/'$ran'/' -e 's/VAL/'$(bc<<<"$i*0.001")'/g' params.param > tmp.param; ../../qDE < tmp.param; done
for ((i=75; i<=325; i+=5)); do echo $(bc<<<"$i*0.0001"); ran=$RANDOM;  sed -e 's/SEED/'$ran'/' -e 's/VAL/'$(bc<<<"$i*0.0001")'/g' params.param > tmp.param; ../../qDE < tmp.param; done
