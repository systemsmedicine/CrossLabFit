#!/bin/zsh

### Script to get parameter distribution using bootstraping ###

# Define the path to your executable and parameter file
executable_path="PATH_TO_EXECUTABLE/CrossLabFit/cudaDE/qDEcode_LV/qDE"
param_file="cycle.param"

# Function to run simulation with modified parameters
run_simulation() {
    local template=$1

    # Replace SEED with a random number 
    local rand_seed=$RANDOM
    sed -e "s/SEED/$rand_seed/" $template > tmp.param

    # Run the simulation
    $executable_path < tmp.param
}

# Number of runs
N=100

for ((i=0; i<N; i++)); do
    echo "Run $i"
	run_simulation $param_file
done

