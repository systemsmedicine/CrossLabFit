#!/bin/bash

### Script to get likelihood profiles ###

# Define the path to your executable and parameter file
executable_path="PATH_TO_EXECUTABLE/CrossLabFit/cudaDE/qDEcode_LV/qDE"
param_file="cycle.param"
template_file="template.param"

# Function to run simulation with modified parameters
run_simulation() {
    local param_value=$1
    local template=$2

    # Replace SEED with a random number and parameter with a specific value
    local rand_seed=$RANDOM
    sed -e "s/SEED/$rand_seed/" -e "s/VAL/$param_value/g" $template > tmp.param

    # Run the simulation
    $executable_path < tmp.param
}

# Parameters and their respective ranges
# Syntax: "param_name:start_value:increment:end_value"
declare -a param_settings=(
    "a0:0.05:0.08:0.30" 
    "a1:0.05:0.08:0.30" 
    "a2:0.06:0.004:0.26" 
    "a3:0.05:0.08:0.30" 
    "a4:0.05:0.08:0.30" 
    "a5:0.05:0.08:0.30" 
    "a6:0.02:0.004:0.22" 
    "a7:0.05:0.08:0.30" 
    "a8:0.05:0.08:0.30" 
    "a9:0.05:0.08:0.30" 
)

# Parameters to skip
declare -A skip_params
skip_params[a0]=1
skip_params[a1]=1
skip_params[a3]=1
skip_params[a4]=1
skip_params[a5]=1
skip_params[a7]=1
skip_params[a8]=1
skip_params[a9]=1

# Iterate over each parameter and their defined range
line=20		# Start line of the first parameter range
for param_info in $param_settings
do
	IFS=':' read param_name start increment end <<< $param_info
	if [[ -z ${skip_params[$param_name]} ]]; then  # Check if param_name is not in skip_params
        # Modify parameter line in param file
		sed "${line}s/.*/[VAL : VAL]  # $param_name/" $param_file > $template_file

		# Calculate values for each parameter step
		for exponent in $(seq $start $increment $end)
		do
			echo "Running simulation for $param_name with value $exponent"
			run_simulation $exponent $template_file
		done

		mv bestPars.dat ${param_name}Profile.dat
	fi

	line=$((line+1))
done

echo "All simulations completed."
