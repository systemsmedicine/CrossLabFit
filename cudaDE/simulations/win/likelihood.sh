#!/bin/bash

### Script to get likelihood profiles ###

# Define the path to your executable and parameter file
executable_path="/home/rodolfo/Proyectos/crossLabFit/codes/CrossLabFit/cudaDE/qDEcode_influenza/qDE"
param_file="influenza.param"
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
    "V0:2.00:0.08:6.00"      	# V0 from 1e2 to 1e6
    "beta:-8.00:0.08:-4.00"  	# beta from 1e-8 to 1e-4
    "delI:-8.00:0.08:-4.00" 	# del_I similar to beta
    "p:-2.00:0.08:2.00"       	# p from 1e-2 to 1e2
    "c:-2.00:0.08:2.00"       	# c similar to p
    "r:-8.00:0.08:-4.00"   	# r similar to beta
    "delT:-2.00:0.08:2.00"  	# del_T similar to p
)

# Parameters to skip
declare -A skip_params
skip_params[V0]=1
skip_params[delT]=1
skip_params[c]=1
skip_params[delI]=1
skip_params[p]=1
skip_params[r]=1

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

		mv bestPars.dat ${param_name}_profile_ql.dat
	fi

	line=$((line+1))
done

echo "All simulations completed."
