#!/bin/bash

module purge
module load GCC/14.2.0 OpenMPI/5.0.7-GCC-14.2.0

source /home6/s6699804/zn_case/.venv_zn_scripts/bin/activate

LAMMPS_EXEC="/home6/s6699804/zn_case/lammps/build/lmp"

# Cleaning old data
# rm -rf log/* sfe_data/basal/* sfe_data/prism/*
# shopt -s extglob # Enables complex pattern amtching like !(file) syntax for "except file"
# rm -rf sfe_data/pyramidal_I/!(sfe.lmp)
# rm -rf sfe_data/pyramidal_II/!(sfe.lmp)
# rm -rf sfe_data/pyramidal_II_stable/!(sfe.lmp)
# shopt -u extglob # Disable the extension
# rm -f basal.csv prism.csv potline_bench_output.yaml log.lammps restart.equil

# Calculating lattice parameters
srun $LAMMPS_EXEC -in in.lattice

# Extracting lattice parameters for next simulations
A_LATTICE=$(awk '/^a_lattice:/ {print $2}' potline_bench_output.yaml)
C_LATTICE=$(awk '/^c_lattice:/ {print $2}' potline_bench_output.yaml)

# Calculating vacancy energy
srun $LAMMPS_EXEC -var a "$A_LATTICE" -var c "$C_LATTICE" -in in.vac

# Calculating elastic parameters
srun $LAMMPS_EXEC -var a "$A_LATTICE" -var c "$C_LATTICE" -in in.elastic

# Calculating TS curves
srun $LAMMPS_EXEC -var a "$A_LATTICE" -var c "$C_LATTICE" -in in.basal
srun $LAMMPS_EXEC -var a "$A_LATTICE" -var c "$C_LATTICE" -in in.prism

# Calculating SFE curves
# Basal plane
OUTPUT_DIR="sfe_data/basal"
DUMP_DIR="$OUTPUT_DIR/dumps"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$DUMP_DIR"

for i in {0..0}
do
    for j in {0..39}
    do
        OUTPUT_FILE=$(printf "%s/%02d_%02d_data" "$OUTPUT_DIR" $i $j)

        srun $LAMMPS_EXEC -var x_d $i -var y_d $j -var dump_dir "$DUMP_DIR" < in.gsfe_basal > "$OUTPUT_FILE"
        
    done
done

grep "SFE" "$OUTPUT_DIR"/*_data > "$OUTPUT_DIR/sfe.txt"

# Prismatic plane
OUTPUT_DIR="sfe_data/prism"
DUMP_DIR="$OUTPUT_DIR/dumps"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$DUMP_DIR"

for i in {0..19}
do
    for j in {0..0}
    do
        OUTPUT_FILE=$(printf "%s/%02d_%02d_data" "$OUTPUT_DIR" $i $j)

        srun $LAMMPS_EXEC -var x_d $i -var y_d $j -var dump_dir "$DUMP_DIR" < in.gsfe_prism > "$OUTPUT_FILE"
        
    done
done

grep "SFE" "$OUTPUT_DIR"/*_data > "$OUTPUT_DIR/sfe.txt"

# Pyramidal I plane
OUTPUT_DIR="sfe_data/pyramidal_I"
DUMP_DIR="$OUTPUT_DIR/dumps"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$DUMP_DIR"

for i in {0..19}
do
    for j in {0..0}
    do
        OUTPUT_FILE=$(printf "%s/%02d_%02d_data" "$OUTPUT_DIR" $i $j)

        srun $LAMMPS_EXEC -var x_d $i -var y_d $j -var dump_dir "$DUMP_DIR" -var data_dir "$OUTPUT_DIR" < in.gsfe_pyrI > "$OUTPUT_FILE"
        
    done
done

grep "SFE" "$OUTPUT_DIR"/*_data > "$OUTPUT_DIR/sfe.txt"

# Pyramidal II plane
OUTPUT_DIR="sfe_data/pyramidal_II"
DUMP_DIR="$OUTPUT_DIR/dumps"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$DUMP_DIR"

for i in {0..19}
do
    for j in {0..0}
    do
        OUTPUT_FILE=$(printf "%s/%02d_%02d_data" "$OUTPUT_DIR" $i $j)

        srun $LAMMPS_EXEC -var x_d $i -var y_d $j -var dump_dir "$DUMP_DIR" -var data_dir "$OUTPUT_DIR" < in.gsfe_pyrII > "$OUTPUT_FILE"
        
    done
done

grep "SFE" "$OUTPUT_DIR"/*_data > "$OUTPUT_DIR/sfe.txt"

# Pyramidal II plane with all degrees of freedom relaxed
OUTPUT_DIR="sfe_data/pyramidal_II_stable"
DUMP_DIR="$OUTPUT_DIR/dumps"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$DUMP_DIR"

for i in {0..19}
do
    for j in {0..0}
    do
        OUTPUT_FILE=$(printf "%s/%02d_%02d_data" "$OUTPUT_DIR" $i $j)

        srun $LAMMPS_EXEC -var x_d $i -var y_d $j -var dump_dir "$DUMP_DIR" -var data_dir "$OUTPUT_DIR" < in.gsfe_pyrII_stable > "$OUTPUT_FILE"
        
    done
done

grep "SFE" "$OUTPUT_DIR"/*_data > "$OUTPUT_DIR/sfe.txt"
# Getting the minimum
python extract_sfe_min.py

# Getting inference time and number of cores
echo "cores: ${SLURM_NTASKS}" >> potline_bench_output.yaml
srun $LAMMPS_EXEC -in in.dynamic
python get_inference_time.py

# Running the data collector
python zn_data_collect.py
