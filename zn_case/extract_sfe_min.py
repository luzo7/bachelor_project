def main():
    input_file = "sfe_data/pyramidal_II_stable/sfe.txt"
    output_file = "model_info.yaml"

    values = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if 'SFE=' in line:
                try:
                    value_str = line.split('SFE=')[1].strip()
                    value = float(value_str)
                    if value > 1e-2:
                        values.append(value)
                except ValueError:
                    continue
    
    if values:
        minimum = min(values)
        with open(output_file, 'a') as f:
            f.write(f"sfe_dof_relaxation_minimum: {minimum}\n")


if __name__ == "__main__":
    main()