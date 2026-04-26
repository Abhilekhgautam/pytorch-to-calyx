import random
import struct
import subprocess
import sys
import os

NUM_TEST_BATCHES = 10
SAMPLES_PER_BATCH = 300
VERILATOR_EXEC = "./obj_dir/Vmain"
VERILOG_OUTPUT_FILE = "data/mem_2.out" 

def float_to_hex32(f):
    """Converts a Python float to an IEEE 754 32-bit hex string"""
    return struct.pack('>f', f).hex()

def generate_data(num_samples):
    """Generates new random f32 inputs and expected ReLU outputs"""
    inputs = [random.uniform(-1000000.0, 100000.0) for _ in range(num_samples)]
    expected_outputs = [x if x > 0.0 else 0.0 for x in inputs]
    
    with open("data/mem_0.dat", "w") as f_in, open("expected.dat", "w") as f_out:
        for i in range(num_samples):
            f_in.write(f"{float_to_hex32(inputs[i])}\n")
            f_out.write(f"{float_to_hex32(expected_outputs[i])}\n")

def compare_results(batch_idx):
    """Compares the Verilog output file with the Expected output file"""
    if not os.path.exists(VERILOG_OUTPUT_FILE):
        print(f"Error: Verilog did not produce '{VERILOG_OUTPUT_FILE}'.")
        return False

    with open("expected.dat", "r") as f_exp, open(VERILOG_OUTPUT_FILE, "r") as f_act:
        expected_lines = f_exp.read().splitlines()
        actual_lines = f_act.read().splitlines()

    if len(expected_lines) != len(actual_lines):
        print(f"Error: Line count mismatch. Expected {len(expected_lines)}, got {len(actual_lines)}")
        return False

    passed = True
    for i, (exp, act) in enumerate(zip(expected_lines, actual_lines)):
        if exp.strip().lower() != act.strip().lower():
            print(f"  -> Mismatch at index {i} | Expected: {exp} | Actual: {act}")
            passed = False

    return passed

def run_automated_tests():
    if not os.path.exists(VERILATOR_EXEC):
        print(f"Error: Could not find '{VERILATOR_EXEC}'. Did you compile?")
        sys.exit(1)

    # Make sure the data directory exists!
    os.makedirs("data", exist_ok=True)

    print(f" Starting Automated Verification...")
    passed_batches = 0

    for batch_idx in range(1, NUM_TEST_BATCHES + 1):
        if os.path.exists(VERILOG_OUTPUT_FILE):
            os.remove(VERILOG_OUTPUT_FILE)

        generate_data(SAMPLES_PER_BATCH)
        
        # Pass SAMPLES_PER_BATCH as a command line argument to the C++ testbench
        result = subprocess.run([VERILATOR_EXEC, str(SAMPLES_PER_BATCH)], capture_output=True, text=True)
        
        # Uncomment this if you want to see C++ print statements for debugging
        if result.stdout: print(result.stdout)
        
        if compare_results(batch_idx):
            print(f"Batch {batch_idx:02d}/{NUM_TEST_BATCHES} :  PASS")
            passed_batches += 1
        else:
            print(f"Batch {batch_idx:02d}/{NUM_TEST_BATCHES} :  FAIL")
            break # Stop testing on first failure

    print("\n" + "="*30)
    if passed_batches == NUM_TEST_BATCHES:
        print(" ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print(f" VERIFICATION FAILED! ({passed_batches}/{NUM_TEST_BATCHES} passed)")
        sys.exit(1)

if __name__ == "__main__":
    run_automated_tests()
