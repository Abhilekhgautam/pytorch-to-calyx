// created by void at 2026-04-26 22:39:37
#include "Vmain.h"
#include "verilated.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cstdint>
#include <cstdio>

#define MEM_SIZE 512

class CalyxMem {
public:
    uint32_t data[MEM_SIZE] = {0};
    uint32_t read_data_reg = 0;
    uint8_t done_reg = 0;          
    std::string name;
    bool debug = false;

    CalyxMem(std::string n) : name(n) {}

    void get_outputs(uint32_t& read_data, uint8_t& done) const {
        read_data = read_data_reg;
        done = done_reg;
    }

    void clock_edge(bool content_en, bool write_en, uint32_t addr, uint32_t write_data, int cycle) {
        if (content_en) {
            if (write_en) {
                // Write
                data[addr % MEM_SIZE] = write_data;
                if (name == "mem_1" && (addr % MEM_SIZE) < 20) {
                    printf("[Cycle %5d] %s WRITE addr=%3u data=%08X\n", cycle, name.c_str(), addr % MEM_SIZE, write_data);
                }
                done_reg = 1;
            } else {
                // Read: latch data for next cycle
                read_data_reg = data[addr % MEM_SIZE];
                if (debug && name == "mem_0" && (addr % MEM_SIZE) < 10) {
                    printf("[Cycle %5d] %s READ  addr=%3u -> %08X\n", cycle, name.c_str(), addr % MEM_SIZE, read_data_reg);
                }
                done_reg = 1;
            }
        } else {
            done_reg = 0;
        }
    }
};

void save_mem(CalyxMem& mem, int samples, const std::string& filename) {
    std::ofstream out(filename);
    for (int i = 0; i < samples; i++) {
        out << std::hex << std::setw(8) << std::setfill('0') << mem.data[i] << "\n";
    }
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vmain* dut = new Vmain;

    int samples = 300;
    if (argc > 1) samples = std::atoi(argv[1]);

    CalyxMem mem0("mem_0");
    CalyxMem mem1("mem_1");
    CalyxMem mem2("mem_2");
    CalyxMem mem3("mem_3");
    CalyxMem mem4("mem_4");

    mem0.debug = true;  

    std::ifstream file("data/mem_0.dat");
    if (!file.is_open()) {
        printf("ERROR: Cannot open data/mem_0.dat\n");
        exit(1);
    }
    std::string line;
    int idx = 0;
    while (std::getline(file, line) && idx < MEM_SIZE) {
        mem0.data[idx++] = std::stoul(line, nullptr, 16);
    }
    printf("Loaded %d inputs into mem0\n", idx);

    // Reset sequence
    dut->clk = 0;
    dut->reset = 1;
    dut->go = 0;
    dut->eval();

    // Initial memory outputs
    dut->mem_0_read_data = 0;
    dut->mem_1_read_data = 0;
    dut->mem_2_read_data = 0;
    dut->mem_3_read_data = 0;
    dut->mem_4_read_data = 0;
    dut->mem_0_done = 0;
    dut->mem_1_done = 0;
    dut->mem_2_done = 0;
    dut->mem_3_done = 0;
    dut->mem_4_done = 0;
    dut->eval();

    // Hold reset for 5 cycles
    for (int i = 0; i < 5; i++) {
        dut->clk = 1; dut->eval();
        dut->clk = 0; dut->eval();
    }
    dut->reset = 0;
    dut->go = 1;          // level-sensitive start
    dut->eval();

    // Preload read data registers with address 0 content
    mem0.read_data_reg = mem0.data[0];
    mem1.read_data_reg = mem1.data[0];
    mem2.read_data_reg = mem2.data[0];
    mem3.read_data_reg = mem3.data[0];
    mem4.read_data_reg = mem4.data[0];
    mem0.done_reg = 0;
    mem1.done_reg = 0;
    mem2.done_reg = 0;
    mem3.done_reg = 0;
    mem4.done_reg = 0;

    int cycle = 0;
    const int MAX_CYCLES = 20000;
    bool done_seen = false;

    // Main simulation loop
    while (cycle < MAX_CYCLES && !done_seen) {
        // ----- FALLING EDGE -----
        dut->clk = 0;
        mem0.get_outputs(dut->mem_0_read_data, dut->mem_0_done);
        mem1.get_outputs(dut->mem_1_read_data, dut->mem_1_done);
        mem2.get_outputs(dut->mem_2_read_data, dut->mem_2_done);
        mem3.get_outputs(dut->mem_3_read_data, dut->mem_3_done);
        mem4.get_outputs(dut->mem_4_read_data, dut->mem_4_done);
        dut->eval();

        bool ce0 = dut->mem_0_content_en, we0 = dut->mem_0_write_en;
        uint32_t a0 = dut->mem_0_addr0, wd0 = dut->mem_0_write_data;
        bool ce1 = dut->mem_1_content_en, we1 = dut->mem_1_write_en;
        uint32_t a1 = dut->mem_1_addr0, wd1 = dut->mem_1_write_data;
        bool ce2 = dut->mem_2_content_en, we2 = dut->mem_2_write_en;
        uint32_t a2 = dut->mem_2_addr0, wd2 = dut->mem_2_write_data;
        bool ce3 = dut->mem_3_content_en, we3 = dut->mem_3_write_en;
        uint32_t a3 = dut->mem_3_addr0, wd3 = dut->mem_3_write_data;
        bool ce4 = dut->mem_4_content_en, we4 = dut->mem_4_write_en;
        uint32_t a4 = dut->mem_4_addr0, wd4 = dut->mem_4_write_data;

        // ----- RISING EDGE -----
        dut->clk = 1;
        dut->eval();

        // Update memory models with captured signals
        mem0.clock_edge(ce0, we0, a0, wd0, cycle);
        mem1.clock_edge(ce1, we1, a1, wd1, cycle);
        mem2.clock_edge(ce2, we2, a2, wd2, cycle);
        mem3.clock_edge(ce3, we3, a3, wd3, cycle);
        mem4.clock_edge(ce4, we4, a4, wd4, cycle);

        if (dut->done && cycle > 10) {
            printf("DONE asserted at cycle %d\n", cycle);
            done_seen = true;
        }
        cycle++;
    }

    if (!done_seen) printf("TIMEOUT after %d cycles\n", MAX_CYCLES);

    printf("\n=== Memory contents after simulation ===\n");
    for (int i = 0; i < 10; i++) {
        printf("mem0[%2d]=%08X  mem1[%2d]=%08X  mem2[%2d]=%08X  mem3[%2d]=%08X  mem4[%2d]=%08X\n",
               i, mem0.data[i], i, mem1.data[i], i, mem2.data[i], i, mem3.data[i], i, mem4.data[i]);
    }

    save_mem(mem1, samples, "data/mem_1.out");
    save_mem(mem2, samples, "data/mem_2.out");
    save_mem(mem3, samples, "data/mem_3.out");
    save_mem(mem4, samples, "data/mem_4.out");

    dut->final();
    delete dut;
    return 0;
}
