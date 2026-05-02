// Compiled by morty-0.9.0 / 2026-05-02 22:31:20.852547692 +05:45:00

/**
Implements a memory with sequential reads and writes.
- Both reads and writes take one cycle to perform.
- Attempting to read and write at the same time is an error.
- The out signal is registered to the last value requested by the read_en signal.
- The out signal is undefined once write_en is asserted.
*/
module seq_mem_d1 #(
    parameter WIDTH = 32,
    parameter SIZE = 16,
    parameter IDX_SIZE = 4
) (
   // Common signals
   input wire logic clk,
   input wire logic reset,
   input wire logic [IDX_SIZE-1:0] addr0,
   input wire logic content_en,
   output logic done,

   // Read signal
   output logic [ WIDTH-1:0] read_data,

   // Write signals
   input wire logic [ WIDTH-1:0] write_data,
   input wire logic write_en
);
  // Internal memory
  logic [WIDTH-1:0] mem[SIZE-1:0];

  // Register for the read output
  logic [WIDTH-1:0] read_out;
  assign read_data = read_out;

  // Read value from the memory
  always_ff @(posedge clk) begin
    if (reset) begin
      read_out <= '0;
    end else if (content_en && !write_en) begin
      /* verilator lint_off WIDTH */
      read_out <= mem[addr0];
    end else if (content_en && write_en) begin
      // Explicitly clobber the read output when a write is performed
      read_out <= 'x;
    end else begin
      read_out <= read_out;
    end
  end

  // Propagate the done signal
  always_ff @(posedge clk) begin
    if (reset) begin
      done <= '0;
    end else if (content_en) begin
      done <= '1;
    end else begin
      done <= '0;
    end
  end

  // Write value to the memory
  always_ff @(posedge clk) begin
    if (!reset && content_en && write_en)
      mem[addr0] <= write_data;
  end

  // Check for out of bounds access
  
endmodule

module seq_mem_d2 #(
    parameter WIDTH = 32,
    parameter D0_SIZE = 16,
    parameter D1_SIZE = 16,
    parameter D0_IDX_SIZE = 4,
    parameter D1_IDX_SIZE = 4
) (
   // Common signals
   input wire logic clk,
   input wire logic reset,
   input wire logic [D0_IDX_SIZE-1:0] addr0,
   input wire logic [D1_IDX_SIZE-1:0] addr1,
   input wire logic content_en,
   output logic done,

   // Read signal
   output logic [WIDTH-1:0] read_data,

   // Write signals
   input wire logic write_en,
   input wire logic [ WIDTH-1:0] write_data
);
  wire [D0_IDX_SIZE+D1_IDX_SIZE-1:0] addr;
  assign addr = addr0 * D1_SIZE + addr1;

  seq_mem_d1 #(.WIDTH(WIDTH), .SIZE(D0_SIZE * D1_SIZE), .IDX_SIZE(D0_IDX_SIZE+D1_IDX_SIZE)) mem
     (.clk(clk), .reset(reset), .addr0(addr),
    .content_en(content_en), .read_data(read_data), .write_data(write_data), .write_en(write_en),
    .done(done));
endmodule

module seq_mem_d3 #(
    parameter WIDTH = 32,
    parameter D0_SIZE = 16,
    parameter D1_SIZE = 16,
    parameter D2_SIZE = 16,
    parameter D0_IDX_SIZE = 4,
    parameter D1_IDX_SIZE = 4,
    parameter D2_IDX_SIZE = 4
) (
   // Common signals
   input wire logic clk,
   input wire logic reset,
   input wire logic [D0_IDX_SIZE-1:0] addr0,
   input wire logic [D1_IDX_SIZE-1:0] addr1,
   input wire logic [D2_IDX_SIZE-1:0] addr2,
   input wire logic content_en,
   output logic done,

   // Read signal
   output logic [WIDTH-1:0] read_data,

   // Write signals
   input wire logic write_en,
   input wire logic [ WIDTH-1:0] write_data
);
  wire [D0_IDX_SIZE+D1_IDX_SIZE+D2_IDX_SIZE-1:0] addr;
  assign addr = addr0 * (D1_SIZE * D2_SIZE) + addr1 * (D2_SIZE) + addr2;

  seq_mem_d1 #(.WIDTH(WIDTH), .SIZE(D0_SIZE * D1_SIZE * D2_SIZE), .IDX_SIZE(D0_IDX_SIZE+D1_IDX_SIZE+D2_IDX_SIZE)) mem
     (.clk(clk), .reset(reset), .addr0(addr),
    .content_en(content_en), .read_data(read_data), .write_data(write_data), .write_en(write_en),
    .done(done));
endmodule

module seq_mem_d4 #(
    parameter WIDTH = 32,
    parameter D0_SIZE = 16,
    parameter D1_SIZE = 16,
    parameter D2_SIZE = 16,
    parameter D3_SIZE = 16,
    parameter D0_IDX_SIZE = 4,
    parameter D1_IDX_SIZE = 4,
    parameter D2_IDX_SIZE = 4,
    parameter D3_IDX_SIZE = 4
) (
   // Common signals
   input wire logic clk,
   input wire logic reset,
   input wire logic [D0_IDX_SIZE-1:0] addr0,
   input wire logic [D1_IDX_SIZE-1:0] addr1,
   input wire logic [D2_IDX_SIZE-1:0] addr2,
   input wire logic [D3_IDX_SIZE-1:0] addr3,
   input wire logic content_en,
   output logic done,

   // Read signal
   output logic [WIDTH-1:0] read_data,

   // Write signals
   input wire logic write_en,
   input wire logic [ WIDTH-1:0] write_data
);
  wire [D0_IDX_SIZE+D1_IDX_SIZE+D2_IDX_SIZE+D3_IDX_SIZE-1:0] addr;
  assign addr = addr0 * (D1_SIZE * D2_SIZE * D3_SIZE) + addr1 * (D2_SIZE * D3_SIZE) + addr2 * (D3_SIZE) + addr3;

  seq_mem_d1 #(.WIDTH(WIDTH), .SIZE(D0_SIZE * D1_SIZE * D2_SIZE * D3_SIZE), .IDX_SIZE(D0_IDX_SIZE+D1_IDX_SIZE+D2_IDX_SIZE+D3_IDX_SIZE)) mem
     (.clk(clk), .reset(reset), .addr0(addr),
    .content_en(content_en), .read_data(read_data), .write_data(write_data), .write_en(write_en),
    .done(done));
endmodule
`define __COMPAREFN_V__

module std_compareFN #(parameter expWidth = 8, parameter sigWidth = 24, parameter numWidth = 32) (
    input clk,
    input reset,
    input go,
    input [(expWidth + sigWidth - 1):0] left,
    input [(expWidth + sigWidth - 1):0] right,
    input signaling,
    output logic lt,
    output logic eq,
    output logic gt,
    output logic unordered,
    output logic [4:0] exceptionFlags,
    output done
);

    // Intermediate signals for recoded formats
    wire [(expWidth + sigWidth):0] l_recoded, r_recoded;

    // Convert 'left' and 'right' from standard to recoded format
    fNToRecFN #(expWidth, sigWidth) convert_l(
        .in(left),
        .out(l_recoded)
    );

    fNToRecFN #(expWidth, sigWidth) convert_r(
        .in(right),
        .out(r_recoded)
    );

    // Intermediate signals for comparison outputs
    wire comp_lt, comp_eq, comp_gt, comp_unordered;
    wire [4:0] comp_exceptionFlags;

    // Compare recoded numbers
    compareRecFN #(expWidth, sigWidth) comparator(
        .a(l_recoded),
        .b(r_recoded),
        .signaling(signaling),
        .lt(comp_lt),
        .eq(comp_eq),
        .gt(comp_gt),
        .unordered(comp_unordered),
        .exceptionFlags(comp_exceptionFlags)
    );

    logic done_buf[1:0];

    assign done = done_buf[1];

    // If the done buffer is empty and go is high, execution just started.
    logic start;
    assign start = go;

    // Start sending the done signal.
    always_ff @(posedge clk) begin
        if (start)
            done_buf[0] <= 1;
        else
            done_buf[0] <= 0;
    end

    // Push the done signal through the pipeline.
    always_ff @(posedge clk) begin
        if (go) begin
            done_buf[1] <= done_buf[0];
        end else begin
            done_buf[1] <= 0;
        end
    end

    // Capture the comparison results
    always_ff @(posedge clk) begin
        if (reset) begin
            lt <= 0;
            eq <= 0;
            gt <= 0;
            unordered <= 0;
            exceptionFlags <= 0;
        end else if (go) begin
            lt <= comp_lt;
            eq <= comp_eq;
            gt <= comp_gt;
            unordered <= comp_unordered;
            exceptionFlags <= comp_exceptionFlags;
        end else begin
            lt <= lt;
            eq <= eq;
            gt <= gt;
            unordered <= unordered;
            exceptionFlags <= exceptionFlags;
        end
    end

endmodule


 /* __COMPAREFN_V__ */
`define __ADDFN_V__


/*============================================================================

This Verilog include file is part of the Berkeley HardFloat IEEE Floating-
Point Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define round_near_even   3'b000
`define round_minMag      3'b001
`define round_min         3'b010
`define round_max         3'b011
`define round_near_maxMag 3'b100
`define round_odd         3'b110

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define floatControlWidth 1
`define flControl_tininessBeforeRounding 1'b0
`define flControl_tininessAfterRounding  1'b1

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define flRoundOpt_sigMSBitAlwaysZero  1
`define flRoundOpt_subnormsAlwaysExact 2
`define flRoundOpt_neverUnderflows     4
`define flRoundOpt_neverOverflows      8



module std_addFN #(parameter expWidth = 8, parameter sigWidth = 24, parameter numWidth = 32) (
    input clk,
    input reset,
    input go,
    input [(1 - 1):0] control,
    input subOp,
    input [(expWidth + sigWidth - 1):0] left,
    input [(expWidth + sigWidth - 1):0] right,
    input [2:0] roundingMode,
    output logic [(expWidth + sigWidth - 1):0] out,
    output logic [4:0] exceptionFlags,
    output done
);

    // Intermediate signals for recoded formats
    wire [(expWidth + sigWidth):0] l_recoded, r_recoded;

    // Convert 'a' and 'b' from standard to recoded format
    fNToRecFN #(expWidth, sigWidth) convert_l(
        .in(left),
        .out(l_recoded)
    );

    fNToRecFN #(expWidth, sigWidth) convert_r(
        .in(right),
        .out(r_recoded)
    );

    // Intermediate signals after the adder
    wire [(expWidth + sigWidth):0] res_recoded;
    wire [4:0] except_flag;

    // Compute recoded numbers
    addRecFN #(expWidth, sigWidth) adder(
        .control(control),
        .subOp(subOp),
        .a(l_recoded),
        .b(r_recoded),
        .roundingMode(roundingMode),
        .out(res_recoded),
        .exceptionFlags(except_flag)
    );

    wire [(expWidth + sigWidth - 1):0] res_std;

    // Convert the result back to standard format
    recFNToFN #(expWidth, sigWidth) convert_res(
        .in(res_recoded),
        .out(res_std)
    );

    logic done_buf[1:0];

    assign done = done_buf[1];

    // If the done buffer is completely empty and go is high then execution
    // just started.
    logic start;
    assign start = go;

    // Start sending the done signal.
    always_ff @(posedge clk) begin
        if (start)
            done_buf[0] <= 1;
        else
            done_buf[0] <= 0;
    end

    // Push the done signal through the pipeline.
    always_ff @(posedge clk) begin
        if (go) begin
            done_buf[1] <= done_buf[0];
        end else begin
            done_buf[1] <= 0;
        end
    end

    // Compute the output and save it into out
    always_ff @(posedge clk) begin
        if (reset) begin
            out <= 0;
        end else if (go) begin
            out <= res_std;
        end else begin
            out <= out;
        end
    end

endmodule


 /* __ADDFN_V__ */
/**
 * Core primitives for Calyx.
 * Implements core primitives used by the compiler.
 *
 * Conventions:
 * - All parameter names must be SNAKE_CASE and all caps.
 * - Port names must be snake_case, no caps.
 */

module std_slice #(
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = 32
) (
   input wire                   logic [ IN_WIDTH-1:0] in,
   output logic [OUT_WIDTH-1:0] out
);
  assign out = in[OUT_WIDTH-1:0];

  
endmodule

module std_pad #(
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = 32
) (
   input wire logic [IN_WIDTH-1:0]  in,
   output logic     [OUT_WIDTH-1:0] out
);
  localparam EXTEND = OUT_WIDTH - IN_WIDTH;
  assign out = { {EXTEND {1'b0}}, in};

  
endmodule

module std_cat #(
  parameter LEFT_WIDTH  = 32,
  parameter RIGHT_WIDTH = 32,
  parameter OUT_WIDTH = 64
) (
  input wire logic [LEFT_WIDTH-1:0] left,
  input wire logic [RIGHT_WIDTH-1:0] right,
  output logic [OUT_WIDTH-1:0] out
);
  assign out = {left, right};

  
endmodule

module std_not #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] in,
   output logic [WIDTH-1:0] out
);
  assign out = ~in;
endmodule

module std_and #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left & right;
endmodule

module std_or #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left | right;
endmodule

module std_xor #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left ^ right;
endmodule

module std_sub #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left - right;
endmodule

module std_gt #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left > right;
endmodule

module std_lt #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left < right;
endmodule

module std_eq #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left == right;
endmodule

module std_neq #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left != right;
endmodule

module std_ge #(
    parameter WIDTH = 32
) (
    input wire   logic [WIDTH-1:0] left,
    input wire   logic [WIDTH-1:0] right,
    output logic out
);
  assign out = left >= right;
endmodule

module std_le #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left <= right;
endmodule

module std_rsh #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left >> right;
endmodule

/// this primitive is intended to be used
/// for lowering purposes (not in source programs)
module std_mux #(
    parameter WIDTH = 32
) (
   input wire               logic cond,
   input wire               logic [WIDTH-1:0] tru,
   input wire               logic [WIDTH-1:0] fal,
   output logic [WIDTH-1:0] out
);
  assign out = cond ? tru : fal;
endmodule

module std_bit_slice #(
    parameter IN_WIDTH = 32,
    parameter START_IDX = 0,
    parameter END_IDX = 31,
    parameter OUT_WIDTH = 32
)(
   input wire logic [IN_WIDTH-1:0] in,
   output logic [OUT_WIDTH-1:0] out
);
  assign out = in[END_IDX:START_IDX];

  

endmodule

module std_skid_buffer #(
    parameter WIDTH = 32
)(
    input wire logic [WIDTH-1:0] in,
    input wire logic i_valid,
    input wire logic i_ready,
    input wire logic clk,
    input wire logic reset,
    output logic [WIDTH-1:0] out,
    output logic o_valid,
    output logic o_ready
);
  logic [WIDTH-1:0] val;
  logic bypass_rg;
  always @(posedge clk) begin
    // Reset  
    if (reset) begin      
      // Internal Registers
      val <= '0;     
      bypass_rg <= 1'b1;
    end   
    // Out of reset
    else begin      
      // Bypass state      
      if (bypass_rg) begin         
        if (!i_ready && i_valid) begin
          val <= in;          // Data skid happened, store to buffer
          bypass_rg <= 1'b0;  // To skid mode  
        end 
      end 
      // Skid state
      else begin         
        if (i_ready) begin
          bypass_rg <= 1'b1;  // Back to bypass mode           
        end
      end
    end
  end

  assign o_ready = bypass_rg;
  assign out = bypass_rg ? in : val;
  assign o_valid = bypass_rg ? i_valid : 1'b1;
endmodule

module std_bypass_reg #(
    parameter WIDTH = 32
)(
    input wire logic [WIDTH-1:0] in,
    input wire logic write_en,
    input wire logic clk,
    input wire logic reset,
    output logic [WIDTH-1:0] out,
    output logic done
);
  logic [WIDTH-1:0] val;
  assign out = write_en ? in : val;

  always_ff @(posedge clk) begin
    if (reset) begin
      val <= 0;
      done <= 0;
    end else if (write_en) begin
      val <= in;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

/*============================================================================

This Verilog source file is part of the Berkeley HardFloat IEEE Floating-Point
Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    isSigNaNRecFN#(parameter expWidth = 3, parameter sigWidth = 3) (
        input [(expWidth + sigWidth):0] in, output isSigNaN
    );

    wire isNaN =
        (in[(expWidth + sigWidth - 1):(expWidth + sigWidth - 3)] == 'b111);
    assign isSigNaN = isNaN && !in[sigWidth - 2];

endmodule

module std_float_const #(
    parameter REP = 32,
    parameter WIDTH = 32,
    parameter VALUE = 32
) (
   output logic [WIDTH-1:0] out
);
assign out = VALUE;
endmodule

module undef #(
    parameter WIDTH = 32
) (
   output logic [WIDTH-1:0] out
);
assign out = 'x;
endmodule

module std_const #(
    parameter WIDTH = 32,
    parameter VALUE = 32
) (
   output logic [WIDTH-1:0] out
);
assign out = VALUE;
endmodule

module std_wire #(
    parameter WIDTH = 32
) (
   input wire logic [WIDTH-1:0] in,
   output logic [WIDTH-1:0] out
);
assign out = in;
endmodule

module std_add #(
    parameter WIDTH = 32
) (
   input wire logic [WIDTH-1:0] left,
   input wire logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
assign out = left + right;
endmodule

module std_lsh #(
    parameter WIDTH = 32
) (
   input wire logic [WIDTH-1:0] left,
   input wire logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
assign out = left << right;
endmodule

module std_reg #(
    parameter WIDTH = 32
) (
   input wire logic [WIDTH-1:0] in,
   input wire logic write_en,
   input wire logic clk,
   input wire logic reset,
   output logic [WIDTH-1:0] out,
   output logic done
);
always_ff @(posedge clk) begin
    if (reset) begin
       out <= 0;
       done <= 0;
    end else if (write_en) begin
      out <= in;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

module init_one_reg #(
    parameter WIDTH = 32
) (
   input wire logic [WIDTH-1:0] in,
   input wire logic write_en,
   input wire logic clk,
   input wire logic reset,
   output logic [WIDTH-1:0] out,
   output logic done
);
always_ff @(posedge clk) begin
    if (reset) begin
       out <= 1;
       done <= 0;
    end else if (write_en) begin
      out <= in;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

module main(
  input logic clk,
  input logic reset,
  input logic go,
  output logic done,
  output logic [8:0] mem_4_addr0,
  output logic mem_4_content_en,
  output logic mem_4_write_en,
  output logic [31:0] mem_4_write_data,
  input logic [31:0] mem_4_read_data,
  input logic mem_4_done,
  output logic [8:0] mem_3_addr0,
  output logic mem_3_content_en,
  output logic mem_3_write_en,
  output logic [31:0] mem_3_write_data,
  input logic [31:0] mem_3_read_data,
  input logic mem_3_done,
  output logic [8:0] mem_2_addr0,
  output logic mem_2_content_en,
  output logic mem_2_write_en,
  output logic [31:0] mem_2_write_data,
  input logic [31:0] mem_2_read_data,
  input logic mem_2_done,
  output logic [8:0] mem_1_addr0,
  output logic mem_1_content_en,
  output logic mem_1_write_en,
  output logic [31:0] mem_1_write_data,
  input logic [31:0] mem_1_read_data,
  input logic mem_1_done,
  output logic [8:0] mem_0_addr0,
  output logic mem_0_content_en,
  output logic mem_0_write_en,
  output logic [31:0] mem_0_write_data,
  input logic [31:0] mem_0_read_data,
  input logic mem_0_done
);
// COMPONENT START: main
logic forward_instance_clk;
logic forward_instance_reset;
logic forward_instance_go;
logic forward_instance_done;
logic [31:0] forward_instance_arg_mem_0_read_data;
logic forward_instance_arg_mem_0_done;
logic forward_instance_arg_mem_4_done;
logic [31:0] forward_instance_arg_mem_1_write_data;
logic [31:0] forward_instance_arg_mem_3_read_data;
logic [31:0] forward_instance_arg_mem_2_read_data;
logic [31:0] forward_instance_arg_mem_1_read_data;
logic forward_instance_arg_mem_0_content_en;
logic [31:0] forward_instance_arg_mem_4_write_data;
logic [8:0] forward_instance_arg_mem_3_addr0;
logic [31:0] forward_instance_arg_mem_3_write_data;
logic [8:0] forward_instance_arg_mem_0_addr0;
logic [8:0] forward_instance_arg_mem_4_addr0;
logic forward_instance_arg_mem_3_content_en;
logic forward_instance_arg_mem_0_write_en;
logic forward_instance_arg_mem_3_done;
logic forward_instance_arg_mem_1_done;
logic forward_instance_arg_mem_3_write_en;
logic [8:0] forward_instance_arg_mem_2_addr0;
logic forward_instance_arg_mem_2_done;
logic [31:0] forward_instance_arg_mem_0_write_data;
logic forward_instance_arg_mem_2_content_en;
logic forward_instance_arg_mem_1_write_en;
logic forward_instance_arg_mem_4_content_en;
logic forward_instance_arg_mem_2_write_en;
logic forward_instance_arg_mem_4_write_en;
logic [31:0] forward_instance_arg_mem_4_read_data;
logic [8:0] forward_instance_arg_mem_1_addr0;
logic forward_instance_arg_mem_1_content_en;
logic [31:0] forward_instance_arg_mem_2_write_data;
logic invoke0_go_in;
logic invoke0_go_out;
logic invoke0_done_in;
logic invoke0_done_out;
forward forward_instance (
    .arg_mem_0_addr0(forward_instance_arg_mem_0_addr0),
    .arg_mem_0_content_en(forward_instance_arg_mem_0_content_en),
    .arg_mem_0_done(forward_instance_arg_mem_0_done),
    .arg_mem_0_read_data(forward_instance_arg_mem_0_read_data),
    .arg_mem_0_write_data(forward_instance_arg_mem_0_write_data),
    .arg_mem_0_write_en(forward_instance_arg_mem_0_write_en),
    .arg_mem_1_addr0(forward_instance_arg_mem_1_addr0),
    .arg_mem_1_content_en(forward_instance_arg_mem_1_content_en),
    .arg_mem_1_done(forward_instance_arg_mem_1_done),
    .arg_mem_1_read_data(forward_instance_arg_mem_1_read_data),
    .arg_mem_1_write_data(forward_instance_arg_mem_1_write_data),
    .arg_mem_1_write_en(forward_instance_arg_mem_1_write_en),
    .arg_mem_2_addr0(forward_instance_arg_mem_2_addr0),
    .arg_mem_2_content_en(forward_instance_arg_mem_2_content_en),
    .arg_mem_2_done(forward_instance_arg_mem_2_done),
    .arg_mem_2_read_data(forward_instance_arg_mem_2_read_data),
    .arg_mem_2_write_data(forward_instance_arg_mem_2_write_data),
    .arg_mem_2_write_en(forward_instance_arg_mem_2_write_en),
    .arg_mem_3_addr0(forward_instance_arg_mem_3_addr0),
    .arg_mem_3_content_en(forward_instance_arg_mem_3_content_en),
    .arg_mem_3_done(forward_instance_arg_mem_3_done),
    .arg_mem_3_read_data(forward_instance_arg_mem_3_read_data),
    .arg_mem_3_write_data(forward_instance_arg_mem_3_write_data),
    .arg_mem_3_write_en(forward_instance_arg_mem_3_write_en),
    .arg_mem_4_addr0(forward_instance_arg_mem_4_addr0),
    .arg_mem_4_content_en(forward_instance_arg_mem_4_content_en),
    .arg_mem_4_done(forward_instance_arg_mem_4_done),
    .arg_mem_4_read_data(forward_instance_arg_mem_4_read_data),
    .arg_mem_4_write_data(forward_instance_arg_mem_4_write_data),
    .arg_mem_4_write_en(forward_instance_arg_mem_4_write_en),
    .clk(forward_instance_clk),
    .done(forward_instance_done),
    .go(forward_instance_go),
    .reset(forward_instance_reset)
);
std_wire # (
    .WIDTH(1)
) invoke0_go (
    .in(invoke0_go_in),
    .out(invoke0_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke0_done (
    .in(invoke0_done_in),
    .out(invoke0_done_out)
);
wire _guard0 = 1;
wire _guard1 = invoke0_done_out;
wire _guard2 = invoke0_go_out;
wire _guard3 = invoke0_go_out;
wire _guard4 = invoke0_go_out;
wire _guard5 = invoke0_go_out;
wire _guard6 = invoke0_go_out;
wire _guard7 = invoke0_go_out;
wire _guard8 = invoke0_go_out;
wire _guard9 = invoke0_go_out;
wire _guard10 = invoke0_go_out;
wire _guard11 = invoke0_go_out;
wire _guard12 = invoke0_go_out;
wire _guard13 = invoke0_go_out;
wire _guard14 = invoke0_go_out;
wire _guard15 = invoke0_go_out;
wire _guard16 = invoke0_go_out;
wire _guard17 = invoke0_go_out;
wire _guard18 = invoke0_go_out;
wire _guard19 = invoke0_go_out;
wire _guard20 = invoke0_go_out;
wire _guard21 = invoke0_go_out;
wire _guard22 = invoke0_go_out;
wire _guard23 = invoke0_go_out;
wire _guard24 = invoke0_go_out;
wire _guard25 = invoke0_go_out;
wire _guard26 = invoke0_go_out;
wire _guard27 = invoke0_go_out;
wire _guard28 = invoke0_go_out;
wire _guard29 = invoke0_go_out;
assign done = _guard1;
assign mem_4_addr0 = forward_instance_arg_mem_4_addr0;
assign mem_3_write_data = forward_instance_arg_mem_3_write_data;
assign mem_4_content_en =
  _guard4 ? forward_instance_arg_mem_4_content_en :
  1'd0;
assign mem_2_write_data = forward_instance_arg_mem_2_write_data;
assign mem_4_write_en =
  _guard6 ? forward_instance_arg_mem_4_write_en :
  1'd0;
assign mem_4_write_data = forward_instance_arg_mem_4_write_data;
assign mem_3_content_en =
  _guard8 ? forward_instance_arg_mem_3_content_en :
  1'd0;
assign mem_1_addr0 = forward_instance_arg_mem_1_addr0;
assign mem_1_content_en =
  _guard10 ? forward_instance_arg_mem_1_content_en :
  1'd0;
assign mem_3_write_en =
  _guard11 ? forward_instance_arg_mem_3_write_en :
  1'd0;
assign mem_0_addr0 = forward_instance_arg_mem_0_addr0;
assign mem_2_addr0 = forward_instance_arg_mem_2_addr0;
assign mem_1_write_en =
  _guard14 ? forward_instance_arg_mem_1_write_en :
  1'd0;
assign mem_2_content_en =
  _guard15 ? forward_instance_arg_mem_2_content_en :
  1'd0;
assign mem_1_write_data = forward_instance_arg_mem_1_write_data;
assign mem_0_content_en =
  _guard17 ? forward_instance_arg_mem_0_content_en :
  1'd0;
assign mem_3_addr0 = forward_instance_arg_mem_3_addr0;
assign mem_2_write_en =
  _guard19 ? forward_instance_arg_mem_2_write_en :
  1'd0;
assign invoke0_go_in = go;
assign forward_instance_arg_mem_0_read_data =
  _guard20 ? mem_0_read_data :
  32'd0;
assign forward_instance_arg_mem_0_done =
  _guard21 ? mem_0_done :
  1'd0;
assign forward_instance_arg_mem_4_done =
  _guard22 ? mem_4_done :
  1'd0;
assign forward_instance_arg_mem_3_read_data =
  _guard23 ? mem_3_read_data :
  32'd0;
assign forward_instance_arg_mem_1_read_data =
  _guard24 ? mem_1_read_data :
  32'd0;
assign forward_instance_clk = clk;
assign forward_instance_arg_mem_3_done =
  _guard25 ? mem_3_done :
  1'd0;
assign forward_instance_reset = reset;
assign forward_instance_go = _guard26;
assign forward_instance_arg_mem_1_done =
  _guard27 ? mem_1_done :
  1'd0;
assign forward_instance_arg_mem_2_done =
  _guard28 ? mem_2_done :
  1'd0;
assign forward_instance_arg_mem_4_read_data =
  _guard29 ? mem_4_read_data :
  32'd0;
assign invoke0_done_in = forward_instance_done;
// COMPONENT END: main
endmodule
module relu4d_0(
  input logic clk,
  input logic reset,
  input logic go,
  output logic done,
  output logic [8:0] arg_mem_1_addr0,
  output logic arg_mem_1_content_en,
  output logic arg_mem_1_write_en,
  output logic [31:0] arg_mem_1_write_data,
  input logic [31:0] arg_mem_1_read_data,
  input logic arg_mem_1_done,
  output logic [8:0] arg_mem_0_addr0,
  output logic arg_mem_0_content_en,
  output logic arg_mem_0_write_en,
  output logic [31:0] arg_mem_0_write_data,
  input logic [31:0] arg_mem_0_read_data,
  input logic arg_mem_0_done
);
// COMPONENT START: relu4d_0
logic [31:0] cst_0_out;
logic [31:0] std_slice_3_in;
logic [8:0] std_slice_3_out;
logic [31:0] std_add_1_left;
logic [31:0] std_add_1_right;
logic [31:0] std_add_1_out;
logic [31:0] load_0_reg_in;
logic load_0_reg_write_en;
logic load_0_reg_clk;
logic load_0_reg_reset;
logic [31:0] load_0_reg_out;
logic load_0_reg_done;
logic std_mux_0_cond;
logic [31:0] std_mux_0_tru;
logic [31:0] std_mux_0_fal;
logic [31:0] std_mux_0_out;
logic std_and_0_left;
logic std_and_0_right;
logic std_and_0_out;
logic std_or_0_left;
logic std_or_0_right;
logic std_or_0_out;
logic unordered_port_0_reg_in;
logic unordered_port_0_reg_write_en;
logic unordered_port_0_reg_clk;
logic unordered_port_0_reg_reset;
logic unordered_port_0_reg_out;
logic unordered_port_0_reg_done;
logic compare_port_0_reg_in;
logic compare_port_0_reg_write_en;
logic compare_port_0_reg_clk;
logic compare_port_0_reg_reset;
logic compare_port_0_reg_out;
logic compare_port_0_reg_done;
logic cmpf_0_reg_in;
logic cmpf_0_reg_write_en;
logic cmpf_0_reg_clk;
logic cmpf_0_reg_reset;
logic cmpf_0_reg_out;
logic cmpf_0_reg_done;
logic std_compareFN_0_clk;
logic std_compareFN_0_reset;
logic std_compareFN_0_go;
logic [31:0] std_compareFN_0_left;
logic [31:0] std_compareFN_0_right;
logic std_compareFN_0_signaling;
logic std_compareFN_0_lt;
logic std_compareFN_0_eq;
logic std_compareFN_0_gt;
logic std_compareFN_0_unordered;
logic [4:0] std_compareFN_0_exceptionFlags;
logic std_compareFN_0_done;
logic mem_0_clk;
logic mem_0_reset;
logic [8:0] mem_0_addr0;
logic mem_0_content_en;
logic mem_0_write_en;
logic [31:0] mem_0_write_data;
logic [31:0] mem_0_read_data;
logic mem_0_done;
logic [31:0] for_1_induction_var_reg_in;
logic for_1_induction_var_reg_write_en;
logic for_1_induction_var_reg_clk;
logic for_1_induction_var_reg_reset;
logic [31:0] for_1_induction_var_reg_out;
logic for_1_induction_var_reg_done;
logic [8:0] idx_in;
logic idx_write_en;
logic idx_clk;
logic idx_reset;
logic [8:0] idx_out;
logic idx_done;
logic cond_reg_in;
logic cond_reg_write_en;
logic cond_reg_clk;
logic cond_reg_reset;
logic cond_reg_out;
logic cond_reg_done;
logic [8:0] adder_left;
logic [8:0] adder_right;
logic [8:0] adder_out;
logic [8:0] lt_left;
logic [8:0] lt_right;
logic lt_out;
logic [8:0] idx0_in;
logic idx0_write_en;
logic idx0_clk;
logic idx0_reset;
logic [8:0] idx0_out;
logic idx0_done;
logic cond_reg0_in;
logic cond_reg0_write_en;
logic cond_reg0_clk;
logic cond_reg0_reset;
logic cond_reg0_out;
logic cond_reg0_done;
logic [8:0] adder0_left;
logic [8:0] adder0_right;
logic [8:0] adder0_out;
logic [8:0] lt0_left;
logic [8:0] lt0_right;
logic lt0_out;
logic [1:0] fsm_in;
logic fsm_write_en;
logic fsm_clk;
logic fsm_reset;
logic [1:0] fsm_out;
logic fsm_done;
logic [1:0] adder1_left;
logic [1:0] adder1_right;
logic [1:0] adder1_out;
logic ud_out;
logic [1:0] adder2_left;
logic [1:0] adder2_right;
logic [1:0] adder2_out;
logic ud0_out;
logic signal_reg_in;
logic signal_reg_write_en;
logic signal_reg_clk;
logic signal_reg_reset;
logic signal_reg_out;
logic signal_reg_done;
logic [3:0] fsm0_in;
logic fsm0_write_en;
logic fsm0_clk;
logic fsm0_reset;
logic [3:0] fsm0_out;
logic fsm0_done;
logic bb0_0_go_in;
logic bb0_0_go_out;
logic bb0_0_done_in;
logic bb0_0_done_out;
logic bb0_1_go_in;
logic bb0_1_go_out;
logic bb0_1_done_in;
logic bb0_1_done_out;
logic bb0_5_go_in;
logic bb0_5_go_out;
logic bb0_5_done_in;
logic bb0_5_done_out;
logic invoke0_go_in;
logic invoke0_go_out;
logic invoke0_done_in;
logic invoke0_done_out;
logic invoke3_go_in;
logic invoke3_go_out;
logic invoke3_done_in;
logic invoke3_done_out;
logic invoke6_go_in;
logic invoke6_go_out;
logic invoke6_done_in;
logic invoke6_done_out;
logic init_repeat_go_in;
logic init_repeat_go_out;
logic init_repeat_done_in;
logic init_repeat_done_out;
logic incr_repeat_go_in;
logic incr_repeat_go_out;
logic incr_repeat_done_in;
logic incr_repeat_done_out;
logic init_repeat0_go_in;
logic init_repeat0_go_out;
logic init_repeat0_done_in;
logic init_repeat0_done_out;
logic incr_repeat0_go_in;
logic incr_repeat0_go_out;
logic incr_repeat0_done_in;
logic incr_repeat0_done_out;
logic early_reset_static_seq_go_in;
logic early_reset_static_seq_go_out;
logic early_reset_static_seq_done_in;
logic early_reset_static_seq_done_out;
logic early_reset_static_seq0_go_in;
logic early_reset_static_seq0_go_out;
logic early_reset_static_seq0_done_in;
logic early_reset_static_seq0_done_out;
logic wrapper_early_reset_static_seq_go_in;
logic wrapper_early_reset_static_seq_go_out;
logic wrapper_early_reset_static_seq_done_in;
logic wrapper_early_reset_static_seq_done_out;
logic wrapper_early_reset_static_seq0_go_in;
logic wrapper_early_reset_static_seq0_go_out;
logic wrapper_early_reset_static_seq0_done_in;
logic wrapper_early_reset_static_seq0_done_out;
logic tdcc_go_in;
logic tdcc_go_out;
logic tdcc_done_in;
logic tdcc_done_out;
std_float_const # (
    .REP(0),
    .VALUE(0),
    .WIDTH(32)
) cst_0 (
    .out(cst_0_out)
);
std_slice # (
    .IN_WIDTH(32),
    .OUT_WIDTH(9)
) std_slice_3 (
    .in(std_slice_3_in),
    .out(std_slice_3_out)
);
std_add # (
    .WIDTH(32)
) std_add_1 (
    .left(std_add_1_left),
    .out(std_add_1_out),
    .right(std_add_1_right)
);
std_reg # (
    .WIDTH(32)
) load_0_reg (
    .clk(load_0_reg_clk),
    .done(load_0_reg_done),
    .in(load_0_reg_in),
    .out(load_0_reg_out),
    .reset(load_0_reg_reset),
    .write_en(load_0_reg_write_en)
);
std_mux # (
    .WIDTH(32)
) std_mux_0 (
    .cond(std_mux_0_cond),
    .fal(std_mux_0_fal),
    .out(std_mux_0_out),
    .tru(std_mux_0_tru)
);
std_and # (
    .WIDTH(1)
) std_and_0 (
    .left(std_and_0_left),
    .out(std_and_0_out),
    .right(std_and_0_right)
);
std_or # (
    .WIDTH(1)
) std_or_0 (
    .left(std_or_0_left),
    .out(std_or_0_out),
    .right(std_or_0_right)
);
std_reg # (
    .WIDTH(1)
) unordered_port_0_reg (
    .clk(unordered_port_0_reg_clk),
    .done(unordered_port_0_reg_done),
    .in(unordered_port_0_reg_in),
    .out(unordered_port_0_reg_out),
    .reset(unordered_port_0_reg_reset),
    .write_en(unordered_port_0_reg_write_en)
);
std_reg # (
    .WIDTH(1)
) compare_port_0_reg (
    .clk(compare_port_0_reg_clk),
    .done(compare_port_0_reg_done),
    .in(compare_port_0_reg_in),
    .out(compare_port_0_reg_out),
    .reset(compare_port_0_reg_reset),
    .write_en(compare_port_0_reg_write_en)
);
std_reg # (
    .WIDTH(1)
) cmpf_0_reg (
    .clk(cmpf_0_reg_clk),
    .done(cmpf_0_reg_done),
    .in(cmpf_0_reg_in),
    .out(cmpf_0_reg_out),
    .reset(cmpf_0_reg_reset),
    .write_en(cmpf_0_reg_write_en)
);
std_compareFN # (
    .expWidth(8),
    .numWidth(32),
    .sigWidth(24)
) std_compareFN_0 (
    .clk(std_compareFN_0_clk),
    .done(std_compareFN_0_done),
    .eq(std_compareFN_0_eq),
    .exceptionFlags(std_compareFN_0_exceptionFlags),
    .go(std_compareFN_0_go),
    .gt(std_compareFN_0_gt),
    .left(std_compareFN_0_left),
    .lt(std_compareFN_0_lt),
    .reset(std_compareFN_0_reset),
    .right(std_compareFN_0_right),
    .signaling(std_compareFN_0_signaling),
    .unordered(std_compareFN_0_unordered)
);
seq_mem_d1 # (
    .IDX_SIZE(9),
    .SIZE(300),
    .WIDTH(32)
) mem_0 (
    .addr0(mem_0_addr0),
    .clk(mem_0_clk),
    .content_en(mem_0_content_en),
    .done(mem_0_done),
    .read_data(mem_0_read_data),
    .reset(mem_0_reset),
    .write_data(mem_0_write_data),
    .write_en(mem_0_write_en)
);
std_reg # (
    .WIDTH(32)
) for_1_induction_var_reg (
    .clk(for_1_induction_var_reg_clk),
    .done(for_1_induction_var_reg_done),
    .in(for_1_induction_var_reg_in),
    .out(for_1_induction_var_reg_out),
    .reset(for_1_induction_var_reg_reset),
    .write_en(for_1_induction_var_reg_write_en)
);
std_reg # (
    .WIDTH(9)
) idx (
    .clk(idx_clk),
    .done(idx_done),
    .in(idx_in),
    .out(idx_out),
    .reset(idx_reset),
    .write_en(idx_write_en)
);
std_reg # (
    .WIDTH(1)
) cond_reg (
    .clk(cond_reg_clk),
    .done(cond_reg_done),
    .in(cond_reg_in),
    .out(cond_reg_out),
    .reset(cond_reg_reset),
    .write_en(cond_reg_write_en)
);
std_add # (
    .WIDTH(9)
) adder (
    .left(adder_left),
    .out(adder_out),
    .right(adder_right)
);
std_lt # (
    .WIDTH(9)
) lt (
    .left(lt_left),
    .out(lt_out),
    .right(lt_right)
);
std_reg # (
    .WIDTH(9)
) idx0 (
    .clk(idx0_clk),
    .done(idx0_done),
    .in(idx0_in),
    .out(idx0_out),
    .reset(idx0_reset),
    .write_en(idx0_write_en)
);
std_reg # (
    .WIDTH(1)
) cond_reg0 (
    .clk(cond_reg0_clk),
    .done(cond_reg0_done),
    .in(cond_reg0_in),
    .out(cond_reg0_out),
    .reset(cond_reg0_reset),
    .write_en(cond_reg0_write_en)
);
std_add # (
    .WIDTH(9)
) adder0 (
    .left(adder0_left),
    .out(adder0_out),
    .right(adder0_right)
);
std_lt # (
    .WIDTH(9)
) lt0 (
    .left(lt0_left),
    .out(lt0_out),
    .right(lt0_right)
);
std_reg # (
    .WIDTH(2)
) fsm (
    .clk(fsm_clk),
    .done(fsm_done),
    .in(fsm_in),
    .out(fsm_out),
    .reset(fsm_reset),
    .write_en(fsm_write_en)
);
std_add # (
    .WIDTH(2)
) adder1 (
    .left(adder1_left),
    .out(adder1_out),
    .right(adder1_right)
);
undef # (
    .WIDTH(1)
) ud (
    .out(ud_out)
);
std_add # (
    .WIDTH(2)
) adder2 (
    .left(adder2_left),
    .out(adder2_out),
    .right(adder2_right)
);
undef # (
    .WIDTH(1)
) ud0 (
    .out(ud0_out)
);
std_reg # (
    .WIDTH(1)
) signal_reg (
    .clk(signal_reg_clk),
    .done(signal_reg_done),
    .in(signal_reg_in),
    .out(signal_reg_out),
    .reset(signal_reg_reset),
    .write_en(signal_reg_write_en)
);
std_reg # (
    .WIDTH(4)
) fsm0 (
    .clk(fsm0_clk),
    .done(fsm0_done),
    .in(fsm0_in),
    .out(fsm0_out),
    .reset(fsm0_reset),
    .write_en(fsm0_write_en)
);
std_wire # (
    .WIDTH(1)
) bb0_0_go (
    .in(bb0_0_go_in),
    .out(bb0_0_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_0_done (
    .in(bb0_0_done_in),
    .out(bb0_0_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_1_go (
    .in(bb0_1_go_in),
    .out(bb0_1_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_1_done (
    .in(bb0_1_done_in),
    .out(bb0_1_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_5_go (
    .in(bb0_5_go_in),
    .out(bb0_5_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_5_done (
    .in(bb0_5_done_in),
    .out(bb0_5_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke0_go (
    .in(invoke0_go_in),
    .out(invoke0_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke0_done (
    .in(invoke0_done_in),
    .out(invoke0_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke3_go (
    .in(invoke3_go_in),
    .out(invoke3_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke3_done (
    .in(invoke3_done_in),
    .out(invoke3_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke6_go (
    .in(invoke6_go_in),
    .out(invoke6_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke6_done (
    .in(invoke6_done_in),
    .out(invoke6_done_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat_go (
    .in(init_repeat_go_in),
    .out(init_repeat_go_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat_done (
    .in(init_repeat_done_in),
    .out(init_repeat_done_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat_go (
    .in(incr_repeat_go_in),
    .out(incr_repeat_go_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat_done (
    .in(incr_repeat_done_in),
    .out(incr_repeat_done_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat0_go (
    .in(init_repeat0_go_in),
    .out(init_repeat0_go_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat0_done (
    .in(init_repeat0_done_in),
    .out(init_repeat0_done_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat0_go (
    .in(incr_repeat0_go_in),
    .out(incr_repeat0_go_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat0_done (
    .in(incr_repeat0_done_in),
    .out(incr_repeat0_done_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_static_seq_go (
    .in(early_reset_static_seq_go_in),
    .out(early_reset_static_seq_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_static_seq_done (
    .in(early_reset_static_seq_done_in),
    .out(early_reset_static_seq_done_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_static_seq0_go (
    .in(early_reset_static_seq0_go_in),
    .out(early_reset_static_seq0_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_static_seq0_done (
    .in(early_reset_static_seq0_done_in),
    .out(early_reset_static_seq0_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_seq_go (
    .in(wrapper_early_reset_static_seq_go_in),
    .out(wrapper_early_reset_static_seq_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_seq_done (
    .in(wrapper_early_reset_static_seq_done_in),
    .out(wrapper_early_reset_static_seq_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_seq0_go (
    .in(wrapper_early_reset_static_seq0_go_in),
    .out(wrapper_early_reset_static_seq0_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_seq0_done (
    .in(wrapper_early_reset_static_seq0_done_in),
    .out(wrapper_early_reset_static_seq0_done_out)
);
std_wire # (
    .WIDTH(1)
) tdcc_go (
    .in(tdcc_go_in),
    .out(tdcc_go_out)
);
std_wire # (
    .WIDTH(1)
) tdcc_done (
    .in(tdcc_done_in),
    .out(tdcc_done_out)
);
wire _guard0 = 1;
wire _guard1 = bb0_1_go_out;
wire _guard2 = bb0_1_go_out;
wire _guard3 = early_reset_static_seq_go_out;
wire _guard4 = early_reset_static_seq_go_out;
wire _guard5 = init_repeat_done_out;
wire _guard6 = ~_guard5;
wire _guard7 = fsm0_out == 4'd1;
wire _guard8 = _guard6 & _guard7;
wire _guard9 = tdcc_go_out;
wire _guard10 = _guard8 & _guard9;
wire _guard11 = tdcc_done_out;
wire _guard12 = bb0_5_go_out;
wire _guard13 = bb0_0_go_out;
wire _guard14 = bb0_0_go_out;
wire _guard15 = bb0_5_go_out;
wire _guard16 = bb0_5_go_out;
wire _guard17 = bb0_5_go_out;
wire _guard18 = incr_repeat_go_out;
wire _guard19 = incr_repeat_go_out;
wire _guard20 = fsm_out != 2'd1;
wire _guard21 = early_reset_static_seq_go_out;
wire _guard22 = _guard20 & _guard21;
wire _guard23 = fsm_out == 2'd1;
wire _guard24 = early_reset_static_seq_go_out;
wire _guard25 = _guard23 & _guard24;
wire _guard26 = _guard22 | _guard25;
wire _guard27 = fsm_out != 2'd1;
wire _guard28 = early_reset_static_seq0_go_out;
wire _guard29 = _guard27 & _guard28;
wire _guard30 = _guard26 | _guard29;
wire _guard31 = fsm_out == 2'd1;
wire _guard32 = early_reset_static_seq0_go_out;
wire _guard33 = _guard31 & _guard32;
wire _guard34 = _guard30 | _guard33;
wire _guard35 = fsm_out != 2'd1;
wire _guard36 = early_reset_static_seq_go_out;
wire _guard37 = _guard35 & _guard36;
wire _guard38 = fsm_out == 2'd1;
wire _guard39 = early_reset_static_seq_go_out;
wire _guard40 = _guard38 & _guard39;
wire _guard41 = fsm_out == 2'd1;
wire _guard42 = early_reset_static_seq0_go_out;
wire _guard43 = _guard41 & _guard42;
wire _guard44 = _guard40 | _guard43;
wire _guard45 = fsm_out != 2'd1;
wire _guard46 = early_reset_static_seq0_go_out;
wire _guard47 = _guard45 & _guard46;
wire _guard48 = bb0_0_done_out;
wire _guard49 = ~_guard48;
wire _guard50 = fsm0_out == 4'd2;
wire _guard51 = _guard49 & _guard50;
wire _guard52 = tdcc_go_out;
wire _guard53 = _guard51 & _guard52;
wire _guard54 = fsm_out == 2'd0;
wire _guard55 = early_reset_static_seq_go_out;
wire _guard56 = _guard54 & _guard55;
wire _guard57 = fsm_out == 2'd0;
wire _guard58 = early_reset_static_seq_go_out;
wire _guard59 = _guard57 & _guard58;
wire _guard60 = fsm_out == 2'd0;
wire _guard61 = early_reset_static_seq_go_out;
wire _guard62 = _guard60 & _guard61;
wire _guard63 = invoke0_go_out;
wire _guard64 = fsm_out == 2'd1;
wire _guard65 = early_reset_static_seq_go_out;
wire _guard66 = _guard64 & _guard65;
wire _guard67 = _guard63 | _guard66;
wire _guard68 = fsm_out == 2'd1;
wire _guard69 = early_reset_static_seq0_go_out;
wire _guard70 = _guard68 & _guard69;
wire _guard71 = _guard67 | _guard70;
wire _guard72 = invoke0_go_out;
wire _guard73 = fsm_out == 2'd1;
wire _guard74 = early_reset_static_seq_go_out;
wire _guard75 = _guard73 & _guard74;
wire _guard76 = fsm_out == 2'd1;
wire _guard77 = early_reset_static_seq0_go_out;
wire _guard78 = _guard76 & _guard77;
wire _guard79 = init_repeat0_go_out;
wire _guard80 = incr_repeat0_go_out;
wire _guard81 = _guard79 | _guard80;
wire _guard82 = init_repeat0_go_out;
wire _guard83 = incr_repeat0_go_out;
wire _guard84 = bb0_1_go_out;
wire _guard85 = bb0_1_go_out;
wire _guard86 = wrapper_early_reset_static_seq0_go_out;
wire _guard87 = bb0_0_go_out;
wire _guard88 = fsm_out == 2'd0;
wire _guard89 = early_reset_static_seq_go_out;
wire _guard90 = _guard88 & _guard89;
wire _guard91 = _guard87 | _guard90;
wire _guard92 = bb0_5_go_out;
wire _guard93 = fsm_out == 2'd0;
wire _guard94 = early_reset_static_seq0_go_out;
wire _guard95 = _guard93 & _guard94;
wire _guard96 = _guard92 | _guard95;
wire _guard97 = bb0_1_go_out;
wire _guard98 = bb0_1_go_out;
wire _guard99 = init_repeat0_go_out;
wire _guard100 = incr_repeat0_go_out;
wire _guard101 = _guard99 | _guard100;
wire _guard102 = incr_repeat0_go_out;
wire _guard103 = init_repeat0_go_out;
wire _guard104 = wrapper_early_reset_static_seq0_done_out;
wire _guard105 = ~_guard104;
wire _guard106 = fsm0_out == 4'd8;
wire _guard107 = _guard105 & _guard106;
wire _guard108 = tdcc_go_out;
wire _guard109 = _guard107 & _guard108;
wire _guard110 = fsm_out == 2'd1;
wire _guard111 = early_reset_static_seq_go_out;
wire _guard112 = _guard110 & _guard111;
wire _guard113 = invoke6_go_out;
wire _guard114 = invoke6_go_out;
wire _guard115 = fsm_out == 2'd1;
wire _guard116 = early_reset_static_seq_go_out;
wire _guard117 = _guard115 & _guard116;
wire _guard118 = _guard114 | _guard117;
wire _guard119 = bb0_1_go_out;
wire _guard120 = bb0_1_go_out;
wire _guard121 = invoke0_done_out;
wire _guard122 = ~_guard121;
wire _guard123 = fsm0_out == 4'd0;
wire _guard124 = _guard122 & _guard123;
wire _guard125 = tdcc_go_out;
wire _guard126 = _guard124 & _guard125;
wire _guard127 = incr_repeat_done_out;
wire _guard128 = ~_guard127;
wire _guard129 = fsm0_out == 4'd5;
wire _guard130 = _guard128 & _guard129;
wire _guard131 = tdcc_go_out;
wire _guard132 = _guard130 & _guard131;
wire _guard133 = early_reset_static_seq0_go_out;
wire _guard134 = early_reset_static_seq0_go_out;
wire _guard135 = fsm0_out == 4'd12;
wire _guard136 = fsm0_out == 4'd0;
wire _guard137 = invoke0_done_out;
wire _guard138 = _guard136 & _guard137;
wire _guard139 = tdcc_go_out;
wire _guard140 = _guard138 & _guard139;
wire _guard141 = _guard135 | _guard140;
wire _guard142 = fsm0_out == 4'd1;
wire _guard143 = init_repeat_done_out;
wire _guard144 = cond_reg_out;
wire _guard145 = _guard143 & _guard144;
wire _guard146 = _guard142 & _guard145;
wire _guard147 = tdcc_go_out;
wire _guard148 = _guard146 & _guard147;
wire _guard149 = _guard141 | _guard148;
wire _guard150 = fsm0_out == 4'd5;
wire _guard151 = incr_repeat_done_out;
wire _guard152 = cond_reg_out;
wire _guard153 = _guard151 & _guard152;
wire _guard154 = _guard150 & _guard153;
wire _guard155 = tdcc_go_out;
wire _guard156 = _guard154 & _guard155;
wire _guard157 = _guard149 | _guard156;
wire _guard158 = fsm0_out == 4'd2;
wire _guard159 = bb0_0_done_out;
wire _guard160 = _guard158 & _guard159;
wire _guard161 = tdcc_go_out;
wire _guard162 = _guard160 & _guard161;
wire _guard163 = _guard157 | _guard162;
wire _guard164 = fsm0_out == 4'd3;
wire _guard165 = bb0_1_done_out;
wire _guard166 = _guard164 & _guard165;
wire _guard167 = tdcc_go_out;
wire _guard168 = _guard166 & _guard167;
wire _guard169 = _guard163 | _guard168;
wire _guard170 = fsm0_out == 4'd4;
wire _guard171 = wrapper_early_reset_static_seq_done_out;
wire _guard172 = _guard170 & _guard171;
wire _guard173 = tdcc_go_out;
wire _guard174 = _guard172 & _guard173;
wire _guard175 = _guard169 | _guard174;
wire _guard176 = fsm0_out == 4'd1;
wire _guard177 = init_repeat_done_out;
wire _guard178 = cond_reg_out;
wire _guard179 = ~_guard178;
wire _guard180 = _guard177 & _guard179;
wire _guard181 = _guard176 & _guard180;
wire _guard182 = tdcc_go_out;
wire _guard183 = _guard181 & _guard182;
wire _guard184 = _guard175 | _guard183;
wire _guard185 = fsm0_out == 4'd5;
wire _guard186 = incr_repeat_done_out;
wire _guard187 = cond_reg_out;
wire _guard188 = ~_guard187;
wire _guard189 = _guard186 & _guard188;
wire _guard190 = _guard185 & _guard189;
wire _guard191 = tdcc_go_out;
wire _guard192 = _guard190 & _guard191;
wire _guard193 = _guard184 | _guard192;
wire _guard194 = fsm0_out == 4'd6;
wire _guard195 = invoke3_done_out;
wire _guard196 = _guard194 & _guard195;
wire _guard197 = tdcc_go_out;
wire _guard198 = _guard196 & _guard197;
wire _guard199 = _guard193 | _guard198;
wire _guard200 = fsm0_out == 4'd7;
wire _guard201 = init_repeat0_done_out;
wire _guard202 = cond_reg0_out;
wire _guard203 = _guard201 & _guard202;
wire _guard204 = _guard200 & _guard203;
wire _guard205 = tdcc_go_out;
wire _guard206 = _guard204 & _guard205;
wire _guard207 = _guard199 | _guard206;
wire _guard208 = fsm0_out == 4'd11;
wire _guard209 = incr_repeat0_done_out;
wire _guard210 = cond_reg0_out;
wire _guard211 = _guard209 & _guard210;
wire _guard212 = _guard208 & _guard211;
wire _guard213 = tdcc_go_out;
wire _guard214 = _guard212 & _guard213;
wire _guard215 = _guard207 | _guard214;
wire _guard216 = fsm0_out == 4'd8;
wire _guard217 = wrapper_early_reset_static_seq0_done_out;
wire _guard218 = _guard216 & _guard217;
wire _guard219 = tdcc_go_out;
wire _guard220 = _guard218 & _guard219;
wire _guard221 = _guard215 | _guard220;
wire _guard222 = fsm0_out == 4'd9;
wire _guard223 = bb0_5_done_out;
wire _guard224 = _guard222 & _guard223;
wire _guard225 = tdcc_go_out;
wire _guard226 = _guard224 & _guard225;
wire _guard227 = _guard221 | _guard226;
wire _guard228 = fsm0_out == 4'd10;
wire _guard229 = invoke6_done_out;
wire _guard230 = _guard228 & _guard229;
wire _guard231 = tdcc_go_out;
wire _guard232 = _guard230 & _guard231;
wire _guard233 = _guard227 | _guard232;
wire _guard234 = fsm0_out == 4'd7;
wire _guard235 = init_repeat0_done_out;
wire _guard236 = cond_reg0_out;
wire _guard237 = ~_guard236;
wire _guard238 = _guard235 & _guard237;
wire _guard239 = _guard234 & _guard238;
wire _guard240 = tdcc_go_out;
wire _guard241 = _guard239 & _guard240;
wire _guard242 = _guard233 | _guard241;
wire _guard243 = fsm0_out == 4'd11;
wire _guard244 = incr_repeat0_done_out;
wire _guard245 = cond_reg0_out;
wire _guard246 = ~_guard245;
wire _guard247 = _guard244 & _guard246;
wire _guard248 = _guard243 & _guard247;
wire _guard249 = tdcc_go_out;
wire _guard250 = _guard248 & _guard249;
wire _guard251 = _guard242 | _guard250;
wire _guard252 = fsm0_out == 4'd2;
wire _guard253 = bb0_0_done_out;
wire _guard254 = _guard252 & _guard253;
wire _guard255 = tdcc_go_out;
wire _guard256 = _guard254 & _guard255;
wire _guard257 = fsm0_out == 4'd4;
wire _guard258 = wrapper_early_reset_static_seq_done_out;
wire _guard259 = _guard257 & _guard258;
wire _guard260 = tdcc_go_out;
wire _guard261 = _guard259 & _guard260;
wire _guard262 = fsm0_out == 4'd3;
wire _guard263 = bb0_1_done_out;
wire _guard264 = _guard262 & _guard263;
wire _guard265 = tdcc_go_out;
wire _guard266 = _guard264 & _guard265;
wire _guard267 = fsm0_out == 4'd8;
wire _guard268 = wrapper_early_reset_static_seq0_done_out;
wire _guard269 = _guard267 & _guard268;
wire _guard270 = tdcc_go_out;
wire _guard271 = _guard269 & _guard270;
wire _guard272 = fsm0_out == 4'd9;
wire _guard273 = bb0_5_done_out;
wire _guard274 = _guard272 & _guard273;
wire _guard275 = tdcc_go_out;
wire _guard276 = _guard274 & _guard275;
wire _guard277 = fsm0_out == 4'd12;
wire _guard278 = fsm0_out == 4'd6;
wire _guard279 = invoke3_done_out;
wire _guard280 = _guard278 & _guard279;
wire _guard281 = tdcc_go_out;
wire _guard282 = _guard280 & _guard281;
wire _guard283 = fsm0_out == 4'd1;
wire _guard284 = init_repeat_done_out;
wire _guard285 = cond_reg_out;
wire _guard286 = _guard284 & _guard285;
wire _guard287 = _guard283 & _guard286;
wire _guard288 = tdcc_go_out;
wire _guard289 = _guard287 & _guard288;
wire _guard290 = fsm0_out == 4'd5;
wire _guard291 = incr_repeat_done_out;
wire _guard292 = cond_reg_out;
wire _guard293 = _guard291 & _guard292;
wire _guard294 = _guard290 & _guard293;
wire _guard295 = tdcc_go_out;
wire _guard296 = _guard294 & _guard295;
wire _guard297 = _guard289 | _guard296;
wire _guard298 = fsm0_out == 4'd1;
wire _guard299 = init_repeat_done_out;
wire _guard300 = cond_reg_out;
wire _guard301 = ~_guard300;
wire _guard302 = _guard299 & _guard301;
wire _guard303 = _guard298 & _guard302;
wire _guard304 = tdcc_go_out;
wire _guard305 = _guard303 & _guard304;
wire _guard306 = fsm0_out == 4'd5;
wire _guard307 = incr_repeat_done_out;
wire _guard308 = cond_reg_out;
wire _guard309 = ~_guard308;
wire _guard310 = _guard307 & _guard309;
wire _guard311 = _guard306 & _guard310;
wire _guard312 = tdcc_go_out;
wire _guard313 = _guard311 & _guard312;
wire _guard314 = _guard305 | _guard313;
wire _guard315 = fsm0_out == 4'd7;
wire _guard316 = init_repeat0_done_out;
wire _guard317 = cond_reg0_out;
wire _guard318 = _guard316 & _guard317;
wire _guard319 = _guard315 & _guard318;
wire _guard320 = tdcc_go_out;
wire _guard321 = _guard319 & _guard320;
wire _guard322 = fsm0_out == 4'd11;
wire _guard323 = incr_repeat0_done_out;
wire _guard324 = cond_reg0_out;
wire _guard325 = _guard323 & _guard324;
wire _guard326 = _guard322 & _guard325;
wire _guard327 = tdcc_go_out;
wire _guard328 = _guard326 & _guard327;
wire _guard329 = _guard321 | _guard328;
wire _guard330 = fsm0_out == 4'd7;
wire _guard331 = init_repeat0_done_out;
wire _guard332 = cond_reg0_out;
wire _guard333 = ~_guard332;
wire _guard334 = _guard331 & _guard333;
wire _guard335 = _guard330 & _guard334;
wire _guard336 = tdcc_go_out;
wire _guard337 = _guard335 & _guard336;
wire _guard338 = fsm0_out == 4'd11;
wire _guard339 = incr_repeat0_done_out;
wire _guard340 = cond_reg0_out;
wire _guard341 = ~_guard340;
wire _guard342 = _guard339 & _guard341;
wire _guard343 = _guard338 & _guard342;
wire _guard344 = tdcc_go_out;
wire _guard345 = _guard343 & _guard344;
wire _guard346 = _guard337 | _guard345;
wire _guard347 = fsm0_out == 4'd0;
wire _guard348 = invoke0_done_out;
wire _guard349 = _guard347 & _guard348;
wire _guard350 = tdcc_go_out;
wire _guard351 = _guard349 & _guard350;
wire _guard352 = fsm0_out == 4'd10;
wire _guard353 = invoke6_done_out;
wire _guard354 = _guard352 & _guard353;
wire _guard355 = tdcc_go_out;
wire _guard356 = _guard354 & _guard355;
wire _guard357 = cond_reg0_done;
wire _guard358 = idx0_done;
wire _guard359 = _guard357 & _guard358;
wire _guard360 = init_repeat_go_out;
wire _guard361 = incr_repeat_go_out;
wire _guard362 = _guard360 | _guard361;
wire _guard363 = incr_repeat_go_out;
wire _guard364 = init_repeat_go_out;
wire _guard365 = cond_reg_done;
wire _guard366 = idx_done;
wire _guard367 = _guard365 & _guard366;
wire _guard368 = cond_reg_done;
wire _guard369 = idx_done;
wire _guard370 = _guard368 & _guard369;
wire _guard371 = cond_reg0_done;
wire _guard372 = idx0_done;
wire _guard373 = _guard371 & _guard372;
wire _guard374 = signal_reg_out;
wire _guard375 = incr_repeat0_go_out;
wire _guard376 = incr_repeat0_go_out;
wire _guard377 = init_repeat0_done_out;
wire _guard378 = ~_guard377;
wire _guard379 = fsm0_out == 4'd7;
wire _guard380 = _guard378 & _guard379;
wire _guard381 = tdcc_go_out;
wire _guard382 = _guard380 & _guard381;
wire _guard383 = incr_repeat0_done_out;
wire _guard384 = ~_guard383;
wire _guard385 = fsm0_out == 4'd11;
wire _guard386 = _guard384 & _guard385;
wire _guard387 = tdcc_go_out;
wire _guard388 = _guard386 & _guard387;
wire _guard389 = wrapper_early_reset_static_seq_go_out;
wire _guard390 = fsm_out == 2'd0;
wire _guard391 = early_reset_static_seq_go_out;
wire _guard392 = _guard390 & _guard391;
wire _guard393 = fsm_out == 2'd0;
wire _guard394 = early_reset_static_seq_go_out;
wire _guard395 = _guard393 & _guard394;
wire _guard396 = fsm_out == 2'd0;
wire _guard397 = early_reset_static_seq0_go_out;
wire _guard398 = _guard396 & _guard397;
wire _guard399 = _guard395 | _guard398;
wire _guard400 = fsm_out == 2'd0;
wire _guard401 = early_reset_static_seq_go_out;
wire _guard402 = _guard400 & _guard401;
wire _guard403 = fsm_out == 2'd0;
wire _guard404 = early_reset_static_seq0_go_out;
wire _guard405 = _guard403 & _guard404;
wire _guard406 = _guard402 | _guard405;
wire _guard407 = fsm_out == 2'd0;
wire _guard408 = early_reset_static_seq_go_out;
wire _guard409 = _guard407 & _guard408;
wire _guard410 = signal_reg_out;
wire _guard411 = fsm_out == 2'd1;
wire _guard412 = _guard411 & _guard0;
wire _guard413 = signal_reg_out;
wire _guard414 = ~_guard413;
wire _guard415 = _guard412 & _guard414;
wire _guard416 = wrapper_early_reset_static_seq_go_out;
wire _guard417 = _guard415 & _guard416;
wire _guard418 = _guard410 | _guard417;
wire _guard419 = fsm_out == 2'd1;
wire _guard420 = _guard419 & _guard0;
wire _guard421 = signal_reg_out;
wire _guard422 = ~_guard421;
wire _guard423 = _guard420 & _guard422;
wire _guard424 = wrapper_early_reset_static_seq0_go_out;
wire _guard425 = _guard423 & _guard424;
wire _guard426 = _guard418 | _guard425;
wire _guard427 = fsm_out == 2'd1;
wire _guard428 = _guard427 & _guard0;
wire _guard429 = signal_reg_out;
wire _guard430 = ~_guard429;
wire _guard431 = _guard428 & _guard430;
wire _guard432 = wrapper_early_reset_static_seq_go_out;
wire _guard433 = _guard431 & _guard432;
wire _guard434 = fsm_out == 2'd1;
wire _guard435 = _guard434 & _guard0;
wire _guard436 = signal_reg_out;
wire _guard437 = ~_guard436;
wire _guard438 = _guard435 & _guard437;
wire _guard439 = wrapper_early_reset_static_seq0_go_out;
wire _guard440 = _guard438 & _guard439;
wire _guard441 = _guard433 | _guard440;
wire _guard442 = signal_reg_out;
wire _guard443 = bb0_1_go_out;
wire _guard444 = bb0_1_go_out;
wire _guard445 = fsm0_out == 4'd12;
wire _guard446 = invoke3_go_out;
wire _guard447 = invoke6_go_out;
wire _guard448 = _guard446 | _guard447;
wire _guard449 = invoke3_go_out;
wire _guard450 = invoke6_go_out;
wire _guard451 = incr_repeat_go_out;
wire _guard452 = incr_repeat_go_out;
wire _guard453 = invoke3_done_out;
wire _guard454 = ~_guard453;
wire _guard455 = fsm0_out == 4'd6;
wire _guard456 = _guard454 & _guard455;
wire _guard457 = tdcc_go_out;
wire _guard458 = _guard456 & _guard457;
wire _guard459 = bb0_1_done_out;
wire _guard460 = ~_guard459;
wire _guard461 = fsm0_out == 4'd3;
wire _guard462 = _guard460 & _guard461;
wire _guard463 = tdcc_go_out;
wire _guard464 = _guard462 & _guard463;
wire _guard465 = invoke6_done_out;
wire _guard466 = ~_guard465;
wire _guard467 = fsm0_out == 4'd10;
wire _guard468 = _guard466 & _guard467;
wire _guard469 = tdcc_go_out;
wire _guard470 = _guard468 & _guard469;
wire _guard471 = wrapper_early_reset_static_seq_done_out;
wire _guard472 = ~_guard471;
wire _guard473 = fsm0_out == 4'd4;
wire _guard474 = _guard472 & _guard473;
wire _guard475 = tdcc_go_out;
wire _guard476 = _guard474 & _guard475;
wire _guard477 = signal_reg_out;
wire _guard478 = bb0_1_go_out;
wire _guard479 = std_compareFN_0_done;
wire _guard480 = ~_guard479;
wire _guard481 = bb0_1_go_out;
wire _guard482 = _guard480 & _guard481;
wire _guard483 = bb0_1_go_out;
wire _guard484 = bb0_1_go_out;
wire _guard485 = init_repeat_go_out;
wire _guard486 = incr_repeat_go_out;
wire _guard487 = _guard485 | _guard486;
wire _guard488 = init_repeat_go_out;
wire _guard489 = incr_repeat_go_out;
wire _guard490 = incr_repeat0_go_out;
wire _guard491 = incr_repeat0_go_out;
wire _guard492 = bb0_5_done_out;
wire _guard493 = ~_guard492;
wire _guard494 = fsm0_out == 4'd9;
wire _guard495 = _guard493 & _guard494;
wire _guard496 = tdcc_go_out;
wire _guard497 = _guard495 & _guard496;
assign unordered_port_0_reg_write_en =
  _guard1 ? std_compareFN_0_done :
  1'd0;
assign unordered_port_0_reg_clk = clk;
assign unordered_port_0_reg_reset = reset;
assign unordered_port_0_reg_in = std_compareFN_0_unordered;
assign adder1_left =
  _guard3 ? fsm_out :
  2'd0;
assign adder1_right =
  _guard4 ? 2'd1 :
  2'd0;
assign init_repeat_go_in = _guard10;
assign done = _guard11;
assign arg_mem_1_write_data = load_0_reg_out;
assign arg_mem_0_content_en = _guard13;
assign arg_mem_0_addr0 = std_slice_3_out;
assign arg_mem_1_write_en = _guard15;
assign arg_mem_1_addr0 = std_slice_3_out;
assign arg_mem_1_content_en = _guard17;
assign adder_left =
  _guard18 ? idx_out :
  9'd0;
assign adder_right =
  _guard19 ? 9'd1 :
  9'd0;
assign fsm_write_en = _guard34;
assign fsm_clk = clk;
assign fsm_reset = reset;
assign fsm_in =
  _guard37 ? adder1_out :
  _guard44 ? 2'd0 :
  _guard47 ? adder2_out :
  2'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard47, _guard44, _guard37})) begin
    $fatal(2, "Multiple assignment to port `fsm.in'.");
end
end
assign bb0_0_go_in = _guard53;
assign std_mux_0_cond = cmpf_0_reg_out;
assign std_mux_0_tru = arg_mem_0_read_data;
assign std_mux_0_fal = cst_0_out;
assign load_0_reg_write_en = _guard71;
assign load_0_reg_clk = clk;
assign load_0_reg_reset = reset;
assign load_0_reg_in =
  _guard72 ? 32'd0 :
  _guard75 ? std_add_1_out :
  _guard78 ? mem_0_read_data :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard78, _guard75, _guard72})) begin
    $fatal(2, "Multiple assignment to port `load_0_reg.in'.");
end
end
assign cond_reg0_write_en = _guard81;
assign cond_reg0_clk = clk;
assign cond_reg0_reset = reset;
assign cond_reg0_in =
  _guard82 ? 1'd1 :
  _guard83 ? lt0_out :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard83, _guard82})) begin
    $fatal(2, "Multiple assignment to port `cond_reg0.in'.");
end
end
assign std_and_0_left =
  _guard84 ? compare_port_0_reg_done :
  1'd0;
assign std_and_0_right =
  _guard85 ? unordered_port_0_reg_done :
  1'd0;
assign early_reset_static_seq0_go_in = _guard86;
assign std_slice_3_in =
  _guard91 ? load_0_reg_out :
  _guard96 ? for_1_induction_var_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard96, _guard91})) begin
    $fatal(2, "Multiple assignment to port `std_slice_3.in'.");
end
end
assign compare_port_0_reg_write_en =
  _guard97 ? std_compareFN_0_done :
  1'd0;
assign compare_port_0_reg_clk = clk;
assign compare_port_0_reg_reset = reset;
assign compare_port_0_reg_in = std_compareFN_0_gt;
assign idx0_write_en = _guard101;
assign idx0_clk = clk;
assign idx0_reset = reset;
assign idx0_in =
  _guard102 ? adder0_out :
  _guard103 ? 9'd0 :
  9'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard103, _guard102})) begin
    $fatal(2, "Multiple assignment to port `idx0.in'.");
end
end
assign wrapper_early_reset_static_seq0_go_in = _guard109;
assign std_add_1_left =
  _guard112 ? load_0_reg_out :
  _guard113 ? for_1_induction_var_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard113, _guard112})) begin
    $fatal(2, "Multiple assignment to port `std_add_1.left'.");
end
end
assign std_add_1_right = 32'd1;
assign std_or_0_left = compare_port_0_reg_out;
assign std_or_0_right = unordered_port_0_reg_out;
assign invoke0_go_in = _guard126;
assign bb0_0_done_in = arg_mem_0_done;
assign incr_repeat_go_in = _guard132;
assign tdcc_go_in = go;
assign adder2_left =
  _guard133 ? fsm_out :
  2'd0;
assign adder2_right =
  _guard134 ? 2'd1 :
  2'd0;
assign fsm0_write_en = _guard251;
assign fsm0_clk = clk;
assign fsm0_reset = reset;
assign fsm0_in =
  _guard256 ? 4'd3 :
  _guard261 ? 4'd5 :
  _guard266 ? 4'd4 :
  _guard271 ? 4'd9 :
  _guard276 ? 4'd10 :
  _guard277 ? 4'd0 :
  _guard282 ? 4'd7 :
  _guard297 ? 4'd2 :
  _guard314 ? 4'd6 :
  _guard329 ? 4'd8 :
  _guard346 ? 4'd12 :
  _guard351 ? 4'd1 :
  _guard356 ? 4'd11 :
  4'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard356, _guard351, _guard346, _guard329, _guard314, _guard297, _guard282, _guard277, _guard276, _guard271, _guard266, _guard261, _guard256})) begin
    $fatal(2, "Multiple assignment to port `fsm0.in'.");
end
end
assign invoke3_done_in = for_1_induction_var_reg_done;
assign incr_repeat0_done_in = _guard359;
assign idx_write_en = _guard362;
assign idx_clk = clk;
assign idx_reset = reset;
assign idx_in =
  _guard363 ? adder_out :
  _guard364 ? 9'd0 :
  9'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard364, _guard363})) begin
    $fatal(2, "Multiple assignment to port `idx.in'.");
end
end
assign init_repeat_done_in = _guard367;
assign incr_repeat_done_in = _guard370;
assign init_repeat0_done_in = _guard373;
assign wrapper_early_reset_static_seq_done_in = _guard374;
assign adder0_left =
  _guard375 ? idx0_out :
  9'd0;
assign adder0_right =
  _guard376 ? 9'd1 :
  9'd0;
assign invoke0_done_in = load_0_reg_done;
assign invoke6_done_in = for_1_induction_var_reg_done;
assign init_repeat0_go_in = _guard382;
assign incr_repeat0_go_in = _guard388;
assign early_reset_static_seq_go_in = _guard389;
assign mem_0_write_en = _guard392;
assign mem_0_clk = clk;
assign mem_0_addr0 = std_slice_3_out;
assign mem_0_content_en = _guard406;
assign mem_0_reset = reset;
assign mem_0_write_data = std_mux_0_out;
assign signal_reg_write_en = _guard426;
assign signal_reg_clk = clk;
assign signal_reg_reset = reset;
assign signal_reg_in =
  _guard441 ? 1'd1 :
  _guard442 ? 1'd0 :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard442, _guard441})) begin
    $fatal(2, "Multiple assignment to port `signal_reg.in'.");
end
end
assign bb0_5_done_in = arg_mem_1_done;
assign cmpf_0_reg_write_en =
  _guard443 ? std_and_0_out :
  1'd0;
assign cmpf_0_reg_clk = clk;
assign cmpf_0_reg_reset = reset;
assign cmpf_0_reg_in = std_or_0_out;
assign bb0_1_done_in = cmpf_0_reg_done;
assign early_reset_static_seq_done_in = ud_out;
assign tdcc_done_in = _guard445;
assign for_1_induction_var_reg_write_en = _guard448;
assign for_1_induction_var_reg_clk = clk;
assign for_1_induction_var_reg_reset = reset;
assign for_1_induction_var_reg_in =
  _guard449 ? 32'd0 :
  _guard450 ? std_add_1_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard450, _guard449})) begin
    $fatal(2, "Multiple assignment to port `for_1_induction_var_reg.in'.");
end
end
assign lt_left =
  _guard451 ? adder_out :
  9'd0;
assign lt_right =
  _guard452 ? 9'd300 :
  9'd0;
assign invoke3_go_in = _guard458;
assign bb0_1_go_in = _guard464;
assign invoke6_go_in = _guard470;
assign early_reset_static_seq0_done_in = ud0_out;
assign wrapper_early_reset_static_seq_go_in = _guard476;
assign wrapper_early_reset_static_seq0_done_in = _guard477;
assign std_compareFN_0_clk = clk;
assign std_compareFN_0_left =
  _guard478 ? arg_mem_0_read_data :
  32'd0;
assign std_compareFN_0_reset = reset;
assign std_compareFN_0_go = _guard482;
assign std_compareFN_0_signaling = _guard483;
assign std_compareFN_0_right =
  _guard484 ? cst_0_out :
  32'd0;
assign cond_reg_write_en = _guard487;
assign cond_reg_clk = clk;
assign cond_reg_reset = reset;
assign cond_reg_in =
  _guard488 ? 1'd1 :
  _guard489 ? lt_out :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard489, _guard488})) begin
    $fatal(2, "Multiple assignment to port `cond_reg.in'.");
end
end
assign lt0_left =
  _guard490 ? adder0_out :
  9'd0;
assign lt0_right =
  _guard491 ? 9'd300 :
  9'd0;
assign bb0_5_go_in = _guard497;
// COMPONENT END: relu4d_0
endmodule
module forward(
  input logic clk,
  input logic reset,
  input logic go,
  output logic done,
  output logic [8:0] arg_mem_4_addr0,
  output logic arg_mem_4_content_en,
  output logic arg_mem_4_write_en,
  output logic [31:0] arg_mem_4_write_data,
  input logic [31:0] arg_mem_4_read_data,
  input logic arg_mem_4_done,
  output logic [8:0] arg_mem_3_addr0,
  output logic arg_mem_3_content_en,
  output logic arg_mem_3_write_en,
  output logic [31:0] arg_mem_3_write_data,
  input logic [31:0] arg_mem_3_read_data,
  input logic arg_mem_3_done,
  output logic [8:0] arg_mem_2_addr0,
  output logic arg_mem_2_content_en,
  output logic arg_mem_2_write_en,
  output logic [31:0] arg_mem_2_write_data,
  input logic [31:0] arg_mem_2_read_data,
  input logic arg_mem_2_done,
  output logic [8:0] arg_mem_1_addr0,
  output logic arg_mem_1_content_en,
  output logic arg_mem_1_write_en,
  output logic [31:0] arg_mem_1_write_data,
  input logic [31:0] arg_mem_1_read_data,
  input logic arg_mem_1_done,
  output logic [8:0] arg_mem_0_addr0,
  output logic arg_mem_0_content_en,
  output logic arg_mem_0_write_en,
  output logic [31:0] arg_mem_0_write_data,
  input logic [31:0] arg_mem_0_read_data,
  input logic arg_mem_0_done
);
// COMPONENT START: forward
logic [31:0] std_slice_4_in;
logic [8:0] std_slice_4_out;
logic [31:0] std_add_1_left;
logic [31:0] std_add_1_right;
logic [31:0] std_add_1_out;
logic [31:0] addf_0_reg_in;
logic addf_0_reg_write_en;
logic addf_0_reg_clk;
logic addf_0_reg_reset;
logic [31:0] addf_0_reg_out;
logic addf_0_reg_done;
logic std_addFN_0_clk;
logic std_addFN_0_reset;
logic std_addFN_0_go;
logic std_addFN_0_control;
logic std_addFN_0_subOp;
logic [31:0] std_addFN_0_left;
logic [31:0] std_addFN_0_right;
logic [2:0] std_addFN_0_roundingMode;
logic [31:0] std_addFN_0_out;
logic [4:0] std_addFN_0_exceptionFlags;
logic std_addFN_0_done;
logic [31:0] for_0_induction_var_reg_in;
logic for_0_induction_var_reg_write_en;
logic for_0_induction_var_reg_clk;
logic for_0_induction_var_reg_reset;
logic [31:0] for_0_induction_var_reg_out;
logic for_0_induction_var_reg_done;
logic relu4d_0_instance_clk;
logic relu4d_0_instance_reset;
logic relu4d_0_instance_go;
logic relu4d_0_instance_done;
logic [31:0] relu4d_0_instance_arg_mem_0_read_data;
logic relu4d_0_instance_arg_mem_0_done;
logic [31:0] relu4d_0_instance_arg_mem_1_write_data;
logic [31:0] relu4d_0_instance_arg_mem_1_read_data;
logic relu4d_0_instance_arg_mem_0_content_en;
logic [8:0] relu4d_0_instance_arg_mem_0_addr0;
logic relu4d_0_instance_arg_mem_0_write_en;
logic relu4d_0_instance_arg_mem_1_done;
logic [31:0] relu4d_0_instance_arg_mem_0_write_data;
logic relu4d_0_instance_arg_mem_1_write_en;
logic [8:0] relu4d_0_instance_arg_mem_1_addr0;
logic relu4d_0_instance_arg_mem_1_content_en;
logic [8:0] idx_in;
logic idx_write_en;
logic idx_clk;
logic idx_reset;
logic [8:0] idx_out;
logic idx_done;
logic cond_reg_in;
logic cond_reg_write_en;
logic cond_reg_clk;
logic cond_reg_reset;
logic cond_reg_out;
logic cond_reg_done;
logic [8:0] adder_left;
logic [8:0] adder_right;
logic [8:0] adder_out;
logic [8:0] lt_left;
logic [8:0] lt_right;
logic lt_out;
logic [8:0] idx0_in;
logic idx0_write_en;
logic idx0_clk;
logic idx0_reset;
logic [8:0] idx0_out;
logic idx0_done;
logic cond_reg0_in;
logic cond_reg0_write_en;
logic cond_reg0_clk;
logic cond_reg0_reset;
logic cond_reg0_out;
logic cond_reg0_done;
logic [8:0] adder0_left;
logic [8:0] adder0_right;
logic [8:0] adder0_out;
logic [8:0] lt0_left;
logic [8:0] lt0_right;
logic lt0_out;
logic [3:0] fsm_in;
logic fsm_write_en;
logic fsm_clk;
logic fsm_reset;
logic [3:0] fsm_out;
logic fsm_done;
logic bb0_0_go_in;
logic bb0_0_go_out;
logic bb0_0_done_in;
logic bb0_0_done_out;
logic bb0_1_go_in;
logic bb0_1_go_out;
logic bb0_1_done_in;
logic bb0_1_done_out;
logic bb0_2_go_in;
logic bb0_2_go_out;
logic bb0_2_done_in;
logic bb0_2_done_out;
logic bb0_3_go_in;
logic bb0_3_go_out;
logic bb0_3_done_in;
logic bb0_3_done_out;
logic bb0_4_go_in;
logic bb0_4_go_out;
logic bb0_4_done_in;
logic bb0_4_done_out;
logic bb0_5_go_in;
logic bb0_5_go_out;
logic bb0_5_done_in;
logic bb0_5_done_out;
logic invoke0_go_in;
logic invoke0_go_out;
logic invoke0_done_in;
logic invoke0_done_out;
logic invoke1_go_in;
logic invoke1_go_out;
logic invoke1_done_in;
logic invoke1_done_out;
logic invoke2_go_in;
logic invoke2_go_out;
logic invoke2_done_in;
logic invoke2_done_out;
logic invoke3_go_in;
logic invoke3_go_out;
logic invoke3_done_in;
logic invoke3_done_out;
logic invoke4_go_in;
logic invoke4_go_out;
logic invoke4_done_in;
logic invoke4_done_out;
logic init_repeat_go_in;
logic init_repeat_go_out;
logic init_repeat_done_in;
logic init_repeat_done_out;
logic incr_repeat_go_in;
logic incr_repeat_go_out;
logic incr_repeat_done_in;
logic incr_repeat_done_out;
logic init_repeat0_go_in;
logic init_repeat0_go_out;
logic init_repeat0_done_in;
logic init_repeat0_done_out;
logic incr_repeat0_go_in;
logic incr_repeat0_go_out;
logic incr_repeat0_done_in;
logic incr_repeat0_done_out;
logic tdcc_go_in;
logic tdcc_go_out;
logic tdcc_done_in;
logic tdcc_done_out;
std_slice # (
    .IN_WIDTH(32),
    .OUT_WIDTH(9)
) std_slice_4 (
    .in(std_slice_4_in),
    .out(std_slice_4_out)
);
std_add # (
    .WIDTH(32)
) std_add_1 (
    .left(std_add_1_left),
    .out(std_add_1_out),
    .right(std_add_1_right)
);
std_reg # (
    .WIDTH(32)
) addf_0_reg (
    .clk(addf_0_reg_clk),
    .done(addf_0_reg_done),
    .in(addf_0_reg_in),
    .out(addf_0_reg_out),
    .reset(addf_0_reg_reset),
    .write_en(addf_0_reg_write_en)
);
std_addFN # (
    .expWidth(8),
    .numWidth(32),
    .sigWidth(24)
) std_addFN_0 (
    .clk(std_addFN_0_clk),
    .control(std_addFN_0_control),
    .done(std_addFN_0_done),
    .exceptionFlags(std_addFN_0_exceptionFlags),
    .go(std_addFN_0_go),
    .left(std_addFN_0_left),
    .out(std_addFN_0_out),
    .reset(std_addFN_0_reset),
    .right(std_addFN_0_right),
    .roundingMode(std_addFN_0_roundingMode),
    .subOp(std_addFN_0_subOp)
);
std_reg # (
    .WIDTH(32)
) for_0_induction_var_reg (
    .clk(for_0_induction_var_reg_clk),
    .done(for_0_induction_var_reg_done),
    .in(for_0_induction_var_reg_in),
    .out(for_0_induction_var_reg_out),
    .reset(for_0_induction_var_reg_reset),
    .write_en(for_0_induction_var_reg_write_en)
);
relu4d_0 relu4d_0_instance (
    .arg_mem_0_addr0(relu4d_0_instance_arg_mem_0_addr0),
    .arg_mem_0_content_en(relu4d_0_instance_arg_mem_0_content_en),
    .arg_mem_0_done(relu4d_0_instance_arg_mem_0_done),
    .arg_mem_0_read_data(relu4d_0_instance_arg_mem_0_read_data),
    .arg_mem_0_write_data(relu4d_0_instance_arg_mem_0_write_data),
    .arg_mem_0_write_en(relu4d_0_instance_arg_mem_0_write_en),
    .arg_mem_1_addr0(relu4d_0_instance_arg_mem_1_addr0),
    .arg_mem_1_content_en(relu4d_0_instance_arg_mem_1_content_en),
    .arg_mem_1_done(relu4d_0_instance_arg_mem_1_done),
    .arg_mem_1_read_data(relu4d_0_instance_arg_mem_1_read_data),
    .arg_mem_1_write_data(relu4d_0_instance_arg_mem_1_write_data),
    .arg_mem_1_write_en(relu4d_0_instance_arg_mem_1_write_en),
    .clk(relu4d_0_instance_clk),
    .done(relu4d_0_instance_done),
    .go(relu4d_0_instance_go),
    .reset(relu4d_0_instance_reset)
);
std_reg # (
    .WIDTH(9)
) idx (
    .clk(idx_clk),
    .done(idx_done),
    .in(idx_in),
    .out(idx_out),
    .reset(idx_reset),
    .write_en(idx_write_en)
);
std_reg # (
    .WIDTH(1)
) cond_reg (
    .clk(cond_reg_clk),
    .done(cond_reg_done),
    .in(cond_reg_in),
    .out(cond_reg_out),
    .reset(cond_reg_reset),
    .write_en(cond_reg_write_en)
);
std_add # (
    .WIDTH(9)
) adder (
    .left(adder_left),
    .out(adder_out),
    .right(adder_right)
);
std_lt # (
    .WIDTH(9)
) lt (
    .left(lt_left),
    .out(lt_out),
    .right(lt_right)
);
std_reg # (
    .WIDTH(9)
) idx0 (
    .clk(idx0_clk),
    .done(idx0_done),
    .in(idx0_in),
    .out(idx0_out),
    .reset(idx0_reset),
    .write_en(idx0_write_en)
);
std_reg # (
    .WIDTH(1)
) cond_reg0 (
    .clk(cond_reg0_clk),
    .done(cond_reg0_done),
    .in(cond_reg0_in),
    .out(cond_reg0_out),
    .reset(cond_reg0_reset),
    .write_en(cond_reg0_write_en)
);
std_add # (
    .WIDTH(9)
) adder0 (
    .left(adder0_left),
    .out(adder0_out),
    .right(adder0_right)
);
std_lt # (
    .WIDTH(9)
) lt0 (
    .left(lt0_left),
    .out(lt0_out),
    .right(lt0_right)
);
std_reg # (
    .WIDTH(4)
) fsm (
    .clk(fsm_clk),
    .done(fsm_done),
    .in(fsm_in),
    .out(fsm_out),
    .reset(fsm_reset),
    .write_en(fsm_write_en)
);
std_wire # (
    .WIDTH(1)
) bb0_0_go (
    .in(bb0_0_go_in),
    .out(bb0_0_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_0_done (
    .in(bb0_0_done_in),
    .out(bb0_0_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_1_go (
    .in(bb0_1_go_in),
    .out(bb0_1_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_1_done (
    .in(bb0_1_done_in),
    .out(bb0_1_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_2_go (
    .in(bb0_2_go_in),
    .out(bb0_2_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_2_done (
    .in(bb0_2_done_in),
    .out(bb0_2_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_3_go (
    .in(bb0_3_go_in),
    .out(bb0_3_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_3_done (
    .in(bb0_3_done_in),
    .out(bb0_3_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_4_go (
    .in(bb0_4_go_in),
    .out(bb0_4_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_4_done (
    .in(bb0_4_done_in),
    .out(bb0_4_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_5_go (
    .in(bb0_5_go_in),
    .out(bb0_5_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_5_done (
    .in(bb0_5_done_in),
    .out(bb0_5_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke0_go (
    .in(invoke0_go_in),
    .out(invoke0_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke0_done (
    .in(invoke0_done_in),
    .out(invoke0_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke1_go (
    .in(invoke1_go_in),
    .out(invoke1_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke1_done (
    .in(invoke1_done_in),
    .out(invoke1_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke2_go (
    .in(invoke2_go_in),
    .out(invoke2_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke2_done (
    .in(invoke2_done_in),
    .out(invoke2_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke3_go (
    .in(invoke3_go_in),
    .out(invoke3_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke3_done (
    .in(invoke3_done_in),
    .out(invoke3_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke4_go (
    .in(invoke4_go_in),
    .out(invoke4_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke4_done (
    .in(invoke4_done_in),
    .out(invoke4_done_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat_go (
    .in(init_repeat_go_in),
    .out(init_repeat_go_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat_done (
    .in(init_repeat_done_in),
    .out(init_repeat_done_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat_go (
    .in(incr_repeat_go_in),
    .out(incr_repeat_go_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat_done (
    .in(incr_repeat_done_in),
    .out(incr_repeat_done_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat0_go (
    .in(init_repeat0_go_in),
    .out(init_repeat0_go_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat0_done (
    .in(init_repeat0_done_in),
    .out(init_repeat0_done_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat0_go (
    .in(incr_repeat0_go_in),
    .out(incr_repeat0_go_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat0_done (
    .in(incr_repeat0_done_in),
    .out(incr_repeat0_done_out)
);
std_wire # (
    .WIDTH(1)
) tdcc_go (
    .in(tdcc_go_in),
    .out(tdcc_go_out)
);
std_wire # (
    .WIDTH(1)
) tdcc_done (
    .in(tdcc_done_in),
    .out(tdcc_done_out)
);
wire _guard0 = 1;
wire _guard1 = bb0_0_go_out;
wire _guard2 = bb0_1_go_out;
wire _guard3 = _guard1 | _guard2;
wire _guard4 = bb0_3_go_out;
wire _guard5 = _guard3 | _guard4;
wire _guard6 = bb0_4_go_out;
wire _guard7 = bb0_5_go_out;
wire _guard8 = _guard6 | _guard7;
wire _guard9 = init_repeat_done_out;
wire _guard10 = ~_guard9;
wire _guard11 = fsm_out == 4'd1;
wire _guard12 = _guard10 & _guard11;
wire _guard13 = tdcc_go_out;
wire _guard14 = _guard12 & _guard13;
wire _guard15 = tdcc_done_out;
wire _guard16 = bb0_0_go_out;
wire _guard17 = invoke2_go_out;
wire _guard18 = bb0_3_go_out;
wire _guard19 = invoke2_go_out;
wire _guard20 = bb0_3_go_out;
wire _guard21 = bb0_0_go_out;
wire _guard22 = bb0_4_go_out;
wire _guard23 = invoke2_go_out;
wire _guard24 = bb0_3_go_out;
wire _guard25 = invoke2_go_out;
wire _guard26 = bb0_3_go_out;
wire _guard27 = bb0_5_go_out;
wire _guard28 = bb0_5_go_out;
wire _guard29 = bb0_4_go_out;
wire _guard30 = invoke2_go_out;
wire _guard31 = bb0_5_go_out;
wire _guard32 = invoke2_go_out;
wire _guard33 = bb0_1_go_out;
wire _guard34 = bb0_1_go_out;
wire _guard35 = bb0_5_go_out;
wire _guard36 = incr_repeat_go_out;
wire _guard37 = incr_repeat_go_out;
wire _guard38 = fsm_out == 4'd15;
wire _guard39 = fsm_out == 4'd0;
wire _guard40 = invoke0_done_out;
wire _guard41 = _guard39 & _guard40;
wire _guard42 = tdcc_go_out;
wire _guard43 = _guard41 & _guard42;
wire _guard44 = _guard38 | _guard43;
wire _guard45 = fsm_out == 4'd1;
wire _guard46 = init_repeat_done_out;
wire _guard47 = cond_reg_out;
wire _guard48 = _guard46 & _guard47;
wire _guard49 = _guard45 & _guard48;
wire _guard50 = tdcc_go_out;
wire _guard51 = _guard49 & _guard50;
wire _guard52 = _guard44 | _guard51;
wire _guard53 = fsm_out == 4'd7;
wire _guard54 = incr_repeat_done_out;
wire _guard55 = cond_reg_out;
wire _guard56 = _guard54 & _guard55;
wire _guard57 = _guard53 & _guard56;
wire _guard58 = tdcc_go_out;
wire _guard59 = _guard57 & _guard58;
wire _guard60 = _guard52 | _guard59;
wire _guard61 = fsm_out == 4'd2;
wire _guard62 = bb0_0_done_out;
wire _guard63 = _guard61 & _guard62;
wire _guard64 = tdcc_go_out;
wire _guard65 = _guard63 & _guard64;
wire _guard66 = _guard60 | _guard65;
wire _guard67 = fsm_out == 4'd3;
wire _guard68 = bb0_1_done_out;
wire _guard69 = _guard67 & _guard68;
wire _guard70 = tdcc_go_out;
wire _guard71 = _guard69 & _guard70;
wire _guard72 = _guard66 | _guard71;
wire _guard73 = fsm_out == 4'd4;
wire _guard74 = bb0_2_done_out;
wire _guard75 = _guard73 & _guard74;
wire _guard76 = tdcc_go_out;
wire _guard77 = _guard75 & _guard76;
wire _guard78 = _guard72 | _guard77;
wire _guard79 = fsm_out == 4'd5;
wire _guard80 = bb0_3_done_out;
wire _guard81 = _guard79 & _guard80;
wire _guard82 = tdcc_go_out;
wire _guard83 = _guard81 & _guard82;
wire _guard84 = _guard78 | _guard83;
wire _guard85 = fsm_out == 4'd6;
wire _guard86 = invoke1_done_out;
wire _guard87 = _guard85 & _guard86;
wire _guard88 = tdcc_go_out;
wire _guard89 = _guard87 & _guard88;
wire _guard90 = _guard84 | _guard89;
wire _guard91 = fsm_out == 4'd1;
wire _guard92 = init_repeat_done_out;
wire _guard93 = cond_reg_out;
wire _guard94 = ~_guard93;
wire _guard95 = _guard92 & _guard94;
wire _guard96 = _guard91 & _guard95;
wire _guard97 = tdcc_go_out;
wire _guard98 = _guard96 & _guard97;
wire _guard99 = _guard90 | _guard98;
wire _guard100 = fsm_out == 4'd7;
wire _guard101 = incr_repeat_done_out;
wire _guard102 = cond_reg_out;
wire _guard103 = ~_guard102;
wire _guard104 = _guard101 & _guard103;
wire _guard105 = _guard100 & _guard104;
wire _guard106 = tdcc_go_out;
wire _guard107 = _guard105 & _guard106;
wire _guard108 = _guard99 | _guard107;
wire _guard109 = fsm_out == 4'd8;
wire _guard110 = invoke2_done_out;
wire _guard111 = _guard109 & _guard110;
wire _guard112 = tdcc_go_out;
wire _guard113 = _guard111 & _guard112;
wire _guard114 = _guard108 | _guard113;
wire _guard115 = fsm_out == 4'd9;
wire _guard116 = invoke3_done_out;
wire _guard117 = _guard115 & _guard116;
wire _guard118 = tdcc_go_out;
wire _guard119 = _guard117 & _guard118;
wire _guard120 = _guard114 | _guard119;
wire _guard121 = fsm_out == 4'd10;
wire _guard122 = init_repeat0_done_out;
wire _guard123 = cond_reg0_out;
wire _guard124 = _guard122 & _guard123;
wire _guard125 = _guard121 & _guard124;
wire _guard126 = tdcc_go_out;
wire _guard127 = _guard125 & _guard126;
wire _guard128 = _guard120 | _guard127;
wire _guard129 = fsm_out == 4'd14;
wire _guard130 = incr_repeat0_done_out;
wire _guard131 = cond_reg0_out;
wire _guard132 = _guard130 & _guard131;
wire _guard133 = _guard129 & _guard132;
wire _guard134 = tdcc_go_out;
wire _guard135 = _guard133 & _guard134;
wire _guard136 = _guard128 | _guard135;
wire _guard137 = fsm_out == 4'd11;
wire _guard138 = bb0_4_done_out;
wire _guard139 = _guard137 & _guard138;
wire _guard140 = tdcc_go_out;
wire _guard141 = _guard139 & _guard140;
wire _guard142 = _guard136 | _guard141;
wire _guard143 = fsm_out == 4'd12;
wire _guard144 = bb0_5_done_out;
wire _guard145 = _guard143 & _guard144;
wire _guard146 = tdcc_go_out;
wire _guard147 = _guard145 & _guard146;
wire _guard148 = _guard142 | _guard147;
wire _guard149 = fsm_out == 4'd13;
wire _guard150 = invoke4_done_out;
wire _guard151 = _guard149 & _guard150;
wire _guard152 = tdcc_go_out;
wire _guard153 = _guard151 & _guard152;
wire _guard154 = _guard148 | _guard153;
wire _guard155 = fsm_out == 4'd10;
wire _guard156 = init_repeat0_done_out;
wire _guard157 = cond_reg0_out;
wire _guard158 = ~_guard157;
wire _guard159 = _guard156 & _guard158;
wire _guard160 = _guard155 & _guard159;
wire _guard161 = tdcc_go_out;
wire _guard162 = _guard160 & _guard161;
wire _guard163 = _guard154 | _guard162;
wire _guard164 = fsm_out == 4'd14;
wire _guard165 = incr_repeat0_done_out;
wire _guard166 = cond_reg0_out;
wire _guard167 = ~_guard166;
wire _guard168 = _guard165 & _guard167;
wire _guard169 = _guard164 & _guard168;
wire _guard170 = tdcc_go_out;
wire _guard171 = _guard169 & _guard170;
wire _guard172 = _guard163 | _guard171;
wire _guard173 = fsm_out == 4'd2;
wire _guard174 = bb0_0_done_out;
wire _guard175 = _guard173 & _guard174;
wire _guard176 = tdcc_go_out;
wire _guard177 = _guard175 & _guard176;
wire _guard178 = fsm_out == 4'd4;
wire _guard179 = bb0_2_done_out;
wire _guard180 = _guard178 & _guard179;
wire _guard181 = tdcc_go_out;
wire _guard182 = _guard180 & _guard181;
wire _guard183 = fsm_out == 4'd3;
wire _guard184 = bb0_1_done_out;
wire _guard185 = _guard183 & _guard184;
wire _guard186 = tdcc_go_out;
wire _guard187 = _guard185 & _guard186;
wire _guard188 = fsm_out == 4'd8;
wire _guard189 = invoke2_done_out;
wire _guard190 = _guard188 & _guard189;
wire _guard191 = tdcc_go_out;
wire _guard192 = _guard190 & _guard191;
wire _guard193 = fsm_out == 4'd9;
wire _guard194 = invoke3_done_out;
wire _guard195 = _guard193 & _guard194;
wire _guard196 = tdcc_go_out;
wire _guard197 = _guard195 & _guard196;
wire _guard198 = fsm_out == 4'd15;
wire _guard199 = fsm_out == 4'd13;
wire _guard200 = invoke4_done_out;
wire _guard201 = _guard199 & _guard200;
wire _guard202 = tdcc_go_out;
wire _guard203 = _guard201 & _guard202;
wire _guard204 = fsm_out == 4'd6;
wire _guard205 = invoke1_done_out;
wire _guard206 = _guard204 & _guard205;
wire _guard207 = tdcc_go_out;
wire _guard208 = _guard206 & _guard207;
wire _guard209 = fsm_out == 4'd1;
wire _guard210 = init_repeat_done_out;
wire _guard211 = cond_reg_out;
wire _guard212 = _guard210 & _guard211;
wire _guard213 = _guard209 & _guard212;
wire _guard214 = tdcc_go_out;
wire _guard215 = _guard213 & _guard214;
wire _guard216 = fsm_out == 4'd7;
wire _guard217 = incr_repeat_done_out;
wire _guard218 = cond_reg_out;
wire _guard219 = _guard217 & _guard218;
wire _guard220 = _guard216 & _guard219;
wire _guard221 = tdcc_go_out;
wire _guard222 = _guard220 & _guard221;
wire _guard223 = _guard215 | _guard222;
wire _guard224 = fsm_out == 4'd5;
wire _guard225 = bb0_3_done_out;
wire _guard226 = _guard224 & _guard225;
wire _guard227 = tdcc_go_out;
wire _guard228 = _guard226 & _guard227;
wire _guard229 = fsm_out == 4'd1;
wire _guard230 = init_repeat_done_out;
wire _guard231 = cond_reg_out;
wire _guard232 = ~_guard231;
wire _guard233 = _guard230 & _guard232;
wire _guard234 = _guard229 & _guard233;
wire _guard235 = tdcc_go_out;
wire _guard236 = _guard234 & _guard235;
wire _guard237 = fsm_out == 4'd7;
wire _guard238 = incr_repeat_done_out;
wire _guard239 = cond_reg_out;
wire _guard240 = ~_guard239;
wire _guard241 = _guard238 & _guard240;
wire _guard242 = _guard237 & _guard241;
wire _guard243 = tdcc_go_out;
wire _guard244 = _guard242 & _guard243;
wire _guard245 = _guard236 | _guard244;
wire _guard246 = fsm_out == 4'd11;
wire _guard247 = bb0_4_done_out;
wire _guard248 = _guard246 & _guard247;
wire _guard249 = tdcc_go_out;
wire _guard250 = _guard248 & _guard249;
wire _guard251 = fsm_out == 4'd12;
wire _guard252 = bb0_5_done_out;
wire _guard253 = _guard251 & _guard252;
wire _guard254 = tdcc_go_out;
wire _guard255 = _guard253 & _guard254;
wire _guard256 = fsm_out == 4'd0;
wire _guard257 = invoke0_done_out;
wire _guard258 = _guard256 & _guard257;
wire _guard259 = tdcc_go_out;
wire _guard260 = _guard258 & _guard259;
wire _guard261 = fsm_out == 4'd10;
wire _guard262 = init_repeat0_done_out;
wire _guard263 = cond_reg0_out;
wire _guard264 = ~_guard263;
wire _guard265 = _guard262 & _guard264;
wire _guard266 = _guard261 & _guard265;
wire _guard267 = tdcc_go_out;
wire _guard268 = _guard266 & _guard267;
wire _guard269 = fsm_out == 4'd14;
wire _guard270 = incr_repeat0_done_out;
wire _guard271 = cond_reg0_out;
wire _guard272 = ~_guard271;
wire _guard273 = _guard270 & _guard272;
wire _guard274 = _guard269 & _guard273;
wire _guard275 = tdcc_go_out;
wire _guard276 = _guard274 & _guard275;
wire _guard277 = _guard268 | _guard276;
wire _guard278 = fsm_out == 4'd10;
wire _guard279 = init_repeat0_done_out;
wire _guard280 = cond_reg0_out;
wire _guard281 = _guard279 & _guard280;
wire _guard282 = _guard278 & _guard281;
wire _guard283 = tdcc_go_out;
wire _guard284 = _guard282 & _guard283;
wire _guard285 = fsm_out == 4'd14;
wire _guard286 = incr_repeat0_done_out;
wire _guard287 = cond_reg0_out;
wire _guard288 = _guard286 & _guard287;
wire _guard289 = _guard285 & _guard288;
wire _guard290 = tdcc_go_out;
wire _guard291 = _guard289 & _guard290;
wire _guard292 = _guard284 | _guard291;
wire _guard293 = bb0_0_done_out;
wire _guard294 = ~_guard293;
wire _guard295 = fsm_out == 4'd2;
wire _guard296 = _guard294 & _guard295;
wire _guard297 = tdcc_go_out;
wire _guard298 = _guard296 & _guard297;
wire _guard299 = invoke4_done_out;
wire _guard300 = ~_guard299;
wire _guard301 = fsm_out == 4'd13;
wire _guard302 = _guard300 & _guard301;
wire _guard303 = tdcc_go_out;
wire _guard304 = _guard302 & _guard303;
wire _guard305 = invoke2_done_out;
wire _guard306 = ~_guard305;
wire _guard307 = fsm_out == 4'd8;
wire _guard308 = _guard306 & _guard307;
wire _guard309 = tdcc_go_out;
wire _guard310 = _guard308 & _guard309;
wire _guard311 = init_repeat0_go_out;
wire _guard312 = incr_repeat0_go_out;
wire _guard313 = _guard311 | _guard312;
wire _guard314 = init_repeat0_go_out;
wire _guard315 = incr_repeat0_go_out;
wire _guard316 = invoke0_go_out;
wire _guard317 = invoke1_go_out;
wire _guard318 = _guard316 | _guard317;
wire _guard319 = invoke0_go_out;
wire _guard320 = invoke1_go_out;
wire _guard321 = bb0_2_done_out;
wire _guard322 = ~_guard321;
wire _guard323 = fsm_out == 4'd4;
wire _guard324 = _guard322 & _guard323;
wire _guard325 = tdcc_go_out;
wire _guard326 = _guard324 & _guard325;
wire _guard327 = init_repeat0_go_out;
wire _guard328 = incr_repeat0_go_out;
wire _guard329 = _guard327 | _guard328;
wire _guard330 = incr_repeat0_go_out;
wire _guard331 = init_repeat0_go_out;
wire _guard332 = invoke1_go_out;
wire _guard333 = invoke4_go_out;
wire _guard334 = invoke1_go_out;
wire _guard335 = invoke4_go_out;
wire _guard336 = _guard334 | _guard335;
wire _guard337 = invoke0_done_out;
wire _guard338 = ~_guard337;
wire _guard339 = fsm_out == 4'd0;
wire _guard340 = _guard338 & _guard339;
wire _guard341 = tdcc_go_out;
wire _guard342 = _guard340 & _guard341;
wire _guard343 = incr_repeat_done_out;
wire _guard344 = ~_guard343;
wire _guard345 = fsm_out == 4'd7;
wire _guard346 = _guard344 & _guard345;
wire _guard347 = tdcc_go_out;
wire _guard348 = _guard346 & _guard347;
wire _guard349 = invoke3_go_out;
wire _guard350 = invoke4_go_out;
wire _guard351 = _guard349 | _guard350;
wire _guard352 = bb0_2_go_out;
wire _guard353 = invoke3_go_out;
wire _guard354 = invoke4_go_out;
wire _guard355 = bb0_2_go_out;
wire _guard356 = cond_reg0_done;
wire _guard357 = idx0_done;
wire _guard358 = _guard356 & _guard357;
wire _guard359 = bb0_3_done_out;
wire _guard360 = ~_guard359;
wire _guard361 = fsm_out == 4'd5;
wire _guard362 = _guard360 & _guard361;
wire _guard363 = tdcc_go_out;
wire _guard364 = _guard362 & _guard363;
wire _guard365 = bb0_2_go_out;
wire _guard366 = bb0_2_go_out;
wire _guard367 = std_addFN_0_done;
wire _guard368 = ~_guard367;
wire _guard369 = bb0_2_go_out;
wire _guard370 = _guard368 & _guard369;
wire _guard371 = bb0_2_go_out;
wire _guard372 = invoke2_go_out;
wire _guard373 = invoke2_go_out;
wire _guard374 = invoke2_go_out;
wire _guard375 = invoke2_go_out;
wire _guard376 = init_repeat_go_out;
wire _guard377 = incr_repeat_go_out;
wire _guard378 = _guard376 | _guard377;
wire _guard379 = incr_repeat_go_out;
wire _guard380 = init_repeat_go_out;
wire _guard381 = cond_reg_done;
wire _guard382 = idx_done;
wire _guard383 = _guard381 & _guard382;
wire _guard384 = cond_reg_done;
wire _guard385 = idx_done;
wire _guard386 = _guard384 & _guard385;
wire _guard387 = cond_reg0_done;
wire _guard388 = idx0_done;
wire _guard389 = _guard387 & _guard388;
wire _guard390 = incr_repeat0_go_out;
wire _guard391 = incr_repeat0_go_out;
wire _guard392 = init_repeat0_done_out;
wire _guard393 = ~_guard392;
wire _guard394 = fsm_out == 4'd10;
wire _guard395 = _guard393 & _guard394;
wire _guard396 = tdcc_go_out;
wire _guard397 = _guard395 & _guard396;
wire _guard398 = incr_repeat0_done_out;
wire _guard399 = ~_guard398;
wire _guard400 = fsm_out == 4'd14;
wire _guard401 = _guard399 & _guard400;
wire _guard402 = tdcc_go_out;
wire _guard403 = _guard401 & _guard402;
wire _guard404 = invoke1_done_out;
wire _guard405 = ~_guard404;
wire _guard406 = fsm_out == 4'd6;
wire _guard407 = _guard405 & _guard406;
wire _guard408 = tdcc_go_out;
wire _guard409 = _guard407 & _guard408;
wire _guard410 = fsm_out == 4'd15;
wire _guard411 = incr_repeat_go_out;
wire _guard412 = incr_repeat_go_out;
wire _guard413 = invoke3_done_out;
wire _guard414 = ~_guard413;
wire _guard415 = fsm_out == 4'd9;
wire _guard416 = _guard414 & _guard415;
wire _guard417 = tdcc_go_out;
wire _guard418 = _guard416 & _guard417;
wire _guard419 = bb0_4_done_out;
wire _guard420 = ~_guard419;
wire _guard421 = fsm_out == 4'd11;
wire _guard422 = _guard420 & _guard421;
wire _guard423 = tdcc_go_out;
wire _guard424 = _guard422 & _guard423;
wire _guard425 = bb0_1_done_out;
wire _guard426 = ~_guard425;
wire _guard427 = fsm_out == 4'd3;
wire _guard428 = _guard426 & _guard427;
wire _guard429 = tdcc_go_out;
wire _guard430 = _guard428 & _guard429;
wire _guard431 = init_repeat_go_out;
wire _guard432 = incr_repeat_go_out;
wire _guard433 = _guard431 | _guard432;
wire _guard434 = init_repeat_go_out;
wire _guard435 = incr_repeat_go_out;
wire _guard436 = incr_repeat0_go_out;
wire _guard437 = incr_repeat0_go_out;
wire _guard438 = bb0_5_done_out;
wire _guard439 = ~_guard438;
wire _guard440 = fsm_out == 4'd12;
wire _guard441 = _guard439 & _guard440;
wire _guard442 = tdcc_go_out;
wire _guard443 = _guard441 & _guard442;
assign std_slice_4_in =
  _guard5 ? for_0_induction_var_reg_out :
  _guard8 ? addf_0_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard8, _guard5})) begin
    $fatal(2, "Multiple assignment to port `std_slice_4.in'.");
end
end
assign init_repeat_go_in = _guard14;
assign done = _guard15;
assign arg_mem_0_content_en = _guard16;
assign arg_mem_4_write_data = relu4d_0_instance_arg_mem_1_write_data;
assign arg_mem_3_addr0 =
  _guard18 ? std_slice_4_out :
  _guard19 ? relu4d_0_instance_arg_mem_0_addr0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard19, _guard18})) begin
    $fatal(2, "Multiple assignment to port `_this.arg_mem_3_addr0'.");
end
end
assign arg_mem_3_write_data = addf_0_reg_out;
assign arg_mem_0_addr0 = std_slice_4_out;
assign arg_mem_4_addr0 =
  _guard22 ? std_slice_4_out :
  _guard23 ? relu4d_0_instance_arg_mem_1_addr0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard23, _guard22})) begin
    $fatal(2, "Multiple assignment to port `_this.arg_mem_4_addr0'.");
end
end
assign arg_mem_3_content_en =
  _guard24 ? 1'd1 :
  _guard25 ? relu4d_0_instance_arg_mem_0_content_en :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard25, _guard24})) begin
    $fatal(2, "Multiple assignment to port `_this.arg_mem_3_content_en'.");
end
end
assign arg_mem_3_write_en = _guard26;
assign arg_mem_2_addr0 = std_slice_4_out;
assign arg_mem_2_content_en = _guard28;
assign arg_mem_4_content_en =
  _guard29 ? 1'd1 :
  _guard30 ? relu4d_0_instance_arg_mem_1_content_en :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard30, _guard29})) begin
    $fatal(2, "Multiple assignment to port `_this.arg_mem_4_content_en'.");
end
end
assign arg_mem_2_write_en = _guard31;
assign arg_mem_4_write_en =
  _guard32 ? relu4d_0_instance_arg_mem_1_write_en :
  1'd0;
assign arg_mem_1_addr0 = std_slice_4_out;
assign arg_mem_1_content_en = _guard34;
assign arg_mem_2_write_data = arg_mem_4_read_data;
assign adder_left =
  _guard36 ? idx_out :
  9'd0;
assign adder_right =
  _guard37 ? 9'd1 :
  9'd0;
assign fsm_write_en = _guard172;
assign fsm_clk = clk;
assign fsm_reset = reset;
assign fsm_in =
  _guard177 ? 4'd3 :
  _guard182 ? 4'd5 :
  _guard187 ? 4'd4 :
  _guard192 ? 4'd9 :
  _guard197 ? 4'd10 :
  _guard198 ? 4'd0 :
  _guard203 ? 4'd14 :
  _guard208 ? 4'd7 :
  _guard223 ? 4'd2 :
  _guard228 ? 4'd6 :
  _guard245 ? 4'd8 :
  _guard250 ? 4'd12 :
  _guard255 ? 4'd13 :
  _guard260 ? 4'd1 :
  _guard277 ? 4'd15 :
  _guard292 ? 4'd11 :
  4'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard292, _guard277, _guard260, _guard255, _guard250, _guard245, _guard228, _guard223, _guard208, _guard203, _guard198, _guard197, _guard192, _guard187, _guard182, _guard177})) begin
    $fatal(2, "Multiple assignment to port `fsm.in'.");
end
end
assign bb0_0_go_in = _guard298;
assign invoke4_go_in = _guard304;
assign invoke2_go_in = _guard310;
assign cond_reg0_write_en = _guard313;
assign cond_reg0_clk = clk;
assign cond_reg0_reset = reset;
assign cond_reg0_in =
  _guard314 ? 1'd1 :
  _guard315 ? lt0_out :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard315, _guard314})) begin
    $fatal(2, "Multiple assignment to port `cond_reg0.in'.");
end
end
assign for_0_induction_var_reg_write_en = _guard318;
assign for_0_induction_var_reg_clk = clk;
assign for_0_induction_var_reg_reset = reset;
assign for_0_induction_var_reg_in =
  _guard319 ? 32'd0 :
  _guard320 ? std_add_1_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard320, _guard319})) begin
    $fatal(2, "Multiple assignment to port `for_0_induction_var_reg.in'.");
end
end
assign bb0_2_go_in = _guard326;
assign bb0_3_done_in = arg_mem_3_done;
assign idx0_write_en = _guard329;
assign idx0_clk = clk;
assign idx0_reset = reset;
assign idx0_in =
  _guard330 ? adder0_out :
  _guard331 ? 9'd0 :
  9'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard331, _guard330})) begin
    $fatal(2, "Multiple assignment to port `idx0.in'.");
end
end
assign std_add_1_left =
  _guard332 ? for_0_induction_var_reg_out :
  _guard333 ? addf_0_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard333, _guard332})) begin
    $fatal(2, "Multiple assignment to port `std_add_1.left'.");
end
end
assign std_add_1_right = 32'd1;
assign invoke0_go_in = _guard342;
assign bb0_0_done_in = arg_mem_0_done;
assign incr_repeat_go_in = _guard348;
assign tdcc_go_in = go;
assign addf_0_reg_write_en =
  _guard351 ? 1'd1 :
  _guard352 ? std_addFN_0_done :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard352, _guard351})) begin
    $fatal(2, "Multiple assignment to port `addf_0_reg.write_en'.");
end
end
assign addf_0_reg_clk = clk;
assign addf_0_reg_reset = reset;
assign addf_0_reg_in =
  _guard353 ? 32'd0 :
  _guard354 ? std_add_1_out :
  _guard355 ? std_addFN_0_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard355, _guard354, _guard353})) begin
    $fatal(2, "Multiple assignment to port `addf_0_reg.in'.");
end
end
assign invoke3_done_in = addf_0_reg_done;
assign incr_repeat0_done_in = _guard358;
assign bb0_3_go_in = _guard364;
assign std_addFN_0_roundingMode = 3'd0;
assign std_addFN_0_control = 1'd0;
assign std_addFN_0_clk = clk;
assign std_addFN_0_left =
  _guard365 ? arg_mem_0_read_data :
  32'd0;
assign std_addFN_0_subOp =
  _guard366 ? 1'd0 :
  1'd0;
assign std_addFN_0_reset = reset;
assign std_addFN_0_go = _guard370;
assign std_addFN_0_right =
  _guard371 ? arg_mem_1_read_data :
  32'd0;
assign relu4d_0_instance_arg_mem_0_read_data =
  _guard372 ? arg_mem_3_read_data :
  32'd0;
assign relu4d_0_instance_arg_mem_0_done =
  _guard373 ? arg_mem_3_done :
  1'd0;
assign relu4d_0_instance_clk = clk;
assign relu4d_0_instance_reset = reset;
assign relu4d_0_instance_go = _guard374;
assign relu4d_0_instance_arg_mem_1_done =
  _guard375 ? arg_mem_4_done :
  1'd0;
assign idx_write_en = _guard378;
assign idx_clk = clk;
assign idx_reset = reset;
assign idx_in =
  _guard379 ? adder_out :
  _guard380 ? 9'd0 :
  9'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard380, _guard379})) begin
    $fatal(2, "Multiple assignment to port `idx.in'.");
end
end
assign init_repeat_done_in = _guard383;
assign incr_repeat_done_in = _guard386;
assign init_repeat0_done_in = _guard389;
assign adder0_left =
  _guard390 ? idx0_out :
  9'd0;
assign adder0_right =
  _guard391 ? 9'd1 :
  9'd0;
assign invoke0_done_in = for_0_induction_var_reg_done;
assign init_repeat0_go_in = _guard397;
assign incr_repeat0_go_in = _guard403;
assign invoke1_go_in = _guard409;
assign bb0_5_done_in = arg_mem_2_done;
assign invoke2_done_in = relu4d_0_instance_done;
assign bb0_1_done_in = arg_mem_1_done;
assign tdcc_done_in = _guard410;
assign bb0_2_done_in = addf_0_reg_done;
assign lt_left =
  _guard411 ? adder_out :
  9'd0;
assign lt_right =
  _guard412 ? 9'd300 :
  9'd0;
assign invoke3_go_in = _guard418;
assign bb0_4_go_in = _guard424;
assign bb0_4_done_in = arg_mem_4_done;
assign invoke4_done_in = addf_0_reg_done;
assign bb0_1_go_in = _guard430;
assign invoke1_done_in = for_0_induction_var_reg_done;
assign cond_reg_write_en = _guard433;
assign cond_reg_clk = clk;
assign cond_reg_reset = reset;
assign cond_reg_in =
  _guard434 ? 1'd1 :
  _guard435 ? lt_out :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard435, _guard434})) begin
    $fatal(2, "Multiple assignment to port `cond_reg.in'.");
end
end
assign lt0_left =
  _guard436 ? adder0_out :
  9'd0;
assign lt0_right =
  _guard437 ? 9'd300 :
  9'd0;
assign bb0_5_go_in = _guard443;
// COMPONENT END: forward
endmodule

/*============================================================================

This Verilog source file is part of the Berkeley HardFloat IEEE Floating-Point
Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    reverse#(parameter width = 1) (
        input [(width - 1):0] in, output [(width - 1):0] out
    );

    genvar ix;
    generate
        for (ix = 0; ix < width; ix = ix + 1) begin :Bit
            assign out[ix] = in[width - 1 - ix];
        end
    endgenerate

endmodule

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    lowMaskHiLo#(
        parameter inWidth = 1,
        parameter topBound = 1,
        parameter bottomBound = 0
    ) (
        input [(inWidth - 1):0] in,
        output [(topBound - bottomBound - 1):0] out
    );

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    localparam numInVals = 1<<inWidth;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire signed [numInVals:0] c;
    assign c[numInVals] = 1;
    assign c[(numInVals - 1):0] = 0;
    wire [(topBound - bottomBound - 1):0] reverseOut =
        (c>>>in)>>(numInVals - topBound);
    reverse#(topBound - bottomBound) reverse(reverseOut, out);

endmodule

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    lowMaskLoHi#(
        parameter inWidth = 1,
        parameter topBound = 0,
        parameter bottomBound = 1
    ) (
        input [(inWidth - 1):0] in,
        output [(bottomBound - topBound - 1):0] out
    );

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    localparam numInVals = 1<<inWidth;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire signed [numInVals:0] c;
    assign c[numInVals] = 1;
    assign c[(numInVals - 1):0] = 0;
    wire [(bottomBound - topBound - 1):0] reverseOut =
        (c>>>~in)>>(topBound + 1);
    reverse#(bottomBound - topBound) reverse(reverseOut, out);

endmodule

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    countLeadingZeros#(parameter inWidth = 1, parameter countWidth = 1) (
        input [(inWidth - 1):0] in, output [(countWidth - 1):0] count
    );

    wire [(inWidth - 1):0] reverseIn;
    reverse#(inWidth) reverse_in(in, reverseIn);
    wire [inWidth:0] oneLeastReverseIn =
        {1'b1, reverseIn} & ({1'b0, ~reverseIn} + 1);
    genvar ix;
    generate
        for (ix = 0; ix <= inWidth; ix = ix + 1) begin :Bit
            wire [(countWidth - 1):0] countSoFar;
            if (ix == 0) begin
                assign countSoFar = 0;
            end else begin
                assign countSoFar =
                    Bit[ix - 1].countSoFar | (oneLeastReverseIn[ix] ? ix : 0);
                if (ix == inWidth) assign count = countSoFar;
            end
        end
    endgenerate

endmodule

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    compressBy2#(parameter inWidth = 1) (
        input [(inWidth - 1):0] in, output [((inWidth - 1)/2):0] out
    );

    localparam maxBitNumReduced = (inWidth - 1)/2;
    genvar ix;
    generate
        for (ix = 0; ix < maxBitNumReduced; ix = ix + 1) begin :Bit
            assign out[ix] = |in[(ix*2 + 1):ix*2];
        end
    endgenerate
    assign out[maxBitNumReduced] = |in[(inWidth - 1):maxBitNumReduced*2];

endmodule

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    compressBy4#(parameter inWidth = 1) (
        input [(inWidth - 1):0] in, output [(inWidth - 1)/4:0] out
    );

    localparam maxBitNumReduced = (inWidth - 1)/4;
    genvar ix;
    generate
        for (ix = 0; ix < maxBitNumReduced; ix = ix + 1) begin :Bit
            assign out[ix] = |in[(ix*4 + 3):ix*4];
        end
    endgenerate
    assign out[maxBitNumReduced] = |in[(inWidth - 1):maxBitNumReduced*4];

endmodule


/*============================================================================

This Verilog source file is part of the Berkeley HardFloat IEEE Floating-Point
Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    compareRecFN#(parameter expWidth = 3, parameter sigWidth = 3) (
        input [(expWidth + sigWidth):0] a,
        input [(expWidth + sigWidth):0] b,
        input signaling,
        output lt,
        output eq,
        output gt,
        output unordered,
        output [4:0] exceptionFlags
    );

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire isNaNA, isInfA, isZeroA, signA;
    wire signed [(expWidth + 1):0] sExpA;
    wire [sigWidth:0] sigA;
    recFNToRawFN#(expWidth, sigWidth)
        recFNToRawFN_a(a, isNaNA, isInfA, isZeroA, signA, sExpA, sigA);
    wire isSigNaNA;
    isSigNaNRecFN#(expWidth, sigWidth) isSigNaN_a(a, isSigNaNA);
    wire isNaNB, isInfB, isZeroB, signB;
    wire signed [(expWidth + 1):0] sExpB;
    wire [sigWidth:0] sigB;
    recFNToRawFN#(expWidth, sigWidth)
        recFNToRawFN_b(b, isNaNB, isInfB, isZeroB, signB, sExpB, sigB);
    wire isSigNaNB;
    isSigNaNRecFN#(expWidth, sigWidth) isSigNaN_b(b, isSigNaNB);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire ordered = !isNaNA && !isNaNB;
    wire bothInfs  = isInfA  && isInfB;
    wire bothZeros = isZeroA && isZeroB;
    wire eqExps = (sExpA == sExpB);
    wire common_ltMags = (sExpA < sExpB) || (eqExps && (sigA < sigB));
    wire common_eqMags = eqExps && (sigA == sigB);
    wire ordered_lt =
        !bothZeros
            && ((signA && !signB)
                    || (!bothInfs
                            && ((signA && !common_ltMags && !common_eqMags)
                                    || (!signB && common_ltMags))));
    wire ordered_eq =
        bothZeros || ((signA == signB) && (bothInfs || common_eqMags));
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire invalid = isSigNaNA || isSigNaNB || (signaling && !ordered);
    assign lt = ordered && ordered_lt;
    assign eq = ordered && ordered_eq;
    assign gt = ordered && !ordered_lt && !ordered_eq;
    assign unordered = !ordered;
    assign exceptionFlags = {invalid, 4'b0};

endmodule


/*============================================================================

This Verilog source file is part of the Berkeley HardFloat IEEE Floating-Point
Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    recFNToFN#(parameter expWidth = 3, parameter sigWidth = 3) (
        input [(expWidth + sigWidth):0] in,
        output [(expWidth + sigWidth - 1):0] out
    );

/*============================================================================

This Verilog include file is part of the Berkeley HardFloat IEEE Floating-
Point Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

function integer clog2;
    input integer a;

    begin
        a = a - 1;
        for (clog2 = 0; a > 0; clog2 = clog2 + 1) a = a>>1;
    end

endfunction



    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    localparam [expWidth:0] minNormExp = (1<<(expWidth - 1)) + 2;
    localparam normDistWidth = clog2(sigWidth);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire isNaN, isInf, isZero, sign;
    wire signed [(expWidth + 1):0] sExp;
    wire [sigWidth:0] sig;
    recFNToRawFN#(expWidth, sigWidth)
        recFNToRawFN(in, isNaN, isInf, isZero, sign, sExp, sig);
    wire isSubnormal = (sExp < minNormExp);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire [(normDistWidth - 1):0] denormShiftDist = minNormExp - 1 - sExp;
    wire [(expWidth - 1):0] expOut =
        (isSubnormal ? 0 : sExp - minNormExp + 1)
            | (isNaN || isInf ? {expWidth{1'b1}} : 0);
    wire [(sigWidth - 2):0] fractOut =
        isSubnormal ? (sig>>1)>>denormShiftDist : isInf ? 0 : sig;
    assign out = {sign, expOut, fractOut};

endmodule


/*============================================================================

This Verilog source file is part of the Berkeley HardFloat IEEE Floating-Point
Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/


/*============================================================================

This Verilog include file is part of the Berkeley HardFloat IEEE Floating-
Point Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define round_near_even   3'b000
`define round_minMag      3'b001
`define round_min         3'b010
`define round_max         3'b011
`define round_near_maxMag 3'b100
`define round_odd         3'b110

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define floatControlWidth 1
`define flControl_tininessBeforeRounding 1'b0
`define flControl_tininessAfterRounding  1'b1

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define flRoundOpt_sigMSBitAlwaysZero  1
`define flRoundOpt_subnormsAlwaysExact 2
`define flRoundOpt_neverUnderflows     4
`define flRoundOpt_neverOverflows      8



/*============================================================================

This Verilog include file is part of the Berkeley HardFloat IEEE Floating-
Point Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define flControl_default `flControl_tininessAfterRounding

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
//`define HardFloat_propagateNaNPayloads

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define HardFloat_signDefaultNaN 0
`define HardFloat_fractDefaultNaN(sigWidth) {1'b1, {((sigWidth) - 2){1'b0}}}



/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    addRecFNToRaw#(parameter expWidth = 3, parameter sigWidth = 3) (
        input [(1 - 1):0] control,
        input subOp,
        input [(expWidth + sigWidth):0] a,
        input [(expWidth + sigWidth):0] b,
        input [2:0] roundingMode,
        output invalidExc,
        output out_isNaN,
        output out_isInf,
        output out_isZero,
        output out_sign,
        output signed [(expWidth + 1):0] out_sExp,
        output [(sigWidth + 2):0] out_sig
    );

/*============================================================================

This Verilog include file is part of the Berkeley HardFloat IEEE Floating-
Point Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

function integer clog2;
    input integer a;

    begin
        a = a - 1;
        for (clog2 = 0; a > 0; clog2 = clog2 + 1) a = a>>1;
    end

endfunction



    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    localparam alignDistWidth = clog2(sigWidth);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire isNaNA, isInfA, isZeroA, signA;
    wire signed [(expWidth + 1):0] sExpA;
    wire [sigWidth:0] sigA;
    recFNToRawFN#(expWidth, sigWidth)
        recFNToRawFN_a(a, isNaNA, isInfA, isZeroA, signA, sExpA, sigA);
    wire isSigNaNA;
    isSigNaNRecFN#(expWidth, sigWidth) isSigNaN_a(a, isSigNaNA);
    wire isNaNB, isInfB, isZeroB, signB;
    wire signed [(expWidth + 1):0] sExpB;
    wire [sigWidth:0] sigB;
    recFNToRawFN#(expWidth, sigWidth)
        recFNToRawFN_b(b, isNaNB, isInfB, isZeroB, signB, sExpB, sigB);
    wire effSignB = subOp ? !signB : signB;
    wire isSigNaNB;
    isSigNaNRecFN#(expWidth, sigWidth) isSigNaN_b(b, isSigNaNB);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire eqSigns = (signA == effSignB);
    wire notEqSigns_signZero = (roundingMode == 3'b010) ? 1 : 0;
    wire signed [(expWidth + 1):0] sDiffExps = sExpA - sExpB;
    wire [(alignDistWidth - 1):0] modNatAlignDist =
        (sDiffExps < 0) ? sExpB - sExpA : sDiffExps;
    wire isMaxAlign =
        (sDiffExps>>>alignDistWidth != 0)
            && ((sDiffExps>>>alignDistWidth != -1)
                    || (sDiffExps[(alignDistWidth - 1):0] == 0));
    wire [(alignDistWidth - 1):0] alignDist =
        isMaxAlign ? (1<<alignDistWidth) - 1 : modNatAlignDist;
    wire closeSubMags = !eqSigns && !isMaxAlign && (modNatAlignDist <= 1);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire signed [(sigWidth + 2):0] close_alignedSigA =
          ((0 <= sDiffExps) &&  sDiffExps[0] ? sigA<<2 : 0)
        | ((0 <= sDiffExps) && !sDiffExps[0] ? sigA<<1 : 0)
        | ((sDiffExps < 0)                   ? sigA    : 0);
    wire signed [(sigWidth + 2):0] close_sSigSum =
        close_alignedSigA - (sigB<<1);
    wire [(sigWidth + 1):0] close_sigSum =
        (close_sSigSum < 0) ? -close_sSigSum : close_sSigSum;
    wire [(sigWidth + 1 + (sigWidth & 1)):0] close_adjustedSigSum =
        close_sigSum<<(sigWidth & 1);
    wire [(sigWidth + 1)/2:0] close_reduced2SigSum;
    compressBy2#(sigWidth + 2 + (sigWidth & 1))
        compressBy2_close_sigSum(close_adjustedSigSum, close_reduced2SigSum);
    wire [(alignDistWidth - 1):0] close_normDistReduced2;
    countLeadingZeros#((sigWidth + 3)/2, alignDistWidth)
        countLeadingZeros_close(close_reduced2SigSum, close_normDistReduced2);
    wire [(alignDistWidth - 1):0] close_nearNormDist =
        close_normDistReduced2<<1;
    wire [(sigWidth + 2):0] close_sigOut =
        (close_sigSum<<close_nearNormDist)<<1;
    wire close_totalCancellation =
        !(|close_sigOut[(sigWidth + 2):(sigWidth + 1)]);
    wire close_notTotalCancellation_signOut = signA ^ (close_sSigSum < 0);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire far_signOut = (sDiffExps < 0) ? effSignB : signA;
    wire [(sigWidth - 1):0] far_sigLarger  = (sDiffExps < 0) ? sigB : sigA;
    wire [(sigWidth - 1):0] far_sigSmaller = (sDiffExps < 0) ? sigA : sigB;
    wire [(sigWidth + 4):0] far_mainAlignedSigSmaller =
        {far_sigSmaller, 5'b0}>>alignDist;
    wire [(sigWidth + 1)/4:0] far_reduced4SigSmaller;
    compressBy4#(sigWidth + 2)
        compressBy4_far_sigSmaller(
            {far_sigSmaller, 2'b00}, far_reduced4SigSmaller);
    wire [(sigWidth + 1)/4:0] far_roundExtraMask;
    lowMaskHiLo#(alignDistWidth - 2, (sigWidth + 5)/4, 0)
        lowMask_far_roundExtraMask(
            alignDist[(alignDistWidth - 1):2], far_roundExtraMask);
    wire [(sigWidth + 2):0] far_alignedSigSmaller =
        {far_mainAlignedSigSmaller>>3,
         (|far_mainAlignedSigSmaller[2:0])
             || (|(far_reduced4SigSmaller & far_roundExtraMask))};
    wire far_subMags = !eqSigns;
    wire [(sigWidth + 3):0] far_negAlignedSigSmaller =
        far_subMags ? {1'b1, ~far_alignedSigSmaller}
            : {1'b0, far_alignedSigSmaller};
    wire [(sigWidth + 3):0] far_sigSum =
        (far_sigLarger<<3) + far_negAlignedSigSmaller + far_subMags;
    wire [(sigWidth + 2):0] far_sigOut =
        far_subMags ? far_sigSum : far_sigSum>>1 | far_sigSum[0];
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire notSigNaN_invalidExc = isInfA && isInfB && !eqSigns;
    wire notNaN_isInfOut = isInfA || isInfB;
    wire addZeros = isZeroA && isZeroB;
    wire notNaN_specialCase = notNaN_isInfOut || addZeros;
    wire notNaN_isZeroOut =
        addZeros
            || (!notNaN_isInfOut && closeSubMags && close_totalCancellation);
    wire notNaN_signOut =
           (eqSigns                      && signA              )
        || (isInfA                       && signA              )
        || (isInfB                       && effSignB           )
        || (notNaN_isZeroOut && !eqSigns && notEqSigns_signZero)
        || (!notNaN_specialCase && closeSubMags && !close_totalCancellation
                                        && close_notTotalCancellation_signOut)
        || (!notNaN_specialCase && !closeSubMags && far_signOut);
    wire signed [(expWidth + 1):0] common_sExpOut =
        (closeSubMags || (sDiffExps < 0) ? sExpB : sExpA)
            - (closeSubMags ? close_nearNormDist : far_subMags);
    wire [(sigWidth + 2):0] common_sigOut =
        closeSubMags ? close_sigOut : far_sigOut;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    assign invalidExc = isSigNaNA || isSigNaNB || notSigNaN_invalidExc;
    assign out_isInf = notNaN_isInfOut;
    assign out_isZero = notNaN_isZeroOut;
    assign out_sExp = common_sExpOut;
assign out_isNaN = isNaNA || isNaNB;
    assign out_sign = notNaN_signOut;
    assign out_sig = common_sigOut;


endmodule

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    addRecFN#(parameter expWidth = 3, parameter sigWidth = 3) (
        input [(1 - 1):0] control,
        input subOp,
        input [(expWidth + sigWidth):0] a,
        input [(expWidth + sigWidth):0] b,
        input [2:0] roundingMode,
        output [(expWidth + sigWidth):0] out,
        output [4:0] exceptionFlags
    );

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire invalidExc, out_isNaN, out_isInf, out_isZero, out_sign;
    wire signed [(expWidth + 1):0] out_sExp;
    wire [(sigWidth + 2):0] out_sig;
    addRecFNToRaw#(expWidth, sigWidth)
        addRecFNToRaw(
            control,
            subOp,
            a,
            b,
            roundingMode,
            invalidExc,
            out_isNaN,
            out_isInf,
            out_isZero,
            out_sign,
            out_sExp,
            out_sig
        );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    roundRawFNToRecFN#(expWidth, sigWidth, 2)
        roundRawOut(
            control,
            invalidExc,
            1'b0,
            out_isNaN,
            out_isInf,
            out_isZero,
            out_sign,
            out_sExp,
            out_sig,
            roundingMode,
            out,
            exceptionFlags
        );

endmodule


/*============================================================================

This Verilog source file is part of the Berkeley HardFloat IEEE Floating-Point
Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    fNToRecFN#(parameter expWidth = 3, parameter sigWidth = 3) (
        input [(expWidth + sigWidth - 1):0] in,
        output [(expWidth + sigWidth):0] out
    );

/*============================================================================

This Verilog include file is part of the Berkeley HardFloat IEEE Floating-
Point Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

function integer clog2;
    input integer a;

    begin
        a = a - 1;
        for (clog2 = 0; a > 0; clog2 = clog2 + 1) a = a>>1;
    end

endfunction



    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    localparam normDistWidth = clog2(sigWidth);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire sign;
    wire [(expWidth - 1):0] expIn;
    wire [(sigWidth - 2):0] fractIn;
    assign {sign, expIn, fractIn} = in;
    wire isZeroExpIn = (expIn == 0);
    wire isZeroFractIn = (fractIn == 0);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire [(normDistWidth - 1):0] normDist;
    countLeadingZeros#(sigWidth - 1, normDistWidth)
        countLeadingZeros(fractIn, normDist);
    wire [(sigWidth - 2):0] subnormFract = (fractIn<<normDist)<<1;
    wire [expWidth:0] adjustedExp =
        (isZeroExpIn ? normDist ^ ((1<<(expWidth + 1)) - 1) : expIn)
            + ((1<<(expWidth - 1)) | (isZeroExpIn ? 2 : 1));
    wire isZero = isZeroExpIn && isZeroFractIn;
    wire isSpecial = (adjustedExp[expWidth:(expWidth - 1)] == 'b11);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire [expWidth:0] exp;
    assign exp[expWidth:(expWidth - 2)] =
        isSpecial ? {2'b11, !isZeroFractIn}
            : isZero ? 3'b000 : adjustedExp[expWidth:(expWidth - 2)];
    assign exp[(expWidth - 3):0] = adjustedExp;
    assign out = {sign, exp, isZeroExpIn ? subnormFract : fractIn};

endmodule


/*============================================================================

This Verilog source file is part of the Berkeley HardFloat IEEE Floating-Point
Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/


/*============================================================================

This Verilog include file is part of the Berkeley HardFloat IEEE Floating-
Point Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define round_near_even   3'b000
`define round_minMag      3'b001
`define round_min         3'b010
`define round_max         3'b011
`define round_near_maxMag 3'b100
`define round_odd         3'b110

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define floatControlWidth 1
`define flControl_tininessBeforeRounding 1'b0
`define flControl_tininessAfterRounding  1'b1

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define flRoundOpt_sigMSBitAlwaysZero  1
`define flRoundOpt_subnormsAlwaysExact 2
`define flRoundOpt_neverUnderflows     4
`define flRoundOpt_neverOverflows      8



/*============================================================================

This Verilog include file is part of the Berkeley HardFloat IEEE Floating-
Point Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define flControl_default `flControl_tininessAfterRounding

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
//`define HardFloat_propagateNaNPayloads

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
`define HardFloat_signDefaultNaN 0
`define HardFloat_fractDefaultNaN(sigWidth) {1'b1, {((sigWidth) - 2){1'b0}}}



/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    recFNToRawFN#(parameter expWidth = 3, parameter sigWidth = 3) (
        input [(expWidth + sigWidth):0] in,
        output isNaN,
        output isInf,
        output isZero,
        output sign,
        output signed [(expWidth + 1):0] sExp,
        output [sigWidth:0] sig
    );

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire [expWidth:0] exp;
    wire [(sigWidth - 2):0] fract;
    assign {sign, exp, fract} = in;
    wire isSpecial = (exp>>(expWidth - 1) == 'b11);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    assign isNaN = isSpecial &&  exp[expWidth - 2];
    assign isInf = isSpecial && !exp[expWidth - 2];
    assign isZero = (exp>>(expWidth - 2) == 'b000);
    assign sExp = exp;
    assign sig = {1'b0, !isZero, fract};

endmodule

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    roundAnyRawFNToRecFN#(
        parameter inExpWidth = 3,
        parameter inSigWidth = 3,
        parameter outExpWidth = 3,
        parameter outSigWidth = 3,
        parameter options = 0
    ) (
        input [(1 - 1):0] control,
        input invalidExc,     // overrides 'infiniteExc' and 'in_*' inputs
        input infiniteExc,    // overrides 'in_*' inputs except 'in_sign'
        input in_isNaN,
        input in_isInf,
        input in_isZero,
        input in_sign,
        input signed [(inExpWidth + 1):0] in_sExp,   // limited range allowed
        input [inSigWidth:0] in_sig,
        input [2:0] roundingMode,
        output [(outExpWidth + outSigWidth):0] out,
        output [4:0] exceptionFlags
    );

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    localparam sigMSBitAlwaysZero =
        ((options & 1) != 0);
    localparam effectiveInSigWidth =
        sigMSBitAlwaysZero ? inSigWidth : inSigWidth + 1;
    localparam neverUnderflows =
        ((options
              & (4
                     | 2))
             != 0)
            || (inExpWidth < outExpWidth);
    localparam neverOverflows =
        ((options & 8) != 0)
            || (inExpWidth < outExpWidth);
    localparam adjustedExpWidth =
          (inExpWidth < outExpWidth) ? outExpWidth + 1
        : (inExpWidth == outExpWidth) ? inExpWidth + 2
        : inExpWidth + 3;
    localparam outNaNExp = 7<<(outExpWidth - 2);
    localparam outInfExp = 6<<(outExpWidth - 2);
    localparam outMaxFiniteExp = outInfExp - 1;
    localparam outMinNormExp = (1<<(outExpWidth - 1)) + 2;
    localparam outMinNonzeroExp = outMinNormExp - outSigWidth + 1;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire roundingMode_near_even   = (roundingMode == 3'b000);
    wire roundingMode_minMag      = (roundingMode == 3'b001);
    wire roundingMode_min         = (roundingMode == 3'b010);
    wire roundingMode_max         = (roundingMode == 3'b011);
    wire roundingMode_near_maxMag = (roundingMode == 3'b100);
    wire roundingMode_odd         = (roundingMode == 3'b110);
    wire roundMagUp =
        (roundingMode_min && in_sign) || (roundingMode_max && !in_sign);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire isNaNOut = invalidExc || (!infiniteExc && in_isNaN);
wire propagateNaNPayload = 0;

    wire signed [(adjustedExpWidth - 1):0] sAdjustedExp =
        in_sExp + ((1<<outExpWidth) - (1<<inExpWidth));
    wire [(outSigWidth + 2):0] adjustedSig;
    generate
        if (inSigWidth <= outSigWidth + 2) begin
            assign adjustedSig = in_sig<<(outSigWidth - inSigWidth + 2);
        end else begin
            assign adjustedSig =
                {in_sig[inSigWidth:(inSigWidth - outSigWidth - 1)],
                 |in_sig[(inSigWidth - outSigWidth - 2):0]};
        end
    endgenerate
    wire doShiftSigDown1 =
        sigMSBitAlwaysZero ? 0 : adjustedSig[outSigWidth + 2];
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire [outExpWidth:0] common_expOut;
    wire [(outSigWidth - 2):0] common_fractOut;
    wire common_overflow, common_totalUnderflow, common_underflow;
    wire common_inexact;
    generate
        if (
            neverOverflows && neverUnderflows
                && (effectiveInSigWidth <= outSigWidth)
        ) begin
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            assign common_expOut = sAdjustedExp + doShiftSigDown1;
            assign common_fractOut =
                doShiftSigDown1 ? adjustedSig>>3 : adjustedSig>>2;
            assign common_overflow       = 0;
            assign common_totalUnderflow = 0;
            assign common_underflow      = 0;
            assign common_inexact        = 0;
        end else begin
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            wire [(outSigWidth + 2):0] roundMask;
            if (neverUnderflows) begin
                assign roundMask = {doShiftSigDown1, 2'b11};
            end else begin
                wire [outSigWidth:0] roundMask_main;
                lowMaskLoHi#(
                    outExpWidth + 1,
                    outMinNormExp - outSigWidth - 1,
                    outMinNormExp
                ) lowMask_roundMask(
                        sAdjustedExp[outExpWidth:0]
                            | (propagateNaNPayload ? 1'b1<<outExpWidth : 1'b0),
                        roundMask_main
                    );
                assign roundMask = {roundMask_main | doShiftSigDown1, 2'b11};
            end
            wire [(outSigWidth + 2):0] shiftedRoundMask = roundMask>>1;
            wire [(outSigWidth + 2):0] roundPosMask =
                ~shiftedRoundMask & roundMask;
            wire roundPosBit =
                (|(adjustedSig[(outSigWidth + 2):3]
                       & roundPosMask[(outSigWidth + 2):3]))
                    || ((|(adjustedSig[2:0] & roundPosMask[2:0]))
                            && !propagateNaNPayload);
            wire anyRoundExtra =
                (|(adjustedSig[(outSigWidth + 2):3]
                       & shiftedRoundMask[(outSigWidth + 2):3]))
                    || ((|(adjustedSig[2:0] & shiftedRoundMask[2:0]))
                            && !propagateNaNPayload);
            wire anyRound = roundPosBit || anyRoundExtra;
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            wire roundIncr =
                ((roundingMode_near_even || roundingMode_near_maxMag)
                     && roundPosBit)
                    || (roundMagUp && anyRound);
            wire [(outSigWidth + 1):0] roundedSig =
                roundIncr
                    ? (((adjustedSig | roundMask)>>2) + 1)
                          & ~(roundingMode_near_even && roundPosBit
                                  && !anyRoundExtra
                                  ? roundMask>>1 : 0)
                    : (adjustedSig & ~roundMask)>>2
                          | (roundingMode_odd && anyRound
                                 ? roundPosMask>>1 : 0);
            wire signed [adjustedExpWidth:0] sExtAdjustedExp = sAdjustedExp;
            wire signed [adjustedExpWidth:0] sRoundedExp =
                sExtAdjustedExp + (roundedSig>>outSigWidth);
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            assign common_expOut = sRoundedExp;
            assign common_fractOut =
                doShiftSigDown1 ? roundedSig>>1 : roundedSig;
            assign common_overflow =
                neverOverflows ? 0 : (sRoundedExp>>>(outExpWidth - 1) >= 3);
            assign common_totalUnderflow =
                neverUnderflows ? 0 : (sRoundedExp < outMinNonzeroExp);
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            wire unboundedRange_roundPosBit =
                doShiftSigDown1 ? adjustedSig[2] : adjustedSig[1];
            wire unboundedRange_anyRound =
                (doShiftSigDown1 && adjustedSig[2]) || (|adjustedSig[1:0]);
            wire unboundedRange_roundIncr =
                ((roundingMode_near_even || roundingMode_near_maxMag)
                     && unboundedRange_roundPosBit)
                    || (roundMagUp && unboundedRange_anyRound);
            wire roundCarry =
                doShiftSigDown1
                    ? roundedSig[outSigWidth + 1] : roundedSig[outSigWidth];
            assign common_underflow =
                neverUnderflows ? 0
                    : common_totalUnderflow
                          || (anyRound && (sAdjustedExp>>>outExpWidth <= 0)
                                  && (doShiftSigDown1
                                          ? roundMask[3] : roundMask[2])
                                  && !(((control
                                           & 1'b1)
                                            != 0)
                                           && !(doShiftSigDown1 ? roundMask[4]
                                                    : roundMask[3])
                                           && roundCarry && roundPosBit
                                           && unboundedRange_roundIncr));
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            assign common_inexact = common_totalUnderflow || anyRound;
        end
    endgenerate
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire notNaN_isSpecialInfOut = infiniteExc || in_isInf;
    wire commonCase = !isNaNOut && !notNaN_isSpecialInfOut && !in_isZero;
    wire overflow  = commonCase && common_overflow;
    wire underflow = commonCase && common_underflow;
    wire inexact = overflow || (commonCase && common_inexact);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire overflow_roundMagUp =
        roundingMode_near_even || roundingMode_near_maxMag || roundMagUp;
    wire pegMinNonzeroMagOut =
        commonCase && common_totalUnderflow
            && (roundMagUp || roundingMode_odd);
    wire pegMaxFiniteMagOut = overflow && !overflow_roundMagUp;
    wire notNaN_isInfOut =
        notNaN_isSpecialInfOut || (overflow && overflow_roundMagUp);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
wire signOut = isNaNOut ? 0 : in_sign;

    wire [outExpWidth:0] expOut =
        (common_expOut
             & ~(in_isZero || common_totalUnderflow ? 7<<(outExpWidth - 2) : 0)
             & ~(pegMinNonzeroMagOut               ? ~outMinNonzeroExp    : 0)
             & ~(pegMaxFiniteMagOut                ? 1<<(outExpWidth - 1) : 0)
             & ~(notNaN_isInfOut                   ? 1<<(outExpWidth - 2) : 0))
            | (pegMinNonzeroMagOut ? outMinNonzeroExp : 0)
            | (pegMaxFiniteMagOut  ? outMaxFiniteExp  : 0)
            | (notNaN_isInfOut     ? outInfExp        : 0)
            | (isNaNOut            ? outNaNExp        : 0);
wire [(outSigWidth - 2):0] fractOut =
          (isNaNOut ? {1'b1, {((outSigWidth) - 2){1'b0}}} : 0)
        | (!in_isZero && !common_totalUnderflow
               ? common_fractOut & {1'b1, {((outSigWidth) - 2){1'b0}}} : 0)
        | (!isNaNOut && !in_isZero && !common_totalUnderflow
               ? common_fractOut & ~{1'b1, {((outSigWidth) - 2){1'b0}}}
               : 0)
        | {(outSigWidth - 1){pegMaxFiniteMagOut}};

    assign out = {signOut, expOut, fractOut};
    assign exceptionFlags =
        {invalidExc, infiniteExc, overflow, underflow, inexact};

endmodule

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    roundRawFNToRecFN#(
        parameter expWidth = 3,
        parameter sigWidth = 3,
        parameter options = 0
    ) (
        input [(1 - 1):0] control,
        input invalidExc,     // overrides 'infiniteExc' and 'in_*' inputs
        input infiniteExc,    // overrides 'in_*' inputs except 'in_sign'
        input in_isNaN,
        input in_isInf,
        input in_isZero,
        input in_sign,
        input signed [(expWidth + 1):0] in_sExp,   // limited range allowed
        input [(sigWidth + 2):0] in_sig,
        input [2:0] roundingMode,
        output [(expWidth + sigWidth):0] out,
        output [4:0] exceptionFlags
    );

    roundAnyRawFNToRecFN#(expWidth, sigWidth + 2, expWidth, sigWidth, options)
        roundAnyRawFNToRecFN(
            control,
            invalidExc,
            infiniteExc,
            in_isNaN,
            in_isInf,
            in_isZero,
            in_sign,
            in_sExp,
            in_sig,
            roundingMode,
            out,
            exceptionFlags
        );

endmodule

