// Compiled by morty-0.9.0 / 2026-04-26 22:54:21.987495915 +05:45:00

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
/* verilator lint_off MULTITOP */
/// =================== Unsigned, Fixed Point =========================
module std_fp_add #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    output logic [WIDTH-1:0] out
);
  assign out = left + right;
endmodule

module std_fp_sub #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    output logic [WIDTH-1:0] out
);
  assign out = left - right;
endmodule

module std_fp_mult_pipe #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16,
    parameter SIGNED = 0
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    input  logic             go,
    input  logic             clk,
    input  logic             reset,
    output logic [WIDTH-1:0] out,
    output logic             done
);
  logic [WIDTH-1:0]          rtmp;
  logic [WIDTH-1:0]          ltmp;
  logic [(WIDTH << 1) - 1:0] out_tmp;
  // Buffer used to walk through the 3 cycles of the pipeline.
  logic done_buf[1:0];

  assign done = done_buf[1];

  assign out = out_tmp[(WIDTH << 1) - INT_WIDTH - 1 : WIDTH - INT_WIDTH];

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

  // Register the inputs
  always_ff @(posedge clk) begin
    if (reset) begin
      rtmp <= 0;
      ltmp <= 0;
    end else if (go) begin
      if (SIGNED) begin
        rtmp <= $signed(right);
        ltmp <= $signed(left);
      end else begin
        rtmp <= right;
        ltmp <= left;
      end
    end else begin
      rtmp <= 0;
      ltmp <= 0;
    end

  end

  // Compute the output and save it into out_tmp
  always_ff @(posedge clk) begin
    if (reset) begin
      out_tmp <= 0;
    end else if (go) begin
      if (SIGNED) begin
        // In the first cycle, this performs an invalid computation because
        // ltmp and rtmp only get their actual values in cycle 1
        out_tmp <= $signed(
          { {WIDTH{ltmp[WIDTH-1]}}, ltmp} *
          { {WIDTH{rtmp[WIDTH-1]}}, rtmp}
        );
      end else begin
        out_tmp <= ltmp * rtmp;
      end
    end else begin
      out_tmp <= out_tmp;
    end
  end
endmodule

/* verilator lint_off WIDTH */
module std_fp_div_pipe #(
  parameter WIDTH = 32,
  parameter INT_WIDTH = 16,
  parameter FRAC_WIDTH = 16
) (
    input  logic             go,
    input  logic             clk,
    input  logic             reset,
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    output logic [WIDTH-1:0] out_remainder,
    output logic [WIDTH-1:0] out_quotient,
    output logic             done
);
    localparam ITERATIONS = WIDTH + FRAC_WIDTH;

    logic [WIDTH-1:0] quotient, quotient_next;
    logic [WIDTH:0] acc, acc_next;
    logic [$clog2(ITERATIONS)-1:0] idx;
    logic start, running, finished, dividend_is_zero;

    assign start = go && !running;
    assign dividend_is_zero = start && left == 0;
    assign finished = idx == ITERATIONS - 1 && running;

    always_ff @(posedge clk) begin
      if (reset || finished || dividend_is_zero)
        running <= 0;
      else if (start)
        running <= 1;
      else
        running <= running;
    end

    always @* begin
      if (acc >= {1'b0, right}) begin
        acc_next = acc - right;
        {acc_next, quotient_next} = {acc_next[WIDTH-1:0], quotient, 1'b1};
      end else begin
        {acc_next, quotient_next} = {acc, quotient} << 1;
      end
    end

    // `done` signaling
    always_ff @(posedge clk) begin
      if (dividend_is_zero || finished)
        done <= 1;
      else
        done <= 0;
    end

    always_ff @(posedge clk) begin
      if (running)
        idx <= idx + 1;
      else
        idx <= 0;
    end

    always_ff @(posedge clk) begin
      if (reset) begin
        out_quotient <= 0;
        out_remainder <= 0;
      end else if (start) begin
        out_quotient <= 0;
        out_remainder <= left;
      end else if (go == 0) begin
        out_quotient <= out_quotient;
        out_remainder <= out_remainder;
      end else if (dividend_is_zero) begin
        out_quotient <= 0;
        out_remainder <= 0;
      end else if (finished) begin
        out_quotient <= quotient_next;
        out_remainder <= out_remainder;
      end else begin
        out_quotient <= out_quotient;
        if (right <= out_remainder)
          out_remainder <= out_remainder - right;
        else
          out_remainder <= out_remainder;
      end
    end

    always_ff @(posedge clk) begin
      if (reset) begin
        acc <= 0;
        quotient <= 0;
      end else if (start) begin
        {acc, quotient} <= {{WIDTH{1'b0}}, left, 1'b0};
      end else begin
        acc <= acc_next;
        quotient <= quotient_next;
      end
    end
endmodule

module std_fp_gt #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    output logic             out
);
  assign out = left > right;
endmodule

/// =================== Signed, Fixed Point =========================
module std_fp_sadd #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = $signed(left + right);
endmodule

module std_fp_ssub #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);

  assign out = $signed(left - right);
endmodule

module std_fp_smult_pipe #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  [WIDTH-1:0]              left,
    input  [WIDTH-1:0]              right,
    input  logic                    reset,
    input  logic                    go,
    input  logic                    clk,
    output logic [WIDTH-1:0]        out,
    output logic                    done
);
  std_fp_mult_pipe #(
    .WIDTH(WIDTH),
    .INT_WIDTH(INT_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH),
    .SIGNED(1)
  ) comp (
    .clk(clk),
    .done(done),
    .reset(reset),
    .go(go),
    .left(left),
    .right(right),
    .out(out)
  );
endmodule

module std_fp_sdiv_pipe #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input                     clk,
    input                     go,
    input                     reset,
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out_quotient,
    output signed [WIDTH-1:0] out_remainder,
    output logic              done
);

  logic signed [WIDTH-1:0] left_abs, right_abs, comp_out_q, comp_out_r, right_save, out_rem_intermediate;

  // Registers to figure out how to transform outputs.
  logic different_signs, left_sign, right_sign;

  // Latch the value of control registers so that their available after
  // go signal becomes low.
  always_ff @(posedge clk) begin
    if (go) begin
      right_save <= right_abs;
      left_sign <= left[WIDTH-1];
      right_sign <= right[WIDTH-1];
    end else begin
      left_sign <= left_sign;
      right_save <= right_save;
      right_sign <= right_sign;
    end
  end

  assign right_abs = right[WIDTH-1] ? -right : right;
  assign left_abs = left[WIDTH-1] ? -left : left;

  assign different_signs = left_sign ^ right_sign;
  assign out_quotient = different_signs ? -comp_out_q : comp_out_q;

  // Remainder is computed as:
  //  t0 = |left| % |right|
  //  t1 = if left * right < 0 and t0 != 0 then |right| - t0 else t0
  //  rem = if right < 0 then -t1 else t1
  assign out_rem_intermediate = different_signs & |comp_out_r ? $signed(right_save - comp_out_r) : comp_out_r;
  assign out_remainder = right_sign ? -out_rem_intermediate : out_rem_intermediate;

  std_fp_div_pipe #(
    .WIDTH(WIDTH),
    .INT_WIDTH(INT_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH)
  ) comp (
    .reset(reset),
    .clk(clk),
    .done(done),
    .go(go),
    .left(left_abs),
    .right(right_abs),
    .out_quotient(comp_out_q),
    .out_remainder(comp_out_r)
  );
endmodule

module std_fp_sgt #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic signed [WIDTH-1:0] left,
    input  logic signed [WIDTH-1:0] right,
    output logic signed             out
);
  assign out = $signed(left > right);
endmodule

module std_fp_slt #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
   input logic signed [WIDTH-1:0] left,
   input logic signed [WIDTH-1:0] right,
   output logic signed            out
);
  assign out = $signed(left < right);
endmodule

/// =================== Unsigned, Bitnum =========================
module std_mult_pipe #(
    parameter WIDTH = 32
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    input  logic             reset,
    input  logic             go,
    input  logic             clk,
    output logic [WIDTH-1:0] out,
    output logic             done
);
  std_fp_mult_pipe #(
    .WIDTH(WIDTH),
    .INT_WIDTH(WIDTH),
    .FRAC_WIDTH(0),
    .SIGNED(0)
  ) comp (
    .reset(reset),
    .clk(clk),
    .done(done),
    .go(go),
    .left(left),
    .right(right),
    .out(out)
  );
endmodule

module std_div_pipe #(
    parameter WIDTH = 32
) (
    input                    reset,
    input                    clk,
    input                    go,
    input        [WIDTH-1:0] left,
    input        [WIDTH-1:0] right,
    output logic [WIDTH-1:0] out_remainder,
    output logic [WIDTH-1:0] out_quotient,
    output logic             done
);

  logic [WIDTH-1:0] dividend;
  logic [(WIDTH-1)*2:0] divisor;
  logic [WIDTH-1:0] quotient;
  logic [WIDTH-1:0] quotient_msk;
  logic start, running, finished, dividend_is_zero;

  assign start = go && !running;
  assign finished = quotient_msk == 0 && running;
  assign dividend_is_zero = start && left == 0;

  always_ff @(posedge clk) begin
    // Early return if the divisor is zero.
    if (finished || dividend_is_zero)
      done <= 1;
    else
      done <= 0;
  end

  always_ff @(posedge clk) begin
    if (reset || finished || dividend_is_zero)
      running <= 0;
    else if (start)
      running <= 1;
    else
      running <= running;
  end

  // Outputs
  always_ff @(posedge clk) begin
    if (dividend_is_zero || start) begin
      out_quotient <= 0;
      out_remainder <= 0;
    end else if (finished) begin
      out_quotient <= quotient;
      out_remainder <= dividend;
    end else begin
      // Otherwise, explicitly latch the values.
      out_quotient <= out_quotient;
      out_remainder <= out_remainder;
    end
  end

  // Calculate the quotient mask.
  always_ff @(posedge clk) begin
    if (start)
      quotient_msk <= 1 << WIDTH - 1;
    else if (running)
      quotient_msk <= quotient_msk >> 1;
    else
      quotient_msk <= quotient_msk;
  end

  // Calculate the quotient.
  always_ff @(posedge clk) begin
    if (start)
      quotient <= 0;
    else if (divisor <= dividend)
      quotient <= quotient | quotient_msk;
    else
      quotient <= quotient;
  end

  // Calculate the dividend.
  always_ff @(posedge clk) begin
    if (start)
      dividend <= left;
    else if (divisor <= dividend)
      dividend <= dividend - divisor;
    else
      dividend <= dividend;
  end

  always_ff @(posedge clk) begin
    if (start) begin
      divisor <= right << WIDTH - 1;
    end else if (finished) begin
      divisor <= 0;
    end else begin
      divisor <= divisor >> 1;
    end
  end

  // Simulation self test against unsynthesizable implementation.
  
endmodule

/// =================== Signed, Bitnum =========================
module std_sadd #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = $signed(left + right);
endmodule

module std_ssub #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = $signed(left - right);
endmodule

module std_smult_pipe #(
    parameter WIDTH = 32
) (
    input  logic                    reset,
    input  logic                    go,
    input  logic                    clk,
    input  signed       [WIDTH-1:0] left,
    input  signed       [WIDTH-1:0] right,
    output logic signed [WIDTH-1:0] out,
    output logic                    done
);
  std_fp_mult_pipe #(
    .WIDTH(WIDTH),
    .INT_WIDTH(WIDTH),
    .FRAC_WIDTH(0),
    .SIGNED(1)
  ) comp (
    .reset(reset),
    .clk(clk),
    .done(done),
    .go(go),
    .left(left),
    .right(right),
    .out(out)
  );
endmodule

/* verilator lint_off WIDTH */
module std_sdiv_pipe #(
    parameter WIDTH = 32
) (
    input                           reset,
    input                           clk,
    input                           go,
    input  logic signed [WIDTH-1:0] left,
    input  logic signed [WIDTH-1:0] right,
    output logic signed [WIDTH-1:0] out_quotient,
    output logic signed [WIDTH-1:0] out_remainder,
    output logic                    done
);

  logic signed [WIDTH-1:0] left_abs, right_abs, comp_out_q, comp_out_r, right_save, out_rem_intermediate;

  // Registers to figure out how to transform outputs.
  logic different_signs, left_sign, right_sign;

  // Latch the value of control registers so that their available after
  // go signal becomes low.
  always_ff @(posedge clk) begin
    if (go) begin
      right_save <= right_abs;
      left_sign <= left[WIDTH-1];
      right_sign <= right[WIDTH-1];
    end else begin
      left_sign <= left_sign;
      right_save <= right_save;
      right_sign <= right_sign;
    end
  end

  assign right_abs = right[WIDTH-1] ? -right : right;
  assign left_abs = left[WIDTH-1] ? -left : left;

  assign different_signs = left_sign ^ right_sign;
  assign out_quotient = different_signs ? -comp_out_q : comp_out_q;

  // Remainder is computed as:
  //  t0 = |left| % |right|
  //  t1 = if left * right < 0 and t0 != 0 then |right| - t0 else t0
  //  rem = if right < 0 then -t1 else t1
  assign out_rem_intermediate = different_signs & |comp_out_r ? $signed(right_save - comp_out_r) : comp_out_r;
  assign out_remainder = right_sign ? -out_rem_intermediate : out_rem_intermediate;

  std_div_pipe #(
    .WIDTH(WIDTH)
  ) comp (
    .reset(reset),
    .clk(clk),
    .done(done),
    .go(go),
    .left(left_abs),
    .right(right_abs),
    .out_quotient(comp_out_q),
    .out_remainder(comp_out_r)
  );

  // Simulation self test against unsynthesizable implementation.
  
endmodule

module std_sgt #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left > right);
endmodule

module std_slt #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left < right);
endmodule

module std_seq #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left == right);
endmodule

module std_sneq #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left != right);
endmodule

module std_sge #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left >= right);
endmodule

module std_sle #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left <= right);
endmodule

module std_slsh #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = left <<< right;
endmodule

module std_srsh #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = left >>> right;
endmodule

// Signed extension
module std_signext #(
  parameter IN_WIDTH  = 32,
  parameter OUT_WIDTH = 32
) (
  input wire logic [IN_WIDTH-1:0]  in,
  output logic     [OUT_WIDTH-1:0] out
);
  localparam EXTEND = OUT_WIDTH - IN_WIDTH;
  assign out = { {EXTEND {in[IN_WIDTH-1]}}, in};

  
endmodule

module std_const_mult #(
    parameter WIDTH = 32,
    parameter VALUE = 1
) (
    input  signed [WIDTH-1:0] in,
    output signed [WIDTH-1:0] out
);
  assign out = in * VALUE;
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
logic [31:0] std_add_5_left;
logic [31:0] std_add_5_right;
logic [31:0] std_add_5_out;
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
logic [31:0] muli_1_reg_in;
logic muli_1_reg_write_en;
logic muli_1_reg_clk;
logic muli_1_reg_reset;
logic [31:0] muli_1_reg_out;
logic muli_1_reg_done;
logic std_mult_pipe_1_clk;
logic std_mult_pipe_1_reset;
logic std_mult_pipe_1_go;
logic [31:0] std_mult_pipe_1_left;
logic [31:0] std_mult_pipe_1_right;
logic [31:0] std_mult_pipe_1_out;
logic std_mult_pipe_1_done;
logic mem_0_clk;
logic mem_0_reset;
logic [8:0] mem_0_addr0;
logic mem_0_content_en;
logic mem_0_write_en;
logic [31:0] mem_0_write_data;
logic [31:0] mem_0_read_data;
logic mem_0_done;
logic [31:0] for_2_induction_var_reg_in;
logic for_2_induction_var_reg_write_en;
logic for_2_induction_var_reg_clk;
logic for_2_induction_var_reg_reset;
logic [31:0] for_2_induction_var_reg_out;
logic for_2_induction_var_reg_done;
logic [31:0] for_1_induction_var_reg_in;
logic for_1_induction_var_reg_write_en;
logic for_1_induction_var_reg_clk;
logic for_1_induction_var_reg_reset;
logic [31:0] for_1_induction_var_reg_out;
logic for_1_induction_var_reg_done;
logic [31:0] for_0_induction_var_reg_in;
logic for_0_induction_var_reg_write_en;
logic for_0_induction_var_reg_clk;
logic for_0_induction_var_reg_reset;
logic [31:0] for_0_induction_var_reg_out;
logic for_0_induction_var_reg_done;
logic [3:0] idx_in;
logic idx_write_en;
logic idx_clk;
logic idx_reset;
logic [3:0] idx_out;
logic idx_done;
logic cond_reg_in;
logic cond_reg_write_en;
logic cond_reg_clk;
logic cond_reg_reset;
logic cond_reg_out;
logic cond_reg_done;
logic [3:0] adder_left;
logic [3:0] adder_right;
logic [3:0] adder_out;
logic [3:0] lt_left;
logic [3:0] lt_right;
logic lt_out;
logic [3:0] idx0_in;
logic idx0_write_en;
logic idx0_clk;
logic idx0_reset;
logic [3:0] idx0_out;
logic idx0_done;
logic cond_reg0_in;
logic cond_reg0_write_en;
logic cond_reg0_clk;
logic cond_reg0_reset;
logic cond_reg0_out;
logic cond_reg0_done;
logic [3:0] adder0_left;
logic [3:0] adder0_right;
logic [3:0] adder0_out;
logic [3:0] lt0_left;
logic [3:0] lt0_right;
logic lt0_out;
logic [1:0] idx1_in;
logic idx1_write_en;
logic idx1_clk;
logic idx1_reset;
logic [1:0] idx1_out;
logic idx1_done;
logic cond_reg1_in;
logic cond_reg1_write_en;
logic cond_reg1_clk;
logic cond_reg1_reset;
logic cond_reg1_out;
logic cond_reg1_done;
logic [1:0] adder1_left;
logic [1:0] adder1_right;
logic [1:0] adder1_out;
logic [1:0] lt1_left;
logic [1:0] lt1_right;
logic lt1_out;
logic [8:0] idx2_in;
logic idx2_write_en;
logic idx2_clk;
logic idx2_reset;
logic [8:0] idx2_out;
logic idx2_done;
logic cond_reg2_in;
logic cond_reg2_write_en;
logic cond_reg2_clk;
logic cond_reg2_reset;
logic cond_reg2_out;
logic cond_reg2_done;
logic [8:0] adder2_left;
logic [8:0] adder2_right;
logic [8:0] adder2_out;
logic [8:0] lt2_left;
logic [8:0] lt2_right;
logic lt2_out;
logic [3:0] fsm_in;
logic fsm_write_en;
logic fsm_clk;
logic fsm_reset;
logic [3:0] fsm_out;
logic fsm_done;
logic [3:0] adder3_left;
logic [3:0] adder3_right;
logic [3:0] adder3_out;
logic ud_out;
logic [3:0] adder4_left;
logic [3:0] adder4_right;
logic [3:0] adder4_out;
logic ud0_out;
logic [3:0] adder5_left;
logic [3:0] adder5_right;
logic [3:0] adder5_out;
logic ud1_out;
logic signal_reg_in;
logic signal_reg_write_en;
logic signal_reg_clk;
logic signal_reg_reset;
logic signal_reg_out;
logic signal_reg_done;
logic [4:0] fsm0_in;
logic fsm0_write_en;
logic fsm0_clk;
logic fsm0_reset;
logic [4:0] fsm0_out;
logic fsm0_done;
logic bb0_4_go_in;
logic bb0_4_go_out;
logic bb0_4_done_in;
logic bb0_4_done_out;
logic bb0_5_go_in;
logic bb0_5_go_out;
logic bb0_5_done_in;
logic bb0_5_done_out;
logic bb0_9_go_in;
logic bb0_9_go_out;
logic bb0_9_done_in;
logic bb0_9_done_out;
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
logic invoke9_go_in;
logic invoke9_go_out;
logic invoke9_done_in;
logic invoke9_done_out;
logic invoke10_go_in;
logic invoke10_go_out;
logic invoke10_done_in;
logic invoke10_done_out;
logic invoke11_go_in;
logic invoke11_go_out;
logic invoke11_done_in;
logic invoke11_done_out;
logic invoke14_go_in;
logic invoke14_go_out;
logic invoke14_done_in;
logic invoke14_done_out;
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
logic init_repeat1_go_in;
logic init_repeat1_go_out;
logic init_repeat1_done_in;
logic init_repeat1_done_out;
logic incr_repeat1_go_in;
logic incr_repeat1_go_out;
logic incr_repeat1_done_in;
logic incr_repeat1_done_out;
logic init_repeat2_go_in;
logic init_repeat2_go_out;
logic init_repeat2_done_in;
logic init_repeat2_done_out;
logic incr_repeat2_go_in;
logic incr_repeat2_go_out;
logic incr_repeat2_done_in;
logic incr_repeat2_done_out;
logic early_reset_static_seq_go_in;
logic early_reset_static_seq_go_out;
logic early_reset_static_seq_done_in;
logic early_reset_static_seq_done_out;
logic early_reset_static_seq0_go_in;
logic early_reset_static_seq0_go_out;
logic early_reset_static_seq0_done_in;
logic early_reset_static_seq0_done_out;
logic early_reset_static_seq1_go_in;
logic early_reset_static_seq1_go_out;
logic early_reset_static_seq1_done_in;
logic early_reset_static_seq1_done_out;
logic wrapper_early_reset_static_seq_go_in;
logic wrapper_early_reset_static_seq_go_out;
logic wrapper_early_reset_static_seq_done_in;
logic wrapper_early_reset_static_seq_done_out;
logic wrapper_early_reset_static_seq0_go_in;
logic wrapper_early_reset_static_seq0_go_out;
logic wrapper_early_reset_static_seq0_done_in;
logic wrapper_early_reset_static_seq0_done_out;
logic wrapper_early_reset_static_seq1_go_in;
logic wrapper_early_reset_static_seq1_go_out;
logic wrapper_early_reset_static_seq1_done_in;
logic wrapper_early_reset_static_seq1_done_out;
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
) std_add_5 (
    .left(std_add_5_left),
    .out(std_add_5_out),
    .right(std_add_5_right)
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
std_reg # (
    .WIDTH(32)
) muli_1_reg (
    .clk(muli_1_reg_clk),
    .done(muli_1_reg_done),
    .in(muli_1_reg_in),
    .out(muli_1_reg_out),
    .reset(muli_1_reg_reset),
    .write_en(muli_1_reg_write_en)
);
std_mult_pipe # (
    .WIDTH(32)
) std_mult_pipe_1 (
    .clk(std_mult_pipe_1_clk),
    .done(std_mult_pipe_1_done),
    .go(std_mult_pipe_1_go),
    .left(std_mult_pipe_1_left),
    .out(std_mult_pipe_1_out),
    .reset(std_mult_pipe_1_reset),
    .right(std_mult_pipe_1_right)
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
) for_2_induction_var_reg (
    .clk(for_2_induction_var_reg_clk),
    .done(for_2_induction_var_reg_done),
    .in(for_2_induction_var_reg_in),
    .out(for_2_induction_var_reg_out),
    .reset(for_2_induction_var_reg_reset),
    .write_en(for_2_induction_var_reg_write_en)
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
    .WIDTH(32)
) for_0_induction_var_reg (
    .clk(for_0_induction_var_reg_clk),
    .done(for_0_induction_var_reg_done),
    .in(for_0_induction_var_reg_in),
    .out(for_0_induction_var_reg_out),
    .reset(for_0_induction_var_reg_reset),
    .write_en(for_0_induction_var_reg_write_en)
);
std_reg # (
    .WIDTH(4)
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
    .WIDTH(4)
) adder (
    .left(adder_left),
    .out(adder_out),
    .right(adder_right)
);
std_lt # (
    .WIDTH(4)
) lt (
    .left(lt_left),
    .out(lt_out),
    .right(lt_right)
);
std_reg # (
    .WIDTH(4)
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
    .WIDTH(4)
) adder0 (
    .left(adder0_left),
    .out(adder0_out),
    .right(adder0_right)
);
std_lt # (
    .WIDTH(4)
) lt0 (
    .left(lt0_left),
    .out(lt0_out),
    .right(lt0_right)
);
std_reg # (
    .WIDTH(2)
) idx1 (
    .clk(idx1_clk),
    .done(idx1_done),
    .in(idx1_in),
    .out(idx1_out),
    .reset(idx1_reset),
    .write_en(idx1_write_en)
);
std_reg # (
    .WIDTH(1)
) cond_reg1 (
    .clk(cond_reg1_clk),
    .done(cond_reg1_done),
    .in(cond_reg1_in),
    .out(cond_reg1_out),
    .reset(cond_reg1_reset),
    .write_en(cond_reg1_write_en)
);
std_add # (
    .WIDTH(2)
) adder1 (
    .left(adder1_left),
    .out(adder1_out),
    .right(adder1_right)
);
std_lt # (
    .WIDTH(2)
) lt1 (
    .left(lt1_left),
    .out(lt1_out),
    .right(lt1_right)
);
std_reg # (
    .WIDTH(9)
) idx2 (
    .clk(idx2_clk),
    .done(idx2_done),
    .in(idx2_in),
    .out(idx2_out),
    .reset(idx2_reset),
    .write_en(idx2_write_en)
);
std_reg # (
    .WIDTH(1)
) cond_reg2 (
    .clk(cond_reg2_clk),
    .done(cond_reg2_done),
    .in(cond_reg2_in),
    .out(cond_reg2_out),
    .reset(cond_reg2_reset),
    .write_en(cond_reg2_write_en)
);
std_add # (
    .WIDTH(9)
) adder2 (
    .left(adder2_left),
    .out(adder2_out),
    .right(adder2_right)
);
std_lt # (
    .WIDTH(9)
) lt2 (
    .left(lt2_left),
    .out(lt2_out),
    .right(lt2_right)
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
std_add # (
    .WIDTH(4)
) adder3 (
    .left(adder3_left),
    .out(adder3_out),
    .right(adder3_right)
);
undef # (
    .WIDTH(1)
) ud (
    .out(ud_out)
);
std_add # (
    .WIDTH(4)
) adder4 (
    .left(adder4_left),
    .out(adder4_out),
    .right(adder4_right)
);
undef # (
    .WIDTH(1)
) ud0 (
    .out(ud0_out)
);
std_add # (
    .WIDTH(4)
) adder5 (
    .left(adder5_left),
    .out(adder5_out),
    .right(adder5_right)
);
undef # (
    .WIDTH(1)
) ud1 (
    .out(ud1_out)
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
    .WIDTH(5)
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
) bb0_9_go (
    .in(bb0_9_go_in),
    .out(bb0_9_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_9_done (
    .in(bb0_9_done_in),
    .out(bb0_9_done_out)
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
) invoke9_go (
    .in(invoke9_go_in),
    .out(invoke9_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke9_done (
    .in(invoke9_done_in),
    .out(invoke9_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke10_go (
    .in(invoke10_go_in),
    .out(invoke10_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke10_done (
    .in(invoke10_done_in),
    .out(invoke10_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke11_go (
    .in(invoke11_go_in),
    .out(invoke11_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke11_done (
    .in(invoke11_done_in),
    .out(invoke11_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke14_go (
    .in(invoke14_go_in),
    .out(invoke14_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke14_done (
    .in(invoke14_done_in),
    .out(invoke14_done_out)
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
) init_repeat1_go (
    .in(init_repeat1_go_in),
    .out(init_repeat1_go_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat1_done (
    .in(init_repeat1_done_in),
    .out(init_repeat1_done_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat1_go (
    .in(incr_repeat1_go_in),
    .out(incr_repeat1_go_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat1_done (
    .in(incr_repeat1_done_in),
    .out(incr_repeat1_done_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat2_go (
    .in(init_repeat2_go_in),
    .out(init_repeat2_go_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat2_done (
    .in(init_repeat2_done_in),
    .out(init_repeat2_done_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat2_go (
    .in(incr_repeat2_go_in),
    .out(incr_repeat2_go_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat2_done (
    .in(incr_repeat2_done_in),
    .out(incr_repeat2_done_out)
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
) early_reset_static_seq1_go (
    .in(early_reset_static_seq1_go_in),
    .out(early_reset_static_seq1_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_static_seq1_done (
    .in(early_reset_static_seq1_done_in),
    .out(early_reset_static_seq1_done_out)
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
) wrapper_early_reset_static_seq1_go (
    .in(wrapper_early_reset_static_seq1_go_in),
    .out(wrapper_early_reset_static_seq1_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_seq1_done (
    .in(wrapper_early_reset_static_seq1_done_in),
    .out(wrapper_early_reset_static_seq1_done_out)
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
wire _guard1 = fsm_out == 4'd1;
wire _guard2 = early_reset_static_seq0_go_out;
wire _guard3 = _guard1 & _guard2;
wire _guard4 = bb0_4_go_out;
wire _guard5 = fsm_out >= 4'd4;
wire _guard6 = fsm_out < 4'd7;
wire _guard7 = _guard5 & _guard6;
wire _guard8 = early_reset_static_seq_go_out;
wire _guard9 = _guard7 & _guard8;
wire _guard10 = _guard4 | _guard9;
wire _guard11 = fsm_out == 4'd0;
wire _guard12 = early_reset_static_seq0_go_out;
wire _guard13 = _guard11 & _guard12;
wire _guard14 = _guard10 | _guard13;
wire _guard15 = invoke10_go_out;
wire _guard16 = invoke14_go_out;
wire _guard17 = _guard15 | _guard16;
wire _guard18 = invoke9_go_out;
wire _guard19 = bb0_4_go_out;
wire _guard20 = fsm_out == 4'd0;
wire _guard21 = early_reset_static_seq0_go_out;
wire _guard22 = _guard20 & _guard21;
wire _guard23 = _guard19 | _guard22;
wire _guard24 = invoke9_go_out;
wire _guard25 = invoke10_go_out;
wire _guard26 = _guard24 | _guard25;
wire _guard27 = invoke14_go_out;
wire _guard28 = _guard26 | _guard27;
wire _guard29 = fsm_out == 4'd1;
wire _guard30 = early_reset_static_seq0_go_out;
wire _guard31 = _guard29 & _guard30;
wire _guard32 = _guard28 | _guard31;
wire _guard33 = fsm_out >= 4'd4;
wire _guard34 = fsm_out < 4'd7;
wire _guard35 = _guard33 & _guard34;
wire _guard36 = early_reset_static_seq_go_out;
wire _guard37 = _guard35 & _guard36;
wire _guard38 = bb0_5_go_out;
wire _guard39 = bb0_5_go_out;
wire _guard40 = init_repeat1_go_out;
wire _guard41 = incr_repeat1_go_out;
wire _guard42 = _guard40 | _guard41;
wire _guard43 = init_repeat1_go_out;
wire _guard44 = incr_repeat1_go_out;
wire _guard45 = incr_repeat1_go_out;
wire _guard46 = incr_repeat1_go_out;
wire _guard47 = invoke9_done_out;
wire _guard48 = ~_guard47;
wire _guard49 = fsm0_out == 5'd11;
wire _guard50 = _guard48 & _guard49;
wire _guard51 = tdcc_go_out;
wire _guard52 = _guard50 & _guard51;
wire _guard53 = init_repeat_done_out;
wire _guard54 = ~_guard53;
wire _guard55 = fsm0_out == 5'd5;
wire _guard56 = _guard54 & _guard55;
wire _guard57 = tdcc_go_out;
wire _guard58 = _guard56 & _guard57;
wire _guard59 = tdcc_done_out;
wire _guard60 = bb0_9_go_out;
wire _guard61 = bb0_4_go_out;
wire _guard62 = bb0_4_go_out;
wire _guard63 = bb0_9_go_out;
wire _guard64 = bb0_9_go_out;
wire _guard65 = bb0_9_go_out;
wire _guard66 = incr_repeat_go_out;
wire _guard67 = incr_repeat_go_out;
wire _guard68 = fsm_out != 4'd7;
wire _guard69 = early_reset_static_seq_go_out;
wire _guard70 = _guard68 & _guard69;
wire _guard71 = fsm_out == 4'd7;
wire _guard72 = early_reset_static_seq_go_out;
wire _guard73 = _guard71 & _guard72;
wire _guard74 = _guard70 | _guard73;
wire _guard75 = fsm_out != 4'd1;
wire _guard76 = early_reset_static_seq0_go_out;
wire _guard77 = _guard75 & _guard76;
wire _guard78 = _guard74 | _guard77;
wire _guard79 = fsm_out == 4'd1;
wire _guard80 = early_reset_static_seq0_go_out;
wire _guard81 = _guard79 & _guard80;
wire _guard82 = _guard78 | _guard81;
wire _guard83 = fsm_out != 4'd1;
wire _guard84 = early_reset_static_seq1_go_out;
wire _guard85 = _guard83 & _guard84;
wire _guard86 = _guard82 | _guard85;
wire _guard87 = fsm_out == 4'd1;
wire _guard88 = early_reset_static_seq1_go_out;
wire _guard89 = _guard87 & _guard88;
wire _guard90 = _guard86 | _guard89;
wire _guard91 = fsm_out != 4'd1;
wire _guard92 = early_reset_static_seq0_go_out;
wire _guard93 = _guard91 & _guard92;
wire _guard94 = fsm_out == 4'd7;
wire _guard95 = early_reset_static_seq_go_out;
wire _guard96 = _guard94 & _guard95;
wire _guard97 = fsm_out == 4'd1;
wire _guard98 = early_reset_static_seq0_go_out;
wire _guard99 = _guard97 & _guard98;
wire _guard100 = _guard96 | _guard99;
wire _guard101 = fsm_out == 4'd1;
wire _guard102 = early_reset_static_seq1_go_out;
wire _guard103 = _guard101 & _guard102;
wire _guard104 = _guard100 | _guard103;
wire _guard105 = fsm_out != 4'd7;
wire _guard106 = early_reset_static_seq_go_out;
wire _guard107 = _guard105 & _guard106;
wire _guard108 = fsm_out != 4'd1;
wire _guard109 = early_reset_static_seq1_go_out;
wire _guard110 = _guard108 & _guard109;
wire _guard111 = incr_repeat1_done_out;
wire _guard112 = ~_guard111;
wire _guard113 = fsm0_out == 5'd14;
wire _guard114 = _guard112 & _guard113;
wire _guard115 = tdcc_go_out;
wire _guard116 = _guard114 & _guard115;
wire _guard117 = fsm_out == 4'd0;
wire _guard118 = early_reset_static_seq0_go_out;
wire _guard119 = _guard117 & _guard118;
wire _guard120 = fsm_out == 4'd0;
wire _guard121 = early_reset_static_seq0_go_out;
wire _guard122 = _guard120 & _guard121;
wire _guard123 = fsm_out == 4'd0;
wire _guard124 = early_reset_static_seq0_go_out;
wire _guard125 = _guard123 & _guard124;
wire _guard126 = init_repeat1_go_out;
wire _guard127 = incr_repeat1_go_out;
wire _guard128 = _guard126 | _guard127;
wire _guard129 = incr_repeat1_go_out;
wire _guard130 = init_repeat1_go_out;
wire _guard131 = incr_repeat1_go_out;
wire _guard132 = incr_repeat1_go_out;
wire _guard133 = incr_repeat2_go_out;
wire _guard134 = incr_repeat2_go_out;
wire _guard135 = early_reset_static_seq0_go_out;
wire _guard136 = early_reset_static_seq0_go_out;
wire _guard137 = invoke2_done_out;
wire _guard138 = ~_guard137;
wire _guard139 = fsm0_out == 5'd4;
wire _guard140 = _guard138 & _guard139;
wire _guard141 = tdcc_go_out;
wire _guard142 = _guard140 & _guard141;
wire _guard143 = init_repeat0_go_out;
wire _guard144 = incr_repeat0_go_out;
wire _guard145 = _guard143 | _guard144;
wire _guard146 = init_repeat0_go_out;
wire _guard147 = incr_repeat0_go_out;
wire _guard148 = bb0_5_go_out;
wire _guard149 = bb0_5_go_out;
wire _guard150 = invoke2_go_out;
wire _guard151 = fsm_out == 4'd1;
wire _guard152 = early_reset_static_seq0_go_out;
wire _guard153 = _guard151 & _guard152;
wire _guard154 = _guard150 | _guard153;
wire _guard155 = fsm_out == 4'd1;
wire _guard156 = early_reset_static_seq0_go_out;
wire _guard157 = _guard155 & _guard156;
wire _guard158 = invoke2_go_out;
wire _guard159 = init_repeat1_done_out;
wire _guard160 = ~_guard159;
wire _guard161 = fsm0_out == 5'd1;
wire _guard162 = _guard160 & _guard161;
wire _guard163 = tdcc_go_out;
wire _guard164 = _guard162 & _guard163;
wire _guard165 = wrapper_early_reset_static_seq0_go_out;
wire _guard166 = signal_reg_out;
wire _guard167 = bb0_4_go_out;
wire _guard168 = fsm_out == 4'd0;
wire _guard169 = early_reset_static_seq0_go_out;
wire _guard170 = _guard168 & _guard169;
wire _guard171 = _guard167 | _guard170;
wire _guard172 = bb0_9_go_out;
wire _guard173 = fsm_out == 4'd0;
wire _guard174 = early_reset_static_seq1_go_out;
wire _guard175 = _guard173 & _guard174;
wire _guard176 = _guard172 | _guard175;
wire _guard177 = bb0_5_go_out;
wire _guard178 = bb0_5_go_out;
wire _guard179 = init_repeat0_go_out;
wire _guard180 = incr_repeat0_go_out;
wire _guard181 = _guard179 | _guard180;
wire _guard182 = init_repeat0_go_out;
wire _guard183 = incr_repeat0_go_out;
wire _guard184 = bb0_9_done_out;
wire _guard185 = ~_guard184;
wire _guard186 = fsm0_out == 5'd18;
wire _guard187 = _guard185 & _guard186;
wire _guard188 = tdcc_go_out;
wire _guard189 = _guard187 & _guard188;
wire _guard190 = invoke11_done_out;
wire _guard191 = ~_guard190;
wire _guard192 = fsm0_out == 5'd15;
wire _guard193 = _guard191 & _guard192;
wire _guard194 = tdcc_go_out;
wire _guard195 = _guard193 & _guard194;
wire _guard196 = wrapper_early_reset_static_seq0_done_out;
wire _guard197 = ~_guard196;
wire _guard198 = fsm0_out == 5'd9;
wire _guard199 = _guard197 & _guard198;
wire _guard200 = tdcc_go_out;
wire _guard201 = _guard199 & _guard200;
wire _guard202 = bb0_5_go_out;
wire _guard203 = bb0_5_go_out;
wire _guard204 = fsm_out == 4'd3;
wire _guard205 = fsm_out == 4'd7;
wire _guard206 = _guard204 | _guard205;
wire _guard207 = early_reset_static_seq_go_out;
wire _guard208 = _guard206 & _guard207;
wire _guard209 = fsm_out == 4'd1;
wire _guard210 = early_reset_static_seq1_go_out;
wire _guard211 = _guard209 & _guard210;
wire _guard212 = _guard208 | _guard211;
wire _guard213 = fsm_out == 4'd1;
wire _guard214 = early_reset_static_seq1_go_out;
wire _guard215 = _guard213 & _guard214;
wire _guard216 = fsm_out == 4'd3;
wire _guard217 = fsm_out == 4'd7;
wire _guard218 = _guard216 | _guard217;
wire _guard219 = early_reset_static_seq_go_out;
wire _guard220 = _guard218 & _guard219;
wire _guard221 = invoke0_done_out;
wire _guard222 = ~_guard221;
wire _guard223 = fsm0_out == 5'd0;
wire _guard224 = _guard222 & _guard223;
wire _guard225 = tdcc_go_out;
wire _guard226 = _guard224 & _guard225;
wire _guard227 = incr_repeat_done_out;
wire _guard228 = ~_guard227;
wire _guard229 = fsm0_out == 5'd10;
wire _guard230 = _guard228 & _guard229;
wire _guard231 = tdcc_go_out;
wire _guard232 = _guard230 & _guard231;
wire _guard233 = cond_reg1_done;
wire _guard234 = idx1_done;
wire _guard235 = _guard233 & _guard234;
wire _guard236 = wrapper_early_reset_static_seq1_go_out;
wire _guard237 = incr_repeat2_go_out;
wire _guard238 = incr_repeat2_go_out;
wire _guard239 = fsm0_out == 5'd21;
wire _guard240 = fsm0_out == 5'd0;
wire _guard241 = invoke0_done_out;
wire _guard242 = _guard240 & _guard241;
wire _guard243 = tdcc_go_out;
wire _guard244 = _guard242 & _guard243;
wire _guard245 = _guard239 | _guard244;
wire _guard246 = fsm0_out == 5'd1;
wire _guard247 = init_repeat1_done_out;
wire _guard248 = cond_reg1_out;
wire _guard249 = _guard247 & _guard248;
wire _guard250 = _guard246 & _guard249;
wire _guard251 = tdcc_go_out;
wire _guard252 = _guard250 & _guard251;
wire _guard253 = _guard245 | _guard252;
wire _guard254 = fsm0_out == 5'd14;
wire _guard255 = incr_repeat1_done_out;
wire _guard256 = cond_reg1_out;
wire _guard257 = _guard255 & _guard256;
wire _guard258 = _guard254 & _guard257;
wire _guard259 = tdcc_go_out;
wire _guard260 = _guard258 & _guard259;
wire _guard261 = _guard253 | _guard260;
wire _guard262 = fsm0_out == 5'd2;
wire _guard263 = invoke1_done_out;
wire _guard264 = _guard262 & _guard263;
wire _guard265 = tdcc_go_out;
wire _guard266 = _guard264 & _guard265;
wire _guard267 = _guard261 | _guard266;
wire _guard268 = fsm0_out == 5'd3;
wire _guard269 = init_repeat0_done_out;
wire _guard270 = cond_reg0_out;
wire _guard271 = _guard269 & _guard270;
wire _guard272 = _guard268 & _guard271;
wire _guard273 = tdcc_go_out;
wire _guard274 = _guard272 & _guard273;
wire _guard275 = _guard267 | _guard274;
wire _guard276 = fsm0_out == 5'd12;
wire _guard277 = incr_repeat0_done_out;
wire _guard278 = cond_reg0_out;
wire _guard279 = _guard277 & _guard278;
wire _guard280 = _guard276 & _guard279;
wire _guard281 = tdcc_go_out;
wire _guard282 = _guard280 & _guard281;
wire _guard283 = _guard275 | _guard282;
wire _guard284 = fsm0_out == 5'd4;
wire _guard285 = invoke2_done_out;
wire _guard286 = _guard284 & _guard285;
wire _guard287 = tdcc_go_out;
wire _guard288 = _guard286 & _guard287;
wire _guard289 = _guard283 | _guard288;
wire _guard290 = fsm0_out == 5'd5;
wire _guard291 = init_repeat_done_out;
wire _guard292 = cond_reg_out;
wire _guard293 = _guard291 & _guard292;
wire _guard294 = _guard290 & _guard293;
wire _guard295 = tdcc_go_out;
wire _guard296 = _guard294 & _guard295;
wire _guard297 = _guard289 | _guard296;
wire _guard298 = fsm0_out == 5'd10;
wire _guard299 = incr_repeat_done_out;
wire _guard300 = cond_reg_out;
wire _guard301 = _guard299 & _guard300;
wire _guard302 = _guard298 & _guard301;
wire _guard303 = tdcc_go_out;
wire _guard304 = _guard302 & _guard303;
wire _guard305 = _guard297 | _guard304;
wire _guard306 = fsm0_out == 5'd6;
wire _guard307 = wrapper_early_reset_static_seq_done_out;
wire _guard308 = _guard306 & _guard307;
wire _guard309 = tdcc_go_out;
wire _guard310 = _guard308 & _guard309;
wire _guard311 = _guard305 | _guard310;
wire _guard312 = fsm0_out == 5'd7;
wire _guard313 = bb0_4_done_out;
wire _guard314 = _guard312 & _guard313;
wire _guard315 = tdcc_go_out;
wire _guard316 = _guard314 & _guard315;
wire _guard317 = _guard311 | _guard316;
wire _guard318 = fsm0_out == 5'd8;
wire _guard319 = bb0_5_done_out;
wire _guard320 = _guard318 & _guard319;
wire _guard321 = tdcc_go_out;
wire _guard322 = _guard320 & _guard321;
wire _guard323 = _guard317 | _guard322;
wire _guard324 = fsm0_out == 5'd9;
wire _guard325 = wrapper_early_reset_static_seq0_done_out;
wire _guard326 = _guard324 & _guard325;
wire _guard327 = tdcc_go_out;
wire _guard328 = _guard326 & _guard327;
wire _guard329 = _guard323 | _guard328;
wire _guard330 = fsm0_out == 5'd5;
wire _guard331 = init_repeat_done_out;
wire _guard332 = cond_reg_out;
wire _guard333 = ~_guard332;
wire _guard334 = _guard331 & _guard333;
wire _guard335 = _guard330 & _guard334;
wire _guard336 = tdcc_go_out;
wire _guard337 = _guard335 & _guard336;
wire _guard338 = _guard329 | _guard337;
wire _guard339 = fsm0_out == 5'd10;
wire _guard340 = incr_repeat_done_out;
wire _guard341 = cond_reg_out;
wire _guard342 = ~_guard341;
wire _guard343 = _guard340 & _guard342;
wire _guard344 = _guard339 & _guard343;
wire _guard345 = tdcc_go_out;
wire _guard346 = _guard344 & _guard345;
wire _guard347 = _guard338 | _guard346;
wire _guard348 = fsm0_out == 5'd11;
wire _guard349 = invoke9_done_out;
wire _guard350 = _guard348 & _guard349;
wire _guard351 = tdcc_go_out;
wire _guard352 = _guard350 & _guard351;
wire _guard353 = _guard347 | _guard352;
wire _guard354 = fsm0_out == 5'd3;
wire _guard355 = init_repeat0_done_out;
wire _guard356 = cond_reg0_out;
wire _guard357 = ~_guard356;
wire _guard358 = _guard355 & _guard357;
wire _guard359 = _guard354 & _guard358;
wire _guard360 = tdcc_go_out;
wire _guard361 = _guard359 & _guard360;
wire _guard362 = _guard353 | _guard361;
wire _guard363 = fsm0_out == 5'd12;
wire _guard364 = incr_repeat0_done_out;
wire _guard365 = cond_reg0_out;
wire _guard366 = ~_guard365;
wire _guard367 = _guard364 & _guard366;
wire _guard368 = _guard363 & _guard367;
wire _guard369 = tdcc_go_out;
wire _guard370 = _guard368 & _guard369;
wire _guard371 = _guard362 | _guard370;
wire _guard372 = fsm0_out == 5'd13;
wire _guard373 = invoke10_done_out;
wire _guard374 = _guard372 & _guard373;
wire _guard375 = tdcc_go_out;
wire _guard376 = _guard374 & _guard375;
wire _guard377 = _guard371 | _guard376;
wire _guard378 = fsm0_out == 5'd1;
wire _guard379 = init_repeat1_done_out;
wire _guard380 = cond_reg1_out;
wire _guard381 = ~_guard380;
wire _guard382 = _guard379 & _guard381;
wire _guard383 = _guard378 & _guard382;
wire _guard384 = tdcc_go_out;
wire _guard385 = _guard383 & _guard384;
wire _guard386 = _guard377 | _guard385;
wire _guard387 = fsm0_out == 5'd14;
wire _guard388 = incr_repeat1_done_out;
wire _guard389 = cond_reg1_out;
wire _guard390 = ~_guard389;
wire _guard391 = _guard388 & _guard390;
wire _guard392 = _guard387 & _guard391;
wire _guard393 = tdcc_go_out;
wire _guard394 = _guard392 & _guard393;
wire _guard395 = _guard386 | _guard394;
wire _guard396 = fsm0_out == 5'd15;
wire _guard397 = invoke11_done_out;
wire _guard398 = _guard396 & _guard397;
wire _guard399 = tdcc_go_out;
wire _guard400 = _guard398 & _guard399;
wire _guard401 = _guard395 | _guard400;
wire _guard402 = fsm0_out == 5'd16;
wire _guard403 = init_repeat2_done_out;
wire _guard404 = cond_reg2_out;
wire _guard405 = _guard403 & _guard404;
wire _guard406 = _guard402 & _guard405;
wire _guard407 = tdcc_go_out;
wire _guard408 = _guard406 & _guard407;
wire _guard409 = _guard401 | _guard408;
wire _guard410 = fsm0_out == 5'd20;
wire _guard411 = incr_repeat2_done_out;
wire _guard412 = cond_reg2_out;
wire _guard413 = _guard411 & _guard412;
wire _guard414 = _guard410 & _guard413;
wire _guard415 = tdcc_go_out;
wire _guard416 = _guard414 & _guard415;
wire _guard417 = _guard409 | _guard416;
wire _guard418 = fsm0_out == 5'd17;
wire _guard419 = wrapper_early_reset_static_seq1_done_out;
wire _guard420 = _guard418 & _guard419;
wire _guard421 = tdcc_go_out;
wire _guard422 = _guard420 & _guard421;
wire _guard423 = _guard417 | _guard422;
wire _guard424 = fsm0_out == 5'd18;
wire _guard425 = bb0_9_done_out;
wire _guard426 = _guard424 & _guard425;
wire _guard427 = tdcc_go_out;
wire _guard428 = _guard426 & _guard427;
wire _guard429 = _guard423 | _guard428;
wire _guard430 = fsm0_out == 5'd19;
wire _guard431 = invoke14_done_out;
wire _guard432 = _guard430 & _guard431;
wire _guard433 = tdcc_go_out;
wire _guard434 = _guard432 & _guard433;
wire _guard435 = _guard429 | _guard434;
wire _guard436 = fsm0_out == 5'd16;
wire _guard437 = init_repeat2_done_out;
wire _guard438 = cond_reg2_out;
wire _guard439 = ~_guard438;
wire _guard440 = _guard437 & _guard439;
wire _guard441 = _guard436 & _guard440;
wire _guard442 = tdcc_go_out;
wire _guard443 = _guard441 & _guard442;
wire _guard444 = _guard435 | _guard443;
wire _guard445 = fsm0_out == 5'd20;
wire _guard446 = incr_repeat2_done_out;
wire _guard447 = cond_reg2_out;
wire _guard448 = ~_guard447;
wire _guard449 = _guard446 & _guard448;
wire _guard450 = _guard445 & _guard449;
wire _guard451 = tdcc_go_out;
wire _guard452 = _guard450 & _guard451;
wire _guard453 = _guard444 | _guard452;
wire _guard454 = fsm0_out == 5'd0;
wire _guard455 = invoke0_done_out;
wire _guard456 = _guard454 & _guard455;
wire _guard457 = tdcc_go_out;
wire _guard458 = _guard456 & _guard457;
wire _guard459 = fsm0_out == 5'd1;
wire _guard460 = init_repeat1_done_out;
wire _guard461 = cond_reg1_out;
wire _guard462 = ~_guard461;
wire _guard463 = _guard460 & _guard462;
wire _guard464 = _guard459 & _guard463;
wire _guard465 = tdcc_go_out;
wire _guard466 = _guard464 & _guard465;
wire _guard467 = fsm0_out == 5'd14;
wire _guard468 = incr_repeat1_done_out;
wire _guard469 = cond_reg1_out;
wire _guard470 = ~_guard469;
wire _guard471 = _guard468 & _guard470;
wire _guard472 = _guard467 & _guard471;
wire _guard473 = tdcc_go_out;
wire _guard474 = _guard472 & _guard473;
wire _guard475 = _guard466 | _guard474;
wire _guard476 = fsm0_out == 5'd17;
wire _guard477 = wrapper_early_reset_static_seq1_done_out;
wire _guard478 = _guard476 & _guard477;
wire _guard479 = tdcc_go_out;
wire _guard480 = _guard478 & _guard479;
wire _guard481 = fsm0_out == 5'd15;
wire _guard482 = invoke11_done_out;
wire _guard483 = _guard481 & _guard482;
wire _guard484 = tdcc_go_out;
wire _guard485 = _guard483 & _guard484;
wire _guard486 = fsm0_out == 5'd21;
wire _guard487 = fsm0_out == 5'd2;
wire _guard488 = invoke1_done_out;
wire _guard489 = _guard487 & _guard488;
wire _guard490 = tdcc_go_out;
wire _guard491 = _guard489 & _guard490;
wire _guard492 = fsm0_out == 5'd3;
wire _guard493 = init_repeat0_done_out;
wire _guard494 = cond_reg0_out;
wire _guard495 = ~_guard494;
wire _guard496 = _guard493 & _guard495;
wire _guard497 = _guard492 & _guard496;
wire _guard498 = tdcc_go_out;
wire _guard499 = _guard497 & _guard498;
wire _guard500 = fsm0_out == 5'd12;
wire _guard501 = incr_repeat0_done_out;
wire _guard502 = cond_reg0_out;
wire _guard503 = ~_guard502;
wire _guard504 = _guard501 & _guard503;
wire _guard505 = _guard500 & _guard504;
wire _guard506 = tdcc_go_out;
wire _guard507 = _guard505 & _guard506;
wire _guard508 = _guard499 | _guard507;
wire _guard509 = fsm0_out == 5'd13;
wire _guard510 = invoke10_done_out;
wire _guard511 = _guard509 & _guard510;
wire _guard512 = tdcc_go_out;
wire _guard513 = _guard511 & _guard512;
wire _guard514 = fsm0_out == 5'd4;
wire _guard515 = invoke2_done_out;
wire _guard516 = _guard514 & _guard515;
wire _guard517 = tdcc_go_out;
wire _guard518 = _guard516 & _guard517;
wire _guard519 = fsm0_out == 5'd11;
wire _guard520 = invoke9_done_out;
wire _guard521 = _guard519 & _guard520;
wire _guard522 = tdcc_go_out;
wire _guard523 = _guard521 & _guard522;
wire _guard524 = fsm0_out == 5'd1;
wire _guard525 = init_repeat1_done_out;
wire _guard526 = cond_reg1_out;
wire _guard527 = _guard525 & _guard526;
wire _guard528 = _guard524 & _guard527;
wire _guard529 = tdcc_go_out;
wire _guard530 = _guard528 & _guard529;
wire _guard531 = fsm0_out == 5'd14;
wire _guard532 = incr_repeat1_done_out;
wire _guard533 = cond_reg1_out;
wire _guard534 = _guard532 & _guard533;
wire _guard535 = _guard531 & _guard534;
wire _guard536 = tdcc_go_out;
wire _guard537 = _guard535 & _guard536;
wire _guard538 = _guard530 | _guard537;
wire _guard539 = fsm0_out == 5'd7;
wire _guard540 = bb0_4_done_out;
wire _guard541 = _guard539 & _guard540;
wire _guard542 = tdcc_go_out;
wire _guard543 = _guard541 & _guard542;
wire _guard544 = fsm0_out == 5'd9;
wire _guard545 = wrapper_early_reset_static_seq0_done_out;
wire _guard546 = _guard544 & _guard545;
wire _guard547 = tdcc_go_out;
wire _guard548 = _guard546 & _guard547;
wire _guard549 = fsm0_out == 5'd6;
wire _guard550 = wrapper_early_reset_static_seq_done_out;
wire _guard551 = _guard549 & _guard550;
wire _guard552 = tdcc_go_out;
wire _guard553 = _guard551 & _guard552;
wire _guard554 = fsm0_out == 5'd5;
wire _guard555 = init_repeat_done_out;
wire _guard556 = cond_reg_out;
wire _guard557 = ~_guard556;
wire _guard558 = _guard555 & _guard557;
wire _guard559 = _guard554 & _guard558;
wire _guard560 = tdcc_go_out;
wire _guard561 = _guard559 & _guard560;
wire _guard562 = fsm0_out == 5'd10;
wire _guard563 = incr_repeat_done_out;
wire _guard564 = cond_reg_out;
wire _guard565 = ~_guard564;
wire _guard566 = _guard563 & _guard565;
wire _guard567 = _guard562 & _guard566;
wire _guard568 = tdcc_go_out;
wire _guard569 = _guard567 & _guard568;
wire _guard570 = _guard561 | _guard569;
wire _guard571 = fsm0_out == 5'd16;
wire _guard572 = init_repeat2_done_out;
wire _guard573 = cond_reg2_out;
wire _guard574 = ~_guard573;
wire _guard575 = _guard572 & _guard574;
wire _guard576 = _guard571 & _guard575;
wire _guard577 = tdcc_go_out;
wire _guard578 = _guard576 & _guard577;
wire _guard579 = fsm0_out == 5'd20;
wire _guard580 = incr_repeat2_done_out;
wire _guard581 = cond_reg2_out;
wire _guard582 = ~_guard581;
wire _guard583 = _guard580 & _guard582;
wire _guard584 = _guard579 & _guard583;
wire _guard585 = tdcc_go_out;
wire _guard586 = _guard584 & _guard585;
wire _guard587 = _guard578 | _guard586;
wire _guard588 = fsm0_out == 5'd18;
wire _guard589 = bb0_9_done_out;
wire _guard590 = _guard588 & _guard589;
wire _guard591 = tdcc_go_out;
wire _guard592 = _guard590 & _guard591;
wire _guard593 = fsm0_out == 5'd3;
wire _guard594 = init_repeat0_done_out;
wire _guard595 = cond_reg0_out;
wire _guard596 = _guard594 & _guard595;
wire _guard597 = _guard593 & _guard596;
wire _guard598 = tdcc_go_out;
wire _guard599 = _guard597 & _guard598;
wire _guard600 = fsm0_out == 5'd12;
wire _guard601 = incr_repeat0_done_out;
wire _guard602 = cond_reg0_out;
wire _guard603 = _guard601 & _guard602;
wire _guard604 = _guard600 & _guard603;
wire _guard605 = tdcc_go_out;
wire _guard606 = _guard604 & _guard605;
wire _guard607 = _guard599 | _guard606;
wire _guard608 = fsm0_out == 5'd5;
wire _guard609 = init_repeat_done_out;
wire _guard610 = cond_reg_out;
wire _guard611 = _guard609 & _guard610;
wire _guard612 = _guard608 & _guard611;
wire _guard613 = tdcc_go_out;
wire _guard614 = _guard612 & _guard613;
wire _guard615 = fsm0_out == 5'd10;
wire _guard616 = incr_repeat_done_out;
wire _guard617 = cond_reg_out;
wire _guard618 = _guard616 & _guard617;
wire _guard619 = _guard615 & _guard618;
wire _guard620 = tdcc_go_out;
wire _guard621 = _guard619 & _guard620;
wire _guard622 = _guard614 | _guard621;
wire _guard623 = fsm0_out == 5'd19;
wire _guard624 = invoke14_done_out;
wire _guard625 = _guard623 & _guard624;
wire _guard626 = tdcc_go_out;
wire _guard627 = _guard625 & _guard626;
wire _guard628 = fsm0_out == 5'd16;
wire _guard629 = init_repeat2_done_out;
wire _guard630 = cond_reg2_out;
wire _guard631 = _guard629 & _guard630;
wire _guard632 = _guard628 & _guard631;
wire _guard633 = tdcc_go_out;
wire _guard634 = _guard632 & _guard633;
wire _guard635 = fsm0_out == 5'd20;
wire _guard636 = incr_repeat2_done_out;
wire _guard637 = cond_reg2_out;
wire _guard638 = _guard636 & _guard637;
wire _guard639 = _guard635 & _guard638;
wire _guard640 = tdcc_go_out;
wire _guard641 = _guard639 & _guard640;
wire _guard642 = _guard634 | _guard641;
wire _guard643 = fsm0_out == 5'd8;
wire _guard644 = bb0_5_done_out;
wire _guard645 = _guard643 & _guard644;
wire _guard646 = tdcc_go_out;
wire _guard647 = _guard645 & _guard646;
wire _guard648 = cond_reg0_done;
wire _guard649 = idx0_done;
wire _guard650 = _guard648 & _guard649;
wire _guard651 = incr_repeat2_done_out;
wire _guard652 = ~_guard651;
wire _guard653 = fsm0_out == 5'd20;
wire _guard654 = _guard652 & _guard653;
wire _guard655 = tdcc_go_out;
wire _guard656 = _guard654 & _guard655;
wire _guard657 = init_repeat_go_out;
wire _guard658 = incr_repeat_go_out;
wire _guard659 = _guard657 | _guard658;
wire _guard660 = incr_repeat_go_out;
wire _guard661 = init_repeat_go_out;
wire _guard662 = early_reset_static_seq_go_out;
wire _guard663 = early_reset_static_seq_go_out;
wire _guard664 = early_reset_static_seq1_go_out;
wire _guard665 = early_reset_static_seq1_go_out;
wire _guard666 = cond_reg_done;
wire _guard667 = idx_done;
wire _guard668 = _guard666 & _guard667;
wire _guard669 = cond_reg_done;
wire _guard670 = idx_done;
wire _guard671 = _guard669 & _guard670;
wire _guard672 = cond_reg0_done;
wire _guard673 = idx0_done;
wire _guard674 = _guard672 & _guard673;
wire _guard675 = signal_reg_out;
wire _guard676 = incr_repeat0_go_out;
wire _guard677 = incr_repeat0_go_out;
wire _guard678 = invoke1_done_out;
wire _guard679 = ~_guard678;
wire _guard680 = fsm0_out == 5'd2;
wire _guard681 = _guard679 & _guard680;
wire _guard682 = tdcc_go_out;
wire _guard683 = _guard681 & _guard682;
wire _guard684 = init_repeat0_done_out;
wire _guard685 = ~_guard684;
wire _guard686 = fsm0_out == 5'd3;
wire _guard687 = _guard685 & _guard686;
wire _guard688 = tdcc_go_out;
wire _guard689 = _guard687 & _guard688;
wire _guard690 = incr_repeat0_done_out;
wire _guard691 = ~_guard690;
wire _guard692 = fsm0_out == 5'd12;
wire _guard693 = _guard691 & _guard692;
wire _guard694 = tdcc_go_out;
wire _guard695 = _guard693 & _guard694;
wire _guard696 = init_repeat2_done_out;
wire _guard697 = ~_guard696;
wire _guard698 = fsm0_out == 5'd16;
wire _guard699 = _guard697 & _guard698;
wire _guard700 = tdcc_go_out;
wire _guard701 = _guard699 & _guard700;
wire _guard702 = wrapper_early_reset_static_seq_go_out;
wire _guard703 = fsm_out == 4'd0;
wire _guard704 = early_reset_static_seq0_go_out;
wire _guard705 = _guard703 & _guard704;
wire _guard706 = fsm_out == 4'd0;
wire _guard707 = early_reset_static_seq0_go_out;
wire _guard708 = _guard706 & _guard707;
wire _guard709 = fsm_out == 4'd0;
wire _guard710 = early_reset_static_seq1_go_out;
wire _guard711 = _guard709 & _guard710;
wire _guard712 = _guard708 | _guard711;
wire _guard713 = fsm_out == 4'd0;
wire _guard714 = early_reset_static_seq0_go_out;
wire _guard715 = _guard713 & _guard714;
wire _guard716 = fsm_out == 4'd0;
wire _guard717 = early_reset_static_seq1_go_out;
wire _guard718 = _guard716 & _guard717;
wire _guard719 = _guard715 | _guard718;
wire _guard720 = fsm_out == 4'd0;
wire _guard721 = early_reset_static_seq0_go_out;
wire _guard722 = _guard720 & _guard721;
wire _guard723 = signal_reg_out;
wire _guard724 = fsm_out == 4'd7;
wire _guard725 = _guard724 & _guard0;
wire _guard726 = signal_reg_out;
wire _guard727 = ~_guard726;
wire _guard728 = _guard725 & _guard727;
wire _guard729 = wrapper_early_reset_static_seq_go_out;
wire _guard730 = _guard728 & _guard729;
wire _guard731 = _guard723 | _guard730;
wire _guard732 = fsm_out == 4'd1;
wire _guard733 = _guard732 & _guard0;
wire _guard734 = signal_reg_out;
wire _guard735 = ~_guard734;
wire _guard736 = _guard733 & _guard735;
wire _guard737 = wrapper_early_reset_static_seq0_go_out;
wire _guard738 = _guard736 & _guard737;
wire _guard739 = _guard731 | _guard738;
wire _guard740 = fsm_out == 4'd1;
wire _guard741 = _guard740 & _guard0;
wire _guard742 = signal_reg_out;
wire _guard743 = ~_guard742;
wire _guard744 = _guard741 & _guard743;
wire _guard745 = wrapper_early_reset_static_seq1_go_out;
wire _guard746 = _guard744 & _guard745;
wire _guard747 = _guard739 | _guard746;
wire _guard748 = fsm_out == 4'd7;
wire _guard749 = _guard748 & _guard0;
wire _guard750 = signal_reg_out;
wire _guard751 = ~_guard750;
wire _guard752 = _guard749 & _guard751;
wire _guard753 = wrapper_early_reset_static_seq_go_out;
wire _guard754 = _guard752 & _guard753;
wire _guard755 = fsm_out == 4'd1;
wire _guard756 = _guard755 & _guard0;
wire _guard757 = signal_reg_out;
wire _guard758 = ~_guard757;
wire _guard759 = _guard756 & _guard758;
wire _guard760 = wrapper_early_reset_static_seq0_go_out;
wire _guard761 = _guard759 & _guard760;
wire _guard762 = _guard754 | _guard761;
wire _guard763 = fsm_out == 4'd1;
wire _guard764 = _guard763 & _guard0;
wire _guard765 = signal_reg_out;
wire _guard766 = ~_guard765;
wire _guard767 = _guard764 & _guard766;
wire _guard768 = wrapper_early_reset_static_seq1_go_out;
wire _guard769 = _guard767 & _guard768;
wire _guard770 = _guard762 | _guard769;
wire _guard771 = signal_reg_out;
wire _guard772 = cond_reg1_done;
wire _guard773 = idx1_done;
wire _guard774 = _guard772 & _guard773;
wire _guard775 = cond_reg2_done;
wire _guard776 = idx2_done;
wire _guard777 = _guard775 & _guard776;
wire _guard778 = invoke0_go_out;
wire _guard779 = invoke10_go_out;
wire _guard780 = _guard778 | _guard779;
wire _guard781 = invoke11_go_out;
wire _guard782 = _guard780 | _guard781;
wire _guard783 = invoke14_go_out;
wire _guard784 = _guard782 | _guard783;
wire _guard785 = invoke10_go_out;
wire _guard786 = invoke14_go_out;
wire _guard787 = _guard785 | _guard786;
wire _guard788 = invoke0_go_out;
wire _guard789 = invoke11_go_out;
wire _guard790 = _guard788 | _guard789;
wire _guard791 = bb0_5_go_out;
wire _guard792 = bb0_5_go_out;
wire _guard793 = fsm_out >= 4'd4;
wire _guard794 = fsm_out < 4'd7;
wire _guard795 = _guard793 & _guard794;
wire _guard796 = early_reset_static_seq_go_out;
wire _guard797 = _guard795 & _guard796;
wire _guard798 = fsm_out < 4'd3;
wire _guard799 = early_reset_static_seq_go_out;
wire _guard800 = _guard798 & _guard799;
wire _guard801 = fsm_out < 4'd3;
wire _guard802 = fsm_out >= 4'd4;
wire _guard803 = fsm_out < 4'd7;
wire _guard804 = _guard802 & _guard803;
wire _guard805 = _guard801 | _guard804;
wire _guard806 = early_reset_static_seq_go_out;
wire _guard807 = _guard805 & _guard806;
wire _guard808 = fsm_out < 4'd3;
wire _guard809 = fsm_out >= 4'd4;
wire _guard810 = fsm_out < 4'd7;
wire _guard811 = _guard809 & _guard810;
wire _guard812 = _guard808 | _guard811;
wire _guard813 = early_reset_static_seq_go_out;
wire _guard814 = _guard812 & _guard813;
wire _guard815 = wrapper_early_reset_static_seq1_done_out;
wire _guard816 = ~_guard815;
wire _guard817 = fsm0_out == 5'd17;
wire _guard818 = _guard816 & _guard817;
wire _guard819 = tdcc_go_out;
wire _guard820 = _guard818 & _guard819;
wire _guard821 = fsm0_out == 5'd21;
wire _guard822 = invoke1_go_out;
wire _guard823 = invoke9_go_out;
wire _guard824 = _guard822 | _guard823;
wire _guard825 = invoke9_go_out;
wire _guard826 = invoke1_go_out;
wire _guard827 = incr_repeat_go_out;
wire _guard828 = incr_repeat_go_out;
wire _guard829 = init_repeat2_go_out;
wire _guard830 = incr_repeat2_go_out;
wire _guard831 = _guard829 | _guard830;
wire _guard832 = init_repeat2_go_out;
wire _guard833 = incr_repeat2_go_out;
wire _guard834 = bb0_4_done_out;
wire _guard835 = ~_guard834;
wire _guard836 = fsm0_out == 5'd7;
wire _guard837 = _guard835 & _guard836;
wire _guard838 = tdcc_go_out;
wire _guard839 = _guard837 & _guard838;
wire _guard840 = wrapper_early_reset_static_seq_done_out;
wire _guard841 = ~_guard840;
wire _guard842 = fsm0_out == 5'd6;
wire _guard843 = _guard841 & _guard842;
wire _guard844 = tdcc_go_out;
wire _guard845 = _guard843 & _guard844;
wire _guard846 = signal_reg_out;
wire _guard847 = bb0_5_go_out;
wire _guard848 = std_compareFN_0_done;
wire _guard849 = ~_guard848;
wire _guard850 = bb0_5_go_out;
wire _guard851 = _guard849 & _guard850;
wire _guard852 = bb0_5_go_out;
wire _guard853 = bb0_5_go_out;
wire _guard854 = init_repeat_go_out;
wire _guard855 = incr_repeat_go_out;
wire _guard856 = _guard854 | _guard855;
wire _guard857 = init_repeat_go_out;
wire _guard858 = incr_repeat_go_out;
wire _guard859 = incr_repeat0_go_out;
wire _guard860 = incr_repeat0_go_out;
wire _guard861 = init_repeat2_go_out;
wire _guard862 = incr_repeat2_go_out;
wire _guard863 = _guard861 | _guard862;
wire _guard864 = incr_repeat2_go_out;
wire _guard865 = init_repeat2_go_out;
wire _guard866 = bb0_5_done_out;
wire _guard867 = ~_guard866;
wire _guard868 = fsm0_out == 5'd8;
wire _guard869 = _guard867 & _guard868;
wire _guard870 = tdcc_go_out;
wire _guard871 = _guard869 & _guard870;
wire _guard872 = invoke10_done_out;
wire _guard873 = ~_guard872;
wire _guard874 = fsm0_out == 5'd13;
wire _guard875 = _guard873 & _guard874;
wire _guard876 = tdcc_go_out;
wire _guard877 = _guard875 & _guard876;
wire _guard878 = invoke14_done_out;
wire _guard879 = ~_guard878;
wire _guard880 = fsm0_out == 5'd19;
wire _guard881 = _guard879 & _guard880;
wire _guard882 = tdcc_go_out;
wire _guard883 = _guard881 & _guard882;
wire _guard884 = cond_reg2_done;
wire _guard885 = idx2_done;
wire _guard886 = _guard884 & _guard885;
assign std_add_5_left =
  _guard3 ? for_0_induction_var_reg_out :
  _guard14 ? muli_1_reg_out :
  _guard17 ? for_2_induction_var_reg_out :
  _guard18 ? for_1_induction_var_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard18, _guard17, _guard14, _guard3})) begin
    $fatal(2, "Multiple assignment to port `std_add_5.left'.");
end
end
assign std_add_5_right =
  _guard23 ? for_0_induction_var_reg_out :
  _guard32 ? 32'd1 :
  _guard37 ? for_1_induction_var_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard37, _guard32, _guard23})) begin
    $fatal(2, "Multiple assignment to port `std_add_5.right'.");
end
end
assign unordered_port_0_reg_write_en =
  _guard38 ? std_compareFN_0_done :
  1'd0;
assign unordered_port_0_reg_clk = clk;
assign unordered_port_0_reg_reset = reset;
assign unordered_port_0_reg_in = std_compareFN_0_unordered;
assign cond_reg1_write_en = _guard42;
assign cond_reg1_clk = clk;
assign cond_reg1_reset = reset;
assign cond_reg1_in =
  _guard43 ? 1'd1 :
  _guard44 ? lt1_out :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard44, _guard43})) begin
    $fatal(2, "Multiple assignment to port `cond_reg1.in'.");
end
end
assign adder1_left =
  _guard45 ? idx1_out :
  2'd0;
assign adder1_right =
  _guard46 ? 2'd1 :
  2'd0;
assign invoke9_go_in = _guard52;
assign invoke9_done_in = for_1_induction_var_reg_done;
assign init_repeat_go_in = _guard58;
assign done = _guard59;
assign arg_mem_1_write_data = muli_1_reg_out;
assign arg_mem_0_content_en = _guard61;
assign arg_mem_0_addr0 = std_slice_3_out;
assign arg_mem_1_write_en = _guard63;
assign arg_mem_1_addr0 = std_slice_3_out;
assign arg_mem_1_content_en = _guard65;
assign adder_left =
  _guard66 ? idx_out :
  4'd0;
assign adder_right =
  _guard67 ? 4'd1 :
  4'd0;
assign fsm_write_en = _guard90;
assign fsm_clk = clk;
assign fsm_reset = reset;
assign fsm_in =
  _guard93 ? adder4_out :
  _guard104 ? 4'd0 :
  _guard107 ? adder3_out :
  _guard110 ? adder5_out :
  4'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard110, _guard107, _guard104, _guard93})) begin
    $fatal(2, "Multiple assignment to port `fsm.in'.");
end
end
assign invoke11_done_in = for_2_induction_var_reg_done;
assign invoke14_done_in = for_2_induction_var_reg_done;
assign incr_repeat1_go_in = _guard116;
assign std_mux_0_cond = cmpf_0_reg_out;
assign std_mux_0_tru = arg_mem_0_read_data;
assign std_mux_0_fal = cst_0_out;
assign idx1_write_en = _guard128;
assign idx1_clk = clk;
assign idx1_reset = reset;
assign idx1_in =
  _guard129 ? adder1_out :
  _guard130 ? 2'd0 :
  2'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard130, _guard129})) begin
    $fatal(2, "Multiple assignment to port `idx1.in'.");
end
end
assign lt1_left =
  _guard131 ? adder1_out :
  2'd0;
assign lt1_right =
  _guard132 ? 2'd3 :
  2'd0;
assign lt2_left =
  _guard133 ? adder2_out :
  9'd0;
assign lt2_right =
  _guard134 ? 9'd300 :
  9'd0;
assign adder4_left =
  _guard135 ? fsm_out :
  4'd0;
assign adder4_right =
  _guard136 ? 4'd1 :
  4'd0;
assign invoke2_go_in = _guard142;
assign cond_reg0_write_en = _guard145;
assign cond_reg0_clk = clk;
assign cond_reg0_reset = reset;
assign cond_reg0_in =
  _guard146 ? 1'd1 :
  _guard147 ? lt0_out :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard147, _guard146})) begin
    $fatal(2, "Multiple assignment to port `cond_reg0.in'.");
end
end
assign std_and_0_left =
  _guard148 ? compare_port_0_reg_done :
  1'd0;
assign std_and_0_right =
  _guard149 ? unordered_port_0_reg_done :
  1'd0;
assign for_0_induction_var_reg_write_en = _guard154;
assign for_0_induction_var_reg_clk = clk;
assign for_0_induction_var_reg_reset = reset;
assign for_0_induction_var_reg_in =
  _guard157 ? std_add_5_out :
  _guard158 ? 32'd0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard158, _guard157})) begin
    $fatal(2, "Multiple assignment to port `for_0_induction_var_reg.in'.");
end
end
assign init_repeat1_go_in = _guard164;
assign early_reset_static_seq0_go_in = _guard165;
assign wrapper_early_reset_static_seq1_done_in = _guard166;
assign std_slice_3_in =
  _guard171 ? std_add_5_out :
  _guard176 ? for_2_induction_var_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard176, _guard171})) begin
    $fatal(2, "Multiple assignment to port `std_slice_3.in'.");
end
end
assign compare_port_0_reg_write_en =
  _guard177 ? std_compareFN_0_done :
  1'd0;
assign compare_port_0_reg_clk = clk;
assign compare_port_0_reg_reset = reset;
assign compare_port_0_reg_in = std_compareFN_0_gt;
assign idx0_write_en = _guard181;
assign idx0_clk = clk;
assign idx0_reset = reset;
assign idx0_in =
  _guard182 ? 4'd0 :
  _guard183 ? adder0_out :
  4'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard183, _guard182})) begin
    $fatal(2, "Multiple assignment to port `idx0.in'.");
end
end
assign bb0_9_go_in = _guard189;
assign invoke11_go_in = _guard195;
assign early_reset_static_seq1_done_in = ud1_out;
assign wrapper_early_reset_static_seq0_go_in = _guard201;
assign std_or_0_left = compare_port_0_reg_out;
assign std_or_0_right = unordered_port_0_reg_out;
assign muli_1_reg_write_en = _guard212;
assign muli_1_reg_clk = clk;
assign muli_1_reg_reset = reset;
assign muli_1_reg_in =
  _guard215 ? mem_0_read_data :
  _guard220 ? std_mult_pipe_1_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard220, _guard215})) begin
    $fatal(2, "Multiple assignment to port `muli_1_reg.in'.");
end
end
assign invoke0_go_in = _guard226;
assign incr_repeat_go_in = _guard232;
assign init_repeat1_done_in = _guard235;
assign early_reset_static_seq1_go_in = _guard236;
assign tdcc_go_in = go;
assign adder2_left =
  _guard237 ? idx2_out :
  9'd0;
assign adder2_right =
  _guard238 ? 9'd1 :
  9'd0;
assign fsm0_write_en = _guard453;
assign fsm0_clk = clk;
assign fsm0_reset = reset;
assign fsm0_in =
  _guard458 ? 5'd1 :
  _guard475 ? 5'd15 :
  _guard480 ? 5'd18 :
  _guard485 ? 5'd16 :
  _guard486 ? 5'd0 :
  _guard491 ? 5'd3 :
  _guard508 ? 5'd13 :
  _guard513 ? 5'd14 :
  _guard518 ? 5'd5 :
  _guard523 ? 5'd12 :
  _guard538 ? 5'd2 :
  _guard543 ? 5'd8 :
  _guard548 ? 5'd10 :
  _guard553 ? 5'd7 :
  _guard570 ? 5'd11 :
  _guard587 ? 5'd21 :
  _guard592 ? 5'd19 :
  _guard607 ? 5'd4 :
  _guard622 ? 5'd6 :
  _guard627 ? 5'd20 :
  _guard642 ? 5'd17 :
  _guard647 ? 5'd9 :
  5'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard647, _guard642, _guard627, _guard622, _guard607, _guard592, _guard587, _guard570, _guard553, _guard548, _guard543, _guard538, _guard523, _guard518, _guard513, _guard508, _guard491, _guard486, _guard485, _guard480, _guard475, _guard458})) begin
    $fatal(2, "Multiple assignment to port `fsm0.in'.");
end
end
assign invoke10_done_in = for_2_induction_var_reg_done;
assign incr_repeat0_done_in = _guard650;
assign incr_repeat2_go_in = _guard656;
assign idx_write_en = _guard659;
assign idx_clk = clk;
assign idx_reset = reset;
assign idx_in =
  _guard660 ? adder_out :
  _guard661 ? 4'd0 :
  4'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard661, _guard660})) begin
    $fatal(2, "Multiple assignment to port `idx.in'.");
end
end
assign adder3_left =
  _guard662 ? fsm_out :
  4'd0;
assign adder3_right =
  _guard663 ? 4'd1 :
  4'd0;
assign adder5_left =
  _guard664 ? fsm_out :
  4'd0;
assign adder5_right =
  _guard665 ? 4'd1 :
  4'd0;
assign bb0_9_done_in = arg_mem_1_done;
assign init_repeat_done_in = _guard668;
assign incr_repeat_done_in = _guard671;
assign init_repeat0_done_in = _guard674;
assign wrapper_early_reset_static_seq_done_in = _guard675;
assign adder0_left =
  _guard676 ? idx0_out :
  4'd0;
assign adder0_right =
  _guard677 ? 4'd1 :
  4'd0;
assign invoke0_done_in = for_2_induction_var_reg_done;
assign invoke1_go_in = _guard683;
assign init_repeat0_go_in = _guard689;
assign incr_repeat0_go_in = _guard695;
assign init_repeat2_go_in = _guard701;
assign early_reset_static_seq_go_in = _guard702;
assign mem_0_write_en = _guard705;
assign mem_0_clk = clk;
assign mem_0_addr0 = std_slice_3_out;
assign mem_0_content_en = _guard719;
assign mem_0_reset = reset;
assign mem_0_write_data = std_mux_0_out;
assign signal_reg_write_en = _guard747;
assign signal_reg_clk = clk;
assign signal_reg_reset = reset;
assign signal_reg_in =
  _guard770 ? 1'd1 :
  _guard771 ? 1'd0 :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard771, _guard770})) begin
    $fatal(2, "Multiple assignment to port `signal_reg.in'.");
end
end
assign bb0_5_done_in = cmpf_0_reg_done;
assign invoke2_done_in = for_0_induction_var_reg_done;
assign incr_repeat1_done_in = _guard774;
assign init_repeat2_done_in = _guard777;
assign for_2_induction_var_reg_write_en = _guard784;
assign for_2_induction_var_reg_clk = clk;
assign for_2_induction_var_reg_reset = reset;
assign for_2_induction_var_reg_in =
  _guard787 ? std_add_5_out :
  _guard790 ? 32'd0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard790, _guard787})) begin
    $fatal(2, "Multiple assignment to port `for_2_induction_var_reg.in'.");
end
end
assign cmpf_0_reg_write_en =
  _guard791 ? std_and_0_out :
  1'd0;
assign cmpf_0_reg_clk = clk;
assign cmpf_0_reg_reset = reset;
assign cmpf_0_reg_in = std_or_0_out;
assign std_mult_pipe_1_clk = clk;
assign std_mult_pipe_1_left =
  _guard797 ? std_add_5_out :
  _guard800 ? for_2_induction_var_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard800, _guard797})) begin
    $fatal(2, "Multiple assignment to port `std_mult_pipe_1.left'.");
end
end
assign std_mult_pipe_1_reset = reset;
assign std_mult_pipe_1_go = _guard807;
assign std_mult_pipe_1_right = 32'd10;
assign early_reset_static_seq_done_in = ud_out;
assign wrapper_early_reset_static_seq1_go_in = _guard820;
assign tdcc_done_in = _guard821;
assign for_1_induction_var_reg_write_en = _guard824;
assign for_1_induction_var_reg_clk = clk;
assign for_1_induction_var_reg_reset = reset;
assign for_1_induction_var_reg_in =
  _guard825 ? std_add_5_out :
  _guard826 ? 32'd0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard826, _guard825})) begin
    $fatal(2, "Multiple assignment to port `for_1_induction_var_reg.in'.");
end
end
assign lt_left =
  _guard827 ? adder_out :
  4'd0;
assign lt_right =
  _guard828 ? 4'd10 :
  4'd0;
assign cond_reg2_write_en = _guard831;
assign cond_reg2_clk = clk;
assign cond_reg2_reset = reset;
assign cond_reg2_in =
  _guard832 ? 1'd1 :
  _guard833 ? lt2_out :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard833, _guard832})) begin
    $fatal(2, "Multiple assignment to port `cond_reg2.in'.");
end
end
assign bb0_4_go_in = _guard839;
assign bb0_4_done_in = arg_mem_0_done;
assign invoke1_done_in = for_1_induction_var_reg_done;
assign early_reset_static_seq0_done_in = ud0_out;
assign wrapper_early_reset_static_seq_go_in = _guard845;
assign wrapper_early_reset_static_seq0_done_in = _guard846;
assign std_compareFN_0_clk = clk;
assign std_compareFN_0_left =
  _guard847 ? arg_mem_0_read_data :
  32'd0;
assign std_compareFN_0_reset = reset;
assign std_compareFN_0_go = _guard851;
assign std_compareFN_0_signaling = _guard852;
assign std_compareFN_0_right =
  _guard853 ? cst_0_out :
  32'd0;
assign cond_reg_write_en = _guard856;
assign cond_reg_clk = clk;
assign cond_reg_reset = reset;
assign cond_reg_in =
  _guard857 ? 1'd1 :
  _guard858 ? lt_out :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard858, _guard857})) begin
    $fatal(2, "Multiple assignment to port `cond_reg.in'.");
end
end
assign lt0_left =
  _guard859 ? adder0_out :
  4'd0;
assign lt0_right =
  _guard860 ? 4'd10 :
  4'd0;
assign idx2_write_en = _guard863;
assign idx2_clk = clk;
assign idx2_reset = reset;
assign idx2_in =
  _guard864 ? adder2_out :
  _guard865 ? 9'd0 :
  9'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard865, _guard864})) begin
    $fatal(2, "Multiple assignment to port `idx2.in'.");
end
end
assign bb0_5_go_in = _guard871;
assign invoke10_go_in = _guard877;
assign invoke14_go_in = _guard883;
assign incr_repeat2_done_in = _guard886;
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
logic [31:0] std_add_5_left;
logic [31:0] std_add_5_right;
logic [31:0] std_add_5_out;
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
logic [31:0] muli_1_reg_in;
logic muli_1_reg_write_en;
logic muli_1_reg_clk;
logic muli_1_reg_reset;
logic [31:0] muli_1_reg_out;
logic muli_1_reg_done;
logic std_mult_pipe_1_clk;
logic std_mult_pipe_1_reset;
logic std_mult_pipe_1_go;
logic [31:0] std_mult_pipe_1_left;
logic [31:0] std_mult_pipe_1_right;
logic [31:0] std_mult_pipe_1_out;
logic std_mult_pipe_1_done;
logic [31:0] for_2_induction_var_reg_in;
logic for_2_induction_var_reg_write_en;
logic for_2_induction_var_reg_clk;
logic for_2_induction_var_reg_reset;
logic [31:0] for_2_induction_var_reg_out;
logic for_2_induction_var_reg_done;
logic [31:0] for_1_induction_var_reg_in;
logic for_1_induction_var_reg_write_en;
logic for_1_induction_var_reg_clk;
logic for_1_induction_var_reg_reset;
logic [31:0] for_1_induction_var_reg_out;
logic for_1_induction_var_reg_done;
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
logic [3:0] idx_in;
logic idx_write_en;
logic idx_clk;
logic idx_reset;
logic [3:0] idx_out;
logic idx_done;
logic cond_reg_in;
logic cond_reg_write_en;
logic cond_reg_clk;
logic cond_reg_reset;
logic cond_reg_out;
logic cond_reg_done;
logic [3:0] adder_left;
logic [3:0] adder_right;
logic [3:0] adder_out;
logic [3:0] lt_left;
logic [3:0] lt_right;
logic lt_out;
logic [3:0] idx0_in;
logic idx0_write_en;
logic idx0_clk;
logic idx0_reset;
logic [3:0] idx0_out;
logic idx0_done;
logic cond_reg0_in;
logic cond_reg0_write_en;
logic cond_reg0_clk;
logic cond_reg0_reset;
logic cond_reg0_out;
logic cond_reg0_done;
logic [3:0] adder0_left;
logic [3:0] adder0_right;
logic [3:0] adder0_out;
logic [3:0] lt0_left;
logic [3:0] lt0_right;
logic lt0_out;
logic [1:0] idx1_in;
logic idx1_write_en;
logic idx1_clk;
logic idx1_reset;
logic [1:0] idx1_out;
logic idx1_done;
logic cond_reg1_in;
logic cond_reg1_write_en;
logic cond_reg1_clk;
logic cond_reg1_reset;
logic cond_reg1_out;
logic cond_reg1_done;
logic [1:0] adder1_left;
logic [1:0] adder1_right;
logic [1:0] adder1_out;
logic [1:0] lt1_left;
logic [1:0] lt1_right;
logic lt1_out;
logic [8:0] idx2_in;
logic idx2_write_en;
logic idx2_clk;
logic idx2_reset;
logic [8:0] idx2_out;
logic idx2_done;
logic cond_reg2_in;
logic cond_reg2_write_en;
logic cond_reg2_clk;
logic cond_reg2_reset;
logic cond_reg2_out;
logic cond_reg2_done;
logic [8:0] adder2_left;
logic [8:0] adder2_right;
logic [8:0] adder2_out;
logic [8:0] lt2_left;
logic [8:0] lt2_right;
logic lt2_out;
logic [3:0] fsm_in;
logic fsm_write_en;
logic fsm_clk;
logic fsm_reset;
logic [3:0] fsm_out;
logic fsm_done;
logic [3:0] adder3_left;
logic [3:0] adder3_right;
logic [3:0] adder3_out;
logic ud_out;
logic signal_reg_in;
logic signal_reg_write_en;
logic signal_reg_clk;
logic signal_reg_reset;
logic signal_reg_out;
logic signal_reg_done;
logic [4:0] fsm0_in;
logic fsm0_write_en;
logic fsm0_clk;
logic fsm0_reset;
logic [4:0] fsm0_out;
logic fsm0_done;
logic bb0_4_go_in;
logic bb0_4_go_out;
logic bb0_4_done_in;
logic bb0_4_done_out;
logic bb0_5_go_in;
logic bb0_5_go_out;
logic bb0_5_done_in;
logic bb0_5_done_out;
logic bb0_6_go_in;
logic bb0_6_go_out;
logic bb0_6_done_in;
logic bb0_6_done_out;
logic bb0_7_go_in;
logic bb0_7_go_out;
logic bb0_7_done_in;
logic bb0_7_done_out;
logic bb0_8_go_in;
logic bb0_8_go_out;
logic bb0_8_done_in;
logic bb0_8_done_out;
logic bb0_9_go_in;
logic bb0_9_go_out;
logic bb0_9_done_in;
logic bb0_9_done_out;
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
logic invoke7_go_in;
logic invoke7_go_out;
logic invoke7_done_in;
logic invoke7_done_out;
logic invoke8_go_in;
logic invoke8_go_out;
logic invoke8_done_in;
logic invoke8_done_out;
logic invoke9_go_in;
logic invoke9_go_out;
logic invoke9_done_in;
logic invoke9_done_out;
logic invoke10_go_in;
logic invoke10_go_out;
logic invoke10_done_in;
logic invoke10_done_out;
logic invoke11_go_in;
logic invoke11_go_out;
logic invoke11_done_in;
logic invoke11_done_out;
logic invoke12_go_in;
logic invoke12_go_out;
logic invoke12_done_in;
logic invoke12_done_out;
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
logic init_repeat1_go_in;
logic init_repeat1_go_out;
logic init_repeat1_done_in;
logic init_repeat1_done_out;
logic incr_repeat1_go_in;
logic incr_repeat1_go_out;
logic incr_repeat1_done_in;
logic incr_repeat1_done_out;
logic init_repeat2_go_in;
logic init_repeat2_go_out;
logic init_repeat2_done_in;
logic init_repeat2_done_out;
logic incr_repeat2_go_in;
logic incr_repeat2_go_out;
logic incr_repeat2_done_in;
logic incr_repeat2_done_out;
logic early_reset_static_seq_go_in;
logic early_reset_static_seq_go_out;
logic early_reset_static_seq_done_in;
logic early_reset_static_seq_done_out;
logic wrapper_early_reset_static_seq_go_in;
logic wrapper_early_reset_static_seq_go_out;
logic wrapper_early_reset_static_seq_done_in;
logic wrapper_early_reset_static_seq_done_out;
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
) std_add_5 (
    .left(std_add_5_left),
    .out(std_add_5_out),
    .right(std_add_5_right)
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
) muli_1_reg (
    .clk(muli_1_reg_clk),
    .done(muli_1_reg_done),
    .in(muli_1_reg_in),
    .out(muli_1_reg_out),
    .reset(muli_1_reg_reset),
    .write_en(muli_1_reg_write_en)
);
std_mult_pipe # (
    .WIDTH(32)
) std_mult_pipe_1 (
    .clk(std_mult_pipe_1_clk),
    .done(std_mult_pipe_1_done),
    .go(std_mult_pipe_1_go),
    .left(std_mult_pipe_1_left),
    .out(std_mult_pipe_1_out),
    .reset(std_mult_pipe_1_reset),
    .right(std_mult_pipe_1_right)
);
std_reg # (
    .WIDTH(32)
) for_2_induction_var_reg (
    .clk(for_2_induction_var_reg_clk),
    .done(for_2_induction_var_reg_done),
    .in(for_2_induction_var_reg_in),
    .out(for_2_induction_var_reg_out),
    .reset(for_2_induction_var_reg_reset),
    .write_en(for_2_induction_var_reg_write_en)
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
    .WIDTH(4)
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
    .WIDTH(4)
) adder (
    .left(adder_left),
    .out(adder_out),
    .right(adder_right)
);
std_lt # (
    .WIDTH(4)
) lt (
    .left(lt_left),
    .out(lt_out),
    .right(lt_right)
);
std_reg # (
    .WIDTH(4)
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
    .WIDTH(4)
) adder0 (
    .left(adder0_left),
    .out(adder0_out),
    .right(adder0_right)
);
std_lt # (
    .WIDTH(4)
) lt0 (
    .left(lt0_left),
    .out(lt0_out),
    .right(lt0_right)
);
std_reg # (
    .WIDTH(2)
) idx1 (
    .clk(idx1_clk),
    .done(idx1_done),
    .in(idx1_in),
    .out(idx1_out),
    .reset(idx1_reset),
    .write_en(idx1_write_en)
);
std_reg # (
    .WIDTH(1)
) cond_reg1 (
    .clk(cond_reg1_clk),
    .done(cond_reg1_done),
    .in(cond_reg1_in),
    .out(cond_reg1_out),
    .reset(cond_reg1_reset),
    .write_en(cond_reg1_write_en)
);
std_add # (
    .WIDTH(2)
) adder1 (
    .left(adder1_left),
    .out(adder1_out),
    .right(adder1_right)
);
std_lt # (
    .WIDTH(2)
) lt1 (
    .left(lt1_left),
    .out(lt1_out),
    .right(lt1_right)
);
std_reg # (
    .WIDTH(9)
) idx2 (
    .clk(idx2_clk),
    .done(idx2_done),
    .in(idx2_in),
    .out(idx2_out),
    .reset(idx2_reset),
    .write_en(idx2_write_en)
);
std_reg # (
    .WIDTH(1)
) cond_reg2 (
    .clk(cond_reg2_clk),
    .done(cond_reg2_done),
    .in(cond_reg2_in),
    .out(cond_reg2_out),
    .reset(cond_reg2_reset),
    .write_en(cond_reg2_write_en)
);
std_add # (
    .WIDTH(9)
) adder2 (
    .left(adder2_left),
    .out(adder2_out),
    .right(adder2_right)
);
std_lt # (
    .WIDTH(9)
) lt2 (
    .left(lt2_left),
    .out(lt2_out),
    .right(lt2_right)
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
std_add # (
    .WIDTH(4)
) adder3 (
    .left(adder3_left),
    .out(adder3_out),
    .right(adder3_right)
);
undef # (
    .WIDTH(1)
) ud (
    .out(ud_out)
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
    .WIDTH(5)
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
) bb0_6_go (
    .in(bb0_6_go_in),
    .out(bb0_6_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_6_done (
    .in(bb0_6_done_in),
    .out(bb0_6_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_7_go (
    .in(bb0_7_go_in),
    .out(bb0_7_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_7_done (
    .in(bb0_7_done_in),
    .out(bb0_7_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_8_go (
    .in(bb0_8_go_in),
    .out(bb0_8_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_8_done (
    .in(bb0_8_done_in),
    .out(bb0_8_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_9_go (
    .in(bb0_9_go_in),
    .out(bb0_9_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_9_done (
    .in(bb0_9_done_in),
    .out(bb0_9_done_out)
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
) invoke7_go (
    .in(invoke7_go_in),
    .out(invoke7_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke7_done (
    .in(invoke7_done_in),
    .out(invoke7_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke8_go (
    .in(invoke8_go_in),
    .out(invoke8_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke8_done (
    .in(invoke8_done_in),
    .out(invoke8_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke9_go (
    .in(invoke9_go_in),
    .out(invoke9_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke9_done (
    .in(invoke9_done_in),
    .out(invoke9_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke10_go (
    .in(invoke10_go_in),
    .out(invoke10_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke10_done (
    .in(invoke10_done_in),
    .out(invoke10_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke11_go (
    .in(invoke11_go_in),
    .out(invoke11_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke11_done (
    .in(invoke11_done_in),
    .out(invoke11_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke12_go (
    .in(invoke12_go_in),
    .out(invoke12_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke12_done (
    .in(invoke12_done_in),
    .out(invoke12_done_out)
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
) init_repeat1_go (
    .in(init_repeat1_go_in),
    .out(init_repeat1_go_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat1_done (
    .in(init_repeat1_done_in),
    .out(init_repeat1_done_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat1_go (
    .in(incr_repeat1_go_in),
    .out(incr_repeat1_go_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat1_done (
    .in(incr_repeat1_done_in),
    .out(incr_repeat1_done_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat2_go (
    .in(init_repeat2_go_in),
    .out(init_repeat2_go_out)
);
std_wire # (
    .WIDTH(1)
) init_repeat2_done (
    .in(init_repeat2_done_in),
    .out(init_repeat2_done_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat2_go (
    .in(incr_repeat2_go_in),
    .out(incr_repeat2_go_out)
);
std_wire # (
    .WIDTH(1)
) incr_repeat2_done (
    .in(incr_repeat2_done_in),
    .out(incr_repeat2_done_out)
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
wire _guard1 = invoke7_go_out;
wire _guard2 = bb0_4_go_out;
wire _guard3 = bb0_5_go_out;
wire _guard4 = _guard2 | _guard3;
wire _guard5 = bb0_7_go_out;
wire _guard6 = _guard4 | _guard5;
wire _guard7 = invoke12_go_out;
wire _guard8 = fsm_out >= 4'd4;
wire _guard9 = fsm_out < 4'd7;
wire _guard10 = _guard8 & _guard9;
wire _guard11 = early_reset_static_seq_go_out;
wire _guard12 = _guard10 & _guard11;
wire _guard13 = _guard7 | _guard12;
wire _guard14 = invoke9_go_out;
wire _guard15 = invoke8_go_out;
wire _guard16 = bb0_4_go_out;
wire _guard17 = bb0_5_go_out;
wire _guard18 = _guard16 | _guard17;
wire _guard19 = bb0_7_go_out;
wire _guard20 = _guard18 | _guard19;
wire _guard21 = invoke7_go_out;
wire _guard22 = invoke8_go_out;
wire _guard23 = _guard21 | _guard22;
wire _guard24 = invoke9_go_out;
wire _guard25 = _guard23 | _guard24;
wire _guard26 = invoke12_go_out;
wire _guard27 = _guard25 | _guard26;
wire _guard28 = fsm_out >= 4'd4;
wire _guard29 = fsm_out < 4'd7;
wire _guard30 = _guard28 & _guard29;
wire _guard31 = early_reset_static_seq_go_out;
wire _guard32 = _guard30 & _guard31;
wire _guard33 = bb0_4_go_out;
wire _guard34 = bb0_5_go_out;
wire _guard35 = _guard33 | _guard34;
wire _guard36 = bb0_7_go_out;
wire _guard37 = _guard35 | _guard36;
wire _guard38 = bb0_8_go_out;
wire _guard39 = bb0_9_go_out;
wire _guard40 = _guard38 | _guard39;
wire _guard41 = init_repeat1_go_out;
wire _guard42 = incr_repeat1_go_out;
wire _guard43 = _guard41 | _guard42;
wire _guard44 = init_repeat1_go_out;
wire _guard45 = incr_repeat1_go_out;
wire _guard46 = incr_repeat1_go_out;
wire _guard47 = incr_repeat1_go_out;
wire _guard48 = invoke9_done_out;
wire _guard49 = ~_guard48;
wire _guard50 = fsm0_out == 5'd15;
wire _guard51 = _guard49 & _guard50;
wire _guard52 = tdcc_go_out;
wire _guard53 = _guard51 & _guard52;
wire _guard54 = init_repeat_done_out;
wire _guard55 = ~_guard54;
wire _guard56 = fsm0_out == 5'd5;
wire _guard57 = _guard55 & _guard56;
wire _guard58 = tdcc_go_out;
wire _guard59 = _guard57 & _guard58;
wire _guard60 = bb0_6_done_out;
wire _guard61 = ~_guard60;
wire _guard62 = fsm0_out == 5'd9;
wire _guard63 = _guard61 & _guard62;
wire _guard64 = tdcc_go_out;
wire _guard65 = _guard63 & _guard64;
wire _guard66 = tdcc_done_out;
wire _guard67 = bb0_4_go_out;
wire _guard68 = invoke10_go_out;
wire _guard69 = bb0_7_go_out;
wire _guard70 = invoke10_go_out;
wire _guard71 = bb0_7_go_out;
wire _guard72 = bb0_4_go_out;
wire _guard73 = bb0_8_go_out;
wire _guard74 = invoke10_go_out;
wire _guard75 = bb0_7_go_out;
wire _guard76 = invoke10_go_out;
wire _guard77 = bb0_7_go_out;
wire _guard78 = bb0_9_go_out;
wire _guard79 = bb0_9_go_out;
wire _guard80 = bb0_8_go_out;
wire _guard81 = invoke10_go_out;
wire _guard82 = bb0_9_go_out;
wire _guard83 = invoke10_go_out;
wire _guard84 = bb0_5_go_out;
wire _guard85 = bb0_5_go_out;
wire _guard86 = bb0_9_go_out;
wire _guard87 = incr_repeat_go_out;
wire _guard88 = incr_repeat_go_out;
wire _guard89 = fsm_out != 4'd7;
wire _guard90 = early_reset_static_seq_go_out;
wire _guard91 = _guard89 & _guard90;
wire _guard92 = fsm_out == 4'd7;
wire _guard93 = early_reset_static_seq_go_out;
wire _guard94 = _guard92 & _guard93;
wire _guard95 = _guard91 | _guard94;
wire _guard96 = fsm_out == 4'd7;
wire _guard97 = early_reset_static_seq_go_out;
wire _guard98 = _guard96 & _guard97;
wire _guard99 = fsm_out != 4'd7;
wire _guard100 = early_reset_static_seq_go_out;
wire _guard101 = _guard99 & _guard100;
wire _guard102 = incr_repeat1_done_out;
wire _guard103 = ~_guard102;
wire _guard104 = fsm0_out == 5'd16;
wire _guard105 = _guard103 & _guard104;
wire _guard106 = tdcc_go_out;
wire _guard107 = _guard105 & _guard106;
wire _guard108 = init_repeat1_go_out;
wire _guard109 = incr_repeat1_go_out;
wire _guard110 = _guard108 | _guard109;
wire _guard111 = incr_repeat1_go_out;
wire _guard112 = init_repeat1_go_out;
wire _guard113 = incr_repeat1_go_out;
wire _guard114 = incr_repeat1_go_out;
wire _guard115 = incr_repeat2_go_out;
wire _guard116 = incr_repeat2_go_out;
wire _guard117 = invoke2_done_out;
wire _guard118 = ~_guard117;
wire _guard119 = fsm0_out == 5'd4;
wire _guard120 = _guard118 & _guard119;
wire _guard121 = tdcc_go_out;
wire _guard122 = _guard120 & _guard121;
wire _guard123 = init_repeat0_go_out;
wire _guard124 = incr_repeat0_go_out;
wire _guard125 = _guard123 | _guard124;
wire _guard126 = init_repeat0_go_out;
wire _guard127 = incr_repeat0_go_out;
wire _guard128 = invoke2_go_out;
wire _guard129 = invoke7_go_out;
wire _guard130 = _guard128 | _guard129;
wire _guard131 = invoke7_go_out;
wire _guard132 = invoke2_go_out;
wire _guard133 = init_repeat1_done_out;
wire _guard134 = ~_guard133;
wire _guard135 = fsm0_out == 5'd1;
wire _guard136 = _guard134 & _guard135;
wire _guard137 = tdcc_go_out;
wire _guard138 = _guard136 & _guard137;
wire _guard139 = init_repeat0_go_out;
wire _guard140 = incr_repeat0_go_out;
wire _guard141 = _guard139 | _guard140;
wire _guard142 = init_repeat0_go_out;
wire _guard143 = incr_repeat0_go_out;
wire _guard144 = bb0_9_done_out;
wire _guard145 = ~_guard144;
wire _guard146 = fsm0_out == 5'd21;
wire _guard147 = _guard145 & _guard146;
wire _guard148 = tdcc_go_out;
wire _guard149 = _guard147 & _guard148;
wire _guard150 = invoke11_done_out;
wire _guard151 = ~_guard150;
wire _guard152 = fsm0_out == 5'd18;
wire _guard153 = _guard151 & _guard152;
wire _guard154 = tdcc_go_out;
wire _guard155 = _guard153 & _guard154;
wire _guard156 = fsm_out == 4'd7;
wire _guard157 = early_reset_static_seq_go_out;
wire _guard158 = _guard156 & _guard157;
wire _guard159 = fsm_out == 4'd7;
wire _guard160 = early_reset_static_seq_go_out;
wire _guard161 = _guard159 & _guard160;
wire _guard162 = invoke0_done_out;
wire _guard163 = ~_guard162;
wire _guard164 = fsm0_out == 5'd0;
wire _guard165 = _guard163 & _guard164;
wire _guard166 = tdcc_go_out;
wire _guard167 = _guard165 & _guard166;
wire _guard168 = incr_repeat_done_out;
wire _guard169 = ~_guard168;
wire _guard170 = fsm0_out == 5'd12;
wire _guard171 = _guard169 & _guard170;
wire _guard172 = tdcc_go_out;
wire _guard173 = _guard171 & _guard172;
wire _guard174 = cond_reg1_done;
wire _guard175 = idx1_done;
wire _guard176 = _guard174 & _guard175;
wire _guard177 = invoke12_done_out;
wire _guard178 = ~_guard177;
wire _guard179 = fsm0_out == 5'd22;
wire _guard180 = _guard178 & _guard179;
wire _guard181 = tdcc_go_out;
wire _guard182 = _guard180 & _guard181;
wire _guard183 = invoke11_go_out;
wire _guard184 = invoke12_go_out;
wire _guard185 = _guard183 | _guard184;
wire _guard186 = fsm_out == 4'd3;
wire _guard187 = early_reset_static_seq_go_out;
wire _guard188 = _guard186 & _guard187;
wire _guard189 = _guard185 | _guard188;
wire _guard190 = bb0_6_go_out;
wire _guard191 = invoke12_go_out;
wire _guard192 = invoke11_go_out;
wire _guard193 = bb0_6_go_out;
wire _guard194 = fsm_out == 4'd3;
wire _guard195 = early_reset_static_seq_go_out;
wire _guard196 = _guard194 & _guard195;
wire _guard197 = incr_repeat2_go_out;
wire _guard198 = incr_repeat2_go_out;
wire _guard199 = fsm0_out == 5'd24;
wire _guard200 = fsm0_out == 5'd0;
wire _guard201 = invoke0_done_out;
wire _guard202 = _guard200 & _guard201;
wire _guard203 = tdcc_go_out;
wire _guard204 = _guard202 & _guard203;
wire _guard205 = _guard199 | _guard204;
wire _guard206 = fsm0_out == 5'd1;
wire _guard207 = init_repeat1_done_out;
wire _guard208 = cond_reg1_out;
wire _guard209 = _guard207 & _guard208;
wire _guard210 = _guard206 & _guard209;
wire _guard211 = tdcc_go_out;
wire _guard212 = _guard210 & _guard211;
wire _guard213 = _guard205 | _guard212;
wire _guard214 = fsm0_out == 5'd16;
wire _guard215 = incr_repeat1_done_out;
wire _guard216 = cond_reg1_out;
wire _guard217 = _guard215 & _guard216;
wire _guard218 = _guard214 & _guard217;
wire _guard219 = tdcc_go_out;
wire _guard220 = _guard218 & _guard219;
wire _guard221 = _guard213 | _guard220;
wire _guard222 = fsm0_out == 5'd2;
wire _guard223 = invoke1_done_out;
wire _guard224 = _guard222 & _guard223;
wire _guard225 = tdcc_go_out;
wire _guard226 = _guard224 & _guard225;
wire _guard227 = _guard221 | _guard226;
wire _guard228 = fsm0_out == 5'd3;
wire _guard229 = init_repeat0_done_out;
wire _guard230 = cond_reg0_out;
wire _guard231 = _guard229 & _guard230;
wire _guard232 = _guard228 & _guard231;
wire _guard233 = tdcc_go_out;
wire _guard234 = _guard232 & _guard233;
wire _guard235 = _guard227 | _guard234;
wire _guard236 = fsm0_out == 5'd14;
wire _guard237 = incr_repeat0_done_out;
wire _guard238 = cond_reg0_out;
wire _guard239 = _guard237 & _guard238;
wire _guard240 = _guard236 & _guard239;
wire _guard241 = tdcc_go_out;
wire _guard242 = _guard240 & _guard241;
wire _guard243 = _guard235 | _guard242;
wire _guard244 = fsm0_out == 5'd4;
wire _guard245 = invoke2_done_out;
wire _guard246 = _guard244 & _guard245;
wire _guard247 = tdcc_go_out;
wire _guard248 = _guard246 & _guard247;
wire _guard249 = _guard243 | _guard248;
wire _guard250 = fsm0_out == 5'd5;
wire _guard251 = init_repeat_done_out;
wire _guard252 = cond_reg_out;
wire _guard253 = _guard251 & _guard252;
wire _guard254 = _guard250 & _guard253;
wire _guard255 = tdcc_go_out;
wire _guard256 = _guard254 & _guard255;
wire _guard257 = _guard249 | _guard256;
wire _guard258 = fsm0_out == 5'd12;
wire _guard259 = incr_repeat_done_out;
wire _guard260 = cond_reg_out;
wire _guard261 = _guard259 & _guard260;
wire _guard262 = _guard258 & _guard261;
wire _guard263 = tdcc_go_out;
wire _guard264 = _guard262 & _guard263;
wire _guard265 = _guard257 | _guard264;
wire _guard266 = fsm0_out == 5'd6;
wire _guard267 = wrapper_early_reset_static_seq_done_out;
wire _guard268 = _guard266 & _guard267;
wire _guard269 = tdcc_go_out;
wire _guard270 = _guard268 & _guard269;
wire _guard271 = _guard265 | _guard270;
wire _guard272 = fsm0_out == 5'd7;
wire _guard273 = bb0_4_done_out;
wire _guard274 = _guard272 & _guard273;
wire _guard275 = tdcc_go_out;
wire _guard276 = _guard274 & _guard275;
wire _guard277 = _guard271 | _guard276;
wire _guard278 = fsm0_out == 5'd8;
wire _guard279 = bb0_5_done_out;
wire _guard280 = _guard278 & _guard279;
wire _guard281 = tdcc_go_out;
wire _guard282 = _guard280 & _guard281;
wire _guard283 = _guard277 | _guard282;
wire _guard284 = fsm0_out == 5'd9;
wire _guard285 = bb0_6_done_out;
wire _guard286 = _guard284 & _guard285;
wire _guard287 = tdcc_go_out;
wire _guard288 = _guard286 & _guard287;
wire _guard289 = _guard283 | _guard288;
wire _guard290 = fsm0_out == 5'd10;
wire _guard291 = bb0_7_done_out;
wire _guard292 = _guard290 & _guard291;
wire _guard293 = tdcc_go_out;
wire _guard294 = _guard292 & _guard293;
wire _guard295 = _guard289 | _guard294;
wire _guard296 = fsm0_out == 5'd11;
wire _guard297 = invoke7_done_out;
wire _guard298 = _guard296 & _guard297;
wire _guard299 = tdcc_go_out;
wire _guard300 = _guard298 & _guard299;
wire _guard301 = _guard295 | _guard300;
wire _guard302 = fsm0_out == 5'd5;
wire _guard303 = init_repeat_done_out;
wire _guard304 = cond_reg_out;
wire _guard305 = ~_guard304;
wire _guard306 = _guard303 & _guard305;
wire _guard307 = _guard302 & _guard306;
wire _guard308 = tdcc_go_out;
wire _guard309 = _guard307 & _guard308;
wire _guard310 = _guard301 | _guard309;
wire _guard311 = fsm0_out == 5'd12;
wire _guard312 = incr_repeat_done_out;
wire _guard313 = cond_reg_out;
wire _guard314 = ~_guard313;
wire _guard315 = _guard312 & _guard314;
wire _guard316 = _guard311 & _guard315;
wire _guard317 = tdcc_go_out;
wire _guard318 = _guard316 & _guard317;
wire _guard319 = _guard310 | _guard318;
wire _guard320 = fsm0_out == 5'd13;
wire _guard321 = invoke8_done_out;
wire _guard322 = _guard320 & _guard321;
wire _guard323 = tdcc_go_out;
wire _guard324 = _guard322 & _guard323;
wire _guard325 = _guard319 | _guard324;
wire _guard326 = fsm0_out == 5'd3;
wire _guard327 = init_repeat0_done_out;
wire _guard328 = cond_reg0_out;
wire _guard329 = ~_guard328;
wire _guard330 = _guard327 & _guard329;
wire _guard331 = _guard326 & _guard330;
wire _guard332 = tdcc_go_out;
wire _guard333 = _guard331 & _guard332;
wire _guard334 = _guard325 | _guard333;
wire _guard335 = fsm0_out == 5'd14;
wire _guard336 = incr_repeat0_done_out;
wire _guard337 = cond_reg0_out;
wire _guard338 = ~_guard337;
wire _guard339 = _guard336 & _guard338;
wire _guard340 = _guard335 & _guard339;
wire _guard341 = tdcc_go_out;
wire _guard342 = _guard340 & _guard341;
wire _guard343 = _guard334 | _guard342;
wire _guard344 = fsm0_out == 5'd15;
wire _guard345 = invoke9_done_out;
wire _guard346 = _guard344 & _guard345;
wire _guard347 = tdcc_go_out;
wire _guard348 = _guard346 & _guard347;
wire _guard349 = _guard343 | _guard348;
wire _guard350 = fsm0_out == 5'd1;
wire _guard351 = init_repeat1_done_out;
wire _guard352 = cond_reg1_out;
wire _guard353 = ~_guard352;
wire _guard354 = _guard351 & _guard353;
wire _guard355 = _guard350 & _guard354;
wire _guard356 = tdcc_go_out;
wire _guard357 = _guard355 & _guard356;
wire _guard358 = _guard349 | _guard357;
wire _guard359 = fsm0_out == 5'd16;
wire _guard360 = incr_repeat1_done_out;
wire _guard361 = cond_reg1_out;
wire _guard362 = ~_guard361;
wire _guard363 = _guard360 & _guard362;
wire _guard364 = _guard359 & _guard363;
wire _guard365 = tdcc_go_out;
wire _guard366 = _guard364 & _guard365;
wire _guard367 = _guard358 | _guard366;
wire _guard368 = fsm0_out == 5'd17;
wire _guard369 = invoke10_done_out;
wire _guard370 = _guard368 & _guard369;
wire _guard371 = tdcc_go_out;
wire _guard372 = _guard370 & _guard371;
wire _guard373 = _guard367 | _guard372;
wire _guard374 = fsm0_out == 5'd18;
wire _guard375 = invoke11_done_out;
wire _guard376 = _guard374 & _guard375;
wire _guard377 = tdcc_go_out;
wire _guard378 = _guard376 & _guard377;
wire _guard379 = _guard373 | _guard378;
wire _guard380 = fsm0_out == 5'd19;
wire _guard381 = init_repeat2_done_out;
wire _guard382 = cond_reg2_out;
wire _guard383 = _guard381 & _guard382;
wire _guard384 = _guard380 & _guard383;
wire _guard385 = tdcc_go_out;
wire _guard386 = _guard384 & _guard385;
wire _guard387 = _guard379 | _guard386;
wire _guard388 = fsm0_out == 5'd23;
wire _guard389 = incr_repeat2_done_out;
wire _guard390 = cond_reg2_out;
wire _guard391 = _guard389 & _guard390;
wire _guard392 = _guard388 & _guard391;
wire _guard393 = tdcc_go_out;
wire _guard394 = _guard392 & _guard393;
wire _guard395 = _guard387 | _guard394;
wire _guard396 = fsm0_out == 5'd20;
wire _guard397 = bb0_8_done_out;
wire _guard398 = _guard396 & _guard397;
wire _guard399 = tdcc_go_out;
wire _guard400 = _guard398 & _guard399;
wire _guard401 = _guard395 | _guard400;
wire _guard402 = fsm0_out == 5'd21;
wire _guard403 = bb0_9_done_out;
wire _guard404 = _guard402 & _guard403;
wire _guard405 = tdcc_go_out;
wire _guard406 = _guard404 & _guard405;
wire _guard407 = _guard401 | _guard406;
wire _guard408 = fsm0_out == 5'd22;
wire _guard409 = invoke12_done_out;
wire _guard410 = _guard408 & _guard409;
wire _guard411 = tdcc_go_out;
wire _guard412 = _guard410 & _guard411;
wire _guard413 = _guard407 | _guard412;
wire _guard414 = fsm0_out == 5'd19;
wire _guard415 = init_repeat2_done_out;
wire _guard416 = cond_reg2_out;
wire _guard417 = ~_guard416;
wire _guard418 = _guard415 & _guard417;
wire _guard419 = _guard414 & _guard418;
wire _guard420 = tdcc_go_out;
wire _guard421 = _guard419 & _guard420;
wire _guard422 = _guard413 | _guard421;
wire _guard423 = fsm0_out == 5'd23;
wire _guard424 = incr_repeat2_done_out;
wire _guard425 = cond_reg2_out;
wire _guard426 = ~_guard425;
wire _guard427 = _guard424 & _guard426;
wire _guard428 = _guard423 & _guard427;
wire _guard429 = tdcc_go_out;
wire _guard430 = _guard428 & _guard429;
wire _guard431 = _guard422 | _guard430;
wire _guard432 = fsm0_out == 5'd0;
wire _guard433 = invoke0_done_out;
wire _guard434 = _guard432 & _guard433;
wire _guard435 = tdcc_go_out;
wire _guard436 = _guard434 & _guard435;
wire _guard437 = fsm0_out == 5'd3;
wire _guard438 = init_repeat0_done_out;
wire _guard439 = cond_reg0_out;
wire _guard440 = ~_guard439;
wire _guard441 = _guard438 & _guard440;
wire _guard442 = _guard437 & _guard441;
wire _guard443 = tdcc_go_out;
wire _guard444 = _guard442 & _guard443;
wire _guard445 = fsm0_out == 5'd14;
wire _guard446 = incr_repeat0_done_out;
wire _guard447 = cond_reg0_out;
wire _guard448 = ~_guard447;
wire _guard449 = _guard446 & _guard448;
wire _guard450 = _guard445 & _guard449;
wire _guard451 = tdcc_go_out;
wire _guard452 = _guard450 & _guard451;
wire _guard453 = _guard444 | _guard452;
wire _guard454 = fsm0_out == 5'd22;
wire _guard455 = invoke12_done_out;
wire _guard456 = _guard454 & _guard455;
wire _guard457 = tdcc_go_out;
wire _guard458 = _guard456 & _guard457;
wire _guard459 = fsm0_out == 5'd17;
wire _guard460 = invoke10_done_out;
wire _guard461 = _guard459 & _guard460;
wire _guard462 = tdcc_go_out;
wire _guard463 = _guard461 & _guard462;
wire _guard464 = fsm0_out == 5'd19;
wire _guard465 = init_repeat2_done_out;
wire _guard466 = cond_reg2_out;
wire _guard467 = ~_guard466;
wire _guard468 = _guard465 & _guard467;
wire _guard469 = _guard464 & _guard468;
wire _guard470 = tdcc_go_out;
wire _guard471 = _guard469 & _guard470;
wire _guard472 = fsm0_out == 5'd23;
wire _guard473 = incr_repeat2_done_out;
wire _guard474 = cond_reg2_out;
wire _guard475 = ~_guard474;
wire _guard476 = _guard473 & _guard475;
wire _guard477 = _guard472 & _guard476;
wire _guard478 = tdcc_go_out;
wire _guard479 = _guard477 & _guard478;
wire _guard480 = _guard471 | _guard479;
wire _guard481 = fsm0_out == 5'd15;
wire _guard482 = invoke9_done_out;
wire _guard483 = _guard481 & _guard482;
wire _guard484 = tdcc_go_out;
wire _guard485 = _guard483 & _guard484;
wire _guard486 = fsm0_out == 5'd24;
wire _guard487 = fsm0_out == 5'd2;
wire _guard488 = invoke1_done_out;
wire _guard489 = _guard487 & _guard488;
wire _guard490 = tdcc_go_out;
wire _guard491 = _guard489 & _guard490;
wire _guard492 = fsm0_out == 5'd5;
wire _guard493 = init_repeat_done_out;
wire _guard494 = cond_reg_out;
wire _guard495 = ~_guard494;
wire _guard496 = _guard493 & _guard495;
wire _guard497 = _guard492 & _guard496;
wire _guard498 = tdcc_go_out;
wire _guard499 = _guard497 & _guard498;
wire _guard500 = fsm0_out == 5'd12;
wire _guard501 = incr_repeat_done_out;
wire _guard502 = cond_reg_out;
wire _guard503 = ~_guard502;
wire _guard504 = _guard501 & _guard503;
wire _guard505 = _guard500 & _guard504;
wire _guard506 = tdcc_go_out;
wire _guard507 = _guard505 & _guard506;
wire _guard508 = _guard499 | _guard507;
wire _guard509 = fsm0_out == 5'd13;
wire _guard510 = invoke8_done_out;
wire _guard511 = _guard509 & _guard510;
wire _guard512 = tdcc_go_out;
wire _guard513 = _guard511 & _guard512;
wire _guard514 = fsm0_out == 5'd4;
wire _guard515 = invoke2_done_out;
wire _guard516 = _guard514 & _guard515;
wire _guard517 = tdcc_go_out;
wire _guard518 = _guard516 & _guard517;
wire _guard519 = fsm0_out == 5'd11;
wire _guard520 = invoke7_done_out;
wire _guard521 = _guard519 & _guard520;
wire _guard522 = tdcc_go_out;
wire _guard523 = _guard521 & _guard522;
wire _guard524 = fsm0_out == 5'd1;
wire _guard525 = init_repeat1_done_out;
wire _guard526 = cond_reg1_out;
wire _guard527 = _guard525 & _guard526;
wire _guard528 = _guard524 & _guard527;
wire _guard529 = tdcc_go_out;
wire _guard530 = _guard528 & _guard529;
wire _guard531 = fsm0_out == 5'd16;
wire _guard532 = incr_repeat1_done_out;
wire _guard533 = cond_reg1_out;
wire _guard534 = _guard532 & _guard533;
wire _guard535 = _guard531 & _guard534;
wire _guard536 = tdcc_go_out;
wire _guard537 = _guard535 & _guard536;
wire _guard538 = _guard530 | _guard537;
wire _guard539 = fsm0_out == 5'd7;
wire _guard540 = bb0_4_done_out;
wire _guard541 = _guard539 & _guard540;
wire _guard542 = tdcc_go_out;
wire _guard543 = _guard541 & _guard542;
wire _guard544 = fsm0_out == 5'd9;
wire _guard545 = bb0_6_done_out;
wire _guard546 = _guard544 & _guard545;
wire _guard547 = tdcc_go_out;
wire _guard548 = _guard546 & _guard547;
wire _guard549 = fsm0_out == 5'd6;
wire _guard550 = wrapper_early_reset_static_seq_done_out;
wire _guard551 = _guard549 & _guard550;
wire _guard552 = tdcc_go_out;
wire _guard553 = _guard551 & _guard552;
wire _guard554 = fsm0_out == 5'd10;
wire _guard555 = bb0_7_done_out;
wire _guard556 = _guard554 & _guard555;
wire _guard557 = tdcc_go_out;
wire _guard558 = _guard556 & _guard557;
wire _guard559 = fsm0_out == 5'd20;
wire _guard560 = bb0_8_done_out;
wire _guard561 = _guard559 & _guard560;
wire _guard562 = tdcc_go_out;
wire _guard563 = _guard561 & _guard562;
wire _guard564 = fsm0_out == 5'd18;
wire _guard565 = invoke11_done_out;
wire _guard566 = _guard564 & _guard565;
wire _guard567 = tdcc_go_out;
wire _guard568 = _guard566 & _guard567;
wire _guard569 = fsm0_out == 5'd21;
wire _guard570 = bb0_9_done_out;
wire _guard571 = _guard569 & _guard570;
wire _guard572 = tdcc_go_out;
wire _guard573 = _guard571 & _guard572;
wire _guard574 = fsm0_out == 5'd3;
wire _guard575 = init_repeat0_done_out;
wire _guard576 = cond_reg0_out;
wire _guard577 = _guard575 & _guard576;
wire _guard578 = _guard574 & _guard577;
wire _guard579 = tdcc_go_out;
wire _guard580 = _guard578 & _guard579;
wire _guard581 = fsm0_out == 5'd14;
wire _guard582 = incr_repeat0_done_out;
wire _guard583 = cond_reg0_out;
wire _guard584 = _guard582 & _guard583;
wire _guard585 = _guard581 & _guard584;
wire _guard586 = tdcc_go_out;
wire _guard587 = _guard585 & _guard586;
wire _guard588 = _guard580 | _guard587;
wire _guard589 = fsm0_out == 5'd5;
wire _guard590 = init_repeat_done_out;
wire _guard591 = cond_reg_out;
wire _guard592 = _guard590 & _guard591;
wire _guard593 = _guard589 & _guard592;
wire _guard594 = tdcc_go_out;
wire _guard595 = _guard593 & _guard594;
wire _guard596 = fsm0_out == 5'd12;
wire _guard597 = incr_repeat_done_out;
wire _guard598 = cond_reg_out;
wire _guard599 = _guard597 & _guard598;
wire _guard600 = _guard596 & _guard599;
wire _guard601 = tdcc_go_out;
wire _guard602 = _guard600 & _guard601;
wire _guard603 = _guard595 | _guard602;
wire _guard604 = fsm0_out == 5'd19;
wire _guard605 = init_repeat2_done_out;
wire _guard606 = cond_reg2_out;
wire _guard607 = _guard605 & _guard606;
wire _guard608 = _guard604 & _guard607;
wire _guard609 = tdcc_go_out;
wire _guard610 = _guard608 & _guard609;
wire _guard611 = fsm0_out == 5'd23;
wire _guard612 = incr_repeat2_done_out;
wire _guard613 = cond_reg2_out;
wire _guard614 = _guard612 & _guard613;
wire _guard615 = _guard611 & _guard614;
wire _guard616 = tdcc_go_out;
wire _guard617 = _guard615 & _guard616;
wire _guard618 = _guard610 | _guard617;
wire _guard619 = fsm0_out == 5'd1;
wire _guard620 = init_repeat1_done_out;
wire _guard621 = cond_reg1_out;
wire _guard622 = ~_guard621;
wire _guard623 = _guard620 & _guard622;
wire _guard624 = _guard619 & _guard623;
wire _guard625 = tdcc_go_out;
wire _guard626 = _guard624 & _guard625;
wire _guard627 = fsm0_out == 5'd16;
wire _guard628 = incr_repeat1_done_out;
wire _guard629 = cond_reg1_out;
wire _guard630 = ~_guard629;
wire _guard631 = _guard628 & _guard630;
wire _guard632 = _guard627 & _guard631;
wire _guard633 = tdcc_go_out;
wire _guard634 = _guard632 & _guard633;
wire _guard635 = _guard626 | _guard634;
wire _guard636 = fsm0_out == 5'd8;
wire _guard637 = bb0_5_done_out;
wire _guard638 = _guard636 & _guard637;
wire _guard639 = tdcc_go_out;
wire _guard640 = _guard638 & _guard639;
wire _guard641 = cond_reg0_done;
wire _guard642 = idx0_done;
wire _guard643 = _guard641 & _guard642;
wire _guard644 = incr_repeat2_done_out;
wire _guard645 = ~_guard644;
wire _guard646 = fsm0_out == 5'd23;
wire _guard647 = _guard645 & _guard646;
wire _guard648 = tdcc_go_out;
wire _guard649 = _guard647 & _guard648;
wire _guard650 = invoke8_done_out;
wire _guard651 = ~_guard650;
wire _guard652 = fsm0_out == 5'd13;
wire _guard653 = _guard651 & _guard652;
wire _guard654 = tdcc_go_out;
wire _guard655 = _guard653 & _guard654;
wire _guard656 = bb0_6_go_out;
wire _guard657 = bb0_6_go_out;
wire _guard658 = std_addFN_0_done;
wire _guard659 = ~_guard658;
wire _guard660 = bb0_6_go_out;
wire _guard661 = _guard659 & _guard660;
wire _guard662 = bb0_6_go_out;
wire _guard663 = invoke10_go_out;
wire _guard664 = invoke10_go_out;
wire _guard665 = invoke10_go_out;
wire _guard666 = invoke10_go_out;
wire _guard667 = init_repeat_go_out;
wire _guard668 = incr_repeat_go_out;
wire _guard669 = _guard667 | _guard668;
wire _guard670 = incr_repeat_go_out;
wire _guard671 = init_repeat_go_out;
wire _guard672 = early_reset_static_seq_go_out;
wire _guard673 = early_reset_static_seq_go_out;
wire _guard674 = cond_reg_done;
wire _guard675 = idx_done;
wire _guard676 = _guard674 & _guard675;
wire _guard677 = cond_reg_done;
wire _guard678 = idx_done;
wire _guard679 = _guard677 & _guard678;
wire _guard680 = cond_reg0_done;
wire _guard681 = idx0_done;
wire _guard682 = _guard680 & _guard681;
wire _guard683 = signal_reg_out;
wire _guard684 = incr_repeat0_go_out;
wire _guard685 = incr_repeat0_go_out;
wire _guard686 = invoke1_done_out;
wire _guard687 = ~_guard686;
wire _guard688 = fsm0_out == 5'd2;
wire _guard689 = _guard687 & _guard688;
wire _guard690 = tdcc_go_out;
wire _guard691 = _guard689 & _guard690;
wire _guard692 = init_repeat0_done_out;
wire _guard693 = ~_guard692;
wire _guard694 = fsm0_out == 5'd3;
wire _guard695 = _guard693 & _guard694;
wire _guard696 = tdcc_go_out;
wire _guard697 = _guard695 & _guard696;
wire _guard698 = incr_repeat0_done_out;
wire _guard699 = ~_guard698;
wire _guard700 = fsm0_out == 5'd14;
wire _guard701 = _guard699 & _guard700;
wire _guard702 = tdcc_go_out;
wire _guard703 = _guard701 & _guard702;
wire _guard704 = init_repeat2_done_out;
wire _guard705 = ~_guard704;
wire _guard706 = fsm0_out == 5'd19;
wire _guard707 = _guard705 & _guard706;
wire _guard708 = tdcc_go_out;
wire _guard709 = _guard707 & _guard708;
wire _guard710 = wrapper_early_reset_static_seq_go_out;
wire _guard711 = signal_reg_out;
wire _guard712 = fsm_out == 4'd7;
wire _guard713 = _guard712 & _guard0;
wire _guard714 = signal_reg_out;
wire _guard715 = ~_guard714;
wire _guard716 = _guard713 & _guard715;
wire _guard717 = wrapper_early_reset_static_seq_go_out;
wire _guard718 = _guard716 & _guard717;
wire _guard719 = _guard711 | _guard718;
wire _guard720 = fsm_out == 4'd7;
wire _guard721 = _guard720 & _guard0;
wire _guard722 = signal_reg_out;
wire _guard723 = ~_guard722;
wire _guard724 = _guard721 & _guard723;
wire _guard725 = wrapper_early_reset_static_seq_go_out;
wire _guard726 = _guard724 & _guard725;
wire _guard727 = signal_reg_out;
wire _guard728 = cond_reg1_done;
wire _guard729 = idx1_done;
wire _guard730 = _guard728 & _guard729;
wire _guard731 = cond_reg2_done;
wire _guard732 = idx2_done;
wire _guard733 = _guard731 & _guard732;
wire _guard734 = invoke0_go_out;
wire _guard735 = invoke9_go_out;
wire _guard736 = _guard734 | _guard735;
wire _guard737 = invoke9_go_out;
wire _guard738 = invoke0_go_out;
wire _guard739 = fsm_out >= 4'd4;
wire _guard740 = fsm_out < 4'd7;
wire _guard741 = _guard739 & _guard740;
wire _guard742 = early_reset_static_seq_go_out;
wire _guard743 = _guard741 & _guard742;
wire _guard744 = fsm_out < 4'd3;
wire _guard745 = early_reset_static_seq_go_out;
wire _guard746 = _guard744 & _guard745;
wire _guard747 = fsm_out < 4'd3;
wire _guard748 = fsm_out >= 4'd4;
wire _guard749 = fsm_out < 4'd7;
wire _guard750 = _guard748 & _guard749;
wire _guard751 = _guard747 | _guard750;
wire _guard752 = early_reset_static_seq_go_out;
wire _guard753 = _guard751 & _guard752;
wire _guard754 = fsm_out < 4'd3;
wire _guard755 = fsm_out >= 4'd4;
wire _guard756 = fsm_out < 4'd7;
wire _guard757 = _guard755 & _guard756;
wire _guard758 = _guard754 | _guard757;
wire _guard759 = early_reset_static_seq_go_out;
wire _guard760 = _guard758 & _guard759;
wire _guard761 = fsm0_out == 5'd24;
wire _guard762 = invoke1_go_out;
wire _guard763 = invoke8_go_out;
wire _guard764 = _guard762 | _guard763;
wire _guard765 = invoke8_go_out;
wire _guard766 = invoke1_go_out;
wire _guard767 = incr_repeat_go_out;
wire _guard768 = incr_repeat_go_out;
wire _guard769 = init_repeat2_go_out;
wire _guard770 = incr_repeat2_go_out;
wire _guard771 = _guard769 | _guard770;
wire _guard772 = init_repeat2_go_out;
wire _guard773 = incr_repeat2_go_out;
wire _guard774 = bb0_4_done_out;
wire _guard775 = ~_guard774;
wire _guard776 = fsm0_out == 5'd7;
wire _guard777 = _guard775 & _guard776;
wire _guard778 = tdcc_go_out;
wire _guard779 = _guard777 & _guard778;
wire _guard780 = bb0_7_done_out;
wire _guard781 = ~_guard780;
wire _guard782 = fsm0_out == 5'd10;
wire _guard783 = _guard781 & _guard782;
wire _guard784 = tdcc_go_out;
wire _guard785 = _guard783 & _guard784;
wire _guard786 = wrapper_early_reset_static_seq_done_out;
wire _guard787 = ~_guard786;
wire _guard788 = fsm0_out == 5'd6;
wire _guard789 = _guard787 & _guard788;
wire _guard790 = tdcc_go_out;
wire _guard791 = _guard789 & _guard790;
wire _guard792 = init_repeat_go_out;
wire _guard793 = incr_repeat_go_out;
wire _guard794 = _guard792 | _guard793;
wire _guard795 = init_repeat_go_out;
wire _guard796 = incr_repeat_go_out;
wire _guard797 = incr_repeat0_go_out;
wire _guard798 = incr_repeat0_go_out;
wire _guard799 = init_repeat2_go_out;
wire _guard800 = incr_repeat2_go_out;
wire _guard801 = _guard799 | _guard800;
wire _guard802 = incr_repeat2_go_out;
wire _guard803 = init_repeat2_go_out;
wire _guard804 = bb0_5_done_out;
wire _guard805 = ~_guard804;
wire _guard806 = fsm0_out == 5'd8;
wire _guard807 = _guard805 & _guard806;
wire _guard808 = tdcc_go_out;
wire _guard809 = _guard807 & _guard808;
wire _guard810 = invoke10_done_out;
wire _guard811 = ~_guard810;
wire _guard812 = fsm0_out == 5'd17;
wire _guard813 = _guard811 & _guard812;
wire _guard814 = tdcc_go_out;
wire _guard815 = _guard813 & _guard814;
wire _guard816 = cond_reg2_done;
wire _guard817 = idx2_done;
wire _guard818 = _guard816 & _guard817;
wire _guard819 = bb0_8_done_out;
wire _guard820 = ~_guard819;
wire _guard821 = fsm0_out == 5'd20;
wire _guard822 = _guard820 & _guard821;
wire _guard823 = tdcc_go_out;
wire _guard824 = _guard822 & _guard823;
wire _guard825 = invoke7_done_out;
wire _guard826 = ~_guard825;
wire _guard827 = fsm0_out == 5'd11;
wire _guard828 = _guard826 & _guard827;
wire _guard829 = tdcc_go_out;
wire _guard830 = _guard828 & _guard829;
assign std_add_5_left =
  _guard1 ? for_0_induction_var_reg_out :
  _guard6 ? muli_1_reg_out :
  _guard13 ? addf_0_reg_out :
  _guard14 ? for_2_induction_var_reg_out :
  _guard15 ? for_1_induction_var_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard15, _guard14, _guard13, _guard6, _guard1})) begin
    $fatal(2, "Multiple assignment to port `std_add_5.left'.");
end
end
assign std_add_5_right =
  _guard20 ? for_0_induction_var_reg_out :
  _guard27 ? 32'd1 :
  _guard32 ? for_1_induction_var_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard32, _guard27, _guard20})) begin
    $fatal(2, "Multiple assignment to port `std_add_5.right'.");
end
end
assign std_slice_4_in =
  _guard37 ? std_add_5_out :
  _guard40 ? addf_0_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard40, _guard37})) begin
    $fatal(2, "Multiple assignment to port `std_slice_4.in'.");
end
end
assign cond_reg1_write_en = _guard43;
assign cond_reg1_clk = clk;
assign cond_reg1_reset = reset;
assign cond_reg1_in =
  _guard44 ? 1'd1 :
  _guard45 ? lt1_out :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard45, _guard44})) begin
    $fatal(2, "Multiple assignment to port `cond_reg1.in'.");
end
end
assign adder1_left =
  _guard46 ? idx1_out :
  2'd0;
assign adder1_right =
  _guard47 ? 2'd1 :
  2'd0;
assign invoke9_go_in = _guard53;
assign invoke9_done_in = for_2_induction_var_reg_done;
assign init_repeat_go_in = _guard59;
assign bb0_6_go_in = _guard65;
assign invoke7_done_in = for_0_induction_var_reg_done;
assign done = _guard66;
assign arg_mem_0_content_en = _guard67;
assign arg_mem_4_write_data = relu4d_0_instance_arg_mem_1_write_data;
assign arg_mem_3_addr0 =
  _guard69 ? std_slice_4_out :
  _guard70 ? relu4d_0_instance_arg_mem_0_addr0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard70, _guard69})) begin
    $fatal(2, "Multiple assignment to port `_this.arg_mem_3_addr0'.");
end
end
assign arg_mem_3_write_data = addf_0_reg_out;
assign arg_mem_0_addr0 = std_slice_4_out;
assign arg_mem_4_addr0 =
  _guard73 ? std_slice_4_out :
  _guard74 ? relu4d_0_instance_arg_mem_1_addr0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard74, _guard73})) begin
    $fatal(2, "Multiple assignment to port `_this.arg_mem_4_addr0'.");
end
end
assign arg_mem_3_content_en =
  _guard75 ? 1'd1 :
  _guard76 ? relu4d_0_instance_arg_mem_0_content_en :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard76, _guard75})) begin
    $fatal(2, "Multiple assignment to port `_this.arg_mem_3_content_en'.");
end
end
assign arg_mem_3_write_en = _guard77;
assign arg_mem_2_addr0 = std_slice_4_out;
assign arg_mem_2_content_en = _guard79;
assign arg_mem_4_content_en =
  _guard80 ? 1'd1 :
  _guard81 ? relu4d_0_instance_arg_mem_1_content_en :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard81, _guard80})) begin
    $fatal(2, "Multiple assignment to port `_this.arg_mem_4_content_en'.");
end
end
assign arg_mem_2_write_en = _guard82;
assign arg_mem_4_write_en =
  _guard83 ? relu4d_0_instance_arg_mem_1_write_en :
  1'd0;
assign arg_mem_1_addr0 = std_slice_4_out;
assign arg_mem_1_content_en = _guard85;
assign arg_mem_2_write_data = arg_mem_4_read_data;
assign adder_left =
  _guard87 ? idx_out :
  4'd0;
assign adder_right =
  _guard88 ? 4'd1 :
  4'd0;
assign fsm_write_en = _guard95;
assign fsm_clk = clk;
assign fsm_reset = reset;
assign fsm_in =
  _guard98 ? 4'd0 :
  _guard101 ? adder3_out :
  4'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard101, _guard98})) begin
    $fatal(2, "Multiple assignment to port `fsm.in'.");
end
end
assign invoke11_done_in = addf_0_reg_done;
assign incr_repeat1_go_in = _guard107;
assign idx1_write_en = _guard110;
assign idx1_clk = clk;
assign idx1_reset = reset;
assign idx1_in =
  _guard111 ? adder1_out :
  _guard112 ? 2'd0 :
  2'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard112, _guard111})) begin
    $fatal(2, "Multiple assignment to port `idx1.in'.");
end
end
assign lt1_left =
  _guard113 ? adder1_out :
  2'd0;
assign lt1_right =
  _guard114 ? 2'd3 :
  2'd0;
assign lt2_left =
  _guard115 ? adder2_out :
  9'd0;
assign lt2_right =
  _guard116 ? 9'd300 :
  9'd0;
assign invoke2_go_in = _guard122;
assign cond_reg0_write_en = _guard125;
assign cond_reg0_clk = clk;
assign cond_reg0_reset = reset;
assign cond_reg0_in =
  _guard126 ? 1'd1 :
  _guard127 ? lt0_out :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard127, _guard126})) begin
    $fatal(2, "Multiple assignment to port `cond_reg0.in'.");
end
end
assign for_0_induction_var_reg_write_en = _guard130;
assign for_0_induction_var_reg_clk = clk;
assign for_0_induction_var_reg_reset = reset;
assign for_0_induction_var_reg_in =
  _guard131 ? std_add_5_out :
  _guard132 ? 32'd0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard132, _guard131})) begin
    $fatal(2, "Multiple assignment to port `for_0_induction_var_reg.in'.");
end
end
assign init_repeat1_go_in = _guard138;
assign idx0_write_en = _guard141;
assign idx0_clk = clk;
assign idx0_reset = reset;
assign idx0_in =
  _guard142 ? 4'd0 :
  _guard143 ? adder0_out :
  4'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard143, _guard142})) begin
    $fatal(2, "Multiple assignment to port `idx0.in'.");
end
end
assign bb0_9_go_in = _guard149;
assign invoke11_go_in = _guard155;
assign muli_1_reg_write_en = _guard158;
assign muli_1_reg_clk = clk;
assign muli_1_reg_reset = reset;
assign muli_1_reg_in = std_mult_pipe_1_out;
assign invoke0_go_in = _guard167;
assign incr_repeat_go_in = _guard173;
assign init_repeat1_done_in = _guard176;
assign tdcc_go_in = go;
assign invoke12_go_in = _guard182;
assign addf_0_reg_write_en =
  _guard189 ? 1'd1 :
  _guard190 ? std_addFN_0_done :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard190, _guard189})) begin
    $fatal(2, "Multiple assignment to port `addf_0_reg.write_en'.");
end
end
assign addf_0_reg_clk = clk;
assign addf_0_reg_reset = reset;
assign addf_0_reg_in =
  _guard191 ? std_add_5_out :
  _guard192 ? 32'd0 :
  _guard193 ? std_addFN_0_out :
  _guard196 ? std_mult_pipe_1_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard196, _guard193, _guard192, _guard191})) begin
    $fatal(2, "Multiple assignment to port `addf_0_reg.in'.");
end
end
assign adder2_left =
  _guard197 ? idx2_out :
  9'd0;
assign adder2_right =
  _guard198 ? 9'd1 :
  9'd0;
assign fsm0_write_en = _guard431;
assign fsm0_clk = clk;
assign fsm0_reset = reset;
assign fsm0_in =
  _guard436 ? 5'd1 :
  _guard453 ? 5'd15 :
  _guard458 ? 5'd23 :
  _guard463 ? 5'd18 :
  _guard480 ? 5'd24 :
  _guard485 ? 5'd16 :
  _guard486 ? 5'd0 :
  _guard491 ? 5'd3 :
  _guard508 ? 5'd13 :
  _guard513 ? 5'd14 :
  _guard518 ? 5'd5 :
  _guard523 ? 5'd12 :
  _guard538 ? 5'd2 :
  _guard543 ? 5'd8 :
  _guard548 ? 5'd10 :
  _guard553 ? 5'd7 :
  _guard558 ? 5'd11 :
  _guard563 ? 5'd21 :
  _guard568 ? 5'd19 :
  _guard573 ? 5'd22 :
  _guard588 ? 5'd4 :
  _guard603 ? 5'd6 :
  _guard618 ? 5'd20 :
  _guard635 ? 5'd17 :
  _guard640 ? 5'd9 :
  5'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard640, _guard635, _guard618, _guard603, _guard588, _guard573, _guard568, _guard563, _guard558, _guard553, _guard548, _guard543, _guard538, _guard523, _guard518, _guard513, _guard508, _guard491, _guard486, _guard485, _guard480, _guard463, _guard458, _guard453, _guard436})) begin
    $fatal(2, "Multiple assignment to port `fsm0.in'.");
end
end
assign invoke10_done_in = relu4d_0_instance_done;
assign incr_repeat0_done_in = _guard643;
assign incr_repeat2_go_in = _guard649;
assign invoke8_go_in = _guard655;
assign std_addFN_0_roundingMode = 3'd0;
assign std_addFN_0_control = 1'd0;
assign std_addFN_0_clk = clk;
assign std_addFN_0_left =
  _guard656 ? arg_mem_0_read_data :
  32'd0;
assign std_addFN_0_subOp =
  _guard657 ? 1'd0 :
  1'd0;
assign std_addFN_0_reset = reset;
assign std_addFN_0_go = _guard661;
assign std_addFN_0_right =
  _guard662 ? arg_mem_1_read_data :
  32'd0;
assign relu4d_0_instance_arg_mem_0_read_data =
  _guard663 ? arg_mem_3_read_data :
  32'd0;
assign relu4d_0_instance_arg_mem_0_done =
  _guard664 ? arg_mem_3_done :
  1'd0;
assign relu4d_0_instance_clk = clk;
assign relu4d_0_instance_reset = reset;
assign relu4d_0_instance_go = _guard665;
assign relu4d_0_instance_arg_mem_1_done =
  _guard666 ? arg_mem_4_done :
  1'd0;
assign idx_write_en = _guard669;
assign idx_clk = clk;
assign idx_reset = reset;
assign idx_in =
  _guard670 ? adder_out :
  _guard671 ? 4'd0 :
  4'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard671, _guard670})) begin
    $fatal(2, "Multiple assignment to port `idx.in'.");
end
end
assign adder3_left =
  _guard672 ? fsm_out :
  4'd0;
assign adder3_right =
  _guard673 ? 4'd1 :
  4'd0;
assign bb0_9_done_in = arg_mem_2_done;
assign init_repeat_done_in = _guard676;
assign incr_repeat_done_in = _guard679;
assign init_repeat0_done_in = _guard682;
assign wrapper_early_reset_static_seq_done_in = _guard683;
assign invoke8_done_in = for_1_induction_var_reg_done;
assign invoke12_done_in = addf_0_reg_done;
assign adder0_left =
  _guard684 ? idx0_out :
  4'd0;
assign adder0_right =
  _guard685 ? 4'd1 :
  4'd0;
assign invoke0_done_in = for_2_induction_var_reg_done;
assign invoke1_go_in = _guard691;
assign init_repeat0_go_in = _guard697;
assign incr_repeat0_go_in = _guard703;
assign init_repeat2_go_in = _guard709;
assign early_reset_static_seq_go_in = _guard710;
assign bb0_7_done_in = arg_mem_3_done;
assign signal_reg_write_en = _guard719;
assign signal_reg_clk = clk;
assign signal_reg_reset = reset;
assign signal_reg_in =
  _guard726 ? 1'd1 :
  _guard727 ? 1'd0 :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard727, _guard726})) begin
    $fatal(2, "Multiple assignment to port `signal_reg.in'.");
end
end
assign bb0_5_done_in = arg_mem_1_done;
assign invoke2_done_in = for_0_induction_var_reg_done;
assign incr_repeat1_done_in = _guard730;
assign init_repeat2_done_in = _guard733;
assign bb0_6_done_in = addf_0_reg_done;
assign bb0_8_done_in = arg_mem_4_done;
assign for_2_induction_var_reg_write_en = _guard736;
assign for_2_induction_var_reg_clk = clk;
assign for_2_induction_var_reg_reset = reset;
assign for_2_induction_var_reg_in =
  _guard737 ? std_add_5_out :
  _guard738 ? 32'd0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard738, _guard737})) begin
    $fatal(2, "Multiple assignment to port `for_2_induction_var_reg.in'.");
end
end
assign std_mult_pipe_1_clk = clk;
assign std_mult_pipe_1_left =
  _guard743 ? std_add_5_out :
  _guard746 ? for_2_induction_var_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard746, _guard743})) begin
    $fatal(2, "Multiple assignment to port `std_mult_pipe_1.left'.");
end
end
assign std_mult_pipe_1_reset = reset;
assign std_mult_pipe_1_go = _guard753;
assign std_mult_pipe_1_right = 32'd10;
assign early_reset_static_seq_done_in = ud_out;
assign tdcc_done_in = _guard761;
assign for_1_induction_var_reg_write_en = _guard764;
assign for_1_induction_var_reg_clk = clk;
assign for_1_induction_var_reg_reset = reset;
assign for_1_induction_var_reg_in =
  _guard765 ? std_add_5_out :
  _guard766 ? 32'd0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard766, _guard765})) begin
    $fatal(2, "Multiple assignment to port `for_1_induction_var_reg.in'.");
end
end
assign lt_left =
  _guard767 ? adder_out :
  4'd0;
assign lt_right =
  _guard768 ? 4'd10 :
  4'd0;
assign cond_reg2_write_en = _guard771;
assign cond_reg2_clk = clk;
assign cond_reg2_reset = reset;
assign cond_reg2_in =
  _guard772 ? 1'd1 :
  _guard773 ? lt2_out :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard773, _guard772})) begin
    $fatal(2, "Multiple assignment to port `cond_reg2.in'.");
end
end
assign bb0_4_go_in = _guard779;
assign bb0_4_done_in = arg_mem_0_done;
assign bb0_7_go_in = _guard785;
assign invoke1_done_in = for_1_induction_var_reg_done;
assign wrapper_early_reset_static_seq_go_in = _guard791;
assign cond_reg_write_en = _guard794;
assign cond_reg_clk = clk;
assign cond_reg_reset = reset;
assign cond_reg_in =
  _guard795 ? 1'd1 :
  _guard796 ? lt_out :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard796, _guard795})) begin
    $fatal(2, "Multiple assignment to port `cond_reg.in'.");
end
end
assign lt0_left =
  _guard797 ? adder0_out :
  4'd0;
assign lt0_right =
  _guard798 ? 4'd10 :
  4'd0;
assign idx2_write_en = _guard801;
assign idx2_clk = clk;
assign idx2_reset = reset;
assign idx2_in =
  _guard802 ? adder2_out :
  _guard803 ? 9'd0 :
  9'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard803, _guard802})) begin
    $fatal(2, "Multiple assignment to port `idx2.in'.");
end
end
assign bb0_5_go_in = _guard809;
assign invoke10_go_in = _guard815;
assign incr_repeat2_done_in = _guard818;
assign bb0_8_go_in = _guard824;
assign invoke7_go_in = _guard830;
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
    isSigNaNRecFN#(parameter expWidth = 3, parameter sigWidth = 3) (
        input [(expWidth + sigWidth):0] in, output isSigNaN
    );

    wire isNaN =
        (in[(expWidth + sigWidth - 1):(expWidth + sigWidth - 3)] == 'b111);
    assign isSigNaN = isNaN && !in[sigWidth - 2];

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

