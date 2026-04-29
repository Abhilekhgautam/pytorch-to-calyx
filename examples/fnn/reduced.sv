// Compiled by morty-0.9.0 / 2026-04-30 1:20:49.223929601 +05:45:00

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
logic [31:0] std_slt_3_left;
logic [31:0] std_slt_3_right;
logic std_slt_3_out;
logic [31:0] std_add_6_left;
logic [31:0] std_add_6_right;
logic [31:0] std_add_6_out;
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
logic [31:0] std_add_0_left;
logic [31:0] std_add_0_right;
logic [31:0] std_add_0_out;
logic mem_0_clk;
logic mem_0_reset;
logic [8:0] mem_0_addr0;
logic mem_0_content_en;
logic mem_0_write_en;
logic [31:0] mem_0_write_data;
logic [31:0] mem_0_read_data;
logic mem_0_done;
logic [31:0] while_2_arg2_reg_in;
logic while_2_arg2_reg_write_en;
logic while_2_arg2_reg_clk;
logic while_2_arg2_reg_reset;
logic [31:0] while_2_arg2_reg_out;
logic while_2_arg2_reg_done;
logic [31:0] while_2_arg1_reg_in;
logic while_2_arg1_reg_write_en;
logic while_2_arg1_reg_clk;
logic while_2_arg1_reg_reset;
logic [31:0] while_2_arg1_reg_out;
logic while_2_arg1_reg_done;
logic [31:0] while_2_arg0_reg_in;
logic while_2_arg0_reg_write_en;
logic while_2_arg0_reg_clk;
logic while_2_arg0_reg_reset;
logic [31:0] while_2_arg0_reg_out;
logic while_2_arg0_reg_done;
logic [31:0] while_1_arg1_reg_in;
logic while_1_arg1_reg_write_en;
logic while_1_arg1_reg_clk;
logic while_1_arg1_reg_reset;
logic [31:0] while_1_arg1_reg_out;
logic while_1_arg1_reg_done;
logic [31:0] while_1_arg0_reg_in;
logic while_1_arg0_reg_write_en;
logic while_1_arg0_reg_clk;
logic while_1_arg0_reg_reset;
logic [31:0] while_1_arg0_reg_out;
logic while_1_arg0_reg_done;
logic [31:0] while_0_arg0_reg_in;
logic while_0_arg0_reg_write_en;
logic while_0_arg0_reg_clk;
logic while_0_arg0_reg_reset;
logic [31:0] while_0_arg0_reg_out;
logic while_0_arg0_reg_done;
logic comb_reg_in;
logic comb_reg_write_en;
logic comb_reg_clk;
logic comb_reg_reset;
logic comb_reg_out;
logic comb_reg_done;
logic comb_reg0_in;
logic comb_reg0_write_en;
logic comb_reg0_clk;
logic comb_reg0_reset;
logic comb_reg0_out;
logic comb_reg0_done;
logic comb_reg1_in;
logic comb_reg1_write_en;
logic comb_reg1_clk;
logic comb_reg1_reset;
logic comb_reg1_out;
logic comb_reg1_done;
logic comb_reg2_in;
logic comb_reg2_write_en;
logic comb_reg2_clk;
logic comb_reg2_reset;
logic comb_reg2_out;
logic comb_reg2_done;
logic [1:0] fsm_in;
logic fsm_write_en;
logic fsm_clk;
logic fsm_reset;
logic [1:0] fsm_out;
logic fsm_done;
logic ud_out;
logic ud1_out;
logic [1:0] adder_left;
logic [1:0] adder_right;
logic [1:0] adder_out;
logic ud4_out;
logic ud5_out;
logic ud6_out;
logic ud7_out;
logic [1:0] adder0_left;
logic [1:0] adder0_right;
logic [1:0] adder0_out;
logic ud9_out;
logic ud10_out;
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
logic bb0_7_go_in;
logic bb0_7_go_out;
logic bb0_7_done_in;
logic bb0_7_done_out;
logic bb0_8_go_in;
logic bb0_8_go_out;
logic bb0_8_done_in;
logic bb0_8_done_out;
logic assign_while_1_latch_go_in;
logic assign_while_1_latch_go_out;
logic assign_while_1_latch_done_in;
logic assign_while_1_latch_done_out;
logic assign_while_2_latch_go_in;
logic assign_while_2_latch_go_out;
logic assign_while_2_latch_done_in;
logic assign_while_2_latch_done_out;
logic bb0_17_go_in;
logic bb0_17_go_out;
logic bb0_17_done_in;
logic bb0_17_done_out;
logic invoke5_go_in;
logic invoke5_go_out;
logic invoke5_done_in;
logic invoke5_done_out;
logic invoke8_go_in;
logic invoke8_go_out;
logic invoke8_done_in;
logic invoke8_done_out;
logic invoke11_go_in;
logic invoke11_go_out;
logic invoke11_done_in;
logic invoke11_done_out;
logic early_reset_static_par_thread_go_in;
logic early_reset_static_par_thread_go_out;
logic early_reset_static_par_thread_done_in;
logic early_reset_static_par_thread_done_out;
logic early_reset_static_par_thread0_go_in;
logic early_reset_static_par_thread0_go_out;
logic early_reset_static_par_thread0_done_in;
logic early_reset_static_par_thread0_done_out;
logic early_reset_static_seq_go_in;
logic early_reset_static_seq_go_out;
logic early_reset_static_seq_done_in;
logic early_reset_static_seq_done_out;
logic early_reset_bb0_400_go_in;
logic early_reset_bb0_400_go_out;
logic early_reset_bb0_400_done_in;
logic early_reset_bb0_400_done_out;
logic early_reset_bb0_200_go_in;
logic early_reset_bb0_200_go_out;
logic early_reset_bb0_200_done_in;
logic early_reset_bb0_200_done_out;
logic early_reset_bb0_000_go_in;
logic early_reset_bb0_000_go_out;
logic early_reset_bb0_000_done_in;
logic early_reset_bb0_000_done_out;
logic early_reset_static_seq0_go_in;
logic early_reset_static_seq0_go_out;
logic early_reset_static_seq0_done_in;
logic early_reset_static_seq0_done_out;
logic early_reset_bb0_1400_go_in;
logic early_reset_bb0_1400_go_out;
logic early_reset_bb0_1400_done_in;
logic early_reset_bb0_1400_done_out;
logic wrapper_early_reset_static_par_thread_go_in;
logic wrapper_early_reset_static_par_thread_go_out;
logic wrapper_early_reset_static_par_thread_done_in;
logic wrapper_early_reset_static_par_thread_done_out;
logic wrapper_early_reset_bb0_000_go_in;
logic wrapper_early_reset_bb0_000_go_out;
logic wrapper_early_reset_bb0_000_done_in;
logic wrapper_early_reset_bb0_000_done_out;
logic wrapper_early_reset_static_par_thread0_go_in;
logic wrapper_early_reset_static_par_thread0_go_out;
logic wrapper_early_reset_static_par_thread0_done_in;
logic wrapper_early_reset_static_par_thread0_done_out;
logic wrapper_early_reset_bb0_200_go_in;
logic wrapper_early_reset_bb0_200_go_out;
logic wrapper_early_reset_bb0_200_done_in;
logic wrapper_early_reset_bb0_200_done_out;
logic wrapper_early_reset_bb0_400_go_in;
logic wrapper_early_reset_bb0_400_go_out;
logic wrapper_early_reset_bb0_400_done_in;
logic wrapper_early_reset_bb0_400_done_out;
logic wrapper_early_reset_static_seq_go_in;
logic wrapper_early_reset_static_seq_go_out;
logic wrapper_early_reset_static_seq_done_in;
logic wrapper_early_reset_static_seq_done_out;
logic wrapper_early_reset_bb0_1400_go_in;
logic wrapper_early_reset_bb0_1400_go_out;
logic wrapper_early_reset_bb0_1400_done_in;
logic wrapper_early_reset_bb0_1400_done_out;
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
std_slt # (
    .WIDTH(32)
) std_slt_3 (
    .left(std_slt_3_left),
    .out(std_slt_3_out),
    .right(std_slt_3_right)
);
std_add # (
    .WIDTH(32)
) std_add_6 (
    .left(std_add_6_left),
    .out(std_add_6_out),
    .right(std_add_6_right)
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
std_add # (
    .WIDTH(32)
) std_add_0 (
    .left(std_add_0_left),
    .out(std_add_0_out),
    .right(std_add_0_right)
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
) while_2_arg2_reg (
    .clk(while_2_arg2_reg_clk),
    .done(while_2_arg2_reg_done),
    .in(while_2_arg2_reg_in),
    .out(while_2_arg2_reg_out),
    .reset(while_2_arg2_reg_reset),
    .write_en(while_2_arg2_reg_write_en)
);
std_reg # (
    .WIDTH(32)
) while_2_arg1_reg (
    .clk(while_2_arg1_reg_clk),
    .done(while_2_arg1_reg_done),
    .in(while_2_arg1_reg_in),
    .out(while_2_arg1_reg_out),
    .reset(while_2_arg1_reg_reset),
    .write_en(while_2_arg1_reg_write_en)
);
std_reg # (
    .WIDTH(32)
) while_2_arg0_reg (
    .clk(while_2_arg0_reg_clk),
    .done(while_2_arg0_reg_done),
    .in(while_2_arg0_reg_in),
    .out(while_2_arg0_reg_out),
    .reset(while_2_arg0_reg_reset),
    .write_en(while_2_arg0_reg_write_en)
);
std_reg # (
    .WIDTH(32)
) while_1_arg1_reg (
    .clk(while_1_arg1_reg_clk),
    .done(while_1_arg1_reg_done),
    .in(while_1_arg1_reg_in),
    .out(while_1_arg1_reg_out),
    .reset(while_1_arg1_reg_reset),
    .write_en(while_1_arg1_reg_write_en)
);
std_reg # (
    .WIDTH(32)
) while_1_arg0_reg (
    .clk(while_1_arg0_reg_clk),
    .done(while_1_arg0_reg_done),
    .in(while_1_arg0_reg_in),
    .out(while_1_arg0_reg_out),
    .reset(while_1_arg0_reg_reset),
    .write_en(while_1_arg0_reg_write_en)
);
std_reg # (
    .WIDTH(32)
) while_0_arg0_reg (
    .clk(while_0_arg0_reg_clk),
    .done(while_0_arg0_reg_done),
    .in(while_0_arg0_reg_in),
    .out(while_0_arg0_reg_out),
    .reset(while_0_arg0_reg_reset),
    .write_en(while_0_arg0_reg_write_en)
);
std_reg # (
    .WIDTH(1)
) comb_reg (
    .clk(comb_reg_clk),
    .done(comb_reg_done),
    .in(comb_reg_in),
    .out(comb_reg_out),
    .reset(comb_reg_reset),
    .write_en(comb_reg_write_en)
);
std_reg # (
    .WIDTH(1)
) comb_reg0 (
    .clk(comb_reg0_clk),
    .done(comb_reg0_done),
    .in(comb_reg0_in),
    .out(comb_reg0_out),
    .reset(comb_reg0_reset),
    .write_en(comb_reg0_write_en)
);
std_reg # (
    .WIDTH(1)
) comb_reg1 (
    .clk(comb_reg1_clk),
    .done(comb_reg1_done),
    .in(comb_reg1_in),
    .out(comb_reg1_out),
    .reset(comb_reg1_reset),
    .write_en(comb_reg1_write_en)
);
std_reg # (
    .WIDTH(1)
) comb_reg2 (
    .clk(comb_reg2_clk),
    .done(comb_reg2_done),
    .in(comb_reg2_in),
    .out(comb_reg2_out),
    .reset(comb_reg2_reset),
    .write_en(comb_reg2_write_en)
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
undef # (
    .WIDTH(1)
) ud (
    .out(ud_out)
);
undef # (
    .WIDTH(1)
) ud1 (
    .out(ud1_out)
);
std_add # (
    .WIDTH(2)
) adder (
    .left(adder_left),
    .out(adder_out),
    .right(adder_right)
);
undef # (
    .WIDTH(1)
) ud4 (
    .out(ud4_out)
);
undef # (
    .WIDTH(1)
) ud5 (
    .out(ud5_out)
);
undef # (
    .WIDTH(1)
) ud6 (
    .out(ud6_out)
);
undef # (
    .WIDTH(1)
) ud7 (
    .out(ud7_out)
);
std_add # (
    .WIDTH(2)
) adder0 (
    .left(adder0_left),
    .out(adder0_out),
    .right(adder0_right)
);
undef # (
    .WIDTH(1)
) ud9 (
    .out(ud9_out)
);
undef # (
    .WIDTH(1)
) ud10 (
    .out(ud10_out)
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
) assign_while_1_latch_go (
    .in(assign_while_1_latch_go_in),
    .out(assign_while_1_latch_go_out)
);
std_wire # (
    .WIDTH(1)
) assign_while_1_latch_done (
    .in(assign_while_1_latch_done_in),
    .out(assign_while_1_latch_done_out)
);
std_wire # (
    .WIDTH(1)
) assign_while_2_latch_go (
    .in(assign_while_2_latch_go_in),
    .out(assign_while_2_latch_go_out)
);
std_wire # (
    .WIDTH(1)
) assign_while_2_latch_done (
    .in(assign_while_2_latch_done_in),
    .out(assign_while_2_latch_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_17_go (
    .in(bb0_17_go_in),
    .out(bb0_17_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_17_done (
    .in(bb0_17_done_in),
    .out(bb0_17_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke5_go (
    .in(invoke5_go_in),
    .out(invoke5_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke5_done (
    .in(invoke5_done_in),
    .out(invoke5_done_out)
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
) early_reset_static_par_thread_go (
    .in(early_reset_static_par_thread_go_in),
    .out(early_reset_static_par_thread_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_static_par_thread_done (
    .in(early_reset_static_par_thread_done_in),
    .out(early_reset_static_par_thread_done_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_static_par_thread0_go (
    .in(early_reset_static_par_thread0_go_in),
    .out(early_reset_static_par_thread0_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_static_par_thread0_done (
    .in(early_reset_static_par_thread0_done_in),
    .out(early_reset_static_par_thread0_done_out)
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
) early_reset_bb0_400_go (
    .in(early_reset_bb0_400_go_in),
    .out(early_reset_bb0_400_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_400_done (
    .in(early_reset_bb0_400_done_in),
    .out(early_reset_bb0_400_done_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_200_go (
    .in(early_reset_bb0_200_go_in),
    .out(early_reset_bb0_200_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_200_done (
    .in(early_reset_bb0_200_done_in),
    .out(early_reset_bb0_200_done_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_000_go (
    .in(early_reset_bb0_000_go_in),
    .out(early_reset_bb0_000_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_000_done (
    .in(early_reset_bb0_000_done_in),
    .out(early_reset_bb0_000_done_out)
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
) early_reset_bb0_1400_go (
    .in(early_reset_bb0_1400_go_in),
    .out(early_reset_bb0_1400_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_1400_done (
    .in(early_reset_bb0_1400_done_in),
    .out(early_reset_bb0_1400_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_par_thread_go (
    .in(wrapper_early_reset_static_par_thread_go_in),
    .out(wrapper_early_reset_static_par_thread_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_par_thread_done (
    .in(wrapper_early_reset_static_par_thread_done_in),
    .out(wrapper_early_reset_static_par_thread_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_000_go (
    .in(wrapper_early_reset_bb0_000_go_in),
    .out(wrapper_early_reset_bb0_000_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_000_done (
    .in(wrapper_early_reset_bb0_000_done_in),
    .out(wrapper_early_reset_bb0_000_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_par_thread0_go (
    .in(wrapper_early_reset_static_par_thread0_go_in),
    .out(wrapper_early_reset_static_par_thread0_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_par_thread0_done (
    .in(wrapper_early_reset_static_par_thread0_done_in),
    .out(wrapper_early_reset_static_par_thread0_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_200_go (
    .in(wrapper_early_reset_bb0_200_go_in),
    .out(wrapper_early_reset_bb0_200_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_200_done (
    .in(wrapper_early_reset_bb0_200_done_in),
    .out(wrapper_early_reset_bb0_200_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_400_go (
    .in(wrapper_early_reset_bb0_400_go_in),
    .out(wrapper_early_reset_bb0_400_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_400_done (
    .in(wrapper_early_reset_bb0_400_done_in),
    .out(wrapper_early_reset_bb0_400_done_out)
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
) wrapper_early_reset_bb0_1400_go (
    .in(wrapper_early_reset_bb0_1400_go_in),
    .out(wrapper_early_reset_bb0_1400_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_1400_done (
    .in(wrapper_early_reset_bb0_1400_done_in),
    .out(wrapper_early_reset_bb0_1400_done_out)
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
wire _guard1 = assign_while_1_latch_go_out;
wire _guard2 = assign_while_2_latch_go_out;
wire _guard3 = assign_while_1_latch_go_out;
wire _guard4 = assign_while_2_latch_go_out;
wire _guard5 = bb0_8_go_out;
wire _guard6 = bb0_8_go_out;
wire _guard7 = assign_while_1_latch_done_out;
wire _guard8 = ~_guard7;
wire _guard9 = fsm0_out == 5'd10;
wire _guard10 = _guard8 & _guard9;
wire _guard11 = tdcc_go_out;
wire _guard12 = _guard10 & _guard11;
wire _guard13 = signal_reg_out;
wire _guard14 = tdcc_done_out;
wire _guard15 = bb0_17_go_out;
wire _guard16 = bb0_7_go_out;
wire _guard17 = bb0_7_go_out;
wire _guard18 = bb0_17_go_out;
wire _guard19 = bb0_17_go_out;
wire _guard20 = bb0_17_go_out;
wire _guard21 = fsm_out != 2'd1;
wire _guard22 = early_reset_static_seq_go_out;
wire _guard23 = _guard21 & _guard22;
wire _guard24 = fsm_out == 2'd1;
wire _guard25 = early_reset_static_seq_go_out;
wire _guard26 = _guard24 & _guard25;
wire _guard27 = _guard23 | _guard26;
wire _guard28 = fsm_out != 2'd1;
wire _guard29 = early_reset_static_seq0_go_out;
wire _guard30 = _guard28 & _guard29;
wire _guard31 = _guard27 | _guard30;
wire _guard32 = fsm_out == 2'd1;
wire _guard33 = early_reset_static_seq0_go_out;
wire _guard34 = _guard32 & _guard33;
wire _guard35 = _guard31 | _guard34;
wire _guard36 = fsm_out != 2'd1;
wire _guard37 = early_reset_static_seq_go_out;
wire _guard38 = _guard36 & _guard37;
wire _guard39 = fsm_out == 2'd1;
wire _guard40 = early_reset_static_seq_go_out;
wire _guard41 = _guard39 & _guard40;
wire _guard42 = fsm_out == 2'd1;
wire _guard43 = early_reset_static_seq0_go_out;
wire _guard44 = _guard42 & _guard43;
wire _guard45 = _guard41 | _guard44;
wire _guard46 = fsm_out != 2'd1;
wire _guard47 = early_reset_static_seq0_go_out;
wire _guard48 = _guard46 & _guard47;
wire _guard49 = early_reset_static_seq_go_out;
wire _guard50 = early_reset_static_seq_go_out;
wire _guard51 = while_2_arg2_reg_done;
wire _guard52 = while_2_arg1_reg_done;
wire _guard53 = _guard51 & _guard52;
wire _guard54 = while_2_arg0_reg_done;
wire _guard55 = _guard53 & _guard54;
wire _guard56 = fsm_out == 2'd0;
wire _guard57 = early_reset_static_seq_go_out;
wire _guard58 = _guard56 & _guard57;
wire _guard59 = fsm_out == 2'd0;
wire _guard60 = early_reset_static_seq_go_out;
wire _guard61 = _guard59 & _guard60;
wire _guard62 = fsm_out == 2'd0;
wire _guard63 = early_reset_static_seq_go_out;
wire _guard64 = _guard62 & _guard63;
wire _guard65 = wrapper_early_reset_static_par_thread_done_out;
wire _guard66 = ~_guard65;
wire _guard67 = fsm0_out == 5'd0;
wire _guard68 = _guard66 & _guard67;
wire _guard69 = tdcc_go_out;
wire _guard70 = _guard68 & _guard69;
wire _guard71 = assign_while_1_latch_go_out;
wire _guard72 = early_reset_static_par_thread0_go_out;
wire _guard73 = _guard71 | _guard72;
wire _guard74 = assign_while_1_latch_go_out;
wire _guard75 = early_reset_static_par_thread0_go_out;
wire _guard76 = early_reset_bb0_000_go_out;
wire _guard77 = early_reset_bb0_000_go_out;
wire _guard78 = signal_reg_out;
wire _guard79 = bb0_8_go_out;
wire _guard80 = bb0_8_go_out;
wire _guard81 = wrapper_early_reset_static_seq0_go_out;
wire _guard82 = bb0_7_go_out;
wire _guard83 = fsm_out == 2'd0;
wire _guard84 = early_reset_static_seq_go_out;
wire _guard85 = _guard83 & _guard84;
wire _guard86 = _guard82 | _guard85;
wire _guard87 = bb0_17_go_out;
wire _guard88 = fsm_out == 2'd0;
wire _guard89 = early_reset_static_seq0_go_out;
wire _guard90 = _guard88 & _guard89;
wire _guard91 = _guard87 | _guard90;
wire _guard92 = bb0_8_go_out;
wire _guard93 = bb0_8_go_out;
wire _guard94 = invoke5_go_out;
wire _guard95 = fsm_out == 2'd1;
wire _guard96 = early_reset_static_seq_go_out;
wire _guard97 = _guard95 & _guard96;
wire _guard98 = _guard94 | _guard97;
wire _guard99 = invoke5_go_out;
wire _guard100 = fsm_out == 2'd1;
wire _guard101 = early_reset_static_seq_go_out;
wire _guard102 = _guard100 & _guard101;
wire _guard103 = early_reset_bb0_400_go_out;
wire _guard104 = early_reset_bb0_400_go_out;
wire _guard105 = assign_while_2_latch_done_out;
wire _guard106 = ~_guard105;
wire _guard107 = fsm0_out == 5'd12;
wire _guard108 = _guard106 & _guard107;
wire _guard109 = tdcc_go_out;
wire _guard110 = _guard108 & _guard109;
wire _guard111 = invoke5_done_out;
wire _guard112 = ~_guard111;
wire _guard113 = fsm0_out == 5'd4;
wire _guard114 = _guard112 & _guard113;
wire _guard115 = tdcc_go_out;
wire _guard116 = _guard114 & _guard115;
wire _guard117 = invoke11_done_out;
wire _guard118 = ~_guard117;
wire _guard119 = fsm0_out == 5'd18;
wire _guard120 = _guard118 & _guard119;
wire _guard121 = tdcc_go_out;
wire _guard122 = _guard120 & _guard121;
wire _guard123 = signal_reg_out;
wire _guard124 = wrapper_early_reset_static_seq0_done_out;
wire _guard125 = ~_guard124;
wire _guard126 = fsm0_out == 5'd16;
wire _guard127 = _guard125 & _guard126;
wire _guard128 = tdcc_go_out;
wire _guard129 = _guard127 & _guard128;
wire _guard130 = bb0_8_go_out;
wire _guard131 = bb0_8_go_out;
wire _guard132 = early_reset_bb0_200_go_out;
wire _guard133 = early_reset_bb0_200_go_out;
wire _guard134 = assign_while_2_latch_go_out;
wire _guard135 = assign_while_2_latch_go_out;
wire _guard136 = assign_while_2_latch_go_out;
wire _guard137 = early_reset_static_par_thread_go_out;
wire _guard138 = _guard136 | _guard137;
wire _guard139 = early_reset_static_par_thread_go_out;
wire _guard140 = assign_while_2_latch_go_out;
wire _guard141 = early_reset_bb0_1400_go_out;
wire _guard142 = early_reset_bb0_1400_go_out;
wire _guard143 = fsm0_out == 5'd20;
wire _guard144 = fsm0_out == 5'd0;
wire _guard145 = wrapper_early_reset_static_par_thread_done_out;
wire _guard146 = _guard144 & _guard145;
wire _guard147 = tdcc_go_out;
wire _guard148 = _guard146 & _guard147;
wire _guard149 = _guard143 | _guard148;
wire _guard150 = fsm0_out == 5'd1;
wire _guard151 = wrapper_early_reset_bb0_000_done_out;
wire _guard152 = comb_reg_out;
wire _guard153 = _guard151 & _guard152;
wire _guard154 = _guard150 & _guard153;
wire _guard155 = tdcc_go_out;
wire _guard156 = _guard154 & _guard155;
wire _guard157 = _guard149 | _guard156;
wire _guard158 = fsm0_out == 5'd13;
wire _guard159 = wrapper_early_reset_bb0_000_done_out;
wire _guard160 = comb_reg_out;
wire _guard161 = _guard159 & _guard160;
wire _guard162 = _guard158 & _guard161;
wire _guard163 = tdcc_go_out;
wire _guard164 = _guard162 & _guard163;
wire _guard165 = _guard157 | _guard164;
wire _guard166 = fsm0_out == 5'd2;
wire _guard167 = wrapper_early_reset_static_par_thread0_done_out;
wire _guard168 = _guard166 & _guard167;
wire _guard169 = tdcc_go_out;
wire _guard170 = _guard168 & _guard169;
wire _guard171 = _guard165 | _guard170;
wire _guard172 = fsm0_out == 5'd3;
wire _guard173 = wrapper_early_reset_bb0_200_done_out;
wire _guard174 = comb_reg0_out;
wire _guard175 = _guard173 & _guard174;
wire _guard176 = _guard172 & _guard175;
wire _guard177 = tdcc_go_out;
wire _guard178 = _guard176 & _guard177;
wire _guard179 = _guard171 | _guard178;
wire _guard180 = fsm0_out == 5'd11;
wire _guard181 = wrapper_early_reset_bb0_200_done_out;
wire _guard182 = comb_reg0_out;
wire _guard183 = _guard181 & _guard182;
wire _guard184 = _guard180 & _guard183;
wire _guard185 = tdcc_go_out;
wire _guard186 = _guard184 & _guard185;
wire _guard187 = _guard179 | _guard186;
wire _guard188 = fsm0_out == 5'd4;
wire _guard189 = invoke5_done_out;
wire _guard190 = _guard188 & _guard189;
wire _guard191 = tdcc_go_out;
wire _guard192 = _guard190 & _guard191;
wire _guard193 = _guard187 | _guard192;
wire _guard194 = fsm0_out == 5'd5;
wire _guard195 = wrapper_early_reset_bb0_400_done_out;
wire _guard196 = comb_reg1_out;
wire _guard197 = _guard195 & _guard196;
wire _guard198 = _guard194 & _guard197;
wire _guard199 = tdcc_go_out;
wire _guard200 = _guard198 & _guard199;
wire _guard201 = _guard193 | _guard200;
wire _guard202 = fsm0_out == 5'd9;
wire _guard203 = wrapper_early_reset_bb0_400_done_out;
wire _guard204 = comb_reg1_out;
wire _guard205 = _guard203 & _guard204;
wire _guard206 = _guard202 & _guard205;
wire _guard207 = tdcc_go_out;
wire _guard208 = _guard206 & _guard207;
wire _guard209 = _guard201 | _guard208;
wire _guard210 = fsm0_out == 5'd6;
wire _guard211 = bb0_7_done_out;
wire _guard212 = _guard210 & _guard211;
wire _guard213 = tdcc_go_out;
wire _guard214 = _guard212 & _guard213;
wire _guard215 = _guard209 | _guard214;
wire _guard216 = fsm0_out == 5'd7;
wire _guard217 = bb0_8_done_out;
wire _guard218 = _guard216 & _guard217;
wire _guard219 = tdcc_go_out;
wire _guard220 = _guard218 & _guard219;
wire _guard221 = _guard215 | _guard220;
wire _guard222 = fsm0_out == 5'd8;
wire _guard223 = wrapper_early_reset_static_seq_done_out;
wire _guard224 = _guard222 & _guard223;
wire _guard225 = tdcc_go_out;
wire _guard226 = _guard224 & _guard225;
wire _guard227 = _guard221 | _guard226;
wire _guard228 = fsm0_out == 5'd5;
wire _guard229 = wrapper_early_reset_bb0_400_done_out;
wire _guard230 = comb_reg1_out;
wire _guard231 = ~_guard230;
wire _guard232 = _guard229 & _guard231;
wire _guard233 = _guard228 & _guard232;
wire _guard234 = tdcc_go_out;
wire _guard235 = _guard233 & _guard234;
wire _guard236 = _guard227 | _guard235;
wire _guard237 = fsm0_out == 5'd9;
wire _guard238 = wrapper_early_reset_bb0_400_done_out;
wire _guard239 = comb_reg1_out;
wire _guard240 = ~_guard239;
wire _guard241 = _guard238 & _guard240;
wire _guard242 = _guard237 & _guard241;
wire _guard243 = tdcc_go_out;
wire _guard244 = _guard242 & _guard243;
wire _guard245 = _guard236 | _guard244;
wire _guard246 = fsm0_out == 5'd10;
wire _guard247 = assign_while_1_latch_done_out;
wire _guard248 = _guard246 & _guard247;
wire _guard249 = tdcc_go_out;
wire _guard250 = _guard248 & _guard249;
wire _guard251 = _guard245 | _guard250;
wire _guard252 = fsm0_out == 5'd3;
wire _guard253 = wrapper_early_reset_bb0_200_done_out;
wire _guard254 = comb_reg0_out;
wire _guard255 = ~_guard254;
wire _guard256 = _guard253 & _guard255;
wire _guard257 = _guard252 & _guard256;
wire _guard258 = tdcc_go_out;
wire _guard259 = _guard257 & _guard258;
wire _guard260 = _guard251 | _guard259;
wire _guard261 = fsm0_out == 5'd11;
wire _guard262 = wrapper_early_reset_bb0_200_done_out;
wire _guard263 = comb_reg0_out;
wire _guard264 = ~_guard263;
wire _guard265 = _guard262 & _guard264;
wire _guard266 = _guard261 & _guard265;
wire _guard267 = tdcc_go_out;
wire _guard268 = _guard266 & _guard267;
wire _guard269 = _guard260 | _guard268;
wire _guard270 = fsm0_out == 5'd12;
wire _guard271 = assign_while_2_latch_done_out;
wire _guard272 = _guard270 & _guard271;
wire _guard273 = tdcc_go_out;
wire _guard274 = _guard272 & _guard273;
wire _guard275 = _guard269 | _guard274;
wire _guard276 = fsm0_out == 5'd1;
wire _guard277 = wrapper_early_reset_bb0_000_done_out;
wire _guard278 = comb_reg_out;
wire _guard279 = ~_guard278;
wire _guard280 = _guard277 & _guard279;
wire _guard281 = _guard276 & _guard280;
wire _guard282 = tdcc_go_out;
wire _guard283 = _guard281 & _guard282;
wire _guard284 = _guard275 | _guard283;
wire _guard285 = fsm0_out == 5'd13;
wire _guard286 = wrapper_early_reset_bb0_000_done_out;
wire _guard287 = comb_reg_out;
wire _guard288 = ~_guard287;
wire _guard289 = _guard286 & _guard288;
wire _guard290 = _guard285 & _guard289;
wire _guard291 = tdcc_go_out;
wire _guard292 = _guard290 & _guard291;
wire _guard293 = _guard284 | _guard292;
wire _guard294 = fsm0_out == 5'd14;
wire _guard295 = invoke8_done_out;
wire _guard296 = _guard294 & _guard295;
wire _guard297 = tdcc_go_out;
wire _guard298 = _guard296 & _guard297;
wire _guard299 = _guard293 | _guard298;
wire _guard300 = fsm0_out == 5'd15;
wire _guard301 = wrapper_early_reset_bb0_1400_done_out;
wire _guard302 = comb_reg2_out;
wire _guard303 = _guard301 & _guard302;
wire _guard304 = _guard300 & _guard303;
wire _guard305 = tdcc_go_out;
wire _guard306 = _guard304 & _guard305;
wire _guard307 = _guard299 | _guard306;
wire _guard308 = fsm0_out == 5'd19;
wire _guard309 = wrapper_early_reset_bb0_1400_done_out;
wire _guard310 = comb_reg2_out;
wire _guard311 = _guard309 & _guard310;
wire _guard312 = _guard308 & _guard311;
wire _guard313 = tdcc_go_out;
wire _guard314 = _guard312 & _guard313;
wire _guard315 = _guard307 | _guard314;
wire _guard316 = fsm0_out == 5'd16;
wire _guard317 = wrapper_early_reset_static_seq0_done_out;
wire _guard318 = _guard316 & _guard317;
wire _guard319 = tdcc_go_out;
wire _guard320 = _guard318 & _guard319;
wire _guard321 = _guard315 | _guard320;
wire _guard322 = fsm0_out == 5'd17;
wire _guard323 = bb0_17_done_out;
wire _guard324 = _guard322 & _guard323;
wire _guard325 = tdcc_go_out;
wire _guard326 = _guard324 & _guard325;
wire _guard327 = _guard321 | _guard326;
wire _guard328 = fsm0_out == 5'd18;
wire _guard329 = invoke11_done_out;
wire _guard330 = _guard328 & _guard329;
wire _guard331 = tdcc_go_out;
wire _guard332 = _guard330 & _guard331;
wire _guard333 = _guard327 | _guard332;
wire _guard334 = fsm0_out == 5'd15;
wire _guard335 = wrapper_early_reset_bb0_1400_done_out;
wire _guard336 = comb_reg2_out;
wire _guard337 = ~_guard336;
wire _guard338 = _guard335 & _guard337;
wire _guard339 = _guard334 & _guard338;
wire _guard340 = tdcc_go_out;
wire _guard341 = _guard339 & _guard340;
wire _guard342 = _guard333 | _guard341;
wire _guard343 = fsm0_out == 5'd19;
wire _guard344 = wrapper_early_reset_bb0_1400_done_out;
wire _guard345 = comb_reg2_out;
wire _guard346 = ~_guard345;
wire _guard347 = _guard344 & _guard346;
wire _guard348 = _guard343 & _guard347;
wire _guard349 = tdcc_go_out;
wire _guard350 = _guard348 & _guard349;
wire _guard351 = _guard342 | _guard350;
wire _guard352 = fsm0_out == 5'd0;
wire _guard353 = wrapper_early_reset_static_par_thread_done_out;
wire _guard354 = _guard352 & _guard353;
wire _guard355 = tdcc_go_out;
wire _guard356 = _guard354 & _guard355;
wire _guard357 = fsm0_out == 5'd14;
wire _guard358 = invoke8_done_out;
wire _guard359 = _guard357 & _guard358;
wire _guard360 = tdcc_go_out;
wire _guard361 = _guard359 & _guard360;
wire _guard362 = fsm0_out == 5'd17;
wire _guard363 = bb0_17_done_out;
wire _guard364 = _guard362 & _guard363;
wire _guard365 = tdcc_go_out;
wire _guard366 = _guard364 & _guard365;
wire _guard367 = fsm0_out == 5'd15;
wire _guard368 = wrapper_early_reset_bb0_1400_done_out;
wire _guard369 = comb_reg2_out;
wire _guard370 = _guard368 & _guard369;
wire _guard371 = _guard367 & _guard370;
wire _guard372 = tdcc_go_out;
wire _guard373 = _guard371 & _guard372;
wire _guard374 = fsm0_out == 5'd19;
wire _guard375 = wrapper_early_reset_bb0_1400_done_out;
wire _guard376 = comb_reg2_out;
wire _guard377 = _guard375 & _guard376;
wire _guard378 = _guard374 & _guard377;
wire _guard379 = tdcc_go_out;
wire _guard380 = _guard378 & _guard379;
wire _guard381 = _guard373 | _guard380;
wire _guard382 = fsm0_out == 5'd20;
wire _guard383 = fsm0_out == 5'd2;
wire _guard384 = wrapper_early_reset_static_par_thread0_done_out;
wire _guard385 = _guard383 & _guard384;
wire _guard386 = tdcc_go_out;
wire _guard387 = _guard385 & _guard386;
wire _guard388 = fsm0_out == 5'd12;
wire _guard389 = assign_while_2_latch_done_out;
wire _guard390 = _guard388 & _guard389;
wire _guard391 = tdcc_go_out;
wire _guard392 = _guard390 & _guard391;
wire _guard393 = fsm0_out == 5'd1;
wire _guard394 = wrapper_early_reset_bb0_000_done_out;
wire _guard395 = comb_reg_out;
wire _guard396 = ~_guard395;
wire _guard397 = _guard394 & _guard396;
wire _guard398 = _guard393 & _guard397;
wire _guard399 = tdcc_go_out;
wire _guard400 = _guard398 & _guard399;
wire _guard401 = fsm0_out == 5'd13;
wire _guard402 = wrapper_early_reset_bb0_000_done_out;
wire _guard403 = comb_reg_out;
wire _guard404 = ~_guard403;
wire _guard405 = _guard402 & _guard404;
wire _guard406 = _guard401 & _guard405;
wire _guard407 = tdcc_go_out;
wire _guard408 = _guard406 & _guard407;
wire _guard409 = _guard400 | _guard408;
wire _guard410 = fsm0_out == 5'd4;
wire _guard411 = invoke5_done_out;
wire _guard412 = _guard410 & _guard411;
wire _guard413 = tdcc_go_out;
wire _guard414 = _guard412 & _guard413;
wire _guard415 = fsm0_out == 5'd3;
wire _guard416 = wrapper_early_reset_bb0_200_done_out;
wire _guard417 = comb_reg0_out;
wire _guard418 = ~_guard417;
wire _guard419 = _guard416 & _guard418;
wire _guard420 = _guard415 & _guard419;
wire _guard421 = tdcc_go_out;
wire _guard422 = _guard420 & _guard421;
wire _guard423 = fsm0_out == 5'd11;
wire _guard424 = wrapper_early_reset_bb0_200_done_out;
wire _guard425 = comb_reg0_out;
wire _guard426 = ~_guard425;
wire _guard427 = _guard424 & _guard426;
wire _guard428 = _guard423 & _guard427;
wire _guard429 = tdcc_go_out;
wire _guard430 = _guard428 & _guard429;
wire _guard431 = _guard422 | _guard430;
wire _guard432 = fsm0_out == 5'd1;
wire _guard433 = wrapper_early_reset_bb0_000_done_out;
wire _guard434 = comb_reg_out;
wire _guard435 = _guard433 & _guard434;
wire _guard436 = _guard432 & _guard435;
wire _guard437 = tdcc_go_out;
wire _guard438 = _guard436 & _guard437;
wire _guard439 = fsm0_out == 5'd13;
wire _guard440 = wrapper_early_reset_bb0_000_done_out;
wire _guard441 = comb_reg_out;
wire _guard442 = _guard440 & _guard441;
wire _guard443 = _guard439 & _guard442;
wire _guard444 = tdcc_go_out;
wire _guard445 = _guard443 & _guard444;
wire _guard446 = _guard438 | _guard445;
wire _guard447 = fsm0_out == 5'd7;
wire _guard448 = bb0_8_done_out;
wire _guard449 = _guard447 & _guard448;
wire _guard450 = tdcc_go_out;
wire _guard451 = _guard449 & _guard450;
wire _guard452 = fsm0_out == 5'd5;
wire _guard453 = wrapper_early_reset_bb0_400_done_out;
wire _guard454 = comb_reg1_out;
wire _guard455 = ~_guard454;
wire _guard456 = _guard453 & _guard455;
wire _guard457 = _guard452 & _guard456;
wire _guard458 = tdcc_go_out;
wire _guard459 = _guard457 & _guard458;
wire _guard460 = fsm0_out == 5'd9;
wire _guard461 = wrapper_early_reset_bb0_400_done_out;
wire _guard462 = comb_reg1_out;
wire _guard463 = ~_guard462;
wire _guard464 = _guard461 & _guard463;
wire _guard465 = _guard460 & _guard464;
wire _guard466 = tdcc_go_out;
wire _guard467 = _guard465 & _guard466;
wire _guard468 = _guard459 | _guard467;
wire _guard469 = fsm0_out == 5'd6;
wire _guard470 = bb0_7_done_out;
wire _guard471 = _guard469 & _guard470;
wire _guard472 = tdcc_go_out;
wire _guard473 = _guard471 & _guard472;
wire _guard474 = fsm0_out == 5'd10;
wire _guard475 = assign_while_1_latch_done_out;
wire _guard476 = _guard474 & _guard475;
wire _guard477 = tdcc_go_out;
wire _guard478 = _guard476 & _guard477;
wire _guard479 = fsm0_out == 5'd18;
wire _guard480 = invoke11_done_out;
wire _guard481 = _guard479 & _guard480;
wire _guard482 = tdcc_go_out;
wire _guard483 = _guard481 & _guard482;
wire _guard484 = fsm0_out == 5'd3;
wire _guard485 = wrapper_early_reset_bb0_200_done_out;
wire _guard486 = comb_reg0_out;
wire _guard487 = _guard485 & _guard486;
wire _guard488 = _guard484 & _guard487;
wire _guard489 = tdcc_go_out;
wire _guard490 = _guard488 & _guard489;
wire _guard491 = fsm0_out == 5'd11;
wire _guard492 = wrapper_early_reset_bb0_200_done_out;
wire _guard493 = comb_reg0_out;
wire _guard494 = _guard492 & _guard493;
wire _guard495 = _guard491 & _guard494;
wire _guard496 = tdcc_go_out;
wire _guard497 = _guard495 & _guard496;
wire _guard498 = _guard490 | _guard497;
wire _guard499 = fsm0_out == 5'd5;
wire _guard500 = wrapper_early_reset_bb0_400_done_out;
wire _guard501 = comb_reg1_out;
wire _guard502 = _guard500 & _guard501;
wire _guard503 = _guard499 & _guard502;
wire _guard504 = tdcc_go_out;
wire _guard505 = _guard503 & _guard504;
wire _guard506 = fsm0_out == 5'd9;
wire _guard507 = wrapper_early_reset_bb0_400_done_out;
wire _guard508 = comb_reg1_out;
wire _guard509 = _guard507 & _guard508;
wire _guard510 = _guard506 & _guard509;
wire _guard511 = tdcc_go_out;
wire _guard512 = _guard510 & _guard511;
wire _guard513 = _guard505 | _guard512;
wire _guard514 = fsm0_out == 5'd15;
wire _guard515 = wrapper_early_reset_bb0_1400_done_out;
wire _guard516 = comb_reg2_out;
wire _guard517 = ~_guard516;
wire _guard518 = _guard515 & _guard517;
wire _guard519 = _guard514 & _guard518;
wire _guard520 = tdcc_go_out;
wire _guard521 = _guard519 & _guard520;
wire _guard522 = fsm0_out == 5'd19;
wire _guard523 = wrapper_early_reset_bb0_1400_done_out;
wire _guard524 = comb_reg2_out;
wire _guard525 = ~_guard524;
wire _guard526 = _guard523 & _guard525;
wire _guard527 = _guard522 & _guard526;
wire _guard528 = tdcc_go_out;
wire _guard529 = _guard527 & _guard528;
wire _guard530 = _guard521 | _guard529;
wire _guard531 = fsm0_out == 5'd16;
wire _guard532 = wrapper_early_reset_static_seq0_done_out;
wire _guard533 = _guard531 & _guard532;
wire _guard534 = tdcc_go_out;
wire _guard535 = _guard533 & _guard534;
wire _guard536 = fsm0_out == 5'd8;
wire _guard537 = wrapper_early_reset_static_seq_done_out;
wire _guard538 = _guard536 & _guard537;
wire _guard539 = tdcc_go_out;
wire _guard540 = _guard538 & _guard539;
wire _guard541 = invoke8_done_out;
wire _guard542 = ~_guard541;
wire _guard543 = fsm0_out == 5'd14;
wire _guard544 = _guard542 & _guard543;
wire _guard545 = tdcc_go_out;
wire _guard546 = _guard544 & _guard545;
wire _guard547 = wrapper_early_reset_bb0_200_done_out;
wire _guard548 = ~_guard547;
wire _guard549 = fsm0_out == 5'd3;
wire _guard550 = _guard548 & _guard549;
wire _guard551 = tdcc_go_out;
wire _guard552 = _guard550 & _guard551;
wire _guard553 = wrapper_early_reset_bb0_200_done_out;
wire _guard554 = ~_guard553;
wire _guard555 = fsm0_out == 5'd11;
wire _guard556 = _guard554 & _guard555;
wire _guard557 = tdcc_go_out;
wire _guard558 = _guard556 & _guard557;
wire _guard559 = _guard552 | _guard558;
wire _guard560 = fsm_out == 2'd1;
wire _guard561 = early_reset_static_seq_go_out;
wire _guard562 = _guard560 & _guard561;
wire _guard563 = bb0_7_go_out;
wire _guard564 = assign_while_1_latch_go_out;
wire _guard565 = _guard563 | _guard564;
wire _guard566 = fsm_out == 2'd0;
wire _guard567 = early_reset_static_seq_go_out;
wire _guard568 = _guard566 & _guard567;
wire _guard569 = _guard565 | _guard568;
wire _guard570 = assign_while_2_latch_go_out;
wire _guard571 = invoke11_go_out;
wire _guard572 = bb0_7_go_out;
wire _guard573 = fsm_out == 2'd0;
wire _guard574 = early_reset_static_seq_go_out;
wire _guard575 = _guard573 & _guard574;
wire _guard576 = _guard572 | _guard575;
wire _guard577 = invoke11_go_out;
wire _guard578 = fsm_out == 2'd1;
wire _guard579 = early_reset_static_seq_go_out;
wire _guard580 = _guard578 & _guard579;
wire _guard581 = _guard577 | _guard580;
wire _guard582 = assign_while_1_latch_go_out;
wire _guard583 = assign_while_2_latch_go_out;
wire _guard584 = assign_while_1_latch_go_out;
wire _guard585 = early_reset_static_par_thread0_go_out;
wire _guard586 = _guard584 | _guard585;
wire _guard587 = assign_while_1_latch_go_out;
wire _guard588 = early_reset_static_par_thread0_go_out;
wire _guard589 = bb0_17_done_out;
wire _guard590 = ~_guard589;
wire _guard591 = fsm0_out == 5'd17;
wire _guard592 = _guard590 & _guard591;
wire _guard593 = tdcc_go_out;
wire _guard594 = _guard592 & _guard593;
wire _guard595 = wrapper_early_reset_bb0_000_go_out;
wire _guard596 = signal_reg_out;
wire _guard597 = wrapper_early_reset_bb0_1400_done_out;
wire _guard598 = ~_guard597;
wire _guard599 = fsm0_out == 5'd15;
wire _guard600 = _guard598 & _guard599;
wire _guard601 = tdcc_go_out;
wire _guard602 = _guard600 & _guard601;
wire _guard603 = wrapper_early_reset_bb0_1400_done_out;
wire _guard604 = ~_guard603;
wire _guard605 = fsm0_out == 5'd19;
wire _guard606 = _guard604 & _guard605;
wire _guard607 = tdcc_go_out;
wire _guard608 = _guard606 & _guard607;
wire _guard609 = _guard602 | _guard608;
wire _guard610 = early_reset_static_seq0_go_out;
wire _guard611 = early_reset_static_seq0_go_out;
wire _guard612 = wrapper_early_reset_static_seq_go_out;
wire _guard613 = wrapper_early_reset_static_par_thread0_done_out;
wire _guard614 = ~_guard613;
wire _guard615 = fsm0_out == 5'd2;
wire _guard616 = _guard614 & _guard615;
wire _guard617 = tdcc_go_out;
wire _guard618 = _guard616 & _guard617;
wire _guard619 = fsm_out == 2'd0;
wire _guard620 = early_reset_static_seq_go_out;
wire _guard621 = _guard619 & _guard620;
wire _guard622 = fsm_out == 2'd0;
wire _guard623 = early_reset_static_seq_go_out;
wire _guard624 = _guard622 & _guard623;
wire _guard625 = fsm_out == 2'd0;
wire _guard626 = early_reset_static_seq0_go_out;
wire _guard627 = _guard625 & _guard626;
wire _guard628 = _guard624 | _guard627;
wire _guard629 = fsm_out == 2'd0;
wire _guard630 = early_reset_static_seq_go_out;
wire _guard631 = _guard629 & _guard630;
wire _guard632 = fsm_out == 2'd0;
wire _guard633 = early_reset_static_seq0_go_out;
wire _guard634 = _guard632 & _guard633;
wire _guard635 = _guard631 | _guard634;
wire _guard636 = fsm_out == 2'd0;
wire _guard637 = early_reset_static_seq_go_out;
wire _guard638 = _guard636 & _guard637;
wire _guard639 = signal_reg_out;
wire _guard640 = _guard0 & _guard0;
wire _guard641 = signal_reg_out;
wire _guard642 = ~_guard641;
wire _guard643 = _guard640 & _guard642;
wire _guard644 = wrapper_early_reset_static_par_thread_go_out;
wire _guard645 = _guard643 & _guard644;
wire _guard646 = _guard639 | _guard645;
wire _guard647 = _guard0 & _guard0;
wire _guard648 = signal_reg_out;
wire _guard649 = ~_guard648;
wire _guard650 = _guard647 & _guard649;
wire _guard651 = wrapper_early_reset_bb0_000_go_out;
wire _guard652 = _guard650 & _guard651;
wire _guard653 = _guard646 | _guard652;
wire _guard654 = _guard0 & _guard0;
wire _guard655 = signal_reg_out;
wire _guard656 = ~_guard655;
wire _guard657 = _guard654 & _guard656;
wire _guard658 = wrapper_early_reset_static_par_thread0_go_out;
wire _guard659 = _guard657 & _guard658;
wire _guard660 = _guard653 | _guard659;
wire _guard661 = _guard0 & _guard0;
wire _guard662 = signal_reg_out;
wire _guard663 = ~_guard662;
wire _guard664 = _guard661 & _guard663;
wire _guard665 = wrapper_early_reset_bb0_200_go_out;
wire _guard666 = _guard664 & _guard665;
wire _guard667 = _guard660 | _guard666;
wire _guard668 = _guard0 & _guard0;
wire _guard669 = signal_reg_out;
wire _guard670 = ~_guard669;
wire _guard671 = _guard668 & _guard670;
wire _guard672 = wrapper_early_reset_bb0_400_go_out;
wire _guard673 = _guard671 & _guard672;
wire _guard674 = _guard667 | _guard673;
wire _guard675 = fsm_out == 2'd1;
wire _guard676 = _guard675 & _guard0;
wire _guard677 = signal_reg_out;
wire _guard678 = ~_guard677;
wire _guard679 = _guard676 & _guard678;
wire _guard680 = wrapper_early_reset_static_seq_go_out;
wire _guard681 = _guard679 & _guard680;
wire _guard682 = _guard674 | _guard681;
wire _guard683 = _guard0 & _guard0;
wire _guard684 = signal_reg_out;
wire _guard685 = ~_guard684;
wire _guard686 = _guard683 & _guard685;
wire _guard687 = wrapper_early_reset_bb0_1400_go_out;
wire _guard688 = _guard686 & _guard687;
wire _guard689 = _guard682 | _guard688;
wire _guard690 = fsm_out == 2'd1;
wire _guard691 = _guard690 & _guard0;
wire _guard692 = signal_reg_out;
wire _guard693 = ~_guard692;
wire _guard694 = _guard691 & _guard693;
wire _guard695 = wrapper_early_reset_static_seq0_go_out;
wire _guard696 = _guard694 & _guard695;
wire _guard697 = _guard689 | _guard696;
wire _guard698 = _guard0 & _guard0;
wire _guard699 = signal_reg_out;
wire _guard700 = ~_guard699;
wire _guard701 = _guard698 & _guard700;
wire _guard702 = wrapper_early_reset_static_par_thread_go_out;
wire _guard703 = _guard701 & _guard702;
wire _guard704 = _guard0 & _guard0;
wire _guard705 = signal_reg_out;
wire _guard706 = ~_guard705;
wire _guard707 = _guard704 & _guard706;
wire _guard708 = wrapper_early_reset_bb0_000_go_out;
wire _guard709 = _guard707 & _guard708;
wire _guard710 = _guard703 | _guard709;
wire _guard711 = _guard0 & _guard0;
wire _guard712 = signal_reg_out;
wire _guard713 = ~_guard712;
wire _guard714 = _guard711 & _guard713;
wire _guard715 = wrapper_early_reset_static_par_thread0_go_out;
wire _guard716 = _guard714 & _guard715;
wire _guard717 = _guard710 | _guard716;
wire _guard718 = _guard0 & _guard0;
wire _guard719 = signal_reg_out;
wire _guard720 = ~_guard719;
wire _guard721 = _guard718 & _guard720;
wire _guard722 = wrapper_early_reset_bb0_200_go_out;
wire _guard723 = _guard721 & _guard722;
wire _guard724 = _guard717 | _guard723;
wire _guard725 = _guard0 & _guard0;
wire _guard726 = signal_reg_out;
wire _guard727 = ~_guard726;
wire _guard728 = _guard725 & _guard727;
wire _guard729 = wrapper_early_reset_bb0_400_go_out;
wire _guard730 = _guard728 & _guard729;
wire _guard731 = _guard724 | _guard730;
wire _guard732 = fsm_out == 2'd1;
wire _guard733 = _guard732 & _guard0;
wire _guard734 = signal_reg_out;
wire _guard735 = ~_guard734;
wire _guard736 = _guard733 & _guard735;
wire _guard737 = wrapper_early_reset_static_seq_go_out;
wire _guard738 = _guard736 & _guard737;
wire _guard739 = _guard731 | _guard738;
wire _guard740 = _guard0 & _guard0;
wire _guard741 = signal_reg_out;
wire _guard742 = ~_guard741;
wire _guard743 = _guard740 & _guard742;
wire _guard744 = wrapper_early_reset_bb0_1400_go_out;
wire _guard745 = _guard743 & _guard744;
wire _guard746 = _guard739 | _guard745;
wire _guard747 = fsm_out == 2'd1;
wire _guard748 = _guard747 & _guard0;
wire _guard749 = signal_reg_out;
wire _guard750 = ~_guard749;
wire _guard751 = _guard748 & _guard750;
wire _guard752 = wrapper_early_reset_static_seq0_go_out;
wire _guard753 = _guard751 & _guard752;
wire _guard754 = _guard746 | _guard753;
wire _guard755 = signal_reg_out;
wire _guard756 = wrapper_early_reset_static_par_thread_go_out;
wire _guard757 = signal_reg_out;
wire _guard758 = early_reset_bb0_200_go_out;
wire _guard759 = early_reset_bb0_400_go_out;
wire _guard760 = early_reset_bb0_000_go_out;
wire _guard761 = early_reset_bb0_1400_go_out;
wire _guard762 = early_reset_bb0_1400_go_out;
wire _guard763 = early_reset_bb0_000_go_out;
wire _guard764 = early_reset_bb0_400_go_out;
wire _guard765 = early_reset_bb0_200_go_out;
wire _guard766 = _guard764 | _guard765;
wire _guard767 = while_1_arg1_reg_done;
wire _guard768 = while_1_arg0_reg_done;
wire _guard769 = _guard767 & _guard768;
wire _guard770 = wrapper_early_reset_static_par_thread0_go_out;
wire _guard771 = wrapper_early_reset_bb0_200_go_out;
wire _guard772 = signal_reg_out;
wire _guard773 = wrapper_early_reset_bb0_000_done_out;
wire _guard774 = ~_guard773;
wire _guard775 = fsm0_out == 5'd1;
wire _guard776 = _guard774 & _guard775;
wire _guard777 = tdcc_go_out;
wire _guard778 = _guard776 & _guard777;
wire _guard779 = wrapper_early_reset_bb0_000_done_out;
wire _guard780 = ~_guard779;
wire _guard781 = fsm0_out == 5'd13;
wire _guard782 = _guard780 & _guard781;
wire _guard783 = tdcc_go_out;
wire _guard784 = _guard782 & _guard783;
wire _guard785 = _guard778 | _guard784;
wire _guard786 = bb0_8_go_out;
wire _guard787 = bb0_8_go_out;
wire _guard788 = wrapper_early_reset_bb0_1400_go_out;
wire _guard789 = signal_reg_out;
wire _guard790 = wrapper_early_reset_bb0_400_done_out;
wire _guard791 = ~_guard790;
wire _guard792 = fsm0_out == 5'd5;
wire _guard793 = _guard791 & _guard792;
wire _guard794 = tdcc_go_out;
wire _guard795 = _guard793 & _guard794;
wire _guard796 = wrapper_early_reset_bb0_400_done_out;
wire _guard797 = ~_guard796;
wire _guard798 = fsm0_out == 5'd9;
wire _guard799 = _guard797 & _guard798;
wire _guard800 = tdcc_go_out;
wire _guard801 = _guard799 & _guard800;
wire _guard802 = _guard795 | _guard801;
wire _guard803 = fsm0_out == 5'd20;
wire _guard804 = assign_while_2_latch_go_out;
wire _guard805 = early_reset_static_par_thread_go_out;
wire _guard806 = _guard804 | _guard805;
wire _guard807 = fsm_out == 2'd1;
wire _guard808 = early_reset_static_seq0_go_out;
wire _guard809 = _guard807 & _guard808;
wire _guard810 = _guard806 | _guard809;
wire _guard811 = early_reset_static_par_thread_go_out;
wire _guard812 = assign_while_2_latch_go_out;
wire _guard813 = fsm_out == 2'd1;
wire _guard814 = early_reset_static_seq0_go_out;
wire _guard815 = _guard813 & _guard814;
wire _guard816 = bb0_7_done_out;
wire _guard817 = ~_guard816;
wire _guard818 = fsm0_out == 5'd6;
wire _guard819 = _guard817 & _guard818;
wire _guard820 = tdcc_go_out;
wire _guard821 = _guard819 & _guard820;
wire _guard822 = assign_while_2_latch_go_out;
wire _guard823 = invoke8_go_out;
wire _guard824 = _guard822 | _guard823;
wire _guard825 = invoke11_go_out;
wire _guard826 = _guard824 | _guard825;
wire _guard827 = early_reset_static_par_thread_go_out;
wire _guard828 = _guard826 | _guard827;
wire _guard829 = assign_while_2_latch_go_out;
wire _guard830 = invoke8_go_out;
wire _guard831 = early_reset_static_par_thread_go_out;
wire _guard832 = _guard830 | _guard831;
wire _guard833 = invoke11_go_out;
wire _guard834 = wrapper_early_reset_bb0_400_go_out;
wire _guard835 = wrapper_early_reset_static_seq_done_out;
wire _guard836 = ~_guard835;
wire _guard837 = fsm0_out == 5'd8;
wire _guard838 = _guard836 & _guard837;
wire _guard839 = tdcc_go_out;
wire _guard840 = _guard838 & _guard839;
wire _guard841 = signal_reg_out;
wire _guard842 = bb0_8_go_out;
wire _guard843 = std_compareFN_0_done;
wire _guard844 = ~_guard843;
wire _guard845 = bb0_8_go_out;
wire _guard846 = _guard844 & _guard845;
wire _guard847 = bb0_8_go_out;
wire _guard848 = bb0_8_go_out;
wire _guard849 = bb0_8_done_out;
wire _guard850 = ~_guard849;
wire _guard851 = fsm0_out == 5'd7;
wire _guard852 = _guard850 & _guard851;
wire _guard853 = tdcc_go_out;
wire _guard854 = _guard852 & _guard853;
assign std_add_5_left =
  _guard1 ? while_1_arg0_reg_out :
  _guard2 ? while_2_arg1_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard2, _guard1})) begin
    $fatal(2, "Multiple assignment to port `std_add_5.left'.");
end
end
assign std_add_5_right =
  _guard3 ? 32'd1 :
  _guard4 ? 32'd10 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard4, _guard3})) begin
    $fatal(2, "Multiple assignment to port `std_add_5.right'.");
end
end
assign unordered_port_0_reg_write_en =
  _guard5 ? std_compareFN_0_done :
  1'd0;
assign unordered_port_0_reg_clk = clk;
assign unordered_port_0_reg_reset = reset;
assign unordered_port_0_reg_in = std_compareFN_0_unordered;
assign assign_while_1_latch_go_in = _guard12;
assign wrapper_early_reset_bb0_200_done_in = _guard13;
assign done = _guard14;
assign arg_mem_1_write_data = while_2_arg2_reg_out;
assign arg_mem_0_content_en = _guard16;
assign arg_mem_0_addr0 = std_slice_3_out;
assign arg_mem_1_write_en = _guard18;
assign arg_mem_1_addr0 = std_slice_3_out;
assign arg_mem_1_content_en = _guard20;
assign fsm_write_en = _guard35;
assign fsm_clk = clk;
assign fsm_reset = reset;
assign fsm_in =
  _guard38 ? adder_out :
  _guard45 ? 2'd0 :
  _guard48 ? adder0_out :
  2'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard48, _guard45, _guard38})) begin
    $fatal(2, "Multiple assignment to port `fsm.in'.");
end
end
assign adder_left =
  _guard49 ? fsm_out :
  2'd0;
assign adder_right =
  _guard50 ? 2'd1 :
  2'd0;
assign assign_while_2_latch_done_in = _guard55;
assign invoke11_done_in = while_2_arg1_reg_done;
assign std_mux_0_cond = cmpf_0_reg_out;
assign std_mux_0_tru = arg_mem_0_read_data;
assign std_mux_0_fal = cst_0_out;
assign wrapper_early_reset_static_par_thread_go_in = _guard70;
assign while_1_arg0_reg_write_en = _guard73;
assign while_1_arg0_reg_clk = clk;
assign while_1_arg0_reg_reset = reset;
assign while_1_arg0_reg_in =
  _guard74 ? std_add_5_out :
  _guard75 ? 32'd0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard75, _guard74})) begin
    $fatal(2, "Multiple assignment to port `while_1_arg0_reg.in'.");
end
end
assign comb_reg_write_en = _guard76;
assign comb_reg_clk = clk;
assign comb_reg_reset = reset;
assign comb_reg_in =
  _guard77 ? std_slt_3_out :
  1'd0;
assign early_reset_static_par_thread0_done_in = ud1_out;
assign wrapper_early_reset_bb0_400_done_in = _guard78;
assign std_and_0_left =
  _guard79 ? compare_port_0_reg_done :
  1'd0;
assign std_and_0_right =
  _guard80 ? unordered_port_0_reg_done :
  1'd0;
assign early_reset_bb0_200_done_in = ud6_out;
assign early_reset_static_seq0_go_in = _guard81;
assign std_slice_3_in =
  _guard86 ? std_add_6_out :
  _guard91 ? while_2_arg1_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard91, _guard86})) begin
    $fatal(2, "Multiple assignment to port `std_slice_3.in'.");
end
end
assign compare_port_0_reg_write_en =
  _guard92 ? std_compareFN_0_done :
  1'd0;
assign compare_port_0_reg_clk = clk;
assign compare_port_0_reg_reset = reset;
assign compare_port_0_reg_in = std_compareFN_0_gt;
assign while_0_arg0_reg_write_en = _guard98;
assign while_0_arg0_reg_clk = clk;
assign while_0_arg0_reg_reset = reset;
assign while_0_arg0_reg_in =
  _guard99 ? 32'd0 :
  _guard102 ? std_add_6_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard102, _guard99})) begin
    $fatal(2, "Multiple assignment to port `while_0_arg0_reg.in'.");
end
end
assign comb_reg1_write_en = _guard103;
assign comb_reg1_clk = clk;
assign comb_reg1_reset = reset;
assign comb_reg1_in =
  _guard104 ? std_slt_3_out :
  1'd0;
assign assign_while_2_latch_go_in = _guard110;
assign invoke5_go_in = _guard116;
assign invoke5_done_in = while_0_arg0_reg_done;
assign invoke11_go_in = _guard122;
assign wrapper_early_reset_bb0_1400_done_in = _guard123;
assign wrapper_early_reset_static_seq0_go_in = _guard129;
assign std_or_0_left = compare_port_0_reg_out;
assign std_or_0_right = unordered_port_0_reg_out;
assign comb_reg0_write_en = _guard132;
assign comb_reg0_clk = clk;
assign comb_reg0_reset = reset;
assign comb_reg0_in =
  _guard133 ? std_slt_3_out :
  1'd0;
assign tdcc_go_in = go;
assign std_add_0_left = while_2_arg0_reg_out;
assign std_add_0_right = 32'd1;
assign while_2_arg0_reg_write_en = _guard138;
assign while_2_arg0_reg_clk = clk;
assign while_2_arg0_reg_reset = reset;
assign while_2_arg0_reg_in =
  _guard139 ? 32'd0 :
  _guard140 ? std_add_0_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard140, _guard139})) begin
    $fatal(2, "Multiple assignment to port `while_2_arg0_reg.in'.");
end
end
assign comb_reg2_write_en = _guard141;
assign comb_reg2_clk = clk;
assign comb_reg2_reset = reset;
assign comb_reg2_in =
  _guard142 ? std_slt_3_out :
  1'd0;
assign fsm0_write_en = _guard351;
assign fsm0_clk = clk;
assign fsm0_reset = reset;
assign fsm0_in =
  _guard356 ? 5'd1 :
  _guard361 ? 5'd15 :
  _guard366 ? 5'd18 :
  _guard381 ? 5'd16 :
  _guard382 ? 5'd0 :
  _guard387 ? 5'd3 :
  _guard392 ? 5'd13 :
  _guard409 ? 5'd14 :
  _guard414 ? 5'd5 :
  _guard431 ? 5'd12 :
  _guard446 ? 5'd2 :
  _guard451 ? 5'd8 :
  _guard468 ? 5'd10 :
  _guard473 ? 5'd7 :
  _guard478 ? 5'd11 :
  _guard483 ? 5'd19 :
  _guard498 ? 5'd4 :
  _guard513 ? 5'd6 :
  _guard530 ? 5'd20 :
  _guard535 ? 5'd17 :
  _guard540 ? 5'd9 :
  5'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard540, _guard535, _guard530, _guard513, _guard498, _guard483, _guard478, _guard473, _guard468, _guard451, _guard446, _guard431, _guard414, _guard409, _guard392, _guard387, _guard382, _guard381, _guard366, _guard361, _guard356})) begin
    $fatal(2, "Multiple assignment to port `fsm0.in'.");
end
end
assign invoke8_go_in = _guard546;
assign wrapper_early_reset_bb0_200_go_in = _guard559;
assign std_add_6_left =
  _guard562 ? while_0_arg0_reg_out :
  _guard569 ? while_1_arg1_reg_out :
  _guard570 ? while_2_arg2_reg_out :
  _guard571 ? while_2_arg1_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard571, _guard570, _guard569, _guard562})) begin
    $fatal(2, "Multiple assignment to port `std_add_6.left'.");
end
end
assign std_add_6_right =
  _guard576 ? while_0_arg0_reg_out :
  _guard581 ? 32'd1 :
  _guard582 ? 32'd10 :
  _guard583 ? 32'd100 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard583, _guard582, _guard581, _guard576})) begin
    $fatal(2, "Multiple assignment to port `std_add_6.right'.");
end
end
assign while_1_arg1_reg_write_en = _guard586;
assign while_1_arg1_reg_clk = clk;
assign while_1_arg1_reg_reset = reset;
assign while_1_arg1_reg_in =
  _guard587 ? std_add_6_out :
  _guard588 ? while_2_arg2_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard588, _guard587})) begin
    $fatal(2, "Multiple assignment to port `while_1_arg1_reg.in'.");
end
end
assign bb0_17_go_in = _guard594;
assign invoke8_done_in = while_2_arg1_reg_done;
assign early_reset_bb0_000_go_in = _guard595;
assign wrapper_early_reset_static_seq_done_in = _guard596;
assign wrapper_early_reset_bb0_1400_go_in = _guard609;
assign adder0_left =
  _guard610 ? fsm_out :
  2'd0;
assign adder0_right =
  _guard611 ? 2'd1 :
  2'd0;
assign bb0_7_done_in = arg_mem_0_done;
assign bb0_17_done_in = arg_mem_1_done;
assign early_reset_static_seq_go_in = _guard612;
assign wrapper_early_reset_static_par_thread0_go_in = _guard618;
assign mem_0_write_en = _guard621;
assign mem_0_clk = clk;
assign mem_0_addr0 = std_slice_3_out;
assign mem_0_content_en = _guard635;
assign mem_0_reset = reset;
assign mem_0_write_data = std_mux_0_out;
assign signal_reg_write_en = _guard697;
assign signal_reg_clk = clk;
assign signal_reg_reset = reset;
assign signal_reg_in =
  _guard754 ? 1'd1 :
  _guard755 ? 1'd0 :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard755, _guard754})) begin
    $fatal(2, "Multiple assignment to port `signal_reg.in'.");
end
end
assign bb0_8_done_in = cmpf_0_reg_done;
assign early_reset_static_par_thread_go_in = _guard756;
assign wrapper_early_reset_static_par_thread0_done_in = _guard757;
assign std_slt_3_left =
  _guard758 ? while_1_arg0_reg_out :
  _guard759 ? while_0_arg0_reg_out :
  _guard760 ? while_2_arg0_reg_out :
  _guard761 ? while_2_arg1_reg_out :
  32'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard761, _guard760, _guard759, _guard758})) begin
    $fatal(2, "Multiple assignment to port `std_slt_3.left'.");
end
end
assign std_slt_3_right =
  _guard762 ? 32'd300 :
  _guard763 ? 32'd3 :
  _guard766 ? 32'd10 :
  32'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard766, _guard763, _guard762})) begin
    $fatal(2, "Multiple assignment to port `std_slt_3.right'.");
end
end
assign assign_while_1_latch_done_in = _guard769;
assign early_reset_static_par_thread0_go_in = _guard770;
assign early_reset_bb0_200_go_in = _guard771;
assign wrapper_early_reset_static_par_thread_done_in = _guard772;
assign wrapper_early_reset_bb0_000_go_in = _guard785;
assign cmpf_0_reg_write_en =
  _guard786 ? std_and_0_out :
  1'd0;
assign cmpf_0_reg_clk = clk;
assign cmpf_0_reg_reset = reset;
assign cmpf_0_reg_in = std_or_0_out;
assign early_reset_static_seq_done_in = ud4_out;
assign early_reset_bb0_000_done_in = ud7_out;
assign early_reset_bb0_1400_go_in = _guard788;
assign wrapper_early_reset_bb0_000_done_in = _guard789;
assign wrapper_early_reset_bb0_400_go_in = _guard802;
assign tdcc_done_in = _guard803;
assign while_2_arg2_reg_write_en = _guard810;
assign while_2_arg2_reg_clk = clk;
assign while_2_arg2_reg_reset = reset;
assign while_2_arg2_reg_in =
  _guard811 ? 32'd0 :
  _guard812 ? std_add_6_out :
  _guard815 ? mem_0_read_data :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard815, _guard812, _guard811})) begin
    $fatal(2, "Multiple assignment to port `while_2_arg2_reg.in'.");
end
end
assign bb0_7_go_in = _guard821;
assign early_reset_bb0_400_done_in = ud5_out;
assign while_2_arg1_reg_write_en = _guard828;
assign while_2_arg1_reg_clk = clk;
assign while_2_arg1_reg_reset = reset;
assign while_2_arg1_reg_in =
  _guard829 ? std_add_5_out :
  _guard832 ? 32'd0 :
  _guard833 ? std_add_6_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard833, _guard832, _guard829})) begin
    $fatal(2, "Multiple assignment to port `while_2_arg1_reg.in'.");
end
end
assign early_reset_static_par_thread_done_in = ud_out;
assign early_reset_bb0_400_go_in = _guard834;
assign early_reset_static_seq0_done_in = ud9_out;
assign wrapper_early_reset_static_seq_go_in = _guard840;
assign wrapper_early_reset_static_seq0_done_in = _guard841;
assign std_compareFN_0_clk = clk;
assign std_compareFN_0_left =
  _guard842 ? arg_mem_0_read_data :
  32'd0;
assign std_compareFN_0_reset = reset;
assign std_compareFN_0_go = _guard846;
assign std_compareFN_0_signaling = _guard847;
assign std_compareFN_0_right =
  _guard848 ? cst_0_out :
  32'd0;
assign bb0_8_go_in = _guard854;
assign early_reset_bb0_1400_done_in = ud10_out;
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
logic [31:0] std_slt_3_left;
logic [31:0] std_slt_3_right;
logic std_slt_3_out;
logic [31:0] std_add_6_left;
logic [31:0] std_add_6_right;
logic [31:0] std_add_6_out;
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
logic [31:0] std_add_0_left;
logic [31:0] std_add_0_right;
logic [31:0] std_add_0_out;
logic [31:0] while_2_arg2_reg_in;
logic while_2_arg2_reg_write_en;
logic while_2_arg2_reg_clk;
logic while_2_arg2_reg_reset;
logic [31:0] while_2_arg2_reg_out;
logic while_2_arg2_reg_done;
logic [31:0] while_2_arg1_reg_in;
logic while_2_arg1_reg_write_en;
logic while_2_arg1_reg_clk;
logic while_2_arg1_reg_reset;
logic [31:0] while_2_arg1_reg_out;
logic while_2_arg1_reg_done;
logic [31:0] while_2_arg0_reg_in;
logic while_2_arg0_reg_write_en;
logic while_2_arg0_reg_clk;
logic while_2_arg0_reg_reset;
logic [31:0] while_2_arg0_reg_out;
logic while_2_arg0_reg_done;
logic [31:0] while_1_arg1_reg_in;
logic while_1_arg1_reg_write_en;
logic while_1_arg1_reg_clk;
logic while_1_arg1_reg_reset;
logic [31:0] while_1_arg1_reg_out;
logic while_1_arg1_reg_done;
logic [31:0] while_1_arg0_reg_in;
logic while_1_arg0_reg_write_en;
logic while_1_arg0_reg_clk;
logic while_1_arg0_reg_reset;
logic [31:0] while_1_arg0_reg_out;
logic while_1_arg0_reg_done;
logic [31:0] while_0_arg0_reg_in;
logic while_0_arg0_reg_write_en;
logic while_0_arg0_reg_clk;
logic while_0_arg0_reg_reset;
logic [31:0] while_0_arg0_reg_out;
logic while_0_arg0_reg_done;
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
logic comb_reg_in;
logic comb_reg_write_en;
logic comb_reg_clk;
logic comb_reg_reset;
logic comb_reg_out;
logic comb_reg_done;
logic comb_reg0_in;
logic comb_reg0_write_en;
logic comb_reg0_clk;
logic comb_reg0_reset;
logic comb_reg0_out;
logic comb_reg0_done;
logic comb_reg1_in;
logic comb_reg1_write_en;
logic comb_reg1_clk;
logic comb_reg1_reset;
logic comb_reg1_out;
logic comb_reg1_done;
logic comb_reg2_in;
logic comb_reg2_write_en;
logic comb_reg2_clk;
logic comb_reg2_reset;
logic comb_reg2_out;
logic comb_reg2_done;
logic ud_out;
logic ud1_out;
logic ud4_out;
logic ud5_out;
logic ud6_out;
logic ud8_out;
logic signal_reg_in;
logic signal_reg_write_en;
logic signal_reg_clk;
logic signal_reg_reset;
logic signal_reg_out;
logic signal_reg_done;
logic signal_reg0_in;
logic signal_reg0_write_en;
logic signal_reg0_clk;
logic signal_reg0_reset;
logic signal_reg0_out;
logic signal_reg0_done;
logic signal_reg1_in;
logic signal_reg1_write_en;
logic signal_reg1_clk;
logic signal_reg1_reset;
logic signal_reg1_out;
logic signal_reg1_done;
logic signal_reg2_in;
logic signal_reg2_write_en;
logic signal_reg2_clk;
logic signal_reg2_reset;
logic signal_reg2_out;
logic signal_reg2_done;
logic signal_reg3_in;
logic signal_reg3_write_en;
logic signal_reg3_clk;
logic signal_reg3_reset;
logic signal_reg3_out;
logic signal_reg3_done;
logic signal_reg4_in;
logic signal_reg4_write_en;
logic signal_reg4_clk;
logic signal_reg4_reset;
logic signal_reg4_out;
logic signal_reg4_done;
logic [4:0] fsm_in;
logic fsm_write_en;
logic fsm_clk;
logic fsm_reset;
logic [4:0] fsm_out;
logic fsm_done;
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
logic bb0_10_go_in;
logic bb0_10_go_out;
logic bb0_10_done_in;
logic bb0_10_done_out;
logic assign_while_1_latch_go_in;
logic assign_while_1_latch_go_out;
logic assign_while_1_latch_done_in;
logic assign_while_1_latch_done_out;
logic assign_while_2_latch_go_in;
logic assign_while_2_latch_go_out;
logic assign_while_2_latch_done_in;
logic assign_while_2_latch_done_out;
logic bb0_16_go_in;
logic bb0_16_go_out;
logic bb0_16_done_in;
logic bb0_16_done_out;
logic bb0_17_go_in;
logic bb0_17_go_out;
logic bb0_17_done_in;
logic bb0_17_done_out;
logic invoke5_go_in;
logic invoke5_go_out;
logic invoke5_done_in;
logic invoke5_done_out;
logic invoke6_go_in;
logic invoke6_go_out;
logic invoke6_done_in;
logic invoke6_done_out;
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
logic early_reset_static_par_thread_go_in;
logic early_reset_static_par_thread_go_out;
logic early_reset_static_par_thread_done_in;
logic early_reset_static_par_thread_done_out;
logic early_reset_static_par_thread0_go_in;
logic early_reset_static_par_thread0_go_out;
logic early_reset_static_par_thread0_done_in;
logic early_reset_static_par_thread0_done_out;
logic early_reset_bb0_400_go_in;
logic early_reset_bb0_400_go_out;
logic early_reset_bb0_400_done_in;
logic early_reset_bb0_400_done_out;
logic early_reset_bb0_200_go_in;
logic early_reset_bb0_200_go_out;
logic early_reset_bb0_200_done_in;
logic early_reset_bb0_200_done_out;
logic early_reset_bb0_000_go_in;
logic early_reset_bb0_000_go_out;
logic early_reset_bb0_000_done_in;
logic early_reset_bb0_000_done_out;
logic early_reset_bb0_1400_go_in;
logic early_reset_bb0_1400_go_out;
logic early_reset_bb0_1400_done_in;
logic early_reset_bb0_1400_done_out;
logic wrapper_early_reset_static_par_thread_go_in;
logic wrapper_early_reset_static_par_thread_go_out;
logic wrapper_early_reset_static_par_thread_done_in;
logic wrapper_early_reset_static_par_thread_done_out;
logic wrapper_early_reset_bb0_000_go_in;
logic wrapper_early_reset_bb0_000_go_out;
logic wrapper_early_reset_bb0_000_done_in;
logic wrapper_early_reset_bb0_000_done_out;
logic wrapper_early_reset_static_par_thread0_go_in;
logic wrapper_early_reset_static_par_thread0_go_out;
logic wrapper_early_reset_static_par_thread0_done_in;
logic wrapper_early_reset_static_par_thread0_done_out;
logic wrapper_early_reset_bb0_200_go_in;
logic wrapper_early_reset_bb0_200_go_out;
logic wrapper_early_reset_bb0_200_done_in;
logic wrapper_early_reset_bb0_200_done_out;
logic wrapper_early_reset_bb0_400_go_in;
logic wrapper_early_reset_bb0_400_go_out;
logic wrapper_early_reset_bb0_400_done_in;
logic wrapper_early_reset_bb0_400_done_out;
logic wrapper_early_reset_bb0_1400_go_in;
logic wrapper_early_reset_bb0_1400_go_out;
logic wrapper_early_reset_bb0_1400_done_in;
logic wrapper_early_reset_bb0_1400_done_out;
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
std_slt # (
    .WIDTH(32)
) std_slt_3 (
    .left(std_slt_3_left),
    .out(std_slt_3_out),
    .right(std_slt_3_right)
);
std_add # (
    .WIDTH(32)
) std_add_6 (
    .left(std_add_6_left),
    .out(std_add_6_out),
    .right(std_add_6_right)
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
std_add # (
    .WIDTH(32)
) std_add_0 (
    .left(std_add_0_left),
    .out(std_add_0_out),
    .right(std_add_0_right)
);
std_reg # (
    .WIDTH(32)
) while_2_arg2_reg (
    .clk(while_2_arg2_reg_clk),
    .done(while_2_arg2_reg_done),
    .in(while_2_arg2_reg_in),
    .out(while_2_arg2_reg_out),
    .reset(while_2_arg2_reg_reset),
    .write_en(while_2_arg2_reg_write_en)
);
std_reg # (
    .WIDTH(32)
) while_2_arg1_reg (
    .clk(while_2_arg1_reg_clk),
    .done(while_2_arg1_reg_done),
    .in(while_2_arg1_reg_in),
    .out(while_2_arg1_reg_out),
    .reset(while_2_arg1_reg_reset),
    .write_en(while_2_arg1_reg_write_en)
);
std_reg # (
    .WIDTH(32)
) while_2_arg0_reg (
    .clk(while_2_arg0_reg_clk),
    .done(while_2_arg0_reg_done),
    .in(while_2_arg0_reg_in),
    .out(while_2_arg0_reg_out),
    .reset(while_2_arg0_reg_reset),
    .write_en(while_2_arg0_reg_write_en)
);
std_reg # (
    .WIDTH(32)
) while_1_arg1_reg (
    .clk(while_1_arg1_reg_clk),
    .done(while_1_arg1_reg_done),
    .in(while_1_arg1_reg_in),
    .out(while_1_arg1_reg_out),
    .reset(while_1_arg1_reg_reset),
    .write_en(while_1_arg1_reg_write_en)
);
std_reg # (
    .WIDTH(32)
) while_1_arg0_reg (
    .clk(while_1_arg0_reg_clk),
    .done(while_1_arg0_reg_done),
    .in(while_1_arg0_reg_in),
    .out(while_1_arg0_reg_out),
    .reset(while_1_arg0_reg_reset),
    .write_en(while_1_arg0_reg_write_en)
);
std_reg # (
    .WIDTH(32)
) while_0_arg0_reg (
    .clk(while_0_arg0_reg_clk),
    .done(while_0_arg0_reg_done),
    .in(while_0_arg0_reg_in),
    .out(while_0_arg0_reg_out),
    .reset(while_0_arg0_reg_reset),
    .write_en(while_0_arg0_reg_write_en)
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
    .WIDTH(1)
) comb_reg (
    .clk(comb_reg_clk),
    .done(comb_reg_done),
    .in(comb_reg_in),
    .out(comb_reg_out),
    .reset(comb_reg_reset),
    .write_en(comb_reg_write_en)
);
std_reg # (
    .WIDTH(1)
) comb_reg0 (
    .clk(comb_reg0_clk),
    .done(comb_reg0_done),
    .in(comb_reg0_in),
    .out(comb_reg0_out),
    .reset(comb_reg0_reset),
    .write_en(comb_reg0_write_en)
);
std_reg # (
    .WIDTH(1)
) comb_reg1 (
    .clk(comb_reg1_clk),
    .done(comb_reg1_done),
    .in(comb_reg1_in),
    .out(comb_reg1_out),
    .reset(comb_reg1_reset),
    .write_en(comb_reg1_write_en)
);
std_reg # (
    .WIDTH(1)
) comb_reg2 (
    .clk(comb_reg2_clk),
    .done(comb_reg2_done),
    .in(comb_reg2_in),
    .out(comb_reg2_out),
    .reset(comb_reg2_reset),
    .write_en(comb_reg2_write_en)
);
undef # (
    .WIDTH(1)
) ud (
    .out(ud_out)
);
undef # (
    .WIDTH(1)
) ud1 (
    .out(ud1_out)
);
undef # (
    .WIDTH(1)
) ud4 (
    .out(ud4_out)
);
undef # (
    .WIDTH(1)
) ud5 (
    .out(ud5_out)
);
undef # (
    .WIDTH(1)
) ud6 (
    .out(ud6_out)
);
undef # (
    .WIDTH(1)
) ud8 (
    .out(ud8_out)
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
    .WIDTH(1)
) signal_reg0 (
    .clk(signal_reg0_clk),
    .done(signal_reg0_done),
    .in(signal_reg0_in),
    .out(signal_reg0_out),
    .reset(signal_reg0_reset),
    .write_en(signal_reg0_write_en)
);
std_reg # (
    .WIDTH(1)
) signal_reg1 (
    .clk(signal_reg1_clk),
    .done(signal_reg1_done),
    .in(signal_reg1_in),
    .out(signal_reg1_out),
    .reset(signal_reg1_reset),
    .write_en(signal_reg1_write_en)
);
std_reg # (
    .WIDTH(1)
) signal_reg2 (
    .clk(signal_reg2_clk),
    .done(signal_reg2_done),
    .in(signal_reg2_in),
    .out(signal_reg2_out),
    .reset(signal_reg2_reset),
    .write_en(signal_reg2_write_en)
);
std_reg # (
    .WIDTH(1)
) signal_reg3 (
    .clk(signal_reg3_clk),
    .done(signal_reg3_done),
    .in(signal_reg3_in),
    .out(signal_reg3_out),
    .reset(signal_reg3_reset),
    .write_en(signal_reg3_write_en)
);
std_reg # (
    .WIDTH(1)
) signal_reg4 (
    .clk(signal_reg4_clk),
    .done(signal_reg4_done),
    .in(signal_reg4_in),
    .out(signal_reg4_out),
    .reset(signal_reg4_reset),
    .write_en(signal_reg4_write_en)
);
std_reg # (
    .WIDTH(5)
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
) bb0_10_go (
    .in(bb0_10_go_in),
    .out(bb0_10_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_10_done (
    .in(bb0_10_done_in),
    .out(bb0_10_done_out)
);
std_wire # (
    .WIDTH(1)
) assign_while_1_latch_go (
    .in(assign_while_1_latch_go_in),
    .out(assign_while_1_latch_go_out)
);
std_wire # (
    .WIDTH(1)
) assign_while_1_latch_done (
    .in(assign_while_1_latch_done_in),
    .out(assign_while_1_latch_done_out)
);
std_wire # (
    .WIDTH(1)
) assign_while_2_latch_go (
    .in(assign_while_2_latch_go_in),
    .out(assign_while_2_latch_go_out)
);
std_wire # (
    .WIDTH(1)
) assign_while_2_latch_done (
    .in(assign_while_2_latch_done_in),
    .out(assign_while_2_latch_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_16_go (
    .in(bb0_16_go_in),
    .out(bb0_16_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_16_done (
    .in(bb0_16_done_in),
    .out(bb0_16_done_out)
);
std_wire # (
    .WIDTH(1)
) bb0_17_go (
    .in(bb0_17_go_in),
    .out(bb0_17_go_out)
);
std_wire # (
    .WIDTH(1)
) bb0_17_done (
    .in(bb0_17_done_in),
    .out(bb0_17_done_out)
);
std_wire # (
    .WIDTH(1)
) invoke5_go (
    .in(invoke5_go_in),
    .out(invoke5_go_out)
);
std_wire # (
    .WIDTH(1)
) invoke5_done (
    .in(invoke5_done_in),
    .out(invoke5_done_out)
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
) early_reset_static_par_thread_go (
    .in(early_reset_static_par_thread_go_in),
    .out(early_reset_static_par_thread_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_static_par_thread_done (
    .in(early_reset_static_par_thread_done_in),
    .out(early_reset_static_par_thread_done_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_static_par_thread0_go (
    .in(early_reset_static_par_thread0_go_in),
    .out(early_reset_static_par_thread0_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_static_par_thread0_done (
    .in(early_reset_static_par_thread0_done_in),
    .out(early_reset_static_par_thread0_done_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_400_go (
    .in(early_reset_bb0_400_go_in),
    .out(early_reset_bb0_400_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_400_done (
    .in(early_reset_bb0_400_done_in),
    .out(early_reset_bb0_400_done_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_200_go (
    .in(early_reset_bb0_200_go_in),
    .out(early_reset_bb0_200_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_200_done (
    .in(early_reset_bb0_200_done_in),
    .out(early_reset_bb0_200_done_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_000_go (
    .in(early_reset_bb0_000_go_in),
    .out(early_reset_bb0_000_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_000_done (
    .in(early_reset_bb0_000_done_in),
    .out(early_reset_bb0_000_done_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_1400_go (
    .in(early_reset_bb0_1400_go_in),
    .out(early_reset_bb0_1400_go_out)
);
std_wire # (
    .WIDTH(1)
) early_reset_bb0_1400_done (
    .in(early_reset_bb0_1400_done_in),
    .out(early_reset_bb0_1400_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_par_thread_go (
    .in(wrapper_early_reset_static_par_thread_go_in),
    .out(wrapper_early_reset_static_par_thread_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_par_thread_done (
    .in(wrapper_early_reset_static_par_thread_done_in),
    .out(wrapper_early_reset_static_par_thread_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_000_go (
    .in(wrapper_early_reset_bb0_000_go_in),
    .out(wrapper_early_reset_bb0_000_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_000_done (
    .in(wrapper_early_reset_bb0_000_done_in),
    .out(wrapper_early_reset_bb0_000_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_par_thread0_go (
    .in(wrapper_early_reset_static_par_thread0_go_in),
    .out(wrapper_early_reset_static_par_thread0_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_static_par_thread0_done (
    .in(wrapper_early_reset_static_par_thread0_done_in),
    .out(wrapper_early_reset_static_par_thread0_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_200_go (
    .in(wrapper_early_reset_bb0_200_go_in),
    .out(wrapper_early_reset_bb0_200_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_200_done (
    .in(wrapper_early_reset_bb0_200_done_in),
    .out(wrapper_early_reset_bb0_200_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_400_go (
    .in(wrapper_early_reset_bb0_400_go_in),
    .out(wrapper_early_reset_bb0_400_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_400_done (
    .in(wrapper_early_reset_bb0_400_done_in),
    .out(wrapper_early_reset_bb0_400_done_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_1400_go (
    .in(wrapper_early_reset_bb0_1400_go_in),
    .out(wrapper_early_reset_bb0_1400_go_out)
);
std_wire # (
    .WIDTH(1)
) wrapper_early_reset_bb0_1400_done (
    .in(wrapper_early_reset_bb0_1400_done_in),
    .out(wrapper_early_reset_bb0_1400_done_out)
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
wire _guard1 = assign_while_1_latch_go_out;
wire _guard2 = assign_while_2_latch_go_out;
wire _guard3 = assign_while_1_latch_go_out;
wire _guard4 = assign_while_2_latch_go_out;
wire _guard5 = bb0_16_go_out;
wire _guard6 = bb0_17_go_out;
wire _guard7 = _guard5 | _guard6;
wire _guard8 = bb0_7_go_out;
wire _guard9 = bb0_8_go_out;
wire _guard10 = _guard8 | _guard9;
wire _guard11 = bb0_10_go_out;
wire _guard12 = _guard10 | _guard11;
wire _guard13 = signal_reg1_out;
wire _guard14 = _guard0 & _guard0;
wire _guard15 = signal_reg1_out;
wire _guard16 = ~_guard15;
wire _guard17 = _guard14 & _guard16;
wire _guard18 = wrapper_early_reset_static_par_thread0_go_out;
wire _guard19 = _guard17 & _guard18;
wire _guard20 = _guard13 | _guard19;
wire _guard21 = _guard0 & _guard0;
wire _guard22 = signal_reg1_out;
wire _guard23 = ~_guard22;
wire _guard24 = _guard21 & _guard23;
wire _guard25 = wrapper_early_reset_static_par_thread0_go_out;
wire _guard26 = _guard24 & _guard25;
wire _guard27 = signal_reg1_out;
wire _guard28 = assign_while_1_latch_done_out;
wire _guard29 = ~_guard28;
wire _guard30 = fsm_out == 5'd12;
wire _guard31 = _guard29 & _guard30;
wire _guard32 = tdcc_go_out;
wire _guard33 = _guard31 & _guard32;
wire _guard34 = signal_reg2_out;
wire _guard35 = invoke9_done_out;
wire _guard36 = ~_guard35;
wire _guard37 = fsm_out == 5'd21;
wire _guard38 = _guard36 & _guard37;
wire _guard39 = tdcc_go_out;
wire _guard40 = _guard38 & _guard39;
wire _guard41 = tdcc_done_out;
wire _guard42 = bb0_7_go_out;
wire _guard43 = invoke7_go_out;
wire _guard44 = bb0_10_go_out;
wire _guard45 = invoke7_go_out;
wire _guard46 = bb0_10_go_out;
wire _guard47 = bb0_7_go_out;
wire _guard48 = bb0_16_go_out;
wire _guard49 = invoke7_go_out;
wire _guard50 = bb0_10_go_out;
wire _guard51 = invoke7_go_out;
wire _guard52 = bb0_10_go_out;
wire _guard53 = bb0_17_go_out;
wire _guard54 = bb0_17_go_out;
wire _guard55 = bb0_16_go_out;
wire _guard56 = invoke7_go_out;
wire _guard57 = bb0_17_go_out;
wire _guard58 = invoke7_go_out;
wire _guard59 = bb0_8_go_out;
wire _guard60 = bb0_8_go_out;
wire _guard61 = bb0_17_go_out;
wire _guard62 = fsm_out == 5'd23;
wire _guard63 = fsm_out == 5'd0;
wire _guard64 = wrapper_early_reset_static_par_thread_done_out;
wire _guard65 = _guard63 & _guard64;
wire _guard66 = tdcc_go_out;
wire _guard67 = _guard65 & _guard66;
wire _guard68 = _guard62 | _guard67;
wire _guard69 = fsm_out == 5'd1;
wire _guard70 = wrapper_early_reset_bb0_000_done_out;
wire _guard71 = comb_reg_out;
wire _guard72 = _guard70 & _guard71;
wire _guard73 = _guard69 & _guard72;
wire _guard74 = tdcc_go_out;
wire _guard75 = _guard73 & _guard74;
wire _guard76 = _guard68 | _guard75;
wire _guard77 = fsm_out == 5'd15;
wire _guard78 = wrapper_early_reset_bb0_000_done_out;
wire _guard79 = comb_reg_out;
wire _guard80 = _guard78 & _guard79;
wire _guard81 = _guard77 & _guard80;
wire _guard82 = tdcc_go_out;
wire _guard83 = _guard81 & _guard82;
wire _guard84 = _guard76 | _guard83;
wire _guard85 = fsm_out == 5'd2;
wire _guard86 = wrapper_early_reset_static_par_thread0_done_out;
wire _guard87 = _guard85 & _guard86;
wire _guard88 = tdcc_go_out;
wire _guard89 = _guard87 & _guard88;
wire _guard90 = _guard84 | _guard89;
wire _guard91 = fsm_out == 5'd3;
wire _guard92 = wrapper_early_reset_bb0_200_done_out;
wire _guard93 = comb_reg0_out;
wire _guard94 = _guard92 & _guard93;
wire _guard95 = _guard91 & _guard94;
wire _guard96 = tdcc_go_out;
wire _guard97 = _guard95 & _guard96;
wire _guard98 = _guard90 | _guard97;
wire _guard99 = fsm_out == 5'd13;
wire _guard100 = wrapper_early_reset_bb0_200_done_out;
wire _guard101 = comb_reg0_out;
wire _guard102 = _guard100 & _guard101;
wire _guard103 = _guard99 & _guard102;
wire _guard104 = tdcc_go_out;
wire _guard105 = _guard103 & _guard104;
wire _guard106 = _guard98 | _guard105;
wire _guard107 = fsm_out == 5'd4;
wire _guard108 = invoke5_done_out;
wire _guard109 = _guard107 & _guard108;
wire _guard110 = tdcc_go_out;
wire _guard111 = _guard109 & _guard110;
wire _guard112 = _guard106 | _guard111;
wire _guard113 = fsm_out == 5'd5;
wire _guard114 = wrapper_early_reset_bb0_400_done_out;
wire _guard115 = comb_reg1_out;
wire _guard116 = _guard114 & _guard115;
wire _guard117 = _guard113 & _guard116;
wire _guard118 = tdcc_go_out;
wire _guard119 = _guard117 & _guard118;
wire _guard120 = _guard112 | _guard119;
wire _guard121 = fsm_out == 5'd11;
wire _guard122 = wrapper_early_reset_bb0_400_done_out;
wire _guard123 = comb_reg1_out;
wire _guard124 = _guard122 & _guard123;
wire _guard125 = _guard121 & _guard124;
wire _guard126 = tdcc_go_out;
wire _guard127 = _guard125 & _guard126;
wire _guard128 = _guard120 | _guard127;
wire _guard129 = fsm_out == 5'd6;
wire _guard130 = bb0_7_done_out;
wire _guard131 = _guard129 & _guard130;
wire _guard132 = tdcc_go_out;
wire _guard133 = _guard131 & _guard132;
wire _guard134 = _guard128 | _guard133;
wire _guard135 = fsm_out == 5'd7;
wire _guard136 = bb0_8_done_out;
wire _guard137 = _guard135 & _guard136;
wire _guard138 = tdcc_go_out;
wire _guard139 = _guard137 & _guard138;
wire _guard140 = _guard134 | _guard139;
wire _guard141 = fsm_out == 5'd8;
wire _guard142 = bb0_9_done_out;
wire _guard143 = _guard141 & _guard142;
wire _guard144 = tdcc_go_out;
wire _guard145 = _guard143 & _guard144;
wire _guard146 = _guard140 | _guard145;
wire _guard147 = fsm_out == 5'd9;
wire _guard148 = bb0_10_done_out;
wire _guard149 = _guard147 & _guard148;
wire _guard150 = tdcc_go_out;
wire _guard151 = _guard149 & _guard150;
wire _guard152 = _guard146 | _guard151;
wire _guard153 = fsm_out == 5'd10;
wire _guard154 = invoke6_done_out;
wire _guard155 = _guard153 & _guard154;
wire _guard156 = tdcc_go_out;
wire _guard157 = _guard155 & _guard156;
wire _guard158 = _guard152 | _guard157;
wire _guard159 = fsm_out == 5'd5;
wire _guard160 = wrapper_early_reset_bb0_400_done_out;
wire _guard161 = comb_reg1_out;
wire _guard162 = ~_guard161;
wire _guard163 = _guard160 & _guard162;
wire _guard164 = _guard159 & _guard163;
wire _guard165 = tdcc_go_out;
wire _guard166 = _guard164 & _guard165;
wire _guard167 = _guard158 | _guard166;
wire _guard168 = fsm_out == 5'd11;
wire _guard169 = wrapper_early_reset_bb0_400_done_out;
wire _guard170 = comb_reg1_out;
wire _guard171 = ~_guard170;
wire _guard172 = _guard169 & _guard171;
wire _guard173 = _guard168 & _guard172;
wire _guard174 = tdcc_go_out;
wire _guard175 = _guard173 & _guard174;
wire _guard176 = _guard167 | _guard175;
wire _guard177 = fsm_out == 5'd12;
wire _guard178 = assign_while_1_latch_done_out;
wire _guard179 = _guard177 & _guard178;
wire _guard180 = tdcc_go_out;
wire _guard181 = _guard179 & _guard180;
wire _guard182 = _guard176 | _guard181;
wire _guard183 = fsm_out == 5'd3;
wire _guard184 = wrapper_early_reset_bb0_200_done_out;
wire _guard185 = comb_reg0_out;
wire _guard186 = ~_guard185;
wire _guard187 = _guard184 & _guard186;
wire _guard188 = _guard183 & _guard187;
wire _guard189 = tdcc_go_out;
wire _guard190 = _guard188 & _guard189;
wire _guard191 = _guard182 | _guard190;
wire _guard192 = fsm_out == 5'd13;
wire _guard193 = wrapper_early_reset_bb0_200_done_out;
wire _guard194 = comb_reg0_out;
wire _guard195 = ~_guard194;
wire _guard196 = _guard193 & _guard195;
wire _guard197 = _guard192 & _guard196;
wire _guard198 = tdcc_go_out;
wire _guard199 = _guard197 & _guard198;
wire _guard200 = _guard191 | _guard199;
wire _guard201 = fsm_out == 5'd14;
wire _guard202 = assign_while_2_latch_done_out;
wire _guard203 = _guard201 & _guard202;
wire _guard204 = tdcc_go_out;
wire _guard205 = _guard203 & _guard204;
wire _guard206 = _guard200 | _guard205;
wire _guard207 = fsm_out == 5'd1;
wire _guard208 = wrapper_early_reset_bb0_000_done_out;
wire _guard209 = comb_reg_out;
wire _guard210 = ~_guard209;
wire _guard211 = _guard208 & _guard210;
wire _guard212 = _guard207 & _guard211;
wire _guard213 = tdcc_go_out;
wire _guard214 = _guard212 & _guard213;
wire _guard215 = _guard206 | _guard214;
wire _guard216 = fsm_out == 5'd15;
wire _guard217 = wrapper_early_reset_bb0_000_done_out;
wire _guard218 = comb_reg_out;
wire _guard219 = ~_guard218;
wire _guard220 = _guard217 & _guard219;
wire _guard221 = _guard216 & _guard220;
wire _guard222 = tdcc_go_out;
wire _guard223 = _guard221 & _guard222;
wire _guard224 = _guard215 | _guard223;
wire _guard225 = fsm_out == 5'd16;
wire _guard226 = invoke7_done_out;
wire _guard227 = _guard225 & _guard226;
wire _guard228 = tdcc_go_out;
wire _guard229 = _guard227 & _guard228;
wire _guard230 = _guard224 | _guard229;
wire _guard231 = fsm_out == 5'd17;
wire _guard232 = invoke8_done_out;
wire _guard233 = _guard231 & _guard232;
wire _guard234 = tdcc_go_out;
wire _guard235 = _guard233 & _guard234;
wire _guard236 = _guard230 | _guard235;
wire _guard237 = fsm_out == 5'd18;
wire _guard238 = wrapper_early_reset_bb0_1400_done_out;
wire _guard239 = comb_reg2_out;
wire _guard240 = _guard238 & _guard239;
wire _guard241 = _guard237 & _guard240;
wire _guard242 = tdcc_go_out;
wire _guard243 = _guard241 & _guard242;
wire _guard244 = _guard236 | _guard243;
wire _guard245 = fsm_out == 5'd22;
wire _guard246 = wrapper_early_reset_bb0_1400_done_out;
wire _guard247 = comb_reg2_out;
wire _guard248 = _guard246 & _guard247;
wire _guard249 = _guard245 & _guard248;
wire _guard250 = tdcc_go_out;
wire _guard251 = _guard249 & _guard250;
wire _guard252 = _guard244 | _guard251;
wire _guard253 = fsm_out == 5'd19;
wire _guard254 = bb0_16_done_out;
wire _guard255 = _guard253 & _guard254;
wire _guard256 = tdcc_go_out;
wire _guard257 = _guard255 & _guard256;
wire _guard258 = _guard252 | _guard257;
wire _guard259 = fsm_out == 5'd20;
wire _guard260 = bb0_17_done_out;
wire _guard261 = _guard259 & _guard260;
wire _guard262 = tdcc_go_out;
wire _guard263 = _guard261 & _guard262;
wire _guard264 = _guard258 | _guard263;
wire _guard265 = fsm_out == 5'd21;
wire _guard266 = invoke9_done_out;
wire _guard267 = _guard265 & _guard266;
wire _guard268 = tdcc_go_out;
wire _guard269 = _guard267 & _guard268;
wire _guard270 = _guard264 | _guard269;
wire _guard271 = fsm_out == 5'd18;
wire _guard272 = wrapper_early_reset_bb0_1400_done_out;
wire _guard273 = comb_reg2_out;
wire _guard274 = ~_guard273;
wire _guard275 = _guard272 & _guard274;
wire _guard276 = _guard271 & _guard275;
wire _guard277 = tdcc_go_out;
wire _guard278 = _guard276 & _guard277;
wire _guard279 = _guard270 | _guard278;
wire _guard280 = fsm_out == 5'd22;
wire _guard281 = wrapper_early_reset_bb0_1400_done_out;
wire _guard282 = comb_reg2_out;
wire _guard283 = ~_guard282;
wire _guard284 = _guard281 & _guard283;
wire _guard285 = _guard280 & _guard284;
wire _guard286 = tdcc_go_out;
wire _guard287 = _guard285 & _guard286;
wire _guard288 = _guard279 | _guard287;
wire _guard289 = fsm_out == 5'd0;
wire _guard290 = wrapper_early_reset_static_par_thread_done_out;
wire _guard291 = _guard289 & _guard290;
wire _guard292 = tdcc_go_out;
wire _guard293 = _guard291 & _guard292;
wire _guard294 = fsm_out == 5'd14;
wire _guard295 = assign_while_2_latch_done_out;
wire _guard296 = _guard294 & _guard295;
wire _guard297 = tdcc_go_out;
wire _guard298 = _guard296 & _guard297;
wire _guard299 = fsm_out == 5'd18;
wire _guard300 = wrapper_early_reset_bb0_1400_done_out;
wire _guard301 = comb_reg2_out;
wire _guard302 = ~_guard301;
wire _guard303 = _guard300 & _guard302;
wire _guard304 = _guard299 & _guard303;
wire _guard305 = tdcc_go_out;
wire _guard306 = _guard304 & _guard305;
wire _guard307 = fsm_out == 5'd22;
wire _guard308 = wrapper_early_reset_bb0_1400_done_out;
wire _guard309 = comb_reg2_out;
wire _guard310 = ~_guard309;
wire _guard311 = _guard308 & _guard310;
wire _guard312 = _guard307 & _guard311;
wire _guard313 = tdcc_go_out;
wire _guard314 = _guard312 & _guard313;
wire _guard315 = _guard306 | _guard314;
wire _guard316 = fsm_out == 5'd17;
wire _guard317 = invoke8_done_out;
wire _guard318 = _guard316 & _guard317;
wire _guard319 = tdcc_go_out;
wire _guard320 = _guard318 & _guard319;
wire _guard321 = fsm_out == 5'd1;
wire _guard322 = wrapper_early_reset_bb0_000_done_out;
wire _guard323 = comb_reg_out;
wire _guard324 = ~_guard323;
wire _guard325 = _guard322 & _guard324;
wire _guard326 = _guard321 & _guard325;
wire _guard327 = tdcc_go_out;
wire _guard328 = _guard326 & _guard327;
wire _guard329 = fsm_out == 5'd15;
wire _guard330 = wrapper_early_reset_bb0_000_done_out;
wire _guard331 = comb_reg_out;
wire _guard332 = ~_guard331;
wire _guard333 = _guard330 & _guard332;
wire _guard334 = _guard329 & _guard333;
wire _guard335 = tdcc_go_out;
wire _guard336 = _guard334 & _guard335;
wire _guard337 = _guard328 | _guard336;
wire _guard338 = fsm_out == 5'd23;
wire _guard339 = fsm_out == 5'd2;
wire _guard340 = wrapper_early_reset_static_par_thread0_done_out;
wire _guard341 = _guard339 & _guard340;
wire _guard342 = tdcc_go_out;
wire _guard343 = _guard341 & _guard342;
wire _guard344 = fsm_out == 5'd12;
wire _guard345 = assign_while_1_latch_done_out;
wire _guard346 = _guard344 & _guard345;
wire _guard347 = tdcc_go_out;
wire _guard348 = _guard346 & _guard347;
wire _guard349 = fsm_out == 5'd3;
wire _guard350 = wrapper_early_reset_bb0_200_done_out;
wire _guard351 = comb_reg0_out;
wire _guard352 = ~_guard351;
wire _guard353 = _guard350 & _guard352;
wire _guard354 = _guard349 & _guard353;
wire _guard355 = tdcc_go_out;
wire _guard356 = _guard354 & _guard355;
wire _guard357 = fsm_out == 5'd13;
wire _guard358 = wrapper_early_reset_bb0_200_done_out;
wire _guard359 = comb_reg0_out;
wire _guard360 = ~_guard359;
wire _guard361 = _guard358 & _guard360;
wire _guard362 = _guard357 & _guard361;
wire _guard363 = tdcc_go_out;
wire _guard364 = _guard362 & _guard363;
wire _guard365 = _guard356 | _guard364;
wire _guard366 = fsm_out == 5'd4;
wire _guard367 = invoke5_done_out;
wire _guard368 = _guard366 & _guard367;
wire _guard369 = tdcc_go_out;
wire _guard370 = _guard368 & _guard369;
wire _guard371 = fsm_out == 5'd5;
wire _guard372 = wrapper_early_reset_bb0_400_done_out;
wire _guard373 = comb_reg1_out;
wire _guard374 = ~_guard373;
wire _guard375 = _guard372 & _guard374;
wire _guard376 = _guard371 & _guard375;
wire _guard377 = tdcc_go_out;
wire _guard378 = _guard376 & _guard377;
wire _guard379 = fsm_out == 5'd11;
wire _guard380 = wrapper_early_reset_bb0_400_done_out;
wire _guard381 = comb_reg1_out;
wire _guard382 = ~_guard381;
wire _guard383 = _guard380 & _guard382;
wire _guard384 = _guard379 & _guard383;
wire _guard385 = tdcc_go_out;
wire _guard386 = _guard384 & _guard385;
wire _guard387 = _guard378 | _guard386;
wire _guard388 = fsm_out == 5'd1;
wire _guard389 = wrapper_early_reset_bb0_000_done_out;
wire _guard390 = comb_reg_out;
wire _guard391 = _guard389 & _guard390;
wire _guard392 = _guard388 & _guard391;
wire _guard393 = tdcc_go_out;
wire _guard394 = _guard392 & _guard393;
wire _guard395 = fsm_out == 5'd15;
wire _guard396 = wrapper_early_reset_bb0_000_done_out;
wire _guard397 = comb_reg_out;
wire _guard398 = _guard396 & _guard397;
wire _guard399 = _guard395 & _guard398;
wire _guard400 = tdcc_go_out;
wire _guard401 = _guard399 & _guard400;
wire _guard402 = _guard394 | _guard401;
wire _guard403 = fsm_out == 5'd7;
wire _guard404 = bb0_8_done_out;
wire _guard405 = _guard403 & _guard404;
wire _guard406 = tdcc_go_out;
wire _guard407 = _guard405 & _guard406;
wire _guard408 = fsm_out == 5'd9;
wire _guard409 = bb0_10_done_out;
wire _guard410 = _guard408 & _guard409;
wire _guard411 = tdcc_go_out;
wire _guard412 = _guard410 & _guard411;
wire _guard413 = fsm_out == 5'd6;
wire _guard414 = bb0_7_done_out;
wire _guard415 = _guard413 & _guard414;
wire _guard416 = tdcc_go_out;
wire _guard417 = _guard415 & _guard416;
wire _guard418 = fsm_out == 5'd10;
wire _guard419 = invoke6_done_out;
wire _guard420 = _guard418 & _guard419;
wire _guard421 = tdcc_go_out;
wire _guard422 = _guard420 & _guard421;
wire _guard423 = fsm_out == 5'd20;
wire _guard424 = bb0_17_done_out;
wire _guard425 = _guard423 & _guard424;
wire _guard426 = tdcc_go_out;
wire _guard427 = _guard425 & _guard426;
wire _guard428 = fsm_out == 5'd18;
wire _guard429 = wrapper_early_reset_bb0_1400_done_out;
wire _guard430 = comb_reg2_out;
wire _guard431 = _guard429 & _guard430;
wire _guard432 = _guard428 & _guard431;
wire _guard433 = tdcc_go_out;
wire _guard434 = _guard432 & _guard433;
wire _guard435 = fsm_out == 5'd22;
wire _guard436 = wrapper_early_reset_bb0_1400_done_out;
wire _guard437 = comb_reg2_out;
wire _guard438 = _guard436 & _guard437;
wire _guard439 = _guard435 & _guard438;
wire _guard440 = tdcc_go_out;
wire _guard441 = _guard439 & _guard440;
wire _guard442 = _guard434 | _guard441;
wire _guard443 = fsm_out == 5'd21;
wire _guard444 = invoke9_done_out;
wire _guard445 = _guard443 & _guard444;
wire _guard446 = tdcc_go_out;
wire _guard447 = _guard445 & _guard446;
wire _guard448 = fsm_out == 5'd3;
wire _guard449 = wrapper_early_reset_bb0_200_done_out;
wire _guard450 = comb_reg0_out;
wire _guard451 = _guard449 & _guard450;
wire _guard452 = _guard448 & _guard451;
wire _guard453 = tdcc_go_out;
wire _guard454 = _guard452 & _guard453;
wire _guard455 = fsm_out == 5'd13;
wire _guard456 = wrapper_early_reset_bb0_200_done_out;
wire _guard457 = comb_reg0_out;
wire _guard458 = _guard456 & _guard457;
wire _guard459 = _guard455 & _guard458;
wire _guard460 = tdcc_go_out;
wire _guard461 = _guard459 & _guard460;
wire _guard462 = _guard454 | _guard461;
wire _guard463 = fsm_out == 5'd5;
wire _guard464 = wrapper_early_reset_bb0_400_done_out;
wire _guard465 = comb_reg1_out;
wire _guard466 = _guard464 & _guard465;
wire _guard467 = _guard463 & _guard466;
wire _guard468 = tdcc_go_out;
wire _guard469 = _guard467 & _guard468;
wire _guard470 = fsm_out == 5'd11;
wire _guard471 = wrapper_early_reset_bb0_400_done_out;
wire _guard472 = comb_reg1_out;
wire _guard473 = _guard471 & _guard472;
wire _guard474 = _guard470 & _guard473;
wire _guard475 = tdcc_go_out;
wire _guard476 = _guard474 & _guard475;
wire _guard477 = _guard469 | _guard476;
wire _guard478 = fsm_out == 5'd19;
wire _guard479 = bb0_16_done_out;
wire _guard480 = _guard478 & _guard479;
wire _guard481 = tdcc_go_out;
wire _guard482 = _guard480 & _guard481;
wire _guard483 = fsm_out == 5'd16;
wire _guard484 = invoke7_done_out;
wire _guard485 = _guard483 & _guard484;
wire _guard486 = tdcc_go_out;
wire _guard487 = _guard485 & _guard486;
wire _guard488 = fsm_out == 5'd8;
wire _guard489 = bb0_9_done_out;
wire _guard490 = _guard488 & _guard489;
wire _guard491 = tdcc_go_out;
wire _guard492 = _guard490 & _guard491;
wire _guard493 = while_2_arg2_reg_done;
wire _guard494 = while_2_arg1_reg_done;
wire _guard495 = _guard493 & _guard494;
wire _guard496 = while_2_arg0_reg_done;
wire _guard497 = _guard495 & _guard496;
wire _guard498 = signal_reg0_out;
wire _guard499 = _guard0 & _guard0;
wire _guard500 = signal_reg0_out;
wire _guard501 = ~_guard500;
wire _guard502 = _guard499 & _guard501;
wire _guard503 = wrapper_early_reset_bb0_000_go_out;
wire _guard504 = _guard502 & _guard503;
wire _guard505 = _guard498 | _guard504;
wire _guard506 = _guard0 & _guard0;
wire _guard507 = signal_reg0_out;
wire _guard508 = ~_guard507;
wire _guard509 = _guard506 & _guard508;
wire _guard510 = wrapper_early_reset_bb0_000_go_out;
wire _guard511 = _guard509 & _guard510;
wire _guard512 = signal_reg0_out;
wire _guard513 = wrapper_early_reset_static_par_thread_done_out;
wire _guard514 = ~_guard513;
wire _guard515 = fsm_out == 5'd0;
wire _guard516 = _guard514 & _guard515;
wire _guard517 = tdcc_go_out;
wire _guard518 = _guard516 & _guard517;
wire _guard519 = assign_while_1_latch_go_out;
wire _guard520 = early_reset_static_par_thread0_go_out;
wire _guard521 = _guard519 | _guard520;
wire _guard522 = assign_while_1_latch_go_out;
wire _guard523 = early_reset_static_par_thread0_go_out;
wire _guard524 = early_reset_bb0_000_go_out;
wire _guard525 = early_reset_bb0_000_go_out;
wire _guard526 = signal_reg3_out;
wire _guard527 = _guard0 & _guard0;
wire _guard528 = signal_reg3_out;
wire _guard529 = ~_guard528;
wire _guard530 = _guard527 & _guard529;
wire _guard531 = wrapper_early_reset_bb0_400_go_out;
wire _guard532 = _guard530 & _guard531;
wire _guard533 = _guard526 | _guard532;
wire _guard534 = _guard0 & _guard0;
wire _guard535 = signal_reg3_out;
wire _guard536 = ~_guard535;
wire _guard537 = _guard534 & _guard536;
wire _guard538 = wrapper_early_reset_bb0_400_go_out;
wire _guard539 = _guard537 & _guard538;
wire _guard540 = signal_reg3_out;
wire _guard541 = signal_reg3_out;
wire _guard542 = invoke5_go_out;
wire _guard543 = invoke6_go_out;
wire _guard544 = _guard542 | _guard543;
wire _guard545 = invoke5_go_out;
wire _guard546 = invoke6_go_out;
wire _guard547 = early_reset_bb0_400_go_out;
wire _guard548 = early_reset_bb0_400_go_out;
wire _guard549 = assign_while_2_latch_done_out;
wire _guard550 = ~_guard549;
wire _guard551 = fsm_out == 5'd14;
wire _guard552 = _guard550 & _guard551;
wire _guard553 = tdcc_go_out;
wire _guard554 = _guard552 & _guard553;
wire _guard555 = invoke5_done_out;
wire _guard556 = ~_guard555;
wire _guard557 = fsm_out == 5'd4;
wire _guard558 = _guard556 & _guard557;
wire _guard559 = tdcc_go_out;
wire _guard560 = _guard558 & _guard559;
wire _guard561 = signal_reg4_out;
wire _guard562 = bb0_9_done_out;
wire _guard563 = ~_guard562;
wire _guard564 = fsm_out == 5'd8;
wire _guard565 = _guard563 & _guard564;
wire _guard566 = tdcc_go_out;
wire _guard567 = _guard565 & _guard566;
wire _guard568 = bb0_10_done_out;
wire _guard569 = ~_guard568;
wire _guard570 = fsm_out == 5'd9;
wire _guard571 = _guard569 & _guard570;
wire _guard572 = tdcc_go_out;
wire _guard573 = _guard571 & _guard572;
wire _guard574 = early_reset_bb0_200_go_out;
wire _guard575 = early_reset_bb0_200_go_out;
wire _guard576 = bb0_16_done_out;
wire _guard577 = ~_guard576;
wire _guard578 = fsm_out == 5'd19;
wire _guard579 = _guard577 & _guard578;
wire _guard580 = tdcc_go_out;
wire _guard581 = _guard579 & _guard580;
wire _guard582 = assign_while_2_latch_go_out;
wire _guard583 = assign_while_2_latch_go_out;
wire _guard584 = assign_while_2_latch_go_out;
wire _guard585 = early_reset_static_par_thread_go_out;
wire _guard586 = _guard584 | _guard585;
wire _guard587 = early_reset_static_par_thread_go_out;
wire _guard588 = assign_while_2_latch_go_out;
wire _guard589 = invoke8_go_out;
wire _guard590 = invoke9_go_out;
wire _guard591 = _guard589 | _guard590;
wire _guard592 = bb0_9_go_out;
wire _guard593 = invoke8_go_out;
wire _guard594 = invoke9_go_out;
wire _guard595 = bb0_9_go_out;
wire _guard596 = early_reset_bb0_1400_go_out;
wire _guard597 = early_reset_bb0_1400_go_out;
wire _guard598 = invoke8_done_out;
wire _guard599 = ~_guard598;
wire _guard600 = fsm_out == 5'd17;
wire _guard601 = _guard599 & _guard600;
wire _guard602 = tdcc_go_out;
wire _guard603 = _guard601 & _guard602;
wire _guard604 = wrapper_early_reset_bb0_200_done_out;
wire _guard605 = ~_guard604;
wire _guard606 = fsm_out == 5'd3;
wire _guard607 = _guard605 & _guard606;
wire _guard608 = tdcc_go_out;
wire _guard609 = _guard607 & _guard608;
wire _guard610 = wrapper_early_reset_bb0_200_done_out;
wire _guard611 = ~_guard610;
wire _guard612 = fsm_out == 5'd13;
wire _guard613 = _guard611 & _guard612;
wire _guard614 = tdcc_go_out;
wire _guard615 = _guard613 & _guard614;
wire _guard616 = _guard609 | _guard615;
wire _guard617 = invoke6_go_out;
wire _guard618 = invoke9_go_out;
wire _guard619 = bb0_7_go_out;
wire _guard620 = bb0_8_go_out;
wire _guard621 = _guard619 | _guard620;
wire _guard622 = bb0_10_go_out;
wire _guard623 = _guard621 | _guard622;
wire _guard624 = assign_while_1_latch_go_out;
wire _guard625 = _guard623 | _guard624;
wire _guard626 = assign_while_2_latch_go_out;
wire _guard627 = bb0_7_go_out;
wire _guard628 = bb0_8_go_out;
wire _guard629 = _guard627 | _guard628;
wire _guard630 = bb0_10_go_out;
wire _guard631 = _guard629 | _guard630;
wire _guard632 = invoke6_go_out;
wire _guard633 = invoke9_go_out;
wire _guard634 = _guard632 | _guard633;
wire _guard635 = assign_while_1_latch_go_out;
wire _guard636 = assign_while_2_latch_go_out;
wire _guard637 = assign_while_1_latch_go_out;
wire _guard638 = early_reset_static_par_thread0_go_out;
wire _guard639 = _guard637 | _guard638;
wire _guard640 = assign_while_1_latch_go_out;
wire _guard641 = early_reset_static_par_thread0_go_out;
wire _guard642 = bb0_9_go_out;
wire _guard643 = bb0_9_go_out;
wire _guard644 = std_addFN_0_done;
wire _guard645 = ~_guard644;
wire _guard646 = bb0_9_go_out;
wire _guard647 = _guard645 & _guard646;
wire _guard648 = bb0_9_go_out;
wire _guard649 = invoke7_go_out;
wire _guard650 = invoke7_go_out;
wire _guard651 = invoke7_go_out;
wire _guard652 = invoke7_go_out;
wire _guard653 = bb0_17_done_out;
wire _guard654 = ~_guard653;
wire _guard655 = fsm_out == 5'd20;
wire _guard656 = _guard654 & _guard655;
wire _guard657 = tdcc_go_out;
wire _guard658 = _guard656 & _guard657;
wire _guard659 = wrapper_early_reset_bb0_000_go_out;
wire _guard660 = wrapper_early_reset_bb0_1400_done_out;
wire _guard661 = ~_guard660;
wire _guard662 = fsm_out == 5'd18;
wire _guard663 = _guard661 & _guard662;
wire _guard664 = tdcc_go_out;
wire _guard665 = _guard663 & _guard664;
wire _guard666 = wrapper_early_reset_bb0_1400_done_out;
wire _guard667 = ~_guard666;
wire _guard668 = fsm_out == 5'd22;
wire _guard669 = _guard667 & _guard668;
wire _guard670 = tdcc_go_out;
wire _guard671 = _guard669 & _guard670;
wire _guard672 = _guard665 | _guard671;
wire _guard673 = signal_reg2_out;
wire _guard674 = _guard0 & _guard0;
wire _guard675 = signal_reg2_out;
wire _guard676 = ~_guard675;
wire _guard677 = _guard674 & _guard676;
wire _guard678 = wrapper_early_reset_bb0_200_go_out;
wire _guard679 = _guard677 & _guard678;
wire _guard680 = _guard673 | _guard679;
wire _guard681 = _guard0 & _guard0;
wire _guard682 = signal_reg2_out;
wire _guard683 = ~_guard682;
wire _guard684 = _guard681 & _guard683;
wire _guard685 = wrapper_early_reset_bb0_200_go_out;
wire _guard686 = _guard684 & _guard685;
wire _guard687 = signal_reg2_out;
wire _guard688 = wrapper_early_reset_static_par_thread0_done_out;
wire _guard689 = ~_guard688;
wire _guard690 = fsm_out == 5'd2;
wire _guard691 = _guard689 & _guard690;
wire _guard692 = tdcc_go_out;
wire _guard693 = _guard691 & _guard692;
wire _guard694 = signal_reg_out;
wire _guard695 = _guard0 & _guard0;
wire _guard696 = signal_reg_out;
wire _guard697 = ~_guard696;
wire _guard698 = _guard695 & _guard697;
wire _guard699 = wrapper_early_reset_static_par_thread_go_out;
wire _guard700 = _guard698 & _guard699;
wire _guard701 = _guard694 | _guard700;
wire _guard702 = _guard0 & _guard0;
wire _guard703 = signal_reg_out;
wire _guard704 = ~_guard703;
wire _guard705 = _guard702 & _guard704;
wire _guard706 = wrapper_early_reset_static_par_thread_go_out;
wire _guard707 = _guard705 & _guard706;
wire _guard708 = signal_reg_out;
wire _guard709 = signal_reg4_out;
wire _guard710 = _guard0 & _guard0;
wire _guard711 = signal_reg4_out;
wire _guard712 = ~_guard711;
wire _guard713 = _guard710 & _guard712;
wire _guard714 = wrapper_early_reset_bb0_1400_go_out;
wire _guard715 = _guard713 & _guard714;
wire _guard716 = _guard709 | _guard715;
wire _guard717 = _guard0 & _guard0;
wire _guard718 = signal_reg4_out;
wire _guard719 = ~_guard718;
wire _guard720 = _guard717 & _guard719;
wire _guard721 = wrapper_early_reset_bb0_1400_go_out;
wire _guard722 = _guard720 & _guard721;
wire _guard723 = signal_reg4_out;
wire _guard724 = wrapper_early_reset_static_par_thread_go_out;
wire _guard725 = signal_reg1_out;
wire _guard726 = early_reset_bb0_200_go_out;
wire _guard727 = early_reset_bb0_400_go_out;
wire _guard728 = early_reset_bb0_000_go_out;
wire _guard729 = early_reset_bb0_1400_go_out;
wire _guard730 = early_reset_bb0_1400_go_out;
wire _guard731 = early_reset_bb0_000_go_out;
wire _guard732 = early_reset_bb0_400_go_out;
wire _guard733 = early_reset_bb0_200_go_out;
wire _guard734 = _guard732 | _guard733;
wire _guard735 = while_1_arg1_reg_done;
wire _guard736 = while_1_arg0_reg_done;
wire _guard737 = _guard735 & _guard736;
wire _guard738 = wrapper_early_reset_static_par_thread0_go_out;
wire _guard739 = wrapper_early_reset_bb0_200_go_out;
wire _guard740 = signal_reg_out;
wire _guard741 = wrapper_early_reset_bb0_000_done_out;
wire _guard742 = ~_guard741;
wire _guard743 = fsm_out == 5'd1;
wire _guard744 = _guard742 & _guard743;
wire _guard745 = tdcc_go_out;
wire _guard746 = _guard744 & _guard745;
wire _guard747 = wrapper_early_reset_bb0_000_done_out;
wire _guard748 = ~_guard747;
wire _guard749 = fsm_out == 5'd15;
wire _guard750 = _guard748 & _guard749;
wire _guard751 = tdcc_go_out;
wire _guard752 = _guard750 & _guard751;
wire _guard753 = _guard746 | _guard752;
wire _guard754 = wrapper_early_reset_bb0_1400_go_out;
wire _guard755 = signal_reg0_out;
wire _guard756 = wrapper_early_reset_bb0_400_done_out;
wire _guard757 = ~_guard756;
wire _guard758 = fsm_out == 5'd5;
wire _guard759 = _guard757 & _guard758;
wire _guard760 = tdcc_go_out;
wire _guard761 = _guard759 & _guard760;
wire _guard762 = wrapper_early_reset_bb0_400_done_out;
wire _guard763 = ~_guard762;
wire _guard764 = fsm_out == 5'd11;
wire _guard765 = _guard763 & _guard764;
wire _guard766 = tdcc_go_out;
wire _guard767 = _guard765 & _guard766;
wire _guard768 = _guard761 | _guard767;
wire _guard769 = fsm_out == 5'd23;
wire _guard770 = assign_while_2_latch_go_out;
wire _guard771 = early_reset_static_par_thread_go_out;
wire _guard772 = _guard770 | _guard771;
wire _guard773 = early_reset_static_par_thread_go_out;
wire _guard774 = assign_while_2_latch_go_out;
wire _guard775 = bb0_7_done_out;
wire _guard776 = ~_guard775;
wire _guard777 = fsm_out == 5'd6;
wire _guard778 = _guard776 & _guard777;
wire _guard779 = tdcc_go_out;
wire _guard780 = _guard778 & _guard779;
wire _guard781 = assign_while_2_latch_go_out;
wire _guard782 = early_reset_static_par_thread_go_out;
wire _guard783 = _guard781 | _guard782;
wire _guard784 = assign_while_2_latch_go_out;
wire _guard785 = early_reset_static_par_thread_go_out;
wire _guard786 = wrapper_early_reset_bb0_400_go_out;
wire _guard787 = invoke6_done_out;
wire _guard788 = ~_guard787;
wire _guard789 = fsm_out == 5'd10;
wire _guard790 = _guard788 & _guard789;
wire _guard791 = tdcc_go_out;
wire _guard792 = _guard790 & _guard791;
wire _guard793 = bb0_8_done_out;
wire _guard794 = ~_guard793;
wire _guard795 = fsm_out == 5'd7;
wire _guard796 = _guard794 & _guard795;
wire _guard797 = tdcc_go_out;
wire _guard798 = _guard796 & _guard797;
wire _guard799 = invoke7_done_out;
wire _guard800 = ~_guard799;
wire _guard801 = fsm_out == 5'd16;
wire _guard802 = _guard800 & _guard801;
wire _guard803 = tdcc_go_out;
wire _guard804 = _guard802 & _guard803;
assign std_add_5_left =
  _guard1 ? while_1_arg0_reg_out :
  _guard2 ? while_2_arg1_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard2, _guard1})) begin
    $fatal(2, "Multiple assignment to port `std_add_5.left'.");
end
end
assign std_add_5_right =
  _guard3 ? 32'd1 :
  _guard4 ? 32'd10 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard4, _guard3})) begin
    $fatal(2, "Multiple assignment to port `std_add_5.right'.");
end
end
assign std_slice_4_in =
  _guard7 ? addf_0_reg_out :
  _guard12 ? std_add_6_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard12, _guard7})) begin
    $fatal(2, "Multiple assignment to port `std_slice_4.in'.");
end
end
assign signal_reg1_write_en = _guard20;
assign signal_reg1_clk = clk;
assign signal_reg1_reset = reset;
assign signal_reg1_in =
  _guard26 ? 1'd1 :
  _guard27 ? 1'd0 :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard27, _guard26})) begin
    $fatal(2, "Multiple assignment to port `signal_reg1.in'.");
end
end
assign assign_while_1_latch_go_in = _guard33;
assign wrapper_early_reset_bb0_200_done_in = _guard34;
assign invoke7_done_in = relu4d_0_instance_done;
assign invoke9_go_in = _guard40;
assign invoke9_done_in = addf_0_reg_done;
assign done = _guard41;
assign arg_mem_0_content_en = _guard42;
assign arg_mem_4_write_data = relu4d_0_instance_arg_mem_1_write_data;
assign arg_mem_3_addr0 =
  _guard44 ? std_slice_4_out :
  _guard45 ? relu4d_0_instance_arg_mem_0_addr0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard45, _guard44})) begin
    $fatal(2, "Multiple assignment to port `_this.arg_mem_3_addr0'.");
end
end
assign arg_mem_3_write_data = addf_0_reg_out;
assign arg_mem_0_addr0 = std_slice_4_out;
assign arg_mem_4_addr0 =
  _guard48 ? std_slice_4_out :
  _guard49 ? relu4d_0_instance_arg_mem_1_addr0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard49, _guard48})) begin
    $fatal(2, "Multiple assignment to port `_this.arg_mem_4_addr0'.");
end
end
assign arg_mem_3_content_en =
  _guard50 ? 1'd1 :
  _guard51 ? relu4d_0_instance_arg_mem_0_content_en :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard51, _guard50})) begin
    $fatal(2, "Multiple assignment to port `_this.arg_mem_3_content_en'.");
end
end
assign arg_mem_3_write_en = _guard52;
assign arg_mem_2_addr0 = std_slice_4_out;
assign arg_mem_2_content_en = _guard54;
assign arg_mem_4_content_en =
  _guard55 ? 1'd1 :
  _guard56 ? relu4d_0_instance_arg_mem_1_content_en :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard56, _guard55})) begin
    $fatal(2, "Multiple assignment to port `_this.arg_mem_4_content_en'.");
end
end
assign arg_mem_2_write_en = _guard57;
assign arg_mem_4_write_en =
  _guard58 ? relu4d_0_instance_arg_mem_1_write_en :
  1'd0;
assign arg_mem_1_addr0 = std_slice_4_out;
assign arg_mem_1_content_en = _guard60;
assign arg_mem_2_write_data = arg_mem_4_read_data;
assign fsm_write_en = _guard288;
assign fsm_clk = clk;
assign fsm_reset = reset;
assign fsm_in =
  _guard293 ? 5'd1 :
  _guard298 ? 5'd15 :
  _guard315 ? 5'd23 :
  _guard320 ? 5'd18 :
  _guard337 ? 5'd16 :
  _guard338 ? 5'd0 :
  _guard343 ? 5'd3 :
  _guard348 ? 5'd13 :
  _guard365 ? 5'd14 :
  _guard370 ? 5'd5 :
  _guard387 ? 5'd12 :
  _guard402 ? 5'd2 :
  _guard407 ? 5'd8 :
  _guard412 ? 5'd10 :
  _guard417 ? 5'd7 :
  _guard422 ? 5'd11 :
  _guard427 ? 5'd21 :
  _guard442 ? 5'd19 :
  _guard447 ? 5'd22 :
  _guard462 ? 5'd4 :
  _guard477 ? 5'd6 :
  _guard482 ? 5'd20 :
  _guard487 ? 5'd17 :
  _guard492 ? 5'd9 :
  5'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard492, _guard487, _guard482, _guard477, _guard462, _guard447, _guard442, _guard427, _guard422, _guard417, _guard412, _guard407, _guard402, _guard387, _guard370, _guard365, _guard348, _guard343, _guard338, _guard337, _guard320, _guard315, _guard298, _guard293})) begin
    $fatal(2, "Multiple assignment to port `fsm.in'.");
end
end
assign assign_while_2_latch_done_in = _guard497;
assign signal_reg0_write_en = _guard505;
assign signal_reg0_clk = clk;
assign signal_reg0_reset = reset;
assign signal_reg0_in =
  _guard511 ? 1'd1 :
  _guard512 ? 1'd0 :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard512, _guard511})) begin
    $fatal(2, "Multiple assignment to port `signal_reg0.in'.");
end
end
assign wrapper_early_reset_static_par_thread_go_in = _guard518;
assign while_1_arg0_reg_write_en = _guard521;
assign while_1_arg0_reg_clk = clk;
assign while_1_arg0_reg_reset = reset;
assign while_1_arg0_reg_in =
  _guard522 ? std_add_5_out :
  _guard523 ? 32'd0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard523, _guard522})) begin
    $fatal(2, "Multiple assignment to port `while_1_arg0_reg.in'.");
end
end
assign comb_reg_write_en = _guard524;
assign comb_reg_clk = clk;
assign comb_reg_reset = reset;
assign comb_reg_in =
  _guard525 ? std_slt_3_out :
  1'd0;
assign signal_reg3_write_en = _guard533;
assign signal_reg3_clk = clk;
assign signal_reg3_reset = reset;
assign signal_reg3_in =
  _guard539 ? 1'd1 :
  _guard540 ? 1'd0 :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard540, _guard539})) begin
    $fatal(2, "Multiple assignment to port `signal_reg3.in'.");
end
end
assign early_reset_static_par_thread0_done_in = ud1_out;
assign wrapper_early_reset_bb0_400_done_in = _guard541;
assign early_reset_bb0_200_done_in = ud5_out;
assign while_0_arg0_reg_write_en = _guard544;
assign while_0_arg0_reg_clk = clk;
assign while_0_arg0_reg_reset = reset;
assign while_0_arg0_reg_in =
  _guard545 ? 32'd0 :
  _guard546 ? std_add_6_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard546, _guard545})) begin
    $fatal(2, "Multiple assignment to port `while_0_arg0_reg.in'.");
end
end
assign comb_reg1_write_en = _guard547;
assign comb_reg1_clk = clk;
assign comb_reg1_reset = reset;
assign comb_reg1_in =
  _guard548 ? std_slt_3_out :
  1'd0;
assign assign_while_2_latch_go_in = _guard554;
assign invoke5_go_in = _guard560;
assign invoke5_done_in = while_0_arg0_reg_done;
assign wrapper_early_reset_bb0_1400_done_in = _guard561;
assign bb0_9_go_in = _guard567;
assign bb0_10_go_in = _guard573;
assign comb_reg0_write_en = _guard574;
assign comb_reg0_clk = clk;
assign comb_reg0_reset = reset;
assign comb_reg0_in =
  _guard575 ? std_slt_3_out :
  1'd0;
assign tdcc_go_in = go;
assign bb0_16_go_in = _guard581;
assign std_add_0_left = while_2_arg0_reg_out;
assign std_add_0_right = 32'd1;
assign while_2_arg0_reg_write_en = _guard586;
assign while_2_arg0_reg_clk = clk;
assign while_2_arg0_reg_reset = reset;
assign while_2_arg0_reg_in =
  _guard587 ? 32'd0 :
  _guard588 ? std_add_0_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard588, _guard587})) begin
    $fatal(2, "Multiple assignment to port `while_2_arg0_reg.in'.");
end
end
assign addf_0_reg_write_en =
  _guard591 ? 1'd1 :
  _guard592 ? std_addFN_0_done :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard592, _guard591})) begin
    $fatal(2, "Multiple assignment to port `addf_0_reg.write_en'.");
end
end
assign addf_0_reg_clk = clk;
assign addf_0_reg_reset = reset;
assign addf_0_reg_in =
  _guard593 ? 32'd0 :
  _guard594 ? std_add_6_out :
  _guard595 ? std_addFN_0_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard595, _guard594, _guard593})) begin
    $fatal(2, "Multiple assignment to port `addf_0_reg.in'.");
end
end
assign comb_reg2_write_en = _guard596;
assign comb_reg2_clk = clk;
assign comb_reg2_reset = reset;
assign comb_reg2_in =
  _guard597 ? std_slt_3_out :
  1'd0;
assign invoke8_go_in = _guard603;
assign wrapper_early_reset_bb0_200_go_in = _guard616;
assign std_add_6_left =
  _guard617 ? while_0_arg0_reg_out :
  _guard618 ? addf_0_reg_out :
  _guard625 ? while_1_arg1_reg_out :
  _guard626 ? while_2_arg2_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard626, _guard625, _guard618, _guard617})) begin
    $fatal(2, "Multiple assignment to port `std_add_6.left'.");
end
end
assign std_add_6_right =
  _guard631 ? while_0_arg0_reg_out :
  _guard634 ? 32'd1 :
  _guard635 ? 32'd10 :
  _guard636 ? 32'd100 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard636, _guard635, _guard634, _guard631})) begin
    $fatal(2, "Multiple assignment to port `std_add_6.right'.");
end
end
assign while_1_arg1_reg_write_en = _guard639;
assign while_1_arg1_reg_clk = clk;
assign while_1_arg1_reg_reset = reset;
assign while_1_arg1_reg_in =
  _guard640 ? std_add_6_out :
  _guard641 ? while_2_arg2_reg_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard641, _guard640})) begin
    $fatal(2, "Multiple assignment to port `while_1_arg1_reg.in'.");
end
end
assign std_addFN_0_roundingMode = 3'd0;
assign std_addFN_0_control = 1'd0;
assign std_addFN_0_clk = clk;
assign std_addFN_0_left =
  _guard642 ? arg_mem_0_read_data :
  32'd0;
assign std_addFN_0_subOp =
  _guard643 ? 1'd0 :
  1'd0;
assign std_addFN_0_reset = reset;
assign std_addFN_0_go = _guard647;
assign std_addFN_0_right =
  _guard648 ? arg_mem_1_read_data :
  32'd0;
assign relu4d_0_instance_arg_mem_0_read_data =
  _guard649 ? arg_mem_3_read_data :
  32'd0;
assign relu4d_0_instance_arg_mem_0_done =
  _guard650 ? arg_mem_3_done :
  1'd0;
assign relu4d_0_instance_clk = clk;
assign relu4d_0_instance_reset = reset;
assign relu4d_0_instance_go = _guard651;
assign relu4d_0_instance_arg_mem_1_done =
  _guard652 ? arg_mem_4_done :
  1'd0;
assign bb0_17_go_in = _guard658;
assign invoke8_done_in = addf_0_reg_done;
assign early_reset_bb0_000_go_in = _guard659;
assign wrapper_early_reset_bb0_1400_go_in = _guard672;
assign bb0_9_done_in = addf_0_reg_done;
assign signal_reg2_write_en = _guard680;
assign signal_reg2_clk = clk;
assign signal_reg2_reset = reset;
assign signal_reg2_in =
  _guard686 ? 1'd1 :
  _guard687 ? 1'd0 :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard687, _guard686})) begin
    $fatal(2, "Multiple assignment to port `signal_reg2.in'.");
end
end
assign bb0_7_done_in = arg_mem_0_done;
assign bb0_17_done_in = arg_mem_2_done;
assign wrapper_early_reset_static_par_thread0_go_in = _guard693;
assign bb0_16_done_in = arg_mem_4_done;
assign invoke6_done_in = while_0_arg0_reg_done;
assign signal_reg_write_en = _guard701;
assign signal_reg_clk = clk;
assign signal_reg_reset = reset;
assign signal_reg_in =
  _guard707 ? 1'd1 :
  _guard708 ? 1'd0 :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard708, _guard707})) begin
    $fatal(2, "Multiple assignment to port `signal_reg.in'.");
end
end
assign signal_reg4_write_en = _guard716;
assign signal_reg4_clk = clk;
assign signal_reg4_reset = reset;
assign signal_reg4_in =
  _guard722 ? 1'd1 :
  _guard723 ? 1'd0 :
  1'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard723, _guard722})) begin
    $fatal(2, "Multiple assignment to port `signal_reg4.in'.");
end
end
assign bb0_8_done_in = arg_mem_1_done;
assign early_reset_static_par_thread_go_in = _guard724;
assign wrapper_early_reset_static_par_thread0_done_in = _guard725;
assign std_slt_3_left =
  _guard726 ? while_1_arg0_reg_out :
  _guard727 ? while_0_arg0_reg_out :
  _guard728 ? while_2_arg0_reg_out :
  _guard729 ? addf_0_reg_out :
  32'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard729, _guard728, _guard727, _guard726})) begin
    $fatal(2, "Multiple assignment to port `std_slt_3.left'.");
end
end
assign std_slt_3_right =
  _guard730 ? 32'd300 :
  _guard731 ? 32'd3 :
  _guard734 ? 32'd10 :
  32'd0;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard734, _guard731, _guard730})) begin
    $fatal(2, "Multiple assignment to port `std_slt_3.right'.");
end
end
assign assign_while_1_latch_done_in = _guard737;
assign early_reset_static_par_thread0_go_in = _guard738;
assign early_reset_bb0_200_go_in = _guard739;
assign wrapper_early_reset_static_par_thread_done_in = _guard740;
assign wrapper_early_reset_bb0_000_go_in = _guard753;
assign early_reset_bb0_000_done_in = ud6_out;
assign early_reset_bb0_1400_go_in = _guard754;
assign wrapper_early_reset_bb0_000_done_in = _guard755;
assign wrapper_early_reset_bb0_400_go_in = _guard768;
assign tdcc_done_in = _guard769;
assign while_2_arg2_reg_write_en = _guard772;
assign while_2_arg2_reg_clk = clk;
assign while_2_arg2_reg_reset = reset;
assign while_2_arg2_reg_in =
  _guard773 ? 32'd0 :
  _guard774 ? std_add_6_out :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard774, _guard773})) begin
    $fatal(2, "Multiple assignment to port `while_2_arg2_reg.in'.");
end
end
assign bb0_7_go_in = _guard780;
assign early_reset_bb0_400_done_in = ud4_out;
assign while_2_arg1_reg_write_en = _guard783;
assign while_2_arg1_reg_clk = clk;
assign while_2_arg1_reg_reset = reset;
assign while_2_arg1_reg_in =
  _guard784 ? std_add_5_out :
  _guard785 ? 32'd0 :
  'x;
always_ff @(posedge clk) begin
  if(~$onehot0({_guard785, _guard784})) begin
    $fatal(2, "Multiple assignment to port `while_2_arg1_reg.in'.");
end
end
assign early_reset_static_par_thread_done_in = ud_out;
assign early_reset_bb0_400_go_in = _guard786;
assign invoke6_go_in = _guard792;
assign bb0_8_go_in = _guard798;
assign early_reset_bb0_1400_done_in = ud8_out;
assign bb0_10_done_in = arg_mem_3_done;
assign invoke7_go_in = _guard804;
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

