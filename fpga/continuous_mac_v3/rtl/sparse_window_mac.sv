// Window-level dense/sparse Conv+ReLU microkernel. NOT a complete CNN/AXI system.
// Input: K nonnegative INT8 values (0..127). Signed INT8 weights, INT32 bias/output.
// Common input list and synchronous weight banks. One issued tuple per cycle.
// Accumulator range must be proven by host; no intermediate saturation.
module sparse_window_mac #(
    parameter integer K=576, COUT=128, P=2,
    parameter integer KW=(K<2 ? 1 : $clog2(K)),
    parameter integer CW=(COUT<2 ? 1 : $clog2(COUT))
)(
    input wire clk, input wire rst_n,
    input wire cfg_valid, output wire cfg_ready, input wire cfg_is_bias,
    input wire [CW-1:0] cfg_channel, input wire [KW-1:0] cfg_tap,
    input wire signed [31:0] cfg_data,
    input wire start_valid, output wire start_ready, input wire sparse_mode,
    input wire s_valid, output wire s_ready, input wire [6:0] s_data,
    output wire m_valid, input wire m_ready, output wire signed [31:0] m_data,
    output wire [CW-1:0] m_channel, output wire m_last
);
    localparam integer GROUPS=(COUT+P-1)/P;
    localparam integer GW=(GROUPS<2 ? 1 : $clog2(GROUPS));
    localparam integer LW=(P<2 ? 1 : $clog2(P));
    localparam integer NW=$clog2(K+1);
    localparam [2:0] IDLE=0, LOAD=1, PREP=2, RUN=3, EMIT=4;
    reg [2:0] state;
    reg mode;
    reg [KW-1:0] input_tap;
    reg [NW-1:0] count, issue_pos;
    reg [GW-1:0] group_id;
    reg [LW-1:0] emit_lane;
    // One shared compressed list: original tap index + nonzero activation.
    (* ram_style="block" *) reg [KW+6:0] tuple_mem [0:K-1];
    reg [KW+6:0] tuple_q;
    wire issue=(state==RUN && issue_pos<count);
    reg a_valid,a_first,a_last;
    reg b_valid,b_first,b_last;
    reg c_valid,c_first,c_last;
    reg [6:0] b_activation;
    wire signed [7:0] weight_q [0:P-1];
    wire signed [31:0] bias_q [0:P-1];
    reg signed [15:0] product [0:P-1];
    reg signed [31:0] accum [0:P-1];
    reg signed [31:0] result [0:P-1];
    wire signed [31:0] sum_next [0:P-1];
    wire take_cfg=cfg_valid && cfg_ready;
    assign cfg_ready=(state==IDLE);
    // Configuration has priority if both request the idle core.
    assign start_ready=(state==IDLE && !cfg_valid);
    assign s_ready=(state==LOAD);
    assign m_valid=(state==EMIT);
    assign m_data=result[emit_lane];
    assign m_channel=group_id*P+emit_lane;
    assign m_last=(m_channel==COUT-1);

    genvar lane;
    generate for(lane=0;lane<P;lane=lane+1) begin: bank
        (* ram_style="block" *) reg signed [7:0] weights [0:GROUPS*K-1];
        reg signed [7:0] wq;
        reg signed [31:0] biases [0:GROUPS-1];
        // No reset on memories: host reloads configuration before use.
        // Synchronous reads are intentional; no combinational weight array lookup.
        always @(posedge clk) begin
            if(rst_n && take_cfg && cfg_channel<COUT && cfg_channel%P==lane) begin
                if(cfg_is_bias) biases[cfg_channel/P]<=cfg_data;
                else if(cfg_tap<K) weights[(cfg_channel/P)*K+cfg_tap]<=cfg_data[7:0];
            end
            if(rst_n && a_valid) wq<=weights[group_id*K+tuple_q[KW+6:7]];
        end
        assign weight_q[lane]=(group_id*P+lane<COUT) ? wq : 8'sd0;
        assign bias_q[lane]=(group_id*P+lane<COUT) ? biases[group_id] : 32'sd0;
        assign sum_next[lane]=(c_first ? bias_q[lane] : accum[lane]) + {{16{product[lane][15]}},product[lane]};
    end endgenerate

    // Shared synchronous tuple RAM. Dense mode stores every input; sparse mode
    // writes only nonzero entries while accepting the same K input beats.
    always @(posedge clk) begin
        if(rst_n && s_valid && s_ready && (!mode || s_data!=0))
            tuple_mem[count]<={input_tap,s_data};
        if(rst_n && issue) tuple_q<=tuple_mem[issue_pos];
    end

    integer l;
    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;mode<=0;input_tap<=0;count<=0;issue_pos<=0;group_id<=0;emit_lane<=0;
            a_valid<=0;b_valid<=0;c_valid<=0;a_first<=0;b_first<=0;c_first<=0;a_last<=0;b_last<=0;c_last<=0;b_activation<=0;
            for(l=0;l<P;l=l+1) begin product[l]<=0;accum[l]<=0;result[l]<=0;end
        end else begin
            // tuple read -> weight read -> registered multiply -> accumulate.
            a_valid<=issue;
            if(issue) begin a_first<=(issue_pos==0);a_last<=(issue_pos==count-1);end
            b_valid<=a_valid;
            if(a_valid) begin b_activation<=tuple_q[6:0];b_first<=a_first;b_last<=a_last;end
            c_valid<=b_valid;
            if(b_valid) begin
                c_first<=b_first;c_last<=b_last;
                for(l=0;l<P;l=l+1) product[l]<=$signed({1'b0,b_activation})*$signed(weight_q[l]);
            end
            case(state)
            IDLE: if(start_valid && start_ready) begin
                mode<=sparse_mode;input_tap<=0;count<=0;group_id<=0;state<=LOAD;
            end
            LOAD: if(s_valid && s_ready) begin
                if(!mode || s_data!=0) count<=count+1'b1;
                if(input_tap==K-1) state<=PREP;
                else input_tap<=input_tap+1'b1;
            end
            PREP: begin
                emit_lane<=0;issue_pos<=0;
                if(count==0) begin
                    for(l=0;l<P;l=l+1) result[l]<=bias_q[l][31] ? 32'sd0 : bias_q[l];
                    state<=EMIT;
                end else state<=RUN;
            end
            RUN: begin
                if(issue) issue_pos<=issue_pos+1'b1;
                if(c_valid) begin
                    for(l=0;l<P;l=l+1) begin
                        accum[l]<=sum_next[l];
                        if(c_last) result[l]<=sum_next[l][31] ? 32'sd0 : sum_next[l];
                    end
                    if(c_last) begin emit_lane<=0;state<=EMIT;end
                end
            end
            EMIT: if(m_valid && m_ready) begin
                if(m_last) state<=IDLE;
                else if(emit_lane==P-1) begin group_id<=group_id+1'b1;state<=PREP;end
                else emit_lane<=emit_lane+1'b1;
            end
            default: state<=IDLE;
            endcase
        end
    end
endmodule
