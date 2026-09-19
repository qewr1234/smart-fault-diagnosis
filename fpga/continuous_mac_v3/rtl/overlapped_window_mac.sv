// Window MAC + bias + ReLU with window-level overlap. Same arithmetic,
// configuration and output format as continuous_window_mac.
// K,COUT,P,DEPTH >= 1. Activations 0..127; weights signed INT8; INT32 sums.
// Host must prove abs(bias[c])+127*sum(abs(weights[c])) <= 2147483647.
//
// Difference from continuous_window_mac: the tuple RAM is double buffered.
// start_ready is high whenever a tuple bank is free, so the host can stream
// window w+1 (K input beats) while window w is still issuing taps or draining
// outputs. The issue engine moves from the last group of window w straight to
// the first group of window w+1 without a bubble if that window is loaded.
// Result slots (DEPTH) are reserved per group exactly as in v2; a group is
// never issued unless its result slot is guaranteed. Outputs stay in order:
// window order, then channel order, with m_last on channel COUT-1.
//
// A window with no nonzero taps (sparse mode, all-zero input) is handled by
// issuing one pseudo tap per group with the product forced to zero, so the
// bias + ReLU path is the same pipeline and no separate state is needed.
//
// Configuration is accepted only when no window is loading, loaded, or has
// results in flight (core_idle). Configuration wins over start in that state.
module overlapped_window_mac #(
    parameter integer K=576, COUT=128, P=2, DEPTH=2,
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
    localparam integer SW=(DEPTH<2 ? 1 : $clog2(DEPTH));
    localparam integer RW=$clog2(DEPTH+1);
    localparam integer NW=$clog2(K+1);
    localparam integer TW=$clog2(2*K);   // tuple RAM address: {bank, position}

    // Load side: one window at a time into tuple bank load_bank.
    reg load_active, load_bank, mode;
    reg [KW-1:0] input_tap;
    reg [NW-1:0] count;
    // Bank ownership. busy: loading or loaded. loaded: complete, not yet issued.
    reg [1:0] busy, loaded;
    reg [NW-1:0] win_count [0:1];

    // Issue side: consumes tuple bank run_bank.
    reg run_bank;
    reg [NW-1:0] issue_pos;
    reg [GW-1:0] issue_group;
    reg [SW-1:0] head, tail, issue_slot;
    reg [RW-1:0] reserved;
    reg [DEPTH-1:0] slot_ready;
    reg [LW-1:0] emit_lane;
    reg [CW-1:0] output_channel;
    reg signed [31:0] results [0:DEPTH-1][0:P-1];

    (* ram_style="block" *) reg [KW+6:0] tuple_mem [0:2*K-1];
    reg [KW+6:0] tuple_q;
    reg a_valid, a_first, a_last, a_zero;
    reg b_valid, b_first, b_last, b_zero;
    reg c_valid, c_first, c_last;
    reg [GW-1:0] a_group, b_group, c_group;
    reg [SW-1:0] a_slot, b_slot, c_slot;
    reg [6:0] b_activation;
    wire signed [7:0] weight_q [0:P-1];
    wire signed [31:0] bias_q [0:P-1];
    reg signed [15:0] product [0:P-1];
    reg signed [31:0] accum [0:P-1];
    wire signed [31:0] sum_next [0:P-1];

    wire core_idle=!busy[0] && !busy[1] && reserved==0;
    wire take_cfg=cfg_valid && cfg_ready;
    assign cfg_ready=rst_n && core_idle;
    assign start_ready=rst_n && !load_active && !busy[load_bank] && !cfg_valid;
    assign s_ready=rst_n && load_active;
    wire take_start=start_valid && start_ready;
    wire take_input=s_valid && s_ready;
    wire store_input=take_input && (!mode || s_data!=0);
    wire [NW-1:0] one={{(NW-1){1'b0}},1'b1};
    wire [NW-1:0] count_next=store_input ? count+one : count;

    wire run_ready=loaded[run_bank];
    wire [NW-1:0] run_count=win_count[run_bank];
    wire zero_window=(run_count==0);
    wire [NW-1:0] eff_count=zero_window ? one : run_count;
    wire last_pos=(issue_pos==eff_count-1);

    assign m_valid=rst_n && slot_ready[head];
    assign m_data=results[head][emit_lane];
    assign m_channel=output_channel;
    assign m_last=(output_channel==COUT-1);
    wire take_output=m_valid && m_ready;
    wire end_group=(emit_lane==P-1 || m_last);
    wire release_slot=take_output && end_group;
    // Once a group is started it never stalls inside the MAC pipeline.
    wire issue=rst_n && run_ready && (issue_pos!=0 || reserved<DEPTH || release_slot);
    wire reserve_slot=issue && issue_pos==0;
    // Real tap issues (excludes the pseudo tap of an all-zero window).
    wire tap_issue=issue && !zero_window;

    wire [TW-1:0] write_addr=load_bank*K+count;
    wire [TW-1:0] read_addr=run_bank*K+issue_pos;

    genvar lane;
    generate for(lane=0;lane<P;lane=lane+1) begin: bank
        (* ram_style="block" *) reg signed [7:0] weights [0:GROUPS*K-1];
        reg signed [7:0] wq;
        reg signed [31:0] biases [0:GROUPS-1];
        // Memories are deliberately not reset. Configure every valid weight/bias.
        always @(posedge clk) begin
            if(rst_n && take_cfg && cfg_channel<COUT && cfg_channel%P==lane) begin
                if(cfg_is_bias) biases[cfg_channel/P]<=cfg_data;
                else if(cfg_tap<K) weights[(cfg_channel/P)*K+cfg_tap]<=cfg_data[7:0];
            end
            if(rst_n && a_valid) wq<=weights[a_group*K+tuple_q[KW+6:7]];
        end
        assign weight_q[lane]=(b_group*P+lane<COUT) ? wq : 8'sd0;
        assign bias_q[lane]=(c_group*P+lane<COUT) ? biases[c_group] : 32'sd0;
        assign sum_next[lane]=(c_first ? bias_q[lane] : accum[lane]) +
                             {{16{product[lane][15]}},product[lane]};
    end endgenerate

    // Simple dual port: the load side writes one bank, the issue side reads
    // the other. Bank ownership guarantees the banks differ.
    always @(posedge clk) begin
        if(rst_n && store_input) tuple_mem[write_addr]<={input_tap,s_data};
        if(issue) tuple_q<=tuple_mem[read_addr];
    end

    integer l;
    always @(posedge clk) begin
        if(!rst_n) begin
            load_active<=0;load_bank<=0;mode<=0;input_tap<=0;count<=0;
            busy<=0;loaded<=0;win_count[0]<=0;win_count[1]<=0;
            run_bank<=0;issue_pos<=0;issue_group<=0;
            head<=0;tail<=0;issue_slot<=0;reserved<=0;slot_ready<=0;
            emit_lane<=0;output_channel<=0;
            a_valid<=0;b_valid<=0;c_valid<=0;
            a_first<=0;b_first<=0;c_first<=0;a_last<=0;b_last<=0;c_last<=0;
            a_zero<=0;b_zero<=0;
            a_group<=0;b_group<=0;c_group<=0;a_slot<=0;b_slot<=0;c_slot<=0;
            b_activation<=0;
            for(l=0;l<P;l=l+1) begin product[l]<=0;accum[l]<=0;end
        end else begin
            // ---- load side ----
            if(take_start) begin
                load_active<=1;busy[load_bank]<=1;
                mode<=sparse_mode;input_tap<=0;count<=0;
            end else if(take_input) begin
                count<=count_next;
                if(input_tap==K-1) begin
                    loaded[load_bank]<=1;win_count[load_bank]<=count_next;
                    load_active<=0;load_bank<=~load_bank;
                end else input_tap<=input_tap+1'b1;
            end
            // ---- MAC pipeline: tuple read -> weight read -> multiply -> accumulate ----
            a_valid<=issue;
            if(issue) begin
                a_first<=(issue_pos==0);a_last<=last_pos;a_zero<=zero_window;
                a_group<=issue_group;
                a_slot<=(issue_pos==0) ? tail : issue_slot;
            end
            b_valid<=a_valid;
            if(a_valid) begin
                b_activation<=a_zero ? 7'd0 : tuple_q[6:0];
                b_first<=a_first;b_last<=a_last;b_zero<=a_zero;
                b_group<=a_group;b_slot<=a_slot;
            end
            c_valid<=b_valid;
            if(b_valid) begin
                c_first<=b_first;c_last<=b_last;c_group<=b_group;c_slot<=b_slot;
                for(l=0;l<P;l=l+1)
                    product[l]<=b_zero ? 16'sd0 : $signed({1'b0,b_activation})*$signed(weight_q[l]);
            end
            if(c_valid) begin
                for(l=0;l<P;l=l+1) begin
                    accum[l]<=sum_next[l];
                    if(c_last) results[c_slot][l]<=sum_next[l][31] ? 32'sd0 : sum_next[l];
                end
                if(c_last) slot_ready[c_slot]<=1;
            end
            // ---- issue side ----
            case({reserve_slot,release_slot})
            2'b10: reserved<=reserved+1'b1;
            2'b01: reserved<=reserved-1'b1;
            default: ;
            endcase
            if(reserve_slot) begin
                issue_slot<=tail;
                tail<=(tail==DEPTH-1) ? 0 : tail+1'b1;
            end
            if(issue) begin
                if(last_pos) begin
                    issue_pos<=0;
                    if(issue_group==GROUPS-1) begin
                        issue_group<=0;
                        loaded[run_bank]<=0;busy[run_bank]<=0;
                        run_bank<=~run_bank;
                    end else issue_group<=issue_group+1'b1;
                end else issue_pos<=issue_pos+1'b1;
            end
            // ---- output side ----
            if(take_output) begin
                if(end_group) begin
                    slot_ready[head]<=0;
                    head<=(head==DEPTH-1) ? 0 : head+1'b1;
                    emit_lane<=0;
                end else emit_lane<=emit_lane+1'b1;
                output_channel<=m_last ? {CW{1'b0}} : output_channel+1'b1;
            end
        end
    end
endmodule
