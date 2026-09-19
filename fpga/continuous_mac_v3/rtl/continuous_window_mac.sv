// Window MAC + bias + ReLU. Same arithmetic/configuration as sparse_window_mac.
// K,COUT,P,DEPTH >= 1. Activations 0..127; weights signed INT8; INT32 sums.
// Host must prove abs(bias[c])+127*sum(abs(weights[c])) <= 2147483647.
// DEPTH is the number of reserved P-channel result slots (registers, not BRAM).
// Reserve before issuing a group's first tap. Release after its last output.
// This protects in-flight results under arbitrary output backpressure.
// A slot freed on this edge can be reserved on the same edge.
module continuous_window_mac #(
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
    localparam [2:0] IDLE=0, LOAD=1, PREP=2, RUN=3, ZERO_EMIT=4;
    reg [2:0] state;
    reg mode;
    reg [KW-1:0] input_tap;
    reg [NW-1:0] count, issue_pos;
    reg [GW-1:0] issue_group, zero_group;
    reg issued_all;
    reg [SW-1:0] head, tail, issue_slot;
    reg [RW-1:0] reserved;
    reg [DEPTH-1:0] slot_ready;
    reg [LW-1:0] emit_lane;
    reg [CW-1:0] output_channel;
    reg signed [31:0] results [0:DEPTH-1][0:P-1];
    reg signed [31:0] zero_result [0:P-1];

    (* ram_style="block" *) reg [KW+6:0] tuple_mem [0:K-1];
    reg [KW+6:0] tuple_q;
    reg a_valid, a_first, a_last;
    reg b_valid, b_first, b_last;
    reg c_valid, c_first, c_last;
    reg [GW-1:0] a_group, b_group, c_group;
    reg [SW-1:0] a_slot, b_slot, c_slot;
    reg [6:0] b_activation;
    wire signed [7:0] weight_q [0:P-1];
    wire signed [31:0] bias_q [0:P-1], zero_bias [0:P-1];
    reg signed [15:0] product [0:P-1];
    reg signed [31:0] accum [0:P-1];
    wire signed [31:0] sum_next [0:P-1];

    wire take_cfg=cfg_valid && cfg_ready;
    assign cfg_ready=rst_n && state==IDLE;
    assign start_ready=rst_n && state==IDLE && !cfg_valid;
    assign s_ready=rst_n && state==LOAD;
    assign m_valid=rst_n && ((state==RUN && slot_ready[head]) || state==ZERO_EMIT);
    assign m_data=(state==ZERO_EMIT) ? zero_result[emit_lane] : results[head][emit_lane];
    assign m_channel=output_channel;
    assign m_last=(output_channel==COUT-1);
    wire take_output=m_valid && m_ready;
    wire end_group=(emit_lane==P-1 || m_last);
    wire release_slot=(state==RUN && take_output && end_group);
    // Once a group is started it never stalls inside the MAC pipeline.
    wire issue=rst_n && state==RUN && !issued_all &&
               (issue_pos!=0 || reserved<DEPTH || release_slot);
    wire reserve_slot=issue && issue_pos==0;

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
        // Group tags must travel WITH the pipeline; issue_group can be ahead.
        assign weight_q[lane]=(b_group*P+lane<COUT) ? wq : 8'sd0;
        assign bias_q[lane]=(c_group*P+lane<COUT) ? biases[c_group] : 32'sd0;
        assign zero_bias[lane]=(zero_group*P+lane<COUT) ? biases[zero_group] : 32'sd0;
        assign sum_next[lane]=(c_first ? bias_q[lane] : accum[lane]) +
                             {{16{product[lane][15]}},product[lane]};
    end endgenerate

    always @(posedge clk) begin
        if(rst_n && s_valid && s_ready && (!mode || s_data!=0))
            tuple_mem[count]<={input_tap,s_data};
        if(issue) tuple_q<=tuple_mem[issue_pos];
    end

    integer l;
    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;mode<=0;input_tap<=0;count<=0;issue_pos<=0;
            issue_group<=0;zero_group<=0;issued_all<=0;
            head<=0;tail<=0;issue_slot<=0;reserved<=0;slot_ready<=0;
            emit_lane<=0;output_channel<=0;
            a_valid<=0;b_valid<=0;c_valid<=0;
            a_first<=0;b_first<=0;c_first<=0;a_last<=0;b_last<=0;c_last<=0;
            a_group<=0;b_group<=0;c_group<=0;a_slot<=0;b_slot<=0;c_slot<=0;
            b_activation<=0;
            for(l=0;l<P;l=l+1) begin product[l]<=0;accum[l]<=0;zero_result[l]<=0;end
        end else begin
            a_valid<=issue;
            if(issue) begin
                a_first<=(issue_pos==0);a_last<=(issue_pos==count-1);
                a_group<=issue_group;
                a_slot<=(issue_pos==0) ? tail : issue_slot;
            end
            b_valid<=a_valid;
            if(a_valid) begin
                b_activation<=tuple_q[6:0];b_first<=a_first;b_last<=a_last;
                b_group<=a_group;b_slot<=a_slot;
            end
            c_valid<=b_valid;
            if(b_valid) begin
                c_first<=b_first;c_last<=b_last;c_group<=b_group;c_slot<=b_slot;
                for(l=0;l<P;l=l+1) product[l]<=$signed({1'b0,b_activation})*$signed(weight_q[l]);
            end
            case(state)
            IDLE: if(start_valid && start_ready) begin
                mode<=sparse_mode;input_tap<=0;count<=0;issue_pos<=0;
                issue_group<=0;zero_group<=0;issued_all<=0;
                head<=0;tail<=0;issue_slot<=0;reserved<=0;slot_ready<=0;
                emit_lane<=0;output_channel<=0;state<=LOAD;
            end
            LOAD: if(s_valid && s_ready) begin
                if(!mode || s_data!=0) count<=count+1'b1;
                if(input_tap==K-1) state<=PREP;
                else input_tap<=input_tap+1'b1;
            end
            PREP: begin
                if(count==0) begin
                    for(l=0;l<P;l=l+1) zero_result[l]<=zero_bias[l][31] ? 32'sd0 : zero_bias[l];
                    state<=ZERO_EMIT;
                end else state<=RUN;
            end
            RUN: begin
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
                    if(issue_pos==count-1) begin
                        issue_pos<=0;
                        if(issue_group==GROUPS-1) issued_all<=1;
                        else issue_group<=issue_group+1'b1;
                    end else issue_pos<=issue_pos+1'b1;
                end
                if(c_valid) begin
                    for(l=0;l<P;l=l+1) begin
                        accum[l]<=sum_next[l];
                        if(c_last) results[c_slot][l]<=sum_next[l][31] ? 32'sd0 : sum_next[l];
                    end
                    if(c_last) slot_ready[c_slot]<=1;
                end
                if(take_output) begin
                    if(end_group) begin
                        slot_ready[head]<=0;
                        head<=(head==DEPTH-1) ? 0 : head+1'b1;
                        emit_lane<=0;
                    end else emit_lane<=emit_lane+1'b1;
                    if(m_last) state<=IDLE;
                    else output_channel<=output_channel+1'b1;
                end
            end
            ZERO_EMIT: if(take_output) begin
                if(m_last) state<=IDLE;
                else begin
                    output_channel<=output_channel+1'b1;
                    if(end_group) begin
                        zero_group<=zero_group+1'b1;emit_lane<=0;state<=PREP;
                    end else emit_lane<=emit_lane+1'b1;
                end
            end
            default: state<=IDLE;
            endcase
        end
    end
endmodule
