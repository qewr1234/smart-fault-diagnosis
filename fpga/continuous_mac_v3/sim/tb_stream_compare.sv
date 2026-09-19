`timescale 1ns/1ps
// Streams a sequence of windows back to back into one core and checks every
// output against the integer gold values. Works for all three cores:
//   IMPL=0 sparse_window_mac (v1)   IMPL=1 continuous_window_mac (v2)
//   IMPL=2 overlapped_window_mac (v3)
// The producer offers the next start on the same negedge it drops the last
// input beat, so a core that can overlap windows is allowed to. Cores that
// cannot simply hold start_ready low and are measured on the same stream.
// Cycle metric per window: start-accept edge and last-output-accept edge, both
// as posedge indices. The stream total is end[last]-start[0].
module tb_stream_compare;
    parameter integer K=27,COUT=9,P=2,DEPTH=2,N=8,IMPL=2;
    parameter integer STALL_LEN=0,PATTERN=0,INPUT_GAPS=0,MODE_SEQ=2,RESET_TEST=0;
    localparam integer KW=(K<2 ? 1:$clog2(K)),CW=(COUT<2 ? 1:$clog2(COUT));
    localparam integer NWIN=(MODE_SEQ==2) ? 2*N : N;
    localparam integer GROUPS=(COUT+P-1)/P;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,cfg_valid=0,cfg_is_bias=0,start_valid=0,sparse_mode=0,s_valid=0,m_ready=1;
    wire cfg_ready,start_ready,s_ready,m_valid,m_last;
    reg [CW-1:0] cfg_channel=0;
    reg [KW-1:0] cfg_tap=0;
    reg signed [31:0] cfg_data=0;
    reg [6:0] s_data=0;
    wire signed [31:0] m_data;wire [CW-1:0] m_channel;
    wire issue,tap_issue;wire [31:0] issue_pos;
    generate if(IMPL==0) begin: v1
        sparse_window_mac #(.K(K),.COUT(COUT),.P(P)) dut(.*);
        assign issue=dut.issue;assign tap_issue=dut.issue;assign issue_pos=dut.issue_pos;
    end else if(IMPL==1) begin: v2
        continuous_window_mac #(.K(K),.COUT(COUT),.P(P),.DEPTH(DEPTH)) dut(.*);
        assign issue=dut.issue;assign tap_issue=dut.issue;assign issue_pos=dut.issue_pos;
        always @(posedge clk) if(rst_n) begin
            if(dut.reserved>DEPTH) $fatal(1,"result reservations overflow");
            if(dut.release_slot && dut.reserved==0) $fatal(1,"result reservation underflow");
            if(dut.c_valid && dut.c_last && dut.slot_ready[dut.c_slot]) $fatal(1,"overwriting an unconsumed result");
        end
    end else begin: v3
        overlapped_window_mac #(.K(K),.COUT(COUT),.P(P),.DEPTH(DEPTH)) dut(.*);
        assign issue=dut.issue;assign tap_issue=dut.tap_issue;assign issue_pos=dut.issue_pos;
        always @(posedge clk) if(rst_n) begin
            if(dut.reserved>DEPTH) $fatal(1,"result reservations overflow");
            if(dut.release_slot && dut.reserved==0) $fatal(1,"result reservation underflow");
            if(dut.c_valid && dut.c_last && dut.slot_ready[dut.c_slot]) $fatal(1,"overwriting an unconsumed result");
            if(dut.head>=DEPTH || dut.tail>=DEPTH) $fatal(1,"FIFO pointer outside depth");
            // Bank ownership: never write the bank being issued, never issue an unloaded bank.
            if(dut.store_input && dut.issue && dut.load_bank==dut.run_bank) $fatal(1,"tuple bank read/write collision");
            if(dut.take_start && dut.busy[dut.load_bank]) $fatal(1,"start accepted into a busy bank");
            if(dut.issue && !dut.loaded[dut.run_bank]) $fatal(1,"issue from an unloaded bank");
            if(dut.take_cfg && !dut.core_idle) $fatal(1,"configuration accepted while not idle");
        end
    end endgenerate
    reg [7:0] weights [0:COUT*K-1];
    reg [31:0] biases [0:COUT-1];
    reg [7:0] inputs [0:N*K-1];
    reg [31:0] gold [0:N*COUT-1];
    string vec;
    // Posedge index. At the negedge after posedge e, cycle==e.
    integer cycle=0;
    reg start_acc=0,s_acc=0,m_acc=0,m_acc_last=0;
    reg [31:0] m_acc_data;reg [CW-1:0] m_acc_ch;
    always @(posedge clk) begin
        cycle<=cycle+1;
        start_acc<=start_valid&&start_ready;
        s_acc<=s_valid&&s_ready;
        m_acc<=m_valid&&m_ready;m_acc_last<=m_last;m_acc_data<=m_data;m_acc_ch<=m_channel;
    end
    integer start_cycle [0:NWIN-1];
    integer end_cycle [0:NWIN-1];
    integer t0=0;
    integer win_out=0,ch_out=0,checked=0,input_beats=0,issues=0,previous_issue=-1;
    integer csv,ci,t,idx,age,w,nz,fr,md,expect_issues,output_stalls=0;
    reg active=0,force_block=0,held=0;
    reg [31:0] held_data;reg [CW-1:0] held_channel;reg held_last;
    function automatic integer win_frame(input integer wi);
        win_frame=(MODE_SEQ==2) ? wi/2 : wi;
    endfunction
    function automatic integer win_mode(input integer wi);
        win_mode=(MODE_SEQ==2) ? wi%2 : MODE_SEQ;
    endfunction
    function automatic integer ready_at(input integer elapsed);
        begin
            if(PATTERN==1) ready_at=!((elapsed%113)<17 || elapsed%7==0);
            else if(PATTERN==2) ready_at=(elapsed>=K+200 && elapsed%19>=7);
            else ready_at=(elapsed%16>=STALL_LEN);
        end
    endfunction
    // m_ready for posedge e+1 is decided at the negedge after posedge e (cycle==e).
    always @(negedge clk) m_ready=(!force_block && ready_at(cycle-t0));
    // Output stability under stall, output checking, bookkeeping.
    always @(posedge clk) begin
        if(!rst_n) held<=0;
        else begin
            if(held && (!m_valid || m_data!==held_data || m_channel!==held_channel || m_last!==held_last))
                $fatal(1,"output changed while stalled");
            held<=m_valid&&!m_ready;
            if(m_valid&&!m_ready) begin held_data<=m_data;held_channel<=m_channel;held_last<=m_last;end
            if(active && m_valid && !m_ready) output_stalls<=output_stalls+1;
            if(active && tap_issue) begin
                if(issue_pos>0 && cycle-previous_issue!=1) $fatal(1,"within-group issue interval != 1");
                previous_issue<=cycle;issues<=issues+1;
            end
        end
        if(cycle>400000000) $fatal(1,"global timeout");
    end
    always @(negedge clk) if(active) begin
        if(s_acc) input_beats=input_beats+1;
        if(m_acc) begin
            if(win_out>=NWIN) $fatal(1,"output after the last window");
            fr=win_frame(win_out);
            if(m_acc_ch!==ch_out || m_acc_data!==gold[fr*COUT+ch_out] || m_acc_last!==(ch_out==COUT-1))
                $fatal(1,"mismatch window=%0d frame=%0d mode=%0d ch=%0d got=%h expected=%h",win_out,fr,win_mode(win_out),ch_out,m_acc_data,gold[fr*COUT+ch_out]);
            checked=checked+1;
            if(ch_out==COUT-1) begin end_cycle[win_out]=cycle;win_out=win_out+1;ch_out=0;end
            else ch_out=ch_out+1;
        end
    end
    task configure;
        begin
            for(ci=0;ci<COUT;ci=ci+1) begin
                for(t=0;t<K;t=t+1) begin
                    @(negedge clk);cfg_valid=1;cfg_is_bias=0;cfg_channel=ci;cfg_tap=t;
                    cfg_data={{24{weights[ci*K+t][7]}},weights[ci*K+t]};
                    start_valid=1; // configuration must win in the idle core
                    @(posedge clk);if(!cfg_ready || start_ready) $fatal(1,"configuration priority failed");
                end
                @(negedge clk);cfg_valid=1;cfg_is_bias=1;cfg_channel=ci;cfg_data=biases[ci];
                @(posedge clk);if(!cfg_ready || start_ready) $fatal(1,"bias configuration priority failed");
            end
            @(negedge clk);cfg_valid=0;start_valid=0;
        end
    endtask
    // Offer start (called at a negedge), return at the negedge after acceptance.
    task offer_start(input integer mode_value,output integer acc_cycle);
        begin
            sparse_mode=mode_value;start_valid=1;
            @(negedge clk);while(!start_acc) @(negedge clk);
            acc_cycle=cycle;start_valid=0;
        end
    endtask
    // Send K beats of frame f (called at a negedge), return at the negedge after the last accept.
    task send_input(input integer f,input integer limit);
        begin
            idx=0;age=0;
            while(idx<limit) begin
                s_valid=!(INPUT_GAPS && (age%9==3 || age%9==4));s_data=inputs[f*K+idx][6:0];
                if(inputs[f*K+idx]>127) $fatal(1,"input outside proof domain");
                @(negedge clk);if(s_acc) idx=idx+1;
                age=age+1;
            end
            s_valid=0;
        end
    endtask
    task abort_and_reset;
        begin
            @(negedge clk);active=0;rst_n=0;s_valid=0;start_valid=0;cfg_valid=0;
            repeat(3) @(negedge clk);
            if(m_valid) $fatal(1,"valid survived reset");
            if(IMPL>=1 && (cfg_ready || start_ready || s_ready)) $fatal(1,"ready high during reset");
            rst_n=1;force_block=0;
            repeat(5) @(negedge clk);
            if(m_valid || !start_ready) $fatal(1,"stale output or not idle after reset");
        end
    endtask
    integer dummy;
    initial begin
        if(!$value$plusargs("VEC=%s",vec)) $fatal(1,"missing +VEC");
        $readmemh({vec,"/weights.hex"},weights);$readmemh({vec,"/bias.hex"},biases);
        $readmemh({vec,"/input.hex"},inputs);$readmemh({vec,"/gold.hex"},gold);
        csv=$fopen("rtl_cycles.csv","w");if(!csv) $fatal(1,"CSV open failed");
        $fwrite(csv,"window,frame,sparse_mode,K,COUT,P,depth,implementation,nonzero_inputs,start_cycle,end_cycle,stall_len,pattern,input_gaps\n");
        repeat(4) @(negedge clk);rst_n=1;
        configure();
        if(RESET_TEST) begin
            // 1. abort while waiting for input
            @(negedge clk);offer_start(0,dummy);abort_and_reset();
            // 2. abort with the MAC pipeline active
            @(negedge clk);offer_start(0,dummy);send_input(1,K);
            wait(issue);repeat(6) @(negedge clk);abort_and_reset();
            // 3. abort with output blocked (and, for v3, a second window loading)
            force_block=1;@(negedge clk);offer_start(0,dummy);send_input(1,K);wait(m_valid);
            repeat(24) @(negedge clk);
            if(IMPL==2) begin
                @(negedge clk);offer_start(1,dummy);send_input(0,(K>=2) ? K/2 : 1);
                repeat(8) @(negedge clk);
            end
            abort_and_reset();
        end
        // Main stream.
        expect_issues=0;
        for(w=0;w<NWIN;w=w+1) begin
            nz=0;for(t=0;t<K;t=t+1) if(inputs[win_frame(w)*K+t]!=0) nz=nz+1;
            expect_issues=expect_issues+GROUPS*(win_mode(w) ? nz : K);
        end
        @(negedge clk);active=1;
        for(w=0;w<NWIN;w=w+1) begin
            offer_start(win_mode(w),start_cycle[w]);
            if(w==0) t0=start_cycle[0];
            send_input(win_frame(w),K);
        end
        wait(win_out==NWIN);@(negedge clk);
        if(input_beats!=NWIN*K) $fatal(1,"input beat count %0d != %0d",input_beats,NWIN*K);
        if(issues!=expect_issues) $fatal(1,"issued tap count %0d != %0d",issues,expect_issues);
        repeat(5) @(negedge clk);
        if(m_valid || !start_ready || !cfg_ready) $fatal(1,"not idle after completion");
        for(w=0;w<NWIN;w=w+1) begin
            nz=0;for(t=0;t<K;t=t+1) if(inputs[win_frame(w)*K+t]!=0) nz=nz+1;
            $fwrite(csv,"%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                w,win_frame(w),win_mode(w),K,COUT,P,DEPTH,IMPL,nz,start_cycle[w],end_cycle[w],STALL_LEN,PATTERN,INPUT_GAPS);
        end
        $fclose(csv);
        $display("PASS ALL checked_values=%0d windows=%0d total_cycles=%0d K=%0d COUT=%0d P=%0d DEPTH=%0d IMPL=%0d STALL=%0d PATTERN=%0d GAPS=%0d RESET=%0d",
                 checked,NWIN,end_cycle[NWIN-1]-start_cycle[0],K,COUT,P,DEPTH,IMPL,STALL_LEN,PATTERN,INPUT_GAPS,RESET_TEST);
        $finish;
    end
endmodule
