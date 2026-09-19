"""Per-edge protocol models of the three cores driven by the tb_stream_compare producer.

Edge e is a clock posedge index; edge 0 is the edge at which the producer first
offers start (the idle core accepts it there, so window 0 starts at edge 0).
Signals driven by the testbench at the negedge before edge e are the values the
core sees at edge e. The producer offers the next start on the same negedge it
drops the last input beat. m_ready seen at edge e is ready_at(e-1) (decided at
the negedge after edge e-1, when the testbench cycle counter equals e-1).

Only timing is modeled here: the tuple counts per window (K in dense mode, the
nonzero count in sparse mode) decide how many taps each group issues. Values are
checked by the RTL testbench against the integer oracle.

Returns [(start_edge, end_edge)] per window. end_edge is the edge at which the
COUT-1 output of that window is accepted.
"""


def ready_at(t, stall, pattern, k):
    if pattern == 1:
        return not (t % 113 < 17 or t % 7 == 0)
    if pattern == 2:
        return t >= k+200 and t % 19 >= 7
    return t % 16 >= stall


class Producer:
    """start_valid / s_valid sequence of tb_stream_compare for a list of windows."""

    def __init__(self, k, taps, gaps):
        self.k, self.taps, self.gaps = k, taps, gaps
        self.w = 0                 # window being offered or loaded
        self.phase = 'start'       # 'start', 'input', 'done'
        self.idx = 0
        self.age = 0

    def signals(self):
        if self.phase == 'start':
            return True, False, False
        if self.phase == 'input':
            s_valid = not (self.gaps and self.age % 9 in (3, 4))
            return False, s_valid, True
        return False, False, False

    def after_edge(self, take_start, take_input):
        if self.phase == 'start' and take_start:
            self.phase, self.idx, self.age = 'input', 0, 0
        elif self.phase == 'input':
            if take_input:
                self.idx += 1
            self.age += 1
            if self.idx == self.k:
                self.w += 1
                self.phase = 'start' if self.w < len(self.taps) else 'done'

    def tuple_of_beat(self):
        # Sparse-mode windows store only nonzero beats: the model needs only the
        # count, so beat idx is "nonzero" iff idx < taps (order does not matter).
        return self.idx < self.taps[self.w]


class V1:
    """sparse_window_mac: load, then per group issue -> drain, serial."""

    def __init__(self, k, cout, p, depth):
        self.k, self.cout, self.p = k, cout, p
        self.groups = (cout+p-1)//p
        self.state = 'IDLE'
        self.count = self.input_tap = self.issue_pos = self.group = self.emit_lane = 0
        self.pending = None

    def combinational(self, m_ready):
        c = {}
        c['start_ready'] = self.state == 'IDLE'
        c['s_ready'] = self.state == 'LOAD'
        c['m_valid'] = self.state == 'EMIT'
        c['m_last'] = self.group*self.p+self.emit_lane == self.cout-1
        c['take_output'] = c['m_valid'] and m_ready
        c['issue'] = self.state == 'RUN' and self.issue_pos < self.count
        return c

    def update(self, e, c, take_start, take_input, store):
        if self.state == 'IDLE':
            if take_start:
                self.count = self.input_tap = self.group = 0
                self.state = 'LOAD'
        elif self.state == 'LOAD':
            if take_input:
                self.count += store
                if self.input_tap == self.k-1:
                    self.state = 'PREP'
                else:
                    self.input_tap += 1
        elif self.state == 'PREP':
            self.emit_lane = self.issue_pos = 0
            self.state = 'EMIT' if self.count == 0 else 'RUN'
        elif self.state == 'RUN':
            if self.pending == e:
                self.state, self.emit_lane, self.pending = 'EMIT', 0, None
            if c['issue']:
                if self.issue_pos == self.count-1:
                    self.pending = e+3
                self.issue_pos += 1
        elif self.state == 'EMIT':
            if c['take_output']:
                if c['m_last']:
                    self.state = 'IDLE'
                elif self.emit_lane == self.p-1:
                    self.group += 1
                    self.state = 'PREP'
                else:
                    self.emit_lane += 1


class SlotFifo:
    """Reserved result slots shared by v2 and v3."""

    def __init__(self, depth):
        self.depth = depth
        self.head = self.tail = self.issue_slot = self.reserved = 0
        self.slot_ready = [False]*depth
        self.pending = []   # (edge, slot): slot_ready set after that edge

    def reset(self):
        self.__init__(self.depth)

    def apply_pending(self, e):
        keep = []
        for edge, slot in self.pending:
            if edge == e:
                assert not self.slot_ready[slot], 'overwriting an unconsumed result'
                self.slot_ready[slot] = True
            else:
                keep.append((edge, slot))
        self.pending = keep


class V2:
    """continuous_window_mac: one window at a time, groups issued back to back."""

    def __init__(self, k, cout, p, depth):
        self.k, self.cout, self.p, self.depth = k, cout, p, depth
        self.groups = (cout+p-1)//p
        self.state = 'IDLE'
        self.count = self.input_tap = self.issue_pos = self.issue_group = self.zero_group = 0
        self.issued_all = False
        self.emit_lane = self.output_channel = 0
        self.fifo = SlotFifo(depth)

    def combinational(self, m_ready):
        f = self.fifo
        c = {}
        c['start_ready'] = self.state == 'IDLE'
        c['s_ready'] = self.state == 'LOAD'
        c['m_valid'] = (self.state == 'RUN' and f.slot_ready[f.head]) or self.state == 'ZERO_EMIT'
        c['m_last'] = self.output_channel == self.cout-1
        c['take_output'] = c['m_valid'] and m_ready
        c['end_group'] = self.emit_lane == self.p-1 or c['m_last']
        c['release'] = self.state == 'RUN' and c['take_output'] and c['end_group']
        c['issue'] = (self.state == 'RUN' and not self.issued_all and
                      (self.issue_pos != 0 or f.reserved < self.depth or c['release']))
        c['reserve'] = c['issue'] and self.issue_pos == 0
        return c

    def update(self, e, c, take_start, take_input, store):
        f = self.fifo
        if self.state == 'IDLE':
            if take_start:
                self.count = self.input_tap = self.issue_pos = self.issue_group = self.zero_group = 0
                self.issued_all = False
                self.emit_lane = self.output_channel = 0
                f.reset()
                self.state = 'LOAD'
        elif self.state == 'LOAD':
            if take_input:
                self.count += store
                if self.input_tap == self.k-1:
                    self.state = 'PREP'
                else:
                    self.input_tap += 1
        elif self.state == 'PREP':
            self.state = 'ZERO_EMIT' if self.count == 0 else 'RUN'
        elif self.state == 'RUN':
            f.apply_pending(e)
            f.reserved += int(c['reserve'])-int(c['release'])
            slot = f.tail if self.issue_pos == 0 else f.issue_slot
            if c['reserve']:
                f.issue_slot = f.tail
                f.tail = (f.tail+1) % self.depth
            if c['issue']:
                if self.issue_pos == self.count-1:
                    f.pending.append((e+3, slot))
                    self.issue_pos = 0
                    if self.issue_group == self.groups-1:
                        self.issued_all = True
                    else:
                        self.issue_group += 1
                else:
                    self.issue_pos += 1
            if c['take_output']:
                if c['end_group']:
                    f.slot_ready[f.head] = False
                    f.head = (f.head+1) % self.depth
                    self.emit_lane = 0
                else:
                    self.emit_lane += 1
                if c['m_last']:
                    self.state = 'IDLE'
                else:
                    self.output_channel += 1
        elif self.state == 'ZERO_EMIT':
            if c['take_output']:
                if c['m_last']:
                    self.state = 'IDLE'
                else:
                    self.output_channel += 1
                    if c['end_group']:
                        self.zero_group += 1
                        self.emit_lane = 0
                        self.state = 'PREP'
                    else:
                        self.emit_lane += 1


class V3:
    """overlapped_window_mac: double-buffered tuple RAM, continuous issue across windows."""

    def __init__(self, k, cout, p, depth):
        self.k, self.cout, self.p, self.depth = k, cout, p, depth
        self.groups = (cout+p-1)//p
        self.load_active = False
        self.load_bank = self.run_bank = 0
        self.count = self.input_tap = 0
        self.busy = [False, False]
        self.loaded = [False, False]
        self.win_count = [0, 0]
        self.issue_pos = self.issue_group = 0
        self.emit_lane = self.output_channel = 0
        self.fifo = SlotFifo(depth)

    def combinational(self, m_ready):
        f = self.fifo
        c = {}
        c['start_ready'] = not self.load_active and not self.busy[self.load_bank]
        c['s_ready'] = self.load_active
        run_ready = self.loaded[self.run_bank]
        eff = max(1, self.win_count[self.run_bank])
        c['last_pos'] = self.issue_pos == eff-1
        c['m_valid'] = f.slot_ready[f.head]
        c['m_last'] = self.output_channel == self.cout-1
        c['take_output'] = c['m_valid'] and m_ready
        c['end_group'] = self.emit_lane == self.p-1 or c['m_last']
        c['release'] = c['take_output'] and c['end_group']
        c['issue'] = run_ready and (self.issue_pos != 0 or f.reserved < self.depth or c['release'])
        c['reserve'] = c['issue'] and self.issue_pos == 0
        return c

    def update(self, e, c, take_start, take_input, store):
        f = self.fifo
        if take_start:
            self.load_active = True
            self.busy[self.load_bank] = True
            self.count = self.input_tap = 0
        elif take_input:
            self.count += store
            if self.input_tap == self.k-1:
                self.loaded[self.load_bank] = True
                self.win_count[self.load_bank] = self.count
                self.load_active = False
                self.load_bank ^= 1
            else:
                self.input_tap += 1
        f.apply_pending(e)
        f.reserved += int(c['reserve'])-int(c['release'])
        assert 0 <= f.reserved <= self.depth
        slot = f.tail if self.issue_pos == 0 else f.issue_slot
        if c['reserve']:
            f.issue_slot = f.tail
            f.tail = (f.tail+1) % self.depth
        if c['issue']:
            if c['last_pos']:
                f.pending.append((e+3, slot))
                self.issue_pos = 0
                if self.issue_group == self.groups-1:
                    self.issue_group = 0
                    self.loaded[self.run_bank] = False
                    self.busy[self.run_bank] = False
                    self.run_bank ^= 1
                else:
                    self.issue_group += 1
            else:
                self.issue_pos += 1
        if c['take_output']:
            if c['end_group']:
                f.slot_ready[f.head] = False
                f.head = (f.head+1) % self.depth
                self.emit_lane = 0
            else:
                self.emit_lane += 1
            self.output_channel = 0 if c['m_last'] else self.output_channel+1


def stream_model(k, cout, p, depth, impl, taps, stall=0, pattern=0, gaps=0, limit=400000000):
    core = {0: V1, 1: V2, 2: V3}[impl](k, cout, p, depth)
    prod = Producer(k, taps, gaps)
    starts, ends = [], []
    e = 0
    while len(ends) < len(taps):
        m_ready = ready_at(e-1, stall, pattern, k) if e > 0 else True
        start_valid, s_valid, _ = prod.signals()
        c = core.combinational(m_ready)
        take_start = start_valid and c['start_ready']
        take_input = s_valid and c['s_ready']
        store = 0
        if take_input:
            # Dense windows store every beat; sparse windows store nonzero beats.
            store = 1 if prod.taps[prod.w] == k or prod.tuple_of_beat() else 0
        if take_start:
            starts.append(e)
        if c['take_output'] and c['m_last']:
            ends.append(e)
        core.update(e, c, take_start, take_input, store)
        prod.after_edge(take_start, take_input)
        e += 1
        if e > limit:
            raise RuntimeError('model timeout')
    return list(zip(starts, ends))


if __name__ == '__main__':
    import sys
    k, c, p, d, impl = (int(v) for v in sys.argv[1:6])
    taps = [int(v) for v in sys.argv[6:]] or [k, k//2, 0, k]
    for w, (s, t) in enumerate(stream_model(k, c, p, d, impl, taps)):
        print(w, s, t, t-s)
