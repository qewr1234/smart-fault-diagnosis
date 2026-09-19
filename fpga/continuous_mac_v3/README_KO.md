# Continuous MAC v3 — 창 간 중첩(overlapped) 코어

`continuous_mac_rtl_v2` 패키지의 다음 단계입니다. v2는 **한 창 안에서** 출력 채널 그룹을 연속 발행했습니다.
v3(`overlapped_window_mac.sv`)는 같은 예약 메커니즘을 **창 사이**로 확장합니다.

- tuple RAM을 2개 bank로 두어, 창 w가 계산·출력 중일 때 창 w+1의 K개 입력을 받습니다.
- 발행 엔진은 창 w의 마지막 그룹에서 창 w+1의 첫 그룹으로 bubble 없이 넘어갑니다(다음 창이 적재되어 있으면).
- 결과 슬롯 예약(DEPTH), 출력 순서, 산술, 인터페이스는 v2와 동일합니다. `start_ready`가 IDLE 외에도 bank가 비어 있으면 올라간다는 점만 다릅니다.
- 전부 0인 sparse 창은 별도 상태 없이 "곱을 0으로 강제한 pseudo tap" 한 개를 그룹당 발행해 같은 파이프라인으로 bias+ReLU를 냅니다.
- configuration은 적재 중/적재된 창이 없고 진행 중 결과가 없을 때(`core_idle`)만 받습니다. start와 동시에 오면 configuration이 우선합니다.

## 왜 이것인가

v2 검증 로그 기준 evaluation sparse 창의 cycle 구성:

| P | 총 cycle | LOAD (K=576) | issue (GROUPS × nnz) | 그 외 |
|--:|--:|--:|--:|--:|
| 2 | 21,558 | 576 (2.7%) | 20,976 (97.3%) | 6 |
| 8 | 5,832 | 576 (9.9%) | 5,244 (89.9%) | 12 |

v2가 없앤 그룹 오버헤드는 이제 창당 6~12 cycle뿐입니다. 남은 것은 LOAD와 issue이고, LOAD는 창 간 중첩으로 숨길 수 있습니다.
P가 클수록 LOAD 비중이 커지므로 개선 폭도 커집니다.

## 실행

Python 3.9+, Icarus Verilog(`iverilog`, `vvp`)가 필요합니다.

```
python scripts/run_stream_checks.py --suite smoke --jobs 4   # 합성 벡터 100개 설정
python scripts/run_stream_checks.py --suite full  --jobs 4   # + 실제 VGG11 features.3 창 512개 × 2 모드
```

`sim/tb_stream_compare.sv`는 창을 **back-to-back**으로 흘립니다. 마지막 입력 beat를 내린 같은 negedge에 다음 start를 올리므로
중첩이 가능한 코어는 중첩하고, v1/v2는 `start_ready`를 낮춰 직렬로 처리됩니다. 세 코어가 같은 스트림, 같은 `m_ready` 패턴을 받습니다.

측정값: 창별 start 수락 edge와 마지막 출력 수락 edge. 스트림 전체 cycle = `end[last] - start[0]`.
중첩 코어는 창당 latency(end−start)가 대기 시간 때문에 커질 수 있으므로 **throughput(총 cycle / 창 수)** 으로 비교해야 합니다.

`scripts/stream_model.py`는 세 코어의 프로토콜을 edge 단위로 다시 구현한 독립 cycle 모델입니다. 모든 창의 (start, end)가 RTL과 일치해야 PASS입니다.
같은 사람이 작성한 두 번째 구현이므로 "독립"의 강도는 v2 패키지의 해석적 모델과 같은 수준입니다.

검사 항목: 출력값(정수 oracle), 출력 순서/`m_last`, stall 중 출력 안정성, 그룹 내 발행 간격 1, 예약 overflow/underflow/덮어쓰기,
tuple bank 읽기/쓰기 충돌 없음, 적재되지 않은 bank 발행 없음, 비-idle 중 configuration 거부, 발행 tap 총수, 입력 beat 총수,
리셋(입력 대기 중 / 파이프라인 동작 중 / 출력 막힌 상태 + v3는 두 번째 창 적재 중) 후 재실행.

## 결과

`RESULTS_KO.md`를 보세요. 동봉 실행 로그는 `verification/`에 있습니다.

## 다음 후보: bank-parallel sparse issue (제안 B, RTL 아직 없음)

`scripts/bank_parallel_model.py`는 tap을 `tap % T`로 T개 bank에 인터리브해 cycle당 T개 tuple을 발행하는 구조의 cycle을 실제 창으로 투영합니다.
그룹당 cycle이 nnz에서 max(bank별 nnz)로 바뀝니다. 정적 인터리브의 불균형 손실(evaluation): T=2 6.2%, T=4 15.7%, T=8 35.1%.
곱셈기가 P×T배로 늘고 weight bank 읽기 폭이 T배가 되므로, 자원과 Fmax 대가는 합성 없이는 알 수 없습니다.

## 범위 밖

Vivado 합성/배치배선, Fmax, 자원, 전력, AXI/DMA, 전체 CNN 시간, 보드 동작은 이 패키지에서 확인하지 않았습니다.
v2 패키지의 `synth_vivado.tcl`에 `overlapped_window_mac`을 추가하면 같은 OOC 흐름으로 세 코어를 비교할 수 있습니다.
