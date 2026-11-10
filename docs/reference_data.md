# `hal_pot_code` 公開データ互換性

確認日: 2026-07-28

参照先は MIT License の
[`tkmiriya/hal_pot_code`](https://github.com/tkmiriya/hal_pot_code) である。
同 repository の sample は `L=48`, XiXi, wall source, 4 configurations,
格子 cutoff 2.194 GeV, pion mass 510 MeV と説明されている。

## Python 版の LO 規約

`notes/hal_pot_single_ch.py` は reduced mass `m_red=m/2` を使い、次の3成分を
構成する。

```text
spatial    = (∇²R/R) / (2 m_red)
first_time = -(R(t+1)-R(t-1)) / (2 R(t))
second_time= (R(t+1)-2R(t)+R(t-1)) / (8 m_red R(t))
```

したがって、`HALQCD.lo_potential_equal_mass(...; mass=m)` の
`spatial`, `first_time`, `second_time` と同じ規約になる。Python 版は
nearest-neighbor periodic lattice Laplacian を使う。

## R-correlator binary

Python の出力 `Rcorr_*.dat` は NumPy の native-endian `complex128` であり、
論理 shape は `(bin,x,y,z)`、flat 化は C order である。Julia importer
`load_hal_pot_rcorr` はこれを `(x,y,z,bin)` に変換する。

```julia
rcorr = load_hal_pot_rcorr(
    "Rcorr_1s0_xixi_t010_2bin_4conf.dat";
    lattice_size=48,
    bins=2,
)
```

期待 file size は `16 * bins * L^3` bytes である。不一致は黙って reshape
せず `DimensionMismatch` とする。

## 圧縮 NBS の double-singlet importer

decompressed NBS wavefunction は big-endian `Float64` で、Python 版は
sink/source spin index、real/imag index、parity projection、spin projection、
さらに cubic `A1+` projection を適用する。元の compressed sample は外部
decompressor `PH1.compress48` に依存する。しかし公開 XiXi sample の圧縮
container は4個の Fortran record からなり、cubic orbit representative
ごとに32個の `Float64` を保存する。`1S0` では sink と source の二重 singlet
縮約が空間回転に対して scalar になるため、一般の spinor field を展開せず
`load_ph1_compressed_a1plus_singlet` で直接3次元場を復元できる。

これは一般 decompressor ではない。triplet、tensor、非自明な cubic
representation には使わない。

## 2026-07-29 の SSH 回帰

SSH 先の Linux 計算機の上、Julia 1.10.10、reference commit
`3446378916df74eca887cb36c380ed0ab151192e` で、12個の圧縮 NBS file と Xi
2点相関関数から `t=10` の `1S0` potential を純 Julia で再構成した。
4 input sample を2個ずつ binning し、2個の delete-one jackknife sample を
作った。

公開 text table との全2925 representative 比較では次を得た。

| component | mean 最大絶対差 | jackknife error 最大絶対差 |
|---|---:|---:|
| spatial | `3.4803e-7` | `5.0000e-10` |
| first time | `4.9995e-10` | `2.8417e-9` |
| second time | `4.9818e-10` | `4.9879e-10` |
| total | `8.8502e-8` | `3.9369e-9` |

最大差は公開 table の10進印字丸めと整合する。全 `48^3=110592` site が
division mask を通過した。初回の wall time は11.68秒、最大 resident memory
は427380 KiB だった。

ただし sample directory は同一 gauge-configuration label `000410` の異なる
`rot_*` measurement に見える。この sample の jackknife error を独立な gauge
configuration に基づく物理的統計誤差とは解釈しない。

## 回帰試験で記録する量

- reference repository commit
- Python/Julia version
- configuration 数と bin size
- time slice
- hadron mass と単位
- spatial/first-time/second-time/total の tolerance
- mask された site と理由
- axis order と flatten order
