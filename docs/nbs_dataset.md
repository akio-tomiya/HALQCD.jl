# NBS correlator dataset

更新日: 2026-07-31

## 目的

`NBSCorrelatorDataset` は、空間分解した二粒子四点相関と、その規格化に使う
二つの一粒子相関を、sourceや標本を混同せず保持するためのin-memory型である。
同じ情報は版番号付きdisk形式で保存・復元できる。

## 軸

四点相関の軸順は固定である。

```text
(x, y, z, time, source, sample)
```

二つの一粒子相関の軸順も固定である。

```text
(time, source, sample)
```

最終軸を`configuration`ではなく`sample`と呼ぶ。raw gauge configurationと、
delete-one jackknifeなどの派生標本を同じ言葉で呼ばないためである。
`sample_kind`は次のどちらかである。

- `:raw_configuration`
- `:derived_resample`

派生標本では`resampling_method`と`parent_configuration_ids`が必須である。
sourceはsampleと別軸なので、同じconfiguration上の複数sourceを独立した
configurationとして数えない。

## 最小例

```julia
using HALQCD

F = zeros(ComplexF64, 4, 4, 4, 8, 2, 10)
C1 = ones(ComplexF64, 8, 2, 10)
C2 = ones(ComplexF64, 8, 2, 10)

metadata = (
    channel="XiXi",
    spin="singlet",
    source_operator="point-Xi",
    sink_operator="local-Xi",
    temporal_boundary_condition="antiperiodic-quark",
    lattice_shape=(4, 4, 4, 8),
    lattice_spacing=(1.0, 1.0, 1.0, 1.0),
    magnetic_flux=0,
    sample_ids=["cfg-$i" for i in 1:10],
    source_ids=["src-0", "src-1"],
    source_time=1,
    source_positions=["(1, 1, 1, 1)", "(2, 1, 1, 1)"],
    code_commit="full-git-commit",
)

dataset = NBSCorrelatorDataset(F, C1, C2; metadata=metadata)
R = r_correlator(dataset)
potential = lo_potential_equal_mass(dataset, 2; mass=1.5)
```

datasetからポテンシャルを抽出する場合、空間・時間格子間隔は
`metadata.lattice_spacing`を既定値として使う。解析時に別の値を仮定する場合だけ
`spacing=(a_x,a_y,a_z), at=a_t`を明示して上書きする。

raw configurationからdelete-one標本を作る場合は、四点相関と二つの一粒子相関を
別々に平均してから各標本の規格化を行う。

```julia
jackknife_dataset = delete_one_jackknife_dataset(dataset)
jackknife_R = r_correlator(jackknife_dataset)
```

configurationごとに先に`R`を作ってからdelete-one平均する経路とは、一般には
一致しない。規格化が非線形だからである。

diskへ保存し、別のprocessやcheckoutで復元するには次を使う。

```julia
write_nbs_dataset("output/nbs-example", dataset)
loaded = read_nbs_dataset("output/nbs-example")
```

保存先directoryには、人間が読める`manifest.toml`、その
`manifest.sha256`、および三つのlittle-endian binary payloadを置く。manifest
にはformat/schema version、軸順、型、shape、storage order、payload hash、
metadataを記録する。既存の保存先は上書きしない。

v1で保存できる要素型は`Float32`、`Float64`、`ComplexF32`、
`ComplexF64`である。metadataはTOML scalarと一次元sequenceに制限する。
manifest hashは偶発的破損や片方だけの編集を検出するが、攻撃者がmanifestと
hashを同時に変更する場合の電子署名ではない。

constructorは次を拒否する。

- 未対応schema versionまたは未知のsample種別。
- 四点相関と一粒子相関のtime/source/sample軸不一致。
- lattice shapeと配列shapeの不一致。
- 空または重複したsample/source ID。
- 空のchannel、spin、演算子、境界条件、commit。
- 非整数または非正のlattice shape。
- 非正または非有限なlattice spacing。
- 派生標本で再標本化法または元configuration IDがない場合。

IDとshapeはconstructor内でimmutable tupleへ変換する。呼び出し側が元のID
vectorを後から変更しても、検証済みmetadataは変化しない。

## 現在の限界

- dataset自体は統計的独立性を証明しない。configurationの生成履歴や自己相関は
  別途検査する。
- `sample_kind`は標本の意味を区別するが、raw標本と派生標本を同じ解析で混ぜて
  よいことを意味しない。
- source/sink smearing、quark mass、solver residualなど、計算固有metadataは
  `metadata`へ追加して保存する。production file schemaでは必須fieldを増やす。
- source時刻を比較するデータでは、`source_time`と全`source_positions`の
  時間成分を両方保存する。比較CLIは両者の一致、全四次元座標の格子範囲、
  比較する二datasetの空間source座標の一致を確認し、一つでも不正なら解析を
  開始しない。solver閾値、重心和、相対座標、単一hadron spin縮約、
  規格化方針も固定条件として照合する。旧データでは`source_positions`から
  時刻を復元する。六quark縮約方式が一方でも未記録なら既定では拒否する。
  commit差分や独立回帰で縮約経路を外部監査した旧データだけ、
  `HALQCD_ALLOW_UNRECORDED_SIX_QUARK_CONTRACTION=true`を明示して許可し、
  その許可を比較結果にも保存する。
- v1は単一directory形式であり、大規模ensemble向けのchunking、圧縮、並列I/O
  はまだ扱わない。
- hashは完全性の検査であり、誰が生成したかを保証する真正性の検査ではない。

## 実装済みの回帰と次の拡張

固定fixtureのwrite/read、hash破損、軸・metadata・版番号の負のテストは
GitHubの新規clone相当CIへ入っている。JuliaQCDで生成した
\(2^3\times5\)の全相対位置・全時刻の二\(\Xi\)四点相関と二つの
単一\(\Xi\)相関も同じ形式へ保存し、読み戻した配列の完全一致と
`R -> V_LO`接続を確認した。

次の拡張は次である。

1. 独立gauge configuration、複数source、smearingを含む実データで必須とする
   metadata fieldを固定する。
2. 大規模ensembleのデータ量とaccess patternを測ってから、chunking、圧縮、
   HDF5/JLD2 extensionの必要性を判断する。
3. 版移行が必要になった場合は、v1 fixtureを読み続ける互換testを先に追加する。
