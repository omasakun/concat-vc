<br>
<div align="center">
  <h1>Concat VC</h1>
  <p>A Personal Experiment in Real-Time Voice Conversion</p>
  <p>学習が高速なリアルタイム声質変換を作ってみる個人的な実験</p>
</div>
<br>
<br>

<!-- TODO: Add link to the demo and wandb. -->

## 開発のはじめかた

Linux や WSL2 での開発を想定しています。

```bash
# Install `asdf` before you start if you haven't already.
# asdf: https://asdf-vm.com/guide/getting-started.html

# Clone this repository.
git clone https://github.com/omasakun/concat-vc.git
cd concat-vc

# Install the required tools and packages.
asdf install
pdm  install -G :all

# Now you are ready to go!
source .venv/bin/activate
python engine/hello.py
```

<!-- TODO: 環境構築の方法だけじゃなくて、学習の走らせ方などまで書きたい。 -->

## やってみたこと

### Attempt 01: 単純な音声の切り貼り

変換先の音声が十分にあるなら、変換元の声の発音に近い変換先話者の音声を切り貼りするだけでもそれなりにうまく変換できるかもしれないと思ったので、試してみた。

[関連する Notebook](engine/attempt01.ipynb)

<!-- TODO: 生成結果の音声を貼る :: 動画形式にすれば GitHub のプレビューに埋め込める -->

### Attempt 02: 音程によらない表現をつかう

先の Attempt 01 では、音程があっていない音声を無理やりつなげているせいでうまくいかないように見えた。

なので今度は、音程を調節してから切り貼りするようにしたら良くなるか試してみる。

→ 音程によらない表現を作るのがうまくできなかったので、ひとまず後回しにした。

<!-- TODO: [関連する Notebook](engine/attempt02.ipynb) -->

### Attempt 03: FragmentVC をつかう

Fragment VC がどれくらいの性能なのか、実際に確認してみる。

公式の実装があるので、それを使ってみる。

<!-- TODO: [関連する Notebook](engine/attempt03.ipynb) -->

<!-- TODO: Write more details, results, observations, and conclusions. -->

## 参考にしたものなど

- [Faiss](https://github.com/facebookresearch/faiss) (efficient similarity search)
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477) (phonetic feature extraction)
- [CREPE](https://arxiv.org/abs/1802.06182) (pitch estimation)
- [AdaSpeech](https://arxiv.org/abs/2103.00993) (conditional layer normalization)
- [HiFi-GAN](https://github.com/jik876/hifi-gan) (audio waveform generation)
- [JVS corpus](https://arxiv.org/abs/1908.06248) (free multi-speaker voice corpus)
- [FastSpeech 2](https://arxiv.org/abs/2006.04558), [FastPitch](https://arxiv.org/abs/2006.06873) (introduced me to the world of voice conversion)
- [FragmentVC](https://arxiv.org/abs/2010.14150) (inspired me to use a similarity search)

<!-- TODO: Comprehensive list of references. -->
