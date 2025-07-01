# Mahjong AI – Quick Start

## Prerequisite
Install the Chinese National Standard (GB) Mahjong scoring library:

```bash
pip install git+https://github.com/ailab-pku/PyMahjongGB.git
```

---

## Usage

```bash
python preprocess.py   # preprocess game logs into the data/ directory
python supervised.py   # start supervised‑learning training
```

* Checkpoints and logs are saved to `log/checkpoint/`.
* After training finishes, you can upload the model to [**Botzone**](http://botzone.org.cn) for live matches.
* For data information, see `data/README-en.txt`. Full dataset can be found in [**IJCAI 2024 MahjongCompetition**]https://botzone.org.cn/static/gamecontest2024a.html.


