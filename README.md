# Running the Repo

Installing the Repo
```bash
pip install -e .
```

Running experiments (add --help to check options/arguments)
```bash
./scripts/train.py
```
Default arguments - grok/training.py/lines 80-131

Possible math operators - grok/data.py/lines 19-40

To plot metrics 
```bash
python scripts/plot_metrics.py --metric="default/version_{i}/metrics.csv"
```

To run multiple cardinality experiments, update the **MODULUS** variable in *grok/data.py* to desired cardinality.
