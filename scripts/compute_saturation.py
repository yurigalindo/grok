#!/usr/bin/env python

import os
import grok

parser = grok.training.add_args()
parser.set_defaults(logdir=os.environ.get("LOGDIR", "."))
hparams = parser.parse_args()
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)


print(hparams)

# ckpts = [f"/Users/tsili/Downloads/Addition_97_old/init.pt"]
# ckpts = [f"/Users/tsili/Downloads/Addition_97_old/epoch_{2**i}.ckpt" for i in range(17)]
ckpts = [f"/Users/tsili/Projects/grok/Addition_97/epoch_{2**i}.ckpt" for i in range(17)]
ckpts.append("/Users/tsili/Projects/grok/Addition_97/epoch_100000.ckpt")
grok.training.compute_saturation(hparams, ckpts)
