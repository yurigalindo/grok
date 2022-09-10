#!/usr/bin/env python

import os
import grok

parser = grok.training.add_args()
parser.set_defaults(logdir=os.environ.get("LOGDIR", "."))
hparams = parser.parse_args()
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)


print(hparams)

ckpts = [f"checkpoints/Addition_97_old/epoch_{(2)**i}.ckpt" for i in range(20)]
grok.training.compute_saturation(hparams, ckpts)
