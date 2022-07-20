#!/usr/bin/env python

import grok
import os

# Create parser object with parameters
parser = grok.training.add_args()
#Setup default logging directory
parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
# Generate a Namespace Object
hparams = parser.parse_args()
#Normalizing pathnames
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)


print(hparams)
#Start training parameters
print(grok.training.train(hparams))
