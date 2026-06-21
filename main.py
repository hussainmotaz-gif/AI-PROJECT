

import sys
import os
#sys.path.append(os.path.abspath("Bot4"))
#sys.path.append(os.path.abspath("Bot1"))
sys.path.append(os.path.abspath("Bot2"))
# sys.path.append(os.path.abspath("Bot3"))


from Bot2.run import run
#from Bot3.run import run

# Pour exécuter ce script, utilisez la commande suivante dans le terminal :
# set `TF_ENABLE_ONEDNN_OPTS=0` && py main.py
run()
