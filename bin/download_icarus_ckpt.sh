#!/bin/bash
# v0.1 ... iteration-127500-epoch-7500.ckpt ... ID = 1OckQGtwWaEViQJlPZc0DOjeLj7s5IKHR
#
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1OckQGtwWaEViQJlPZc0DOjeLj7s5IKHR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1OckQGtwWaEViQJlPZc0DOjeLj7s5IKHR" -O icarus_siren.ckpt
gdown https://drive.google.com/uc?id=1OckQGtwWaEViQJlPZc0DOjeLj7s5IKHR -O icarus_siren.ckpt