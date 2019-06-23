#!/bin/bash
mencoder mf://*.png -mf w=1735:h=1140:fps=6:type=png -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o out.avi
