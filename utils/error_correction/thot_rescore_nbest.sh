# Author: Daniel Ortiz Mart\'inez
# *- bash -*

# Rescores an n-best list file generated by means of the thot_process_wg
# tool given a set of weights. The n-best list file is read from
# standard input.

if [ $# -eq 0 ]; then
    echo "Usage: thot_rescore_nbest \"<weight1> <weight2> ... <weightn>\""
else
    weights=$1
    ${AWK} -v weights="$weights" \
           'BEGIN{
             numw=split(weights,weightArray," ")
            }
            {
             if(FNR==1)
             {
              printf"# %s\n",weights
             }
             else
             { 
              newscore=0;
              for(i=1;i<=numw;++i)
              {
               newscore+=$(i+2)*weightArray[i]
              }
              $1=newscore
              printf"%s\n",$0
             }
            }'
fi
