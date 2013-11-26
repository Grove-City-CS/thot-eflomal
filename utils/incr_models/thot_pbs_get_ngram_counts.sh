# Author: Daniel Ortiz Mart\'inez
# *- bash -*

# Extracts n-gram counts from a monolingual corpus.

print_desc()
{
    echo "pbs_get_ngram_counts written by Daniel Ortiz"
    echo "pbs_get_ngram_counts extracts n-grams counts from a monolingual corpus"
    echo "type \"pbs_get_ngram_counts --help\" to get usage information"
}

version()
{
    echo "pbs_get_ngram_counts is part of the thot package"
    echo "thot version "${version}
    echo "thot is GNU software written by Daniel Ortiz"
}

usage()
{
    echo "pbs_get_ngram_counts -pr <int>"
    echo "                     -c <string> -o <string> -n <int> [-unk]"
    echo "                     [-qs <string>] [-tdir <string>] [-sdir <string>]"
    echo "                     [-debug] [--help] [--version]"
    echo ""
    echo "-pr <int>          : Number of processors."
    echo "-c <string>        : Corpus file (give absolute path when"
    echo "                     using pbs clusters)."
    echo "-o <string>        : Prefix of output files."
    echo "-n <int>           : Order of the n-grams."
    echo "-unk               : Reserve probability mass for the unknown word."
    echo "-qs <string>       : Specific options to be given to the qsub command"
    echo "                     (example: -qs \"-l pmem=1gb\")."
    echo "-tdir <string>     : Directory for temporary files."
    echo "                     NOTES:"
    echo "                      a) give absolute paths when using pbs clusters."
    echo "                      b) ensure there is enough disk space in the partition."
    echo "-sdir <string>     : Absolute path of a directory common to all"
    echo "                     processors. If not given, the directory for"
    echo "                     temporaries will be used (/tmp or the"
    echo "                     directory given by means of the -tdir option)."
    echo "                     NOTES:"
    echo "                      a) give absolute paths when using pbs clusters."
    echo "                      b) ensure there is enough disk space in the partition."
    echo "-debug             : After ending, do not delete temporary files"
    echo "                     (for debugging purposes)."
    echo "--help             : Display this help and exit."
    echo "--version          : Output version information and exit."
}

pipe_fail()
{
    # test if there is at least one command to exit with a non-zero status
    for pipe_status_elem in ${PIPESTATUS[*]}; do 
        if test ${pipe_status_elem} -ne 0; then 
            return 1; 
        fi 
    done
    return 0
}

set_tmp_dir()
{
    if [ -d ${tdir} ]; then
        TMP=${tdir}
    else
        echo "Error: temporary directory does not exist"
        return 1;
    fi
}

set_shared_dir()
{
    if [ -z "$sdir" ]; then
        # if not given, SDIR will be created in the $TMP directory
        SDIR="${TMP}/pbs_get_ngram_counts_sdir_${PPID}_$$"
        mkdir $SDIR || { echo "Error: shared directory cannot be created" ; return 1; }
    else
        SDIR="${sdir}/pbs_get_ngram_counts_sdir_${PPID}_$$"
        mkdir $SDIR || { echo "Error: shared directory cannot be created" ; return 1; }
    fi

    # Create temporary subdirectories
    chunks_dir=$SDIR/chunks
    scripts_dir=$SDIR/scripts
    counts_per_chunk_dir=$SDIR/counts_per_chunk
    mkdir ${chunks_dir} || return 1
    mkdir ${scripts_dir} || return 1
    mkdir ${counts_per_chunk_dir} || return 1

    # Function executed correctly
    return 0
}

split_input()
{
    echo "*** Splitting input: ${corpus}..." >> $SDIR/log

    # Determine fragment size
    local input_size=`wc -l ${corpus} 2>/dev/null | ${AWK} '{printf"%d",$1}'`
    if [ ${input_size} -lt ${pr_val} ]; then
        echo "Error: problem too small"
        exit 1
    fi
    local chunk_size=`expr ${input_size} / ${pr_val}`
    local chunk_size=`expr ${chunk_size} + 1`

    # Split input 
    ${SPLIT} -l ${chunk_size} ${corpus} ${chunks_dir}/chunk\_ || return 1
}

remove_temp()
{
    # remove shared directory
    if [ "$debug" -eq 0 ]; then
        rm -rf $SDIR 2>/dev/null
    fi
}

exclude_readonly_vars()
{
    ${AWK} -F "=" 'BEGIN{
                         readonlyvars["BASHOPTS"]=1
                         readonlyvars["BASH_VERSINFO"]=1
                         readonlyvars["EUID"]=1
                         readonlyvars["PPID"]=1
                         readonlyvars["SHELLOPTS"]=1
                         readonlyvars["UID"]=1
                        }
                        {
                         if(!($1 in readonlyvars)) printf"%s\n",$0
                        }'
}

create_script()
{
    # Init variables
    local name=$1
    local command=$2

    # Write environment information
    declare | exclude_readonly_vars > ${name}

    # If sh is being used as command interpreter, functions have to be
    # added explicitly using the -f option of the "declare" command
    command_interp=$(basename "${BASH}")
    if [ ${command_interp} = "sh" ]; then
        declare -f >> ${name}
    fi

    # Write PBS directives
    echo "#PBS -o ${name}.o\${PBS_JOBID}" >> ${name}
    echo "#PBS -e ${name}.e\${PBS_JOBID}" >> ${name}

    # Write command to be executed
    echo "${command}" >> ${name}

    # Give execution permission
    chmod u+x ${name}
}

launch()
{
    local job_deps=$1
    local program=$2
    local suffix=$3
    local outvar=$4

    if [ -z "$QSUB" ]; then
        $program &
        eval "${outvar}=$!"
    else
        # Synchronization is carried out by explicitly defining
        # dependencies between jobs when executing qsub
            
        # Create script
        create_script ${scripts_dir}/${program}${suffix}.sh $program

        # Define qsub option declaring job dependencies
        local depend_opt=""
        if [ ! -z "$job_deps" ]; then
            job_deps=`echo ${job_deps} | $AWK '{for(i=1;i<NF;++i) printf"%s:",$i; printf"%s",$NF}'`
            depend_opt="-W depend=afterok:${job_deps}"
        fi

        # Execute qsub command. The -h option is used to hold
        # jobs. All jobs are released at the end of the script. This
        # ensures that job dependencies are defined over existing
        # jobs
        local jid=$(${QSUB} -h ${depend_opt} ${qs_opts} ${scripts_dir}/${program}${suffix}.sh)

            # Set value of output variable
        eval "${outvar}='${jid}'"

        # Uncomment line to show debug information
        # echo $program ${depend_opt} $jid
    fi
}

sync()
{
    if [ -z "$QSUB" ]; then
        wait
        return 0
    fi
}

release_job_holds()
{
    job_ids=$1
    ${QRLS} ${job_ids}
    
    # Uncomment line to get debugging information
    # echo ${job_id_list}
}

add_length_col()
{
    ${AWK} '{printf"%d %s\n",NF,$0}'
}

remove_length_col()
{
    ${AWK} '{for(i=2;i<=NF-1;++i)printf"%s ",$i; printf"%d\n",$NF}'
}

sort_counts()
{
    # Set sort command options
    export LC_ALL=""
    export LC_COLLATE=C
    if test ${sortT} = "yes"; then
        SORT_TMP="-T $TMP"
    else
        SORT_TMP=""
    fi

    ${SORT} ${SORT_TMP} -t " " ${sortpars}
}

add_chunk_id()
{
    ${AWK} -v c=${chunk_id} '{printf"%s %s\n",$0,c}'    
}

proc_chunk()
{
    # Write date to log file
    echo "** Processing chunk ${chunk} (started at "`date`")..." >> $SDIR/log

    # Determine if -unk should be used
    if [ ${unk_given} -eq 1 -a ${chunk_id} -eq 1 ]; then
        unk_opt="-unk"
    else
        unk_opt=""
    fi

    # Extract counts from chunk and sort them
    $bindir/thot_get_ngram_counts_mr -c ${chunks_dir}/${chunk} -n ${n_val} -tdir $TMP ${unk_opt} | \
        add_length_col | sort_counts | add_chunk_id > ${counts_per_chunk_dir}/${chunk}_sorted_counts ; pipe_fail || return 1
    
    # Write date to log file
    echo "Processing of chunk ${chunk} finished ("`date`")" >> $SDIR/log 

    return 0
}

merge_sort()
{
    # Set sort command options
    export LC_ALL=""
    export LC_COLLATE=C
    if test ${sortT} = "yes"; then
        SORT_TMP="-T $TMP"
    else
        SORT_TMP=""
    fi

    ${SORT} ${SORT_TMP} -t " " ${sortpars} -m ${counts_per_chunk_dir}/*_sorted_counts
}

generate_counts_file()
{
    # Delete output file if exists
    if [ -f ${output} ]; then
        rm ${output}
    fi

    # Merge count files
    merge_sort | remove_length_col | ${bindir}/thot_merge_ngram_counts > ${output} ; pipe_fail || return 1

    # Copy log file
    echo "**** Parallel process finished at: "`date` >> $SDIR/log
    cp $SDIR/log ${output}.log
}

pr_given=0
unk_given=0
sdir=""
qs_given=0
c_given=0
n_given=0
o_given=0
tdir="/tmp"
debug=0

if [ $# -eq 0 ]; then
    print_desc
    exit 1
fi

while [ $# -ne 0 ]; do
    case $1 in
        "--help") usage
            exit 0
            ;;
        "--version") version
            exit 0
            ;;
        "-pr") shift
            if [ $# -ne 0 ]; then
                pr_val=$1
                pr_given=1
            fi
            ;;
        "-sdir") shift
            if [ $# -ne 0 ]; then
                sdir=$1                
            else
                sdir=""
            fi
            ;;
        "-c") shift
            if [ $# -ne 0 ]; then
                corpus=$1
                c_given=1
            else
                c_given=0
            fi
            ;;
        "-n") shift
            if [ $# -ne 0 ]; then
                n_val=$1
                n_given=1
            else
                n_given=0
            fi
            ;;
        "-o") shift
            if [ $# -ne 0 ]; then
                output=$1
                o_given=1
            else
                o_given=0
            fi
            ;;
        "-qs") shift
            if [ $# -ne 0 ]; then
                qs_opts=$1
                qs_given=1
            else
                qs_given=0
            fi
            ;;
        "-tdir") shift
            if [ $# -ne 0 ]; then
                tdir=$1
            else
                tdir="./"
            fi
            ;;
        "-unk") unk_given=1
            ;;
        "-debug") debug=1
            ;;
    esac
    shift
done

# verify parameters

if [ ${c_given} -eq 0 ]; then
    echo "Error: corpus file not given"
    exit 1
else
    if [ ! -f  "${corpus}" ]; then
        echo "Error: file ${corpus} with training sentences does not exist"
        exit 1
    fi
fi

if [ ${n_given} -eq 0 ]; then
    echo "Error: order of the n-grams not provided"
    exit 1
fi

if [ ${pr_given} -eq 0 ]; then
    # invalid parameters 
    echo "Error: number of processors must be given"
    exit 1
fi

# parameters are ok

# Set TMP directory
set_tmp_dir || exit 1

# Set shared directory (global variables are declared)
declare chunks_dir="" 
declare counts_per_chunk_dir="" 
declare scripts_dir="" 

set_shared_dir || exit 1

# Create log file
echo "**** Parallel process started at: "`date` > $SDIR/log

# Split shuffled input into chunks and process them separately...
# job_deps=""
split_input || exit 1

# Declare job id list variable
declare job_id_list=""

# Generate best alignment
chunk_id=0
for i in `ls ${chunks_dir}/chunk\_*`; do
    # Initialize variables
    chunk=`${BASENAME} $i`
    chunk_id=`expr $chunk_id + 1`
        
    # Process chunk
    job_deps=""
    launch "${job_deps}" proc_chunk "_${chunk_id}" job_id || exit 1
    pc_job_ids=${job_id}" "${pc_job_ids}
done

# Extract counts synchronization
sync "${pc_job_ids}" "proc_chunk" || exit 1

# Generate counts file
job_deps=${pc_job_ids}
launch "${job_deps}" generate_counts_file "" gcf_job_id || exit 1

# Generate counts file synchronization
sync || exit 1

# Remove temporary files
job_deps=${gcf_job_id}
launch "${job_deps}" remove_temp "" rt_job_id || exit 1
# Remove temporary files synchronization
sync || exit 1

# Update job_id_list
job_id_list="${pc_job_ids} ${gcf_job_id} ${rt_job_id}"

# Release job holds
if [ ! -z "${QSUB}" ]; then
    release_job_holds "${job_id_list}"
fi