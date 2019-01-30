import argparse
from ExampleDb import ExampleDb
from helper_functions import parse_output_path, parse_input_path
import h5py
import numpy as np
from os.path import isfile, basename, splitext
from TrainingRead import TrainingRead

parser = argparse.ArgumentParser(description='Create ZODB database of training reads from fast5-directory.')
parser.add_argument('-o', '--db-dir', type=str, required=True,
                    help='Name for new database directory')
parser.add_argument('-n', '--normalization', type=str, required=False, default='median',
                    help='Specify how the raw data should be normalized.')
parser.add_argument('-w', '--width', type=int, required=True,
                    help='Width of window around the target to include.')
parser.add_argument('-i', '--input', type=str, required=False,
                      help='Specify location of reads')
parser.add_argument('-k', '--kmer-size', type=int, required=False, default=5,
                    help='k-mer size to construct db for')
parser.add_argument('--hdf-path', type=str, required=False, default='Analyses/Basecall_1D_000',
                    help='Internal path in fast5-files, at which analysis files can be found.')
parser.add_argument('--use-tombo', action='store_true', default=False,
                    help='Given hdf path refers to a tombo analysis.')
parser.add_argument('--clipped-bases', type=int, required=False, default=10,
                    help='Define how many bases should be clipped off ends of training reads.'
                         'Overruled by --use-tombo (in which case base clipping will depend on'
                         'an alignment).')

args = parser.parse_args()

file_list = parse_input_path(args.input)
out_path = parse_output_path(args.db_dir)
db_name = out_path+'db.fs'
error_fn = out_path+'failed_reads.txt'
npz_path = out_path + 'test_squiggles/'
npz_path = parse_output_path(npz_path)
if isfile(db_name):
    raise ValueError('DB  by this name already exists!')

# Very light check on tombo usage
if args.use_tombo and 'RawGenomeCorrected' not in args.hdf_path:
    raise UserWarning('Tombo files should be used, but hdf path does not seem tombo-generated...')

db = ExampleDb(db_name=db_name, width=args.width)
nb_files = len(file_list)
count_pct_lim = 5
for i, files in enumerate(file_list):
    with h5py.File(files, 'r') as f:
        try:
            tr = TrainingRead(f,
                              normalization=args.normalization,
                              hdf_path=args.hdf_path,
                              clipped_bases=args.clipped_bases,
                              kmer_size=args.kmer_size,
                              use_tombo=args.use_tombo)
            #~ db.add_training_read(training_read=tr)
        except ValueError as e:
            with open(error_fn, 'a') as efn:
                efn.write('{fn}\t{err}\n'.format(err=e, fn=basename(files)))
            continue
    
    np.savez(npz_path + splitext(basename(files))[0],
             base_labels=tr.classified, 
             raw=tr.raw[: tr.final_signal])

    if not i+1 % 10:  # Every 10 reads remove history of transactions ('pack' the database) to reduce size
        db.pack_db()
    percentage_processed = int( (i+1) / nb_files * 100)
    if percentage_processed >= count_pct_lim:
        print('{pct} % of reads processed, {pos} positives and {neg} negatives in DB'.format(pct=percentage_processed,
                                                                                          neg=db.nb_neg,
                                                                                          pos=db.nb_pos))
        count_pct_lim += 5
db.pack_db()
